"""A move-prediction scorecard that top-k accuracy cannot give you.

Top-3 at 80% is ambiguous: it could be one forced recapture the model nailed, or three equally
plausible moves it could not separate. Accuracy counts hits and ignores both how hard the
position was and how confident the model was.

What this reports instead:

* **Perplexity** on the human's actual move - exp of the cross-entropy. A proper scoring rule,
  and directly readable: "the model was effectively choosing among N moves". Confident-and-wrong
  is punished, which accuracy never does.
* **Accuracy by CONFIDENCE decile** - when the model says it is sure, is it? This is the direct
  answer to "was that top-3 hit three plausible moves or one obvious one": if accuracy tracks
  confidence, the probabilities mean something.
* **Accuracy on HARD positions only** - difficulty defined by a fixed REFERENCE model's entropy,
  not by the model being scored, so it cannot grade its own homework. The headline number is
  accuracy on the top difficulty tercile: forced moves are excluded by construction.
* **Calibration error** - mean |confidence - accuracy| across deciles. Near zero means the
  model's stated probability is trustworthy, which is what a clone needs in order to SAMPLE
  human-like moves rather than always play its argmax.
* **Top-3 given top-1 missed** - are the extra two slots doing real work, or padding?

    PYTHONPATH=. python -u scripts/move_scorecard.py --models checkpoints/base_300k_best.pt
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from sahformer.model.config import ModelConfig
from sahformer.training.loop import build_model, load_model
from sahformer.training.losses import move_target_index

DEV = "cuda" if torch.cuda.is_available() else "cpu"
KEYS = ("board", "history", "elo_self", "elo_opp", "temporal")


def load_any(path):
    st = torch.load(path, map_location="cpu", weights_only=False)
    if "model_cfg" in st and "model_state" in st and "step" in st:
        m = build_model("full", ModelConfig(**st["model_cfg"]))
        m.load_state_dict(st["model_state"])
        return m.to(DEV).eval(), f"{os.path.basename(os.path.dirname(path))}@{st['step']:,}"
    m, _ = load_model(path)
    return m.to(DEV).eval(), os.path.basename(path).replace(".pt", "")


@torch.no_grad()
def collect(model, arrs, tgt, bs=128):
    """Per-position: log-prob of the played move, its rank, and the model's entropy."""
    n = len(tgt)
    lp = np.zeros(n, np.float32); rank = np.zeros(n, np.int32)
    ent = np.zeros(n, np.float32); conf = np.zeros(n, np.float32)
    for s in range(0, n, bs):
        j = slice(s, min(s + bs, n))
        b = {k: torch.from_numpy(np.ascontiguousarray(arrs[k][j])).float().to(DEV) for k in KEYS}
        with torch.autocast("cuda", enabled=(DEV == "cuda")):
            lg = model(b)["move_logits"].float()
        logp = F.log_softmax(lg, dim=-1)
        t = tgt[j].to(DEV)
        lp[j] = logp.gather(1, t[:, None]).squeeze(1).cpu().numpy()
        rank[j] = (logp > logp.gather(1, t[:, None])).sum(1).cpu().numpy()   # 0 = top-1
        p = logp.exp()
        ent[j] = (-(p * logp).sum(1)).cpu().numpy()
        conf[j] = p.max(1).values.cpu().numpy()
    return lp, rank, ent, conf


def scorecard(name, lp, rank, conf, hard_mask):
    ce = float(-lp.mean())
    out = {
        "model": name,
        "top1": float(100 * (rank == 0).mean()),
        "top3": float(100 * (rank < 3).mean()),
        "top5": float(100 * (rank < 5).mean()),
        "ce_nats": ce,
        "perplexity": float(math.exp(ce)),
        "top1_hard": float(100 * (rank[hard_mask] == 0).mean()),
        "ppl_hard": float(math.exp(-lp[hard_mask].mean())),
        "top3_given_top1_miss": float(100 * ((rank < 3) & (rank > 0)).sum() / max((rank > 0).sum(), 1)),
    }
    # calibration: does stated confidence match observed accuracy?
    q = np.quantile(conf, np.linspace(0, 1, 11))
    dec, gaps = [], []
    for i in range(10):
        m = (conf >= q[i]) & (conf <= q[i + 1] if i == 9 else conf < q[i + 1])
        if m.sum() < 50:
            continue
        c, a = float(conf[m].mean() * 100), float(100 * (rank[m] == 0).mean())
        dec.append({"decile": i + 1, "confidence": c, "accuracy": a, "n": int(m.sum())})
        gaps.append(abs(c - a))
    out["calibration_err"] = float(np.mean(gaps)) if gaps else float("nan")
    out["deciles"] = dec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["checkpoints/base_300k_best.pt"])
    ap.add_argument("--reference", default="checkpoints/base_300k_best.pt",
                    help="model whose entropy DEFINES position difficulty (so no model grades "
                         "its own homework)")
    ap.add_argument("--raw", default="cache/timehead_raw")
    ap.add_argument("--split", default="cache/timehead_split.json")
    ap.add_argument("--out", default="move_scorecard.json")
    ap.add_argument("--bs", type=int, default=128)
    args = ap.parse_args()

    names = json.load(open(args.split, encoding="utf-8"))["test"]
    parts = []
    for nm in names:
        p = os.path.join(args.raw, f"{nm}.npz")
        if os.path.exists(p):
            z = np.load(p)
            parts.append({k: z[k] for k in KEYS + ("move_from", "move_to", "promo")})
    arrs = {k: np.concatenate([p[k] for p in parts]) for k in KEYS + ("move_from", "move_to", "promo")}
    tgt = move_target_index(torch.from_numpy(arrs["move_from"]), torch.from_numpy(arrs["move_to"]),
                            torch.from_numpy(arrs["promo"]))
    print(f"{len(tgt):,} positions from {len(parts)} held-out players\n")

    ref, refname = load_any(args.reference)
    _, _, ref_ent, _ = collect(ref, arrs, tgt, args.bs)
    thr = np.quantile(ref_ent, 2 / 3)
    hard = ref_ent >= thr
    print(f"difficulty defined by {refname} entropy; HARD = top tercile "
          f"({hard.sum():,} positions, entropy >= {thr:.2f} nats)\n")
    del ref; torch.cuda.empty_cache()

    rows = []
    for mp in args.models:
        m, nm = load_any(mp)
        lp, rank, ent, conf = collect(m, arrs, tgt, args.bs)
        rows.append(scorecard(nm, lp, rank, conf, hard))
        del m; torch.cuda.empty_cache()

    print(f"{'model':<26} {'top1':>6} {'top3':>6} {'PPL':>6} | {'top1':>6} {'PPL':>6} | "
          f"{'calib':>6} {'t3|miss':>8}")
    print(f"{'':<26} {'':>6} {'':>6} {'':>6} | {'HARD positions':^13} | {'err':>6} {'':>8}")
    for r in rows:
        print(f"{r['model']:<26} {r['top1']:>5.2f}% {r['top3']:>5.2f}% {r['perplexity']:>6.2f} | "
              f"{r['top1_hard']:>5.2f}% {r['ppl_hard']:>6.2f} | {r['calibration_err']:>5.1f}pp "
              f"{r['top3_given_top1_miss']:>7.1f}%")

    for r in rows:
        print(f"\n{r['model']} - accuracy by confidence decile:")
        print(f"   {'decile':>7} {'says':>7} {'is right':>9} {'n':>8}")
        for d in r["deciles"]:
            print(f"   {d['decile']:>7} {d['confidence']:>6.1f}% {d['accuracy']:>8.1f}% {d['n']:>8,}")

    json.dump(rows, open(args.out, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
