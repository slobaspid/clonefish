"""How does the newly-trained base's TIMING compare to the incumbent?

Both models carry an MDN think-time head, so this scores them the same way the 2026-09-03
bake-off scored everything: SAMPLED draws for distribution match (never E[t] - that collapses
the snap/tank shape), and r at E[t] for conditional skill, reported side by side.

CONTAMINATION WARNING, read before believing anything here. shards_v3 was built from every
corpus directory including data/chesscom_bands_v2, which holds these 20 test players' games. So
base_v3 has trained on them. At step 10k the model has seen ~5.1M of 155.5M positions (~3.3% of
the corpus), so expected exposure to any single test player is small - but it is not zero, and
base_v3's numbers here should be read as an optimistic bound, not a held-out result.

base_300k (the incumbent) was trained on data/chesscom_corpus, the high-elo slice, and these
players come from the band-balanced crawl - so it is closer to clean, though a test player could
still appear inside some crawled opponent's file.

    PYTHONPATH=. python -u scripts/timehead_compare_bases.py
"""
import argparse
import json
import os

import numpy as np
import torch

from sahformer.model.config import ModelConfig
from sahformer.training.loop import build_model, load_model

DEV = "cuda" if torch.cuda.is_available() else "cpu"
KEYS = ("board", "history", "elo_self", "elo_opp", "temporal")


def wasserstein1(a, b):
    n = min(len(a), len(b))
    a = np.sort(np.asarray(a, np.float64)[:n]); b = np.sort(np.asarray(b, np.float64)[:n])
    return float(np.abs(a - b).mean())


def pearson(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def load_v3(path):
    st = torch.load(path, map_location=DEV, weights_only=False)
    mc = ModelConfig(**st["model_cfg"])
    m = build_model("full", mc).to(DEV)
    m.load_state_dict(st["model_state"])
    m.eval()
    return m, st["step"]


@torch.no_grad()
def mdn_readouts(model, arrs, clock, bs=256, seed=0):
    """(sampled draw, E[t]) from a mixture-of-lognormals head, clock-clamped."""
    rng = torch.Generator(device="cpu").manual_seed(seed)
    draws, means = [], []
    n = len(clock)
    for s in range(0, n, bs):
        j = slice(s, min(s + bs, n))
        b = {k: torch.from_numpy(np.ascontiguousarray(arrs[k][j])).float().to(DEV) for k in KEYS}
        with torch.autocast("cuda", enabled=(DEV == "cuda")):
            out = model(b)
        pi, mu, sg = (t.float() for t in out["mdn"])
        w = torch.softmax(pi, -1)
        sig = torch.nn.functional.softplus(sg) + 1e-3
        k = torch.multinomial(w, 1).squeeze(1)
        m_ = mu.gather(1, k[:, None]).squeeze(1)
        v_ = sig.gather(1, k[:, None]).squeeze(1)
        draws.append(torch.exp(m_ + v_ * torch.randn_like(v_)).cpu().numpy())
        means.append((w * torch.exp(mu + 0.5 * sig ** 2)).sum(-1).cpu().numpy())
    d = np.minimum(np.concatenate(draws), clock)
    return d, np.minimum(np.concatenate(means), clock)


def report(name, samp, et, actual, pid, extra=""):
    per = lambda f: np.mean([f(samp[pid == u], actual[pid == u]) for u in np.unique(pid)])
    return {
        "model": name + extra,
        "median": float(np.median(samp)), "median_err": float(np.median(samp) - np.median(actual)),
        "snap": float((samp < 1).mean() * 100), "snap_err": float((samp < 1).mean() * 100 - (actual < 1).mean() * 100),
        "tail": float((samp > 10).mean() * 100), "tail_err": float((samp > 10).mean() * 100 - (actual > 10).mean() * 100),
        "w1": float(per(lambda a, b: wasserstein1(a, b))),
        "r_Et": pearson(et, actual),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="cache/timehead_raw")
    ap.add_argument("--split", default="cache/timehead_split.json")
    ap.add_argument("--old", default="checkpoints/base_300k_best.pt")
    ap.add_argument("--new", default="checkpoints/base_v3/last.pt")
    ap.add_argument("--out", default="timehead_base_compare.json")
    args = ap.parse_args()

    names = json.load(open(args.split, encoding="utf-8"))["test"]
    parts, pids = [], []
    for i, nm in enumerate(names):
        p = os.path.join(args.raw, f"{nm}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        parts.append({k: z[k] for k in KEYS + ("think_time",)})
        pids.append(np.full(len(z["think_time"]), i))
    arrs = {k: np.concatenate([p[k] for p in parts]) for k in KEYS + ("think_time",)}
    pid = np.concatenate(pids)
    actual = arrs["think_time"].astype(np.float64)
    clock = (arrs["temporal"][:, 0] * 180.0).astype(np.float64)
    print(f"{len(np.unique(pid))} players | {len(actual):,} positions")
    print(f"REAL: median {np.median(actual):.2f}s | snap {100*(actual<1).mean():.1f}% "
          f"| tail {100*(actual>10).mean():.2f}%\n")

    rows = []
    old, _ = load_model(args.old); old.to(DEV).eval()
    s, e = mdn_readouts(old, arrs, clock)
    rows.append(report("base_300k (incumbent)", s, e, actual, pid))
    del old; torch.cuda.empty_cache()

    new, step = load_v3(args.new)
    s, e = mdn_readouts(new, arrs, clock)
    rows.append(report(f"base_v3 @ step {step:,}", s, e, actual, pid, "  [CONTAMINATED]"))

    print(f"{'model':<32} {'median':>7} {'medErr':>7} {'snap':>7} {'snapE':>7} "
          f"{'tail':>7} {'tailE':>7} {'W1':>7} {'r@E[t]':>7}")
    for r in rows:
        print(f"{r['model']:<32} {r['median']:>6.2f}s {r['median_err']:>+6.2f}s "
              f"{r['snap']:>6.1f}% {r['snap_err']:>+6.1f} {r['tail']:>6.2f}% {r['tail_err']:>+6.2f} "
              f"{r['w1']:>7.3f} {r['r_Et']:>7.3f}")
    json.dump({"real_median": float(np.median(actual)),
               "real_snap": float(100 * (actual < 1).mean()),
               "real_tail": float(100 * (actual > 10).mean()), "rows": rows},
              open(args.out, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
