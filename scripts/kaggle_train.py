"""Kaggle training entrypoint: scale the base backbone, with resumable checkpoints.

Kaggle GPU sessions are time-limited and get killed, so this is built to be re-run: it writes a
rolling checkpoint every `--save-every` steps and, on start, resumes from the newest one it finds
in `--out`. Re-running the same notebook cell continues the run rather than restarting it.

WHAT THIS TRAINS, and what it does not:

* Backbone is scaled (dim_vit 512, 12 blocks ~ 30M) per the 08-26 spec section 4.1.
* An Elo head is added (section 4.3) - still untested and still motivated: HANDOFF-08-16 section 3
  found the Elo dial only moved move CONFIDENCE, not move choice.
* The bucket TIME head from that spec is deliberately NOT the point. The 2026-09-03 findings
  showed head architecture changes distribution match by ~7% while the readout changes it 2.6x,
  and that conditional skill saturates regardless. The time head here stays as-is; timing is
  fixed at inference (sample, don't average; self-clock), not by retraining.

DATA CAVEAT: the HF corpus is the leaderboard-seeded one, elo_self 2012-3074. This run therefore
answers "does capacity improve move-match", NOT "does the base serve clonefish's users", who sit
below 2000 and are not represented here.

    python kaggle_train.py --shards /kaggle/input/... --out /kaggle/working/ckpt
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sahformer.model.config import ModelConfig
from sahformer.training.loop import build_model
from sahformer.training.losses import move_target_index

DEV = "cuda" if torch.cuda.is_available() else "cpu"
KEYS = ("board", "history", "elo_self", "elo_opp", "temporal")


class EloHead(nn.Module):
    """Predicts the mover's own rating from the pooled summary (08-26 spec s4.3).

    Forces the representation to encode strength rather than using Elo only as a confidence
    knob, and gives clonefish a strength meter for free.
    """

    def __init__(self, dim, hid=256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hid), nn.ReLU(),
                                 nn.Linear(hid, 1))

    def forward(self, pooled):
        return self.net(pooled).squeeze(-1)


FIELDS = KEYS + ("move_from", "move_to", "promo", "result", "think_time")


def shard_stream(paths, bs, seed=0, shuffle_shards=True):
    """Stream batches shard by shard so peak RAM stays flat regardless of corpus size.

    MATERIALISE each shard once. np.load on an .npz returns a LAZY handle, and every `d[k]`
    access re-decompresses that whole array out of the zip - for `history` that is 250k x 5,376
    bytes = 1.34 GB per key per batch. Measured: 2,553 ms per batch lazy versus 0.1 ms once the
    arrays are in memory, i.e. the GPU sat idle through ~100% of every step. One shard resident
    is ~1.2 GB, which is nothing against the box's RAM.
    """
    rng = np.random.default_rng(seed)
    order = list(paths)
    while True:
        if shuffle_shards:
            rng.shuffle(order)
        for p in order:
            with np.load(p) as z:
                d = {k: z[k] for k in FIELDS}       # decompress ONCE, not once per batch
            n = len(d["think_time"])
            idx = rng.permutation(n)
            for s in range(0, n - bs + 1, bs):
                i = np.sort(idx[s:s + bs])
                yield {k: d[k][i] for k in FIELDS}
            del d


def to_dev(b):
    out = {k: torch.from_numpy(np.ascontiguousarray(b[k])).float().to(DEV) for k in KEYS}
    out["move_from"] = torch.from_numpy(np.ascontiguousarray(b["move_from"])).to(DEV)
    out["move_to"] = torch.from_numpy(np.ascontiguousarray(b["move_to"])).to(DEV)
    out["promo"] = torch.from_numpy(np.ascontiguousarray(b["promo"])).to(DEV)
    out["result"] = torch.from_numpy(np.ascontiguousarray(b["result"])).long().to(DEV)
    out["think_time"] = torch.from_numpy(np.ascontiguousarray(b["think_time"])).float().to(DEV)
    out["elo_raw"] = torch.from_numpy(np.ascontiguousarray(b["elo_self"])).float().to(DEV)
    return out


def newest_ckpt(out, extra=()):
    """Newest checkpoint in `out`, else in any `extra` dir.

    Across Kaggle sessions the previous run's /kaggle/working is mounted read-only as an INPUT,
    so a resume has to look there too or every session silently restarts from zero.
    """
    cands = []
    for d in (out,) + tuple(extra):
        for pat in ("last.pt", "step_*.pt"):
            cands += glob.glob(os.path.join(d, pat))
    return max(cands, key=os.path.getmtime) if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="glob or directory of .npz shards")
    ap.add_argument("--out", default="/kaggle/working/ckpt")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--blocks", type=int, default=12)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--steps", type=int, default=400000)
    ap.add_argument("--bs", type=int, default=256, help="micro-batch that must fit in VRAM")
    ap.add_argument("--accum", type=int, default=4,
                    help="gradient accumulation; EFFECTIVE batch = bs * accum. The 08-26 recipe "
                         "specifies batch 512 with lr 4e-5, and a 6GB card cannot hold 512 - "
                         "accumulating keeps the agreed recipe instead of improvising a new lr "
                         "for a smaller batch, which is how this repo diverged before (3ab430d).")
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--w-policy", type=float, default=1.0)
    ap.add_argument("--w-value", type=float, default=0.1)
    ap.add_argument("--w-time", type=float, default=0.2)
    ap.add_argument("--w-elo", type=float, default=0.05)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--grad-clip", type=float, default=3.5)
    ap.add_argument("--resume-dirs", nargs="*", default=[],
                    help="extra dirs to search for a checkpoint (previous Kaggle outputs)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    paths = (sorted(glob.glob(os.path.join(args.shards, "*.npz")))
             if os.path.isdir(args.shards) else sorted(glob.glob(args.shards)))
    if not paths:
        raise SystemExit(f"no shards matched {args.shards}")
    print(f"{len(paths)} shards | device {DEV} | micro-batch {args.bs} x accum {args.accum} "
          f"= EFFECTIVE BATCH {args.bs * args.accum}", flush=True)

    mc = ModelConfig(dim_vit=args.dim, num_blocks=args.blocks, num_heads=args.heads)
    model = build_model("full", mc).to(DEV)
    elo_head = EloHead(args.dim).to(DEV)
    params = list(model.parameters()) + list(elo_head.parameters())
    print(f"model {sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"(+{sum(p.numel() for p in elo_head.parameters())/1e6:.2f}M elo head)", flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEV == "cuda"))
    step0, hist = 0, []

    ck = newest_ckpt(args.out, tuple(args.resume_dirs))
    if ck:
        st = torch.load(ck, map_location=DEV, weights_only=False)
        model.load_state_dict(st["model_state"])
        elo_head.load_state_dict(st["elo_state"])
        opt.load_state_dict(st["opt_state"])
        scaler.load_state_dict(st["scaler_state"])
        step0 = st["step"]; hist = st.get("hist", [])
        print(f"RESUMED from {ck} at step {step0:,}", flush=True)
    else:
        print("fresh run (no checkpoint found)", flush=True)

    def save(step, tag="last"):
        torch.save({"step": step, "model_state": model.state_dict(),
                    "elo_state": elo_head.state_dict(), "opt_state": opt.state_dict(),
                    "scaler_state": scaler.state_dict(), "model_cfg": vars(mc),
                    "hist": hist, "args": vars(args)},
                   os.path.join(args.out, f"{tag}.pt"))
        json.dump(hist, open(os.path.join(args.out, "history.json"), "w"), indent=2)

    stream = shard_stream(paths, args.bs, seed=args.seed + step0)
    t0, acc, seen = time.time(), 0.0, 0
    skipped = [0]
    model.train()
    for step in range(step0 + 1, args.steps + 1):
        for gp in opt.param_groups:                       # linear warmup
            gp["lr"] = args.lr * min(1.0, step / max(args.warmup, 1))
        opt.zero_grad(set_to_none=True)
        for micro in range(args.accum):
            b = to_dev(next(stream))
            with torch.autocast("cuda", enabled=(DEV == "cuda")):
                out = model(b)
                tgt = move_target_index(b["move_from"], b["move_to"], b["promo"])
                pol = F.cross_entropy(out["move_logits"].float(), tgt)
                val = F.cross_entropy(out["value_logits"].float(), b["result"])
                pi, mu, sg = out["mdn"]
                from sahformer.model.heads import mdn_nll
                tim = mdn_nll(pi.float(), mu.float(), sg.float(), b["think_time"])
                elo = F.smooth_l1_loss(elo_head(out["pooled"].float()),
                                       (b["elo_raw"] - 1500.0) / 500.0)
                loss = (args.w_policy * pol + args.w_value * val +
                        args.w_time * tim + args.w_elo * elo)
            # scale so the accumulated gradient equals a true batch of bs*accum
            scaler.scale(loss / args.accum).backward()
            if micro == 0:
                acc += float((out["move_logits"].argmax(-1) == tgt).float().mean()); seen += 1
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        if torch.isfinite(gnorm):
            scaler.step(opt)
        else:
            skipped[0] += 1              # NaN batch: drop the step rather than poison the weights
        scaler.update()
        if step % args.log_every == 0:
            rec = {"step": step, "loss": float(loss), "policy": float(pol),
                   "value": float(val), "time": float(tim), "elo": float(elo),
                   "move_acc": acc / max(seen, 1), "skipped": skipped[0],
                   "elapsed": round(time.time() - t0, 1)}
            hist.append(rec); acc, seen = 0.0, 0
            print(f"step {step:>7} | loss {rec['loss']:.4f} pol {rec['policy']:.4f} "
                  f"move_acc {rec['move_acc']*100:.2f}% | {rec['elapsed']:.0f}s", flush=True)
        if step % args.save_every == 0:
            save(step)
            print(f"  checkpoint saved at step {step:,}", flush=True)
    save(args.steps)
    print("done", flush=True)


if __name__ == "__main__":
    main()
