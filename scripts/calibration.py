"""Calibration check: does the model's Elo dial actually track real strength?

For a sample of real positions with KNOWN player rating, sweep elo_self across a grid
and, for each position, find the Elo setting that makes the player's ACTUAL move most
likely (its "revealed Elo"). If the model is calibrated, mean revealed Elo per true-rating
band comes out ~diagonal.

    PYTHONPATH=. python scripts/calibration.py checkpoints/base_300k_best.pt

NOTE: runs on the training shards (model saw them) -> optimistic; a held-out version with
fresh games is the stricter follow-up. This first pass answers "does the dial do anything?"
"""
import argparse, glob, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from sahformer.training.loop import load_model
from sahformer.model.heads import move_to_index


def sample_positions(shard_dir, per_shard, n_shards):
    paths = sorted(glob.glob(os.path.join(shard_dir, "*.npz")))
    pick = paths[:: max(1, len(paths) // n_shards)][:n_shards]
    cols = {k: [] for k in ["board", "history", "temporal", "elo_self",
                            "move_from", "move_to", "promo", "think_time"]}
    for p in pick:
        d = np.load(p, mmap_mode="r")
        n = min(per_shard, d["board"].shape[0])
        idx = np.linspace(0, d["board"].shape[0] - 1, n).astype(int)
        for k in cols:
            cols[k].append(np.asarray(d[k][idx]))
    return {k: np.concatenate(v) for k, v in cols.items()}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--shard-dir", default="data/chesscom_balanced_shards")
    ap.add_argument("--per-shard", type=int, default=160)
    ap.add_argument("--n-shards", type=int, default=28)
    ap.add_argument("--grid", default="1000:2800:100")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() or 4))  # use all CPU cores
    lo, hi, step = (int(x) for x in args.grid.split(":"))
    grid = np.arange(lo, hi + 1, step)
    print(f"loading {args.ckpt} ... ({torch.get_num_threads()} threads)", flush=True)
    model, mcfg = load_model(args.ckpt)
    model.eval()

    S = sample_positions(args.shard_dir, args.per_shard, args.n_shards)
    N = S["board"].shape[0]
    true_elo = S["elo_self"].astype(int)
    actual_idx = np.array([move_to_index(int(f), int(t), int(p))
                           for f, t, p in zip(S["move_from"], S["move_to"], S["promo"])])
    print(f"{N} positions sampled | true Elo range {true_elo.min()}-{true_elo.max()} "
          f"| sweeping {grid[0]}..{grid[-1]} step {step}\n")

    board = torch.from_numpy(S["board"]).float()
    hist = torch.from_numpy(S["history"]).float()
    temporal = torch.from_numpy(S["temporal"]).float()
    aidx = torch.from_numpy(actual_idx).long()

    # logP(actual move | elo=E) for every position, every grid Elo
    logP = np.zeros((N, len(grid)), dtype=np.float32)
    for gi, E in enumerate(grid):
        for s in range(0, N, args.batch):
            e = slice(s, min(N, s + args.batch))
            bsz = board[e].shape[0]
            batch = {
                "board": board[e], "history": hist[e], "temporal": temporal[e],
                "elo_self": torch.full((bsz,), int(E)),
                "elo_opp": torch.full((bsz,), int(E)),
            }
            out = model(batch)
            lp = F.log_softmax(out["move_logits"], dim=-1)
            logP[e, gi] = lp.gather(1, aidx[e].unsqueeze(1)).squeeze(1).numpy()
        print(f"  swept Elo {E}  ({gi+1}/{len(grid)})", flush=True)

    revealed = grid[logP.argmax(axis=1)]
    np.savez(os.path.join(os.path.dirname(args.ckpt) or ".", "calib_arrays.npz"),
             true_elo=true_elo, revealed=revealed)

    # calibration table: bin by true Elo (200-wide), mean revealed
    print("true band     n     mean revealed Elo")
    print("-" * 42)
    bands = np.arange(1000, 3001, 200)
    for b in bands[:-1]:
        m = (true_elo >= b) & (true_elo < b + 200)
        if m.sum() < 10:
            continue
        mr = revealed[m].mean()
        bar = "#" * int(round((mr - 1000) / 60))
        print(f"{b}-{b+199}  {m.sum():5d}   {mr:7.0f}  {bar}")

    # AGGREGATED calibration: group moves into pseudo-"players" of K moves (the granularity
    # the strength-meter actually works at), correlate per-group means. Noise averages out.
    order = np.argsort(true_elo)
    te, rv = true_elo[order], revealed[order]
    print("-" * 42)
    print("moves per 'player'   #groups   correlation")
    for K in [1, 10, 25, 50, 100, 200]:
        ng = len(te) // K
        if ng < 3:
            break
        gt = te[:ng * K].reshape(ng, K).mean(1)
        gr = rv[:ng * K].reshape(ng, K).mean(1)
        rr = np.corrcoef(gt, gr)[0, 1]
        print(f"{K:>15}   {ng:>7}     {rr:.3f}")
    print("-" * 42)
    print("(K=1 is the noisy per-move floor; the strength-meter aggregates many moves/player)")


if __name__ == "__main__":
    main()
