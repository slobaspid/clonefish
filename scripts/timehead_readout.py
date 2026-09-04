"""Is the win the bucket head, or just SAMPLING instead of taking E[t]?

The 2026-08-26 spec adopts a bucket time-head because the MDN "hedges into the middle" and
under-snaps. But the bake-off behind that decision compared the MDN's EXPECTED VALUE against
the bucket head's SAMPLED draws - two changes at once (parameterisation and readout), credited
to one of them.

This scores every trained head under BOTH readouts on the same test humans:

  sampled - draw from the predicted distribution (clock-masked)
  E[t]    - the distribution's mean, which is what ChessMimic itself uses at inference

If both families collapse under E[t] and both recover under sampling, the parameterisation is a
red herring and the fix is one line at inference rather than a new head.

Also bootstraps over test PLAYERS, because with 20 humans a 0.005 gap in RPS means nothing
without an interval around it.

    PYTHONPATH=. python -u scripts/timehead_readout.py
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from sahformer.model.timebuckets import expected_time
from sahformer.training.timeloss import masked_probs

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "bakeoff", os.path.join(os.path.dirname(__file__), "timehead_bakeoff.py"))
BO = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BO)

DEV = BO.DEV


def dist_stats(pred, actual, pid):
    """Per-player distribution match, averaged over players."""
    w1, snap, tail = [], [], []
    for u in np.unique(pid):
        m = pid == u
        w1.append(BO.wasserstein1(pred[m], actual[m]))
        snap.append(abs((pred[m] < 1.0).mean() - (actual[m] < 1.0).mean()))
        tail.append(abs((pred[m] > 10.0).mean() - (actual[m] > 10.0).mean()))
    return np.array(w1), np.array(snap) * 100, np.array(tail) * 100


def boot_ci(per_player, n=2000, seed=0):
    """Bootstrap over PLAYERS - the unit of independence here is the human, not the position."""
    rng = np.random.default_rng(seed)
    k = len(per_player)
    draws = per_player[rng.integers(0, k, size=(n, k))].mean(axis=1)
    return float(per_player.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


@torch.no_grad()
def readouts(head, kind, X, clock, rng):
    """(sampled, deterministic E[t]) for one head."""
    if kind == "mdn":
        draws, means = [], []
        for s in range(0, len(X), 8192):
            pi, mu, sg = head(X[s:s + 8192].to(DEV))
            w = torch.softmax(pi, -1)
            sig = torch.nn.functional.softplus(sg) + 1e-3
            k = torch.multinomial(w, 1).squeeze(1)
            m = mu.gather(1, k[:, None]).squeeze(1)
            v = sig.gather(1, k[:, None]).squeeze(1)
            draws.append(torch.exp(m + v * torch.randn_like(v)).cpu().numpy())
            means.append(torch.exp(mu + 0.5 * sig ** 2).mul(w).sum(-1).cpu().numpy())
        c = clock.numpy()
        return np.minimum(np.concatenate(draws), c), np.minimum(np.concatenate(means), c)

    probs = []
    for s in range(0, len(X), 8192):
        probs.append(masked_probs(head(X[s:s + 8192].to(DEV)), clock[s:s + 8192].to(DEV)).cpu())
    probs = torch.cat(probs)
    return BO.sample_from_probs(probs, clock.numpy(), rng), expected_time(probs).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/timehead")
    ap.add_argument("--ckpts", default="checkpoints/timehead")
    ap.add_argument("--out", default="timehead_readout.json")
    ap.add_argument("--test-players", type=int, default=20)
    ap.add_argument("--val-players", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    players = BO.load_players(args.cache)
    order = np.random.default_rng(args.seed).permutation(len(players))
    te = [players[i] for i in order[:args.test_players]]
    print(f"test on {len(te)} held-out humans\n")

    kinds = {"mdn": "mdn", "hurdle": "hurdle", "ordinal": "ordinal"}
    rows = []
    for ck in sorted(glob.glob(os.path.join(args.ckpts, "*.pt"))):
        name = os.path.basename(ck)[:-3]
        kind = next((v for k, v in kinds.items() if name.startswith(k)), "bucket")
        use_diff = "nodiff" not in name
        Xte, tte, cte, pte = BO.stack(te, use_diff=use_diff)
        head = BO.make_head(kind, Xte.shape[1]).to(DEV)
        head.load_state_dict(torch.load(ck, map_location=DEV))
        head.eval()
        rng = np.random.default_rng(args.seed)
        samp, det = readouts(head, kind, Xte, cte, rng)
        actual, pid = tte.numpy(), pte.numpy()
        for label, pred in (("sampled", samp), ("E[t]", det)):
            w1, sn, tl = dist_stats(pred, actual, pid)
            m, lo, hi = boot_ci(w1)
            rows.append({
                "variant": name, "readout": label,
                "w1": m, "w1_lo": lo, "w1_hi": hi,
                "snap_err": float(sn.mean()), "tail_err": float(tl.mean()),
                "pred_snap_pct": float((pred < 1.0).mean() * 100),
                "pred_tail_pct": float((pred > 10.0).mean() * 100),
                "r": BO.pearson(pred, actual),
            })

    real_snap = float((np.concatenate([d["think"] for d in te]) < 1.0).mean() * 100)
    real_tail = float((np.concatenate([d["think"] for d in te]) > 10.0).mean() * 100)
    print(f"REAL: snap(<1s) {real_snap:.1f}%   tail(>10s) {real_tail:.2f}%\n")
    print(f"{'variant':<20} {'readout':<8} {'W1 (95% CI)':>22} {'snapErr':>8} {'tailErr':>8} "
          f"{'snap%':>7} {'tail%':>7} {'r':>7}")
    for r in sorted(rows, key=lambda x: (x["readout"] != "sampled", x["w1"])):
        ci = f"{r['w1']:.3f} [{r['w1_lo']:.3f},{r['w1_hi']:.3f}]"
        print(f"{r['variant']:<20} {r['readout']:<8} {ci:>22} {r['snap_err']:>7.1f}pp "
              f"{r['tail_err']:>7.2f}pp {r['pred_snap_pct']:>6.1f}% {r['pred_tail_pct']:>6.2f}% "
              f"{r['r']:>7.3f}")

    json.dump({"real_snap_pct": real_snap, "real_tail_pct": real_tail, "rows": rows},
              open(args.out, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
