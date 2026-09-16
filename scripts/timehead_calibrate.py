"""How close can we actually get to the real time distribution?

Two questions the bake-off left open.

1. **The sampling temperature.** Sampled draws over-snap slightly (19-21% against a real 17.2%).
   Temperature interpolates between argmax (T->0, which is the E[t]-style collapse) and the raw
   distribution (T=1) and beyond. Sweeping it maps the trade-off frontier between distribution
   match (W1) and conditional skill (r) explicitly, instead of picking one end by assumption.
   T is chosen on VAL players and reported on TEST players.

2. **Does it track the individual, or just predict the population mean?** Snap rate ranges
   5.5%-31.7% across humans in this cache. A model that emitted the population average for
   everyone would still post a decent average error while being useless for a clone. The test
   is the per-player regression of predicted snap rate on actual: slope ~1 means it tracks the
   person, slope ~0 means it has memorised one number and is blind to who is playing.

    PYTHONPATH=. python -u scripts/timehead_calibrate.py
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from sahformer.training.timeloss import masked_probs

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "bakeoff", os.path.join(os.path.dirname(__file__), "timehead_bakeoff.py"))
BO = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BO)
DEV = BO.DEV


@torch.no_grad()
def probs_for(head, X, clock, chunk=8192):
    out = []
    for s in range(0, len(X), chunk):
        out.append(masked_probs(head(X[s:s + chunk].to(DEV)), clock[s:s + chunk].to(DEV)).cpu())
    return torch.cat(out)


def sample_at_T(probs, clock, T, rng):
    """Re-temper an already-masked distribution, then sample. T->0 becomes argmax."""
    if T <= 1e-6:
        p = torch.zeros_like(probs).scatter_(1, probs.argmax(1, keepdim=True), 1.0)
    else:
        lg = torch.log(probs.clamp_min(1e-12)) / T
        p = torch.softmax(lg, dim=-1)
    return BO.sample_from_probs(p, clock, rng)


def per_player(pred, actual, pid, fn):
    return np.array([fn(pred[pid == u], actual[pid == u]) for u in np.unique(pid)])


def w1_mean(pred, actual, pid):
    return float(per_player(pred, actual, pid, lambda a, b: BO.wasserstein1(a, b)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/timehead")
    ap.add_argument("--ckpts", default="checkpoints/timehead")
    ap.add_argument("--heads", nargs="+", default=["bucket-brier", "bucket-ce", "hurdle-rps"])
    ap.add_argument("--out", default="timehead_calibrate.json")
    ap.add_argument("--test-players", type=int, default=20)
    ap.add_argument("--val-players", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    players = BO.load_players(args.cache)
    order = np.random.default_rng(args.seed).permutation(len(players))
    te = [players[i] for i in order[:args.test_players]]
    va = [players[i] for i in order[args.test_players:args.test_players + args.val_players]]

    Xva, tva, cva, pva = BO.stack(va)
    Xte, tte, cte, pte = BO.stack(te)
    real_snap = float((tte.numpy() < 1.0).mean() * 100)
    print(f"val {len(va)} / test {len(te)} humans | test real snap {real_snap:.1f}%  "
          f"tail {float((tte.numpy()>10).mean()*100):.2f}%\n")

    Ts = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    report = {}
    for name in args.heads:
        ck = os.path.join(args.ckpts, f"{name}.pt")
        if not os.path.exists(ck):
            continue
        head = BO.make_head("hurdle" if name.startswith("hurdle") else "bucket", Xte.shape[1]).to(DEV)
        head.load_state_dict(torch.load(ck, map_location=DEV, weights_only=True))
        head.eval()
        pv, pt = probs_for(head, Xva, cva), probs_for(head, Xte, cte)

        print(f"=== {name} ===")
        print(f"{'T':>5} {'valW1':>7} | {'testW1':>7} {'snap%':>7} {'tail%':>7} {'r':>7}")
        rows, best = [], (None, 9e9)
        for T in Ts:
            sv = sample_at_T(pv, cva.numpy(), T, np.random.default_rng(args.seed))
            st = sample_at_T(pt, cte.numpy(), T, np.random.default_rng(args.seed))
            vw = w1_mean(sv, tva.numpy(), pva.numpy())
            tw = w1_mean(st, tte.numpy(), pte.numpy())
            row = {"T": T, "val_w1": vw, "test_w1": tw,
                   "snap_pct": float((st < 1).mean() * 100),
                   "tail_pct": float((st > 10).mean() * 100),
                   "r": BO.pearson(st, tte.numpy())}
            rows.append(row)
            print(f"{T:>5.2f} {vw:>7.3f} | {tw:>7.3f} {row['snap_pct']:>6.1f}% "
                  f"{row['tail_pct']:>6.2f}% {row['r']:>7.3f}")
            if vw < best[1]:
                best = (T, vw)
        print(f"  -> T selected on val: {best[0]}")

        # does it track the individual, or emit one number for everyone?
        st = sample_at_T(pt, cte.numpy(), best[0], np.random.default_rng(args.seed))
        pid = pte.numpy(); act = tte.numpy()
        a_snap = per_player(st, act, pid, lambda a, b: (b < 1).mean() * 100)
        p_snap = per_player(st, act, pid, lambda a, b: (a < 1).mean() * 100)
        slope, icpt = np.polyfit(a_snap, p_snap, 1)
        r_snap = BO.pearson(a_snap, p_snap)
        print(f"  per-player snap: actual {a_snap.min():.1f}-{a_snap.max():.1f}%  "
              f"predicted {p_snap.min():.1f}-{p_snap.max():.1f}%")
        print(f"  tracking: slope {slope:.2f} (1.0 = tracks the person, 0.0 = predicts the mean), "
              f"r {r_snap:.3f}\n")
        report[name] = {"rows": rows, "T": best[0], "snap_slope": float(slope),
                        "snap_r": float(r_snap), "actual_snap": a_snap.tolist(),
                        "pred_snap": p_snap.tolist()}

    json.dump(report, open(args.out, "w", encoding="utf-8"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
