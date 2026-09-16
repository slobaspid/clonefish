"""Does a tiny per-player calibration close the under-dispersion gap?

The population head ranks people well (per-player snap rate r=0.77) but captures only ~39% of
the real spread: a 29.6% snapper is predicted at 25.6%, a 6.2% snapper at 13.4%. Everyone's
clone drifts toward the average player, which is the exact opposite of what clonefish sells.

This fits TWO scalars per person - a snap-bucket logit bias and a sampling temperature - on
their earlier games, and scores them on their later ones. Two parameters is deliberately
almost nothing: if two scalars close most of the gap, the population model already knows the
shape and is only mis-scaled, and no architecture change is needed.

Splits are CHRONOLOGICAL within each player (calibrate on early games, score on later ones),
never random, because a player's tempo drifts and a random split would leak the future.

    PYTHONPATH=. python -u scripts/timehead_perplayer.py
"""
import argparse
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


def adjust(probs, bias, T):
    """Re-temper and shift snap mass, then renormalise. bias>0 = snap more."""
    lg = torch.log(probs.clamp_min(1e-12))
    lg[:, 0] += bias
    return torch.softmax(lg / max(T, 1e-3), dim=-1)


def fit_player(probs, clock, actual, seed=0, biases=None, temps=None):
    """Grid-search the two scalars on this player's CALIBRATION games only."""
    biases = biases if biases is not None else np.arange(-2.0, 2.01, 0.25)
    temps = temps if temps is not None else np.arange(0.8, 1.45, 0.05)
    best = (0.0, 1.0, 9e9)
    for b in biases:
        for T in temps:
            s = BO.sample_from_probs(adjust(probs, float(b), float(T)), clock,
                                     np.random.default_rng(seed))
            w = BO.wasserstein1(s, actual)
            if w < best[2]:
                best = (float(b), float(T), w)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/timehead")
    ap.add_argument("--ckpt", default="checkpoints/timehead/bucket-brier.pt")
    ap.add_argument("--kind", default="bucket")
    ap.add_argument("--out", default="timehead_perplayer.json")
    ap.add_argument("--test-players", type=int, default=20)
    ap.add_argument("--calib-frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    players = BO.load_players(args.cache)
    order = np.random.default_rng(args.seed).permutation(len(players))
    te = [players[i] for i in order[:args.test_players]]

    Xall, _, _, _ = BO.stack(te)
    head = BO.make_head(args.kind, Xall.shape[1]).to(DEV)
    head.load_state_dict(torch.load(args.ckpt, map_location=DEV, weights_only=True))
    head.eval()

    rows = []
    for d in te:
        X = torch.from_numpy(np.concatenate(
            [d["pooled"].astype(np.float32), d["diff"].astype(np.float32)], axis=1))
        t, c, gid = d["think"], d["my_clock"], d["game_id"]
        cut = int(gid.max() * args.calib_frac)
        cal, tst = gid <= cut, gid > cut          # chronological, by GAME
        if tst.sum() < 400 or cal.sum() < 400:
            continue
        with torch.no_grad():
            pr = torch.cat([masked_probs(head(X[s:s + 8192].to(DEV)),
                                         torch.from_numpy(c[s:s + 8192]).to(DEV)).cpu()
                            for s in range(0, len(X), 8192)])

        rng = np.random.default_rng(args.seed)
        base = BO.sample_from_probs(pr[tst], c[tst], rng)
        b, T, _ = fit_player(pr[cal], c[cal], t[cal], seed=args.seed)
        tuned = BO.sample_from_probs(adjust(pr[tst], b, T), c[tst], np.random.default_rng(args.seed))

        rows.append({
            "player": d["name"], "n_test": int(tst.sum()), "bias": b, "T": T,
            "real_snap": float((t[tst] < 1).mean() * 100),
            "base_snap": float((base < 1).mean() * 100),
            "tuned_snap": float((tuned < 1).mean() * 100),
            "real_tail": float((t[tst] > 10).mean() * 100),
            "base_tail": float((base > 10).mean() * 100),
            "tuned_tail": float((tuned > 10).mean() * 100),
            "base_w1": BO.wasserstein1(base, t[tst]),
            "tuned_w1": BO.wasserstein1(tuned, t[tst]),
        })

    bw = np.array([r["base_w1"] for r in rows])
    tw = np.array([r["tuned_w1"] for r in rows])
    rs = np.array([r["real_snap"] for r in rows])
    bs = np.array([r["base_snap"] for r in rows])
    ts_ = np.array([r["tuned_snap"] for r in rows])

    print(f"{len(rows)} players, calibrated on first {int(args.calib_frac*100)}% of each "
          f"player's games, scored on their LATER games\n")
    print(f"{'player':<22} {'realSnap':>9} {'base':>7} {'tuned':>7} | {'baseW1':>7} {'tunedW1':>8}")
    for r in sorted(rows, key=lambda x: x["real_snap"]):
        print(f"{r['player'][:22]:<22} {r['real_snap']:>8.1f}% {r['base_snap']:>6.1f}% "
              f"{r['tuned_snap']:>6.1f}% | {r['base_w1']:>7.3f} {r['tuned_w1']:>8.3f}")

    def slope(pred):
        return float(np.polyfit(rs, pred, 1)[0])

    print(f"\nW1        base {bw.mean():.3f}  ->  tuned {tw.mean():.3f}  "
          f"({100*(bw.mean()-tw.mean())/bw.mean():+.1f}%)")
    print(f"improved for {int((tw < bw).sum())}/{len(rows)} players")
    print(f"snap slope (1.0 = tracks the person): base {slope(bs):.2f}  ->  tuned {slope(ts_):.2f}")
    print(f"snap spread: real {rs.min():.1f}-{rs.max():.1f}%  base {bs.min():.1f}-{bs.max():.1f}%  "
          f"tuned {ts_.min():.1f}-{ts_.max():.1f}%")
    json.dump(rows, open(args.out, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
