"""Does self-consistent clock accounting fix the 11% flag rate?

Teacher-forced sampling masks each move against the REAL clock from the human's game, so the
model's own overspending never comes back to bite it - and in 10.8% of games its think-times sum
past the 180s budget, which is impossible in play.

This replays each game in order, carrying the model's OWN simulated clock: mask against what the
model has left, sample, subtract, continue. That is what free-running play would do, and it is
the cheapest possible budget mechanism - no retraining, no new head.

Reports flag rate, drift, and distribution match under three regimes:

  teacher-forced  mask on the human's real clock (what the bake-off measured)
  self-clocked    mask on the model's own remaining clock
  self+pace       self-clocked, plus a mild pace prior that discourages spending far ahead of
                  schedule (an explicit budget, rather than only a hard wall at zero)

HONEST LIMIT: the input features were encoded with the human's real temporal vector, so the
model still SEES the real clock even when masked against a simulated one. Only the mask is
self-consistent. A full simulation needs the base model re-run per ply, which the cache cannot
do. This therefore isolates one question - does budget-aware masking alone stop the overspend -
and cannot tell us how the model would behave if it also perceived its own clock.

    PYTHONPATH=. python -u scripts/timehead_budget.py
"""
import argparse
import json
import os

import numpy as np
import torch

from sahformer.model.timebuckets import EDGES, N_BUCKETS
from sahformer.training.timeloss import masked_probs

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "bakeoff", os.path.join(os.path.dirname(__file__), "timehead_bakeoff.py"))
BO = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BO)
DEV = BO.DEV

def load_split_players(cache_dir, split_path="cache/timehead_split.json", which="test"):
    """Load exactly the players a head was held out from, BY NAME.

    Never select the test set with permutation(len(players)): the cache grows, the permutation
    changes, and the "held-out" set silently becomes partly training data. That happened - 4 of
    20 supposedly-held-out players in the first drift run were training players.
    """
    import json as _json
    names = set(_json.load(open(split_path, encoding="utf-8"))[which])
    got = [d for d in BO.load_players(cache_dir) if d["name"] in names]
    missing = names - {d["name"] for d in got}
    if missing:
        raise SystemExit(f"split lists {len(missing)} players not in the cache: {sorted(missing)[:3]}")
    return got

BUDGET = 180.0
LO = np.asarray(EDGES[:-1])
HI = np.asarray([e if np.isfinite(e) else 60.0 for e in EDGES[1:]])


def draw(p_row, clock, rng, pace_penalty=0.0, moves_left=None):
    """Sample one think-time from a bucket row, masked by `clock`.

    pace_penalty>0 multiplies down buckets that would spend more than an even share of the
    remaining clock, so the model is nudged to hold something back rather than only being
    stopped by the wall at zero.
    """
    p = p_row.copy()
    p *= (LO < clock)
    if pace_penalty > 0 and moves_left and moves_left > 0:
        share = clock / moves_left
        over = np.maximum(LO / max(share, 1e-6), 1.0)
        p *= np.exp(-pace_penalty * (over - 1.0))
    s = p.sum()
    if s <= 0:
        return float(max(min(clock, 0.1), 0.0))
    p /= s
    i = int((p.cumsum() < rng.random()).sum())
    i = min(i, N_BUCKETS - 1)
    a, b = LO[i], min(HI[i], clock)
    return float(a if b <= a else rng.uniform(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/timehead")
    ap.add_argument("--ckpt", default="checkpoints/timehead/bucket-brier.pt")
    ap.add_argument("--out", default="timehead_budget.json")
    ap.add_argument("--test-players", type=int, default=20)
    ap.add_argument("--pace", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    te = load_split_players(args.cache)
    Xp, _, _, _ = BO.stack(te)
    head = BO.make_head("bucket", Xp.shape[1]).to(DEV)
    head.load_state_dict(torch.load(args.ckpt, map_location=DEV, weights_only=True))
    head.eval()

    modes = ["teacher-forced", "self-clocked", "self+pace"]
    tot = {m: [] for m in modes}
    allt = {m: [] for m in modes}
    actual_tot, actual_all = [], []

    for d in te:
        X = torch.from_numpy(np.concatenate(
            [d["pooled"].astype(np.float32), d["diff"].astype(np.float32)], axis=1))
        t, c, gid = d["think"], d["my_clock"], d["game_id"]
        with torch.no_grad():
            pr = torch.cat([masked_probs(head(X[s:s + 8192].to(DEV)),
                                         torch.from_numpy(c[s:s + 8192]).to(DEV)).cpu()
                            for s in range(0, len(X), 8192)]).numpy().astype(np.float64)
        for g in np.unique(gid):
            m = np.where(gid == g)[0]
            if len(m) < 8:
                continue
            actual_tot.append(float(t[m].sum())); actual_all.extend(t[m].tolist())
            for mode in modes:
                rng = np.random.default_rng(args.seed + int(g))
                clock, spent, seq = float(c[m[0]]), 0.0, []
                for k, idx in enumerate(m):
                    use = float(c[idx]) if mode == "teacher-forced" else clock
                    x = draw(pr[idx], max(use, 0.0), rng,
                             pace_penalty=(args.pace if mode == "self+pace" else 0.0),
                             moves_left=max(len(m) - k, 1))
                    seq.append(x); spent += x; clock = max(clock - x, 0.0)
                tot[mode].append(spent); allt[mode].extend(seq)

    at = np.array(actual_tot); aa = np.array(actual_all)
    print(f"{len(at):,} games | human total {at.mean():.1f}s/game\n")
    print(f"{'mode':<16} {'total/game':>11} {'drift':>8} {'FLAG %':>8} {'snap%':>7} "
          f"{'tail%':>7} {'W1':>7}")
    print(f"{'human':<16} {at.mean():>10.1f}s {'':>8} {100*(at>BUDGET).mean():>7.2f}% "
          f"{100*(aa<1).mean():>6.1f}% {100*(aa>10).mean():>6.2f}% {'':>7}")
    res = {}
    for mode in modes:
        tt = np.array(tot[mode]); sa = np.array(allt[mode])
        w1 = BO.wasserstein1(sa, aa)
        res[mode] = {"total": float(tt.mean()), "drift": float(tt.mean() - at.mean()),
                     "flag_pct": float(100 * (tt > BUDGET).mean()),
                     "snap_pct": float(100 * (sa < 1).mean()),
                     "tail_pct": float(100 * (sa > 10).mean()), "w1": float(w1)}
        r = res[mode]
        print(f"{mode:<16} {r['total']:>10.1f}s {r['drift']:>+7.1f}s {r['flag_pct']:>7.2f}% "
              f"{r['snap_pct']:>6.1f}% {r['tail_pct']:>6.2f}% {w1:>7.3f}")

    json.dump({"human_total": float(at.mean()), "human_snap": float(100 * (aa < 1).mean()),
               "human_tail": float(100 * (aa > 10).mean()), "modes": res},
              open(args.out, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
