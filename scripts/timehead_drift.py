"""Clock drift: do per-move timing errors cancel, or compound across a game?

Per-position metrics (W1, snap rate) can look healthy while the model still spends the wrong
TOTAL amount of time. A 2pp over-snap on every move compounds over ~36 moves into a clone that
finishes with a minute to spare and never feels time pressure - which is exactly the "human
feel" the whole time head exists for.

Measures, per game, against the same held-out humans as the bake-off:

  * total time spent, predicted vs actual
  * signed drift (seconds and as a share of the 180s budget)
  * where the clock trajectory has diverged by each quarter of the game
  * end-of-game clock: how often the model would still be flush when the human was scrambling

CAVEAT, and it matters: predictions here are TEACHER-FORCED. Each one is conditioned on the
real game's clock and the human's real last-5 think-times, so the input never drifts even when
the output does. Free-running self-play would compound further, because the model's own wrong
clock would feed the next prediction. Treat these as a LOWER BOUND.

    PYTHONPATH=. python -u scripts/timehead_drift.py
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/timehead")
    ap.add_argument("--ckpt", default="checkpoints/timehead/bucket-brier.pt")
    ap.add_argument("--kind", default="bucket")
    ap.add_argument("--out", default="timehead_drift.json")
    ap.add_argument("--test-players", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    te = load_split_players(args.cache)

    Xp, _, _, _ = BO.stack(te)
    head = BO.make_head(args.kind, Xp.shape[1]).to(DEV)
    head.load_state_dict(torch.load(args.ckpt, map_location=DEV, weights_only=True))
    head.eval()

    per_game, quart = [], [[], [], [], []]
    for d in te:
        X = torch.from_numpy(np.concatenate(
            [d["pooled"].astype(np.float32), d["diff"].astype(np.float32)], axis=1))
        t, c, gid = d["think"], d["my_clock"], d["game_id"]
        with torch.no_grad():
            pr = torch.cat([masked_probs(head(X[s:s + 8192].to(DEV)),
                                         torch.from_numpy(c[s:s + 8192]).to(DEV)).cpu()
                            for s in range(0, len(X), 8192)])
        samp = BO.sample_from_probs(pr, c, np.random.default_rng(args.seed))

        for g in np.unique(gid):
            m = gid == g
            if m.sum() < 8:
                continue
            a, p = t[m], samp[m]
            per_game.append({"player": d["name"], "plies": int(m.sum()),
                             "actual_total": float(a.sum()), "pred_total": float(p.sum())})
            ca, cp = np.cumsum(a), np.cumsum(p)
            for q in range(4):
                i = min(int(len(a) * (q + 1) / 4) - 1, len(a) - 1)
                quart[q].append(cp[i] - ca[i])

    at = np.array([r["actual_total"] for r in per_game])
    pt = np.array([r["pred_total"] for r in per_game])
    drift = pt - at

    print(f"{len(per_game):,} games from {len(te)} held-out humans\n")
    print(f"total time spent per game   actual {at.mean():6.1f}s   predicted {pt.mean():6.1f}s")
    print(f"drift                       {drift.mean():+6.1f}s per game "
          f"({100*drift.mean()/BUDGET:+.1f}% of the 180s budget)")
    print(f"                            median {np.median(drift):+.1f}s | "
          f"abs mean {np.abs(drift).mean():.1f}s | sd {drift.std():.1f}s")
    print(f"\ncumulative divergence by point in game (signed, seconds):")
    for q in range(4):
        v = np.array(quart[q])
        print(f"  after {25*(q+1):>3}% of plies   {v.mean():+6.2f}s   (abs {np.abs(v).mean():5.2f}s)")

    print(f"\nend-of-game clock (180s budget):")
    print(f"  human left  {BUDGET - at.mean():6.1f}s on average | "
          f"scrambles (<10s left) in {100*(BUDGET-at > 0).mean()*0 + 100*((BUDGET-at) < 10).mean():.1f}% of games")
    print(f"  model left  {BUDGET - pt.mean():6.1f}s on average | "
          f"scrambles (<10s left) in {100*((BUDGET-pt) < 10).mean():.1f}% of games")
    over = 100 * (drift > 0).mean()
    print(f"\nmodel spent MORE time than the human in {over:.1f}% of games")

    json.dump({"n_games": len(per_game), "drift_mean_s": float(drift.mean()),
               "drift_median_s": float(np.median(drift)),
               "drift_abs_mean_s": float(np.abs(drift).mean()),
               "actual_total_mean_s": float(at.mean()), "pred_total_mean_s": float(pt.mean()),
               "quartile_divergence_s": [float(np.mean(q)) for q in quart],
               "human_scramble_pct": float(100 * ((BUDGET - at) < 10).mean()),
               "model_scramble_pct": float(100 * ((BUDGET - pt) < 10).mean())},
              open(args.out, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
