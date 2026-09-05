"""Where does the +2.5s/game come from?

The headline drift is small, but it sits next to a contradiction: the model spends MORE total
time per game than the human (118.3s vs 115.9s) while ending under 10s LESS often (15.3% vs
18.7%). A uniform shift cannot do both. So the per-game total distribution must be shifted up in
the body and compressed at the top - the model over-spends in ordinary games and under-spends in
exactly the games where the human burned the clock.

Four decompositions:

  A. per PLAYER - is +2.5s uniform, or the average of large opposing biases?
  B. quantiles of per-game total - where does the distribution actually differ?
  C. per BUCKET, in seconds contributed - which think-lengths are over/under produced?
  D. by clock regime - does the drift come from calm play or from time pressure?

    PYTHONPATH=. python -u scripts/timehead_drift_decomp.py
"""
import argparse
import json
import os

import numpy as np
import torch

from sahformer.model.timebuckets import EDGES, N_BUCKETS, to_bucket
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
    ap.add_argument("--out", default="timehead_drift_decomp.json")
    ap.add_argument("--test-players", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    te = load_split_players(args.cache)
    Xp, _, _, _ = BO.stack(te)
    head = BO.make_head("bucket", Xp.shape[1]).to(DEV)
    head.load_state_dict(torch.load(args.ckpt, map_location=DEV, weights_only=True))
    head.eval()

    P = []       # per player
    A, S, C = [], [], []       # pooled actual, sampled, clock
    tot_a, tot_p = [], []
    for d in te:
        X = torch.from_numpy(np.concatenate(
            [d["pooled"].astype(np.float32), d["diff"].astype(np.float32)], axis=1))
        t, c, gid = d["think"], d["my_clock"], d["game_id"]
        with torch.no_grad():
            pr = torch.cat([masked_probs(head(X[s:s + 8192].to(DEV)),
                                         torch.from_numpy(c[s:s + 8192]).to(DEV)).cpu()
                            for s in range(0, len(X), 8192)])
        samp = BO.sample_from_probs(pr, c, np.random.default_rng(args.seed))
        A.append(t); S.append(samp); C.append(c)
        ga = np.array([t[gid == g].sum() for g in np.unique(gid) if (gid == g).sum() >= 8])
        gp = np.array([samp[gid == g].sum() for g in np.unique(gid) if (gid == g).sum() >= 8])
        tot_a.append(ga); tot_p.append(gp)
        P.append({"player": d["name"], "n": len(ga), "drift": float((gp - ga).mean()),
                  "actual": float(ga.mean()), "pred": float(gp.mean())})

    A = np.concatenate(A); S = np.concatenate(S); C = np.concatenate(C)
    ta = np.concatenate(tot_a); tp = np.concatenate(tot_p)

    # ---- A. per player
    dr = np.array([p["drift"] for p in P])
    print(f"A. PER-PLAYER DRIFT  (mean {dr.mean():+.2f}s)")
    print(f"   range {dr.min():+.1f}s to {dr.max():+.1f}s | sd {dr.std():.1f}s | "
          f"{int((dr > 0).sum())}/{len(dr)} players over-spend")
    for p in sorted(P, key=lambda x: x["drift"])[:3] + sorted(P, key=lambda x: -x["drift"])[:3]:
        print(f"     {p['player'][:24]:<24} {p['drift']:+7.1f}s  "
              f"(actual {p['actual']:5.1f}s -> pred {p['pred']:5.1f}s)")

    # ---- B. quantiles of per-game total
    print(f"\nB. PER-GAME TOTAL, BY QUANTILE")
    print(f"   {'pct':>5} {'human':>8} {'model':>8} {'drift':>8}")
    qs = [5, 10, 25, 50, 75, 90, 95, 99]
    quant = []
    for q in qs:
        ha, mo = np.percentile(ta, q), np.percentile(tp, q)
        quant.append({"q": q, "human": float(ha), "model": float(mo)})
        print(f"   {q:>4}% {ha:>7.1f}s {mo:>7.1f}s {mo-ha:>+7.1f}s")
    print(f"   spread p5-p95: human {np.percentile(ta,95)-np.percentile(ta,5):.1f}s  "
          f"model {np.percentile(tp,95)-np.percentile(tp,5):.1f}s")

    # ---- C. seconds contributed per bucket
    ba, bs = to_bucket(A), to_bucket(S)
    print(f"\nC. SECONDS CONTRIBUTED PER BUCKET  (total over all test positions, per game)")
    ngames = len(ta)
    print(f"   {'bucket':>12} {'human s/g':>10} {'model s/g':>10} {'drift':>9} {'moves/g':>9}")
    rows = []
    groups = [(0, 1, "<1s snap"), (1, 3, "1-3s"), (3, 6, "3-6s"), (6, 11, "6-11s"),
              (11, 21, "11-21s"), (21, N_BUCKETS, "21s+ tank")]
    for lo, hi, lab in groups:
        ma = ((ba >= lo) & (ba < hi)); ms = ((bs >= lo) & (bs < hi))
        ha, mo = A[ma].sum() / ngames, S[ms].sum() / ngames
        dn = (ms.sum() - ma.sum()) / ngames
        rows.append({"bucket": lab, "human_s": float(ha), "model_s": float(mo),
                     "drift_s": float(mo - ha), "move_drift": float(dn)})
        print(f"   {lab:>12} {ha:>9.2f}s {mo:>9.2f}s {mo-ha:>+8.2f}s {dn:>+8.2f}")
    print(f"   {'TOTAL':>12} {A.sum()/ngames:>9.2f}s {S.sum()/ngames:>9.2f}s "
          f"{(S.sum()-A.sum())/ngames:>+8.2f}s")

    # ---- D. by clock regime
    print(f"\nD. DRIFT BY CLOCK REGIME  (seconds per game contributed)")
    regimes = [(120, 181, ">120s calm"), (60, 120, "60-120s"), (20, 60, "20-60s"),
               (10, 20, "10-20s"), (0, 10, "<10s scramble")]
    drows = []
    for lo, hi, lab in regimes:
        m = (C >= lo) & (C < hi)
        ha, mo = A[m].sum() / ngames, S[m].sum() / ngames
        drows.append({"regime": lab, "human_s": float(ha), "model_s": float(mo),
                      "drift_s": float(mo - ha)})
        print(f"   {lab:>14} {ha:>8.2f}s {mo:>8.2f}s {mo-ha:>+8.2f}s   "
              f"({100*m.mean():4.1f}% of moves)")

    json.dump({"per_player": P, "quantiles": quant, "buckets": rows, "regimes": drows},
              open(args.out, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
