"""Search readouts for one that matches the DISTRIBUTION and the MEDIAN at once.

Everything measured so far tracked the tails - snap rate (<1s) and tank rate (>10s) - plus W1 and
correlation. Nobody checked the middle. A readout can hit both tails and still put the median in
the wrong place, which is what a player would actually notice first: every move feeling a beat too
slow or too fast.

Sweeps every trained head against a family of readouts:

  E[t]              the distribution's mean (what ChessMimic uses)
  median            the distribution's 50th percentile - for a right-skewed law this sits well
                    below the mean, so it is a genuinely different point estimate
  q<p>              other fixed quantiles, to see whether any point estimate can carry a shape
  sample T=<t>      tempered sampling
  self-clocked      sampling that masks against the model's OWN depleting clock (the fix that
                    removed the 12.8% clock overruns)
  blend a=<a>       a*sample + (1-a)*E[t] - trades distributional spread for point accuracy

Reported per readout: median error, quartile errors, snap and tail error, W1 per player, and r.
Ranked by a combined distribution score so that "matches the shape" and "matches the middle" have
to hold together.

    PYTHONPATH=. python -u scripts/timehead_readout_search.py
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from sahformer.model.timebuckets import EDGES, N_BUCKETS, expected_time
from sahformer.training.timeloss import masked_probs

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "bakeoff", os.path.join(os.path.dirname(__file__), "timehead_bakeoff.py"))
BO = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BO)
DEV = BO.DEV
LO = np.asarray(EDGES[:-1])
HI = np.asarray([e if np.isfinite(e) else 60.0 for e in EDGES[1:]])


def quantile_readout(probs, q):
    """The q-quantile of each row's bucket distribution, interpolated inside the bucket."""
    p = probs.numpy().astype(np.float64)
    cdf = p.cumsum(1)
    idx = (cdf < q).sum(1).clip(0, N_BUCKETS - 1)
    below = np.where(idx > 0, cdf[np.arange(len(p)), np.maximum(idx - 1, 0)], 0.0)
    mass = p[np.arange(len(p)), idx]
    frac = np.where(mass > 1e-12, (q - below) / np.maximum(mass, 1e-12), 0.5).clip(0, 1)
    return LO[idx] + frac * (np.minimum(HI[idx], 60.0) - LO[idx])


def sample_T(probs, clock, T, rng):
    if T <= 1e-6:
        p = torch.zeros_like(probs).scatter_(1, probs.argmax(1, keepdim=True), 1.0)
    else:
        p = torch.softmax(torch.log(probs.clamp_min(1e-12)) / T, dim=-1)
    return BO.sample_from_probs(p, clock, rng)


def self_clocked(probs, clock, gid, rng):
    """Sample carrying the model's own depleting clock rather than the human's."""
    p = probs.numpy().astype(np.float64)
    out = np.zeros(len(p))
    for g in np.unique(gid):
        m = np.where(gid == g)[0]
        c = float(clock[m[0]])
        for i in m:
            row = p[i] * (LO < c)
            s = row.sum()
            if s <= 0:
                x = max(min(c, 0.1), 0.0)
            else:
                row = row / s
                k = min(int((row.cumsum() < rng.random()).sum()), N_BUCKETS - 1)
                a, b = LO[k], min(HI[k], c)
                x = a if b <= a else rng.uniform(a, b)
            out[i] = x
            c = max(c - x, 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/timehead")
    ap.add_argument("--ckpts", default="checkpoints/timehead")
    ap.add_argument("--split", default="cache/timehead_split.json")
    ap.add_argument("--out", default="timehead_readout_search.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names = set(json.load(open(args.split, encoding="utf-8"))["test"])
    te = [d for d in BO.load_players(args.cache) if d["name"] in names]
    print(f"{len(te)} held-out humans (pinned split)\n")

    rows = []
    for ck in sorted(glob.glob(os.path.join(args.ckpts, "*.pt"))):
        head_name = os.path.basename(ck)[:-3]
        kind = ("mdn" if head_name.startswith("mdn") else
                "hurdle" if head_name.startswith("hurdle") else
                "ordinal" if head_name.startswith("ordinal") else "bucket")
        if kind == "mdn":
            continue                       # no bucket distribution to take quantiles of
        use_diff = "nodiff" not in head_name

        A, C, G, P = [], [], [], []
        Xs, _, _, _ = BO.stack(te, use_diff=use_diff)
        head = BO.make_head(kind, Xs.shape[1]).to(DEV)
        head.load_state_dict(torch.load(ck, map_location=DEV, weights_only=True))
        head.eval()
        off = 0
        for d in te:
            f = [d["pooled"].astype(np.float32)]
            if use_diff:
                f.append(d["diff"].astype(np.float32))
            X = torch.from_numpy(np.concatenate(f, 1))
            with torch.no_grad():
                pr = torch.cat([masked_probs(head(X[s:s + 8192].to(DEV)),
                                             torch.from_numpy(d["my_clock"][s:s + 8192]).to(DEV)).cpu()
                                for s in range(0, len(X), 8192)])
            A.append(d["think"]); C.append(d["my_clock"]); P.append(pr)
            G.append(d["game_id"].astype(np.int64) + off * 100000); off += 1
        actual = np.concatenate(A); clock = np.concatenate(C)
        gid = np.concatenate(G); probs = torch.cat(P)
        pid = np.concatenate([np.full(len(d["think"]), i) for i, d in enumerate(te)])
        et = expected_time(probs).numpy()

        readouts = {"E[t]": lambda: et,
                    "median": lambda: quantile_readout(probs, 0.5)}
        for q in (0.35, 0.45, 0.55, 0.65):
            readouts[f"q{q:.2f}"] = (lambda q=q: quantile_readout(probs, q))
        for T in (0.75, 1.0, 1.25):
            readouts[f"sample T={T}"] = (lambda T=T: sample_T(
                probs, clock, T, np.random.default_rng(args.seed)))
        readouts["self-clocked"] = lambda: self_clocked(
            probs, clock, gid, np.random.default_rng(args.seed))
        for a in (0.5, 0.75):
            readouts[f"blend a={a}"] = (lambda a=a: a * sample_T(
                probs, clock, 1.0, np.random.default_rng(args.seed)) + (1 - a) * et)

        for label, fn in readouts.items():
            pred = np.asarray(fn(), dtype=np.float64)
            med_h, med_m = np.median(actual), np.median(pred)
            pp_med = np.mean([abs(np.median(pred[pid == u]) - np.median(actual[pid == u]))
                              for u in np.unique(pid)])
            w1 = np.mean([BO.wasserstein1(pred[pid == u], actual[pid == u])
                          for u in np.unique(pid)])
            rows.append({
                "head": head_name, "readout": label,
                "median_h": float(med_h), "median_m": float(med_m),
                "median_err": float(med_m - med_h), "pp_median_err": float(pp_med),
                "q25_err": float(np.percentile(pred, 25) - np.percentile(actual, 25)),
                "q75_err": float(np.percentile(pred, 75) - np.percentile(actual, 75)),
                "snap_err": float((pred < 1).mean() * 100 - (actual < 1).mean() * 100),
                "tail_err": float((pred > 10).mean() * 100 - (actual > 10).mean() * 100),
                "w1": float(w1), "r": BO.pearson(pred, actual)})

    # combined: median must be right AND the shape must be right
    for r in rows:
        r["score"] = abs(r["pp_median_err"]) + r["w1"] + abs(r["snap_err"]) / 20 + abs(r["tail_err"]) / 5

    print(f"human median {rows[0]['median_h']:.2f}s\n")
    print(f"{'head':<20} {'readout':<14} {'med':>6} {'medErr':>7} {'ppMed':>6} {'q25':>6} "
          f"{'q75':>6} {'snapE':>6} {'tailE':>6} {'W1':>6} {'r':>6}")
    for r in sorted(rows, key=lambda x: x["score"])[:22]:
        print(f"{r['head']:<20} {r['readout']:<14} {r['median_m']:>6.2f} {r['median_err']:>+7.2f} "
              f"{r['pp_median_err']:>6.2f} {r['q25_err']:>+6.2f} {r['q75_err']:>+6.2f} "
              f"{r['snap_err']:>+6.1f} {r['tail_err']:>+6.2f} {r['w1']:>6.3f} {r['r']:>6.3f}")

    json.dump(rows, open(args.out, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {args.out}  ({len(rows)} head x readout combinations)")


if __name__ == "__main__":
    main()
