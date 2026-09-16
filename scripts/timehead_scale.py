"""Is the time head's conditional skill limited by DATA or by the frozen base features?

The bake-off showed the head architecture and loss barely matter, and that distribution match is
gameable by a lookup table. What the model uniquely provides is conditional skill - knowing which
positions deserve time - measured as correlation between E[t] and the human's actual think-time
(r ~0.55 on 48 training players).

This trains the same head on a growing number of training PLAYERS against a fixed test set:

  * if r keeps climbing, the lever is data and the corpus scrape pays off for timing too
  * if r plateaus, the frozen base representation is the ceiling and no time head can fix it -
    the next move would be a better backbone, not a better head

Test and validation players are held fixed across every size, so the only thing changing is how
many humans the head has seen. Checkpointed per size, so it resumes.

    PYTHONPATH=. python -u scripts/timehead_scale.py --sizes 12 24 48 96 192 336
"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch

from sahformer.model.timebuckets import expected_time, to_bucket
from sahformer.training.timeloss import masked_probs

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "bakeoff", os.path.join(os.path.dirname(__file__), "timehead_bakeoff.py"))
BO = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BO)
DEV = BO.DEV


@torch.no_grad()
def evaluate(head, X, t, c, pid):
    probs = torch.cat([masked_probs(head(X[s:s + 8192].to(DEV)), c[s:s + 8192].to(DEV)).cpu()
                       for s in range(0, len(X), 8192)])
    et = expected_time(probs).numpy()
    samp = BO.sample_from_probs(probs, c.numpy(), np.random.default_rng(0))
    actual = t.numpy()
    y = torch.from_numpy(to_bucket(actual)).long()
    p_cum = probs.cumsum(-1)
    idx = torch.arange(probs.shape[1])[None, :]
    rps = float(((p_cum - (idx >= y[:, None]).float()) ** 2).sum(-1).mean())
    w1 = float(np.mean([BO.wasserstein1(samp[pid.numpy() == u], actual[pid.numpy() == u])
                        for u in np.unique(pid.numpy())]))
    return {"r_Et": BO.pearson(et, actual), "spearman_Et": BO.spearman(et, actual),
            "mae_Et": float(np.abs(et - actual).mean()), "rps": rps, "w1_sampled": w1,
            "r_sampled": BO.pearson(samp, actual)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/timehead")
    ap.add_argument("--out", default="checkpoints/timehead_scale")
    ap.add_argument("--results", default="timehead_scale.json")
    ap.add_argument("--sizes", type=int, nargs="+", default=[12, 24, 48, 96, 192, 336])
    ap.add_argument("--test-players", type=int, default=20)
    ap.add_argument("--val-players", type=int, default=12)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    players = BO.load_players(args.cache)
    # Split on a hash of the player's NAME, not an index into the loaded list. An index
    # permutation depends on how many players are cached, so the test and val humans would
    # silently change every time the cache grew - and a scaling curve whose test set moves
    # underneath it measures nothing.
    def band(d):
        return int(hashlib.blake2b(d["name"].encode(), digest_size=4).hexdigest(), 16) % 1000

    # Assign by a fixed THRESHOLD on the hash, not by rank. Ranking still reshuffles when the
    # pool grows, because a new low-hash name displaces an incumbent; a threshold pins each
    # person permanently. ~5% test, ~3% val.
    te = [d for d in players if band(d) < 50]
    va = [d for d in players if 50 <= band(d) < 80]
    pool = sorted([d for d in players if band(d) >= 80], key=band)
    print(f"{len(players)} cached | test {len(te)} / val {len(va)} | train pool {len(pool)}")
    print(f"test humans: {', '.join(d['name'] for d in te[:5])} ...")

    Xte, tte, cte, pte = BO.stack(te)
    Xva, tva, cva, pva = BO.stack(va)

    rows = []
    for n in [s for s in args.sizes if s <= len(pool)]:
        ck = os.path.join(args.out, f"n{n}.pt")
        Xtr, ttr, ctr, _ = BO.stack(pool[:n])
        head = BO.make_head("bucket", Xtr.shape[1]).to(DEV)
        if os.path.exists(ck):
            head.load_state_dict(torch.load(ck, map_location=DEV, weights_only=True))
            print(f"[n={n}] loaded")
        else:
            print(f"[n={n}] training on {len(Xtr):,} positions from {n} humans...")
            head = BO.train_head("bucket", "ce", Xtr, ttr, ctr, False,
                                 args.steps, args.bs, args.lr, args.seed)
            torch.save(head.state_dict(), ck)
        head.eval()
        m = evaluate(head, Xte, tte, cte, pte)
        m.update({"n_players": n, "n_positions": int(len(Xtr)),
                  "val_r_Et": evaluate(head, Xva, tva, cva, pva)["r_Et"]})
        rows.append(m)
        print(f"  test r@E[t] {m['r_Et']:.4f} | spearman {m['spearman_Et']:.4f} | "
              f"RPS {m['rps']:.4f} | W1 {m['w1_sampled']:.3f} | MAE {m['mae_Et']:.3f}")
        json.dump(rows, open(args.results, "w", encoding="utf-8"), indent=2)

    print(f"\n{'players':>8} {'positions':>11} {'r@E[t]':>8} {'spearman':>9} {'RPS':>8} {'W1':>7}")
    for r in rows:
        print(f"{r['n_players']:>8} {r['n_positions']:>11,} {r['r_Et']:>8.4f} "
              f"{r['spearman_Et']:>9.4f} {r['rps']:>8.4f} {r['w1_sampled']:>7.3f}")
    if len(rows) >= 2:
        d = rows[-1]["r_Et"] - rows[-2]["r_Et"]
        mult = rows[-1]["n_players"] / max(rows[-2]["n_players"], 1)
        print(f"\nlast doubling ({mult:.1f}x players): r moved {d:+.4f}")
        print("plateau => the frozen base representation is the ceiling, not the head or the data")
    print(f"wrote {args.results}")


if __name__ == "__main__":
    main()
