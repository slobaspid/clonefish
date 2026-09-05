"""Does letting gradients into the BACKBONE break the r=0.575 ceiling?

Every head in this session's bake-off trained on FROZEN features from base_300k_best.pt, and the
scaling probe concluded "the base representation is the ceiling". That conclusion has a hole: a
frozen backbone was never given a reason to encode timing-relevant structure. If it is discarding
information that a jointly-trained backbone would keep, the ceiling is an artifact of the setup
rather than a property of the model.

Two arms, identical in every other respect - same players, same split, same steps, same schedule:

  frozen  backbone requires_grad=False, only the time head learns  (reproduces the ceiling)
  joint   backbone trains too

Trained on the SAME 48 train players the frozen heads used, not on data/chesscom_balanced_shards
- those shards are the leaderboard-seeded corpus (elo 2012-3074, median think 1.40s, snap 35.4%)
against test players at median 2.10s / snap 17.2%, so training there would confound capability
with domain shift.

Positions are the crawled player's own moves only, matching how the cached features were built.
Policy and value losses are kept on in the joint arm so the backbone cannot simply forget how to
play chess in order to fit think-time.

    PYTHONPATH=. python -u scripts/timehead_jointtrain.py --arm joint
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

from sahformer.download import iter_games_from_zst
from sahformer.model.heads import policy_difficulty
from sahformer.model.timebuckets import N_BUCKETS, expected_time, to_bucket
from sahformer.records import game_to_records, player_id_of
from sahformer.shards import records_to_arrays
from sahformer.training.loop import load_model
from sahformer.training.losses import move_target_index
from sahformer.training.timeloss import masked_probs

DEV = "cuda" if torch.cuda.is_available() else "cpu"
KEYS = ("board", "history", "elo_self", "elo_opp", "temporal")


def modal_player(path, probe=40):
    import collections
    f = collections.Counter()
    for i, g in enumerate(iter_games_from_zst(path)):
        for s in ("White", "Black"):
            p = player_id_of(g.headers.get(s, ""))
            if p:
                f[p] += 1
        if i + 1 >= probe:
            break
    return f.most_common(1)[0][0] if f else 0


def build_cache(names, pgn_dir, out, max_games=200):
    """Encode the crawled player's own positions into one .npz per player (resumable)."""
    os.makedirs(out, exist_ok=True)
    for nm in names:
        dst = os.path.join(out, f"{nm}.npz")
        if os.path.exists(dst):
            continue
        src = os.path.join(pgn_dir, f"{nm}.pgn.zst")
        target = modal_player(src)
        recs, gi = [], 0
        for g in iter_games_from_zst(src):
            if gi >= max_games:
                break
            if "abandon" in g.headers.get("Termination", "").lower():
                continue
            mine = [r for r in game_to_records(g) if r.player_id == target]
            if len(mine) < 5:
                continue
            recs.extend(mine); gi += 1
        if recs:
            np.savez_compressed(dst, **records_to_arrays(recs))
            print(f"  {nm}: {gi} games, {len(recs):,} positions", flush=True)


def load_arrays(names, cache):
    out = {}
    parts = [np.load(os.path.join(cache, f"{n}.npz")) for n in names
             if os.path.exists(os.path.join(cache, f"{n}.npz"))]
    for k in KEYS + ("think_time", "move_from", "move_to", "promo", "result"):
        out[k] = np.concatenate([p[k] for p in parts])
    return out


class Joint(nn.Module):
    """Base backbone + a bucket time head on [pooled, policy_difficulty].

    The head takes the same 514-dim input the frozen bake-off heads did, so a converged frozen
    head can be loaded straight in. That is what makes this experiment cheap: the run STARTS at
    the frozen ceiling (r=0.5753) instead of spending ten hours re-converging a head, and the
    only thing left to learn is whether the backbone can do better than the features it was
    handing over.
    """

    def __init__(self, base, dim=514, hid=256):
        super().__init__()
        self.base = base
        self.time = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hid),
                                  nn.ReLU(), nn.Linear(hid, N_BUCKETS))

    def forward(self, batch):
        out = self.base(batch)
        d = policy_difficulty(out["move_logits"])
        return out, self.time(torch.cat([out["pooled"], d], dim=-1))


def batches(arr, bs, steps, seed=0):
    n = len(arr["think_time"])
    g = np.random.default_rng(seed)
    for _ in range(steps):
        i = g.integers(0, n, bs)
        b = {k: torch.from_numpy(np.ascontiguousarray(arr[k][i])).float().to(DEV) for k in KEYS}
        tgt = move_target_index(
            torch.from_numpy(np.ascontiguousarray(arr["move_from"][i])),
            torch.from_numpy(np.ascontiguousarray(arr["move_to"][i])),
            torch.from_numpy(np.ascontiguousarray(arr["promo"][i]))).to(DEV)
        yield (b,
               torch.from_numpy(to_bucket(arr["think_time"][i])).long().to(DEV),
               tgt,
               torch.from_numpy(np.ascontiguousarray(arr["result"][i])).long().to(DEV),
               torch.from_numpy(np.ascontiguousarray(arr["temporal"][i, 0] * 180.0)).float().to(DEV))


@torch.no_grad()
def eval_r(model, arr, bs=256):
    model.eval()
    ets, acts = [], []
    n = len(arr["think_time"])
    for s in range(0, n, bs):
        j = slice(s, min(s + bs, n))
        b = {k: torch.from_numpy(np.ascontiguousarray(arr[k][j])).float().to(DEV) for k in KEYS}
        with torch.autocast("cuda", enabled=(DEV == "cuda")):
            _, tl = model(b)
        p = masked_probs(tl.float(), torch.from_numpy(
            np.ascontiguousarray(arr["temporal"][j, 0] * 180.0)).float().to(DEV))
        ets.append(expected_time(p).cpu().numpy()); acts.append(arr["think_time"][j])
    model.train()
    e, a = np.concatenate(ets), np.concatenate(acts)
    return float(np.corrcoef(e, a)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["frozen", "joint"], required=True)
    ap.add_argument("--split", default="cache/timehead_split.json")
    ap.add_argument("--pgn-dir", default="data/chesscom_bands_v2")
    ap.add_argument("--cache", default="cache/timehead_raw")
    ap.add_argument("--ckpt", default="checkpoints/base_300k_best.pt")
    ap.add_argument("--out", default="checkpoints/timehead_joint")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--w-time", type=float, default=1.0)
    ap.add_argument("--w-policy", type=float, default=1.0)
    ap.add_argument("--w-value", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--init-head", default="checkpoints/timehead/bucket-ce.pt",
                    help="converged frozen head to start from ('' = random init)")
    args = ap.parse_args()

    sp = json.load(open(args.split, encoding="utf-8"))
    os.makedirs(args.out, exist_ok=True)
    print("building raw caches (resumable)...", flush=True)
    build_cache(sp["train"] + sp["val"] + sp["test"], args.pgn_dir, args.cache)

    tr = load_arrays(sp["train"], args.cache)
    te = load_arrays(sp["test"], args.cache)
    print(f"train {len(tr['think_time']):,} positions | test {len(te['think_time']):,}", flush=True)

    base, _ = load_model(args.ckpt)
    model = Joint(base).to(DEV)
    if args.init_head and os.path.exists(args.init_head):
        sd = torch.load(args.init_head, map_location=DEV, weights_only=True)
        model.time.load_state_dict({k.replace("net.", ""): v for k, v in sd.items()})
        print(f"time head initialised from {args.init_head}", flush=True)
    if args.arm == "frozen":
        for p in model.base.parameters():
            p.requires_grad = False
    lr = args.lr if args.lr else (1e-5 if args.arm == "joint" else 3e-4)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"arm={args.arm} | trainable {sum(p.numel() for p in params)/1e6:.2f}M | lr {lr}", flush=True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEV == "cuda"))

    hist = []
    r0 = eval_r(model, te)
    print(f"step 0: test r@E[t] {r0:.4f}  (untrained head)", flush=True)
    t0 = time.time()
    model.train()
    for step, (b, y, mtgt, res, clk) in enumerate(batches(tr, args.bs, args.steps), 1):
        with torch.autocast("cuda", enabled=(DEV == "cuda")):
            out, tl = model(b)
            p = masked_probs(tl.float(), clk)
            tloss = F.nll_loss(torch.log(p.clamp_min(1e-9)), y)
            loss = args.w_time * tloss
            if args.arm == "joint":
                # keep the backbone honest: without these it can forget how to play chess
                # in order to fit think-time, and the resulting r would mean nothing
                ploss = F.cross_entropy(out["move_logits"].float(), mtgt)
                vloss = F.cross_entropy(out["value_logits"].float(), res)
                loss = loss + args.w_policy * ploss + args.w_value * vloss
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 3.5)
        scaler.step(opt); scaler.update()
        if step % args.eval_every == 0 or step == args.steps:
            r = eval_r(model, te)
            hist.append({"step": step, "loss": float(loss.item()),
                         "time_loss": float(tloss.item()), "test_r": r})
            print(f"step {step:>6}: loss {loss.item():.4f} time {tloss.item():.4f} | test r@E[t] {r:.4f} "
                  f"| {time.time()-t0:.0f}s", flush=True)
            torch.save({"arm": args.arm, "step": step,
                        "time_head": model.time.state_dict(),
                        "base_state": model.base.state_dict() if args.arm == "joint" else None},
                       os.path.join(args.out, f"{args.arm}.pt"))
            json.dump(hist, open(os.path.join(args.out, f"{args.arm}.json"), "w"), indent=2)

    best = max(h["test_r"] for h in hist)
    print(f"\narm={args.arm}: best test r@E[t] = {best:.4f}   (frozen-feature ceiling was 0.5753)")


if __name__ == "__main__":
    main()
