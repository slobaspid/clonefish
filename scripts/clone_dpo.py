"""STAGE 2+3 of the clone judge/DPO (docs/superpowers/specs/2026-08-21-clone-dpo-judge-localizer-spec.md).

Localize the positions where the clone STICKS OUT (its top move != the player's real move) and
run a KL-leashed DPO preference update on the clone's MOVE head toward the real move — anchored by
an imitation term so it can't drift into a caricature. Time head frozen.

    PYTHONPATH=. python scripts/clone_dpo.py --start clones/Gerry_h120.pt \
        --pgn data/lichess_scale/Gerry_Grob.pgn.zst --name Gerry_Grob \
        --max-games 120 --out clones/Gerry_dpo.pt

Pairs (offline, simple-honest): a+ = the real move, a- = the starting clone's argmax move where it
disagrees. Reference policy pi_ref = the starting clone (frozen). Only the move head moves.
"""
import argparse
import copy
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F

from sahformer.training.loop import load_model
from sahformer.clone import load_clone, save_clone
from clone_fit import extract


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="starting clone (imitation-fit) = pi_ref")
    ap.add_argument("--pgn", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", default="checkpoints/base_300k_best.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-games", type=int, default=120, help="TRAIN games (keep < eval split!)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--beta", type=float, default=0.1, help="DPO KL-leash strength")
    ap.add_argument("--lambda-imit", type=float, default=0.5, help="imitation anchor weight")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows, seen, games = extract(args.pgn, args.name, args.max_games)
    N = len(rows)
    print(f"{args.name}: {N} train moves across {games} games")

    model, _ = load_model(args.ckpt); model.eval().to(dev)
    for p in model.parameters():
        p.requires_grad_(False)

    # cache frozen base features (pooled + base move logits) for the train positions
    cP, cML, cAct = [], [], []
    B = 512
    avg = int(load_clone(args.start).meta.get("avg_elo", 1500))
    with torch.no_grad():
        for s in range(0, N, B):
            batch = rows[s:s + B]; nb = len(batch)
            out = model({
                "board": torch.from_numpy(np.stack([r[0] for r in batch])).float().to(dev),
                "history": torch.from_numpy(np.stack([r[1] for r in batch])).float().to(dev),
                "elo_self": torch.full((nb,), avg).to(dev),
                "elo_opp": torch.full((nb,), avg).to(dev),
                "temporal": torch.from_numpy(np.stack([r[2] for r in batch])).float().to(dev),
            })
            cP.append(out["pooled"]); cML.append(out["move_logits"])
            cAct.append(torch.tensor([r[5] for r in batch], device=dev))
    pooled = torch.cat(cP); base_ml = torch.cat(cML); act = torch.cat(cAct)

    ref = load_clone(args.start, device=dev)          # frozen reference
    for p in ref.parameters():
        p.requires_grad_(False)
    theta = copy.deepcopy(ref).to(dev)                # the one we fine-tune
    for p in theta.time.parameters():                 # freeze TIME head — moves only
        p.requires_grad_(False)

    # ---- LOCALIZER: pairs where the starting clone sticks out (top move != real move) ----
    with torch.no_grad():
        ref_logits = base_ml + ref.move_residual(pooled)       # pi_ref move logits
        ref_logp = F.log_softmax(ref_logits, dim=-1)
        a_neg = ref_logits.argmax(dim=-1)                      # the clone's tell
    stick = a_neg != act                                       # the localized tell-points
    idx = stick.nonzero(as_tuple=True)[0]
    print(f"localizer: {len(idx)}/{N} positions where the clone sticks out "
          f"({len(idx)/N*100:.0f}%) -> DPO pairs")
    ai = act[idx]; ni = a_neg[idx]
    ref_dpos = ref_logp[idx, ai]; ref_dneg = ref_logp[idx, ni]  # frozen ref log-probs

    opt = torch.optim.AdamW([p for p in theta.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)
    M = len(idx)
    for ep in range(args.epochs):
        theta.train(); perm = torch.randperm(M, device=dev); tot = nb = 0
        for s in range(0, M, args.batch_size):
            b = perm[s:s + args.batch_size]; g = idx[b]
            logits = base_ml[g] + theta.move_residual(pooled[g])
            logp = F.log_softmax(logits, dim=-1)
            dpos = logp[torch.arange(len(g), device=dev), ai[b]] - ref_dpos[b]
            dneg = logp[torch.arange(len(g), device=dev), ni[b]] - ref_dneg[b]
            dpo = -F.logsigmoid(args.beta * (dpos - dneg)).mean()
            imit = F.cross_entropy(logits, ai[b])              # anchor: real move stays likely
            loss = dpo + args.lambda_imit * imit
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"epoch {ep+1}/{args.epochs}  loss {tot/max(1,nb):.4f}")

    theta.eval()
    meta = dict(ref.meta); meta["dpo"] = {"beta": args.beta, "lambda_imit": args.lambda_imit,
                                          "pairs": int(M), "start": args.start}
    save_clone(args.out, theta, meta)
    print(f"\nsaved DPO clone -> {args.out}")


if __name__ == "__main__":
    main()
