"""Retrieval-augmented clone prototype. Instead of a 512-number code, keep the player's whole
history: index every past position by the base's fingerprint (pooled 512-d), and at each new
position pull the k most-similar past positions, see what the player played there, and boost those
moves. Tests whether retrieval cracks the MIDGAME where the static code added ~0.

    PYTHONPATH=. python scripts/retrieval_clone.py --name latebloomer --data-dir data/lichess_1k --max-games 1200

Non-learned first pass (similarity-weighted vote) — if the signal is here, learned attention comes next.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
from clone_sweep import extract_split, base_feats, MAXLEG, DEV
from sahformer.training.loop import load_model

BUCKETS = [(0, 12, "opening"), (12, 24, "early-mid"), (24, 40, "middlegame"), (40, 9999, "late/endgame")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data-dir", default="data/lichess_1k")
    ap.add_argument("--max-games", type=int, default=1200)
    ap.add_argument("--ckpt", default="checkpoints/base_300k_best.pt")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--lams", default="0,1,2,4,8")
    args = ap.parse_args()

    model, _ = load_model(args.ckpt); model.eval().to(DEV)
    sp = extract_split(f"{args.data_dir}/{args.name}.pgn.zst", args.name, args.max_games, 40)
    if sp is None:
        print("not enough games"); return
    train, test, gid, meta_tr, meta_te = sp
    Ptr, BLtr, LGtr, LNtr, ACtr = base_feats(model, train)
    Pte, BLte, LGte, LNte, ACte = base_feats(model, test)
    ply_te = torch.tensor([m[0] for m in meta_te])
    print(f"{args.name}: {Ptr.shape[0]} train pos, {Pte.shape[0]} test pos, k={args.k}")

    ftr = F.normalize(Ptr.float().to(DEV), dim=1)                     # train fingerprints [Ntr,512]
    played_tr = LGtr.to(DEV).gather(1, ACtr.to(DEV)[:, None]).squeeze(1)   # full move idx played [Ntr]
    aleg = torch.arange(MAXLEG, device=DEV)
    lams = [float(x) for x in args.lams.split(",")]

    # accumulate per-phase, per-lam: correct counts
    agg = {lam: {lab: [0, 0] for _,_,lab in BUCKETS} for lam in lams}   # [correct, n] (n same across lam)
    gated = {lam: {lab: [0, 0] for _,_,lab in BUCKETS} for lam in lams} # CONFIDENCE-GATED retrieval
    pure = {lab: [0, 0] for _,_,lab in BUCKETS}                         # pure-retrieval correct
    # per-position noise analysis (noise = 1 - neighbor agreement); collect (conc, base_ok, gated_ok@lam1)
    NL = 1.0 if 1.0 in lams else lams[1]
    conc_all, baseok_all, gok_all = [], [], []
    B = 256
    with torch.no_grad():
        for s in range(0, Pte.shape[0], B):
            e = slice(s, s+B)
            fte = F.normalize(Pte[e].float().to(DEV), dim=1)           # [b,512]
            legal = LGte[e].to(DEV); lens = LNte[e].to(DEV); ac = ACte[e].to(DEV)
            pad = aleg[None,:] >= lens[:,None]
            base_lg = BLte[e].float().to(DEV).masked_fill(pad, -1e4)
            ply = ply_te[e]
            sim = fte @ ftr.T                                          # [b,Ntr]
            topv, topi = sim.topk(args.k, dim=1)                       # [b,k]
            nb = played_tr[topi]                                       # [b,k] neighbor moves
            # retrieval vote over this position's legal moves: sum sim of neighbors who played each legal move
            match = (nb[:,:,None] == legal[:,None,:]).float()         # [b,k,MAXLEG]
            rscore = (topv.clamp(min=0)[:,:,None] * match).sum(1)     # [b,MAXLEG]
            rscore = rscore.masked_fill(pad, 0.0)
            rsum = rscore.sum(1, keepdim=True)                        # total vote weight
            rdist = rscore / rsum.clamp(min=1e-6)                     # vote distribution over legal
            conc = rdist.max(1, keepdim=True).values                 # concentration (agreement) in [0,1]
            for lam in lams:
                pred = (base_lg + lam * rscore).argmax(1)
                # GATED: boost scaled by how much the neighbors AGREE (conc). openings->strong, midgame->~off
                gpred = (base_lg + lam * conc * rscore).argmax(1)
                for lo,hi,lab in BUCKETS:
                    msk = (ply>=lo)&(ply<hi)
                    agg[lam][lab][0] += int((pred.cpu()[msk]==ac.cpu()[msk]).sum()); agg[lam][lab][1] += int(msk.sum())
                    gated[lam][lab][0] += int((gpred.cpu()[msk]==ac.cpu()[msk]).sum()); gated[lam][lab][1] += int(msk.sum())
            # pure retrieval: argmax rscore where any vote, else base
            has = rscore.sum(1) > 0
            ppred = torch.where(has, rscore.argmax(1), base_lg.argmax(1))
            for lo,hi,lab in BUCKETS:
                msk = (ply>=lo)&(ply<hi)
                pure[lab][0] += int((ppred.cpu()[msk]==ac.cpu()[msk]).sum()); pure[lab][1] += int(msk.sum())
            # noise analysis: agreement (conc) vs correctness
            gok = ((base_lg + NL*conc*rscore).argmax(1) == ac).cpu()
            conc_all.append(conc.squeeze(1).cpu()); baseok_all.append((base_lg.argmax(1)==ac).cpu()); gok_all.append(gok)

    N = sum(agg[lams[0]][x][1] for _,_,x in BUCKETS)
    def overall(d, l): return sum(d[l][x][0] for _,_,x in BUCKETS)/N*100
    best_plain = max([l for l in lams if l>0], key=lambda l: overall(agg, l))
    best_gated = max([l for l in lams if l>0], key=lambda l: overall(gated, l))
    print(f"\nbest plain λ={best_plain} | best GATED λ={best_gated}")
    print(f"{'phase':>13}{'n':>8}{'base':>9}{'plainRetr':>11}{'GATED':>9}")
    for _,_,lab in BUCKETS:
        n = agg[lams[0]][lab][1]
        if n == 0: continue
        print(f"{lab:>13}{n:>8}{agg[0.0][lab][0]/n*100:>8.1f}%{agg[best_plain][lab][0]/n*100:>10.1f}%{gated[best_gated][lab][0]/n*100:>8.1f}%")
    print(f"{'ALL':>13}{N:>8}{overall(agg,0.0):>8.1f}%{overall(agg,best_plain):>10.1f}%{overall(gated,best_gated):>8.1f}%")
    print(f"\n(GATED = retrieval scaled by neighbor-agreement: fires in openings, backs off in scattered midgames)")

    # ---- NOISE analysis: bucket positions by how PREDICTABLE they are (neighbor agreement) ----
    conc = torch.cat(conc_all).numpy(); bok = torch.cat(baseok_all).numpy().astype(float); gok = torch.cat(gok_all).numpy().astype(float)
    qs = np.quantile(conc, [0.25, 0.5, 0.75])
    nb = [("high-noise (least predictable)", conc <= qs[0]),
          ("mid-low", (conc > qs[0]) & (conc <= qs[1])),
          ("mid-high", (conc > qs[1]) & (conc <= qs[2])),
          ("LOW-noise (most predictable)", conc > qs[2])]
    print(f"\n=== by per-position PREDICTABILITY (neighbor agreement) ===")
    print(f"{'bucket':>32}{'n':>7}{'base':>8}{'clone(gated)':>14}")
    for lab, msk in nb:
        if msk.sum() == 0: continue
        print(f"{lab:>32}{int(msk.sum()):>7}{bok[msk].mean()*100:>7.1f}%{gok[msk].mean()*100:>13.1f}%")
    print(f"\n-> on your most-predictable quarter, clone top-1 = {gok[conc>qs[2]].mean()*100:.1f}% (vs {bok[conc>qs[2]].mean()*100:.1f}% base).")


if __name__ == "__main__":
    main()
