"""Top-1 / top-3 / top-5 move-match by phase, base vs clone (gated retrieval). Tests whether
personalization shows up in a softer metric (is your move in the top-k?) that top-1 misses —
especially in the midgame.

    PYTHONPATH=. python scripts/topk_phase.py --name latebloomer --data-dir data/lichess_1k
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
from clone_sweep import extract_split, base_feats, MAXLEG, DEV
from sahformer.training.loop import load_model

BUCKETS = [(0, 12, "opening"), (12, 24, "early-mid"), (24, 40, "middlegame"), (40, 9999, "endgame")]
KS = [1, 3, 5]


def topk_hit(logits, target_col, k):
    """is target_col among the top-k columns of logits? -> bool tensor [n]"""
    tk = logits.topk(k, dim=1).indices                     # [n,k]
    return (tk == target_col[:, None]).any(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data-dir", default="data/lichess_1k")
    ap.add_argument("--max-games", type=int, default=1200)
    ap.add_argument("--ckpt", default="checkpoints/base_300k_best.pt")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--lam", type=float, default=1.0)
    args = ap.parse_args()

    model, _ = load_model(args.ckpt); model.eval().to(DEV)
    sp = extract_split(f"{args.data_dir}/{args.name}.pgn.zst", args.name, args.max_games, 40)
    train, test, gid, meta_tr, meta_te = sp
    Ptr, BLtr, LGtr, LNtr, ACtr = base_feats(model, train)
    Pte, BLte, LGte, LNte, ACte = base_feats(model, test)
    ply = torch.tensor([m[0] for m in meta_te])
    ftr = F.normalize(Ptr.float().to(DEV), dim=1)
    played_tr = LGtr.to(DEV).gather(1, ACtr.to(DEV)[:, None]).squeeze(1)
    aleg = torch.arange(MAXLEG, device=DEV)

    # agg[phase] = {'n':, 'base':{k:hits}, 'clone':{k:hits}}
    agg = {lab: {"n": 0, "base": {k:0 for k in KS}, "clone": {k:0 for k in KS}} for _,_,lab in BUCKETS}
    with torch.no_grad():
        for s in range(0, Pte.shape[0], 256):
            e = slice(s, s+256)
            fte = F.normalize(Pte[e].float().to(DEV), 1)
            legal = LGte[e].to(DEV); lens = LNte[e].to(DEV); ac = ACte[e].to(DEV)
            pad = aleg[None,:] >= lens[:,None]
            base_lg = BLte[e].float().to(DEV).masked_fill(pad, -1e4)
            sim = fte @ ftr.T; topv, topi = sim.topk(args.k, 1)
            nb = played_tr[topi]
            match = (nb[:,:,None] == legal[:,None,:]).float()
            rscore = (topv.clamp(min=0)[:,:,None]*match).sum(1).masked_fill(pad, 0.0)
            conc = (rscore / rscore.sum(1,keepdim=True).clamp(min=1e-6)).max(1,keepdim=True).values
            clone_lg = base_lg + args.lam * conc * rscore
            pl = ply[e]
            for lo,hi,lab in BUCKETS:
                m = (pl>=lo)&(pl<hi)
                if not m.any(): continue
                mi = m.to(DEV)
                agg[lab]["n"] += int(m.sum())
                for k in KS:
                    agg[lab]["base"][k] += int(topk_hit(base_lg, ac, k)[mi].sum())
                    agg[lab]["clone"][k] += int(topk_hit(clone_lg, ac, k)[mi].sum())

    print(f"\n{args.name}: move-match by phase — base vs clone (gated retrieval)")
    hdr = "".join(f"  base@{k}  clone@{k}   Δ" for k in KS)
    print(f"{'phase':>11}{'n':>7}" + "".join(f"{'b@'+str(k):>8}{'c@'+str(k):>8}{'Δ':>6}" for k in KS))
    for _,_,lab in BUCKETS:
        a = agg[lab]; n = a["n"]
        if not n: continue
        row = "".join(f"{a['base'][k]/n*100:>7.1f}{a['clone'][k]/n*100:>8.1f}{(a['clone'][k]-a['base'][k])/n*100:>+6.1f}" for k in KS)
        print(f"{lab:>11}{n:>7}{row}")
    tot = {k: (sum(agg[l]['base'][k] for _,_,l in BUCKETS), sum(agg[l]['clone'][k] for _,_,l in BUCKETS)) for k in KS}
    N = sum(agg[l]['n'] for _,_,l in BUCKETS)
    row = "".join(f"{tot[k][0]/N*100:>7.1f}{tot[k][1]/N*100:>8.1f}{(tot[k][1]-tot[k][0])/N*100:>+6.1f}" for k in KS)
    print(f"{'ALL':>11}{N:>7}{row}")


if __name__ == "__main__":
    main()
