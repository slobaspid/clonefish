"""Decompose predictability: for each held-out position, measure
  CHOICE      = base move-entropy over legal moves (low=forced, high=genuine choice)
  CONSISTENCY = how concentrated YOUR moves were in similar past positions (retrieval agreement)
The personalization GOLD is CHOICE + CONSISTENT. Question: does any of it live in the midgame?

    PYTHONPATH=. python scripts/decompose.py --name latebloomer --data-dir data/lichess_1k
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
from clone_sweep import extract_split, base_feats, MAXLEG, DEV
from sahformer.training.loop import load_model


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
    ply = torch.tensor([m[0] for m in meta_te]).numpy()
    ftr = F.normalize(Ptr.float().to(DEV), dim=1)
    played_tr = LGtr.to(DEV).gather(1, ACtr.to(DEV)[:, None]).squeeze(1)
    aleg = torch.arange(MAXLEG, device=DEV)

    ent, conc, bok, cok = [], [], [], []
    with torch.no_grad():
        for s in range(0, Pte.shape[0], 256):
            e = slice(s, s+256)
            fte = F.normalize(Pte[e].float().to(DEV), 1)
            legal = LGte[e].to(DEV); lens = LNte[e].to(DEV); ac = ACte[e].to(DEV)
            pad = aleg[None,:] >= lens[:,None]
            base_lg = BLte[e].float().to(DEV).masked_fill(pad, -1e4)
            p = F.softmax(base_lg, 1)
            ent.append((-(p*torch.log(p+1e-9)).sum(1)).cpu())            # base entropy = CHOICE
            sim = fte @ ftr.T; topv, topi = sim.topk(args.k, 1)
            nb = played_tr[topi]
            match = (nb[:,:,None] == legal[:,None,:]).float()
            rscore = (topv.clamp(min=0)[:,:,None]*match).sum(1).masked_fill(pad, 0.0)
            c = (rscore / rscore.sum(1,keepdim=True).clamp(min=1e-6)).max(1,keepdim=True).values  # CONSISTENCY
            conc.append(c.squeeze(1).cpu())
            bok.append((base_lg.argmax(1)==ac).cpu())
            cok.append(((base_lg + args.lam*c*rscore).argmax(1)==ac).cpu())
    ent=torch.cat(ent).numpy(); conc=torch.cat(conc).numpy()
    bok=torch.cat(bok).numpy().astype(float); cok=torch.cat(cok).numpy().astype(float)
    mid = (ply>=12)&(ply<40)

    def cell(msk):
        n=int(msk.sum());
        return f"{n:>6} {bok[msk].mean()*100 if n else 0:>6.1f}%{cok[msk].mean()*100 if n else 0:>8.1f}%  mid%{mid[msk].mean()*100 if n else 0:>5.0f}"
    eh = ent > np.median(ent); ch = conc > np.median(conc)
    print(f"{args.name}: {len(ent)} test positions  (median entropy {np.median(ent):.2f}, median consistency {np.median(conc):.2f})")
    print(f"\n2x2 (n, base top1, clone top1, %of-cell-that's-midgame):")
    print(f"                       CHOICE-low(forced)      CHOICE-high(real choice)")
    print(f"  CONSISTENT-high   {cell(~eh & ch)}   {cell(eh & ch)}")
    print(f"  CONSISTENT-low    {cell(~eh & ~ch)}   {cell(eh & ~ch)}")
    print(f"\n>>> GOLD cell = CHOICE-high & CONSISTENT-high: {cell(eh & ch)}")

    print(f"\n=== MIDGAME ONLY (ply 12-40, n={int(mid.sum())}): is there consistent-choice here? ===")
    for lab, msk in [("choice-high & consistent-high", mid & eh & ch),
                     ("choice-high & consistent-low ", mid & eh & ~ch),
                     ("choice-low  (forced)         ", mid & ~eh)]:
        n=int(msk.sum())
        if n: print(f"  {lab}: n={n:>5}  base {bok[msk].mean()*100:>5.1f}%  clone {cok[msk].mean()*100:>5.1f}%")


if __name__ == "__main__":
    main()
