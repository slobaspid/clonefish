"""Move-identifier: CosFace-style ADDITIVE MARGIN on move choice (your idea). Instead of plain
cross-entropy, push the player's actual move above the other legal moves by a margin — a contrastive
"which move is most THIS person here, relative to the alternatives" objective. Tests whether a
margin/discriminative loss extracts more per-move personal signal than plain CE.

    PYTHONPATH=. python scripts/move_identifier.py --cache sweep_cache_lichess_1k_100_full.pt

Memory-safe (data on CPU, batch to GPU). margin=0 == plain-CE baseline.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from clone_sweep import PooledResidual, MAXLEG, DEV


def train_eval(cache, margin, emb=512, hidden=512, wd=3e-2, epochs=20, dropout=0.6, res_l2=0.10, lr=1e-3, bs=2048):
    torch.manual_seed(0)
    players = list(cache.keys()); P = len(players)
    Pt, BL, LG, LN, AC, PID = [], [], [], [], [], []
    for i, n in enumerate(players):
        tp, tbl, tlg, tln, tac = cache[n]["train"]
        Pt.append(tp.float()); BL.append(tbl.float()); LG.append(tlg.long()); LN.append(tln); AC.append(tac)
        PID.append(torch.full((tp.shape[0],), i))
    Pt=torch.cat(Pt); BL=torch.cat(BL); LG=torch.cat(LG); LN=torch.cat(LN); AC=torch.cat(AC); PID=torch.cat(PID)
    m = PooledResidual(P, emb, hidden=hidden, dropout=dropout).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    N = Pt.shape[0]; aleg = torch.arange(MAXLEG, device=DEV)
    for _ in range(epochs):
        perm = torch.randperm(N); m.train()
        for s in range(0, N, bs):
            b = perm[s:s+bs]
            pb=Pt[b].to(DEV); bl=BL[b].to(DEV); lg=LG[b].to(DEV); ln=LN[b].to(DEV); ac=AC[b].to(DEV); pid=PID[b].to(DEV)
            pad = aleg[None,:] >= ln[:,None]
            res = m(pb, pid); rl = torch.gather(res, 1, lg)
            logits = (bl + rl).masked_fill(pad, -1e4)
            # ADDITIVE MARGIN: subtract `margin` from the played move's logit during training,
            # forcing the model to separate it from the alternatives by that gap (CosFace/AM-Softmax spirit).
            onehot = F.one_hot(ac, logits.shape[1]).float()
            loss = F.cross_entropy(logits - margin * onehot, ac) + res_l2 * res.pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); dmove = []
    with torch.no_grad():
        for i, n in enumerate(players):
            ep, ebl, elg, eln, eac = cache[n]["test"]
            epf=ep.float().to(DEV); ebf=ebl.float().to(DEV); elf=elg.long().to(DEV); eac_d=eac.to(DEV)
            pad = aleg[None,:] >= eln.to(DEV)[:,None]
            res = m(epf, torch.full((epf.shape[0],), i, device=DEV)); rl = torch.gather(res, 1, elf)
            base_lg = ebf.masked_fill(pad, -1e4); clone_lg = (ebf + rl).masked_fill(pad, -1e4)
            dmove.append(((clone_lg.argmax(1)==eac_d).float().mean() - (base_lg.argmax(1)==eac_d).float().mean()).item()*100)
    return float(np.mean(dmove)), float(np.median(dmove))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="sweep_cache_lichess_1k_100_full.pt")
    ap.add_argument("--margins", default="0,0.5,1.0,2.0")
    args = ap.parse_args()
    cache = torch.load(args.cache, weights_only=False)
    print(f"{args.cache}: {len(cache)} players\n{'margin':>7}{'moveΔ mean':>12}{'median':>9}   (0 = plain-CE baseline)")
    for mg in [float(x) for x in args.margins.split(",")]:
        a, b = train_eval(cache, mg)
        print(f"{mg:>7.2f}{a:>+12.2f}{b:>+9.2f}", flush=True)


if __name__ == "__main__":
    main()
