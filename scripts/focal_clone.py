"""Deviation-weighted (focal) clone training: weight each move's loss by how far the player
deviated from the base's expectation. Big deviation (base gave the played move low prob) -> big
weight; obvious move -> tiny weight. Focuses the residual on where personal style actually lives.

    PYTHONPATH=. python scripts/focal_clone.py --cache sweep_cache_120_full.pt

Compares gamma=0 (uniform = current baseline) vs gamma>0 (deviation-weighted) on held-out moveΔ.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from clone_sweep import PooledResidual, MAXLEG, DEV


def train_and_eval(cache, gamma, emb=512, hidden=512, wd=3e-2, epochs=20, dropout=0.6, res_l2=0.10, lr=1e-3, bs=2048):
    torch.manual_seed(0)
    players = list(cache.keys()); P = len(players)
    Ptr, BLtr, LGtr, LNtr, ACtr, PID = [], [], [], [], [], []
    for i, n in enumerate(players):
        tp, tbl, tlg, tln, tac = cache[n]["train"]
        Ptr.append(tp.float()); BLtr.append(tbl.float()); LGtr.append(tlg.long()); LNtr.append(tln); ACtr.append(tac)
        PID.append(torch.full((tp.shape[0],), i))
    Ptr=torch.cat(Ptr).to(DEV); BLtr=torch.cat(BLtr).to(DEV); LGtr=torch.cat(LGtr).to(DEV)
    LNtr=torch.cat(LNtr).to(DEV); ACtr=torch.cat(ACtr).to(DEV); PID=torch.cat(PID).to(DEV)
    pad = torch.arange(MAXLEG, device=DEV)[None, :] >= LNtr[:, None]
    # per-move deviation weight: 1 - base_prob(played move), ^gamma  (big deviation -> big weight)
    with torch.no_grad():
        base_p = F.softmax(BLtr.masked_fill(pad, -1e4), 1).gather(1, ACtr[:, None]).squeeze(1)
        dev = (1 - base_p).clamp(min=1e-4)
    m = PooledResidual(P, emb, hidden=hidden, dropout=dropout).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    N = Ptr.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(N, device=DEV); m.train()
        for s in range(0, N, bs):
            b = perm[s:s + bs]
            res = m(Ptr[b], PID[b]); rl = torch.gather(res, 1, LGtr[b])
            logits = (BLtr[b] + rl).masked_fill(pad[b], -1e4)
            ce = F.cross_entropy(logits, ACtr[b], reduction="none")
            w = dev[b] ** gamma
            w = w / w.mean().clamp(min=1e-6)                       # keep overall scale comparable
            loss = (ce * w).mean() + res_l2 * res.pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    # eval: held-out top-1 moveΔ per player
    m.eval(); dmove = []
    with torch.no_grad():
        for i, n in enumerate(players):
            ep, ebl, elg, eln, eac = cache[n]["test"]
            epf=ep.float().to(DEV); ebf=ebl.float().to(DEV); elf=elg.long().to(DEV); eac_d=eac.to(DEV)
            epad = torch.arange(MAXLEG, device=DEV)[None, :] >= eln.to(DEV)[:, None]
            pid = torch.full((epf.shape[0],), i, device=DEV)
            res = m(epf, pid); rl = torch.gather(res, 1, elf)
            base_lg = ebf.masked_fill(epad, -1e4); clone_lg = (ebf + rl).masked_fill(epad, -1e4)
            dmove.append(((clone_lg.argmax(1)==eac_d).float().mean() - (base_lg.argmax(1)==eac_d).float().mean()).item()*100)
    return float(np.mean(dmove)), float(np.median(dmove))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="sweep_cache_120_full.pt")
    ap.add_argument("--gammas", default="0,0.5,1,2")
    args = ap.parse_args()
    cache = torch.load(args.cache, weights_only=False)
    print(f"cache {args.cache}: {len(cache)} players\n")
    print(f"{'gamma':>6}  {'moveΔ mean':>11}  {'median':>8}   (gamma=0 = uniform baseline)")
    for g in [float(x) for x in args.gammas.split(",")]:
        mean, med = train_and_eval(cache, g)
        print(f"{g:>6.1f}  {mean:>+11.2f}  {med:>+8.2f}", flush=True)


if __name__ == "__main__":
    main()
