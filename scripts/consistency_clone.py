"""Consistency-weighted clone: weight moves by whether the player's DEVIATION RECURS across their
own games (real style) vs is a one-off (noise). Measured honestly by OUT-OF-FOLD prediction:
train the style model on half a player's games, see if it predicts their deviations in the other half.

weight ∝ how much the out-of-fold style model beats the base on that move (positive only) = the move
is both deviant AND generalizes = consistent style. One-offs get ~0 weight.

    PYTHONPATH=. python scripts/consistency_clone.py --cache sweep_cache_120_full.pt
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from clone_sweep import PooledResidual, MAXLEG, DEV


def load_tensors(cache):
    players = list(cache.keys()); P = len(players)
    Pt, BL, LG, LN, AC, PID, FOLD = [], [], [], [], [], [], []
    for i, n in enumerate(players):
        tp, tbl, tlg, tln, tac = cache[n]["train"]
        gid = cache[n].get("gid")
        fold = (gid % 2).long() if gid is not None else (torch.arange(tp.shape[0]) % 2)
        Pt.append(tp.float()); BL.append(tbl.float()); LG.append(tlg.long()); LN.append(tln); AC.append(tac)
        PID.append(torch.full((tp.shape[0],), i)); FOLD.append(fold)
    t = lambda xs: torch.cat(xs).to(DEV)
    return (players, t(Pt), t(BL), t(LG), t(LN), t(AC), t(PID), t(FOLD))


def train_head(P, Ptr, BL, LG, LN, AC, PID, sel, emb, hidden, wd, epochs, dropout, res_l2, lr, w=None, bs=2048):
    torch.manual_seed(0)
    pad = torch.arange(MAXLEG, device=DEV)[None, :] >= LN[:, None]
    m = PooledResidual(P, emb, hidden=hidden, dropout=dropout).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    idx = sel.nonzero(as_tuple=True)[0]
    for _ in range(epochs):
        perm = idx[torch.randperm(len(idx), device=DEV)]; m.train()
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            res = m(Ptr[b], PID[b]); rl = torch.gather(res, 1, LG[b])
            logits = (BL[b] + rl).masked_fill(pad[b], -1e4)
            ce = F.cross_entropy(logits, AC[b], reduction="none")
            wb = (w[b] if w is not None else torch.ones_like(ce))
            loss = (ce * wb).mean() + res_l2 * res.pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def clone_prob(m, Ptr, BL, LG, LN, AC, PID, sel):
    pad = torch.arange(MAXLEG, device=DEV)[None, :] >= LN[:, None]
    out = torch.zeros(Ptr.shape[0], device=DEV)
    idx = sel.nonzero(as_tuple=True)[0]
    for s in range(0, len(idx), 4096):
        b = idx[s:s + 4096]
        res = m(Ptr[b], PID[b]); rl = torch.gather(res, 1, LG[b])
        p = F.softmax((BL[b] + rl).masked_fill(pad[b], -1e4), 1).gather(1, AC[b][:, None]).squeeze(1)
        out[b] = p
    return out


@torch.no_grad()
def heldout_moveD(m, cache, players):
    dm = []
    for i, n in enumerate(players):
        ep, ebl, elg, eln, eac = cache[n]["test"]
        epf=ep.float().to(DEV); ebf=ebl.float().to(DEV); elf=elg.long().to(DEV); eac_d=eac.to(DEV)
        pad = torch.arange(MAXLEG, device=DEV)[None, :] >= eln.to(DEV)[:, None]
        res = m(epf, torch.full((epf.shape[0],), i, device=DEV)); rl = torch.gather(res, 1, elf)
        base_lg = ebf.masked_fill(pad, -1e4); clone_lg = (ebf + rl).masked_fill(pad, -1e4)
        dm.append(((clone_lg.argmax(1)==eac_d).float().mean() - (base_lg.argmax(1)==eac_d).float().mean()).item()*100)
    return float(np.mean(dm)), float(np.median(dm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="sweep_cache_120_full.pt")
    cfg = dict(emb=512, hidden=512, wd=3e-2, epochs=20, dropout=0.6, res_l2=0.10, lr=1e-3)
    args = ap.parse_args()
    cache = torch.load(args.cache, weights_only=False)
    players, Ptr, BL, LG, LN, AC, PID, FOLD = load_tensors(cache)
    P = len(players)
    pad = torch.arange(MAXLEG, device=DEV)[None, :] >= LN[:, None]
    base_p = F.softmax(BL.masked_fill(pad, -1e4), 1).gather(1, AC[:, None]).squeeze(1)
    has_gid = any("gid" in cache[n] for n in players)
    print(f"{args.cache}: {P} players, {Ptr.shape[0]} train positions | gid folds: {has_gid}\n")

    # OUT-OF-FOLD style probability: train on fold0 -> predict fold1, and vice-versa
    print("computing out-of-fold consistency (train on half a player's games, predict the other half)...", flush=True)
    fe = {k: cfg[k] for k in cfg}; fe = dict(fe); fe["epochs"] = 12
    m0 = train_head(P, Ptr, BL, LG, LN, AC, PID, FOLD == 0, **fe)
    oof = clone_prob(m0, Ptr, BL, LG, LN, AC, PID, FOLD == 1)
    m1 = train_head(P, Ptr, BL, LG, LN, AC, PID, FOLD == 1, **fe)
    oof = oof + clone_prob(m1, Ptr, BL, LG, LN, AC, PID, FOLD == 0)

    # consistency = how much the out-of-fold style model beats base on this move (positive only)
    consistency = (oof - base_p).clamp(min=0)
    frac_style = float((consistency > 0.02).float().mean())
    print(f"  positions where style generalizes (oof beats base by >0.02): {frac_style*100:.0f}%")

    all_sel = torch.ones(Ptr.shape[0], dtype=torch.bool, device=DEV)
    print(f"\n{'weighting':<26}{'moveΔ mean':>11}{'median':>9}")
    # baseline (uniform)
    mu = train_head(P, Ptr, BL, LG, LN, AC, PID, all_sel, **cfg)
    a, b = heldout_moveD(mu, cache, players); print(f"{'uniform (baseline)':<26}{a:>+11.2f}{b:>+9.2f}", flush=True)
    # consistency-weighted, a few strengths (blend uniform + consistency so we don't zero out data)
    cnorm = consistency / consistency.mean().clamp(min=1e-6)
    for lam in [0.5, 1.0, 2.0]:
        w = (1 - min(lam, 1.0)) + lam * cnorm if lam <= 1 else (1.0 + lam * cnorm)
        mc = train_head(P, Ptr, BL, LG, LN, AC, PID, all_sel, w=w, **cfg)
        a, b = heldout_moveD(mc, cache, players)
        print(f"{'consistency lam='+str(lam):<26}{a:>+11.2f}{b:>+9.2f}", flush=True)


if __name__ == "__main__":
    main()
