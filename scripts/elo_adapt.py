"""Elo-adaptability: make the per-player code a FUNCTION of Elo, so a climbing/dropping player's
clone tracks their evolving style. code(player, elo) = base_style + elo_norm * drift. Compares a
static code vs an Elo-conditioned code on held-out (recent) games, overall and on the RATING MOVERS.
Needs an ENRICHED cache (meta = (ply, elo)).

    PYTHONPATH=. python scripts/elo_adapt.py --cache sweep_cache_lichess_1k_100_full.pt
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from clone_sweep import MAXLEG, DEV

MOVERS = {"ajaremile","Yespapa","happyVegan","sawabear","stefane01","latebloomer","T22","zeus2016","Misun","ARQZITRO","TR4207"}


class EloResidual(nn.Module):
    def __init__(self, P, emb_dim=512, dim=512, n_moves=4352, hidden=512, dropout=0.6, elo_cond=True):
        super().__init__()
        self.emb = nn.Embedding(P, emb_dim); nn.init.normal_(self.emb.weight, std=0.01)
        self.elo_cond = elo_cond
        if elo_cond:
            self.slope = nn.Embedding(P, emb_dim); nn.init.zeros_(self.slope.weight)   # start = static
        self.net = nn.Sequential(nn.LayerNorm(dim+emb_dim), nn.Linear(dim+emb_dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, n_moves))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, pooled, pid, elo_norm):
        code = self.emb(pid)
        if self.elo_cond:
            code = code + elo_norm[:, None] * self.slope(pid)
        return self.net(torch.cat([pooled, code], -1))


def load(cache):
    players = list(cache.keys())
    Pt, BL, LG, LN, AC, PID, ELO = [], [], [], [], [], [], []
    for i, n in enumerate(players):
        tp, tbl, tlg, tln, tac = cache[n]["train"]
        Pt.append(tp.float()); BL.append(tbl.float()); LG.append(tlg.long()); LN.append(tln); AC.append(tac)
        PID.append(torch.full((tp.shape[0],), i)); ELO.append(cache[n]["meta_train"][:, 1].float())
    t = lambda xs: torch.cat(xs)
    return players, t(Pt), t(BL), t(LG), t(LN), t(AC), t(PID), t(ELO)


def train(cache, elo_cond, emb=512, hidden=512, wd=3e-2, epochs=20, dropout=0.6, res_l2=0.10, lr=1e-3, bs=2048):
    torch.manual_seed(0)
    players, Pt, BL, LG, LN, AC, PID, ELO = load(cache)
    ELO = (ELO - 1700) / 200.0                                    # normalize elo
    m = EloResidual(len(players), emb, hidden=hidden, dropout=dropout, elo_cond=elo_cond).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    N = Pt.shape[0]; aleg = torch.arange(MAXLEG, device=DEV)
    for _ in range(epochs):
        perm = torch.randperm(N); m.train()
        for s in range(0, N, bs):
            b = perm[s:s+bs]
            pb=Pt[b].to(DEV); bl=BL[b].to(DEV); lg=LG[b].to(DEV); ln=LN[b].to(DEV); ac=AC[b].to(DEV)
            pid=PID[b].to(DEV); el=ELO[b].to(DEV)
            pad = aleg[None,:] >= ln[:,None]
            res = m(pb, pid, el); rl = torch.gather(res, 1, lg)
            loss = F.cross_entropy((bl+rl).masked_fill(pad,-1e4), ac) + res_l2*res.pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return m, players


@torch.no_grad()
def evalp(m, cache, players):
    m.eval(); aleg = torch.arange(MAXLEG, device=DEV); out = {}
    for i, n in enumerate(players):
        ep, ebl, elg, eln, eac = cache[n]["test"]
        epf=ep.float().to(DEV); ebf=ebl.float().to(DEV); elf=elg.long().to(DEV); eac_d=eac.to(DEV)
        el = ((cache[n]["meta_test"][:,1].float()-1700)/200.0).to(DEV)
        pad = aleg[None,:] >= eln.to(DEV)[:,None]
        res = m(epf, torch.full((epf.shape[0],), i, device=DEV), el); rl = torch.gather(res, 1, elf)
        base_lg = ebf.masked_fill(pad,-1e4); clone_lg=(ebf+rl).masked_fill(pad,-1e4)
        out[n] = ((clone_lg.argmax(1)==eac_d).float().mean()-(base_lg.argmax(1)==eac_d).float().mean()).item()*100
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="sweep_cache_lichess_1k_100_full.pt")
    args = ap.parse_args()
    cache = torch.load(args.cache, weights_only=False)
    if "meta_test" not in cache[list(cache)[0]]:
        print("ERROR: cache has no meta (ply/elo) — rebuild with enriched clone_sweep."); return
    res = {}
    for cond in (False, True):
        m, players = train(cache, elo_cond=cond)
        d = evalp(m, cache, players)
        allv = np.array([d[n] for n in players])
        mov = np.array([d[n] for n in players if n in MOVERS])
        res[cond] = (allv, mov)
        tag = "elo-conditioned" if cond else "static code"
        print(f"{tag:16s}  ALL mean {allv.mean():+.2f}pp   MOVERS mean {mov.mean():+.2f}pp (n={len(mov)})", flush=True)
    da, sa = res[True][0].mean()-res[False][0].mean(), res[True][1].mean()-res[False][1].mean()
    print(f"\nElo-conditioning effect:  ALL {da:+.2f}pp   MOVERS {sa:+.2f}pp   (movers should benefit more)")


if __name__ == "__main__":
    main()
