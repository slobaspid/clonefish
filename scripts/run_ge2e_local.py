"""GE2E game-embedding fingerprint with an optional timing channel — local runner.

    PYTHONPATH=. python scripts/run_ge2e_local.py --time      # moves + time
    PYTHONPATH=. python scripts/run_ge2e_local.py --no-time   # moves only  (the ablation)

Reads cache_film.pt (build it first with build_cache_local.py). Trains GE2E, prints
P@1 vs player-count and a games-per-query curve.
"""
import argparse
import os
import sys
import time
import random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL, N_LAYERS, N_HEADS, MAX_PLIES = 256, 4, 4, 60
M_PLAYERS, K_GAMES, STEPS, LR = 32, 5, 3000, 3e-4


class GameEncoder(nn.Module):
    def __init__(self, dp, use_time, d=D_MODEL, layers=N_LAYERS, heads=N_HEADS, L=MAX_PLIES):
        super().__init__()
        self.use_time = use_time
        self.proj = nn.Linear(dp + (1 if use_time else 0), d)
        self.pos = nn.Parameter(torch.zeros(1, L, d))
        enc = nn.TransformerEncoderLayer(d, heads, d * 4, dropout=0.1, batch_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, layers)

    def forward(self, pool, tfeat, mask):
        x = pool.float()
        if self.use_time:
            x = torch.cat([x, tfeat.unsqueeze(-1)], dim=-1)
        x = self.proj(x) + self.pos[:, :x.shape[1]]
        h = self.tr(x, src_key_padding_mask=~mask)
        m = mask.unsqueeze(-1).float()
        return F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1), dim=-1)


class GE2E(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(10.0)); self.b = nn.Parameter(torch.tensor(-5.0))

    def loss(self, emb):
        M, K, d = emb.shape
        cent = F.normalize(emb.mean(1), dim=-1)
        excl = F.normalize((emb.sum(1, keepdim=True) - emb) / (K - 1), dim=-1)
        sim = torch.einsum("mkd,jd->mkj", emb, cent)
        own = (emb * excl).sum(-1)
        ar = torch.arange(M, device=emb.device)
        sim[ar, :, ar] = own
        sim = self.w.clamp(min=1e-6) * sim + self.b
        tgt = ar[:, None].expand(M, K).reshape(-1)
        return F.cross_entropy(sim.reshape(M * K, M), tgt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache_film.pt")
    ap.add_argument("--time", dest="use_time", action="store_true")
    ap.add_argument("--no-time", dest="use_time", action="store_false")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.set_defaults(use_time=True)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); random.seed(0)
    tag = "MOVES+TIME" if args.use_time else "MOVES-ONLY"
    print(f"device: {dev} | mode: {tag}")

    c = torch.load(args.cache, weights_only=False)
    pooled = c["pooled"].float(); think = c["think"].float()
    game = c["game"].long(); pid = c["pid"].long(); test = c["test"].bool()
    names = c["names"]; P = len(names); Dp = pooled.shape[1]

    # ---- group plies into padded games ----
    order = torch.argsort(game, stable=True)
    gs = game[order]
    change = torch.ones(len(gs), dtype=torch.bool); change[1:] = gs[1:] != gs[:-1]
    starts = change.nonzero(as_tuple=True)[0].tolist() + [len(gs)]
    G = len(starts) - 1
    logt = torch.log1p(think.clamp(min=0))
    tmean, tstd = logt.mean(), logt.std().clamp(min=1e-3)
    po, th, pi, te = pooled[order], logt[order], pid[order], test[order]

    gpool = torch.zeros(G, MAX_PLIES, Dp, dtype=torch.float16)
    gtime = torch.zeros(G, MAX_PLIES); gmask = torch.zeros(G, MAX_PLIES, dtype=torch.bool)
    gown = torch.zeros(G, dtype=torch.long); gtest = torch.zeros(G, dtype=torch.bool)
    for i in range(G):
        a, b = starts[i], starts[i + 1]
        l = min(b - a, MAX_PLIES)
        gpool[i, :l] = po[a:a + l].half()
        gtime[i, :l] = (th[a:a + l] - tmean) / tstd
        gmask[i, :l] = True
        gown[i] = pi[a]; gtest[i] = te[a]
    gpool = gpool.to(dev); gtime = gtime.to(dev); gmask = gmask.to(dev)
    gown = gown.to(dev); gtest = gtest.to(dev)
    print(f"{G} games | {int(gtest.sum())} test | avg plies {gmask.float().sum(1).mean():.1f}")

    train_by_player = {}
    for i in range(G):
        if not bool(gtest[i]):
            train_by_player.setdefault(int(gown[i]), []).append(i)
    elig = [p for p, gl in train_by_player.items() if len(gl) >= K_GAMES]

    enc = GameEncoder(Dp, args.use_time).to(dev)
    ge2e = GE2E().to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(ge2e.parameters()), lr=LR, weight_decay=1e-4)
    rng = random.Random(0)
    enc.train(); t0 = time.time()
    for step in range(args.steps):
        players = rng.sample(elig, M_PLAYERS)
        idx = []
        for p in players:
            idx += rng.sample(train_by_player[p], K_GAMES)
        idx = torch.tensor(idx, device=dev)
        e = enc(gpool[idx], gtime[idx], gmask[idx]).view(M_PLAYERS, K_GAMES, -1)
        loss = ge2e.loss(e)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 3.0)
        opt.step()
        if (step + 1) % 250 == 0:
            print(f"step {step+1}/{args.steps}  ge2e loss {loss.item():.4f}  ({time.time()-t0:.0f}s)")

    # ---- eval ----
    @torch.no_grad()
    def embed_all():
        enc.eval(); E = torch.zeros(G, D_MODEL, device=dev); B = 512
        for s in range(0, G, B):
            E[s:s + B] = enc(gpool[s:s + B], gtime[s:s + B], gmask[s:s + B])
        return E

    E = embed_all()
    tr = ~gtest
    cent = torch.zeros(P, D_MODEL, device=dev)
    for p in range(P):
        m = tr & (gown == p)
        if m.any():
            cent[p] = F.normalize(E[m].mean(0), dim=-1)
    cent = F.normalize(cent, dim=-1)
    ti = gtest.nonzero(as_tuple=True)[0]
    Et = E[ti]; own_t = gown[ti]

    def p_at_1(Nsub):
        keep = own_t < Nsub
        return (Et[keep] @ cent[:Nsub].T).argmax(1).eq(own_t[keep]).float().mean().item()

    print(f"\n=== GE2E {tag} — single game / query ===")
    print("players(N)   P@1     chance")
    for Nsub in [10, 30, 60, P]:
        print(f"{Nsub:>8}   {p_at_1(Nsub):.3f}   {1/Nsub:.3f}")

    by = {}
    for j in range(len(ti)):
        by.setdefault(int(own_t[j]), []).append(j)
    owners = [p for p, l in by.items() if len(l) >= 10]
    print(f"\n=== GE2E {tag} — games/query (N={P}) ===")
    print("games   P@1")
    r = random.Random(0)
    for k in (1, 3, 5, 10):
        hit = tot = 0
        for _ in range(400):
            p = r.choice(owners); sel = r.sample(by[p], k)
            q = F.normalize(Et[sel].mean(0), dim=-1)
            hit += int((q @ cent.T).argmax().item() == p); tot += 1
        print(f"{k:>5}   {hit/tot:.3f}")


if __name__ == "__main__":
    main()
