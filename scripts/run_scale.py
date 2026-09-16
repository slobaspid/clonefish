"""Scale-safe fingerprint runner: multiple losses, +/- timing, on a LARGE cache.

    PYTHONPATH=. python scripts/run_scale.py --cache cache_scale.pt --loss arcface --time
    losses: ge2e | supcon | arcface | cosface | proxyanchor      (flip --no-time for moves-only)

One cache build feeds every loss — the loss only touches the training step. Prints P@1 vs
number of candidate players and a games-per-query curve, so you can watch the curves fan apart
(or not) as N climbs. Keeps per-ply features on CPU, gathers each minibatch to the GPU on the
fly, so it scales to thousands of players without OOM.

Sampling per loss:
  tuple   (ge2e)                  -> M players x K games, embeddings [M,K,d]
  flat_mk (supcon)                -> M players x K games, flat [M*K,d] + labels (needs positives)
  class   (arcface/cosface/proxy) -> random reference games + owner labels
"""
import argparse
import os
import sys
import time
import math
import random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL, N_LAYERS, N_HEADS, MAX_PLIES = 256, 4, 4, 60


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


# ---- contrastive / tuple family ----
class GE2E(nn.Module):
    mode = "tuple"

    def __init__(self, **_):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(10.0)); self.b = nn.Parameter(torch.tensor(-5.0))

    def loss(self, emb):                              # emb [M,K,d]
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


class SupCon(nn.Module):
    """Supervised contrastive: for each game, pull all same-player games together against the
    rest of the batch, temperature-scaled. Needs several games per player in the batch."""
    mode = "flat_mk"

    def __init__(self, tau=0.1, **_):
        super().__init__()
        self.tau = tau

    def loss(self, emb, labels):                      # emb [B,d] normalized, labels [B]
        B = emb.shape[0]
        sim = emb @ emb.T / self.tau
        sim = sim - sim.max(1, keepdim=True).values.detach()
        self_mask = torch.eye(B, dtype=torch.bool, device=emb.device)
        exp = torch.exp(sim).masked_fill(self_mask, 0.0)
        log_prob = sim - torch.log(exp.sum(1, keepdim=True) + 1e-12)
        pos = (labels[:, None] == labels[None, :]) & ~self_mask
        pc = pos.sum(1)
        valid = pc > 0
        loss = -(log_prob * pos).sum(1)[valid] / pc[valid].clamp(min=1)
        return loss.mean()


# ---- margin-softmax family ----
class ArcFace(nn.Module):
    mode = "class"

    def __init__(self, d, n, s=30.0, m=0.3, **_):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n, d)); nn.init.xavier_uniform_(self.W)
        self.s, self.m = s, m

    def loss(self, x, labels):
        cos = F.linear(F.normalize(x), F.normalize(self.W)).clamp(-1 + 1e-7, 1 - 1e-7)
        target = torch.cos(torch.acos(cos) + self.m)
        oh = F.one_hot(labels, cos.shape[1]).float()
        return F.cross_entropy(self.s * (oh * target + (1 - oh) * cos), labels)


class CosFace(nn.Module):
    """Additive cosine margin (AM-Softmax) — ArcFace's sibling; margin on cosine, not angle."""
    mode = "class"

    def __init__(self, d, n, s=30.0, m=0.35, **_):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n, d)); nn.init.xavier_uniform_(self.W)
        self.s, self.m = s, m

    def loss(self, x, labels):
        cos = F.linear(F.normalize(x), F.normalize(self.W))
        oh = F.one_hot(labels, cos.shape[1]).float()
        return F.cross_entropy(self.s * (cos - self.m * oh), labels)


# ---- proxy family ----
class ProxyAnchor(nn.Module):
    """One learnable proxy per player; pushes samples toward their proxy, away from others."""
    mode = "class"

    def __init__(self, d, n, alpha=32.0, delta=0.1, **_):
        super().__init__()
        self.P = nn.Parameter(torch.empty(n, d)); nn.init.kaiming_uniform_(self.P, a=math.sqrt(5))
        self.n, self.alpha, self.delta = n, alpha, delta

    def loss(self, x, labels):
        cos = F.normalize(x) @ F.normalize(self.P).T          # [B, n]
        oh = F.one_hot(labels, self.n).float()
        pos = (oh * torch.exp(-self.alpha * (cos - self.delta))).sum(0)   # [n]
        neg = ((1 - oh) * torch.exp(self.alpha * (cos + self.delta))).sum(0)
        with_pos = oh.sum(0) > 0
        npos = with_pos.sum().clamp(min=1)
        pos_term = torch.log1p(pos)[with_pos].sum() / npos
        neg_term = torch.log1p(neg).sum() / self.n
        return pos_term + neg_term


def build_head(name, P, args):
    if name == "ge2e":
        return GE2E()
    if name == "supcon":
        return SupCon(tau=args.tau)
    if name == "arcface":
        return ArcFace(D_MODEL, P, s=args.scale, m=args.margin)
    if name == "cosface":
        return CosFace(D_MODEL, P, s=args.scale, m=args.cos_margin)
    if name == "proxyanchor":
        return ProxyAnchor(D_MODEL, P)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache_scale.pt")
    ap.add_argument("--loss", choices=["ge2e", "supcon", "arcface", "cosface", "proxyanchor"],
                    default="arcface")
    ap.add_argument("--time", dest="use_time", action="store_true")
    ap.add_argument("--no-time", dest="use_time", action="store_false")
    ap.set_defaults(use_time=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--m-players", type=int, default=64)
    ap.add_argument("--k-games", type=int, default=5)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--margin", type=float, default=0.3)       # arcface
    ap.add_argument("--cos-margin", type=float, default=0.35)  # cosface
    ap.add_argument("--scale", type=float, default=30.0)
    ap.add_argument("--tau", type=float, default=0.1)          # supcon temperature
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--save", default=None, help="save the trained GameEncoder here (for the clone judge)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); random.seed(0)
    tag = f"{args.loss.upper()} {'MOVES+TIME' if args.use_time else 'MOVES-ONLY'}"
    print(f"device: {dev} | {tag}")

    c = torch.load(args.cache, weights_only=False)
    names = c["names"]; P = len(names)
    order = torch.argsort(c["game"].long(), stable=True)
    pooled = c["pooled"][order].contiguous()
    logt = torch.log1p(c["think"][order].clamp(min=0).float())
    tmean, tstd = logt.mean(), logt.std().clamp(min=1e-3)
    logt = (logt - tmean) / tstd
    gsort = c["game"][order].long(); psort = c["pid"][order].long(); tsort = c["test"][order].bool()
    Dp = pooled.shape[1]

    change = torch.ones(len(gsort), dtype=torch.bool); change[1:] = gsort[1:] != gsort[:-1]
    starts = change.nonzero(as_tuple=True)[0]
    ends = torch.cat([starts[1:], torch.tensor([len(gsort)])])
    G = len(starts)
    g_start = starts
    g_len = (ends - starts).clamp(max=MAX_PLIES)
    g_owner = psort[starts]
    g_test = tsort[starts]
    print(f"{P} players | {G} games | {pooled.shape[0]} plies | {int(g_test.sum())} test games")

    def gather(ids):
        b = len(ids)
        pl = torch.zeros(b, MAX_PLIES, Dp); tf = torch.zeros(b, MAX_PLIES)
        mk = torch.zeros(b, MAX_PLIES, dtype=torch.bool)
        for j, i in enumerate(ids):
            s = int(g_start[i]); l = int(g_len[i])
            pl[j, :l] = pooled[s:s + l].float()
            tf[j, :l] = logt[s:s + l]
            mk[j, :l] = True
        return pl.to(dev), tf.to(dev), mk.to(dev)

    enc = GameEncoder(Dp, args.use_time).to(dev)
    head = build_head(args.loss, P, args).to(dev)
    mode = head.mode
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=args.lr, weight_decay=1e-4)

    ref_games = (~g_test).nonzero(as_tuple=True)[0].tolist()
    by_player = {}
    for gi in ref_games:
        by_player.setdefault(int(g_owner[gi]), []).append(gi)
    elig = [p for p, gl in by_player.items() if len(gl) >= args.k_games]
    rng = random.Random(0)

    enc.train(); t0 = time.time()
    for step in range(args.steps):
        if mode in ("tuple", "flat_mk"):
            players = rng.sample(elig, min(args.m_players, len(elig)))
            ids = [g for p in players for g in rng.sample(by_player[p], args.k_games)]
            pl, tf, mk = gather(ids)
            e = enc(pl, tf, mk)
            if mode == "tuple":
                loss = head.loss(e.view(len(players), args.k_games, -1))
            else:
                loss = head.loss(e, g_owner[ids].to(dev))
        else:  # class
            ids = rng.sample(ref_games, min(args.batch, len(ref_games)))
            pl, tf, mk = gather(ids)
            e = enc(pl, tf, mk)
            loss = head.loss(e, g_owner[ids].to(dev))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 3.0)
        opt.step()
        if (step + 1) % 500 == 0:
            print(f"step {step+1}/{args.steps}  {args.loss} loss {loss.item():.4f}  ({time.time()-t0:.0f}s)")

    if args.save:
        torch.save({"state_dict": enc.state_dict(), "use_time": args.use_time, "dp": Dp,
                    "d_model": D_MODEL, "layers": N_LAYERS, "heads": N_HEADS, "max_plies": MAX_PLIES,
                    "tmean": float(tmean), "tstd": float(tstd), "loss": args.loss},
                   args.save)
        print(f"saved recognizer -> {args.save}")

    # ---- eval ----
    @torch.no_grad()
    def embed_all():
        enc.eval(); E = torch.zeros(G, D_MODEL, device=dev)
        for s in range(0, G, args.batch):
            ids = list(range(s, min(s + args.batch, G)))
            pl, tf, mk = gather(ids)
            E[s:s + len(ids)] = enc(pl, tf, mk)
        return E

    E = embed_all()
    tr = (~g_test).to(dev)
    owner_dev = g_owner.to(dev)
    cent = torch.zeros(P, D_MODEL, device=dev)
    for p in range(P):
        m = tr & (owner_dev == p)
        if m.any():
            cent[p] = F.normalize(E[m].mean(0), dim=-1)
    cent = F.normalize(cent, dim=-1)
    ti = g_test.nonzero(as_tuple=True)[0].to(dev)
    Et = E[ti]; own_t = owner_dev[ti]

    def p_at_1(Nsub):
        keep = own_t < Nsub
        if keep.sum() == 0:
            return float("nan")
        return (Et[keep] @ cent[:Nsub].T).argmax(1).eq(own_t[keep]).float().mean().item()

    Ns = [n for n in [10, 30, 100, 300, 1000, P] if n <= P]
    print(f"\n=== {tag} — single game / query ===")
    print("players(N)   P@1     chance")
    for Nsub in Ns:
        print(f"{Nsub:>8}   {p_at_1(Nsub):.3f}   {1/Nsub:.4f}")

    by = {}
    for j, p in enumerate(own_t.cpu().tolist()):
        by.setdefault(p, []).append(j)
    owners = [p for p, l in by.items() if len(l) >= 20]
    print(f"\n=== {tag} — games/query (N={P}) ===")
    print("games   P@1")
    r = random.Random(0)
    for k in (1, 5, 10, 20):
        hit = tot = 0
        for _ in range(500):
            p = r.choice(owners); sel = r.sample(by[p], k)
            q = F.normalize(Et[sel].mean(0), dim=-1)
            hit += int((q @ cent.T).argmax().item() == p); tot += 1
        print(f"{k:>5}   {hit/tot:.3f}")


if __name__ == "__main__":
    main()
