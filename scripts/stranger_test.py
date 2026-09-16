"""STRANGER TEST — the architecture's actual promise: clone players the shared head NEVER trained on.

Split the cache players into HEAD-players (train the shared head + their codes) and STRANGERS
(disjoint, never seen by the head). Then FREEZE the head and fit ONLY each stranger's 512-d code
on their train games, eval their held-out games. If stranger moveΔ ~ in-sample moveΔ -> the head
generalizes (clone-anyone works). If strangers collapse -> the head overspilled onto its training set.

    PYTHONPATH=. python scripts/stranger_test.py --cache sweep_cache_300_full.pt --strangers 60
"""
import argparse, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from clone_sweep import PooledResidual, MAXLEG, DEV
from build_pooled_clone import train_shared, heldout_moveD


def gather(cache_sub, players):
    P, BL, LG, LN, AC, PID = [], [], [], [], [], []
    for i, n in enumerate(players):
        tp, tbl, tlg, tln, tac = cache_sub[n]["train"]
        P.append(tp.float()); BL.append(tbl.float()); LG.append(tlg.long()); LN.append(tln); AC.append(tac)
        PID.append(torch.full((tp.shape[0],), i))
    return (torch.cat(P).to(DEV), torch.cat(BL).to(DEV), torch.cat(LG).to(DEV),
            torch.cat(LN).to(DEV), torch.cat(AC).to(DEV), torch.cat(PID).to(DEV))


def fit_stranger_codes(net, cache_str, players, emb_dim, epochs, lr, wd, res_l2, bs=2048):
    """Freeze the head, train ONLY fresh per-stranger codes on their train games."""
    Ptr, BLtr, LGtr, LNtr, ACtr, PID = gather(cache_str, players)
    pad = torch.arange(MAXLEG, device=DEV)[None, :] >= LNtr[:, None]
    emb = nn.Embedding(len(players), emb_dim).to(DEV); nn.init.normal_(emb.weight, std=0.01)
    net.eval()                                                     # frozen + deterministic (dropout off)
    for p in net.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(emb.parameters(), lr=lr, weight_decay=wd)
    N = Ptr.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(N, device=DEV)
        for s in range(0, N, bs):
            b = perm[s:s + bs]
            res = net(torch.cat([Ptr[b], emb(PID[b])], -1))
            rl = torch.gather(res, 1, LGtr[b])
            logits = (BLtr[b] + rl).masked_fill(pad[b], -1e4)
            loss = F.cross_entropy(logits, ACtr[b]) + res_l2 * res.pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return emb


@torch.no_grad()
def eval_moveD(net, emb, cache_sub, players):
    net.eval(); out = {}
    for i, n in enumerate(players):
        ep, ebl, elg, eln, eac = cache_sub[n]["test"]
        epf = ep.float().to(DEV); ebf = ebl.float().to(DEV); elf = elg.long().to(DEV); eac_d = eac.to(DEV)
        pad = torch.arange(MAXLEG, device=DEV)[None, :] >= eln.to(DEV)[:, None]
        code = emb(torch.full((epf.shape[0],), i, device=DEV))
        res = net(torch.cat([epf, code], -1)); rl = torch.gather(res, 1, elf)
        base_lg = ebf.masked_fill(pad, -1e4); clone_lg = (ebf + rl).masked_fill(pad, -1e4)
        out[n] = ((clone_lg.argmax(1) == eac_d).float().mean() - (base_lg.argmax(1) == eac_d).float().mean()).item() * 100
    return out


def summ(tag, d):
    v = np.array(list(d.values()))
    print(f"{tag:16s} mean {v.mean():+.2f}pp  median {np.median(v):+.2f}pp  "
          f">=5pp {int((v>=5).sum())}/{len(v)}  <0 {int((v<0).sum())}/{len(v)}")
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="sweep_cache_300_full.pt")
    ap.add_argument("--strangers", type=int, default=60)
    ap.add_argument("--emb", type=int, default=512); ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--wd", type=float, default=3e-2); ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--dropout", type=float, default=0.6); ap.add_argument("--res-l2", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cache = torch.load(args.cache, weights_only=False)
    names = list(cache.keys()); random.Random(args.seed).shuffle(names)
    strangers = names[:args.strangers]; head_players = names[args.strangers:]
    cache_head = {n: cache[n] for n in head_players}
    cache_str = {n: cache[n] for n in strangers}
    print(f"cache {args.cache}: {len(names)} players -> HEAD {len(head_players)} | STRANGERS {len(strangers)} (disjoint)\n")

    # 1) train shared head + head codes on head-players only
    m, hp = train_shared(cache_head, args.emb, args.hidden, args.wd, args.epochs, args.dropout, args.res_l2, 1e-3)
    insample = heldout_moveD(m, cache_head, hp, per_player=True)

    # 2) FREEZE head, fit ONLY stranger codes, eval their held-out
    emb_str = fit_stranger_codes(m.net, cache_str, strangers, args.emb, args.epochs, 1e-3, args.wd, args.res_l2)
    stranger = eval_moveD(m.net, emb_str, cache_str, strangers)

    print("=== held-out moveΔ (clone vs base, last-40 games) ===")
    vin = summ("IN-SAMPLE", insample)
    vst = summ("STRANGERS", stranger)
    keep = vin.mean() and (vst.mean() / vin.mean() * 100)
    print(f"\nstranger retains {vst.mean()/max(vin.mean(),1e-9)*100:.0f}% of in-sample moveΔ  "
          f"({vst.mean():+.2f} vs {vin.mean():+.2f}pp)")
    srt = sorted(stranger.items(), key=lambda kv: -kv[1])
    print("top strangers:", [f"{n} {d:+.1f}" for n, d in srt[:6]])


if __name__ == "__main__":
    main()
