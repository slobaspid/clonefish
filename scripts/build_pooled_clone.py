"""Train the pooled shared move-residual head across many players (the +4pp winner), then export
ONE player's clone as a self-play-able PooledCloneAdapter (good moves + base clock).

    PYTHONPATH=. python scripts/build_pooled_clone.py --cache sweep_cache_120_full.pt \
        --name Gerry_Grob --out clones/Gerry_pooled.pt

Uses the honest cache (train = all-but-last-40, test = last-40). Same recipe as clone_sweep's
best pooled config. Prints the pooled held-out moveΔ as a sanity check before exporting.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
from clone_sweep import PooledResidual, MAXLEG, DEV
from sahformer.clone import PooledCloneAdapter, save_pooled_clone


def train_shared(cache, emb_dim, hidden, wd, epochs, dropout, res_l2, lr, bs=2048):
    torch.manual_seed(0)
    players = list(cache.keys()); P = len(players)
    Ptr, BLtr, LGtr, LNtr, ACtr, PID = [], [], [], [], [], []
    for i, n in enumerate(players):
        tp, tbl, tlg, tln, tac = cache[n]["train"]
        Ptr.append(tp.float()); BLtr.append(tbl.float()); LGtr.append(tlg.long()); LNtr.append(tln); ACtr.append(tac)
        PID.append(torch.full((tp.shape[0],), i))
    # keep big tensors on CPU; move only each batch to GPU (fits large caches on a 6GB card)
    Ptr=torch.cat(Ptr); BLtr=torch.cat(BLtr); LGtr=torch.cat(LGtr)
    LNtr=torch.cat(LNtr); ACtr=torch.cat(ACtr); PID=torch.cat(PID)
    m = PooledResidual(P, emb_dim, hidden=hidden, dropout=dropout).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    N = Ptr.shape[0]; arange_leg = torch.arange(MAXLEG, device=DEV)
    for ep in range(epochs):
        perm = torch.randperm(N); m.train()
        for s in range(0, N, bs):
            b = perm[s:s+bs]
            pb = Ptr[b].to(DEV); bl = BLtr[b].to(DEV); lg = LGtr[b].to(DEV)
            ln = LNtr[b].to(DEV); ac = ACtr[b].to(DEV); pid = PID[b].to(DEV)
            pad = arange_leg[None,:] >= ln[:,None]
            res = m(pb, pid); rl = torch.gather(res, 1, lg)
            logits = (bl + rl).masked_fill(pad, -1e4)
            loss = F.cross_entropy(logits, ac) + res_l2 * res.pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return m, players


@torch.no_grad()
def heldout_moveD(m, cache, players, per_player=False):
    dmove = {}
    for i, n in enumerate(players):
        ep, ebl, elg, eln, eac = cache[n]["test"]
        epf=ep.float().to(DEV); ebf=ebl.float().to(DEV); elf=elg.long().to(DEV); eac_d=eac.to(DEV)
        epad = torch.arange(MAXLEG, device=DEV)[None,:] >= eln.to(DEV)[:,None]
        pid = torch.full((epf.shape[0],), i, device=DEV)
        res = m(epf, pid); rl = torch.gather(res, 1, elf)
        base_lg = ebf.masked_fill(epad, -1e4); clone_lg = (ebf + rl).masked_fill(epad, -1e4)
        dmove[n] = ((clone_lg.argmax(1)==eac_d).float().mean() - (base_lg.argmax(1)==eac_d).float().mean()).item()*100
    if per_player:
        return dmove
    return float(np.mean(list(dmove.values())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="sweep_cache_120_full.pt")
    ap.add_argument("--out", default=None)
    # best pooled config (~+4pp)
    ap.add_argument("--emb", type=int, default=512); ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--wd", type=float, default=3e-2); ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--dropout", type=float, default=0.6); ap.add_argument("--res-l2", type=float, default=0.10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--name", default=None)
    ap.add_argument("--per-player", action="store_true", help="print sorted per-player moveΔ and exit")
    args = ap.parse_args()

    cache = torch.load(args.cache, weights_only=False)
    m, players = train_shared(cache, args.emb, args.hidden, args.wd, args.epochs, args.dropout, args.res_l2, args.lr)

    if args.per_player:
        per = heldout_moveD(m, cache, players, per_player=True)
        vals = np.array([per[n] for n in players])
        # n_train_games per player from the stored gid (only present in _full caches)
        ng = np.array([int(cache[n]["gid"].max()) + 1 if "gid" in cache[n] else 120 for n in players])
        srt = sorted(per.items(), key=lambda kv: -kv[1])
        print(f"\n=== per-player held-out moveΔ ({len(players)} players) ===")
        print(f"mean {vals.mean():+.2f}pp  median {np.median(vals):+.2f}pp  "
              f">=10pp: {(vals>=10).sum()}  >=5pp: {(vals>=5).sum()}  <0: {(vals<0).sum()}")
        if ng.std() > 0:
            r = float(np.corrcoef(ng, vals)[0, 1])
            print(f"corr(moveΔ, n_train_games) = {r:+.2f}")
            print("bucketed by games:")
            for lo, hi in [(0, 140), (140, 200), (200, 260), (260, 10000)]:
                msk = (ng >= lo) & (ng < hi)
                if msk.any():
                    print(f"  games [{lo:>4},{hi if hi<10000 else '+':>4}): "
                          f"{msk.sum():>3} players  meanΔ {vals[msk].mean():+.2f}pp  "
                          f">=10pp {int((vals[msk]>=10).sum())}")
        print("top 12:"); [print(f"  {n:22s} {per[n]:+.1f}pp  ({int(ng[players.index(n)])}g)") for n, d in srt[:12]]
        print("bottom 5:"); [print(f"  {n:22s} {per[n]:+.1f}pp  ({int(ng[players.index(n)])}g)") for n, d in srt[-5:]]
        return

    print(f"pooled shared head trained on {len(players)} players | held-out moveΔ {heldout_moveD(m, cache, players):+.2f}pp")

    if args.name not in players:
        print(f"ERROR: {args.name} not in cache players. available e.g.: {players[:5]}"); sys.exit(1)
    idx = players.index(args.name)
    code = m.emb.weight[idx].detach().cpu()
    net = m.net.cpu().eval()
    adapter = PooledCloneAdapter(net, code)
    # avg elo for self-play strength: pull from the single-player clone meta if present, else 1500
    avg_elo = 1500
    sp = os.path.join("clones", f"{args.name}.pt")
    if os.path.exists(sp):
        try:
            from sahformer.clone import load_clone
            avg_elo = int(load_clone(sp).meta.get("avg_elo", 1500))
        except Exception:
            pass
    save_pooled_clone(args.out, adapter, {"name": args.name, "avg_elo": avg_elo,
                                          "kind": "pooled", "cache": args.cache, "code_dim": args.emb})
    print(f"saved pooled clone -> {args.out}  (avg_elo {avg_elo}, code dim {args.emb})")


if __name__ == "__main__":
    main()
