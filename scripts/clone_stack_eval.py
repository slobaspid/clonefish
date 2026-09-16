"""Does a personal opening book still add anything ON TOP of the best existing clone (the pooled
code against the frozen 120-player shared head)? And how do per-player temperature and the book
stack? Decides the clonefish architecture.

Per player, CHRONOLOGICAL split: train = all but newest 80 games (capped to the most recent
--max-train), val = games -80..-40 (tunes T and alpha per config), test = newest 40 (reported).
Book AND pooled code both use the SAME train games only (fair comparison).
Configs: base | base+T | pooled | pooled+T | base+book | pooled+book (T, alpha tuned on val).
Players are flagged in_head=True if the shared head's training cache contains them (not a stranger).

    PYTHONPATH=. python scripts/clone_stack_eval.py --out results/book_blend/stack.json
"""
import sys, os, json, random, argparse, importlib.util
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, torch, torch.nn.functional as F
from collections import defaultdict, Counter
from sahformer.training.loop import load_model

spec = importlib.util.spec_from_file_location("bb", os.path.join(ROOT, "scripts", "book_blend_eval.py"))
bb = importlib.util.module_from_spec(spec); spec.loader.exec_module(bb)
DEV = bb.DEV
MAXLEG = 72
bb.ALPHAS = [0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]   # extended: AlexDC's alpha* sat on the old 0.25 edge


@torch.no_grad()
def feats(model, rows):
    """pooled [n,512] (cpu fp32), per-row legal logits (np), legal idx padded [n,MAXLEG], lens, action."""
    P, LG = [], []
    for s in range(0, len(rows), 256):
        ch = rows[s:s + 256]
        batch = {"board": torch.from_numpy(np.stack([r["board"] for r in ch])).float().to(DEV),
                 "history": torch.from_numpy(np.stack([r["hist"] for r in ch])).float().to(DEV),
                 "elo_self": torch.tensor([r["es"] for r in ch]).to(DEV),
                 "elo_opp": torch.tensor([r["eo"] for r in ch]).to(DEV),
                 "temporal": torch.from_numpy(np.stack([r["temporal"] for r in ch])).float().to(DEV)}
        o = model(batch)
        P.append(o["pooled"].float().cpu())
        ml = o["move_logits"].float().cpu().numpy()
        for j, r in enumerate(ch):
            LG.append(ml[j][r["legal"]].astype(np.float64))
    return torch.cat(P) if P else torch.zeros(0, 512), LG


def rows_in_chunks(model, games, me):
    """Build rows game by game and keep only what we need (keeps RAM bounded)."""
    keep, pooled, logits = [], [], []
    for i in range(0, len(games), 50):
        rows = [r for g in games[i:i + 50] for r in bb.rows_plus(g, me)]
        if not rows:
            continue
        p, lg = feats(model, rows)
        pooled.append(p); logits += lg
        keep += [dict(legal=r["legal"], legal_uci=r["legal_uci"], played=r["played"], key=r["key"],
                      own=r["own"], gply=r["gply"]) for r in rows]
    return keep, (torch.cat(pooled) if pooled else torch.zeros(0, 512)), logits


def fit_code(net, pooled, rows, logits, emb_dim=512, epochs=20, lr=1e-3, wd=3e-2, res_l2=0.05, bs=2048):
    n = len(rows)
    li = torch.zeros(n, MAXLEG, dtype=torch.long); ln = torch.zeros(n, dtype=torch.long)
    bl = torch.full((n, MAXLEG), -1e4); ac = torch.tensor([r["played"] for r in rows])
    for i, (r, lg) in enumerate(zip(rows, logits)):
        k = min(len(r["legal"]), MAXLEG)
        li[i, :k] = torch.tensor(r["legal"][:k]); ln[i] = k; bl[i, :k] = torch.from_numpy(lg[:k]).float()
    ok = ac < ln                                            # drop the rare row whose move sits past MAXLEG
    li, ln, bl, ac, P = li[ok].to(DEV), ln[ok].to(DEV), bl[ok].to(DEV), ac[ok].to(DEV), pooled[ok].to(DEV)
    pad = torch.arange(MAXLEG, device=DEV)[None, :] >= ln[:, None]
    code = torch.nn.Parameter(torch.randn(emb_dim, device=DEV) * 0.01)
    opt = torch.optim.AdamW([code], lr=lr, weight_decay=wd)
    N = P.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(N, device=DEV)
        for s in range(0, N, bs):
            b = perm[s:s + bs]
            res = net(torch.cat([P[b], code.expand(len(b), -1)], -1))
            logit = (bl[b] + torch.gather(res, 1, li[b])).masked_fill(pad[b], -1e4)
            loss = F.cross_entropy(logit, ac[b]) + res_l2 * res.pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return code.detach()


@torch.no_grad()
def clone_logits(net, code, pooled, rows, logits):
    out = []
    for s in range(0, len(rows), 1024):
        P = pooled[s:s + 1024].to(DEV)
        res = net(torch.cat([P, code.expand(P.shape[0], -1)], -1)).cpu().numpy()
        for j, (r, lg) in enumerate(zip(rows[s:s + 1024], logits[s:s + 1024])):
            out.append(lg + res[j][r["legal"]].astype(np.float64))
    return out


def tune(rows, lg, counts, use_book):
    T = min(bb.TEMPS, key=lambda t: np.mean([x["b_nll"] for x in bb.evaluate(rows, lg, counts, t, 1.0)]))
    if not use_book:
        return T, None
    A = min(bb.ALPHAS, key=lambda a: np.mean([x["q_nll"] for x in bb.evaluate(rows, lg, counts, T, a)]))
    return T, A


def summarize(ev, which):
    """which='b' -> the model probs, 'q' -> the blend."""
    def s(sel):
        e = [x for x in ev if sel(x)]
        if not e:
            return None
        return dict(n=len(e), top1=float(np.mean([x[which + "_top1"] for x in e])),
                    nll=float(np.mean([x[which + "_nll"] for x in e])), pm=float(np.mean([x[which + "_pm"] for x in e])))
    return dict(all=s(lambda e: True), ply10=s(lambda e: e["gply"] >= 10),
                **{lab: s(lambda e, lo=lo, hi=hi: lo <= e["own"] < hi) for lo, hi, lab in bb.OWN_BUCKETS})


def run_player(model, net, pgn, name, max_train, n_val=40, n_test=40):
    me = name.lower()
    gs = bb.fc.games_of(pgn, me); gs.sort(key=bb.date_key)
    if len(gs) < n_val + n_test + 100:
        return None
    tr = gs[:-(n_val + n_test)][-max_train:]; va = gs[-(n_val + n_test):-n_test]; te = gs[-n_test:]
    counts = defaultdict(Counter)
    for g in tr:
        bb.add_to_book(counts, g, me)
    tr_rows, tr_P, tr_lg = rows_in_chunks(model, tr, me)
    va_rows, va_P, va_lg = rows_in_chunks(model, va, me)
    te_rows, te_P, te_lg = rows_in_chunks(model, te, me)
    code = fit_code(net, tr_P, tr_rows, tr_lg)
    va_c, te_c = clone_logits(net, code, va_P, va_rows, va_lg), clone_logits(net, code, te_P, te_rows, te_lg)
    res = dict(name=name, n_games=len(gs), n_train=len(tr), n_train_moves=len(tr_rows))
    cfg = {}
    for tag, (vl, tl) in {"base": (va_lg, te_lg), "pooled": (va_c, te_c)}.items():
        T, A = tune(va_rows, vl, counts, use_book=True)
        cfg[tag] = summarize(bb.evaluate(te_rows, tl, counts, 1.0, 1.0), "b")
        cfg[tag + "+T"] = summarize(bb.evaluate(te_rows, tl, counts, T, 1.0), "b")
        cfg[tag + "+book"] = summarize(bb.evaluate(te_rows, tl, counts, T, A), "q")
        res[tag + "_T"], res[tag + "_alpha"] = T, A
    res["cfg"] = cfg
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/base_300k_best.pt")
    ap.add_argument("--head", default="clones/latebloomer_pooled.pt")
    ap.add_argument("--head-cache", default="sweep_cache_120_full.pt")
    ap.add_argument("--n-players", type=int, default=16)
    ap.add_argument("--cohort", default="stranger", choices=["stranger", "in_head"],
                    help="which lichess_1k players to evaluate: those NOT in the shared head "
                         "(default) or those IN it. The in_head arm controls the confound that the "
                         "only in-head player measured so far (latebloomer) is a low-base outlier.")
    ap.add_argument("--max-train", type=int, default=1000)
    ap.add_argument("--out", default="results/book_blend/stack.json")
    ap.add_argument("--resume", default=None, help="json of players already done; they are skipped and kept")
    args = ap.parse_args()
    model, _ = load_model(args.ckpt); model.to(DEV).eval()
    net = torch.load(args.head, map_location=DEV, weights_only=False)["net"].to(DEV).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    head_players = set(k.lower() for k in torch.load(args.head_cache, map_location="cpu", mmap=True,
                                                     weights_only=False).keys())
    print(f"shared head trained on {len(head_players)} players", flush=True)
    stems = sorted(f[:-8] for f in os.listdir("data/lichess_1k") if f.endswith(".pgn.zst"))
    strangers = [s for s in stems if s.lower() not in head_players]
    insiders = [s for s in stems if s.lower() in head_players]
    print(f"lichess_1k: {len(stems)} players, {len(strangers)} NOT in the head (strangers), "
          f"{len(insiders)} in the head", flush=True)
    pool = insiders if args.cohort == "in_head" else strangers
    random.Random(1).shuffle(pool)          # same seed/order as before for the stranger arm
    jobs = [(f"data/lichess_1k/{s}.pgn.zst", s) for s in pool[:args.n_players]]
    if args.cohort == "stranger":
        jobs += [("data/clone_dtchess/dtchess.pgn.zst", "dtchess"), ("data/latebloomer_full.pgn", "latebloomer")]
    results = json.load(open(args.resume)) if args.resume else []   # players already done (e.g. before a crash)
    done = {r["name"] for r in results}
    jobs = [j for j in jobs if j[1] not in done]
    for pgn, name in jobs:
        try:
            r = run_player(model, net, pgn, name, args.max_train)
        except Exception as e:
            print(f"{name}: ERROR {type(e).__name__}: {e}", flush=True)
            continue
        if r is None:
            print(f"{name}: skipped", flush=True)
            continue
        r["in_head"] = name.lower() in head_players
        results.append(r)
        c = r["cfg"]
        f = lambda k: c[k]["all"]
        print(f"{name:18s} head={'Y' if r['in_head'] else 'n'} trainG={r['n_train']:4d} | top1 "
              f"base {f('base')['top1']*100:5.1f} pooled {f('pooled')['top1']*100:5.1f} "
              f"base+book {f('base+book')['top1']*100:5.1f} pooled+book {f('pooled+book')['top1']*100:5.1f} | nll "
              f"{f('base')['nll']:.3f} {f('pooled+T')['nll']:.3f} {f('base+book')['nll']:.3f} {f('pooled+book')['nll']:.3f} | "
              f"T base {r['base_T']} pooled {r['pooled_T']} a {r['pooled_alpha']}", flush=True)
        json.dump(results, open(args.out, "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
