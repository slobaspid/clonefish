"""Control for the personal-book result: is the gain PERSONAL, or just Lichess opening habits the
chess.com-trained base doesn't know? Compare, on the same chronological splits as book_blend_eval:
  base | base+cohort book | base+personal book | base+cohort+personal (hierarchical)
Cohort book = the OTHER lichess_1k players' own moves, only from games dated BEFORE the target's first
val game (no look-ahead). Hierarchical: r = (n_cohort + beta*p)/(N_cohort + beta);  q = (n_user + alpha*r)/(N_user + alpha).
beta tuned on val (cohort-only), then alpha given beta. Opening region only (own moves < 20) is stored.

    PYTHONPATH=. python scripts/book_population_control.py --out results/book_blend/population.json
"""
import sys, os, json, random, argparse, importlib.util
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, chess
from collections import defaultdict, Counter
from sahformer.training.loop import load_model

spec = importlib.util.spec_from_file_location("bb", os.path.join(ROOT, "scripts", "book_blend_eval.py"))
bb = importlib.util.module_from_spec(spec); spec.loader.exec_module(bb)
GRID = [0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
MAX_OWN = 20


def own_entries(g, me):
    """[(key, uci)] for the player's own moves among their first MAX_OWN."""
    me_white = (me == (g.headers.get("White", "") or "").lower())
    board = g.board(); out = []; own = 0
    for mv in g.mainline_moves():
        if (board.turn == chess.WHITE) == me_white:
            if own >= MAX_OWN:
                break
            out.append((bb.pos_key(board), mv.uci())); own += 1
        board.push(mv)
    return out


def counts_from(entries):
    c = defaultdict(Counter)
    for k, u in entries:
        c[k][u] += 1
    return c


def vec(c, r):
    x = c.get(r["key"])
    return np.array([x.get(u, 0) for u in r["legal_uci"]], float) if x else np.zeros(len(r["legal_uci"]))


def eval_cfg(rows, lg, cu, cc, alpha, beta, use_user, use_coh):
    top1, nll = [], []
    for r, l in zip(rows, lg):
        p = bb.probs(l, 1.0); k = r["played"]
        if use_coh:
            n = vec(cc, r); p = (n + beta * p) / (n.sum() + beta)
        if use_user:
            n = vec(cu, r); p = (n + alpha * p) / (n.sum() + alpha)
        top1.append(float(p.argmax() == k)); nll.append(-np.log(max(p[k], 1e-12)))
    return float(np.mean(top1)), float(np.mean(nll))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/base_300k_best.pt")
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--out", default="results/book_blend/population.json")
    args = ap.parse_args()
    stems = sorted(f[:-8] for f in os.listdir("data/lichess_1k") if f.endswith(".pgn.zst"))
    print("indexing all lichess_1k players (own opening moves + dates) ...", flush=True)
    idx = {}                                   # name -> list of (date_key, [(key, uci)])
    for s in stems:
        gs = bb.fc.games_of(f"data/lichess_1k/{s}.pgn.zst", s.lower())
        idx[s] = [(bb.date_key(g), own_entries(g, s.lower())) for g in gs]
    print("indexed", len(idx), "players", flush=True)
    model, _ = load_model(args.ckpt); model.to(bb.DEV).eval()
    order = stems[:]; random.Random(0).shuffle(order)          # same 20 players as book_blend_eval
    results = []
    for name in order[:args.n_eval]:
        me = name.lower()
        gs = bb.fc.games_of(f"data/lichess_1k/{name}.pgn.zst", me); gs.sort(key=bb.date_key)
        tr, va, te = gs[:-80], gs[-80:-40], gs[-40:]
        cutoff = bb.date_key(va[0])
        coh = [e for other, lst in idx.items() if other != name for d, ents in lst if d < cutoff for e in ents]
        cc = counts_from(coh)
        cu_val = counts_from([e for g in tr for e in own_entries(g, me)])
        cu_test = counts_from([e for g in tr + va for e in own_entries(g, me)])
        va_rows = [r for g in va for r in bb.rows_plus(g, me)]; te_rows = [r for g in te for r in bb.rows_plus(g, me)]
        va_lg, te_lg = bb.legal_logits(model, va_rows), bb.legal_logits(model, te_rows)
        a_u = min(GRID, key=lambda a: eval_cfg(va_rows, va_lg, cu_val, cc, a, 1, True, False)[1])
        b_c = min(GRID, key=lambda b: eval_cfg(va_rows, va_lg, cu_val, cc, 1, b, False, True)[1])
        a_h = min(GRID, key=lambda a: eval_cfg(va_rows, va_lg, cu_val, cc, a, b_c, True, True)[1])
        res = dict(name=name, n_cohort_moves=len(coh), alpha_user=a_u, beta_coh=b_c, alpha_hier=a_h,
                   base=eval_cfg(te_rows, te_lg, cu_test, cc, 1, 1, False, False),
                   cohort=eval_cfg(te_rows, te_lg, cu_test, cc, 1, b_c, False, True),
                   personal=eval_cfg(te_rows, te_lg, cu_test, cc, a_u, 1, True, False),
                   hier=eval_cfg(te_rows, te_lg, cu_test, cc, a_h, b_c, True, True))
        results.append(res)
        f = lambda k: f"{res[k][0]*100:5.1f}/{res[k][1]:.3f}"
        print(f"{name:16s} top1/nll  base {f('base')}  cohort {f('cohort')}  personal {f('personal')}  "
              f"cohort+personal {f('hier')}  (beta {b_c}, alpha {a_h})", flush=True)
        json.dump(results, open(args.out, "w"), indent=1)
    R = results
    for k in ("cohort", "personal", "hier"):
        d = np.array([(r[k][0] - r["base"][0]) * 100 for r in R]); n = np.array([r[k][1] - r["base"][1] for r in R])
        print(f"MEAN vs base  {k:9s} top1 {d.mean():+.2f}pp  nll {n.mean():+.4f}", flush=True)
    print("saved", args.out)


if __name__ == "__main__":
    main()
