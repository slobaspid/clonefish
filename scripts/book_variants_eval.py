"""Two book variants for the clonefish plan, on the base model (no per-player training):
  E3 recency: weight each past game 0.5 ** (age_in_games / H); H tuned on val (inf = unweighted).
  E5 thin users: book built from only the K most recent games before the eval set (a user who has K games).
Same chronological split as book_blend_eval: train = all but newest 80, val = -80..-40, test = newest 40.
Base logits are computed once per player; variants only change the counts.

    PYTHONPATH=. python scripts/book_variants_eval.py --out results/book_blend/variants.json
"""
import sys, os, json, random, argparse, importlib.util
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from collections import defaultdict, Counter
from sahformer.training.loop import load_model

spec = importlib.util.spec_from_file_location("bb", os.path.join(ROOT, "scripts", "book_blend_eval.py"))
bb = importlib.util.module_from_spec(spec); spec.loader.exec_module(bb)
ALPHAS = [0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]
HALF_LIVES = [None, 800, 400, 200, 100, 50, 25]
KS = [25, 50, 100, 200, 400, None]


def book(games, me, H=None):
    """games ordered oldest->newest; age 0 = the newest game in the list."""
    counts = defaultdict(Counter); n = len(games)
    for i, g in enumerate(games):
        w = 1.0 if H is None else 0.5 ** ((n - 1 - i) / H)
        me_white = (me == (g.headers.get("White", "") or "").lower())
        board = g.board()
        for mv in g.mainline_moves():
            if (board.turn == bb.chess.WHITE) == me_white:
                counts[bb.pos_key(board)][mv.uci()] += w
            board.push(mv)
    return counts


def nll(rows, lg, counts, T, a):
    return float(np.mean([x["q_nll"] for x in bb.evaluate(rows, lg, counts, T, a)]))


def score(rows, lg, counts, T, a):
    ev = bb.evaluate(rows, lg, counts, T, a)
    return dict(top1=float(np.mean([x["q_top1"] for x in ev])), nll=float(np.mean([x["q_nll"] for x in ev])),
                pm=float(np.mean([x["q_pm"] for x in ev])),
                base_top1=float(np.mean([x["b_top1"] for x in ev])), base_nll=float(np.mean([x["b_nll"] for x in ev])))


def run_player(model, pgn, name, n_val=40, n_test=40):
    me = name.lower()
    gs = bb.fc.games_of(pgn, me); gs.sort(key=bb.date_key)
    if len(gs) < n_val + n_test + 100:
        return None
    tr, va, te = gs[:-(n_val + n_test)], gs[-(n_val + n_test):-n_test], gs[-n_test:]
    va_rows = [r for g in va for r in bb.rows_plus(g, me)]; te_rows = [r for g in te for r in bb.rows_plus(g, me)]
    va_lg, te_lg = bb.legal_logits(model, va_rows), bb.legal_logits(model, te_rows)
    T = min(bb.TEMPS, key=lambda t: np.mean([x["b_nll"] for x in bb.evaluate(va_rows, va_lg, {}, t, 1.0)]))
    out = dict(name=name, n_games=len(gs), T=T, recency={}, thin={})
    # E3 recency: select (H, alpha) on val with a train-only book; report test with a train+val book
    best = None
    for H in HALF_LIVES:
        cv = book(tr, me, H)
        a = min(ALPHAS, key=lambda a: nll(va_rows, va_lg, cv, T, a))
        v = nll(va_rows, va_lg, cv, T, a)
        ct = book(tr + va, me, H)
        out["recency"][str(H)] = dict(alpha=a, val_nll=v, test=score(te_rows, te_lg, ct, T, a))
        if best is None or v < best[0]:
            best = (v, H)
    out["recency_best_H"] = best[1]
    # E5 thin users: a user with only K games (the K most recent before the eval set)
    for K in KS:
        cv = book(tr[-K:] if K else tr, me)
        a = min(ALPHAS, key=lambda a: nll(va_rows, va_lg, cv, T, a))
        ct = book((tr + va)[-K:] if K else tr + va, me)
        out["thin"][str(K)] = dict(alpha=a, test=score(te_rows, te_lg, ct, T, a))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/base_300k_best.pt")
    ap.add_argument("--n-lichess", type=int, default=20)
    ap.add_argument("--out", default="results/book_blend/variants.json")
    args = ap.parse_args()
    model, _ = load_model(args.ckpt); model.to(bb.DEV).eval()
    stems = sorted(f[:-8] for f in os.listdir("data/lichess_1k") if f.endswith(".pgn.zst"))
    random.Random(0).shuffle(stems)
    jobs = [(f"data/lichess_1k/{s}.pgn.zst", s) for s in stems[:args.n_lichess]]
    jobs += [("data/latebloomer_full.pgn", "latebloomer"), ("data/clone_dtchess/dtchess.pgn.zst", "dtchess")]
    results = []
    for pgn, name in jobs:
        try:
            r = run_player(model, pgn, name)
        except Exception as e:
            print(f"{name}: ERROR {type(e).__name__}: {e}", flush=True); continue
        if r is None:
            print(f"{name}: skipped", flush=True); continue
        results.append(r)
        rc, th = r["recency"], r["thin"]
        base = rc["None"]["test"]
        print(f"{name:16s} base {base['base_top1']*100:5.1f} | book(all,unweighted) {base['top1']*100:5.1f} | "
              f"best H={r['recency_best_H']} -> {rc[str(r['recency_best_H'])]['test']['top1']*100:5.1f} | thin K: "
              + " ".join(f"{k}:{v['test']['top1']*100:4.1f}" for k, v in th.items()), flush=True)
        json.dump(results, open(args.out, "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
