"""Grounding experiment for the clonefish plan: how much do (a) an exact personal opening book
blended with the base (Dirichlet smoothing), and (b) a per-player temperature, buy on HELD-OUT games?

Honest split per player, CHRONOLOGICAL: book/train = all games except the newest 80;
val = games -80..-40 (tunes alpha and T); test = newest 40 (reported). Base = frozen base_300k_best.
    blend  q(m) = (n_m + alpha * p_base(m)) / (N + alpha)      n_m = player's own count at this position
No per-player training at all. Metrics on the player's own moves: argmax top-1, NLL, and
E[match] = mean prob of the played move (= move-match you actually get when the engine SAMPLES).

    PYTHONPATH=. python scripts/book_blend_eval.py --out results/book_blend/summary.json
"""
import sys, os, json, random, argparse, importlib.util
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, torch, chess, chess.pgn
from collections import defaultdict, Counter
from sahformer.encoding import encode_board, encode_move, build_temporal
from sahformer.records import _stack_history, BASE_SECONDS
from sahformer.model.heads import move_to_index
from sahformer.training.loop import load_model

spec = importlib.util.spec_from_file_location("fc", os.path.join(ROOT, "scripts", "finetune_clone.py"))
fc = importlib.util.module_from_spec(spec); spec.loader.exec_module(fc)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OWN_BUCKETS = [(0, 12, "open"), (12, 24, "earlymid"), (24, 40, "mid"), (40, 9999, "end")]
ALPHAS = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]
TEMPS = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0]


def date_key(g):
    h = g.headers
    return ((h.get("UTCDate") or h.get("Date") or ""), (h.get("UTCTime") or h.get("StartTime") or ""))


def pos_key(board):
    return board.epd()          # no move counters -> transpositions merge


def add_to_book(counts, g, me):
    me_white = (me == (g.headers.get("White", "") or "").lower())
    board = g.board()
    for mv in g.mainline_moves():
        if (board.turn == chess.WHITE) == me_white:
            counts[pos_key(board)][mv.uci()] += 1
        board.push(mv)


def rows_plus(g, me):
    """Mirror of finetune_clone.rows_of (same clock-None skipping) + position key / legal UCIs."""
    me_white = (me == (g.headers.get("White", "") or "").lower())
    we = int(g.headers.get("WhiteElo", 0) or 0); be = int(g.headers.get("BlackElo", 0) or 0)
    board = g.board(); prev = {chess.WHITE: BASE_SECONDS, chess.BLACK: BASE_SECONDS}
    thist = {chess.WHITE: [], chess.BLACK: []}; ph, node, ply, rows, own = [], g, 0, [], 0
    while node.variations:
        node = node.variation(0); mv = node.move; mover = board.turn; ca = node.clock()
        if ca is None:
            board.push(mv); continue
        cur = encode_board(board)
        if (mover == chess.WHITE) == me_white:
            lm = list(board.legal_moves)
            rows.append(dict(board=cur, hist=_stack_history(ph, cur),
                             temporal=build_temporal(prev[mover], prev[not mover], thist[mover], ply),
                             es=(we if me_white else be), eo=(be if me_white else we),
                             legal=[move_to_index(*encode_move(board, m)) for m in lm],
                             legal_uci=[m.uci() for m in lm], played=lm.index(mv),
                             key=pos_key(board), own=own, gply=board.ply()))
            own += 1
        think2 = max(prev[mover] - ca, 0.0); prev[mover] = ca; thist[mover] = [think2] + thist[mover]
        ph.append(cur); board.push(mv); ply += 1
    return rows


@torch.no_grad()
def legal_logits(model, rows):
    out = []
    for s in range(0, len(rows), 256):
        ch = rows[s:s + 256]
        batch = {"board": torch.from_numpy(np.stack([r["board"] for r in ch])).float().to(DEV),
                 "history": torch.from_numpy(np.stack([r["hist"] for r in ch])).float().to(DEV),
                 "elo_self": torch.tensor([r["es"] for r in ch]).to(DEV),
                 "elo_opp": torch.tensor([r["eo"] for r in ch]).to(DEV),
                 "temporal": torch.from_numpy(np.stack([r["temporal"] for r in ch])).float().to(DEV)}
        ml = model(batch)["move_logits"].float().cpu().numpy()
        for j, r in enumerate(ch):
            out.append(ml[j][r["legal"]].astype(np.float64))
    return out


def probs(lg, T):
    z = lg / T; z = z - z.max(); p = np.exp(z); return p / p.sum()


def book_vec(counts, r):
    c = counts.get(r["key"])
    return np.array([c.get(u, 0) for u in r["legal_uci"]], float) if c else np.zeros(len(r["legal_uci"]))


def evaluate(rows, logits, counts, T, alpha):
    out = []
    for r, lg in zip(rows, logits):
        p = probs(lg, T); n = book_vec(counts, r); N = n.sum()
        q = (n + alpha * p) / (N + alpha)
        k = r["played"]
        out.append(dict(own=r["own"], gply=r["gply"], N=N,
                        b_top1=float(p.argmax() == k), b_nll=-np.log(max(p[k], 1e-12)), b_pm=p[k],
                        q_top1=float(q.argmax() == k), q_nll=-np.log(max(q[k], 1e-12)), q_pm=q[k],
                        bk_top1=float(n.argmax() == k) if N > 0 else np.nan))
    return out


def agg(ev, sel=lambda e: True):
    e = [x for x in ev if sel(x)]
    if not e:
        return None
    m = lambda f: float(np.nanmean([x[f] for x in e]))
    return dict(n=len(e), base_top1=m("b_top1"), blend_top1=m("q_top1"), base_nll=m("b_nll"),
                blend_nll=m("q_nll"), base_pm=m("b_pm"), blend_pm=m("q_pm"),
                cov1=float(np.mean([x["N"] >= 1 for x in e])), cov3=float(np.mean([x["N"] >= 3 for x in e])))


def run_player(model, pgn, name, n_val=40, n_test=40):
    me = name.lower()
    gs = fc.games_of(pgn, me)
    gs.sort(key=date_key)
    if len(gs) < n_val + n_test + 100:
        return None
    tr, va, te = gs[:-(n_val + n_test)], gs[-(n_val + n_test):-n_test], gs[-n_test:]
    counts = defaultdict(Counter)
    for g in tr:
        add_to_book(counts, g, me)
    va_rows = [r for g in va for r in rows_plus(g, me)]
    te_rows = [r for g in te for r in rows_plus(g, me)]
    va_lg, te_lg = legal_logits(model, va_rows), legal_logits(model, te_rows)
    # tune T on base NLL (val), then alpha on blend NLL (val) given T*
    T = min(TEMPS, key=lambda t: np.mean([x["b_nll"] for x in evaluate(va_rows, va_lg, counts, t, 1.0)]))
    A = min(ALPHAS, key=lambda a: np.mean([x["q_nll"] for x in evaluate(va_rows, va_lg, counts, T, a)]))
    for g in va:
        add_to_book(counts, g, me)                    # book for test = train + val games
    ev_T1 = evaluate(te_rows, te_lg, counts, 1.0, A)  # base untempered, for reference
    ev = evaluate(te_rows, te_lg, counts, T, A)
    res = dict(name=name, n_games=len(gs), n_book_games=len(tr) + len(va), T=T, alpha=A,
               elo=float(np.median([r["es"] for r in te_rows])) if te_rows else None,
               base_nll_T1=float(np.mean([x["b_nll"] for x in ev_T1])),
               base_pm_T1=float(np.mean([x["b_pm"] for x in ev_T1])),
               all=agg(ev), maia_ply10=agg(ev, lambda e: e["gply"] >= 10),
               phases={lab: agg(ev, lambda e, lo=lo, hi=hi: lo <= e["own"] < hi) for lo, hi, lab in OWN_BUCKETS})
    bk = [x["bk_top1"] for x in ev if x["N"] >= 3]
    res["book_only_top1_where_N3"] = float(np.mean(bk)) if bk else None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/base_300k_best.pt")
    ap.add_argument("--n-lichess", type=int, default=20)
    ap.add_argument("--out", default="results/book_blend/summary.json")
    args = ap.parse_args()
    model, _ = load_model(args.ckpt); model.to(DEV).eval()
    stems = sorted(f[:-8] for f in os.listdir("data/lichess_1k") if f.endswith(".pgn.zst"))
    random.Random(0).shuffle(stems)
    jobs = [("data/latebloomer_full.pgn", "latebloomer"), ("data/clone_dtchess/dtchess.pgn.zst", "dtchess")]
    jobs += [(f"data/lichess_1k/{s}.pgn.zst", s) for s in stems[:args.n_lichess]]
    results = []
    for pgn, name in jobs:
        try:
            r = run_player(model, pgn, name)
        except Exception as e:
            print(f"{name}: ERROR {type(e).__name__}: {e}", flush=True)
            continue
        if r is None:
            print(f"{name}: skipped (too few games)", flush=True)
            continue
        results.append(r)
        a, o = r["all"], r["phases"]["open"]
        print(f"{name:18s} g={r['n_games']:5d} elo={r['elo']:.0f} T*={r['T']:.1f} a*={r['alpha']:>6} | "
              f"ALL top1 {a['base_top1']*100:5.1f}->{a['blend_top1']*100:5.1f}  "
              f"E[match] {r['base_pm_T1']*100:5.1f}/{a['base_pm']*100:5.1f}->{a['blend_pm']*100:5.1f}  "
              f"nll {r['base_nll_T1']:.3f}/{a['base_nll']:.3f}->{a['blend_nll']:.3f} | "
              f"OPEN top1 {o['base_top1']*100:5.1f}->{o['blend_top1']*100:5.1f} cov3 {o['cov3']*100:4.0f}%",
              flush=True)
        json.dump(results, open(args.out, "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
