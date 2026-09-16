"""How much of the ENGINE's move match comes from the personal opening book, and where is headroom left?

Every top-1/top-3 figure measured this session came from the raw model policy, with no book — but the engine blends
the player's own opening book in: q = (n + alpha*p) / (N + alpha). So the engine's real opening accuracy has never
been measured, which matters for one specific reason:

The only mechanism in this project's history with a large move-match gain is retrieval (kNN over the player's own
past positions). On `latebloomer` it gave +8.4pp overall — but ALL of it in the opening (32.1 % -> 65.2 % top-1) and
it HURT the middlegame (52.7 -> 51.1) and endgame (66.2 -> 65.5). Its baseline had no book. If the book already
delivers that opening accuracy, retrieval is redundant and there is nothing to chase; if it does not, there is real
headroom sitting in the engine's opening.

Reports top-1/top-3 by phase for the clone alone vs the clone with its book, on the held-out games the clone never
trained on, plus how often the position is even IN the book (the ceiling on what a book can do).

    PYTHONPATH=. python scripts/clonefish_book_effect.py --player VEGETAL --clone clones/ftval_VEGETAL_bucket.pt
"""
import argparse, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np, torch, torch.nn.functional as F, chess
from finetune_clone import games_of, rows_of, batch_of
from clonefish_uci import player_data
from sahformer.model.heads import move_to_index
from sahformer.encoding import encode_move
from sahformer.training.loop import load_model

CKPT = os.path.join(ROOT, "checkpoints", "base_300k_best.pt")
BANDS = [(0, 10, "opening"), (10, 25, "early-mid"), (25, 40, "middlegame"), (40, 9999, "late/endgame")]
dkey = lambda g: ((g.headers.get("UTCDate") or g.headers.get("Date") or ""),
                  (g.headers.get("UTCTime") or g.headers.get("StartTime") or ""))


def walk(games, me):
    """(row, epd, legal uci list) for every own turn with a clock — one pass, so the book key and the model input
    are guaranteed to describe the same position."""
    out = []
    for g in games:
        if g.board().fen() != chess.STARTING_FEN:
            continue
        rows = rows_of(g, me)
        me_white = (me == (g.headers.get("White", "") or "").lower())
        board, k = g.board(), 0
        for nd in g.mainline():
            mover, c = board.turn, nd.clock()
            if c is None:
                board.push(nd.move); continue
            if (mover == chess.WHITE) == me_white:
                if k < len(rows):
                    out.append((rows[k], board.epd(), [m.uci() for m in board.legal_moves]))
                k += 1
            board.push(nd.move)
    return [(r, e, lg) for r, e, lg in out if r[5] in r[6] and len(r[6]) > 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", required=True); ap.add_argument("--clone", required=True)
    ap.add_argument("--data-dir", default=os.path.join("data", "lichess_5k"))
    ap.add_argument("--alpha", type=float, default=2.0, help="book blend strength, as the engine uses it")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "clonefish", "book_effect.json"))
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    me = a.player.lower(); t0 = time.time()

    model, _ = load_model(CKPT)
    model.load_state_dict(torch.load(a.clone, map_location="cpu", weights_only=False)["model_state"])
    model.eval().to(dev)
    pgn = os.path.join(ROOT, a.data_dir, f"{a.player}.pgn.zst")
    data = player_data(pgn, a.player, exclude_recent=80)
    book = data["counts"]
    gs = games_of(pgn, me); gs.sort(key=dkey)
    items = walk(gs[-80:], me)
    rows = [r for r, _, _ in items]
    print(f"{a.player}: {len(items)} held-out own moves, book has {len(book)} positions", flush=True)

    hits = {b[2]: {"n": 0, "in_book": 0, "c1": 0, "c3": 0, "b1": 0, "b3": 0} for b in BANDS}
    with torch.no_grad():
        for s in range(0, len(rows), 256):
            batch, ch = batch_of(rows, range(s, min(s + 256, len(rows))), dev)
            ml = model(batch)["move_logits"].float().cpu()
            for j, r in enumerate(ch):
                _, epd, legal_uci = items[s + j]
                legal, actual = r[6], r[5]
                p = F.softmax(ml[j][legal], -1).numpy().astype(np.float64)
                cnt = book.get(epd, {})
                n_tot = sum(cnt.values())
                q = p.copy()
                if n_tot:                       # the engine's blend: q = (n + alpha*p) / (N + alpha)
                    n_arr = np.array([cnt.get(u, 0) for u in legal_uci], float)
                    q = (n_arr + a.alpha * p) / (n_tot + a.alpha)
                    q /= q.sum()
                ai = legal.index(actual)
                band = next(nm for lo, hi, nm in BANDS if lo <= r[8] < hi)
                h = hits[band]; h["n"] += 1; h["in_book"] += 1 if n_tot else 0
                for tag, dist in (("c", p), ("b", q)):
                    order = np.argsort(-dist)
                    h[tag + "1"] += int(order[0] == ai)
                    h[tag + "3"] += int(ai in order[:3])

    print(f"  {'phase':<12}{'n':>6}{'in book':>9}{'clone t1':>10}{'+book t1':>10}{'clone t3':>10}{'+book t3':>10}",
          flush=True)
    tot = {k: 0 for k in ("n", "in_book", "c1", "c3", "b1", "b3")}
    for _, _, nm in BANDS:
        h = hits[nm]
        if not h["n"]:
            continue
        for k in tot:
            tot[k] += h[k]
        print(f"  {nm:<12}{h['n']:>6}{100 * h['in_book'] / h['n']:>8.1f}%{100 * h['c1'] / h['n']:>9.1f}%"
              f"{100 * h['b1'] / h['n']:>9.1f}%{100 * h['c3'] / h['n']:>9.1f}%{100 * h['b3'] / h['n']:>9.1f}%",
              flush=True)
    print(f"  {'ALL':<12}{tot['n']:>6}{100 * tot['in_book'] / tot['n']:>8.1f}%{100 * tot['c1'] / tot['n']:>9.1f}%"
          f"{100 * tot['b1'] / tot['n']:>9.1f}%{100 * tot['c3'] / tot['n']:>9.1f}%{100 * tot['b3'] / tot['n']:>9.1f}%",
          flush=True)
    allr = json.load(open(a.out)) if os.path.exists(a.out) else {}
    allr[a.player] = {"bands": hits, "all": tot, "alpha": a.alpha, "book_positions": len(book)}
    json.dump(allr, open(a.out, "w"), indent=1, default=float)
    print(f"wrote {os.path.relpath(a.out, ROOT)} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
