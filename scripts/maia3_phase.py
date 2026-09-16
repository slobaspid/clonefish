"""Phase-split move-matching for the SOTA public model (Maia-3), on our players' held-out games.
Tests whether Maia-3's accuracy shows the same opening-hard / midgame-easier profile we found,
i.e. that the phase lens holds on the strongest public human model (a citable external check).

    PYTHONPATH=. python scripts/maia3_phase.py --model maia3-23m --players latebloomer,couli,OKENITE --games 40
"""
import argparse, os, sys, io, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import zstandard, chess, chess.pgn, chess.engine, numpy as np

BUCKETS = [(0, 12, "opening"), (12, 24, "early-mid"), (24, 40, "middlegame"), (40, 9999, "late/endgame")]


def own_positions(path, me, last_n):
    """Return (board_with_history, actual_move, own_ply_index, my_elo) for the player's moves in their last_n games."""
    me = me.lower()
    text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(path, "rb")), encoding="utf-8", errors="ignore")
    games = []
    while True:
        g = chess.pgn.read_game(text)
        if g is None: break
        w = (g.headers.get("White","") or "").lower(); b = (g.headers.get("Black","") or "").lower()
        if me not in (w, b): continue
        games.append(g)
    out = []
    for g in games[-last_n:]:
        me_white = (me == (g.headers.get("White","") or "").lower())
        elo = int(g.headers.get("WhiteElo", 0) or 0) if me_white else int(g.headers.get("BlackElo", 0) or 0)
        board = g.board(); node = g; own = 0
        while node.variations:
            node = node.variation(0); mv = node.move
            if (board.turn == chess.WHITE) == me_white:
                out.append((board.copy(stack=True), mv, own, elo)); own += 1
            board.push(mv)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="maia3-23m")
    ap.add_argument("--players", default="latebloomer,couli,OKENITE,Yespapa,sawabear")
    ap.add_argument("--data-dir", default="data/lichess_1k")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--cap", type=int, default=1500, help="max positions per player (speed)")
    args = ap.parse_args()

    agg = {lab: [0, 0] for _,_,lab in BUCKETS}   # [correct, n]
    for name in args.players.split(","):
        pos = own_positions(f"{args.data_dir}/{name}.pgn.zst", name, args.games)
        if not pos:
            print(f"{name}: no games"); continue
        elo = int(np.median([p[3] for p in pos if p[3] > 0]) or 1700)
        elo = min(max(elo, 1100), 2000)
        eng = chess.engine.SimpleEngine.popen_uci([args.model+"-uci" if False else "maia3-uci",
                                                   "--model", args.model, "--use-uci-history", "--elo", str(elo)])
        pos = pos[:args.cap]
        c = {lab:[0,0] for _,_,lab in BUCKETS}
        for board, actual, ply, _ in pos:
            pred = eng.play(board, limit=chess.engine.Limit(nodes=1)).move
            for lo,hi,lab in BUCKETS:
                if lo <= ply < hi:
                    c[lab][0] += int(pred == actual); c[lab][1] += 1
                    agg[lab][0] += int(pred == actual); agg[lab][1] += 1
                    break
        eng.close()
        tot = [sum(c[l][k] for _,_,l in BUCKETS) for k in range(2)]
        print(f"{name:14s} elo{elo} n={tot[1]:>4}  overall {tot[0]/max(tot[1],1)*100:.1f}%  | "
              + " ".join(f"{lab.split('/')[0]} {c[lab][0]/max(c[lab][1],1)*100:.0f}%" for _,_,lab in BUCKETS), flush=True)

    print(f"\n=== Maia-3 ({args.model}) move-match by phase, aggregate ===")
    print(f"{'phase':>14}{'n':>8}{'top-1':>9}")
    for _,_,lab in BUCKETS:
        cc, nn = agg[lab]
        if nn: print(f"{lab:>14}{nn:>8}{cc/nn*100:>8.1f}%")
    t = [sum(agg[l][k] for _,_,l in BUCKETS) for k in range(2)]
    print(f"{'ALL':>14}{t[1]:>8}{t[0]/max(t[1],1)*100:>8.1f}%")


if __name__ == "__main__":
    main()
