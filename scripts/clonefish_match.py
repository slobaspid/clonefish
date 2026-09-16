"""Play real-time games between the clonefish UCI engine and an opponent engine, exactly as a GUI would.

Clocks are managed here (wall time per move is charged to the mover), passed to each engine via go
wtime/btime, and every move is written to the PGN with a [%clk] comment, like a Lichess export.

    python scripts/clonefish_match.py --player VEGETAL --clone clones/ftval_VEGETAL_bucket.pt \
        --pgn data/lichess_5k/VEGETAL.pgn.zst --games 1 --tc 180 --opp-elo 1500
"""
import argparse, datetime, os, sys, time
import chess, chess.engine, chess.pgn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SF = os.path.join(ROOT, "tools", "stockfish", "stockfish", "stockfish-windows-x86-64-avx2.exe")


def clk(sec):
    sec = max(0, int(sec)); return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def play(clone, opp, clone_white, tc, max_plies, log):
    board, clocks, node = chess.Board(), {chess.WHITE: float(tc), chess.BLACK: float(tc)}, None
    game = chess.pgn.Game(); node = game; result, reason = "*", "max plies"
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        eng = clone if (board.turn == chess.WHITE) == clone_white else opp
        limit = chess.engine.Limit(white_clock=clocks[chess.WHITE], black_clock=clocks[chess.BLACK],
                                   white_inc=0, black_inc=0)
        t0 = time.perf_counter()
        r = eng.play(board, limit, info=chess.engine.INFO_ALL)
        used = time.perf_counter() - t0
        clocks[board.turn] -= used
        if clocks[board.turn] <= 0:
            result, reason = ("0-1" if board.turn == chess.WHITE else "1-0"), "time forfeit"
            log(f"  FLAG by {'white' if board.turn == chess.WHITE else 'black'} at ply {board.ply()}")
            break
        if r.move is None or r.move not in board.legal_moves:
            raise RuntimeError(f"illegal/no move {r.move} at {board.fen()}")
        who = "clone" if eng is clone else "opp"
        if who == "clone":
            log(f"  ply {board.ply():3d} {board.san(r.move):8s} used {used:5.2f}s clock {clocks[board.turn]:6.1f} "
                f"{(r.info or {}).get('string', '')}")
        node = node.add_variation(r.move); node.comment = f"[%clk {clk(clocks[board.turn])}]"
        board.push(r.move)
    if result == "*" and board.is_game_over(claim_draw=True):
        result = board.result(claim_draw=True); reason = "normal"
    game.headers["Result"] = result; game.headers["Termination"] = reason
    return game, board


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", required=True); ap.add_argument("--clone", required=True)
    ap.add_argument("--pgn", required=True); ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--tc", type=float, default=180); ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--opp-elo", type=int, default=1500)
    ap.add_argument("--book-exclude-recent", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "clonefish", "match.pgn"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    log = lambda s: print(s, flush=True)
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "clonefish_uci.py"), "--player", a.player, "--pgn", a.pgn,
           "--clone", a.clone, "--book-exclude-recent", str(a.book_exclude_recent)]
    clone = chess.engine.SimpleEngine.popen_uci(cmd, timeout=600)
    opp = chess.engine.SimpleEngine.popen_uci(SF, timeout=60)
    opp.configure({"UCI_LimitStrength": True, "UCI_Elo": max(1320, a.opp_elo), "Threads": 1})
    try:
        for gi in range(a.games):
            cw = gi % 2 == 0
            log(f"game {gi + 1}: clone plays {'white' if cw else 'black'}")
            clone.configure({"OppElo": a.opp_elo})
            game, board = play(clone, opp, cw, a.tc, a.max_plies, log)
            game.headers.update({"Event": "clonefish match", "White": a.player if cw else f"stockfish{a.opp_elo}",
                                 "Black": f"stockfish{a.opp_elo}" if cw else a.player,
                                 "Date": datetime.date.today().strftime("%Y.%m.%d"), "TimeControl": f"{int(a.tc)}+0"})
            with open(a.out, "a", encoding="utf-8") as f:
                print(game, file=f, end="\n\n")
            log(f"game {gi + 1} done: {game.headers['Result']} ({game.headers['Termination']}), {board.ply()} plies")
    finally:
        clone.quit(); opp.quit()


if __name__ == "__main__":
    main()
