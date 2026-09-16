"""Replay the player's REAL training games through the ENGINE code path and sample its think times.

Separates two explanations for the free-play scramble gap (clone plays fewer instant moves at 5-10 s):
  pipeline : the engine builds clock / think-history inputs differently from training
  situation: free-play scrambles differ (e.g. the simulated opponent keeps more time than a human would)
For each own move we feed CloneEngine.choose() the real position, the real whole-second clocks and (through its
own clock-reading logic) the real think history, then record the sampled think as a Lichess reading (whole
seconds, uniform clock phase). Variant OPP+60 gives the opponent 60 extra seconds on every move.

    PYTHONPATH=. python scripts/clonefish_engine_replay.py --players VEGETAL:clones/ftval_VEGETAL_bucket.pt,...
"""
import argparse, json, math, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np, torch, chess
from finetune_clone import games_of
from clonefish_uci import CloneEngine, player_data

CKPT = os.path.join(ROOT, "checkpoints", "base_300k_best.pt")
BUCKETS = [(0, 5, "<5s"), (5, 10, "5-10s"), (10, 30, "10-30s"), (30, 60, "30-60s")]
dkey = lambda g: ((g.headers.get("UTCDate") or g.headers.get("Date") or ""),
                  (g.headers.get("UTCTime") or g.headers.get("StartTime") or ""))


def replay(eng, games, me, opp_bonus, rng):
    out = []                                                  # (clock_before, real_reading, engine_reading)
    for g in games:
        if g.board().fen() != chess.STARTING_FEN:
            continue
        me_white = (me == (g.headers.get("White", "") or "").lower())
        board, moves, prev = g.board(), [], {chess.WHITE: 180.0, chess.BLACK: 180.0}
        eng.set_position(chess.STARTING_FEN, []); eng.new_game(); own = 0
        for nd in g.mainline():
            mover, c = board.turn, nd.clock()
            if c is None:
                break
            if (mover == chess.WHITE) == me_white:
                eng.set_position(chess.STARTING_FEN, moves)
                _, info = eng.choose(prev[mover], prev[not mover] + opp_bonus, 0.0, None, sleep=False)
                if own > 0 and prev[mover] < 60:
                    t, u = info["think"], rng.random()
                    out.append((prev[mover], max(prev[mover] - c, 0.0), max(0.0, math.ceil(t - u))))
                own += 1
            prev[mover] = c; board.push(nd.move); moves.append(nd.move.uci())
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", required=True); ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "clonefish", "engine_replay.json"))
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {}
    for spec in a.players.split(","):
        nm, ftp = spec.split(":")
        pgn = os.path.join(ROOT, "data", "lichess_5k", f"{nm}.pgn.zst")
        gs = games_of(pgn, nm.lower()); gs.sort(key=dkey); gs = gs[:-80][-a.games:]
        eng = CloneEngine(CKPT, ftp, player_data(pgn, nm, 80), device=dev, seed=5)
        eng.opt["UseBook"] = False                              # moves are not played, only thinks sampled
        res[nm] = {}
        for tag, bonus in (("ENGINE", 0.0), ("OPP+60", 60.0)):
            A = replay(eng, gs, nm.lower(), bonus, np.random.default_rng(7))
            res[nm][tag] = {}
            print(f"{nm} {tag}: {len(A)} own moves under 60 s", flush=True)
            for lo, hi, lab in BUCKETS:
                m = (A[:, 0] >= lo) & (A[:, 0] < hi)
                if not m.any(): continue
                r = {"n": int(m.sum()), "instant_real": round(100 * float((A[m, 1] == 0).mean()), 1),
                     "instant_engine": round(100 * float((A[m, 2] == 0).mean()), 1),
                     "mean_real": round(float(A[m, 1].mean()), 2), "mean_engine": round(float(A[m, 2].mean()), 2)}
                res[nm][tag][lab] = r
                print(f"  {lab:<7} n={r['n']:5d}  instant real {r['instant_real']:5.1f} / engine {r['instant_engine']:5.1f}   "
                      f"mean real {r['mean_real']:.2f} / engine {r['mean_engine']:.2f}", flush=True)
        del eng; torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1); print("wrote", a.out)


if __name__ == "__main__":
    main()
