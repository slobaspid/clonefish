"""Does the residual clone reproduce the player's DISTRIBUTION (not just per-move)?

    PYTHONPATH=. python scripts/clone_compare.py --pgn data/lichess_scale/<user>.pgn.zst \
        --name <user> --clone clones/<user>.pt

Compares two distribution-level signatures — the things per-move likelihood can't see:
  1. Opening variety: the player's first move as White vs the clone's (does the clone collapse
     onto one pet line, or spread like the human?).
  2. Think-time shape: mean / spread / percentiles of the player's clock vs the clone's.
Generates a batch of clone self-play games and tallies both, side by side with the real games.
"""
import argparse
import io
import os
import sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import zstandard
import chess
import chess.pgn
from sahformer.training.loop import load_model
from sahformer.clone import load_clone
from sahformer.play import self_play


def player_stats(path, me, max_games):
    me = me.lower()
    text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(path, "rb")),
                            encoding="utf-8", errors="ignore")
    first_white = Counter(); thinks = []; ngames = 0
    while ngames < max_games:
        g = chess.pgn.read_game(text)
        if g is None:
            break
        w = (g.headers.get("White", "") or "").lower(); b = (g.headers.get("Black", "") or "").lower()
        if me not in (w, b):
            continue
        me_white = (me == w)
        board = g.board(); prev = {chess.WHITE: 180.0, chess.BLACK: 180.0}
        node = g; first = True
        while node.variations:
            node = node.variation(0); mv = node.move; mover = board.turn; ca = node.clock()
            if (mover == chess.WHITE) == me_white:
                if first and me_white:
                    first_white[board.san(mv)] += 1
                if ca is not None:
                    thinks.append(max(prev[mover] - ca, 0.0))
                first = False
            if ca is not None:
                prev[mover] = ca
            board.push(mv)
        ngames += 1
    return first_white, np.array(thinks)


def clone_stats(model, adapter, elo, ngames):
    first_white = Counter(); thinks = []
    for seed in range(ngames):
        board = chess.Board(); first = True
        for rec in self_play(model, max_plies=80, elo=elo, temperature=1.0, top_p=0.9,
                             seed=seed, adapter=adapter):
            if rec["mover"] == "white" and first:
                first_white[board.san(rec["move"])] += 1
                first = False
            thinks.append(rec["think"])
            board.push(rec["move"])
    return first_white, np.array(thinks)


def show_dist(name, c, top=6):
    tot = sum(c.values()) or 1
    print(f"  {name}:")
    for mv, n in c.most_common(top):
        print(f"    {mv:6s} {n/tot*100:5.1f}%  {'#'*int(n/tot*40)}")


def tstats(name, t):
    if len(t) == 0:
        print(f"  {name}: (none)"); return
    print(f"  {name}: mean {t.mean():.2f}s  median {np.median(t):.2f}s  "
          f"std {t.std():.2f}  p90 {np.percentile(t,90):.2f}s  snap%(<0.5s) {(t<0.5).mean()*100:.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--clone", required=True)
    ap.add_argument("--ckpt", default="checkpoints/base_300k_best.pt")
    ap.add_argument("--real-games", type=int, default=200)
    ap.add_argument("--clone-games", type=int, default=80)
    args = ap.parse_args()

    model, mcfg = load_model(args.ckpt); model.eval()
    adapter = load_clone(args.clone)
    elo = int(adapter.meta.get("avg_elo", 1500))
    print(f"clone: {args.name} @ elo {elo}\n")

    rf, rt = player_stats(args.pgn, args.name, args.real_games)
    print(f"generating {args.clone_games} clone self-play games ...")
    cf, ct = clone_stats(model, adapter, elo, args.clone_games)

    print("\n=== opening variety (first move as White) ===")
    show_dist("REAL ", rf); show_dist("CLONE", cf)
    ren = -sum((n/sum(rf.values()))*np.log2(n/sum(rf.values())) for n in rf.values()) if rf else 0
    cen = -sum((n/sum(cf.values()))*np.log2(n/sum(cf.values())) for n in cf.values()) if cf else 0
    print(f"  variety (entropy bits):  REAL {ren:.2f}   CLONE {cen:.2f}   "
          f"({'clone less varied' if cen < ren-0.3 else 'clone more varied' if cen > ren+0.3 else 'comparable'})")

    print("\n=== think-time shape ===")
    tstats("REAL ", rt); tstats("CLONE", ct)


if __name__ == "__main__":
    main()
