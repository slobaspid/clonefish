"""Matched-position timing test — removes the self-play position confound.

    PYTHONPATH=. python scripts/clone_timing_matched.py --pgn <player>.pgn.zst --name <user> \
        --clone clones/<user>.pt

Runs the clone (and the bare base, as control) at the PLAYER'S OWN held-out positions and
samples a think-time at each. Compares the distribution to the player's ACTUAL think-times at
those same positions. Since the positions are identical, any gap is the model's tempo error,
not a difference in which positions were reached.
"""
import argparse
import io
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import zstandard
import chess
import chess.pgn
from sahformer.encoding import encode_board, build_temporal
from sahformer.records import _stack_history, BASE_SECONDS
from sahformer.training.loop import load_model
from sahformer.clone import load_clone, apply_clone
from sahformer.play import _sample_think_time


def extract(path, me, skip, take):
    """Return held-out positions (games[skip:skip+take]) as feature rows + actual think."""
    me = me.lower()
    text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(path, "rb")),
                            encoding="utf-8", errors="ignore")
    games, gi = [], 0
    while len(games) < skip + take:
        g = chess.pgn.read_game(text)
        if g is None:
            break
        w = (g.headers.get("White", "") or "").lower(); b = (g.headers.get("Black", "") or "").lower()
        if me == w: me_white = True
        elif me == b: me_white = False
        else: continue
        we = int(g.headers.get("WhiteElo", 0) or 0); be = int(g.headers.get("BlackElo", 0) or 0)
        board = g.board(); prev = {chess.WHITE: BASE_SECONDS, chess.BLACK: BASE_SECONDS}
        thist = {chess.WHITE: [], chess.BLACK: []}; ph, node, ply, rows = [], g, 0, []
        while node.variations:
            node = node.variation(0); mv = node.move; mover = board.turn; ca = node.clock()
            if ca is None: board.push(mv); continue
            think = max(prev[mover] - ca, 0.0); cur = encode_board(board)
            if (mover == chess.WHITE) == me_white:
                rows.append((cur, _stack_history(ph, cur),
                             build_temporal(prev[mover], prev[not mover], thist[mover], ply),
                             (we if me_white else be), (be if me_white else we), think))
            prev[mover] = ca; thist[mover] = [think] + thist[mover]; ph.append(cur); board.push(mv); ply += 1
        if len(rows) >= 8:
            games.append(rows)
    return [r for g in games[skip:] for r in g]


def summ(name, t):
    t = np.asarray(t)
    print(f"  {name:6s} mean {t.mean():5.2f}s  median {np.median(t):5.2f}s  std {t.std():5.2f}  "
          f"p90 {np.percentile(t,90):5.2f}s  snap%(<0.5s) {(t<0.5).mean()*100:4.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--clone", required=True); ap.add_argument("--ckpt", default="checkpoints/base_300k_best.pt")
    ap.add_argument("--skip-games", type=int, default=120); ap.add_argument("--take-games", type=int, default=40)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    rows = extract(args.pgn, args.name, args.skip_games, args.take_games)
    print(f"{args.name}: {len(rows)} held-out positions")
    model, mcfg = load_model(args.ckpt); model.eval().to(dev)
    adapter = load_clone(args.clone).to(dev)
    rng = np.random.default_rng(0)

    actual, base_s, clone_s = [], [], []
    B = 256
    with torch.no_grad():
        for s in range(0, len(rows), B):
            ch = rows[s:s+B]
            batch = {"board": torch.from_numpy(np.stack([r[0] for r in ch])).float().to(dev),
                     "history": torch.from_numpy(np.stack([r[1] for r in ch])).float().to(dev),
                     "elo_self": torch.tensor([r[3] for r in ch]).to(dev),
                     "elo_opp": torch.tensor([r[4] for r in ch]).to(dev),
                     "temporal": torch.from_numpy(np.stack([r[2] for r in ch])).float().to(dev)}
            out = model(batch); cl = apply_clone(out, adapter)
            for j, r in enumerate(ch):
                actual.append(r[5])
                bmdn = tuple(m[j:j+1] for m in out["mdn"])
                cmdn = tuple(m[j:j+1] for m in cl["mdn"])
                base_s.append(_sample_think_time(bmdn, rng))
                clone_s.append(_sample_think_time(cmdn, rng))

    print("\n=== think-time at the PLAYER'S OWN positions (matched) ===")
    summ("REAL", actual)          # what the player actually did
    summ("CLONE", clone_s)        # clone at those same positions
    summ("base", base_s)          # bare base at those same positions, control
    a, c, b = np.median(actual), np.median(clone_s), np.median(base_s)
    print(f"\n  median gap to real:  clone {c-a:+.2f}s   base {b-a:+.2f}s   "
          f"({'clone closer' if abs(c-a) < abs(b-a) else 'base closer'})")


if __name__ == "__main__":
    main()
