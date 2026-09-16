"""Quick player-identification test (stylometry baseline, no model training).
Each .pgn.zst file = one player. Build a per-player signature from most of their games,
hold some out, and match each held-out game to the nearest player. Compare signals:
opening repertoire vs timing vs both.

    PYTHONPATH=. python scripts/identify.py --dir data/chesscom_corpus --players 40
"""
import argparse, glob, os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import chess
from sahformer.download import iter_games_from_zst

TIME_BINS = [0, 0.3, 0.7, 1.5, 3, 6, 12, 25, 1e9]   # think-time histogram edges (s)

def game_features(game, me_white):
    """Opening key (first 6 SAN plies) + this player's think-time list + move stats."""
    board = game.board()
    sans, thinks = [], []
    prev = {chess.WHITE: 180.0, chess.BLACK: 180.0}
    caps = 0; myplies = 0; castle = 0
    node = game
    while node.variations:
        node = node.variation(0); mv = node.move; mover = board.turn
        if len(sans) < 6:
            sans.append(board.san(mv))
        is_me = (mover == chess.WHITE) == me_white
        ca = node.clock()
        if is_me:
            myplies += 1
            if board.is_capture(mv): caps += 1
            if board.is_castling(mv): castle += 1
            if ca is not None:
                thinks.append(max(prev[mover] - ca, 0.0))
        if ca is not None: prev[mover] = ca
        board.push(mv)
    opening = "|".join(sans)
    return opening, thinks, (caps, castle, myplies)

def player_games(path, cap=200):
    stem = os.path.splitext(os.path.splitext(os.path.basename(path))[0])[0].lower()
    out = []
    for g in iter_games_from_zst(path):
        w = (g.headers.get("White", "") or "").lower()
        b = (g.headers.get("Black", "") or "").lower()
        if stem == w: me_white = True
        elif stem == b: me_white = False
        else: continue
        feats = game_features(g, me_white)
        if feats[2][2] >= 8:                 # at least 8 of my moves
            out.append(feats)
        if len(out) >= cap: break
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/chesscom_corpus")
    ap.add_argument("--players", type=int, default=40)
    ap.add_argument("--min-games", type=int, default=30)
    ap.add_argument("--test-games", type=int, default=6)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.pgn.zst")))
    players = {}
    for f in files:
        g = player_games(f, cap=args.min_games + args.test_games + 20)
        if len(g) >= args.min_games + args.test_games:
            players[os.path.basename(f)] = g
        if len(players) >= args.players:
            break
    names = list(players)
    N = len(names)
    print(f"{N} players, {sum(len(v) for v in players.values())} games total\n")

    # opening vocabulary
    vocab = {}
    for g in players.values():
        for op, _, _ in g:
            vocab.setdefault(op, len(vocab))
    V = len(vocab)

    def open_vec(op):
        v = np.zeros(V);
        if op in vocab: v[vocab[op]] = 1.0
        return v
    def time_vec(thinks):
        h, _ = np.histogram(thinks, bins=TIME_BINS)
        s = h.sum(); return h / s if s else h
    def norm(x): n = np.linalg.norm(x); return x / n if n else x

    # reference signatures (all but last test-games) + per-player held-out games
    ref_open, ref_time, test_by = {}, {}, {}
    for name, g in players.items():
        ref, tst = g[:-args.test_games], g[-args.test_games:]
        ref_open[name] = norm(np.mean([open_vec(o) for o, _, _ in ref], axis=0))
        ref_time[name] = norm(np.mean([time_vec(t) for _, t, _ in ref], axis=0))
        test_by[name] = [(open_vec(o), time_vec(t)) for o, t, _ in tst]

    def evaluate(R_dict, combine, K):
        R = np.stack([R_dict(n) for n in names])
        tot = cor = 0
        for name, items in test_by.items():
            for s in range(0, len(items) - len(items) % K, K):
                grp = items[s:s + K]
                ov = np.mean([x[0] for x in grp], axis=0)
                tv = np.mean([x[1] for x in grp], axis=0)
                q = norm(combine(ov, tv))
                if names[int((R @ q).argmax())] == name: cor += 1
                tot += 1
        return cor / tot if tot else 0.0

    chance = 1.0 / N
    print(f"random chance = {chance:.3f}  (1 of {N})\n")
    print("games/query   openings    timing      both")
    for K in [1, 2, 3, args.test_games]:
        if K > args.test_games: break
        ao = evaluate(lambda n: ref_open[n], lambda o, t: o, K)
        at = evaluate(lambda n: ref_time[n], lambda o, t: t, K)
        ab = evaluate(lambda n: norm(np.concatenate([ref_open[n], ref_time[n]])),
                      lambda o, t: np.concatenate([norm(o), norm(t)]), K)
        print(f"{K:>9}     {ao:.3f}({ao/chance:.0f}x) {at:.3f}({at/chance:.0f}x) {ab:.3f}({ab/chance:.0f}x)")

if __name__ == "__main__":
    main()
