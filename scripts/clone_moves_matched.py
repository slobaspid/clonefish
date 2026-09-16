"""Matched-position MOVE test: does the clone's move residual predict the player's actual
held-out moves better than the base alone? (log-likelihood + top-1 match, base vs clone.)

    PYTHONPATH=. python scripts/clone_moves_matched.py --pgn <player>.pgn.zst --name <user> \
        --clone clones/<user>.pt
"""
import argparse, io, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F, zstandard, chess, chess.pgn
from sahformer.encoding import encode_board, encode_move, build_temporal
from sahformer.records import _stack_history, BASE_SECONDS
from sahformer.model.heads import move_to_index
from sahformer.training.loop import load_model
from sahformer.clone import load_any_clone as load_clone, apply_clone


def extract(path, me, skip, take):
    me = me.lower()
    text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(path, "rb")),
                            encoding="utf-8", errors="ignore")
    games = []
    while len(games) < skip + take:
        g = chess.pgn.read_game(text)
        if g is None: break
        w = (g.headers.get("White","") or "").lower(); b = (g.headers.get("Black","") or "").lower()
        if me == w: me_white = True
        elif me == b: me_white = False
        else: continue
        we = int(g.headers.get("WhiteElo",0) or 0); be = int(g.headers.get("BlackElo",0) or 0)
        board = g.board(); prev = {chess.WHITE: BASE_SECONDS, chess.BLACK: BASE_SECONDS}
        thist = {chess.WHITE: [], chess.BLACK: []}; ph, node, ply, rows = [], g, 0, []
        while node.variations:
            node = node.variation(0); mv = node.move; mover = board.turn; ca = node.clock()
            if ca is None: board.push(mv); continue
            cur = encode_board(board)
            if (mover == chess.WHITE) == me_white:
                frm, to, pr = encode_move(board, mv)
                legal = [move_to_index(*encode_move(board, m)) for m in board.legal_moves]
                rows.append((cur, _stack_history(ph, cur),
                             build_temporal(prev[mover], prev[not mover], thist[mover], ply),
                             (we if me_white else be), (be if me_white else we),
                             legal, move_to_index(frm, to, pr)))
            think = max(prev[mover]-ca, 0.0); prev[mover] = ca
            thist[mover] = [think] + thist[mover]; ph.append(cur); board.push(mv); ply += 1
        if len(rows) >= 8: games.append(rows)
    return [r for g in games[skip:] for r in g]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--clone", required=True); ap.add_argument("--ckpt", default="checkpoints/base_300k_best.pt")
    ap.add_argument("--skip-games", type=int, default=120); ap.add_argument("--take-games", type=int, default=40)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = extract(args.pgn, args.name, args.skip_games, args.take_games)
    print(f"{args.name}: {len(rows)} held-out positions")
    model, _ = load_model(args.ckpt); model.eval().to(dev)
    adapter = load_clone(args.clone).to(dev)

    b_ll = c_ll = b_hit = c_hit = n = 0
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
            bl = out["move_logits"].cpu(); clg = cl["move_logits"].cpu()
            for j, r in enumerate(ch):
                legal, actual = r[5], r[6]
                if actual not in legal: continue
                ai = legal.index(actual)
                bp = torch.log_softmax(bl[j][legal], 0); cp = torch.log_softmax(clg[j][legal], 0)
                b_ll += bp[ai].item(); c_ll += cp[ai].item()
                b_hit += int(bp.argmax().item() == ai); c_hit += int(cp.argmax().item() == ai)
                n += 1
    n = max(n, 1)
    print(f"\n=== move prediction at the player's own held-out positions ({n} moves) ===")
    print(f"  base   log-lik {b_ll/n:7.4f}   top-1 match {b_hit/n*100:5.1f}%")
    print(f"  CLONE  log-lik {c_ll/n:7.4f}   top-1 match {c_hit/n*100:5.1f}%")
    print(f"\n  clone Δ log-lik {(c_ll-b_ll)/n:+.4f}   Δ top-1 {(c_hit-b_hit)/n*100:+.1f}pp   "
          f"({'clone helps' if c_ll>b_ll else 'clone hurts/flat'})")


if __name__ == "__main__":
    main()
