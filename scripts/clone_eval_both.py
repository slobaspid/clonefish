"""Combined matched-position eval (moves + timing in one encode pass). One compact line:

    NAME  moveΔ  base_top1 clone_top1 | real_med clone_med base_med  clone_gap base_gap  real_snap clone_snap

Used by the 50-player batch. See clone_moves_matched.py / clone_timing_matched.py for the split versions.
"""
import argparse, io, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F, zstandard, chess, chess.pgn
from sahformer.encoding import encode_board, encode_move, build_temporal
from sahformer.records import _stack_history, BASE_SECONDS
from sahformer.model.heads import move_to_index
from sahformer.training.loop import load_model
from sahformer.clone import load_clone, apply_clone
from sahformer.play import _sample_think_time


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
            think = max(prev[mover]-ca, 0.0); cur = encode_board(board)
            if (mover == chess.WHITE) == me_white:
                frm, to, pr = encode_move(board, mv)
                legal = [move_to_index(*encode_move(board, m)) for m in board.legal_moves]
                rows.append((cur, _stack_history(ph, cur),
                             build_temporal(prev[mover], prev[not mover], thist[mover], ply),
                             (we if me_white else be), (be if me_white else we),
                             legal, move_to_index(frm, to, pr), think))
            prev[mover] = ca; thist[mover] = [think]+thist[mover]; ph.append(cur); board.push(mv); ply += 1
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
    if len(rows) < 100:
        print(f"SKIP {args.name} (only {len(rows)} held-out positions)"); return
    model, _ = load_model(args.ckpt); model.eval().to(dev)
    adapter = load_clone(args.clone).to(dev)
    rng = np.random.default_rng(0)
    bh = ch = n = 0; ract, rbase, rclone = [], [], []
    B = 256
    with torch.no_grad():
        for s in range(0, len(rows), B):
            chk = rows[s:s+B]
            batch = {"board": torch.from_numpy(np.stack([r[0] for r in chk])).float().to(dev),
                     "history": torch.from_numpy(np.stack([r[1] for r in chk])).float().to(dev),
                     "elo_self": torch.tensor([r[3] for r in chk]).to(dev),
                     "elo_opp": torch.tensor([r[4] for r in chk]).to(dev),
                     "temporal": torch.from_numpy(np.stack([r[2] for r in chk])).float().to(dev)}
            out = model(batch); cl = apply_clone(out, adapter)
            bl = out["move_logits"].cpu(); clg = cl["move_logits"].cpu()
            for j, r in enumerate(chk):
                legal, actual, think = r[5], r[6], r[7]
                if actual in legal:
                    ai = legal.index(actual)
                    bh += int(bl[j][legal].argmax().item() == ai)
                    ch += int(clg[j][legal].argmax().item() == ai); n += 1
                ract.append(think)
                rbase.append(_sample_think_time(tuple(m[j:j+1] for m in out["mdn"]), rng))
                rclone.append(_sample_think_time(tuple(m[j:j+1] for m in cl["mdn"]), rng))
    n = max(n, 1); ract, rbase, rclone = map(np.asarray, (ract, rbase, rclone))
    rm, cm, bm = np.median(ract), np.median(rclone), np.median(rbase)
    print(f"RESULT {args.name} {(ch-bh)/n*100:+.1f} {bh/n*100:.1f} {ch/n*100:.1f} | "
          f"{rm:.2f} {cm:.2f} {bm:.2f} {cm-rm:+.2f} {bm-rm:+.2f} | "
          f"{(ract<0.5).mean()*100:.0f} {(rclone<0.5).mean()*100:.0f}")


if __name__ == "__main__":
    main()
