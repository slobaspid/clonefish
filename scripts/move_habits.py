"""How different are a player's move-TYPE habits from what the model expects?

On the player's real held-out positions (games the clone never trained on: [-80:]), for each move family
compare  REAL  = fraction of positions where the played move is in the family
with     MODEL = mean probability mass the model puts on that family (base and fine-tuned clone).
Teacher-forced on real positions, so no self-play confound. Gaps here are exactly what a per-family
logit bias would correct.

  PYTHONPATH=. python scripts/move_habits.py --players VEGETAL:clones/ftval_VEGETAL_bucket.pt,...
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F, chess
from finetune_clone import games_of, rows_of, batch_of
from sahformer.training.loop import load_model

FAMILIES = ["capture", "check", "castle", "queen", "king(non-castle)", "pawn", "knight", "bishop", "rook",
            "retreat", "edge-pawn push", "recapture", "promotion", "into enemy half"]
dkey = lambda g: ((g.headers.get("UTCDate") or g.headers.get("Date") or ""),
                  (g.headers.get("UTCTime") or g.headers.get("StartTime") or ""))


def move_feats(board, mv):
    p = board.piece_at(mv.from_square); pt = p.piece_type if p else 0; white = board.turn
    fr, tr = chess.square_rank(mv.from_square), chess.square_rank(mv.to_square)
    fwd = (tr - fr) if white else (fr - tr)
    last = board.peek() if board.move_stack else None
    cap = board.is_capture(mv)
    return [cap, board.gives_check(mv), board.is_castling(mv), pt == chess.QUEEN,
            pt == chess.KING and not board.is_castling(mv), pt == chess.PAWN, pt == chess.KNIGHT,
            pt == chess.BISHOP, pt == chess.ROOK, pt not in (chess.PAWN, chess.KING) and fwd < 0,
            pt == chess.PAWN and not cap and chess.square_file(mv.from_square) in (0, 1, 6, 7),
            cap and last is not None and mv.to_square == last.to_square, mv.promotion is not None,
            (tr >= 4) if white else (tr <= 3)]


def feats_of(g, me):
    """per own-ply feature matrix [n_legal, n_fam], mirroring rows_of's skip rule and legal-move order."""
    me_white = (me == (g.headers.get("White", "") or "").lower())
    board, node, out = g.board(), g, []
    while node.variations:
        node = node.variation(0); mv = node.move; mover = board.turn
        if node.clock() is None: board.push(mv); continue
        if (mover == chess.WHITE) == me_white:
            out.append(np.array([move_feats(board, m) for m in board.legal_moves], dtype=np.float32))
        board.push(mv)
    return out


@torch.no_grad()
def expected(model, rows, feats, dev):
    exp, real, post = [], [], []
    for s in range(0, len(rows), 256):
        b, ch = batch_of(rows, range(s, min(s + 256, len(rows))), dev)
        ml = model(b)["move_logits"].float().cpu()
        for j, r in enumerate(ch):
            legal, actual = r[6], r[5]
            if actual not in legal: continue
            f = feats[s + j]; p = F.softmax(ml[j][legal], -1).numpy()
            exp.append(p @ f); real.append(f[legal.index(actual)]); post.append(r[8] >= 12)
    return np.array(exp), np.array(real), np.array(post)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", required=True, help="NAME:ft_ckpt,...")
    ap.add_argument("--ckpt", default="checkpoints/base_300k_best.pt")
    ap.add_argument("--out", default="results/night_0913/move_habits.json")
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    base, _ = load_model(a.ckpt); base.eval().to(dev)
    res = {}
    for spec in a.players.split(","):
        nm, ftp = spec.split(":")
        gs = games_of(f"data/lichess_5k/{nm}.pgn.zst", nm.lower()); gs.sort(key=dkey)
        rows, feats = [], []
        for g in gs[-80:]:
            r, f = rows_of(g, nm.lower()), feats_of(g, nm.lower())
            assert len(r) == len(f), (len(r), len(f)); rows += r; feats += f
        eb, real, post = expected(base, rows, feats, dev)
        ft, _ = load_model(a.ckpt)
        ft.load_state_dict(torch.load(ftp, map_location="cpu", weights_only=False)["model_state"]); ft.eval().to(dev)
        ec, _, _ = expected(ft, rows, feats, dev)
        del ft; torch.cuda.empty_cache()
        res[nm] = {}
        for region, m in (("all", np.ones_like(post, bool)), ("post-opening", post)):
            n = int(m.sum()); res[nm][region] = {"n": n}
            print(f"\n{nm} [{region}] n={n}   (pp; +/- = 2 standard errors of REAL)")
            print(f"{'family':<18}{'REAL':>7}{'BASE':>7}{'CLONE':>7}{'real-base':>11}{'real-clone':>11}{'+/-':>6}")
            for k, fam in enumerate(FAMILIES):
                rr, bb, cc = real[m, k].mean() * 100, eb[m, k].mean() * 100, ec[m, k].mean() * 100
                se = 2 * np.sqrt(max(rr / 100 * (1 - rr / 100), 1e-9) / n) * 100
                res[nm][region][fam] = [round(rr, 2), round(bb, 2), round(cc, 2), round(se, 2)]
                print(f"{fam:<18}{rr:>7.1f}{bb:>7.1f}{cc:>7.1f}{rr - bb:>+11.1f}{rr - cc:>+11.1f}{se:>6.1f}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1); print("\nwrote", a.out)


if __name__ == "__main__":
    main()
