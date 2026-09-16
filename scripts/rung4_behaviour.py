"""Rung 4 of the clonefish measurement ladder: BEHAVIOUR under the engine's own distribution.

Every other measurement in this project is teacher-forced - the model is only ever asked about
positions the HUMAN reached, so it never has to live with its own mistakes. This script branches real
held-out games at a fixed ply and lets each arm play on, then compares the distribution of move
quality and of think-time against the real continuation.

Arms: REAL | BASE seed A | BASE seed B (the noise FLOOR) | CLONE (a --ft-model fine-tune).
Without the BASE-vs-BASE floor the numbers are uninterpretable, which is the mistake rung 3 made.

Move quality proxy: the base model's own VALUE head (no Stockfish available). Expected score for the
side to move is P(win) + 0.5*P(draw); the drop for a move is
    s_before(mover) - (1 - s_after(opponent to move)).
This is a proxy, so ABSOLUTE blunder rates are not comparable to the literature - but the same
instrument is applied to every arm, so ARM-VS-ARM contrasts are valid.

    PYTHONPATH=. python scripts/rung4_behaviour.py --pgn data/lichess_1k/VEGETAL.pgn.zst \
        --name VEGETAL --ft-model clones/ft5k_VEGETAL.pt --elo 1842
"""
import argparse, importlib.util, io, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, chess, chess.pgn, zstandard

from sahformer.encoding import encode_board, build_temporal
from sahformer.records import _stack_history
from sahformer.training.loop import load_model

_spec = importlib.util.spec_from_file_location(
    "ctri", os.path.join(os.path.dirname(os.path.abspath(__file__)), "clone_triage.py"))
ctri = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ctri)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BLUNDER = 0.10          # value-drop threshold counted as a "mistake" (proxy units, 0..1 score)


def real_games(path, me, n_games):
    """Last n_games of `me`: (moves, me_white, thinks) with thinks aligned to moves (None if no clock)."""
    me = me.lower()
    fh = open(path, "rb")
    text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8",
                            errors="ignore") if path.endswith(".zst") else open(path, encoding="utf-8", errors="ignore")
    out = []
    while True:
        g = chess.pgn.read_game(text)
        if g is None:
            break
        w = (g.headers.get("White", "") or "").lower(); b = (g.headers.get("Black", "") or "").lower()
        if me not in (w, b):
            continue
        mw = (me == w)
        moves, thinks, prev, node = [], [], {chess.WHITE: 180.0, chess.BLACK: 180.0}, g
        board = g.board()
        while node.variations:
            node = node.variation(0); mv = node.move; mover = board.turn; ca = node.clock()
            moves.append(mv)
            thinks.append(max(prev[mover] - ca, 0.0) if ca is not None else None)
            if ca is not None:
                prev[mover] = ca
            board.push(mv)
        out.append((moves, mw, thinks))
    return out[-n_games:]


@torch.no_grad()
def score_positions(model, items, elo):
    """items: list of (cur_planes, hist_planes, ply). Returns expected score for the SIDE TO MOVE."""
    res = []
    for s in range(0, len(items), 256):
        ch = items[s:s + 256]
        batch = {
            "board": torch.from_numpy(np.stack([c[0] for c in ch])).float().to(DEV),
            "history": torch.from_numpy(np.stack([c[1] for c in ch])).float().to(DEV),
            "elo_self": torch.tensor([elo] * len(ch)).to(DEV),
            "elo_opp": torch.tensor([elo] * len(ch)).to(DEV),
            "temporal": torch.from_numpy(np.stack([
                build_temporal(90.0, 90.0, [], c[2]) for c in ch])).float().to(DEV),
        }
        p = torch.softmax(model(batch)["value_logits"].float(), dim=-1).cpu().numpy()
        res.extend((p[:, 2] + 0.5 * p[:, 1]).tolist())          # [loss, draw, win]
    return res


def walk(moves, me_white, thinks, branch, evaluator, elo):
    """Replay a move list; for the TARGET side's moves at or after `branch`, return
    (value_drop, think, clock_left). Value drop uses the shared evaluator."""
    board = chess.Board(); ph = []
    pre, post, meta = [], [], []
    clock = {chess.WHITE: 180.0, chess.BLACK: 180.0}
    for i, mv in enumerate(moves):
        mover = board.turn
        cur = encode_board(board); hist = _stack_history(ph, cur)
        th = thinks[i] if thinks[i] is not None else 0.0
        mine = (mover == chess.WHITE) == me_white
        if mine and i >= branch:
            pre.append((cur, hist, i))
        ph.append(cur); board.push(mv)
        if mine and i >= branch:
            nxt = encode_board(board)
            post.append((nxt, _stack_history(ph, nxt), i + 1))
            meta.append((th, clock[mover]))
        clock[mover] = max(clock[mover] - th, 0.0)
    if not pre:
        return []
    sp = evaluator(pre); sq = evaluator(post)
    return [(sp[j] - (1.0 - sq[j]), meta[j][0], meta[j][1]) for j in range(len(pre))]


def ks(a, b):
    """Two-sample Kolmogorov-Smirnov distance, no scipy dependency."""
    a, b = np.sort(np.asarray(a)), np.sort(np.asarray(b))
    grid = np.concatenate([a, b])
    ca = np.searchsorted(a, grid, side="right") / max(len(a), 1)
    cb = np.searchsorted(b, grid, side="right") / max(len(b), 1)
    return float(np.max(np.abs(ca - cb))) if len(grid) else float("nan")


def summarise(tag, rows, ref=None):
    d = np.array([r[0] for r in rows]); t = np.array([r[1] for r in rows]); c = np.array([r[2] for r in rows])
    line = (f"  {tag:16s} n={len(d):5d}  mean_drop {d.mean():+.4f}  median {np.median(d):+.4f}  "
            f"blunder%(>{BLUNDER:.2f}) {(d > BLUNDER).mean()*100:5.1f}  think_med {np.median(t):5.2f}s")
    if ref is not None:
        line += f"   KS_vs_real {ks(d, np.array([r[0] for r in ref])):.3f}"
    print(line)
    lo = [(0, 10), (10, 30), (30, 60), (60, 200)]
    parts = []
    for a, b in lo:
        m = (c >= a) & (c < b)
        parts.append(f"{a}-{b}s:{(d[m] > BLUNDER).mean()*100:4.1f}%(n={m.sum()})" if m.sum() >= 10 else f"{a}-{b}s: -")
    print(f"                    blunder rate by seconds left | " + "  ".join(parts))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--ft-model", default=None)
    ap.add_argument("--ckpt", default="checkpoints/base_300k_best.pt")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--branch", type=int, default=24, help="branch after this many plies (~own move 12)")
    ap.add_argument("--max-plies", type=int, default=80)
    ap.add_argument("--elo", type=int, default=1500)
    args = ap.parse_args()

    base, _ = load_model(args.ckpt); base.to(DEV).eval()
    evaluator = lambda items: score_positions(base, items, args.elo)   # ONE instrument for all arms

    gms = real_games(args.pgn, args.name, args.games)
    print(f"{args.name}: {len(gms)} held-out games, branching at ply {args.branch}, elo {args.elo}")

    real_rows = []
    for moves, mw, thinks in gms:
        if len(moves) > args.branch:
            real_rows += walk(moves, mw, thinks, args.branch, evaluator, args.elo)

    arms = [("BASE seedA", base, 0), ("BASE seedB", base, 5000)]
    if args.ft_model:
        ft, _ = load_model(args.ckpt)
        ft.load_state_dict(torch.load(args.ft_model, weights_only=False)["model_state"])
        ft.to(DEV).eval()
        arms.append(("CLONE", ft, 0))

    print("\n=== rung 4: behaviour under the engine's own distribution ===")
    real_d = summarise("REAL", real_rows)
    out = {}
    for tag, mdl, off in arms:
        rows = []
        for gi, (moves, mw, thinks) in enumerate(gms):
            if len(moves) <= args.branch:
                continue
            opening = (moves[:args.branch], mw)
            traj, me_white = ctri._seeded_traj(mdl, None, args.elo, DEV, off + gi, opening,
                                               max_plies=args.max_plies)
            mv = [t[0] for t in traj]; th = [t[2] for t in traj]
            rows += walk(mv, me_white, th, args.branch, evaluator, args.elo)
        out[tag] = summarise(tag, rows, ref=real_rows)

    print("\n=== read-out ===")
    if "BASE seedA" in out and "BASE seedB" in out:
        floor = abs(ks(out["BASE seedA"], out["BASE seedB"]))
        print(f"  FLOOR  KS(base seedA, base seedB) = {floor:.3f}  <- differences below this are noise")
        for tag in ("BASE seedA", "CLONE"):
            if tag in out:
                print(f"  {tag:11s} KS vs REAL = {ks(out[tag], real_d):.3f}")
        if "CLONE" in out:
            kb, kc = ks(out['BASE seedA'], real_d), ks(out['CLONE'], real_d)
            verdict = "PASS" if (kb - kc) > floor else "not beyond floor"
            print(f"  clone closer to real than base by {kb - kc:+.3f} KS -> {verdict}")


if __name__ == "__main__":
    main()
