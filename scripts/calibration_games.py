"""Proper calibration check: replay REAL games, restrict to legal moves, and fold in the
TIMING channel. For each player-in-a-game, sweep Elo and find the setting that makes their
actual moves (and, optionally, think-times) most likely -> revealed Elo. Correlate revealed
vs true rating across players. Compares MOVE-ONLY vs MOVE+TIMING.

    PYTHONPATH=. python scripts/calibration_games.py checkpoints/base_300k_best.pt
"""
import argparse, glob, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import chess
from sahformer.download import iter_games_from_zst
from sahformer.encoding import encode_board, encode_move, build_temporal
from sahformer.records import _stack_history, BASE_SECONDS
from sahformer.model.heads import move_to_index
from sahformer.training.loop import load_model

LOG2PI = math.log(2 * math.pi)


def game_plies(game):
    """Per ply: encoded board/history/temporal, mover's true elo, actual move index,
    legal move indices, think-time, and the mover side."""
    white_elo = int(game.headers.get("WhiteElo", 0) or 0)
    black_elo = int(game.headers.get("BlackElo", 0) or 0)
    board = game.board()
    prev = {chess.WHITE: BASE_SECONDS, chess.BLACK: BASE_SECONDS}
    thist = {chess.WHITE: [], chess.BLACK: []}
    plane_hist, node, ply = [], game, 0
    while node.variations:
        node = node.variation(0); move = node.move; mover = board.turn
        ca = node.clock()
        if ca is None:
            board.push(move); continue
        think = max(prev[mover] - ca, 0.0)
        temporal = build_temporal(my_clock=prev[mover], opp_clock=prev[not mover],
                                  own_think_history=thist[mover], ply=ply)
        cur = encode_board(board); hist = _stack_history(plane_hist, cur)
        frm, to, promo = encode_move(board, move)
        legal = [move_to_index(*encode_move(board, m)) for m in board.legal_moves]
        yield dict(board=cur, history=hist, temporal=temporal,
                   elo=(white_elo if mover == chess.WHITE else black_elo),
                   side=("w" if mover == chess.WHITE else "b"),
                   actual=move_to_index(frm, to, promo), legal=legal, think=think)
        prev[mover] = ca; thist[mover] = [think] + thist[mover]
        plane_hist.append(cur); board.push(move); ply += 1


def collect(dirs_specs, per_spec, min_elo=800, max_elo=3200):
    """Pull ~per_spec plies from EACH corpus (band), so ratings span the full range."""
    P, gid = [], 0
    for d, nfiles in dirs_specs:
        files = sorted(glob.glob(os.path.join(d, "*.pgn.zst")))
        step = max(1, len(files) // nfiles)
        got = 0
        for f in files[::step][:nfiles]:
            for g in iter_games_from_zst(f):
                we = int(g.headers.get("WhiteElo", 0) or 0)
                be = int(g.headers.get("BlackElo", 0) or 0)
                if not (min_elo <= we <= max_elo and min_elo <= be <= max_elo):
                    continue
                rows = list(game_plies(g))
                if len(rows) < 8:
                    continue
                for r in rows:
                    r["group"] = f"{gid}{r['side']}"
                P.extend(rows); gid += 1; got += len(rows)
                if got >= per_spec:
                    break
            if got >= per_spec:
                break
    return P


def mdn_logpdf(mdn, t):
    pi = torch.log_softmax(mdn[0], dim=-1)          # [B,K]
    mu = mdn[1]
    sigma = torch.nn.functional.softplus(mdn[2]) + 1e-3
    lt = torch.log(t.clamp(min=0.1)).unsqueeze(1)   # [B,1]
    comp = -lt - torch.log(sigma) - 0.5 * LOG2PI - (lt - mu) ** 2 / (2 * sigma ** 2)
    return torch.logsumexp(pi + comp, dim=-1)       # [B]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--max-plies", type=int, default=7000)
    ap.add_argument("--grid", default="1000:2800:200")
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 4))

    lo, hi, step = (int(x) for x in args.grid.split(":"))
    grid = np.arange(lo, hi + 1, step)
    print(f"loading {args.ckpt} ... ({torch.get_num_threads()} threads)", flush=True)
    model, _ = load_model(args.ckpt)

    # span rating bands: low -> high corpora
    specs = [("data/chesscom_p1", 6), ("data/chesscom_lowmid", 6),
             ("data/chesscom_p2", 6), ("data/chesscom_p3", 6),
             ("data/chesscom_corpus", 10)]
    specs = [(d, n) for d, n in specs if os.path.isdir(d)]
    print("collecting plies from real games...", flush=True)
    P = collect(specs, per_spec=max(400, args.max_plies // len(specs)))
    N = len(P)
    board = torch.from_numpy(np.stack([p["board"] for p in P])).float()
    hist = torch.from_numpy(np.stack([p["history"] for p in P])).float()
    temporal = torch.from_numpy(np.stack([p["temporal"] for p in P])).float()
    think = torch.tensor([p["think"] for p in P]).float()
    actual = [p["actual"] for p in P]
    legal = [p["legal"] for p in P]
    groups = np.array([p["group"] for p in P])
    true_elo = np.array([p["elo"] for p in P])
    print(f"{N} plies from {len(set(groups))} player-games | "
          f"true Elo {true_elo.min()}-{true_elo.max()}\n", flush=True)

    # learn move-space size, then precompute an additive legal mask ONCE (0=legal, -inf=illegal)
    M = model({"board": board[:1], "history": hist[:1], "temporal": temporal[:1],
               "elo_self": torch.full((1,), 1500),
               "elo_opp": torch.full((1,), 1500)})["move_logits"].shape[1]
    legal_mask = torch.full((N, M), float("-inf"))
    for i in range(N):
        legal_mask[i, legal[i]] = 0.0
    act = torch.tensor(actual)

    lpm = np.zeros((N, len(grid)), np.float32)   # move logP (legal-restricted)
    lpt = np.zeros((N, len(grid)), np.float32)   # timing logP
    for gi, E in enumerate(grid):
        for s in range(0, N, args.batch):
            e = slice(s, min(N, s + args.batch)); bsz = e.stop - e.start
            out = model({"board": board[e], "history": hist[e], "temporal": temporal[e],
                         "elo_self": torch.full((bsz,), int(E)),
                         "elo_opp": torch.full((bsz,), int(E))})
            ml = out["move_logits"]
            lse = torch.logsumexp(ml + legal_mask[e], dim=1)          # legal-restricted denom
            lpm[e, gi] = (ml[torch.arange(bsz), act[e]] - lse).numpy()
            lpt[e, gi] = mdn_logpdf(out["mdn"], think[e]).numpy()
        print(f"  swept Elo {E} ({gi+1}/{len(grid)})", flush=True)

    def revealed(logp):
        rows = {}
        for g in set(groups):
            m = groups == g
            tot = logp[m].sum(axis=0)
            rows[g] = (grid[int(tot.argmax())], true_elo[m][0])
        rv = np.array([v[0] for v in rows.values()])
        tr = np.array([v[1] for v in rows.values()])
        return np.corrcoef(tr, rv)[0, 1], rv, tr

    r_move, _, _ = revealed(lpm)
    r_joint, _, _ = revealed(lpm + lpt)
    print("\n=== per-player calibration (aggregated over each player's moves) ===")
    print(f"MOVE ONLY   correlation (true vs revealed Elo): {r_move:.3f}")
    print(f"MOVE+TIMING correlation (true vs revealed Elo): {r_joint:.3f}")
    print(f"timing lift: {r_joint - r_move:+.3f}")


if __name__ == "__main__":
    main()
