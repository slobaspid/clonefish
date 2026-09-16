"""Fit a personal 'clone' adapter from someone's games.

    PYTHONPATH=. python scripts/clone_fit.py --pgn my_games.pgn --name your_username

Reads YOUR moves + clock times out of a Lichess/Chess.com PGN export, freezes the
all-around base model, and trains a small deviation adapter (how your moves and your
clock differ from the generic player). Saves clones/<name>.pt — that file *is* your clone.

Export tips:
  - Lichess:   Profile -> ... -> Export games  (tick "Include clock times").
  - Chess.com: Archive / "Download games (PGN)" already includes clocks.
  - Best results on 3+0 blitz (what the base was trained on); feed those games.
"""
import argparse
import io
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess
import chess.pgn
import numpy as np
import torch
import torch.nn.functional as F

from sahformer.encoding import encode_board, encode_move, build_temporal
from sahformer.records import _stack_history, BASE_SECONDS
from sahformer.model.heads import move_to_index, mdn_nll
from sahformer.training.loop import load_model
from sahformer.clone import CloneAdapter, save_clone


def open_pgn(path):
    if path.endswith(".zst"):
        import zstandard
        fh = open(path, "rb")
        return io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh),
                                encoding="utf-8", errors="ignore")
    return open(path, encoding="utf-8", errors="ignore")


def extract(path, me, max_games):
    """Yield per-ply feature rows for the games where `me` played (their own moves)."""
    me = me.lower()
    text = open_pgn(path)
    rows, games, with_clock, seen = [], 0, 0, 0
    while games < max_games:
        g = chess.pgn.read_game(text)
        if g is None:
            break
        seen += 1
        w = (g.headers.get("White", "") or "").lower()
        b = (g.headers.get("Black", "") or "").lower()
        if me == w:
            me_white = True
        elif me == b:
            me_white = False
        else:
            continue
        we = int(g.headers.get("WhiteElo", 0) or 0)
        be = int(g.headers.get("BlackElo", 0) or 0)
        board = g.board()
        prev = {chess.WHITE: BASE_SECONDS, chess.BLACK: BASE_SECONDS}
        thist = {chess.WHITE: [], chess.BLACK: []}
        ph, node, ply = [], g, 0
        got = 0
        while node.variations:
            node = node.variation(0)
            mv = node.move
            mover = board.turn
            ca = node.clock()
            cur = encode_board(board)
            if ca is not None and (mover == chess.WHITE) == me_white:
                think = max(prev[mover] - ca, 0.0)
                h = _stack_history(ph, cur)
                frm, to, pr = encode_move(board, mv)
                rows.append((cur, h,
                             build_temporal(my_clock=prev[mover], opp_clock=prev[not mover],
                                            own_think_history=thist[mover], ply=ply),
                             (we if me_white else be), (be if me_white else we),
                             move_to_index(frm, to, pr), think))
                got += 1
            if ca is not None:
                prev[mover] = ca
                thist[mover] = [max(prev[mover] - ca, 0.0)] + thist[mover]
            ph.append(cur)
            board.push(mv)
            ply += 1
        if got:
            games += 1
            with_clock += 1
    return rows, seen, games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True, help="PGN file (.pgn or .pgn.zst)")
    ap.add_argument("--name", required=True, help="your username exactly as it appears in the PGN")
    ap.add_argument("--ckpt", default="checkpoints/base_300k_best.pt")
    ap.add_argument("--out", default=None, help="output path (default clones/<name>.pt)")
    ap.add_argument("--max-games", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-time", type=float, default=1.0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")
    print(f"reading {args.pgn} for '{args.name}' ...")
    rows, seen, games = extract(args.pgn, args.name, args.max_games)
    if not rows:
        print(f"No moves found for '{args.name}' in {seen} games. "
              f"Check the username matches the PGN's White/Black headers exactly.")
        sys.exit(1)
    N = len(rows)
    print(f"found {N} of your moves across {games} games (scanned {seen}).")
    if N < 500:
        print("  (heads-up: <500 moves is a thin clone — feed more games for a sharper style.)")

    # your typical strength: average of your own Elo across these games -> the clone is
    # trained (and plays) at this level, so the deviation is "you vs a generic player here".
    own_elos = [r[3] for r in rows if r[3] > 0]
    avg_elo = int(round(sum(own_elos) / len(own_elos))) if own_elos else 1500
    print(f"your average Elo in these games: {avg_elo}  (clone trains + plays at this strength)")

    print(f"loading base model {args.ckpt} ...")
    model, mcfg = load_model(args.ckpt)
    model.eval().to(dev)
    for p in model.parameters():
        p.requires_grad_(False)

    # ---- run the frozen base once; cache what the adapter trains on ----
    print("caching base features (frozen) ...")
    cP, cML, cPi, cMu, cSg, cAct, cTh = ([] for _ in range(7))
    B = 512
    with torch.no_grad():
        for s in range(0, N, B):
            batch = rows[s:s + B]
            nb_ = len(batch)
            out = model({
                "board": torch.from_numpy(np.stack([r[0] for r in batch])).float().to(dev),
                "history": torch.from_numpy(np.stack([r[1] for r in batch])).float().to(dev),
                "elo_self": torch.full((nb_,), avg_elo).to(dev),
                "elo_opp": torch.full((nb_,), avg_elo).to(dev),
                "temporal": torch.from_numpy(np.stack([r[2] for r in batch])).float().to(dev),
            })
            pi, mu, sg = out["mdn"]
            cP.append(out["pooled"].cpu()); cML.append(out["move_logits"].cpu())
            cPi.append(pi.cpu()); cMu.append(mu.cpu()); cSg.append(sg.cpu())
            cAct.append(torch.tensor([r[5] for r in batch]))
            cTh.append(torch.tensor([r[6] for r in batch], dtype=torch.float32))
    pooled = torch.cat(cP).to(dev); base_ml = torch.cat(cML).to(dev)
    base_pi = torch.cat(cPi).to(dev); base_mu = torch.cat(cMu).to(dev); base_sg = torch.cat(cSg).to(dev)
    act = torch.cat(cAct).to(dev); think = torch.cat(cTh).to(dev)
    dim = pooled.shape[1]; n_moves = base_ml.shape[1]; mdn_k = base_pi.shape[1]

    # ---- train the deviation adapter (base stays frozen) ----
    adapter = CloneAdapter(dim, n_moves, mdn_k).to(dev)
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    idx = torch.arange(N, device=dev)
    for ep in range(args.epochs):
        perm = idx[torch.randperm(N, device=dev)]
        adapter.train(); tot = nb = 0
        for s in range(0, N, args.batch_size):
            b = perm[s:s + args.batch_size]
            dpi, dmu, dsg = adapter.time_offset(pooled[b])
            move_loss = F.cross_entropy(base_ml[b] + adapter.move_residual(pooled[b]), act[b])
            time_loss = mdn_nll(base_pi[b] + dpi, base_mu[b] + dmu, base_sg[b] + dsg, think[b])
            loss = move_loss + args.lambda_time * time_loss
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"epoch {ep + 1}/{args.epochs}  loss {tot / max(1, nb):.4f}")

    out_path = args.out or os.path.join("clones", f"{args.name}.pt")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_clone(out_path, adapter, {"name": args.name, "avg_elo": avg_elo,
                                   "n_moves_seen": N, "n_games": games, "base_ckpt": args.ckpt})
    print(f"\nsaved your clone -> {out_path}")
    print(f"watch it play:   PYTHONPATH=. python scripts/clone_play.py --clone {out_path} --base")
    print(f"play against it: PYTHONPATH=. python scripts/clone_uci.py --clone {out_path}  (add to a chess GUI)")


if __name__ == "__main__":
    main()
