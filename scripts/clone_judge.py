"""STAGE 1 of the DPO clone judge (see docs/superpowers/specs/2026-08-21-clone-dpo-judge-localizer-spec.md).

Pure measurement — no training. Answers: does the clone's DISTRIBUTION of games overlap the real
player's, in the recognizer's style space? Plus the human-auditable stats.

    PYTHONPATH=. python scripts/clone_judge.py --name Gerry_Grob \
        --pgn data/lichess_scale/Gerry_Grob.pgn.zst --clone clones/Gerry_Grob.pt \
        --recognizer checkpoints/recognizer_film.pt

Two read-outs:
  (a) recognizer clouds  -> MMD (two-sample distance) + real-vs-clone discriminator AUC (~0.5 = good)
  (b) interpretable stats -> opening variety + think-time shape (real vs clone)

CRITICAL: real games and clone games go through the SAME featurizer (`side_rows` below), so the
recognizer sees a difference of STYLE, never a difference of feature pipeline.
"""
import argparse
import io
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # for sibling script imports
import numpy as np
import torch
import torch.nn.functional as F
import zstandard
import chess
import chess.pgn

from sahformer.encoding import encode_board, build_temporal
from sahformer.records import _stack_history, BASE_SECONDS
from sahformer.training.loop import load_model
from sahformer.clone import load_any_clone as load_clone
from sahformer.play import self_play
from run_scale import GameEncoder, MAX_PLIES


# ---------------------------------------------------------------------------
# ONE featurizer for both real and clone games. Input: a full ply trajectory
# (both sides), a chosen side. Output: the chosen side's own-ply rows, exactly
# as build_cache_scale.extract_games builds them.
# ---------------------------------------------------------------------------
def side_rows(traj, me_white, my_elo, opp_elo):
    """traj = list of (move, mover_is_white, think). Replays the game and emits one row per
    own-ply: (board_encoding, history, temporal, my_elo, opp_elo, think)."""
    board = chess.Board()
    ph, rows, ply = [], [], 0
    prev = {chess.WHITE: BASE_SECONDS, chess.BLACK: BASE_SECONDS}
    thist = {chess.WHITE: [], chess.BLACK: []}
    for mv, mover_white, think in traj:
        mover = chess.WHITE if mover_white else chess.BLACK
        cur = encode_board(board)
        if mover_white == me_white:
            h = _stack_history(ph, cur)
            temporal = build_temporal(my_clock=prev[mover], opp_clock=prev[not mover],
                                      own_think_history=thist[mover], ply=ply)
            rows.append((cur, h, temporal, my_elo, opp_elo, think))
        prev[mover] = max(prev[mover] - think, 0.0)
        thist[mover] = [think] + thist[mover]
        ph.append(cur); board.push(mv); ply += 1
    return rows


def real_games(path, me, max_games, elo_fallback):
    """Parse a player's PGN into per-game trajectories, keep games where `me` played, and split
    each into that side's own-ply rows via side_rows. Returns list[rows], one per game."""
    me = me.lower()
    text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(path, "rb")),
                            encoding="utf-8", errors="ignore")
    out = []
    while len(out) < max_games:
        g = chess.pgn.read_game(text)
        if g is None:
            break
        w = (g.headers.get("White", "") or "").lower(); b = (g.headers.get("Black", "") or "").lower()
        if me not in (w, b):
            continue
        me_white = (me == w)
        we = int(g.headers.get("WhiteElo", 0) or elo_fallback)
        be = int(g.headers.get("BlackElo", 0) or elo_fallback)
        board = g.board(); node = g
        traj = []; prev = {chess.WHITE: BASE_SECONDS, chess.BLACK: BASE_SECONDS}
        while node.variations:
            node = node.variation(0); mv = node.move; mover = board.turn; ca = node.clock()
            think = max(prev[mover] - ca, 0.0) if ca is not None else 0.0
            traj.append((mv, mover == chess.WHITE, think))
            if ca is not None:
                prev[mover] = ca
            board.push(mv)
        rows = side_rows(traj, me_white, we if me_white else be, be if me_white else we)
        if len(rows) >= 8:
            out.append(rows)
    return out


def clone_games(model, adapter, elo, ngames, dev="cpu", max_plies=80, seed_offset=0):
    """Generate clone self-play games; take the WHITE side's own-ply rows of each as one game
    (mirrors a real game = one player's own plies). Same side_rows featurizer as real games."""
    out = []
    for seed in range(seed_offset, seed_offset + ngames):
        traj = []
        for rec in self_play(model, max_plies=max_plies, elo=elo, temperature=1.0, top_p=0.9,
                             seed=seed, adapter=adapter, device=dev):
            traj.append((rec["move"], rec["mover"] == "white", rec["think"]))
        rows = side_rows(traj, True, elo, elo)
        if len(rows) >= 8:
            out.append(rows)
    return out


@torch.no_grad()
def embed_games(games, model, enc, tmean, tstd, dev):
    """Featurize each game's rows through the frozen base -> pooled, then the recognizer -> 256-d."""
    def base_pooled(board, hist, temporal, es, eo):
        tok = model.input_emb(board.float(), hist.float(), es, eo)
        t = model.temporal_enc(temporal)
        film = model.film_gen(t) if model.use_film else None
        e = model.encoder(tok, t=(t if model.use_time_gab else None), film=film)
        return e.mean(1) + model.t_to_d(t)

    embs = []
    for rows in games:
        rows = rows[:MAX_PLIES]
        board = torch.from_numpy(np.stack([r[0] for r in rows])).float().to(dev)
        hist = torch.from_numpy(np.stack([r[1] for r in rows])).float().to(dev)
        temp = torch.from_numpy(np.stack([r[2] for r in rows])).float().to(dev)
        es = torch.tensor([r[3] for r in rows]).to(dev)
        eo = torch.tensor([r[4] for r in rows]).to(dev)
        pooled = base_pooled(board, hist, temp, es, eo).unsqueeze(0)          # [1,L,Dp]
        think = torch.tensor([r[5] for r in rows], device=dev).float()
        logt = ((torch.log1p(think.clamp(min=0)) - tmean) / tstd).unsqueeze(0)  # [1,L]
        mask = torch.ones(1, pooled.shape[1], dtype=torch.bool, device=dev)
        embs.append(enc(pooled, logt, mask).squeeze(0))
    return torch.stack(embs) if embs else torch.zeros(0, enc.pos.shape[-1], device=dev)


# ---------------------------------------------------------------------------
# distribution metrics in recognizer space
# ---------------------------------------------------------------------------
def mmd_rbf(X, Y, bandwidths=(0.5, 1.0, 2.0, 4.0)):
    """Unbiased multi-bandwidth RBF MMD^2. 0 = clouds identical; bigger = more separated."""
    def _k(A, B):
        d2 = torch.cdist(A, B) ** 2
        return sum(torch.exp(-d2 / (2 * bw * bw)) for bw in bandwidths) / len(bandwidths)
    nx, ny = len(X), len(Y)
    kxx = _k(X, X); kyy = _k(Y, Y); kxy = _k(X, Y)
    sxx = (kxx.sum() - kxx.diag().sum()) / (nx * (nx - 1))
    syy = (kyy.sum() - kyy.diag().sum()) / (ny * (ny - 1))
    return float(sxx + syy - 2 * kxy.mean())


def discriminator_auc(X, Y):
    """Leave-one-out 1-NN AUC that real/clone are separable. 0.5 = indistinguishable (good clone),
    1.0 = trivially separable (clone gives itself away). Cheap, hyperparameter-free."""
    Z = torch.cat([X, Y]); lab = torch.cat([torch.zeros(len(X)), torch.ones(len(Y))])
    D = torch.cdist(Z, Z); D.fill_diagonal_(float("inf"))
    nn = D.argmin(1)
    pred_same = (lab[nn] == lab).float()          # 1 if nearest neighbour shares the label
    # AUC-style: fraction of pairs correctly kept apart == mean of "nearest neighbour is same class"
    return float(pred_same.mean())


def stats_dashboard(real, clone):
    def think_of(games):
        return np.array([r[5] for rows in games for r in rows])
    rt, ct = think_of(real), think_of(clone)
    def line(tag, t):
        if len(t) == 0:
            print(f"  {tag}: (none)"); return
        print(f"  {tag}: mean {t.mean():.2f}s  median {np.median(t):.2f}s  std {t.std():.2f}  "
              f"p90 {np.percentile(t,90):.2f}s  snap%(<0.5s) {(t<0.5).mean()*100:.0f}%")
    print("\n=== think-time shape (own moves) ===")
    line("REAL ", rt); line("CLONE", ct)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--clone", default=None, help="adapter-style clone (.pt from clone_fit)")
    ap.add_argument("--ft-model", default=None,
                    help="FULL fine-tuned model (finetune_clone.py --save-ft). Generation uses this "
                         "model; featurization ALWAYS uses the frozen base, so the recognizer space "
                         "is unchanged and real-vs-clone clouds stay comparable.")
    ap.add_argument("--elo", type=int, default=None, help="play/eval Elo (required with --ft-model)")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="shift the self-play seeds. Running the SAME model twice with different "
                         "offsets gives the base-vs-base noise floor for MMD / 1-NN, without which "
                         "those numbers cannot be interpreted (rung 3).")
    ap.add_argument("--recognizer", default="checkpoints/recognizer_film.pt")
    ap.add_argument("--ckpt", default="checkpoints/base_300k_best.pt")
    ap.add_argument("--real-games", type=int, default=120)
    ap.add_argument("--clone-games", type=int, default=120)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(args.ckpt); model.eval().to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    if not args.clone and not args.ft_model:
        raise SystemExit("pass --clone (adapter) or --ft-model (full fine-tune)")
    adapter, gen_model = None, model          # `model` stays the frozen base = the featurizer
    if args.ft_model:
        gen_model, _ = load_model(args.ckpt)
        gen_model.load_state_dict(torch.load(args.ft_model, weights_only=False)["model_state"])
        gen_model.eval().to(dev)
        for p in gen_model.parameters():
            p.requires_grad_(False)
        elo = args.elo or 1500
    else:
        adapter = load_clone(args.clone, device=dev)
        elo = args.elo or int(adapter.meta.get("avg_elo", 1500))

    ck = torch.load(args.recognizer, weights_only=False)
    enc = GameEncoder(ck["dp"], ck["use_time"], d=ck["d_model"], layers=ck["layers"],
                      heads=ck["heads"], L=ck["max_plies"]).to(dev)
    enc.load_state_dict(ck["state_dict"]); enc.eval()
    tmean, tstd = ck["tmean"], ck["tstd"]
    print(f"judge: {args.name} @ elo {elo} | recognizer {args.recognizer} ({ck['loss']})")

    print(f"featurizing {args.real_games} real games ...")
    real = real_games(args.pgn, args.name, args.real_games, elo)
    print(f"generating + featurizing {args.clone_games} clone self-play games ...")
    clone = clone_games(gen_model, adapter, elo, args.clone_games, dev=dev, seed_offset=args.seed_offset)
    print(f"  real games: {len(real)} | clone games: {len(clone)}")

    Xr = embed_games(real, model, enc, tmean, tstd, dev)
    Xc = embed_games(clone, model, enc, tmean, tstd, dev)

    print("\n=== recognizer-space distribution gap ===")
    print(f"  MMD^2 (real vs clone) : {mmd_rbf(Xr.cpu(), Xc.cpu()):.4f}   (0 = identical clouds)")
    print(f"  1-NN same-class rate  : {discriminator_auc(Xr.cpu(), Xc.cpu()):.3f}   "
          f"(0.5 = indistinguishable = GOOD; 1.0 = clone gives itself away)")
    # calibration: real-vs-real (split the real cloud) is the floor MMD should be near
    if len(Xr) >= 20:
        h = len(Xr) // 2
        print(f"  [floor] MMD^2 real-vs-real: {mmd_rbf(Xr[:h].cpu(), Xr[h:].cpu()):.4f}")

    stats_dashboard(real, clone)


if __name__ == "__main__":
    main()
