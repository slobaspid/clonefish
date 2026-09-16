"""Retrieval-clone SIGNAL TEST: does retrieving a player's own past behavior in similar
positions predict their held-out moves + times better than the base model alone?

    PYTHONPATH=. python scripts/clone_retrieval.py --pgn data/lichess_scale/<user>.pgn.zst --name <user>

Roles of the base model here:
  1. encoder — its `pooled` vector is the position-similarity key for retrieval.
  2. prior   — its move/time prediction is the fallback we blend retrieval into.

For each held-out position we compare held-out log-likelihood of the player's ACTUAL move and
think-time under:  base alone   vs   base + retrieval-from-their-own-reference-games.
If retrieval beats base, the paradigm has legs.
"""
import argparse
import io
import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
import zstandard
import chess
import chess.pgn

from sahformer.encoding import encode_board, encode_move, build_temporal
from sahformer.records import _stack_history, BASE_SECONDS
from sahformer.model.heads import move_to_index
from sahformer.training.loop import load_model


def extract(path, me, max_games):
    me = me.lower()
    text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(path, "rb")),
                            encoding="utf-8", errors="ignore")
    games = []
    while len(games) < max_games:
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
                h = _stack_history(ph, cur)
                frm, to, pr = encode_move(board, mv)
                legal = [move_to_index(*encode_move(board, m)) for m in board.legal_moves]
                rows.append((cur, h, build_temporal(prev[mover], prev[not mover], thist[mover], ply),
                             (we if me_white else be), (be if me_white else we),
                             legal, move_to_index(frm, to, pr), think))
            prev[mover] = ca; thist[mover] = [think] + thist[mover]; ph.append(cur); board.push(mv); ply += 1
        if len(rows) >= 8:
            games.append(rows)
    return games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", default="checkpoints/base_300k_best.pt")
    ap.add_argument("--ref-games", type=int, default=120)
    ap.add_argument("--query-games", type=int, default=40)
    ap.add_argument("--k", type=int, default=25, help="neighbors retrieved per query position")
    ap.add_argument("--alpha", type=float, default=0.5, help="retrieval weight in the blend")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    games = extract(args.pgn, args.name, args.ref_games + args.query_games)
    if len(games) < args.ref_games + 5:
        print(f"only {len(games)} games — need more."); return
    ref_games = games[:args.ref_games]; qry_games = games[args.ref_games:]
    print(f"{args.name}: {len(ref_games)} reference games, {len(qry_games)} query games")

    model, mcfg = load_model(args.ckpt); model.eval().to(dev)
    ck = getattr(mcfg, "mdn_components", 3)

    @torch.no_grad()
    def base_forward(rows):
        """rows -> pooled[n,d], legal-move logprobs list, mdn params, per row."""
        out_pool, out_ml, out_mdn = [], [], []
        B = 256
        for s in range(0, len(rows), B):
            chunk = rows[s:s + B]
            batch = {"board": torch.from_numpy(np.stack([r[0] for r in chunk])).float().to(dev),
                     "history": torch.from_numpy(np.stack([r[1] for r in chunk])).float().to(dev),
                     "elo_self": torch.tensor([r[3] for r in chunk]).to(dev),
                     "elo_opp": torch.tensor([r[4] for r in chunk]).to(dev),
                     "temporal": torch.from_numpy(np.stack([r[2] for r in chunk])).float().to(dev)}
            o = model(batch)
            out_pool.append(F.normalize(o["pooled"], dim=-1).cpu())
            out_ml.append(o["move_logits"].cpu())
            out_mdn.append(torch.stack(o["mdn"], 1).cpu())     # [b,3,k]
        return torch.cat(out_pool), torch.cat(out_ml), torch.cat(out_mdn)

    ref_rows = [r for g in ref_games for r in g]
    qry_rows = [r for g in qry_games for r in g]
    print(f"encoding {len(ref_rows)} ref + {len(qry_rows)} query positions through the base ...")
    ref_pool, _, _ = base_forward(ref_rows)
    qry_pool, qry_ml, qry_mdn = base_forward(qry_rows)
    ref_move = torch.tensor([r[6] for r in ref_rows])
    ref_think = torch.tensor([r[7] for r in ref_rows], dtype=torch.float32)
    ref_pool = ref_pool.to(dev)

    def mdn_logp(params, t):                                   # params [3,k]
        k = params.shape[1]; pi = torch.log_softmax(params[0], -1)
        mu = params[1]; sg = F.softplus(params[2]) + 1e-3
        lt = math.log(max(t, 0.1))
        comp = -lt - torch.log(sg) - 0.5 * math.log(2 * math.pi) - (lt - mu) ** 2 / (2 * sg ** 2)
        return torch.logsumexp(pi + comp, 0).item()

    b_move = r_move = b_time = r_time = n = 0.0
    for i, r in enumerate(qry_rows):
        legal = r[5]; actual = r[6]; think = r[7]
        if actual not in legal:
            continue
        # ---- neighbors ----
        sims = (ref_pool @ qry_pool[i].to(dev))
        nn = torch.topk(sims, min(args.k, len(sims))).indices.cpu()
        nb_moves = ref_move[nn]; nb_think = ref_think[nn]

        # ---- MOVE likelihood: base vs base+retrieval blend ----
        base_lp = torch.log_softmax(qry_ml[i][legal], 0)
        base_p = base_lp.exp()
        retr = torch.zeros(len(legal))                         # retrieval vote over legal moves
        legal_idx = {m: j for j, m in enumerate(legal)}
        for m in nb_moves.tolist():
            if m in legal_idx:
                retr[legal_idx[m]] += 1
        if retr.sum() > 0:
            retr = retr / retr.sum()
            blend = (1 - args.alpha) * base_p + args.alpha * retr
        else:
            blend = base_p
        a = legal.index(actual)
        b_move += math.log(base_p[a].item() + 1e-12)
        r_move += math.log(blend[a].item() + 1e-12)

        # ---- TIME likelihood: base MDN vs base+retrieval (log-normal from neighbors' times) ----
        base_t = mdn_logp(qry_mdn[i], think)
        lt_nb = np.log(np.clip(nb_think.numpy(), 0.1, None))
        mu_r, sg_r = float(lt_nb.mean()), max(float(lt_nb.std()), 0.25)
        ltx = math.log(max(think, 0.1))
        retr_t = -ltx - math.log(sg_r) - 0.5 * math.log(2 * math.pi) - (ltx - mu_r) ** 2 / (2 * sg_r ** 2)
        blend_t = np.logaddexp(math.log(1 - args.alpha) + base_t, math.log(args.alpha) + retr_t)
        b_time += base_t
        r_time += float(blend_t)
        n += 1

    n = max(n, 1)
    print(f"\nscored {int(n)} query positions (actual move legal)\n")
    print("channel   base LL      base+retrieval LL   Δ")
    print(f"move     {b_move/n:8.4f}     {r_move/n:8.4f}        {(r_move-b_move)/n:+.4f}")
    print(f"time     {b_time/n:8.4f}     {r_time/n:8.4f}        {(r_time-b_time)/n:+.4f}")
    print("\n(positive Δ = retrieval from the player's own games beats the base alone)")


if __name__ == "__main__":
    main()
