import math
import numpy as np
import chess
import torch
import torch.nn.functional as F
from sahformer.encoding import encode_board, encode_move, build_temporal
from sahformer.records import _stack_history
from sahformer.model.heads import move_to_index
from sahformer.search import SearchConfig, time_to_sims, mcts_move

def _softmax_np(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

def _sample_think_time(mdn, rng, think_temp=1.0):
    """Sample a think-time (seconds) from the MDN mixture-of-log-normals output.
    think_temp scales the randomness: 1.0 = full human spread, lower = calmer (closer
    to the typical time, fewer wild long-thinks), 0 = always the most-likely typical time."""
    pi_logits = mdn[0][0].detach().cpu().numpy()
    mu = mdn[1][0].detach().cpu().numpy()
    sigma_p = mdn[2][0].detach().cpu().numpy()
    if think_temp <= 1e-6:
        k = int(np.argmax(pi_logits))                     # most-likely component
        z = 0.0
    else:
        pi = _softmax_np(pi_logits / think_temp)          # sharpen weights when calmer
        k = int(rng.choice(len(pi), p=pi))
        z = float(rng.standard_normal()) * think_temp     # shrink the spread when calmer
    sigma = float(np.logaddexp(0.0, sigma_p[k])) + 1e-3   # stable softplus
    t = math.exp(mu[k] + sigma * z)
    return float(min(max(t, 0.0), 180.0))

@torch.no_grad()
def self_play(model, max_plies=200, elo=1500, temperature=1.0, top_p=1.0,
              start_clock=180.0, device="cpu", seed=0, think_temp=1.0,
              search=False, search_cfg=None, adapter=None):
    """Model plays both sides. Yields one dict per ply (the move it's about to play,
    its sampled think-time, and both clocks). Ends on game over, ply cap, or a flag.

    If `adapter` (a CloneAdapter) is given, the base's move + clock outputs are deviated
    by it every ply, so the game is played in that person's style and clock rhythm."""
    from sahformer.clone import apply_clone
    rng = np.random.default_rng(seed)
    model.eval()
    board = chess.Board()
    plane_hist = []
    think_hist = {chess.WHITE: [], chess.BLACK: []}
    clock = {chess.WHITE: float(start_clock), chess.BLACK: float(start_clock)}
    ply = 0
    while not board.is_game_over() and ply < max_plies:
        mover = board.turn
        cur = encode_board(board)
        hist = _stack_history(plane_hist, cur)
        temporal = build_temporal(clock[mover], clock[not mover], think_hist[mover], ply)
        batch = {
            "board": torch.from_numpy(cur).float().unsqueeze(0).to(device),
            "history": torch.from_numpy(hist).float().unsqueeze(0).to(device),
            "elo_self": torch.tensor([elo]).to(device),
            "elo_opp": torch.tensor([elo]).to(device),
            "temporal": torch.from_numpy(temporal).float().unsqueeze(0).to(device),
        }
        out = model(batch)
        if adapter is not None:
            out = apply_clone(out, adapter)
        think = _sample_think_time(out["mdn"], rng, think_temp=think_temp)
        if search:
            scfg = search_cfg or SearchConfig(elo=elo, temperature=temperature)
            sims = time_to_sims(think, scfg)
            move, _ = mcts_move(model, board, plane_hist, sims, scfg,
                                my_clock=clock[mover], opp_clock=clock[not mover],
                                ply=ply, device=device, seed=int(rng.integers(1 << 30)))
        else:
            logits = out["move_logits"][0]
            legal = list(board.legal_moves)
            idxs = [move_to_index(*encode_move(board, m)) for m in legal]
            scores = logits[idxs]
            if temperature <= 1e-6:                      # temp 0 = always the top move
                move = legal[int(scores.argmax().item())]
            else:
                probs = F.softmax(scores / temperature, dim=-1).detach().cpu().numpy()
                probs = probs / probs.sum()
                if top_p < 1.0 and len(probs) > 1:
                    order = np.argsort(probs)[::-1]                 # best first
                    csum = np.cumsum(probs[order])
                    keep = order[: int(np.searchsorted(csum, top_p)) + 1]  # nucleus
                    trunc = np.zeros_like(probs)
                    trunc[keep] = probs[keep]
                    probs = trunc / trunc.sum()                     # drop the absurd tail
                move = legal[int(rng.choice(len(legal), p=probs))]

        clock[mover] -= think
        flagged = clock[mover] <= 0.0
        yield {
            "ply": ply, "mover": "white" if mover == chess.WHITE else "black",
            "move": move, "san": board.san(move), "think": think,
            "white_clock": clock[chess.WHITE], "black_clock": clock[chess.BLACK],
            "fen": board.fen(), "flagged": flagged,
        }
        if flagged:
            break
        think_hist[mover] = [think] + think_hist[mover]
        plane_hist.append(cur)
        board.push(move)
        ply += 1

def selfplay_frames(model, max_plies=120, elo=1500, temperature=1.0,
                    start_clock=180.0, seed=0, size=400, think_temp=1.0,
                    search=False, search_cfg=None):
    """Play one self-play game and return (svg_frames, captions) for a viewer.
    frames[0] is the start position; each later frame is the board after a move."""
    import chess.svg
    board = chess.Board()
    frames = [chess.svg.board(board, size=size)]
    caps = [{"san": "", "think": 0.0, "white": float(start_clock),
             "black": float(start_clock), "mover": ""}]
    for rec in self_play(model, max_plies=max_plies, elo=elo, temperature=temperature,
                         start_clock=start_clock, device="cpu", seed=seed,
                         think_temp=think_temp, search=search, search_cfg=search_cfg):
        board.push(rec["move"])
        frames.append(chess.svg.board(board, size=size, lastmove=rec["move"]))
        caps.append({"san": rec["san"], "think": rec["think"],
                     "white": rec["white_clock"], "black": rec["black_clock"],
                     "mover": rec["mover"]})
    return frames, caps
