"""Your clone as a real, playable UCI chess engine.

    PYTHONPATH=. python scripts/clone_uci.py --clone clones/your_username.pt

Then add that command as a UCI engine in any chess GUI (Cute Chess, Nibbler, Arena,
BanksiaGUI) or point a lichess-bot at it — and play against yourself. The engine plays
your moves AND, if MimicClock is on, spends time on each move the way you do (capped so
it never flags).

UCI options exposed to the GUI:
  Elo         - strength to play at   (default: your average Elo from the games)
  Randomness  - move variety /100     (100 = full human; lower = cleaner/stronger)
  MoveFilterTopP - nucleus filter /100 (lower = only sensible moves)
  MimicClock  - spend human-like time per move (default true)
"""
import argparse
import math
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess
import numpy as np
import torch
import torch.nn.functional as F

from sahformer.encoding import encode_board, encode_move, build_temporal
from sahformer.records import _stack_history, BASE_SECONDS
from sahformer.model.heads import move_to_index
from sahformer.training.loop import load_model
from sahformer.clone import load_clone, apply_clone
from sahformer.play import _sample_think_time


def out(line=""):
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


class Engine:
    def __init__(self, ckpt, clone_path):
        self.model, self.mcfg = load_model(ckpt)
        self.model.eval()
        self.adapter = load_clone(clone_path)
        self.name = self.adapter.meta.get("name", "clone")
        self.opt = {
            "Elo": int(self.adapter.meta.get("avg_elo", 1500)),
            "Randomness": 100,
            "MoveFilterTopP": 90,
            "MimicClock": True,
        }
        self.rng = np.random.default_rng()
        self.new_game()

    def new_game(self):
        self.board = chess.Board()
        self.plane_hist = []
        self.think_hist = []           # our own think-times, most recent first

    def set_position(self, tokens):
        # position [startpos | fen <6 fields>] [moves m1 m2 ...]
        self.new_game()
        i = 0
        if tokens and tokens[0] == "startpos":
            i = 1
        elif tokens and tokens[0] == "fen":
            self.board = chess.Board(" ".join(tokens[1:7]))
            i = 7
        moves = []
        if i < len(tokens) and tokens[i] == "moves":
            moves = tokens[i + 1:]
        for uci in moves:
            self.plane_hist.append(encode_board(self.board))
            self.board.push(chess.Move.from_uci(uci))

    @torch.no_grad()
    def choose(self, wtime=None, btime=None, winc=0, binc=0, movetime=None):
        board = self.board
        mover = board.turn
        my_ms = wtime if mover == chess.WHITE else btime
        opp_ms = btime if mover == chess.WHITE else wtime
        my_clock = my_ms / 1000.0 if my_ms is not None else BASE_SECONDS
        opp_clock = opp_ms / 1000.0 if opp_ms is not None else BASE_SECONDS

        cur = encode_board(board)
        hist = _stack_history(self.plane_hist, cur)
        temporal = build_temporal(my_clock, opp_clock, self.think_hist, board.ply())
        elo = self.opt["Elo"]
        batch = {
            "board": torch.from_numpy(cur).float().unsqueeze(0),
            "history": torch.from_numpy(hist).float().unsqueeze(0),
            "elo_self": torch.tensor([elo]),
            "elo_opp": torch.tensor([elo]),
            "temporal": torch.from_numpy(temporal).float().unsqueeze(0),
        }
        o = apply_clone(self.model(batch), self.adapter)

        # ---- pick a move (your style) ----
        logits = o["move_logits"][0]
        legal = list(board.legal_moves)
        idxs = [move_to_index(*encode_move(board, m)) for m in legal]
        scores = logits[idxs]
        temp = max(0.01, self.opt["Randomness"] / 100.0)
        top_p = self.opt["MoveFilterTopP"] / 100.0
        probs = F.softmax(scores / temp, dim=-1).cpu().numpy()
        probs = probs / probs.sum()
        if top_p < 1.0 and len(probs) > 1:
            order = np.argsort(probs)[::-1]
            csum = np.cumsum(probs[order])
            keep = order[: int(np.searchsorted(csum, top_p)) + 1]
            trunc = np.zeros_like(probs)
            trunc[keep] = probs[keep]
            probs = trunc / trunc.sum()
        move = legal[int(self.rng.choice(len(legal), p=probs))]

        # ---- your clock rhythm ----
        think = _sample_think_time(o["mdn"], self.rng, think_temp=1.0)
        if self.opt["MimicClock"]:
            budget = movetime / 1000.0 if movetime else max(0.0, my_clock - 2.0) * 0.4
            time.sleep(max(0.0, min(think, budget)))

        # record our think so the rhythm carries forward
        self.think_hist = [think] + self.think_hist
        return move


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone", required=True)
    ap.add_argument("--ckpt", default="checkpoints/base_300k_best.pt")
    args = ap.parse_args()
    eng = Engine(args.ckpt, args.clone)

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        tok = line.split()
        cmd = tok[0]
        if cmd == "uci":
            out(f"id name Clone-{eng.name}")
            out("id author clonefish")
            out(f"option name Elo type spin default {eng.opt['Elo']} min 600 max 3200")
            out("option name Randomness type spin default 100 min 10 max 150")
            out("option name MoveFilterTopP type spin default 90 min 50 max 100")
            out("option name MimicClock type check default true")
            out("uciok")
        elif cmd == "isready":
            out("readyok")
        elif cmd == "ucinewgame":
            eng.new_game()
        elif cmd == "setoption":
            # setoption name <id> value <v>
            if "name" in tok and "value" in tok:
                nm = " ".join(tok[tok.index("name") + 1:tok.index("value")])
                val = " ".join(tok[tok.index("value") + 1:])
                if nm == "MimicClock":
                    eng.opt[nm] = val.lower() in ("true", "1", "yes")
                elif nm in eng.opt:
                    try:
                        eng.opt[nm] = int(val)
                    except ValueError:
                        pass
        elif cmd == "position":
            eng.set_position(tok[1:])
        elif cmd == "go":
            kv = {}
            for k in ("wtime", "btime", "winc", "binc", "movetime"):
                if k in tok:
                    kv[k] = int(tok[tok.index(k) + 1])
            move = eng.choose(wtime=kv.get("wtime"), btime=kv.get("btime"),
                              winc=kv.get("winc", 0), binc=kv.get("binc", 0),
                              movetime=kv.get("movetime"))
            out(f"bestmove {move.uci()}")
        elif cmd == "quit":
            break


if __name__ == "__main__":
    main()
