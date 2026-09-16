"""clonefish: a player's clone as a real UCI chess engine.

    python scripts/clonefish_uci.py --player VEGETAL --pgn data/lichess_5k/VEGETAL.pgn.zst \
        --clone clones/ftval_VEGETAL_bucket.pt

Each move:
  1. the player's fine-tuned model gives move probabilities (temperature + nucleus filter)
  2. blended with the player's own opening book:  q = (n + alpha * p) / (N + alpha)
  3. the move is SAMPLED, a think-time is SAMPLED from the clone's time head, and that time is really spent
     (minus compute time and move overhead; never lets the clock run out)

Timing inputs mirror the Lichess training data exactly: whole-second clock readings, and the clone's own
think history is read from those clock readings (what the model saw in training), not from exact floats.

UCI options: Elo, OppElo, UseBook, BookAlpha (/100), Temperature (/100), TopP (/100), MimicClock,
MoveOverheadMs. A GUI sending UCI_Opponent sets OppElo automatically.
"""
import argparse, json, math, os, pickle, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from collections import Counter, defaultdict
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, chess
from sahformer.encoding import encode_board, encode_move, build_temporal
from sahformer.records import _stack_history, BASE_SECONDS
from sahformer.model.heads import move_to_index
from sahformer.training.loop import load_model
from sahformer.play import _sample_think_time


def out(line=""):
    sys.stdout.write(line + "\n"); sys.stdout.flush()


PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def material(board, color):
    return sum(PIECE_VALUE.get(p.piece_type, 0) * (1 if p.color == color else -1) for p in board.piece_map().values())


class IdentityHead(nn.Module):
    """A player's identity as a small policy-shaped residual over the trunk's 64 square tokens: (B,64,dim)->(B,4352).
    Trained on the player's real moves by scripts/clonefish_identity_head.py; the engine adds lambda*head(enc).
    Measured: lambda 1.0 (maximum likelihood) is the optimum — amplifying past it makes a WORSE model of the person."""

    def __init__(self, dim, hid=32):
        super().__init__()
        self.f = nn.Linear(dim, hid, bias=False)
        self.t = nn.Linear(dim, hid, bias=False)
        self.promo = nn.Linear(dim, 4)
        self.scale = 1.0 / math.sqrt(hid)
        nn.init.normal_(self.t.weight, std=0.002)          # starts as a no-op, but gradients still flow
        nn.init.zeros_(self.promo.weight); nn.init.zeros_(self.promo.bias)

    def forward(self, enc):
        b = enc.shape[0]
        m = torch.einsum("bid,bjd->bij", self.f(enc), self.t(enc)) * self.scale
        return torch.cat([m.reshape(b, 4096), self.promo(enc).reshape(b, 64 * 4)], dim=-1)


def hook_enc(model):
    """The trunk's per-square encodings are not returned by forward() — grab them without touching the model."""
    box = {}
    model.encoder.register_forward_hook(lambda m, i, o: box.__setitem__("enc", o))
    return box


def load_identity(path, device="cpu"):
    d = torch.load(path, map_location="cpu", weights_only=False)
    h = IdentityHead(d["dim"], d["hid"]); h.load_state_dict(d["head"]); h.eval().to(device)
    return h


def resign_features(p_loss, p_win, mat, my_clock, opp_clock, own_move, hist=None, danger=None):
    """Inputs of the resignation model (shared by clonefish_resign_fit.py and the engine).
    p_loss/p_win: value head, side to move. mat: material me - opponent (pawn=1 .. queen=9).
    hist = (previous own turn's p_loss, previous own turn's material, how many own turns in a row already lost)
    adds the v2 features: people resign just AFTER things collapse, not evenly through a lost game."""
    m = float(max(-15, min(15, mat)))
    mc, oc = max(0.0, my_clock), max(0.0, opp_clock)
    x = [p_loss, p_win, p_loss * p_loss, m / 10.0, min(m, 0.0) / 10.0,
         math.log1p(mc), math.log1p(oc), (mc - oc) / 60.0, min(own_move, 80) / 40.0, p_loss * math.log1p(mc)]
    if hist is None:
        return x
    ppl, pmat, streak = hist
    s = min(float(streak), 20.0) / 10.0
    x = x + [max(0.0, p_loss - ppl), min(0.0, m - float(max(-15, min(15, pmat)))) / 10.0, s, p_loss * s]
    return x if danger is None else x + list(danger)


def danger_features(board):
    """Cheap "am I getting mated" signals for the side to move, no search. The value head saturates — it reads
    "down a rook" and "mate in 3" both as p_loss ~ 0.95 — but mobility collapse separates them: at the moment
    OKENITE resigns he has 5.3 legal moves vs 14.4 in his other losing positions."""
    legal = list(board.legal_moves); ksq = board.king(board.turn)
    esc = sum(1 for mv in legal if mv.from_square == ksq)
    if board.is_check():
        opp_ch = 0
    else:
        board.push(chess.Move.null())
        opp_ch = sum(1 for mv in board.legal_moves if board.gives_check(mv))
        board.pop()
    return [float(board.is_check()), min(len(legal), 40) / 10.0, esc / 8.0, min(opp_ch, 10) / 5.0]


LOST = 0.8          # p_loss above this = "this position is lost" (used for the how-long-lost feature)


def resign_history(hist, p_loss, mat):
    """Roll the (prev p_loss, prev material, lost-streak) state forward one own turn."""
    streak = ((hist[2] if hist else 0) + 1) if p_loss >= LOST else 0
    return (p_loss, mat, streak)


def resign_hazard(model, x):
    z = (np.asarray(x, float) - np.asarray(model["mean"])) / np.asarray(model["std"])
    return float(1.0 / (1.0 + math.exp(-(float(np.dot(z, model["coef"])) + model["intercept"]))))


def player_data(pgn, name, exclude_recent=0, cache_dir=os.path.join(ROOT, "clones")):
    """Opening book (position -> move counts) + typical own/opponent Elo, cached next to the clones."""
    cache = os.path.join(cache_dir, f"{name}_book_x{exclude_recent}_v2.pkl")    # v2: + think readings per book position
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    from finetune_clone import games_of
    me = name.lower()
    dkey = lambda g: ((g.headers.get("UTCDate") or g.headers.get("Date") or ""),
                      (g.headers.get("UTCTime") or g.headers.get("StartTime") or ""))
    gs = games_of(pgn, me); gs.sort(key=dkey)
    if exclude_recent: gs = gs[:-exclude_recent]
    counts, times, elos, opps = defaultdict(Counter), defaultdict(Counter), [], []
    for g in gs:
        if g.board().fen() != chess.STARTING_FEN: continue
        me_white = (me == (g.headers.get("White", "") or "").lower())
        board, prev = g.board(), {chess.WHITE: BASE_SECONDS, chess.BLACK: BASE_SECONDS}
        for nd in g.mainline():
            mv, mover, c = nd.move, board.turn, nd.clock()
            if (mover == chess.WHITE) == me_white:
                counts[board.epd()][mv.uci()] += 1
                if c is not None and board.ply() < 40:          # their real think readings in opening positions
                    times[board.epd()][int(round(max(prev[mover] - c, 0.0)))] += 1
            if c is not None: prev[mover] = c
            board.push(mv)
    for g in gs[-200:]:
        me_white = (me == (g.headers.get("White", "") or "").lower())
        we, be = int(g.headers.get("WhiteElo", 0) or 0), int(g.headers.get("BlackElo", 0) or 0)
        if we and be:
            elos.append(we if me_white else be); opps.append(be if me_white else we)
    data = {"name": name, "counts": dict(counts), "games": len(gs),
            "times": {k: dict(v) for k, v in times.items() if sum(v.values()) >= 3},
            "elo": int(np.mean(elos)) if elos else 1500, "opp_elo": int(np.mean(opps)) if opps else 1500}
    os.makedirs(cache_dir, exist_ok=True); pickle.dump(data, open(cache, "wb"))
    return data


class CloneEngine:
    def __init__(self, ckpt, clone, data, device="cpu", seed=None, whole_seconds=True, resign=None, norm_model=None,
                 identity=None):
        self.dev, self.ws = device, whole_seconds   # ws=False = old behaviour: exact float clocks/history
        self.resign = resign                        # fitted resignation model (clonefish_resign_fit.py) or None
        self.norm_model = norm_model                # the stock model = "the norm"; enables NormPush and norm stats
        self.identity = identity                    # learned per-player identity head, or None
        self.model, _ = load_model(ckpt)
        if clone:
            self.model.load_state_dict(torch.load(clone, map_location="cpu", weights_only=False)["model_state"])
        self.model.eval().to(device)
        self.enc_box = hook_enc(self.model) if identity is not None else None
        self.book = data["counts"] if data else {}
        self.book_times = data.get("times", {}) if data else {}    # position -> {think reading (s): count}
        self.name = data["name"] if data else "base"
        self.opt = {"Elo": data["elo"] if data else 1500, "OppElo": data["opp_elo"] if data else 1500,
                    "UseBook": True, "BookAlpha": 200, "Temperature": 100, "TopP": 90,
                    "MimicClock": True, "MoveOverheadMs": 100,
                    "AllowFlag": True,   # human-like: spend the sampled think even if it runs the clock out
                    "Resign": True,      # resign lost positions like the player (needs a fitted resign model)
                    "NormPush": 0,       # /100: move scores = clone + w*(clone - stock); pushes the clone away from the norm
                    "IdentityPush": 0,   # /100: + lambda * learned identity head (100 = the strength the real moves support)
                    "ResignBias": 0,     # /100 logits added to the resign hazard. The FIELD model is calibrated on the
                                         # opponents' OWN games but fires far more in simulation (VEGETAL's simulated
                                         # opponent resigns 52 % of its losses vs real opponents' 35.8 %), and those
                                         # early resignations end games before anyone reaches a time scramble.
                    "PaceSigma": 0}      # /100: per-GAME pace factor. The time head gives the within-game spread, but
                                         # humans also vary BETWEEN games (burn three minutes in one, blitz the next).
                                         # Simulated opponents match the real clock MEDIANS yet reach <10 s in 23 % of
                                         # games vs real opponents' 49 %, and that missing tail is what produces flags.
                                         # (never-flag mode turned every scramble move instant: 73-80% vs real 16-22%)
        self.rng = np.random.default_rng(seed)
        self.start_fen, self.moves = chess.STARTING_FEN, []
        self.new_game()

    # ---- game state ------------------------------------------------------------------------------------
    def new_game(self):
        self.think_hist = []          # own think-times as the clock records them, most recent first
        self.last_go_clock = None     # our clock (s) at our previous 'go', to read our last think off the clock
        self.last_spend = None        # fallback when the GUI sends no clocks
        self.res_hist = None          # (prev p_loss, prev material, lost-streak) for the resignation model
        s = self.opt["PaceSigma"] / 100.0 if hasattr(self, "opt") else 0.0
        self.pace = float(np.exp(s * self.rng.standard_normal())) if s > 0 else 1.0   # this game's pace, drawn once

    def set_position(self, fen, moves):
        same_game = (fen == self.start_fen and len(moves) > len(self.moves) and moves[:len(self.moves)] == self.moves)
        if not same_game:
            self.new_game()
        self.start_fen, self.moves = fen, list(moves)
        self.board = chess.Board(fen); self.plane_hist = []
        for u in moves:
            self.plane_hist.append(encode_board(self.board)); self.board.push(chess.Move.from_uci(u))

    # ---- one move -------------------------------------------------------------------------------------
    @torch.no_grad()
    def choose(self, my_clock=None, opp_clock=None, inc=0.0, movetime=None, sleep=True):
        t0 = time.perf_counter(); board = self.board
        # our previous think, read off the clock the way a Lichess PGN records it (whole seconds)
        rd = math.floor if self.ws else (lambda x: x)
        if self.last_go_clock is not None and my_clock is not None:
            self.think_hist.insert(0, float(max(0, rd(self.last_go_clock) - rd(my_clock) + inc)))
        elif self.last_spend is not None:
            self.think_hist.insert(0, float(round(self.last_spend)) if self.ws else float(self.last_spend))
        mc = float(rd(my_clock)) if my_clock is not None else BASE_SECONDS
        oc = float(rd(opp_clock)) if opp_clock is not None else BASE_SECONDS
        cur = encode_board(board)
        batch = {"board": torch.from_numpy(cur).float()[None].to(self.dev),
                 "history": torch.from_numpy(_stack_history(self.plane_hist, cur)).float()[None].to(self.dev),
                 "elo_self": torch.tensor([int(self.opt["Elo"])]).to(self.dev),
                 "elo_opp": torch.tensor([int(self.opt["OppElo"])]).to(self.dev),
                 "temporal": torch.from_numpy(build_temporal(mc, oc, self.think_hist, board.ply())).float()[None].to(self.dev)}
        o = self.model(batch)

        resign, hazard = False, 0.0
        if self.resign is not None and self.opt["Resign"]:
            pv = F.softmax(o["value_logits"][0].float(), -1).cpu().numpy()      # side to move: loss, draw, win
            pl, mat = float(pv[0]), material(board, board.turn)
            nf = len(self.resign["coef"])                                       # 10 / 14 / 18 = which fit this is
            x = resign_features(pl, float(pv[2]), mat, mc, oc, board.ply() // 2,
                                hist=(self.res_hist or (pl, mat, 0)) if nf > 10 else None,
                                danger=danger_features(board) if nf > 14 else None)
            hazard = resign_hazard(self.resign, x)
            bias = self.opt["ResignBias"] / 100.0        # logit offset: the FIELD model is calibrated on the
            if bias:                                     # opponents' own games, but in simulation it fires far more
                o_ = math.log(max(hazard, 1e-12) / max(1.0 - hazard, 1e-12))   # (VEGETAL's simulated opponent resigns
                hazard = 1.0 / (1.0 + math.exp(-(o_ + bias)))                  # 52 % of its losses vs a real 19 %)
            resign = bool(self.rng.random() < hazard)
            self.res_hist = resign_history(self.res_hist, pl, mat)

        legal = list(board.legal_moves)
        idx = [move_to_index(*encode_move(board, m)) for m in legal]
        lg = o["move_logits"][0].float().cpu()[idx]
        lb = None
        if self.norm_model is not None:                  # the stock model's view of the same position
            lb = F.log_softmax(self.norm_model(batch)["move_logits"][0].float().cpu()[idx], -1)
            w = self.opt["NormPush"] / 100.0
            if w:
                lc = F.log_softmax(lg, -1); lg = lc + w * (lc - lb)
        if self.identity is not None and self.opt["IdentityPush"]:
            lg = lg + (self.opt["IdentityPush"] / 100.0) * self.identity(self.enc_box["enc"].float())[0].cpu()[idx]
        p = F.softmax(lg / max(0.05, self.opt["Temperature"] / 100.0), -1).numpy().astype(np.float64)
        in_book = 0
        if self.opt["UseBook"]:
            c = self.book.get(board.epd())
            if c:
                n = np.array([c.get(m.uci(), 0) for m in legal], dtype=np.float64); in_book = int(n.sum())
                if in_book:
                    a = self.opt["BookAlpha"] / 100.0; p = (n + a * p) / (in_book + a)
        top_p = self.opt["TopP"] / 100.0
        if top_p < 1.0 and len(p) > 1:
            order = np.argsort(p)[::-1]; keep = order[: int(np.searchsorted(np.cumsum(p[order]), top_p)) + 1]
            q = np.zeros_like(p); q[keep] = p[keep]; p = q
        p /= p.sum()
        move = legal[int(self.rng.choice(len(legal), p=p))]

        think = float(_sample_think_time(o["mdn"], self.rng))
        bt = self.book_times.get(board.epd()) if self.opt["UseBook"] else None
        if bt:          # a position they know: use their own think readings there (e.g. instant book moves),
            n_t = sum(bt.values()); a = self.opt["BookAlpha"] / 100.0      # blended with the model like the moves
            if self.rng.random() < n_t / (n_t + a):
                secs = np.array(list(bt.keys()), float); w = np.array(list(bt.values()), float)
                r = float(self.rng.choice(secs, p=w / w.sum()))
                think = max(0.0, r - float(self.rng.random())) if r > 0 else 0.0   # reading r = true think in (r-1, r]
        think *= self.pace          # this game's pace: applied after the book readings too, so a slow game is slow
                                    # throughout and an instant premove (0 s) stays instant
        if movetime is not None:
            spend = min(think, movetime / 1000.0)
        elif my_clock is not None and not self.opt["AllowFlag"]:
            spend = min(think, max(0.0, my_clock - 1.0))      # safe mode: never run the clock out
        else:
            spend = think
        if sleep and self.opt["MimicClock"]:
            time.sleep(max(0.0, spend - (time.perf_counter() - t0) - self.opt["MoveOverheadMs"] / 1000.0))
        self.last_go_clock, self.last_spend = my_clock, spend
        k = legal.index(move)
        return move, {"think": think, "spend": spend, "book": in_book, "prob": float(p[k]),
                      "resign": resign, "resign_hazard": hazard,
                      "norm_top": (bool(int(lb.argmax()) == k) if lb is not None else None),     # = stock model's 1st choice?
                      "norm_logp": (float(lb[k]) if lb is not None else None)}                    # stock model's log-prob


def parse_position(tok):
    fen, i = chess.STARTING_FEN, 0
    if tok and tok[0] == "startpos":
        i = 1
    elif tok and tok[0] == "fen":
        fen = " ".join(tok[1:7]); i = 7
    moves = tok[i + 1:] if i < len(tok) and tok[i] == "moves" else []
    return fen, moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", default=None); ap.add_argument("--pgn", default=None)
    ap.add_argument("--clone", default=None, help="fine-tuned model (finetune_clone --save-ft); omit = base")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints", "base_300k_best.pt"))
    ap.add_argument("--book-exclude-recent", type=int, default=0)
    ap.add_argument("--resign", default=None, help="resign model json (default clones/<player>_resign.json if present)")
    ap.add_argument("--identity", default=None, help="learned identity head (clones/<player>_identity_<base>.pt)")
    ap.add_argument("--clock", default="whole", choices=["whole", "exact"],
                    help="clock precision of the player's training games: whole = Lichess (whole seconds), exact = chess.com (tenths)")
    ap.add_argument("--device", default="cpu"); ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    data = player_data(a.pgn, a.player, a.book_exclude_recent) if a.player and a.pgn else None
    rp = a.resign or (os.path.join(ROOT, "clones", f"{a.player}_resign.json") if a.player else None)
    resign = json.load(open(rp))["player"] if rp and os.path.exists(rp) else None
    ident = load_identity(a.identity, a.device) if a.identity and os.path.exists(a.identity) else None
    eng = CloneEngine(a.ckpt, a.clone, data, device=a.device, resign=resign, whole_seconds=(a.clock == "whole"),
                      identity=ident)
    eng.set_position(chess.STARTING_FEN, [])

    for raw in sys.stdin:
        tok = raw.strip().split()
        if not tok: continue
        cmd = tok[0]
        if cmd == "uci":
            out(f"id name clonefish-{eng.name}"); out("id author clonefish")
            out(f"option name Elo type spin default {eng.opt['Elo']} min 400 max 3200")
            out(f"option name OppElo type spin default {eng.opt['OppElo']} min 400 max 3200")
            out("option name UseBook type check default true")
            out("option name BookAlpha type spin default 200 min 0 max 10000")
            out("option name Temperature type spin default 100 min 10 max 200")
            out("option name TopP type spin default 90 min 50 max 100")
            out("option name MimicClock type check default true")
            out("option name MoveOverheadMs type spin default 100 min 0 max 2000")
            out("option name AllowFlag type check default true")
            out(f"option name Resign type check default {'true' if eng.resign is not None else 'false'}")
            if eng.identity is not None:
                out("option name IdentityPush type spin default 0 min 0 max 300")
            out("option name PaceSigma type spin default 0 min 0 max 100")
            out("option name ResignBias type spin default 0 min -400 max 400")
            out("option name UCI_Opponent type string default none")
            out("uciok")
        elif cmd == "isready":
            out("readyok")
        elif cmd == "ucinewgame":
            eng.set_position(chess.STARTING_FEN, []); eng.new_game()
        elif cmd == "setoption" and "name" in tok:
            vi = tok.index("value") if "value" in tok else len(tok)
            nm, val = " ".join(tok[tok.index("name") + 1:vi]), " ".join(tok[vi + 1:])
            if nm == "UCI_Opponent":          # "<title> <rating> <computer|human> <name>"
                parts = val.split()
                if len(parts) > 1 and parts[1].isdigit(): eng.opt["OppElo"] = int(parts[1])
            elif nm in ("UseBook", "MimicClock", "AllowFlag", "Resign"):
                eng.opt[nm] = val.lower() in ("true", "1", "yes")
            elif nm in eng.opt:
                try: eng.opt[nm] = int(val)
                except ValueError: pass
        elif cmd == "position":
            eng.set_position(*parse_position(tok[1:]))
        elif cmd == "go":
            kv = {k: int(tok[tok.index(k) + 1]) for k in ("wtime", "btime", "winc", "binc", "movetime")
                  if k in tok and tok.index(k) + 1 < len(tok)}
            white = eng.board.turn == chess.WHITE
            my, opp = (kv.get("wtime"), kv.get("btime")) if white else (kv.get("btime"), kv.get("wtime"))
            inc = (kv.get("winc", 0) if white else kv.get("binc", 0)) / 1000.0
            move, info = eng.choose(my / 1000.0 if my is not None else None, opp / 1000.0 if opp is not None else None,
                                    inc, kv.get("movetime"))
            out(f"info string think {info['think']:.2f}s spend {info['spend']:.2f}s book {info['book']} p {info['prob']:.2f} "
                f"resign_hazard {info['resign_hazard']:.3f}")
            if info["resign"]:
                # UCI has no resign command: GUIs / lichess-bot resign on a hopeless score
                # (lichess-bot: resign enabled, score -9000 or lower, moves 1)
                out("info depth 1 score cp -9999"); out("info string resign")
            out(f"bestmove {move.uci()}")
        elif cmd == "quit":
            break


if __name__ == "__main__":
    main()
