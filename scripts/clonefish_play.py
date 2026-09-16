"""Play a 3+0 game against a player's clone in your browser.

    python scripts/clonefish_play.py            # then open http://localhost:5001

The clone picks moves like the player (fine-tuned model + their opening book), takes as long as the player
would (sampled think time, really waited), and can lose on time. The server owns the clocks: real time,
each side's first move is free (Lichess rule), running out loses unless the opponent can't checkmate.
"""
import argparse, io, os, sys, threading, time, random
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import chess, chess.pgn, torch
from flask import Flask, jsonify, request, send_file
from clonefish_uci import CloneEngine, player_data

CKPT = os.path.join(ROOT, "checkpoints", "base_300k_best.pt")
CLONES = {n: os.path.join(ROOT, "clones", f"ftval_{n}_bucket.pt") for n in ("VEGETAL", "kpowe52", "OKENITE")}
TC = 180.0

app = Flask(__name__)
LOCK = threading.RLock()
ENGINES, INFO = {}, {}
G = {"gen": 0, "result": None}


def load_all():
    torch.set_num_threads(2)
    for name, path in CLONES.items():
        data = player_data(os.path.join(ROOT, "data", "lichess_5k", f"{name}.pgn.zst"), name, 0)
        ENGINES[name] = CloneEngine(CKPT, path, data, device="cpu")
        INFO[name] = {"elo": data["elo"], "opp_elo": data["opp_elo"], "games": data["games"]}
        print(f"loaded clone {name}", flush=True)


# ---- clock helpers (call with LOCK held) ------------------------------------------------------------------
def remaining(side):
    c = G["clock"][side]
    if G["result"] is None and side == G["board"].turn and G["moved"][side]:
        c -= time.monotonic() - G["turn_start"]
    return c


def finish(result, reason):
    G["result"], G["reason"], G["thinking"] = result, reason, False


def check_flag():
    if G.get("board") is None or G["result"] is not None:
        return
    side = G["board"].turn
    if remaining(side) <= 0:
        G["clock"][side] = 0.0
        if G["board"].has_insufficient_material(not side):
            finish("1/2-1/2", f"{'White' if side else 'Black'} ran out of time, but the opponent can't checkmate")
        else:
            finish("0-1" if side == chess.WHITE else "1-0", f"{'White' if side else 'Black'} lost on time")


def push(move):
    """charge the mover's clock, apply the move, start the other side's turn."""
    b, side, now = G["board"], G["board"].turn, time.monotonic()
    if G["moved"][side]:
        G["clock"][side] -= now - G["turn_start"]
    G["san"].append(b.san(move)); b.push(move); G["moves"].append(move.uci())
    G["moved"][side] = True; G["turn_start"] = now
    G["clk_after"].append(max(0.0, G["clock"][side]))
    if b.is_game_over(claim_draw=True):
        o = b.outcome(claim_draw=True)
        finish(b.result(claim_draw=True), o.termination.name.replace("_", " ").lower() if o else "game over")


def clone_turn(gen):
    with LOCK:
        if G["gen"] != gen or G["result"] is not None or G["board"].turn != G["clone_color"]:
            return
        eng, side = ENGINES[G["name"]], G["board"].turn
        eng.set_position(chess.STARTING_FEN, list(G["moves"]))
        my, opp, start = remaining(side), G["clock"][not side], G["turn_start"]
        G["thinking"] = True
    move, info = eng.choose(my, opp, 0.0, None, sleep=False)
    think = info["think"]
    while True:                                   # really wait the sampled think time (or until the flag falls)
        with LOCK:
            if G["gen"] != gen or G["result"] is not None:
                return
            check_flag()
            if G["result"] is not None:
                return
            if time.monotonic() - start >= think:
                G["clone_thinks"].append(round(think, 1))
                push(move); G["thinking"] = False
                return
        time.sleep(0.03)


def state():
    with LOCK:
        check_flag()
        if G.get("board") is None:
            return {"active": False, "clones": INFO, "loaded": len(ENGINES) == len(CLONES)}
        b = G["board"]
        human = not G["clone_color"]
        return {"active": True, "loaded": True, "clones": INFO, "name": G["name"], "fen": b.fen(),
                "human_color": "white" if human == chess.WHITE else "black",
                "turn": "white" if b.turn == chess.WHITE else "black",
                "clock_white": max(0.0, remaining(chess.WHITE)), "clock_black": max(0.0, remaining(chess.BLACK)),
                "san": G["san"], "clone_thinks": G["clone_thinks"], "thinking": G["thinking"],
                "result": G["result"], "reason": G.get("reason", ""),
                "last": G["moves"][-1] if G["moves"] else None, "in_check": b.is_check(), "gen": G["gen"]}


@app.get("/")
def index():
    return send_file(os.path.join(ROOT, "scripts", "clonefish_play.html"))


@app.get("/api/state")
def api_state():
    return jsonify(state())


@app.post("/api/new")
def api_new():
    j = request.get_json(force=True)
    name = j.get("name")
    if name not in ENGINES:
        return jsonify({"error": "clone still loading"}), 409
    color = j.get("color", "random")
    human_white = (color == "white") or (color == "random" and random.random() < 0.5)
    with LOCK:
        eng = ENGINES[name]
        eng.set_position(chess.STARTING_FEN, []); eng.new_game()
        eng.opt["OppElo"] = int(j.get("rating") or INFO[name]["opp_elo"])
        G.clear()
        G.update({"gen": random.getrandbits(32), "name": name, "board": chess.Board(), "moves": [], "san": [],
                  "clock": {chess.WHITE: TC, chess.BLACK: TC}, "moved": {chess.WHITE: False, chess.BLACK: False},
                  "turn_start": time.monotonic(), "clone_color": chess.BLACK if human_white else chess.WHITE,
                  "clk_after": [], "clone_thinks": [], "thinking": False, "result": None, "reason": ""})
        gen = G["gen"]
    if not human_white:
        threading.Thread(target=clone_turn, args=(gen,), daemon=True).start()
    return jsonify(state())


@app.post("/api/move")
def api_move():
    uci = request.get_json(force=True).get("uci", "")
    with LOCK:
        check_flag()
        if G.get("board") is None or G["result"] is not None:
            return jsonify(dict(state(), error="game is over"))
        b = G["board"]
        if b.turn == G["clone_color"]:
            return jsonify(dict(state(), error="not your turn"))
        try:
            mv = chess.Move.from_uci(uci)
            if mv not in b.legal_moves and len(uci) == 4:
                mv = chess.Move.from_uci(uci + "q")
        except ValueError:
            return jsonify(dict(state(), error="bad move"))
        if mv not in b.legal_moves:
            return jsonify(dict(state(), error="illegal move"))
        push(mv); gen = G["gen"]
    threading.Thread(target=clone_turn, args=(gen,), daemon=True).start()
    return jsonify(state())


@app.post("/api/resign")
def api_resign():
    with LOCK:
        if G.get("board") is not None and G["result"] is None:
            finish("0-1" if G["clone_color"] == chess.BLACK else "1-0", "you resigned")
    return jsonify(state())


@app.get("/api/pgn")
def api_pgn():
    with LOCK:
        if G.get("board") is None:
            return ""
        game = chess.pgn.Game(); node = game
        human = "You"
        game.headers.update({"Event": "clonefish 3+0", "TimeControl": "180+0",
                             "White": G["name"] if G["clone_color"] == chess.WHITE else human,
                             "Black": G["name"] if G["clone_color"] == chess.BLACK else human,
                             "Result": G["result"] or "*", "Termination": G.get("reason", "")})
        for u, c in zip(G["moves"], G["clk_after"]):
            node = node.add_variation(chess.Move.from_uci(u))
            s = int(c); node.comment = f"[%clk {s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}]"
        return str(game), 200, {"Content-Type": "text/plain; charset=utf-8"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=5001)
    ap.add_argument("--tc", type=float, default=180.0, help="seconds per side (testing only; clones are built for 3+0)")
    a = ap.parse_args(); TC = a.tc
    threading.Thread(target=load_all, daemon=True).start()
    app.run(host="127.0.0.1", port=a.port, threaded=True)
