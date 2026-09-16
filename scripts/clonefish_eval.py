"""End-to-end check of the clonefish ENGINE: does it play like the player (timing + mistakes)?

Arms per player (games simulated in-process with the real engine code, 3+0 clocks, Lichess-style
whole-second %clk recording; opponent = base model at the player's typical opponent rating):
  REAL        the player's newest 80 games (never used for the clone or its book)
  CLONE       clonefish engine: fine-tuned clone + book (book excludes newest 80) + whole-second timing
  CLONE_OLD   same, but exact-float clocks/think history (the old engine's timing inputs)
  BASE        base model at the player's rating, no book
Metrics: timing stats (median, instant-move %, >=10 s %, p90, clock left at own move 20/30/40, think by
game stage, W1 on log think vs REAL); Stockfish mistake profile (ACPL, mistake/blunder rate, blunders by
clock left); results and game length.

    PYTHONPATH=. python scripts/clonefish_eval.py --players VEGETAL:clones/ftval_VEGETAL_bucket.pt,...
"""
import argparse, json, math, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np, torch, chess, chess.pgn, chess.engine
from clonefish_uci import CloneEngine, player_data, load_identity
from finetune_clone import games_of, rows_of
from sahformer.training.loop import load_model

SF = os.path.join(ROOT, "tools", "stockfish", "stockfish", "stockfish-windows-x86-64-avx2.exe")
CKPT = os.path.join(ROOT, "checkpoints", "base_300k_best.pt")
dkey = lambda g: ((g.headers.get("UTCDate") or g.headers.get("Date") or ""),
                  (g.headers.get("UTCTime") or g.headers.get("StartTime") or ""))


def clk(sec):
    s = max(0, int(math.floor(sec))); return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def simulate(pl, opp, player_white, tc=180.0, max_plies=300, start=()):
    board, moves = chess.Board(), []
    clocks = {chess.WHITE: tc, chess.BLACK: tc}
    game = chess.pgn.Game(); node = game
    game.headers.update({"White": "player" if player_white else "opp", "Black": "opp" if player_white else "player",
                         "WhiteElo": str(pl.opt["Elo"] if player_white else opp.opt["Elo"]),
                         "BlackElo": str(opp.opt["Elo"] if player_white else pl.opt["Elo"])})
    for e in (pl, opp):
        e.set_position(chess.STARTING_FEN, []); e.new_game()
    for u in start:                 # force both sides through an opening line so self-play covers many openings
        m = chess.Move.from_uci(u)
        if m not in board.legal_moves:
            break
        node = node.add_variation(m); node.comment = f"[%clk {clk(tc)}]"
        board.push(m); moves.append(u)
    flagged = resigned = None
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        eng = pl if (board.turn == chess.WHITE) == player_white else opp
        eng.set_position(chess.STARTING_FEN, moves)
        mv, info = eng.choose(clocks[board.turn], clocks[not board.turn], 0.0, None, sleep=False)
        if info.get("resign"):
            resigned = board.turn; break
        if board.ply() >= 2:                    # Lichess: each side's first move doesn't run its clock
            clocks[board.turn] -= info["spend"]
            if clocks[board.turn] <= 0:         # ran out of time before completing the move
                flagged = board.turn; break
        node = node.add_variation(mv); node.comment = f"[%clk {clk(clocks[board.turn])}]"
        board.push(mv); moves.append(mv.uci())
    if resigned is not None:
        game.headers["Result"], game.headers["Termination"] = ("0-1" if resigned == chess.WHITE else "1-0"), "Normal"
    elif flagged is not None:
        game.headers["Result"], game.headers["Termination"] = ("0-1" if flagged == chess.WHITE else "1-0"), "Time forfeit"
    else:
        game.headers["Result"] = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "*"
        game.headers["Termination"] = "Normal"
    return game


def own_rows(games, me):
    return [rows_of(g, me) for g in games]


def timing(G, real_think=None):
    th = np.array([r[7] for g in G for r in g], float)
    clock_at = lambda k: float(np.median([r[2][0] * 180 for g in G for r in g if r[8] == k] or [np.nan]))
    band = lambda lo, hi: float(np.median([r[7] for g in G for r in g if lo <= r[8] < hi] or [np.nan]))
    s = {"n": int(len(th)), "median": float(np.median(th)), "mean": round(float(th.mean()), 2),
         "instant%": round(100 * float((th == 0).mean()), 1), "10s+%": round(100 * float((th >= 10).mean()), 1),
         "p90": float(np.percentile(th, 90)),
         "clock@20": clock_at(19), "clock@30": clock_at(29), "clock@40": clock_at(39),
         "under10s_games%": round(100 * float(np.mean([min([r[2][0] * 180 for r in g] or [180]) < 10 for g in G])), 1),
         "med_think_by_move": [band(0, 10), band(10, 20), band(20, 30), band(30, 40), band(40, 999)]}
    clk = np.array([r[2][0] * 180 for g in G for r in g], float)
    own = np.array([r[8] for g in G for r in g])
    for lo, hi, lab in ((0, 5, "<5s"), (5, 10, "5-10s"), (10, 30, "10-30s"), (30, 60, "30-60s")):
        m = (clk >= lo) & (clk < hi) & (own > 0)
        s[f"scramble{lab}"] = ({"share%": round(100 * float(m.mean()), 1), "instant%": round(100 * float((th[m] == 0).mean()), 1),
                                "mean": round(float(th[m].mean()), 2)} if m.any() else None)
    if real_think is not None:
        a, b = np.sort(np.log1p(th)), np.sort(np.log1p(real_think)); q = np.linspace(0, 1, 201)
        s["W1log_vs_real"] = round(float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q)))), 4)
    return s, th


def mistakes(games, me, sf, depth):
    loss, clockb = [], []
    for g in games:
        me_white = (me == (g.headers.get("White", "") or "").lower())
        board, prev_clk = g.board(), {chess.WHITE: 180.0, chess.BLACK: 180.0}
        nodes = list(g.mainline())
        if len(nodes) < 12: continue
        ev = []
        b = g.board()
        for i in range(len(nodes) + 1):
            if b.is_game_over():
                ev.append(-1000 if b.is_checkmate() else 0)
            else:
                sc = sf.analyse(b, chess.engine.Limit(depth=depth))["score"].relative.score(mate_score=1000)
                ev.append(int(max(-1000, min(1000, sc))))
            if i < len(nodes): b.push(nodes[i].move)
        for i, nd in enumerate(nodes):
            mover = chess.WHITE if i % 2 == 0 else chess.BLACK
            if (mover == chess.WHITE) == me_white and i >= 16:          # skip the first 8 own moves (book)
                loss.append(max(0, min(1000, ev[i] + ev[i + 1]))); clockb.append(prev_clk[mover])
            c = nd.clock()
            if c is not None: prev_clk[mover] = c
    L, C = np.array(loss, float), np.array(clockb, float)
    out = {"n": int(len(L)), "ACPL": round(float(L.mean()), 1), "mistake%": round(100 * float(((L >= 100) & (L < 300)).mean()), 1),
           "blunder%": round(100 * float((L >= 300).mean()), 1)}
    for lo, hi, lab in ((60, 999, ">60s"), (30, 60, "30-60s"), (10, 30, "10-30s"), (0, 10, "<10s")):
        m = (C >= lo) & (C < hi)
        out[f"blunder%{lab}"] = [round(100 * float((L[m] >= 300).mean()), 1) if m.any() else None, int(m.sum())]
    return out


def results(games, me):
    sc = []
    for g in games:
        r = g.headers.get("Result", "*"); w = (me == (g.headers.get("White", "") or "").lower())
        if r in ("1-0", "0-1", "1/2-1/2"):
            sc.append(0.5 if r == "1/2-1/2" else float((r == "1-0") == w))
    plies = [len(list(g.mainline_moves())) for g in games]
    how = {"resigned": 0, "checkmated": 0, "lost_on_time": 0, "won_opp_resigned": 0, "won_mate": 0,
           "won_on_time": 0, "draw": 0, "unfinished": 0}
    for g in games:
        r, t = g.headers.get("Result", "*"), g.headers.get("Termination", "")
        if r not in ("1-0", "0-1", "1/2-1/2"):
            how["unfinished"] += 1; continue
        if r == "1/2-1/2":
            how["draw"] += 1; continue
        won = (r == "1-0") == (me == (g.headers.get("White", "") or "").lower())
        b = g.end().board()
        if t == "Time forfeit":
            how["won_on_time" if won else "lost_on_time"] += 1
        elif b.is_checkmate():
            how["won_mate" if won else "checkmated"] += 1
        else:
            how["won_opp_resigned" if won else "resigned"] += 1
    out = {"score%": round(100 * float(np.mean(sc)), 1) if sc else None, "median_plies": float(np.median(plies)),
           "plies_p25_p75": [float(np.percentile(plies, 25)), float(np.percentile(plies, 75))]}
    out.update({f"{k}%": round(100 * v / max(1, len(games)), 1) for k, v in how.items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", required=True, help="NAME:ft_ckpt,...")
    ap.add_argument("--games", type=int, default=60); ap.add_argument("--sf-games", type=int, default=40)
    ap.add_argument("--sf-depth", type=int, default=10)
    ap.add_argument("--resign", action="store_true", help="use clones/<NAME>_resign.json for clone (player) and opponent (field)")
    ap.add_argument("--no-sf", action="store_true", help="skip the Stockfish mistake profile")
    ap.add_argument("--data-dir", default=os.path.join("data", "lichess_5k"), help="folder with <NAME>.pgn.zst")
    ap.add_argument("--norm-push", default="", help="per-player push strength /100 for the CLONE arm, e.g. VEGETAL=50,kpowe52=100")
    ap.add_argument("--identity", default=None,
                    help="learned identity head, '{name}' allowed (e.g. clones/{name}_identity_stock.pt); adds an IDENT arm")
    ap.add_argument("--identity-push", type=int, default=100, help="/100 lambda for the IDENT arm")
    ap.add_argument("--identity-base", default="clone", choices=["clone", "stock"],
                    help="what the IDENT arm plays: the fine-tuned clone, or the frozen base + head alone")
    ap.add_argument("--opp-model", default=None,
                    help="fine-tuned opponent model (e.g. a Lichess field model from finetune_clone --opponents); default = base")
    ap.add_argument("--opp-resign-bias", type=int, default=0,
                    help="/100 logits on the OPPONENT's resign hazard. The field model fires far more in simulation "
                         "than real opponents do (VEGETAL: 51.4 %% of its losses vs their 36.8 %%), ending games early.")
    ap.add_argument("--target-games", type=int, default=1500,
                    help="real games used for the GAME-LEVEL targets (endings, length). An 80-game median moves "
                         "~9 plies on resampling, which flipped the length bar; the think-time reference stays the 80.")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "clonefish", "eval.json"))
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    arms_by_player, big_by_player, res = {}, {}, {}
    pushes = dict(kv.split("=") for kv in a.norm_push.split(",")) if a.norm_push else {}   # e.g. VEGETAL=50,kpowe52=100
    stock = None

    # ---- phase 1: generate all games on the GPU (no Stockfish running at the same time) --------------------
    for spec in a.players.split(","):
        nm, ftp = spec.split(":")
        pgn = os.path.join(ROOT, a.data_dir, f"{nm}.pgn.zst")
        gs = games_of(pgn, nm.lower()); gs.sort(key=dkey)
        arms = {"REAL": (gs[-80:], nm.lower())}
        big_by_player[nm] = gs[-(80 + a.target_games):-80] if a.target_games else []
        data = player_data(pgn, nm, exclude_recent=80)
        rj = os.path.join(ROOT, "clones", f"{nm}_resign.json")
        rmod = json.load(open(rj)) if (a.resign and os.path.exists(rj)) else {"player": None, "field": None}
        print(f"[{nm}] resign models: player {'on' if rmod['player'] else 'off'}, opponent {'on' if rmod['field'] else 'off'}", flush=True)
        opp_path = a.opp_model.format(name=nm) if a.opp_model else None      # "{name}" = per-player opponent model
        opp = CloneEngine(CKPT, opp_path, None, device=dev, seed=1, resign=rmod["field"])
        print(f"[{nm}] opponent model: {opp_path or 'base'}", flush=True)
        opp.opt.update({"Elo": data["opp_elo"], "OppElo": data["elo"], "UseBook": False,
                        "ResignBias": a.opp_resign_bias})
        def mk(model, d, resign=None, norm=None, ident=None, **opt):
            e = CloneEngine(CKPT, model, d, device=dev, seed=2, resign=resign, norm_model=norm, identity=ident)
            e.opt.update(opt); return e
        push = int(pushes.get(nm, 0))
        if push and stock is None:
            stock = load_model(CKPT)[0].eval().to(dev)
        if push:
            print(f"[{nm}] norm push w={push / 100:.2f}", flush=True)
        makers = {"CLONE": lambda: mk(ftp, data, rmod["player"], stock if push else None, NormPush=push),
                  "BASE": lambda: mk(None, dict(data, counts={}), rmod["field"])}
        if a.identity:
            ipath = a.identity.format(name=nm)
            if os.path.exists(ipath):
                ihead = load_identity(ipath, dev)
                imodel = ftp if a.identity_base == "clone" else None      # stock = frozen base + the head alone
                print(f"[{nm}] identity arm: {ipath} on the {a.identity_base} model, lambda "
                      f"{a.identity_push / 100:.2f}", flush=True)
                makers["IDENT"] = lambda: mk(imodel, data, rmod["player"], None, ihead, IdentityPush=a.identity_push)
            else:
                print(f"[{nm}] no identity head at {ipath} — skipping the IDENT arm", flush=True)
        for arm, maker in makers.items():
            t0 = time.time(); eng = maker(); games = []
            n = a.games
            for gi in range(n):
                games.append(simulate(eng, opp, player_white=(gi % 2 == 0)))
            with open(os.path.join(ROOT, "results", "clonefish", f"{nm}_{arm}{a.tag}.pgn"), "w", encoding="utf-8") as f:
                for g in games: print(g, file=f, end="\n\n")
            arms[arm] = (games, "player")
            del eng; torch.cuda.empty_cache()
            print(f"[{nm}] generated {n} {arm} games in {time.time() - t0:.0f}s", flush=True)
        del opp; torch.cuda.empty_cache()
        arms_by_player[nm] = arms

    # ---- phase 2: timing + results (cheap) ---------------------------------------------------------------
    for nm, arms in arms_by_player.items():
        res[nm] = {}
        real_rows = own_rows(arms["REAL"][0], arms["REAL"][1])
        _, real_th = timing(real_rows)
        for arm, (games, me) in arms.items():
            t, _ = timing(own_rows(games, me), None if arm == "REAL" else real_th)
            res[nm][arm] = {"timing": t, "results": results(games, me)}
            print(f"[{nm}] {arm:<9} timing {t}  results {res[nm][arm]['results']}", flush=True)
        if big_by_player[nm]:       # the target the game-level bars should actually be read against
            res[nm]["REAL"]["results_big"] = results(big_by_player[nm], nm.lower())
            print(f"[{nm}] REAL target over {len(big_by_player[nm])} games: "
                  f"{res[nm]['REAL']['results_big']}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)

    # ---- phase 3: Stockfish mistake profile (CPU, one engine thread) -------------------------------------
    if a.no_sf:
        print("wrote", a.out); return
    sf = chess.engine.SimpleEngine.popen_uci(SF); sf.configure({"Threads": 1, "Hash": 64})
    try:
        for nm, arms in arms_by_player.items():
            for arm in ("REAL", "CLONE", "IDENT", "BASE"):
                if arm not in arms:      # IDENT only exists when --identity was passed
                    continue
                games, me = arms[arm]; t0 = time.time()
                m = mistakes(games[:a.sf_games], me, sf, a.sf_depth)
                res[nm][arm]["mistakes"] = m
                print(f"[{nm}] {arm:<9} mistakes {m}  ({time.time() - t0:.0f}s)", flush=True)
                json.dump(res, open(a.out, "w"), indent=1)
    finally:
        sf.quit()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
