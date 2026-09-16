"""Pull a clone away from the norm by its moves against the norm.

For each player:
  REAL   — on their own training games (never the newest 80), after the opening (own move >= 12): how often their move
           is NOT the stock model's first choice ("off-norm %"), and how surprising their moves are to the stock model
           (mean -log p_stock(move), "norm surprise").
  CLONE  — the clone engine plays the stock model at the player's rating; the same two numbers on the clone's moves,
           for several push strengths w (engine move scores = clone + w * (clone - stock)).
The push strength whose clone numbers match the real player's is saved; nothing is trained.

    PYTHONPATH=. python scripts/clonefish_norm_push.py --players VEGETAL:clones/ftval_VEGETAL_bucket.pt,...
"""
import argparse, json, math, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np, torch, torch.nn.functional as F, chess
from finetune_clone import games_of, rows_of, batch_of
from clonefish_uci import CloneEngine, player_data
from sahformer.training.loop import load_model

CKPT = os.path.join(ROOT, "checkpoints", "base_300k_best.pt")
dkey = lambda g: ((g.headers.get("UTCDate") or g.headers.get("Date") or ""),
                  (g.headers.get("UTCTime") or g.headers.get("StartTime") or ""))


@torch.no_grad()
def real_norm_stats(stock, rows, dev):
    top, surp = [], []
    for s in range(0, len(rows), 512):
        b, ch = batch_of(rows, range(s, min(s + 512, len(rows))), dev)
        ml = stock(b)["move_logits"].float().cpu()
        for j, r in enumerate(ch):
            lb = F.log_softmax(ml[j][r[6]], -1); k = r[6].index(r[5])
            top.append(int(lb.argmax()) == k); surp.append(-float(lb[k]))
    return 100 * (1 - float(np.mean(top))), float(np.mean(surp)), len(top)


def play_collect(pl, opp, player_white, tc=180.0, max_plies=300):
    """one game clone vs stock; returns the clone's (norm_top, norm_logp) for own moves >= 12."""
    board, moves, clocks, got = chess.Board(), [], {chess.WHITE: tc, chess.BLACK: tc}, []
    for e in (pl, opp):
        e.set_position(chess.STARTING_FEN, []); e.new_game()
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        is_pl = (board.turn == chess.WHITE) == player_white
        eng = pl if is_pl else opp
        eng.set_position(chess.STARTING_FEN, moves)
        mv, info = eng.choose(clocks[board.turn], clocks[not board.turn], 0.0, None, sleep=False)
        if info.get("resign"):
            break
        if board.ply() >= 2:
            clocks[board.turn] -= info["spend"]
            if clocks[board.turn] <= 0:
                break
        if is_pl and board.ply() // 2 >= 12 and info["norm_top"] is not None:
            got.append((info["norm_top"], info["norm_logp"]))
        board.push(mv); moves.append(mv.uci())
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", required=True, help="NAME:clone[:data_dir],...")
    ap.add_argument("--grid", default="0,50,100,150,200", help="push strengths in /100")
    ap.add_argument("--games", type=int, default=40); ap.add_argument("--real-games", type=int, default=800)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "clonefish", "norm_push.json"))
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    grid = [int(x) for x in a.grid.split(",")]
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    stock, _ = load_model(CKPT); stock.eval().to(dev)
    for spec in a.players.split(","):
        parts = spec.split(":"); nm, clone = parts[0], parts[1]
        ddir = parts[2] if len(parts) > 2 else os.path.join("data", "lichess_5k")
        pgn = os.path.join(ROOT, ddir, f"{nm}.pgn.zst"); t0 = time.time()
        gs = games_of(pgn, nm.lower()); gs.sort(key=dkey)
        rows = [r for g in gs[:-80][-a.real_games:] for r in rows_of(g, nm.lower()) if r[8] >= 12 and r[5] in r[6]]
        r_off, r_surp, r_n = real_norm_stats(stock, rows, dev)
        data = player_data(pgn, nm, 80)
        rj = os.path.join(ROOT, "clones", f"{nm}_resign.json")
        resign = json.load(open(rj)) if os.path.exists(rj) else {"player": None, "field": None}
        pl = CloneEngine(CKPT, clone, data, device=dev, seed=3, resign=resign["player"], norm_model=stock)
        opp = CloneEngine(CKPT, None, None, device=dev, seed=4, resign=resign["field"])
        opp.opt.update({"Elo": data["opp_elo"], "OppElo": data["elo"], "UseBook": False})
        out = {"real": {"off_norm%": round(r_off, 2), "norm_surprise": round(r_surp, 4), "n": r_n}, "clone": {}}
        print(f"\n{nm}: REAL off-norm {r_off:.1f}%  norm surprise {r_surp:.3f}  (n={r_n} own moves >= 12)", flush=True)
        for w in grid:
            pl.opt["NormPush"] = w; got = []
            for gi in range(a.games):
                got += play_collect(pl, opp, player_white=(gi % 2 == 0))
            off = 100 * (1 - float(np.mean([t for t, _ in got]))); surp = -float(np.mean([l for _, l in got]))
            se = 100 * math.sqrt(max(1e-9, off / 100 * (1 - off / 100)) / max(1, len(got)))
            out["clone"][str(w)] = {"off_norm%": round(off, 2), "norm_surprise": round(surp, 4), "n": len(got), "se_off": round(se, 2)}
            print(f"  push w={w / 100:.2f}: CLONE off-norm {off:5.1f}% (+/-{se:.1f})  surprise {surp:.3f}  n={len(got)}   "
                  f"gap off {off - r_off:+.1f}  surprise {surp - r_surp:+.3f}", flush=True)
        score = lambda c: abs(c["off_norm%"] - r_off) / max(1.0, r_off) + abs(c["norm_surprise"] - r_surp) / max(1e-3, r_surp)
        best = min(out["clone"], key=lambda k: score(out["clone"][k]))
        out["best_push"] = int(best)
        print(f"  -> best push for {nm}: w={int(best) / 100:.2f} ({time.time() - t0:.0f}s)", flush=True)
        res[nm] = out; json.dump(res, open(a.out, "w"), indent=1)
        del pl, opp; torch.cuda.empty_cache()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
