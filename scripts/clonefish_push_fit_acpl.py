"""Pick the norm-push strength by matching the player's real MISTAKE profile (not their off-norm rate).

Round-3 lesson: matching "how often the clone leaves the stock model's first choice" picked a push that broke a clone
whose mistakes already matched. What actually matters is average centipawn loss (ACPL).

Per player:
  1. real ACPL / blunder % on their held-out games (the newest 80, never trained on).
  2. the unpushed clone's ACPL from a short self-play run; if it is NOT cleaner than the player by more than --noise
     (Stockfish's own run-to-run drift, ~2 points), keep w = 0 and stop.
  3. otherwise try pushes up to --max-push and keep the one whose ACPL is closest to the player's.

    PYTHONPATH=. python scripts/clonefish_push_fit_acpl.py --players Alexei_Soroka:clones/ftval_Alexei_Soroka_bucket.pt:data/lichess_1k
"""
import argparse, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np, torch, chess, chess.engine
from finetune_clone import games_of
from clonefish_uci import CloneEngine, player_data
from clonefish_eval import simulate, mistakes, SF, CKPT
from sahformer.training.loop import load_model

dkey = lambda g: ((g.headers.get("UTCDate") or g.headers.get("Date") or ""),
                  (g.headers.get("UTCTime") or g.headers.get("StartTime") or ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", required=True, help="NAME:clone[:data_dir],...")
    ap.add_argument("--grid", default="0,50,100,150", help="push strengths /100, ascending")
    ap.add_argument("--games", type=int, default=30); ap.add_argument("--sf-depth", type=int, default=9)
    ap.add_argument("--noise", type=float, default=2.0, help="ACPL gap below this counts as a match -> no push")
    ap.add_argument("--opp-model", default="clones/field_lichess_bucket.pt")
    ap.add_argument("--seed", type=int, default=2, help="engine seed; re-run with another seed to test stability")
    ap.add_argument("--always-fit", action="store_true", help="try every push even if the unpushed clone is not too clean")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "clonefish", "push_fit_acpl.json"))
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    grid = [int(x) for x in a.grid.split(",")]
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    sf = chess.engine.SimpleEngine.popen_uci(SF); sf.configure({"Threads": 1, "Hash": 64})
    stock, _ = load_model(CKPT); stock.eval().to(dev)
    try:
        for spec in a.players.split(","):
            parts = spec.split(":"); nm, clone = parts[0], parts[1]
            ddir = parts[2] if len(parts) > 2 else os.path.join("data", "lichess_5k")
            pgn = os.path.join(ROOT, ddir, f"{nm}.pgn.zst"); t0 = time.time()
            gs = games_of(pgn, nm.lower()); gs.sort(key=dkey)
            real = mistakes(gs[-80:], nm.lower(), sf, a.sf_depth)
            data = player_data(pgn, nm, 80)
            rj = os.path.join(ROOT, "clones", f"{nm}_resign.json")
            rmod = json.load(open(rj)) if os.path.exists(rj) else {"player": None, "field": None}
            opp = CloneEngine(CKPT, a.opp_model, None, device=dev, seed=a.seed + 1000, resign=rmod["field"])
            opp.opt.update({"Elo": data["opp_elo"], "OppElo": data["elo"], "UseBook": False})
            out = {"real": real, "clone": {}}
            print(f"\n{nm}: REAL ACPL {real['ACPL']} blunder {real['blunder%']}%  (held-out games)", flush=True)
            for w in grid:
                eng = CloneEngine(CKPT, clone, data, device=dev, seed=a.seed, resign=rmod["player"],
                                  norm_model=(stock if w else None))
                eng.opt["NormPush"] = w
                games = [simulate(eng, opp, player_white=(i % 2 == 0)) for i in range(a.games)]
                m = mistakes(games, "player", sf, a.sf_depth)
                out["clone"][str(w)] = m
                print(f"  push {w/100:.2f}: ACPL {m['ACPL']:.1f} (gap {m['ACPL'] - real['ACPL']:+.1f})  "
                      f"blunder {m['blunder%']:.1f}% (gap {m['blunder%'] - real['blunder%']:+.1f})  n={m['n']}", flush=True)
                del eng; torch.cuda.empty_cache()
                if w == 0 and not a.always_fit and m["ACPL"] >= real["ACPL"] - a.noise:
                    print(f"  -> unpushed clone is not too clean (gap {m['ACPL'] - real['ACPL']:+.1f}); keep push 0", flush=True)
                    break
            best = min(out["clone"], key=lambda k: abs(out["clone"][k]["ACPL"] - real["ACPL"]))
            out["best_push"] = int(best)
            print(f"  => {nm}: push {int(best)/100:.2f} ({time.time() - t0:.0f}s)", flush=True)
            res[nm] = out; json.dump(res, open(a.out, "w"), indent=1)
            del opp; torch.cuda.empty_cache()
    finally:
        sf.quit()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
