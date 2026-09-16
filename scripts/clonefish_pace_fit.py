"""Sweep one knob on the SIMULATED OPPONENT and see which way the player's whole-game stats move.

Round 8 killed the pace idea: a per-game pace multiplier (`PaceSigma`) could match the opponent's flag RATE almost
exactly (28.3 % vs real 28.8 %) while making game length WORSE (59 plies vs 81 at sigma 0, real 99), because real
opponents flag at the end of long grinding games and a pace multiplier makes them flag early. Causality runs
length -> flags, not flags -> length.

What actually ends games early is the opponent RESIGNING: VEGETAL's simulated opponent resigns in 52 % of its losses
where his real opponents resign in 35.8 % (measured over 2,500 games — the held-out 80 give a far noisier 19 %).
`ResignBias` is a logit offset on the field model's hazard, and this script fits it.

Targets are estimated from a LARGE window of the player's real games (rates on 80 games are hopeless: 6 resignations
out of 31 losses), while the think-time reference stays the held-out 80 the bars use.

    PYTHONPATH=. python scripts/clonefish_pace_fit.py --player VEGETAL --clone clones/ftval_VEGETAL_bucket.pt \
        --knob ResignBias --grid 0,-50,-100,-150
"""
import argparse, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np, torch, chess
from finetune_clone import games_of
from clonefish_uci import CloneEngine, player_data
from clonefish_eval import simulate, timing, own_rows, CKPT

dkey = lambda g: ((g.headers.get("UTCDate") or g.headers.get("Date") or ""),
                  (g.headers.get("UTCTime") or g.headers.get("StartTime") or ""))


def opp_profile(games, me):
    """what the OPPONENT does: time trouble, flags, resignations (and resignations CONDITIONAL on losing, which is
    the number that matters — a clone that loses more often also gets resigned-to more often)"""
    at = {20: [], 30: [], 40: []}; under = flag = lost = res = n = 0
    for g in games:
        me_w = me == (g.headers.get("White", "") or "").lower()
        b = g.board(); cl = []
        for nd in g.mainline():
            c = nd.clock()
            if c is not None and (b.turn == chess.WHITE) != me_w:
                cl.append(c)
            b.push(nd.move)
        if not cl:
            continue
        n += 1
        for k in at:
            if len(cl) >= k:
                at[k].append(cl[k - 1])
        if min(cl) < 10:
            under += 1
        r = g.headers.get("Result"); term = (g.headers.get("Termination") or "")
        if r in ("1-0", "0-1") and ((r == "1-0") == me_w):      # the OPPONENT lost this game
            lost += 1
            if "time" in term.lower():
                flag += 1
            elif not b.is_checkmate():
                res += 1
    return {"n": n, "under10%": 100 * under / max(1, n), "opp_flag%": 100 * flag / max(1, n),
            "opp_resign%": 100 * res / max(1, n), "resign_given_loss%": 100 * res / max(1, lost),
            "clock": [float(np.median(at[k])) if at[k] else float("nan") for k in (20, 30, 40)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", required=True); ap.add_argument("--clone", required=True)
    ap.add_argument("--data-dir", default=os.path.join("data", "lichess_5k"))
    ap.add_argument("--opp-model", default="clones/field_{name}_bucket.pt")
    ap.add_argument("--knob", default="ResignBias", choices=["ResignBias", "PaceSigma"])
    ap.add_argument("--grid", default="0,-50,-100,-150")
    ap.add_argument("--games", type=int, default=80, help="simulated games per setting")
    ap.add_argument("--target-games", type=int, default=1000, help="real games used to ESTIMATE the targets")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "clonefish", "oppfit.json"))
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    me = a.player.lower(); t0 = time.time()
    grid = [int(x) for x in a.grid.split(",")]

    pgn = os.path.join(ROOT, a.data_dir, f"{a.player}.pgn.zst")
    gs = games_of(pgn, me); gs.sort(key=dkey)
    held = gs[-80:]
    tgt = opp_profile(gs[-(80 + a.target_games):-80], me)          # rates: big window
    held_t = opp_profile(held, me)                                  # what the bars will compare against
    real_rows = own_rows(held, me); _, real_th = timing(real_rows)
    held_plies = float(np.median([sum(1 for _ in g.mainline()) for g in held]))
    print(f"{a.player}: REAL opponents over {tgt['n']} games — resign {tgt['opp_resign%']:.1f}% of games "
          f"({tgt['resign_given_loss%']:.1f}% of their losses), flag {tgt['opp_flag%']:.1f}%, "
          f"under 10 s {tgt['under10%']:.1f}%, clock {'/'.join(f'{c:.0f}' for c in tgt['clock'])}", flush=True)
    print(f"  held-out 80 (what the bars use): opp resign {held_t['opp_resign%']:.1f}% "
          f"({held_t['resign_given_loss%']:.1f}% of losses), flag {held_t['opp_flag%']:.1f}%, "
          f"median plies {held_plies:.0f}", flush=True)

    data = player_data(pgn, a.player, exclude_recent=80)
    rmj = os.path.join(ROOT, "clones", f"{a.player}_resign.json")
    rmod = json.load(open(rmj)) if os.path.exists(rmj) else {"player": None, "field": None}
    opp_path = a.opp_model.format(name=a.player)
    res = {}
    for v in grid:
        pl = CloneEngine(CKPT, a.clone, data, device=dev, seed=2, resign=rmod["player"])
        op = CloneEngine(CKPT, opp_path if os.path.exists(opp_path) else None, None, device=dev, seed=1,
                         resign=rmod["field"])
        op.opt.update({"Elo": data["opp_elo"], "OppElo": data["elo"], "UseBook": False, a.knob: v})
        games = [simulate(pl, op, player_white=(i % 2 == 0)) for i in range(a.games)]
        prof = opp_profile(games, "player")
        t, _ = timing(own_rows(games, "player"), real_th)
        plies = float(np.median([sum(1 for _ in g.mainline()) for g in games]))
        res[v] = {"opp": prof, "W1": t.get("W1log_vs_real"), "median_plies": plies}
        print(f"  {a.knob} {v:>5}: opp resign {prof['opp_resign%']:5.1f}% ({prof['resign_given_loss%']:5.1f}% of "
              f"losses, real {tgt['resign_given_loss%']:.1f}) flag {prof['opp_flag%']:5.1f}% "
              f"under10s {prof['under10%']:5.1f}% | plies {plies:5.0f} (real {held_plies:.0f}) "
              f"| player W1 {t.get('W1log_vs_real')}", flush=True)
        del pl, op; torch.cuda.empty_cache()
    best = min(grid, key=lambda v: abs(res[v]["median_plies"] - held_plies))
    print(f"\n  closest on GAME LENGTH: {a.knob} {best} -> {res[best]['median_plies']:.0f} plies "
          f"vs real {held_plies:.0f}; W1 {res[best]['W1']} vs {res[grid[0]]['W1']} at {grid[0]} "
          f"(reject if W1 got worse)", flush=True)
    allr = json.load(open(a.out)) if os.path.exists(a.out) else {}
    allr[f"{a.player}:{a.knob}"] = {"real_big": tgt, "real_held": held_t, "real_plies": held_plies,
                                    "grid": res, "best_by_length": best}
    json.dump(allr, open(a.out, "w"), indent=1, default=float)
    print(f"wrote {os.path.relpath(a.out, ROOT)} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
