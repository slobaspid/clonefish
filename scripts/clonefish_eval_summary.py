"""Compact real-vs-clone-vs-base table + pre-registered pass bars for a clonefish_eval.py JSON.

    python scripts/clonefish_eval_summary.py results/clonefish/eval5_field.json
"""
import json, sys


def g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def main(path):
    res = json.load(open(path))
    tally = {}
    for nm, arms in res.items():
        print(f"\n{nm}")
        print(f"  {'arm':<6}{'resign':>7}{'mated':>7}{'time':>6} | {'won:res':>7}{'mate':>6}{'time':>6}{'draw':>6} | {'plies':>5} | "
              f"{'inst':>5}{'10s+':>5}{'c@30':>5}{'W1':>7} | {'5-10 inst':>9}{'<5 inst':>8} | {'ACPL':>5}{'blun':>5}")
        for arm in ("REAL", "CLONE", "IDENT", "BASE"):
            if arm not in arms: continue
            t, r, m = arms[arm]["timing"], arms[arm]["results"], arms[arm].get("mistakes", {})
            print(f"  {arm:<6}{r['resigned%']:>7}{r['checkmated%']:>7}{r['lost_on_time%']:>6} | {r['won_opp_resigned%']:>7}"
                  f"{r['won_mate%']:>6}{r['won_on_time%']:>6}{r['draw%']:>6} | {r['median_plies']:>5.0f} | "
                  f"{t['instant%']:>5}{t['10s+%']:>5}{t['clock@30']:>5.0f}{str(t.get('W1log_vs_real', '-')):>7} | "
                  f"{str(g(t, 'scramble5-10s', 'instant%')):>9}{str(g(t, 'scramble<5s', 'instant%')):>8} | "
                  f"{str(m.get('ACPL', '-')):>5}{str(m.get('blunder%', '-')):>5}")
        # diagnostic (not a pre-registered bar): endings CONDITIONAL on the result, which removes the opponent-strength
        # confound (a clone that loses more often to the simulated opponent also resigns more often overall)
        for arm in ("REAL", "CLONE", "IDENT"):
            if arm not in arms: continue
            r = arms[arm]["results"]
            lost = r["resigned%"] + r["checkmated%"] + r["lost_on_time%"]
            won = r["won_opp_resigned%"] + r["won_mate%"] + r["won_on_time%"]
            sh = lambda x, tot: f"{100 * x / tot:4.0f}" if tot else "   -"
            print(f"  {arm:<6}lost {lost:4.1f}% of games -> resign {sh(r['resigned%'], lost)}% mate {sh(r['checkmated%'], lost)}% "
                  f"flag {sh(r['lost_on_time%'], lost)}%  |  won {won:4.1f}% -> opp resign {sh(r['won_opp_resigned%'], won)}% "
                  f"mate {sh(r['won_mate%'], won)}% opp flag {sh(r['won_on_time%'], won)}%")
        R, B = arms["REAL"], arms.get("BASE")

        def bars_for(C):
            """the pre-registered bars, applied to whichever arm is being judged against the player"""
            # game-level targets from the player's whole history when the run recorded them: an 80-game median
            # moves ~9 plies on resampling, which alone flipped the length bar from 1/3 to 3/3
            rr = R.get("results_big") or R["results"]
            cr = C["results"]; rt, ct = R["timing"], C["timing"]
            b = {
                "R2 resign": abs(cr["resigned%"] - rr["resigned%"]) <= 8 and (rr["resigned%"] > 3 or cr["resigned%"] <= 3),
                "R3 lost on time": abs(cr["lost_on_time%"] - rr["lost_on_time%"]) <= 7,
                "R3 length": abs(cr["median_plies"] - rr["median_plies"]) <= 12,
                "G2 timing": (abs(ct["instant%"] - rt["instant%"]) <= 2 and abs(ct["10s+%"] - rt["10s+%"]) <= 2
                              and abs(ct["clock@30"] - rt["clock@30"]) <= 15
                              and (B is None or ct["W1log_vs_real"] < B["timing"]["W1log_vs_real"])),
            }
            if "mistakes" in C and "mistakes" in R and B and "mistakes" in B:
                rm, cm, bm = R["mistakes"], C["mistakes"], B["mistakes"]
                # pre-registered wording is "closer to REAL than BASE": a tie does not pass
                b["G3 mistakes"] = (abs(cm["blunder%"] - rm["blunder%"]) <= 2
                                    and abs(cm["blunder%"] - rm["blunder%"]) < abs(bm["blunder%"] - rm["blunder%"])
                                    and abs(cm["ACPL"] - rm["ACPL"]) < abs(bm["ACPL"] - rm["ACPL"]))
            return b

        bars = bars_for(arms["CLONE"])
        print("  bars: " + "  ".join(f"{k} {'PASS' if v else 'fail'}" for k, v in bars.items()))
        if "IDENT" in arms:      # identity-head arm, same bars, kept out of the OVERALL tally
            print("  IDENT: " + "  ".join(f"{k} {'PASS' if v else 'fail'}" for k, v in bars_for(arms["IDENT"]).items()))
        for k, v in bars.items():
            tally.setdefault(k, []).append(v)
    print("\nOVERALL (need >= 2 of 3): " + "  ".join(f"{k} {sum(v)}/{len(v)}" for k, v in tally.items()))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/clonefish/eval5_field.json")
