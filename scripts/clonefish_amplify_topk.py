"""Clone vs stock across many openings -> where does amplifying the difference actually improve TOP-1 / TOP-3?

Why this is not the circular test I ran before. Every earlier push/head experiment FITTED by log-likelihood and only
reported top-k afterwards — and log-likelihood is the exact loss the clone was fine-tuned with, so lambda = 1 won by
construction. **Top-k accuracy is a different objective** (Lapin 2015): a push can improve top-3 while hurting NLL.
So here w is SELECTED by top-3, never by NLL.

Two halves, and each does the job it can actually do:
  * self-play (clone vs stock, forced through many different openings) says WHERE the two models disagree and how
    often each kind of position actually arises in play — it has no ground-truth player move, so it cannot score.
  * the player's real held-out games supply the labels, because top-1/top-3 only exists where we know what they
    played.

Segmented, because a signal that helps in sharp positions can be invisible once averaged over quiet ones: by phase,
by whether clone and stock even disagree, and by opening. Selection of w is cross-validated BY GAME, so the reported
gain is what you would get choosing w on other games — not the best cell in the table.

    PYTHONPATH=. python scripts/clonefish_amplify_topk.py --player VEGETAL --clone clones/ftval_VEGETAL_bucket.pt
"""
import argparse, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np, torch, chess
from finetune_clone import games_of
from clonefish_push_fit_moves import cache
from clonefish_uci import CloneEngine, player_data
from clonefish_eval import simulate, CKPT
from sahformer.training.loop import load_model

dkey = lambda g: ((g.headers.get("UTCDate") or g.headers.get("Date") or ""),
                  (g.headers.get("UTCTime") or g.headers.get("StartTime") or ""))

# a spread of openings so self-play is not 150 repeats of the clone's favourite line
OPENINGS = [[], ["e2e4", "e7e5"], ["e2e4", "c7c5"], ["e2e4", "e7e6"], ["e2e4", "c7c6"],
            ["d2d4", "d7d5"], ["d2d4", "g8f6"], ["d2d4", "f7f5"], ["c2c4", "e7e5"], ["g1f3", "d7d5"]]


def topk(score, played, mask):
    """top-1 / top-3 of the played move under these scores (ranking only, so no renormalising needed)"""
    s = np.where(mask, score, -np.inf)
    order = np.argsort(-s, axis=1)
    hit1 = (order[:, 0] == played)
    hit3 = (order[:, :3] == played[:, None]).any(1)
    return hit1, hit3


def sweep(C, B, played, mask, grid):
    """top-1/top-3 for every w, vectorised over the whole position set"""
    out = {}
    for w in grid:
        out[w] = topk(C + w * (C - B), played, mask)
    return out


def cv_select(hits, gid, grid, seed=0):
    """5-fold by game: pick w on the other folds by top-3, score this fold. Returns (top1, top3) at the selected w,
    the baseline at w=0, and how often each w got picked."""
    games = np.unique(gid); fold = dict(zip(games, np.random.default_rng(seed).integers(0, 5, len(games))))
    f = np.array([fold[g] for g in gid])
    t1 = t3 = b1 = b3 = n = 0; picks = []
    for k in range(5):
        tr, te = f != k, f == k
        if not te.any() or not tr.any():
            continue
        best = max(grid, key=lambda w: hits[w][1][tr].mean())
        picks.append(best)
        t1 += hits[best][0][te].sum(); t3 += hits[best][1][te].sum()
        b1 += hits[0.0][0][te].sum(); b3 += hits[0.0][1][te].sum()
        n += te.sum()
    if not n:
        return None
    return dict(top1=100 * t1 / n, top3=100 * t3 / n, base1=100 * b1 / n, base3=100 * b3 / n,
                n=int(n), picks=picks)


def report(name, hits, gid, grid, share=None):
    r = cv_select(hits, gid, grid)
    if r is None or r["n"] < 200:
        return None
    sh = f" | {share:4.1f}% of self-play positions" if share is not None else ""
    print(f"  {name:<22} n={r['n']:5d}  top1 {r['base1']:5.2f} -> {r['top1']:5.2f} ({r['top1'] - r['base1']:+.2f})"
          f"  top3 {r['base3']:5.2f} -> {r['top3']:5.2f} ({r['top3'] - r['base3']:+.2f})  w={r['picks']}{sh}",
          flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", required=True); ap.add_argument("--clone", required=True)
    ap.add_argument("--data-dir", default=os.path.join("data", "lichess_5k"))
    ap.add_argument("--games", type=int, default=80, help="held-out real games (never trained on)")
    ap.add_argument("--selfplay", type=int, default=40, help="clone-vs-stock games, spread over the openings")
    ap.add_argument("--grid", default="-0.5,-0.25,0,0.25,0.5,1,1.5,2,3")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "clonefish", "amplify_topk.json"))
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    grid = [float(x) for x in a.grid.split(",")]
    if 0.0 not in grid:
        grid.append(0.0)
    t0 = time.time(); me = a.player.lower()

    stock, mcfg = load_model(CKPT); stock.eval().to(dev)
    clone, _ = load_model(CKPT)
    clone.load_state_dict(torch.load(a.clone, map_location="cpu", weights_only=False)["model_state"])
    clone.eval().to(dev)

    # ---- half 1: self-play across openings — where do clone and stock disagree, and how often? ----------
    pgn = os.path.join(ROOT, a.data_dir, f"{a.player}.pgn.zst")
    data = player_data(pgn, a.player, exclude_recent=80)
    shares = {}
    if a.selfplay:
        pl = CloneEngine(CKPT, a.clone, data, device=dev, seed=2)
        op = CloneEngine(CKPT, None, None, device=dev, seed=3)
        op.opt.update({"Elo": data["opp_elo"], "OppElo": data["elo"], "UseBook": False})
        per = max(1, a.selfplay // len(OPENINGS)); sp = []
        for line in OPENINGS:
            for j in range(per):
                sp.append(simulate(pl, op, player_white=(j % 2 == 0), start=line))
        del pl, op; torch.cuda.empty_cache()
        plies = [sum(1 for _ in g.mainline()) for g in sp]
        # how often would ANY amplification even bite? it can only change a move where clone and stock disagree
        Cs, Bs, ps, mvs, _ = cache(clone, stock, sp, "player", dev)
        msk = Cs > -1e8
        ag = np.array([np.argmax(np.where(msk[i], Cs[i], -np.inf)) == np.argmax(np.where(msk[i], Bs[i], -np.inf))
                       for i in range(len(ps))])
        shares = {"opening (<10)": 100 * np.mean(mvs < 10), "middle (10-25)": 100 * np.mean((mvs >= 10) & (mvs < 25)),
                  "late (>25)": 100 * np.mean(mvs >= 25),
                  "clone==stock": 100 * ag.mean(), "clone!=stock": 100 * (~ag).mean()}
        print(f"{a.player}: {len(sp)} clone-vs-stock games over {len(OPENINGS)} openings, "
              f"median {np.median(plies):.0f} plies; in those games clone and stock pick DIFFERENT moves in "
              f"{100 * (~ag).mean():.1f}% of the clone's own positions ({time.time() - t0:.0f}s)", flush=True)

    # ---- half 2: the player's real held-out games — the only place top-1/top-3 exists ---------------------
    gs = games_of(pgn, me); gs.sort(key=dkey)
    held = gs[-a.games:]
    C, B, played, mv, gid = cache(clone, stock, held, me, dev)
    mask = C > -1e8
    Cn = np.where(mask, C, -np.inf); Bn = np.where(mask, B, -np.inf)
    first = {}
    for i, g in enumerate(held):
        m = list(g.mainline_moves())
        first[i] = m[0].uci() if m else ""
    fm = np.array([first.get(int(g), "") for g in gid])
    agree = np.array([np.argmax(Cn[i]) == np.argmax(Bn[i]) for i in range(len(played))])
    print(f"  {len(played)} held-out own moves; clone and stock pick the same move in "
          f"{100 * agree.mean():.1f}% of them", flush=True)

    hits = sweep(Cn, Bn, played, mask, grid)
    print(f"\n{a.player}: amplification selected BY TOP-3, 5-fold CV by game "
          f"(w picked per fold; w=0 is the plain clone)", flush=True)
    res = {"all": report("all moves", hits, gid, grid)}
    segs = {"opening (<10)": mv < 10, "middle (10-25)": (mv >= 10) & (mv < 25), "late (>25)": mv >= 25,
            "clone==stock": agree, "clone!=stock": ~agree,
            "1.e4 games": fm == "e2e4", "1.d4 games": fm == "d2d4"}
    for nm, m_ in segs.items():
        if m_.sum() < 200:
            continue
        sub = {w: (h[0][m_], h[1][m_]) for w, h in hits.items()}
        res[nm] = report(nm, sub, gid[m_], grid, shares.get(nm))
    allres = json.load(open(a.out)) if os.path.exists(a.out) else {}
    allres[a.player] = res
    json.dump(allres, open(a.out, "w"), indent=1, default=float)
    print(f"\nwrote {os.path.relpath(a.out, ROOT)} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
