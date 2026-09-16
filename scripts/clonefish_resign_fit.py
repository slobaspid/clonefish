"""Learn when a player resigns, from their own games (never the newest 80 = the test games).

For every turn of a side (with clock), features = resign_features(...) from clonefish_uci: the model's own
win/loss estimate for the side to move, material balance, both clocks, move number. Label = 1 on that side's
last turn in games it RESIGNED (Termination Normal, it lost, final position not checkmate), else 0.
A logistic regression then gives a per-turn resignation probability, sampled by the engine each move.

Two models per player file:
  player : the player's turns, features from their fine-tuned CLONE (what the engine runs)
  field  : their opponents' turns, features from the BASE model (used for the simulated opponent)

    PYTHONPATH=. python scripts/clonefish_resign_fit.py --player VEGETAL --clone clones/ftval_VEGETAL_bucket.pt
"""
import argparse, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np, torch, torch.nn.functional as F, chess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from finetune_clone import games_of, rows_of, batch_of
from clonefish_uci import resign_features, resign_history, danger_features, material
from sahformer.training.loop import load_model

CKPT = os.path.join(ROOT, "checkpoints", "base_300k_best.pt")
dkey = lambda g: ((g.headers.get("UTCDate") or g.headers.get("Date") or ""),
                  (g.headers.get("UTCTime") or g.headers.get("StartTime") or ""))


def side_rows(g, name):
    """rows_of for `name` + per-row (material, my_clock, opp_clock) + resigned label on the last row."""
    rows = rows_of(g, name)
    if not rows:
        return [], [], []
    me_white = (name == (g.headers.get("White", "") or "").lower())
    board, prev, extra = g.board(), {chess.WHITE: 180.0, chess.BLACK: 180.0}, []
    for nd in g.mainline():
        mover, c = board.turn, nd.clock()
        if c is None:
            board.push(nd.move); continue
        if (mover == chess.WHITE) == me_white:
            extra.append((material(board, mover), prev[mover], prev[not mover], danger_features(board)))
        prev[mover] = c; board.push(nd.move)
    assert len(extra) == len(rows), (len(extra), len(rows))
    r = g.headers.get("Result")
    lost = r in ("1-0", "0-1") and ((r == "1-0") != me_white)
    resigned = g.headers.get("Termination") == "Normal" and lost and not board.is_checkmate()
    labels = [0] * len(rows)
    if resigned:
        labels[-1] = 1
    return rows, extra, labels


@torch.no_grad()
def featurize(model, rows, extra, dev, gid):
    """Returns (all 18 features, own move number per row); v1/v2 are the first 10/14 columns. Built exactly as the
    engine builds them: one pass per game, rolling (previous p_loss, previous material, lost-streak) forward."""
    raw = []
    for s in range(0, len(rows), 512):
        b, ch = batch_of(rows, range(s, min(s + 512, len(rows))), dev)
        pv = F.softmax(model(b)["value_logits"].float(), -1).cpu().numpy()
        for j, r in enumerate(ch):
            mat, mc, oc, dg = extra[s + j]
            raw.append((float(pv[j, 0]), float(pv[j, 2]), mat, mc, oc, r[8], dg))
    X, h, prev = [], None, object()
    for i, a in enumerate(raw):
        if gid[i] != prev:
            h, prev = None, gid[i]
        X.append(resign_features(*a[:6], hist=(h or (a[0], a[2], 0)), danger=a[6]))
        h = resign_history(h, a[0], a[2])
    return np.array(X, float), np.array([a[5] for a in raw], float)


def game_report(oof, y, gid, mv, tag):
    """What the engine actually does: roll the hazard forward and stop at the first firing. Per-row calibration can
    read 1.00 while this is 30 % low, because a long lost stretch spreads mass the real process concentrates."""
    fire, move, w, real_r, real_m = [], [], [], 0, []
    for g in np.unique(gid):
        i = np.flatnonzero(gid == g); h = np.clip(oof[i], 1e-12, 1 - 1e-9); mm = mv[i]
        p = h * np.concatenate([[1.0], np.cumprod(1 - h)[:-1]]); tot = p.sum()
        fire.append(tot)
        if tot > 0:
            move.append((p * mm).sum() / tot); w.append(tot)
        if y[i].sum():
            real_r += 1; real_m.append(mm[y[i] == 1][0])
    fr, rr = float(np.mean(fire)), real_r / max(1, len(np.unique(gid)))
    fm = float(np.average(move, weights=w)) if w else float("nan")
    rm = float(np.mean(real_m)) if real_m else float("nan")
    print(f"  [{tag}] per GAME: fires in {100 * fr:.1f}% of games (real {100 * rr:.1f}%) at mean move {fm:.1f} "
          f"(real {rm:.1f})", flush=True)
    return {"fire_rate": fr, "real_rate": rr, "fire_move": fm, "real_move": rm}


def ply_report(oof, y, gid, mv, tag):
    """How deep into the game the hazard fires, vs when the player really resigned (own move numbers).
    Per resigned game: E[move of the first firing | it fires at all] from the out-of-sample hazards."""
    pred, real = [], []
    for g in np.unique(gid[y == 1]):
        m = gid == g
        h = np.clip(oof[m], 1e-9, 1 - 1e-9); mvs = mv[m]
        p = h * np.concatenate([[1.0], np.cumprod(1 - h)[:-1]])
        if p.sum() <= 0:
            continue
        pred.append(float((p * mvs).sum() / p.sum())); real.append(float(mvs[y[m] == 1][0]))
    if not pred:
        return None
    pred, real = np.array(pred), np.array(real)
    print(f"  [{tag}] resign timing: predicted move {np.mean(pred):.1f} vs real {np.mean(real):.1f} "
          f"(median {np.median(pred):.1f} vs {np.median(real):.1f}, mean gap {np.mean(pred - real):+.1f} moves, n={len(pred)})",
          flush=True)
    return float(np.mean(pred - real))


def fit(X, y, gid, tag, mv=None):
    if y.sum() < 5:        # e.g. kpowe52 never resigns: a flat, smoothed per-turn rate instead of a regression
        rate = (y.sum() + 0.5) / (len(y) + 1.0)
        print(f"  [{tag}] rows {len(y)} resignations {int(y.sum())} | too few to fit -> constant per-turn rate {rate:.2e} "
              f"(~{rate * len(y) / max(1, len(np.unique(gid))):.4f} resignations per game)", flush=True)
        return {"mean": X.mean(0).tolist(), "std": (X.std(0) + 1e-6).tolist(), "coef": [0.0] * X.shape[1],
                "intercept": float(np.log(rate / (1 - rate))), "val_auc": None, "constant": True,
                "n_rows": int(len(y)), "n_resign": int(y.sum())}
    # 5-fold cross-validation grouped by game: every resignation is scored once out-of-sample, so the calibration
    # ratio uses all positives (a single 10 % split had only ~10-45 resignations and was too noisy to read)
    games = np.unique(gid); rng = np.random.default_rng(0); fold_of = dict(zip(games, rng.integers(0, 5, len(games))))
    folds = np.array([fold_of[g] for g in gid]); oof = np.zeros(len(y))
    for k in range(5):
        tr, te = folds != k, folds == k
        if y[tr].sum() < 2: continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        oof[te] = LogisticRegression(C=1.0, max_iter=3000).fit((X[tr] - mu) / sd, y[tr]).predict_proba((X[te] - mu) / sd)[:, 1]
    auc = roc_auc_score(y, oof) if 0 < y.sum() < len(y) else float("nan")
    ratio = oof.sum() / max(1, y.sum())
    lost_rows = X[:, 0] > 0.8                                       # value head says the side to move is losing
    print(f"  [{tag}] rows {len(y)} resignations {int(y.sum())} | 5-fold AUC {auc:.3f} | out-of-sample predicted/real "
          f"resignations {oof.sum():.1f}/{int(y.sum())} = {ratio:.2f} | in lost positions (p_loss>0.8): predicted "
          f"{oof[lost_rows].sum():.1f} vs real {int(y[lost_rows].sum())}", flush=True)
    ply_gap = ply_report(oof, y, gid, mv, tag) if mv is not None else None
    game = game_report(oof, y, gid, mv, tag) if mv is not None else None
    mean, std = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=1.0, max_iter=3000).fit((X - mean) / std, y)      # refit on everything
    return {"mean": mean.tolist(), "std": std.tolist(), "coef": clf.coef_[0].tolist(),
            "intercept": float(clf.intercept_[0]), "cv_auc": auc, "cv_calibration": float(ratio),
            "cv_ply_gap": ply_gap, "cv_game": game, "n_rows": int(len(y)), "n_resign": int(y.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", required=True); ap.add_argument("--clone", required=True)
    ap.add_argument("--pgn", default=None); ap.add_argument("--games", type=int, default=2500)
    ap.add_argument("--exclude-recent", type=int, default=80)
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    pgn = a.pgn or os.path.join(ROOT, "data", "lichess_5k", f"{a.player}.pgn.zst")
    me = a.player.lower(); t0 = time.time()
    gs = games_of(pgn, me); gs.sort(key=dkey)
    gs = (gs[:-a.exclude_recent] if a.exclude_recent > 0 else gs)[-a.games:]      # --exclude-recent 0 = use all games
    P = {"rows": [], "extra": [], "y": [], "gid": []}; O = {"rows": [], "extra": [], "y": [], "gid": []}
    for gi, g in enumerate(gs):
        if g.board().fen() != chess.STARTING_FEN:
            continue
        opp = ((g.headers.get("Black") if me == (g.headers.get("White", "") or "").lower() else g.headers.get("White")) or "").lower()
        for D, nm in ((P, me), (O, opp)):
            rows, extra, y = side_rows(g, nm)
            D["rows"] += rows; D["extra"] += extra; D["y"] += y; D["gid"] += [gi] * len(rows)
    print(f"{a.player}: {len(gs)} games parsed in {time.time() - t0:.0f}s", flush=True)

    gP, gO = np.array(P["gid"]), np.array(O["gid"])
    clone, _ = load_model(CKPT)
    clone.load_state_dict(torch.load(a.clone, map_location="cpu", weights_only=False)["model_state"]); clone.eval().to(dev)
    XP, mvP = featurize(clone, P["rows"], P["extra"], dev, gP); del clone; torch.cuda.empty_cache()
    base, _ = load_model(CKPT); base.eval().to(dev)
    XO, mvO = featurize(base, O["rows"], O["extra"], dev, gO); del base; torch.cuda.empty_cache()

    np.savez_compressed(os.path.join(ROOT, "clones", f"{a.player}_resign_features.npz"),       # re-analyse without a GPU
                        XP=XP, yP=np.array(P["y"]), gP=gP, mvP=mvP, XO=XO, yO=np.array(O["y"]), gO=gO, mvO=mvO)

    def best(X, y, gid, mv, tag):
        """v1 = clock/material/value features, v2 = + history, v3 = + mate-danger. Keep the best out-of-sample AUC
        (a constant-rate player has nothing to choose, so keep v1 and save the engine the danger computation)."""
        cand = {"v1": fit(X[:, :10], y, gid, tag + " v1", mv), "v2": fit(X[:, :14], y, gid, tag + " v2", mv),
                "v3": fit(X, y, gid, tag + " v3", mv)}
        pick = "v1" if cand["v1"].get("constant") else max(cand, key=lambda k: cand[k].get("cv_auc") or 0.0)
        print(f"  [{tag}] -> {pick}", flush=True)
        return dict(cand[pick], picked=pick), cand

    y_, yo_ = np.array(P["y"]), np.array(O["y"])
    pl, plc = best(XP, y_, gP, mvP, "player"); fl, flc = best(XO, yo_, gO, mvO, "field")
    out = {"player": pl, "field": fl, "player_variants": plc, "field_variants": flc,
           "games": len(gs), "exclude_recent": a.exclude_recent, "clone": a.clone}
    path = os.path.join(ROOT, "clones", f"{a.player}_resign.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"saved {path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
