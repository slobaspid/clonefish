"""Fit the norm push against the player's REAL moves instead of against ACPL in self-play.

Why: `score = clone + w*(clone - stock)` is linear extrapolation in log-space borrowed from classifier-free
guidance — one scalar for every position, phase and clock — and fitting w by matching self-play ACPL was
seed-noisy (the same player's fit flipped between 0 and 0.5 on the seed alone). The player's own held-out games
have no simulation noise, so fit there and let the data pick the shape.

Families (all scored as log-probabilities over the legal moves, no book, no top-p):
  linear w  : s = lc + w*(lc - lb)                      (the current engine option; w=0 is the plain clone)
  affine    : s = a*lc + b*lb                           (strictly more general: linear push is a=1+w, b=-w,
                                                         temperature is a=b=1/T)
  gap-power : s = lc + w*sign(d)*|d|^p,  d = lc - lb     (push hard where clone and norm disagree a lot, or the
                                                         opposite, instead of proportionally)
  phase     : affine with its own (a,b) per phase bucket (own move <10 / 10-25 / >25)

Everything is cross-validated 5-fold BY GAME, so a family with more parameters has to earn it.

    PYTHONPATH=. python scripts/clonefish_push_fit_moves.py --players VEGETAL:clones/ftval_VEGETAL_bucket.pt
"""
import argparse, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np, torch, torch.nn.functional as F, chess
from finetune_clone import games_of, rows_of, batch_of
from sahformer.training.loop import load_model

CKPT = os.path.join(ROOT, "checkpoints", "base_300k_best.pt")
dkey = lambda g: ((g.headers.get("UTCDate") or g.headers.get("Date") or ""),
                  (g.headers.get("UTCTime") or g.headers.get("StartTime") or ""))


@torch.no_grad()
def cache(clone, stock, games, me, dev):
    """clone/stock logits over the legal moves of every own turn, + which one the player actually played"""
    rows, gid = [], []
    for i, g in enumerate(games):
        if g.board().fen() != chess.STARTING_FEN:
            continue
        r = rows_of(g, me)
        rows += r; gid += [i] * len(r)
    keep = [j for j, r in enumerate(rows) if r[5] in r[6] and len(r[6]) > 1]
    rows = [rows[j] for j in keep]; gid = np.array([gid[j] for j in keep])
    L = max(len(r[6]) for r in rows)
    C = np.full((len(rows), L), -1e9, np.float32); B = np.full((len(rows), L), -1e9, np.float32)
    played = np.zeros(len(rows), np.int64); mv = np.zeros(len(rows), np.int64)
    for s in range(0, len(rows), 256):
        idx = range(s, min(s + 256, len(rows)))
        b, ch = batch_of(rows, idx, dev)
        lc = clone(b)["move_logits"].float().cpu().numpy()
        lb = stock(b)["move_logits"].float().cpu().numpy()
        for j, r in enumerate(ch):
            legal = r[6]; n = len(legal)
            C[s + j, :n] = lc[j][legal]; B[s + j, :n] = lb[j][legal]
            played[s + j] = legal.index(r[5]); mv[s + j] = r[8]
    return C, B, played, mv, gid


def logsm(x):
    return x - torch.logsumexp(x, -1, keepdim=True)


def make(family, phase):
    """returns (init params, score function) — score maps (lc, lb) to un-normalised move scores"""
    if family == "linear":
        return torch.zeros(1), lambda p, lc, lb, ph: lc + p[0] * (lc - lb)
    if family == "affine":
        return torch.tensor([1.0, 0.0]), lambda p, lc, lb, ph: p[0] * lc + p[1] * lb
    if family == "gap":
        return torch.tensor([0.0, 1.0]), lambda p, lc, lb, ph: lc + p[0] * torch.sign(lc - lb) * (lc - lb).abs().clamp(
            min=1e-6) ** p[1].clamp(0.2, 3.0)
    if family == "phase":
        return torch.tensor([1.0, 0.0] * phase), lambda p, lc, lb, ph: (
            p.view(-1, 2)[ph, 0:1] * lc + p.view(-1, 2)[ph, 1:2] * lb)
    raise ValueError(family)


def nll_acc(score, played, mask):
    lp = logsm(score.masked_fill(~mask, -1e9))
    nll = -lp.gather(1, played[:, None]).squeeze(1).mean()
    top = lp.topk(3, -1).indices
    return nll, (top[:, 0] == played).float().mean().item(), (top == played[:, None]).any(1).float().mean().item()


def run(family, C, B, played, ph, mask, folds, nph, steps=400):
    """5-fold-by-game CV: fit on the other folds, score the held-out one"""
    out = []
    for k in range(5):
        tr, te = folds != k, folds == k
        p, f = make(family, nph); p = p.clone().requires_grad_(True)
        opt = torch.optim.Adam([p], lr=0.05)
        for _ in range(steps):
            opt.zero_grad()
            loss, _, _ = nll_acc(f(p, C[tr], B[tr], ph[tr]), played[tr], mask[tr])
            loss.backward(); opt.step()
        with torch.no_grad():
            nll, t1, t3 = nll_acc(f(p.detach(), C[te], B[te], ph[te]), played[te], mask[te])
        out.append((nll.item(), t1, t3, p.detach().clone()))
    n, t1, t3 = (float(np.mean([o[i] for o in out])) for i in range(3))
    return n, t1, t3, torch.stack([o[3] for o in out]).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", required=True, help="NAME:clone[:data_dir],...")
    ap.add_argument("--games", type=int, default=80, help="newest N games, never trained on")
    ap.add_argument("--post-opening", type=int, default=5, help="also report own moves after this one")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "clonefish", "push_fit_moves.json"))
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    stock, _ = load_model(CKPT); stock.eval().to(dev)
    for spec in a.players.split(","):
        parts = spec.split(":"); nm, cp = parts[0], parts[1]
        ddir = parts[2] if len(parts) > 2 else os.path.join("data", "lichess_5k")
        t0 = time.time()
        clone, _ = load_model(CKPT)
        clone.load_state_dict(torch.load(cp, map_location="cpu", weights_only=False)["model_state"])
        clone.eval().to(dev)
        gs = games_of(os.path.join(ROOT, ddir, f"{nm}.pgn.zst"), nm.lower()); gs.sort(key=dkey)
        C, B, played, mv, gid = cache(clone, stock, gs[-a.games:], nm.lower(), dev)
        del clone; torch.cuda.empty_cache()
        mask = torch.from_numpy(C > -1e8)
        C = logsm(torch.from_numpy(C).masked_fill(~mask, -1e9))
        B = logsm(torch.from_numpy(B).masked_fill(~mask, -1e9))
        played = torch.from_numpy(played)
        ph = torch.from_numpy(np.digitize(mv, [10, 25]))
        rng = np.random.default_rng(0); g2f = {g: rng.integers(0, 5) for g in np.unique(gid)}
        folds = torch.from_numpy(np.array([g2f[g] for g in gid]))
        res[nm] = {}
        for tag, sel in (("all moves", torch.ones(len(played), dtype=torch.bool)),
                         ("post-opening", torch.from_numpy(mv > a.post_opening))):
            print(f"\n{nm} — {tag} ({int(sel.sum())} positions, {len(np.unique(gid))} held-out games)", flush=True)
            base = nll_acc(C[sel], played[sel], mask[sel])
            print(f"   {'clone as-is':<12} NLL {base[0].item():.4f}  top1 {100 * base[1]:.2f}%  top3 {100 * base[2]:.2f}%",
                  flush=True)
            res[nm][tag] = {"clone": [base[0].item(), base[1], base[2]]}
            for fam in ("linear", "affine", "gap", "phase"):
                n, t1, t3, p = run(fam, C[sel], B[sel], played[sel], ph[sel], mask[sel], folds[sel], 3)
                print(f"   {fam:<12} NLL {n:.4f} ({n - base[0].item():+.4f})  top1 {100 * t1:.2f}% "
                      f"({100 * (t1 - base[1]):+.2f})  top3 {100 * t3:.2f}% ({100 * (t3 - base[2]):+.2f})  "
                      f"params {np.round(p.numpy(), 3).tolist()}", flush=True)
                res[nm][tag][fam] = {"nll": n, "top1": t1, "top3": t3, "params": p.numpy().tolist()}
            json.dump(res, open(a.out, "w"), indent=1)
        print(f"[{nm}] {time.time() - t0:.0f}s", flush=True)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
