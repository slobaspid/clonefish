"""Night test 2026-09-12: K1 weight dial (WiSE-FT) + K4 per-user timing calibration.
Plan: results/night_0913/PLAN.md. Reuses finetune_clone's exact game loading / sort / row building.

  PYTHONPATH=. python scripts/night_dial_eval.py --pgn data/lichess_5k/VEGETAL.pgn.zst --name VEGETAL \
      --ft clones/ftval_VEGETAL.pt            # stage 1: val = games [-80,-40), test = [-40:]
  ... --stage0 --ft clones/ft5k_VEGETAL.pt    # smoke test: cross-fit dial on odd/even test games
"""
import argparse, os, sys, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, chess
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finetune_clone import games_of, rows_of, batch_of
from sahformer.records import BASE_SECONDS
from sahformer.training.loop import load_model

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
CAP = 60                                     # readings above this are pooled into one bucket


def clocks_of(g, me):
    """clock-before-move for every row rows_of emits (same iteration order)."""
    me_white = (me == (g.headers.get("White", "") or "").lower())
    board, prev, out, node = g.board(), {chess.WHITE: BASE_SECONDS, chess.BLACK: BASE_SECONDS}, [], g
    while node.variations:
        node = node.variation(0); mover = board.turn; ca = node.clock()
        if ca is None: board.push(node.move); continue
        if (mover == chess.WHITE) == me_white: out.append(prev[mover])
        prev[mover] = ca; board.push(node.move)
    return out


def load_split(pgn, name, stage0, test_n=40, val_n=40):
    dkey = lambda g: ((g.headers.get("UTCDate") or g.headers.get("Date") or ""),
                      (g.headers.get("UTCTime") or g.headers.get("StartTime") or ""))
    gs = games_of(pgn, name.lower()); gs.sort(key=dkey)
    def rows(glist):
        R = []
        for gi, g in enumerate(glist):
            rs, cs = rows_of(g, name.lower()), clocks_of(g, name.lower())
            assert len(rs) == len(cs)
            R += [(r, c, gi) for r, c in zip(rs, cs)]
        return R
    test = rows(gs[-test_n:])
    val = [] if stage0 else rows(gs[-(test_n + val_n):-test_n])
    return test, val


@torch.no_grad()
def forward_all(model, R, dev):
    rows = [x[0] for x in R]; hit1, hit3, pis, mus, sgs = [], [], [], [], []
    for s in range(0, len(rows), 256):
        b, ch = batch_of(rows, range(s, min(s + 256, len(rows))), dev)
        out = model(b); ml = out["move_logits"].float().cpu()
        pi, mu, sg = [t.float().cpu() for t in out["mdn"]]
        for j, r in enumerate(ch):
            legal, actual = r[6], r[5]
            if actual in legal:
                lg = ml[j][legal]; ai = legal.index(actual)
                top = lg.topk(min(3, len(legal))).indices.tolist()
                hit1.append(int(top[0] == ai)); hit3.append(int(ai in top))
            else:
                hit1.append(-1); hit3.append(-1)
        pis.append(pi); mus.append(mu); sgs.append(sg)
    return (np.array(hit1), np.array(hit3), torch.cat(pis).double(), torch.cat(mus).double(),
            torch.cat(sgs).double())


def mix(pi, mu, sgp, cal=None):
    """calibrated mixture params. cal = (shift, log_sigma_scale, snap_bias)."""
    logpi = F.log_softmax(pi, -1); sig = F.softplus(sgp) + 1e-3
    if cal is not None:
        s, g, b = cal
        mu = mu + s; sig = sig * torch.exp(g)
        snap = F.one_hot(mu.argmin(-1), mu.shape[-1]).double()
        logpi = F.log_softmax(logpi + b * snap, -1)
    return logpi, mu, sig


def reading_logprob(logpi, mu, sig, k, n_q=24):
    """log P(clock reading = k). True think t in (k-1, k+1) with triangular weight (clock phase)."""
    k = k.double().unsqueeze(-1)
    lo = (k - 1).clamp_min(0); hi = k + 1
    u = (torch.arange(n_q, dtype=torch.float64) + 0.5) / n_q
    t = lo + (hi - lo) * u                                        # (N, n_q)
    w = (1 - (t - k).abs()).clamp_min(0) * (hi - lo) / n_q        # triangular kernel * dt
    lt = t.log().unsqueeze(-1)                                    # (N, n_q, 1)
    comp = -0.5 * ((lt - mu.unsqueeze(1)) / sig.unsqueeze(1)) ** 2 - sig.unsqueeze(1).log() \
        - 0.5 * math.log(2 * math.pi) - lt
    dens = torch.logsumexp(logpi.unsqueeze(1) + comp, -1).exp()    # pdf of t
    p = (dens * w).sum(-1)
    return p.clamp_min(1e-12).log()


def sample_readings(logpi, mu, sig, gen):
    c = torch.multinomial(logpi.exp(), 1, generator=gen).squeeze(-1)
    m = mu.gather(-1, c[:, None]).squeeze(-1); s = sig.gather(-1, c[:, None]).squeeze(-1)
    t = torch.exp(m + s * torch.randn(m.shape, generator=gen, dtype=torch.float64))
    u = torch.rand(m.shape, generator=gen, dtype=torch.float64)
    return torch.ceil(t - u).clamp_min(0)


def w1_log(a, b):
    a = np.sort(np.log1p(np.asarray(a, float))); b = np.sort(np.log1p(np.asarray(b, float)))
    q = np.linspace(0, 1, 201)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def time_stats(sampled, actual):
    s, a = np.asarray(sampled), np.asarray(actual)
    return {"W1log": round(w1_log(s, a), 4), "snap_sim": round(float((s == 0).mean()), 3),
            "snap_real": round(float((a == 0).mean()), 3), "tank_sim": round(float((s >= 10).mean()), 3),
            "tank_real": round(float((a >= 10).mean()), 3), "median_sim": float(np.median(s)),
            "median_real": float(np.median(a)),
            "r_log": round(float(np.corrcoef(np.log1p(s), np.log1p(a))[0, 1]), 3)}


def bucket(clock, own):
    cb = 0 if clock < 30 else 1 if clock < 60 else 2 if clock < 120 else 3
    mb = 0 if own < 10 else 1 if own < 25 else 2 if own < 40 else 3
    return cb * 4 + mb


def lookup_null(fit_R, eval_R, gen):
    """empirical reading histogram per (clock bucket x move bucket) from fit rows; add-0.5 smoothing."""
    H = np.full((16, CAP + 1), 0.5)
    for r, c, _ in fit_R: H[bucket(c, r[8]), min(int(round(r[7])), CAP)] += 1
    H /= H.sum(1, keepdims=True)
    nll = -np.mean([math.log(H[bucket(c, r[8]), min(int(round(r[7])), CAP)]) for r, c, _ in eval_R])
    rng = np.random.default_rng(0)
    samp = [rng.choice(CAP + 1, p=H[bucket(c, r[8])]) for r, c, _ in eval_R]
    return nll, samp


def fit_cal(pi, mu, sg, k, steps=300, no_snap=False):
    p = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    keep = torch.tensor([1.0, 1.0, 0.0 if no_snap else 1.0], dtype=torch.float64)
    opt = torch.optim.Adam([p], lr=0.05)
    for _ in range(steps):
        q = p * keep
        lp, m, s = mix(pi, mu, sg, (q[0], q[1], q[2]))
        loss = -reading_logprob(lp, m, s, k).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return (p * keep).detach()


def post_acc(h1, h3, R, mask=None):
    sel = np.array([x[0][8] >= 12 for x in R]) & (h1 >= 0)
    if mask is not None: sel &= mask
    return 100 * h1[sel].mean(), 100 * h3[sel].mean(), sel


def boot_diff(hA, hB, sel, gid, n=1000):
    rng = np.random.default_rng(0); games = np.unique(gid[sel]); out = []
    idx = {g: np.where(sel & (gid == g))[0] for g in games}
    for _ in range(n):
        pick = np.concatenate([idx[g] for g in rng.choice(games, len(games))])
        out.append(100 * (hA[pick].mean() - hB[pick].mean()))
    return [round(float(np.percentile(out, 2.5)), 2), round(float(np.percentile(out, 97.5)), 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--ft", required=True); ap.add_argument("--ckpt", default="checkpoints/base_300k_best.pt")
    ap.add_argument("--stage0", action="store_true"); ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/night_0913")
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--no-snap", action="store_true", help="K4 fits only shift + sigma scale")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(); dev = a.device
    global ALPHAS; ALPHAS = [float(x) for x in a.alphas.split(",")]
    test, val = load_split(a.pgn, a.name, a.stage0)
    print(f"{a.name}: test rows {len(test)}, val rows {len(val)}", flush=True)
    base, _ = load_model(a.ckpt)
    bsd = {k: v.clone() for k, v in base.state_dict().items()}
    fsd = torch.load(a.ft, map_location="cpu", weights_only=False)["model_state"]
    model = base.to(dev).eval()
    kt = torch.tensor([x[0][7] for x in test]).round(); gid_t = np.array([x[2] for x in test])
    zero_frac = float((kt == 0).double().mean())
    res = {"name": a.name, "stage0": a.stage0, "zero_second_frac_test": round(zero_frac, 3), "arms": {}}
    fw = {}
    for al in ALPHAS:
        sd = {k: (bsd[k] + al * (fsd[k] - bsd[k]) if bsd[k].is_floating_point() else bsd[k]) for k in bsd}
        model.load_state_dict(sd)
        fw[al] = (forward_all(model, test, dev), forward_all(model, val, dev) if val else None)
        h1, h3, *_ = fw[al][0]; t1, t3, _ = post_acc(h1, h3, test)
        print(f"  alpha {al:.2f}: test post-opening top1 {t1:.2f} top3 {t3:.2f}", flush=True)
        res["arms"][f"a{al}"] = {"top1": round(t1, 2), "top3": round(t3, 2)}

    # ---- dial selection -------------------------------------------------------------------------
    def pick(hits_by_alpha, R, mask=None):
        best = max(ALPHAS, key=lambda al: (post_acc(*hits_by_alpha[al][:2], R, mask)[1],
                                           post_acc(*hits_by_alpha[al][:2], R, mask)[0]))
        return best
    ft1, ft3, sel = post_acc(*fw[1.0][0][:2], test)
    if a.stage0:                                  # cross-fit on odd/even test games
        parts = []
        for fit_par in (0, 1):
            fitm = (gid_t % 2) == fit_par; repm = ~fitm
            al = pick({x: fw[x][0] for x in ALPHAS}, test, fitm)
            r1, r3, _ = post_acc(*fw[al][0][:2], test, repm); f1, f3, _ = post_acc(*fw[1.0][0][:2], test, repm)
            parts.append({"alpha": al, "top1": round(r1, 2), "top3": round(r3, 2),
                          "ft_top1": round(f1, 2), "ft_top3": round(f3, 2)})
        res["dial_crossfit"] = parts; picked = None
    else:
        picked = pick({x: fw[x][1] for x in ALPHAS}, val)
        p1, p3, _ = post_acc(*fw[picked][0][:2], test)
        res["dial_picked"] = {"alpha": picked, "top1": round(p1, 2), "top3": round(p3, 2),
                              "d_top1_vs_ft": round(p1 - ft1, 2), "d_top3_vs_ft": round(p3 - ft3, 2),
                              "ci_top3_vs_ft": boot_diff(fw[picked][0][1], fw[1.0][0][1], sel, gid_t),
                              "ci_top1_vs_ft": boot_diff(fw[picked][0][0], fw[1.0][0][0], sel, gid_t)}
    print(json.dumps({k: v for k, v in res.items() if k != "arms"}, indent=1), flush=True)

    # ---- timing ---------------------------------------------------------------------------------
    gen = torch.Generator().manual_seed(0); T = {}
    arms = [("base", 0.0), ("ft", 1.0)] + ([("dial", picked)] if picked not in (None, 0.0, 1.0) else [])
    for nm, al in arms:
        _, _, pi, mu, sg = fw[al][0]
        lp, m, s = mix(pi, mu, sg)
        T[nm] = {"bucket_nll": round(float(-reading_logprob(lp, m, s, kt).mean()), 4),
                 **time_stats(sample_readings(lp, m, s, gen).numpy(), kt.numpy())}
        if val:                                   # K4: 3 scalars fitted on VALIDATION games only
            _, _, vpi, vmu, vsg = fw[al][1]
            kv = torch.tensor([x[0][7] for x in val]).round()
            cal = fit_cal(vpi, vmu, vsg, kv, no_snap=a.no_snap)
            lp2, m2, s2 = mix(pi, mu, sg, tuple(cal))
            T[nm + "+K4"] = {"cal(shift,logsig,snapb)": [round(float(c), 3) for c in cal],
                             "bucket_nll": round(float(-reading_logprob(lp2, m2, s2, kt).mean()), 4),
                             **time_stats(sample_readings(lp2, m2, s2, gen).numpy(), kt.numpy())}
    if val:
        nll, samp = lookup_null(val, test, gen)
        T["lookup_null(val)"] = {"bucket_nll": round(nll, 4), **time_stats(samp, kt.numpy())}
    res["timing"] = T
    for k, v in T.items(): print(f"  [time] {k}: {v}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, f"dial_{a.name}{'_stage0' if a.stage0 else ''}{a.tag}.json")
    json.dump(res, open(fn, "w"), indent=1); print("wrote", fn)


if __name__ == "__main__":
    main()
