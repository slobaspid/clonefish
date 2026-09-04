"""Stage 2+3: train every think-time head on identical frozen features and score them on how
closely they reproduce the REAL human time distribution.

Leakage control - positions inside a game are not independent, because the temporal vector
carries that player's own last-5 think-times. So the split is BY PLAYER: the test players are
humans the head has never seen. A position-level or even game-level split would let the head
memorise a person's tempo and score far better than it deserves.

Two baselines exist to catch the obvious trap. A model that ignores the position entirely and
samples from the training histogram will match the marginal distribution almost perfectly while
having zero skill. So:

  * `marginal`  - sample from the global training think-time histogram
  * `clock-marginal` - same, conditioned on the remaining-clock bucket

Any head that fails to beat these on the CONDITIONAL metrics (RPS, correlation) has learned
nothing, no matter how good its distribution match looks.

Checkpointed: each trained head is saved and skipped on re-run.

    PYTHONPATH=. python -u scripts/timehead_bakeoff.py --cache cache/timehead \
        --out checkpoints/timehead --results timehead_results.json
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from sahformer.model.heads import mdn_nll_per_sample
from sahformer.model.timebuckets import CENTERS, EDGES, N_BUCKETS, expected_time, to_bucket
from sahformer.training.timeloss import (
    LOSSES,
    inverse_frequency_weights,
    masked_probs,
    sample_weights,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- data

def load_players(cache_dir):
    out = []
    for p in sorted(glob.glob(os.path.join(cache_dir, "*.pt"))):
        d = torch.load(p, weights_only=False)
        d["name"] = os.path.basename(p)[:-3]
        out.append(d)
    return out


def stack(players, use_diff=True):
    feats, think, clock, pid = [], [], [], []
    for i, d in enumerate(players):
        f = d["pooled"].astype(np.float32)
        if use_diff:
            f = np.concatenate([f, d["diff"].astype(np.float32)], axis=1)
        feats.append(f)
        think.append(d["think"])
        clock.append(d["my_clock"])
        pid.append(np.full(len(d["think"]), i, np.int32))
    return (torch.from_numpy(np.concatenate(feats)),
            torch.from_numpy(np.concatenate(think)),
            torch.from_numpy(np.concatenate(clock)),
            torch.from_numpy(np.concatenate(pid)))


# ---------------------------------------------------------------- heads

class BucketHead(nn.Module):
    def __init__(self, in_dim, hid=256, k=N_BUCKETS):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hid),
                                 nn.ReLU(), nn.Linear(hid, k))

    def forward(self, x):
        return self.net(x)


class HurdleHead(nn.Module):
    """Two-process head, straight from the reaction-time literature.

    Human RT is standardly modelled as a mixture of a fast/automatic process and a deliberate
    one (ex-Gaussian, two-state "fast guess" models). In blitz that is premove-or-snap versus
    actually calculating. This head makes the split explicit: a gate for P(snap, bucket 0) and
    a separate distribution over the thinking buckets, instead of asking one softmax to carry
    both a spike at zero and a fat tail.

    Emits LOG-probabilities as its logits, so downstream softmax and clock-masking are exact
    (softmax of a log-probability vector returns the same vector).
    """

    def __init__(self, in_dim, hid=256, k=N_BUCKETS):
        super().__init__()
        self.body = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hid), nn.ReLU())
        self.gate = nn.Linear(hid, 1)            # P(snap)
        self.rest = nn.Linear(hid, k - 1)        # distribution over the thinking buckets

    def forward(self, x):
        h = self.body(x)
        g = self.gate(h)
        log_snap = nn.functional.logsigmoid(g)               # log P(snap)
        log_think = nn.functional.logsigmoid(-g)             # log P(not snap)
        return torch.cat([log_snap, log_think + torch.log_softmax(self.rest(h), -1)], dim=-1)


class OrderedLogitHead(nn.Module):
    """Cumulative-link ordinal head (OrderedLogitNN, arXiv 2507.00736).

    Instead of 30 free logits, predicts one latent 'how long does this position deserve' score
    and a set of monotone cutpoints, so the output is a valid ordinal distribution by
    construction rather than by hope. Increments are softplus'd to keep the cutpoints ordered.
    """

    def __init__(self, in_dim, hid=256, k=N_BUCKETS):
        super().__init__()
        self.body = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hid), nn.ReLU(),
                                  nn.Linear(hid, 1))
        self.first = nn.Parameter(torch.tensor(-2.0))
        self.inc = nn.Parameter(torch.full((k - 2,), -1.0))
        self.k = k

    def forward(self, x):
        z = self.body(x)                                       # (B,1) latent severity
        cuts = torch.cat([self.first[None],
                          self.first[None] + torch.cumsum(nn.functional.softplus(self.inc), 0)])
        cdf = torch.sigmoid(cuts[None, :] - z)                 # (B, k-1), non-decreasing
        one = torch.ones_like(cdf[:, :1])
        zero = torch.zeros_like(cdf[:, :1])
        full = torch.cat([zero, cdf, one], dim=1)
        p = (full[:, 1:] - full[:, :-1]).clamp_min(1e-9)
        return torch.log(p)


class MDNHead(nn.Module):
    """The incumbent: mixture of 3 log-normals, NLL in log-space."""

    def __init__(self, in_dim, hid=256, m=3):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU())
        self.pi, self.mu, self.sigma = (nn.Linear(hid, m) for _ in range(3))

    def forward(self, x):
        h = self.trunk(x)
        return self.pi(h), self.mu(h), self.sigma(h)


# ---------------------------------------------------------------- metrics

def wasserstein1(a, b):
    """1-Wasserstein between two equal-size empirical samples = mean |sorted difference|."""
    n = min(len(a), len(b))
    a = np.sort(np.asarray(a, np.float64)[:n])
    b = np.sort(np.asarray(b, np.float64)[:n])
    return float(np.abs(a - b).mean())


def pearson(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return pearson(ra, rb)


def score(name, sampled, etime, actual, pid, probs=None, targets=None):
    """Conditional skill (RPS/r) AND marginal distribution match, reported side by side."""
    res = {"variant": name}
    if probs is not None:
        p_cum = probs.cumsum(-1)
        idx = torch.arange(N_BUCKETS)[None, :]
        y_cum = (idx >= targets[:, None]).float()
        res["rps"] = float(((p_cum - y_cum) ** 2).sum(-1).mean())
        oh = torch.zeros_like(probs).scatter_(1, targets[:, None], 1.0)
        res["brier"] = float(((probs - oh) ** 2).sum(-1).mean())
        res["nll"] = float(-torch.log(probs.gather(1, targets[:, None]).squeeze(1) + 1e-9).mean())
        res["top_bucket_acc"] = float((probs.argmax(-1) == targets).float().mean())

    res["pearson_Et"] = pearson(etime, actual)
    res["spearman_Et"] = spearman(etime, actual)
    res["mae_Et"] = float(np.abs(np.asarray(etime) - np.asarray(actual)).mean())

    # distribution match, computed PER PLAYER then averaged - a pooled match can hide the fact
    # that every individual is wrong in a different direction
    w1, dsnap, dtail = [], [], []
    for u in np.unique(pid):
        m = pid == u
        w1.append(wasserstein1(sampled[m], actual[m]))
        dsnap.append((sampled[m] < 1.0).mean() - (actual[m] < 1.0).mean())
        dtail.append((sampled[m] > 10.0).mean() - (actual[m] > 10.0).mean())
    res["w1_per_player"] = float(np.mean(w1))
    # log-space W1: raw W1 is dominated by the crowded 0-3s region, so a model can look close
    # while getting the tail badly wrong. log1p spreads the scale and exposes that.
    res["w1_log_per_player"] = float(np.mean([
        wasserstein1(np.log1p(sampled[pid == u]), np.log1p(actual[pid == u]))
        for u in np.unique(pid)]))
    res["snap_err_pp"] = float(np.mean(np.abs(dsnap)) * 100)
    res["tail_err_pp"] = float(np.mean(np.abs(dtail)) * 100)
    res["snap_bias_pp"] = float(np.mean(dsnap) * 100)
    res["tail_bias_pp"] = float(np.mean(dtail) * 100)
    res["sampled_snap_pct"] = float((sampled < 1.0).mean() * 100)
    res["sampled_tail_pct"] = float((sampled > 10.0).mean() * 100)
    return res


def sample_from_probs(probs, clock, rng):
    """Vectorised clock-masked sampling: pick a bucket, then a time uniformly inside it."""
    lo = np.asarray(EDGES[:-1])
    hi = np.asarray([e if np.isfinite(e) else 60.0 for e in EDGES[1:]])
    p = probs.numpy().astype(np.float64)
    ok = lo[None, :] < np.asarray(clock)[:, None]
    p = p * ok
    s = p.sum(1, keepdims=True)
    dead = s[:, 0] <= 0
    p[dead] = 0.0
    p[dead, 0] = 1.0
    s = p.sum(1, keepdims=True)
    p = p / s
    cdf = p.cumsum(1)
    u = rng.random(len(p))[:, None]
    idx = (cdf < u).sum(1).clip(0, N_BUCKETS - 1)
    a, b = lo[idx], np.minimum(hi[idx], np.asarray(clock))
    b = np.maximum(b, a)
    return a + rng.random(len(a)) * (b - a)


# ---------------------------------------------------------------- training

def make_head(kind, in_dim):
    return {"mdn": MDNHead, "bucket": BucketHead,
            "hurdle": HurdleHead, "ordinal": OrderedLogitHead}[kind](in_dim)


def train_head(kind, loss_name, Xtr, ttr, ctr, balanced, steps, bs, lr, seed, log_every=0,
               cap=20.0):
    torch.manual_seed(seed)
    in_dim = Xtr.shape[1]
    head = make_head(kind, in_dim).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    ytr = torch.from_numpy(to_bucket(ttr.numpy())).long()

    bw = None
    if balanced:
        bw = inverse_frequency_weights(ytr, cap=cap)

    n = len(Xtr)
    # Features are small enough to live on the GPU for the whole sweep; moving them once
    # instead of per batch is the difference between hours and minutes across 27 runs.
    try:
        Xg, tg, cg, yg = (Xtr.to(DEV), ttr.to(DEV), ctr.to(DEV), ytr.to(DEV))
        bwg = bw.to(DEV) if bw is not None else None
    except RuntimeError:                     # not enough VRAM - fall back to per-batch transfer
        Xg, tg, cg, yg, bwg = Xtr, ttr, ctr, ytr, bw

    g = torch.Generator(device=Xg.device).manual_seed(seed)
    for step in range(steps):
        i = torch.randint(0, n, (bs,), device=Xg.device, generator=g)
        xb = Xg[i].to(DEV)
        if kind == "mdn":
            pi, mu, sg = head(xb)
            loss = mdn_nll_per_sample(pi, mu, sg, tg[i].to(DEV)).mean()
        else:
            logits = head(xb)
            w = sample_weights(yg[i], bwg).to(DEV) if bwg is not None else None
            loss = LOSSES[loss_name](logits, yg[i].to(DEV), cg[i].to(DEV), w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 3.5)
        opt.step()
        if log_every and step % log_every == 0:
            print(f"    step {step:>5} loss {loss.item():.4f}", flush=True)
    return head


@torch.no_grad()
def predict(head, kind, X, clock, rng, chunk=8192):
    """Returns (sampled_times, E[t], probs or None)."""
    if kind == "mdn":
        draws, means = [], []
        for s in range(0, len(X), chunk):
            pi, mu, sg = head(X[s:s + chunk].to(DEV))
            w = torch.softmax(pi, -1)
            sig = torch.nn.functional.softplus(sg) + 1e-3
            k = torch.multinomial(w, 1).squeeze(1)
            m = mu.gather(1, k[:, None]).squeeze(1)
            v = sig.gather(1, k[:, None]).squeeze(1)
            d = torch.exp(m + v * torch.randn_like(v))
            draws.append(d.cpu().numpy())
            means.append(torch.exp(mu + 0.5 * sig ** 2).mul(w).sum(-1).cpu().numpy())
        d = np.concatenate(draws)
        return np.minimum(d, clock.numpy()), np.concatenate(means), None

    probs = []
    for s in range(0, len(X), chunk):
        lg = head(X[s:s + chunk].to(DEV))
        probs.append(masked_probs(lg, clock[s:s + chunk].to(DEV)).cpu())
    probs = torch.cat(probs)
    return sample_from_probs(probs, clock.numpy(), rng), expected_time(probs).numpy(), probs


@torch.no_grad()
def val_metric(head, kind, X, t, c):
    """Selection metric, each method judged by its own native loss so none is handicapped.

    A shared learning rate would rig this: RPS gradients are far smaller in magnitude than
    cross-entropy's, so at one lr the ordinal loss simply trains slower and looks worse.
    """
    y = torch.from_numpy(to_bucket(t.numpy())).long()
    if kind == "mdn":
        tot, n = 0.0, 0
        for s in range(0, len(X), 8192):
            pi, mu, sg = head(X[s:s + 8192].to(DEV))
            tot += float(mdn_nll_per_sample(pi, mu, sg, t[s:s + 8192].to(DEV)).sum())
            n += len(pi)
        return tot / n
    probs = []
    for s in range(0, len(X), 8192):
        probs.append(masked_probs(head(X[s:s + 8192].to(DEV)), c[s:s + 8192].to(DEV)).cpu())
    probs = torch.cat(probs)
    p_cum = probs.cumsum(-1)
    idx = torch.arange(N_BUCKETS)[None, :]
    y_cum = (idx >= y[:, None]).float()
    return float(((p_cum - y_cum) ** 2).sum(-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/timehead")
    ap.add_argument("--out", default="checkpoints/timehead")
    ap.add_argument("--results", default="timehead_results.json")
    ap.add_argument("--test-players", type=int, default=20)
    ap.add_argument("--val-players", type=int, default=12)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lrs", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3])
    ap.add_argument("--caps", type=float, nargs="+", default=[3.0, 20.0])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    players = load_players(args.cache)
    rng0 = np.random.default_rng(args.seed)
    order = rng0.permutation(len(players))
    te_idx = order[:args.test_players]
    va_idx = order[args.test_players:args.test_players + args.val_players]
    tr_idx = order[args.test_players + args.val_players:]
    tr = [players[i] for i in tr_idx]
    va = [players[i] for i in va_idx]
    te = [players[i] for i in te_idx]
    print(f"{len(players)} players -> train {len(tr)} / val {len(va)} / test {len(te)}  "
          f"(DISJOINT humans; hyperparameters chosen on VAL, never on test)")

    variants = [
        ("mdn",            dict(kind="mdn",    loss=None,    diff=True,  bal=False)),
        ("bucket-ce",      dict(kind="bucket", loss="ce",    diff=True,  bal=False)),
        ("bucket-brier",   dict(kind="bucket", loss="brier", diff=True,  bal=False)),
        ("bucket-rps",     dict(kind="bucket", loss="rps",   diff=True,  bal=False)),
        ("bucket-rps-bal", dict(kind="bucket", loss="rps",   diff=True,  bal=True)),
        ("bucket-ce-bal",  dict(kind="bucket", loss="ce",    diff=True,  bal=True)),
        ("bucket-rps-nodiff", dict(kind="bucket", loss="rps", diff=False, bal=False)),
        ("hurdle-rps",     dict(kind="hurdle",  loss="rps",   diff=True,  bal=False)),
        ("hurdle-ce",      dict(kind="hurdle",  loss="ce",    diff=True,  bal=False)),
        ("ordinal-rps",    dict(kind="ordinal", loss="rps",   diff=True,  bal=False)),
    ]

    results = []
    for name, v in variants:
        t0 = time.time()
        Xtr, ttr, ctr, _ = stack(tr, use_diff=v["diff"])
        Xva, tva, cva, _ = stack(va, use_diff=v["diff"])
        Xte, tte, cte, pte = stack(te, use_diff=v["diff"])
        ckpt = os.path.join(args.out, f"{name}.pt")
        meta_p = os.path.join(args.out, f"{name}.json")

        if os.path.exists(ckpt):
            head = make_head(v["kind"], Xtr.shape[1]).to(DEV)
            head.load_state_dict(torch.load(ckpt, map_location=DEV))
            meta = json.load(open(meta_p, encoding="utf-8")) if os.path.exists(meta_p) else {}
            print(f"[{name}] loaded checkpoint (lr={meta.get('lr')} cap={meta.get('cap')})")
        else:
            grid = [(lr, cap) for lr in args.lrs
                    for cap in (args.caps if v["bal"] else [None])]
            print(f"[{name}] sweeping {len(grid)} setting(s) on {len(Xtr):,} train positions...")
            best = (None, float("inf"), None)
            for lr, cap in grid:
                h = train_head(v["kind"], v["loss"], Xtr, ttr, ctr, v["bal"],
                               args.steps, args.bs, lr, args.seed,
                               cap=(cap if cap is not None else 20.0))
                h.eval()
                m = val_metric(h, v["kind"], Xva, tva, cva)
                print(f"    lr={lr:<7g} cap={cap} -> val {m:.4f}")
                if m < best[1]:
                    best = (h, m, (lr, cap))
            head, _, (lr, cap) = best
            print(f"  selected lr={lr} cap={cap}")
            torch.save(head.state_dict(), ckpt)
            json.dump({"lr": lr, "cap": cap, "steps": args.steps},
                      open(meta_p, "w", encoding="utf-8"))
        head.eval()
        rng = np.random.default_rng(args.seed)
        sampled, etime, probs = predict(head, v["kind"], Xte, cte, rng)
        targets = torch.from_numpy(to_bucket(tte.numpy())).long()
        r = score(name, sampled, etime, tte.numpy(), pte.numpy(), probs, targets)
        r["train_s"] = round(time.time() - t0, 1)
        results.append(r)
        print(f"  -> RPS {r.get('rps', float('nan')):.4f} | W1 {r['w1_per_player']:.3f}s | "
              f"snap err {r['snap_err_pp']:.1f}pp | tail err {r['tail_err_pp']:.2f}pp | "
              f"r {r['pearson_Et']:.3f}")

    # ---- baselines that must be beaten -------------------------------------------
    _, ttr_all, ctr_all, _ = stack(tr)
    _, tte_all, cte_all, pte_all = stack(te)
    rng = np.random.default_rng(args.seed)

    draw = rng.choice(ttr_all.numpy(), size=len(tte_all), replace=True)
    draw = np.minimum(draw, cte_all.numpy())
    results.append(score("BASELINE-marginal", draw,
                         np.full(len(tte_all), float(ttr_all.mean())), tte_all.numpy(),
                         pte_all.numpy()))

    tr_cb = np.digitize(ctr_all.numpy(), [5, 15, 30, 60, 120])
    te_cb = np.digitize(cte_all.numpy(), [5, 15, 30, 60, 120])
    draw2 = np.zeros(len(tte_all))
    et2 = np.zeros(len(tte_all))
    for b in np.unique(te_cb):
        pool = ttr_all.numpy()[tr_cb == b]
        if len(pool) == 0:
            pool = ttr_all.numpy()
        m = te_cb == b
        draw2[m] = rng.choice(pool, size=int(m.sum()), replace=True)
        et2[m] = pool.mean()
    draw2 = np.minimum(draw2, cte_all.numpy())
    results.append(score("BASELINE-clock-marginal", draw2, et2, tte_all.numpy(), pte_all.numpy()))

    with open(args.results, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'variant':<24} {'RPS':>7} {'W1(s)':>7} {'snapErr':>8} {'tailErr':>8} "
          f"{'r':>7} {'MAE':>6}")
    for r in sorted(results, key=lambda x: x.get("rps", 9e9)):
        print(f"{r['variant']:<24} {r.get('rps', float('nan')):>7.4f} {r['w1_per_player']:>7.3f} "
              f"{r['snap_err_pp']:>7.1f}pp {r['tail_err_pp']:>7.2f}pp "
              f"{r['pearson_Et']:>7.3f} {r['mae_Et']:>6.2f}")
    print(f"\nwrote {args.results}")


if __name__ == "__main__":
    main()
