"""Discretisation of think-time into ordered buckets, plus a clock-masked sampler.

Layout (design 2026-08-26 s4.2, matching ChessMimic arXiv 2606.04473): 27 one-second bins
covering [0,27) - which the paper reports is ~86% of blitz moves - then three widening tail
bins [27,32), [32,40), [40,inf). K=30.

The buckets are ORDINAL, and that matters for the loss: predicting 3s when the truth is 4s is
a near miss, predicting 40s is not. Cross-entropy and plain Brier score both identically.
See `sahformer/training/timeloss.py`.

Play must SAMPLE from the distribution, not take E[t]. Collapsing to the mean is what turns a
snap-or-tank distribution back into the hedged middle the MDN produced - and it is what
ChessMimic itself does at inference, which is the likeliest reason their reported correlation
(r=0.41) sits below Allie's scalar head (r=0.70).
"""
import numpy as np
import torch

EDGES = tuple([float(i) for i in range(28)] + [32.0, 40.0, float("inf")])
N_BUCKETS = len(EDGES) - 1

# Representative time for each bucket. Open-ended tail bucket gets a finite stand-in so that
# E[t] is computable; it is deliberately not huge, since a 3+0 game caps think-time anyway.
CENTERS = tuple(
    [i + 0.5 for i in range(27)] + [29.5, 36.0, 48.0]
)

assert N_BUCKETS == 30 and len(CENTERS) == N_BUCKETS


def to_bucket(t):
    """Seconds -> bucket index. Vectorised over arrays, scalar-safe."""
    edges = np.asarray(EDGES[1:-1])          # interior edges: 1..27, 32, 40
    idx = np.searchsorted(edges, np.asarray(t, dtype=np.float64), side="right")
    return int(idx) if np.isscalar(t) or np.ndim(t) == 0 else idx.astype(np.int64)


def bucket_upper(i):
    return EDGES[i + 1]


def clock_mask(remaining_clock, device=None):
    """Bool mask of buckets a player could actually still spend, given the clock.

    A bucket is allowed if its LOWER edge is under the remaining clock, so the sampler can
    never pick a time that would flag the player.
    """
    lower = torch.tensor(EDGES[:-1], dtype=torch.float32, device=device)
    rc = torch.as_tensor(remaining_clock, dtype=torch.float32, device=device)
    if rc.ndim == 0:
        return lower < rc
    return lower[None, :] < rc[:, None]


def expected_time(probs):
    """E[t] = sum P(bucket) * center(bucket). For reporting/comparability only - not for play."""
    c = torch.tensor(CENTERS, dtype=probs.dtype, device=probs.device)
    return (probs * c).sum(-1)


def sample_think_time(logits, remaining_clock, rng=None, temperature=1.0):
    """Draw one think-time (seconds) from a bucket distribution, respecting the clock.

    Masks unreachable buckets, samples a bucket, then draws uniformly WITHIN the bucket so the
    result is continuous rather than piling up on bucket centres. The open tail bucket draws
    from [40, min(clock, 60)).
    """
    rng = rng or np.random.default_rng()
    lg = logits.detach().float().flatten()
    mask = clock_mask(float(remaining_clock), device=lg.device)
    lg = lg.masked_fill(~mask, float("-inf"))
    if not torch.isfinite(lg).any():          # clock under 1s: nothing legal, snap
        return float(max(min(remaining_clock, 0.1), 0.0))
    p = torch.softmax(lg / max(temperature, 1e-6), dim=-1).cpu().numpy()
    p = np.nan_to_num(p, nan=0.0)
    if p.sum() <= 0:
        return float(max(min(remaining_clock, 0.1), 0.0))
    p = p / p.sum()
    i = int(rng.choice(N_BUCKETS, p=p))
    lo = EDGES[i]
    hi = EDGES[i + 1] if np.isfinite(EDGES[i + 1]) else 60.0
    hi = min(hi, float(remaining_clock))
    if hi <= lo:
        return float(max(lo, 0.0))
    return float(rng.uniform(lo, hi))
