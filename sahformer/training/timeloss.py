"""Losses for the bucketed think-time head.

The buckets are ORDERED. Cross-entropy and plain Brier both ignore that: with the truth at 4s,
predicting 3s and predicting 40s cost the same. The Ranked Probability Score fixes it by
scoring the CUMULATIVE distribution, so error is paid in proportion to ordinal distance. RPS is
a proper scoring rule and is the discrete analogue of CRPS; for K=2 it reduces to Brier.

Think-time is also badly imbalanced - ChessMimic reports ~86% of blitz moves under 27s, and the
>10s tail measured here is ~5%. An unweighted ordinal loss drifts toward the majority buckets,
which is exactly the MDN failure this head exists to fix (under-snaps AND kills the tail). The
`weights` argument carries the inverse-frequency correction (Balanced DRPS, arXiv 2507.00736).

Clock-masking follows ChessMimic: buckets holding a time the player could not afford are
masked BEFORE scoring, so probability mass is never spent on impossible outcomes.
"""
import torch
import torch.nn.functional as F

from sahformer.model.timebuckets import N_BUCKETS, clock_mask


def masked_probs(logits, remaining_clock=None):
    """Softmax over only the buckets the clock allows."""
    if remaining_clock is None:
        return torch.softmax(logits, dim=-1)
    mask = clock_mask(remaining_clock, device=logits.device)
    lg = logits.masked_fill(~mask, float("-inf"))
    allnone = ~mask.any(dim=-1, keepdim=True)          # clock under 1s: fall back to bucket 0
    lg = torch.where(allnone, logits, lg)
    return torch.softmax(lg, dim=-1)


def _onehot(target, k=N_BUCKETS):
    return F.one_hot(target.long().clamp(0, k - 1), num_classes=k).to(torch.float32)


def _reduce(per_sample, weights):
    if weights is None:
        return per_sample.mean()
    w = weights.to(per_sample.dtype)
    return (per_sample * w).sum() / w.sum().clamp_min(1e-8)


def ce_loss(logits, target, remaining_clock=None, weights=None):
    """Cross-entropy. Ordinal-blind; kept as the baseline the 08-26 spec proposed."""
    p = masked_probs(logits, remaining_clock)
    ll = torch.log(p.gather(-1, target.long().clamp(0, N_BUCKETS - 1)[:, None]).squeeze(-1) + 1e-9)
    return _reduce(-ll, weights)


def brier_loss(logits, target, remaining_clock=None, weights=None):
    """Masked Brier - ChessMimic's choice. Proper, but ordinal-blind."""
    p = masked_probs(logits, remaining_clock)
    per = ((p - _onehot(target)) ** 2).sum(-1)
    return _reduce(per, weights)


def rps_loss(logits, target, remaining_clock=None, weights=None):
    """Ranked Probability Score: Brier on the CUMULATIVE distribution. Ordinal-aware."""
    p = masked_probs(logits, remaining_clock)
    p_cum = p.cumsum(-1)
    idx = torch.arange(N_BUCKETS, device=logits.device)[None, :]
    y_cum = (idx >= target.long().clamp(0, N_BUCKETS - 1)[:, None]).to(p.dtype)
    per = ((p_cum - y_cum) ** 2).sum(-1)
    return _reduce(per, weights)


LOSSES = {"ce": ce_loss, "brier": brier_loss, "rps": rps_loss}


def inverse_frequency_weights(targets, k=N_BUCKETS, smoothing=1.0, cap=None):
    """Per-BUCKET weights ~ 1/frequency over the training set. Index with `sample_weights`.

    Buckets never observed get weight 0 and are excluded from the normalisation. Giving them
    1/(0+smoothing) would hand the RAREST possible weight to buckets that contribute nothing,
    dragging the mean up and shrinking every real weight.

    `cap` is the maximum ratio between the largest and smallest weight. Without it, a bucket
    seen a handful of times gets a weight thousands of times the majority's and its few samples
    dominate the gradient.
    """
    counts = torch.bincount(targets.long().clamp(0, k - 1), minlength=k).to(torch.float32)
    seen = counts > 0
    w = torch.zeros(k, dtype=torch.float32)
    if not seen.any():
        return w
    w[seen] = 1.0 / (counts[seen] + smoothing)
    if cap is not None:
        w[seen] = w[seen].clamp(max=float(cap) * w[seen].min())
    w[seen] = w[seen] / w[seen].mean()
    return w


def sample_weights(targets, bucket_w):
    """Per-bucket weights -> per-sample weights, which is what the losses take."""
    return bucket_w.to(targets.device)[targets.long().clamp(0, bucket_w.numel() - 1)]
