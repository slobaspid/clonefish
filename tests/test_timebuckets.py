"""Bucket layout, clock-masked sampler, and the ordinal losses."""
import numpy as np
import torch

from sahformer.model.timebuckets import (
    CENTERS,
    EDGES,
    N_BUCKETS,
    clock_mask,
    expected_time,
    sample_think_time,
    to_bucket,
)
from sahformer.training.timeloss import (
    brier_loss,
    ce_loss,
    inverse_frequency_weights,
    masked_probs,
    rps_loss,
    sample_weights,
)


def test_bucket_layout():
    assert N_BUCKETS == 30
    assert len(EDGES) == N_BUCKETS + 1
    assert EDGES[0] == 0.0 and EDGES[-1] == float("inf")
    assert to_bucket(0.2) == 0            # snap bin [0,1)
    assert to_bucket(0.99) == 0
    assert to_bucket(1.0) == 1
    assert to_bucket(26.5) == 26
    assert to_bucket(27.5) == 27          # first wide bin [27,32)
    assert to_bucket(35.0) == 28          # [32,40)
    assert to_bucket(50.0) == N_BUCKETS - 1   # tail 40+


def test_to_bucket_vectorised_matches_scalar():
    ts = np.array([0.0, 0.5, 1.0, 5.4, 26.99, 27.0, 31.9, 32.0, 39.9, 40.0, 300.0])
    vec = to_bucket(ts)
    assert vec.tolist() == [to_bucket(float(t)) for t in ts]


def test_centers_lie_inside_their_bucket():
    for i, c in enumerate(CENTERS):
        assert EDGES[i] <= c
        if np.isfinite(EDGES[i + 1]):
            assert c < EDGES[i + 1]


# --- sampler ---------------------------------------------------------------------

def test_sampler_never_exceeds_clock():
    logits = torch.zeros(N_BUCKETS)
    logits[-1] = 10.0                      # begging to tank
    rng = np.random.default_rng(0)
    for _ in range(300):
        t = sample_think_time(logits, remaining_clock=3.0, rng=rng)
        assert 0.0 <= t <= 3.0


def test_sampler_can_snap_and_tank():
    rng = np.random.default_rng(0)
    snap = torch.full((N_BUCKETS,), -10.0); snap[0] = 10.0
    tank = torch.full((N_BUCKETS,), -10.0); tank[-1] = 10.0
    s = [sample_think_time(snap, 180.0, rng) for _ in range(200)]
    t = [sample_think_time(tank, 180.0, rng) for _ in range(200)]
    assert np.mean(s) < 1.0
    assert np.mean(t) > 20.0


def test_sampler_survives_a_dead_clock():
    logits = torch.zeros(N_BUCKETS)
    t = sample_think_time(logits, remaining_clock=0.3, rng=np.random.default_rng(0))
    assert 0.0 <= t <= 0.3                 # must not flag, must not crash


def test_sampled_distribution_tracks_the_logits():
    """Over many draws the sampled bucket mix should match the softmax it came from."""
    rng = np.random.default_rng(0)
    logits = torch.full((N_BUCKETS,), -8.0)
    logits[0] = 2.0      # snap
    logits[5] = 1.0      # ~5s
    logits[-1] = 0.5     # tank
    want = torch.softmax(logits, -1).numpy()
    draws = np.array([to_bucket(sample_think_time(logits, 180.0, rng)) for _ in range(4000)])
    got = np.bincount(draws, minlength=N_BUCKETS) / len(draws)
    for i in (0, 5, N_BUCKETS - 1):
        assert abs(got[i] - want[i]) < 0.05


# --- clock masking ---------------------------------------------------------------

def test_clock_mask_blocks_unaffordable_buckets():
    m = clock_mask(torch.tensor([3.0, 180.0]))
    assert m.shape == (2, N_BUCKETS)
    assert m[0, 0] and m[0, 2] and not m[0, 3]     # 3s left -> bucket [3,4) is unaffordable
    assert m[1].all()


def test_masked_probs_put_zero_mass_on_impossible_buckets():
    logits = torch.zeros(2, N_BUCKETS)
    p = masked_probs(logits, torch.tensor([3.0, 180.0]))
    assert torch.allclose(p.sum(-1), torch.ones(2), atol=1e-5)
    assert p[0, 5:].sum().item() < 1e-6


# --- the ordinal point -----------------------------------------------------------

def test_rps_penalises_by_ordinal_distance_but_ce_and_brier_do_not():
    """The whole reason for RPS: a near miss must cost less than a far miss."""
    target = torch.tensor([4])
    near = torch.full((1, N_BUCKETS), -20.0); near[0, 3] = 20.0    # off by one bucket
    far = torch.full((1, N_BUCKETS), -20.0); far[0, 29] = 20.0     # off by 25 buckets

    assert rps_loss(near, target) < rps_loss(far, target)
    # CE and Brier cannot tell them apart - both are confidently wrong by the same amount
    assert abs(ce_loss(near, target).item() - ce_loss(far, target).item()) < 1e-3
    assert abs(brier_loss(near, target).item() - brier_loss(far, target).item()) < 1e-3


def test_all_losses_are_minimised_by_the_truth():
    target = torch.tensor([7])
    right = torch.full((1, N_BUCKETS), -20.0); right[0, 7] = 20.0
    wrong = torch.full((1, N_BUCKETS), -20.0); wrong[0, 20] = 20.0
    for fn in (ce_loss, brier_loss, rps_loss):
        assert fn(right, target) < fn(wrong, target)
        assert fn(right, target).item() >= 0.0


def test_losses_are_finite_on_a_uniform_prediction():
    logits = torch.zeros(8, N_BUCKETS)
    target = torch.randint(0, N_BUCKETS, (8,))
    clock = torch.full((8,), 60.0)
    for fn in (ce_loss, brier_loss, rps_loss):
        v = fn(logits, target, clock)
        assert torch.isfinite(v)


# --- balancing -------------------------------------------------------------------

def test_inverse_frequency_upweights_rare_buckets():
    targets = torch.tensor([0] * 900 + [1] * 90 + [29] * 10)
    w = inverse_frequency_weights(targets)
    assert w[29] > w[1] > w[0]
    seen = w > 0
    assert abs(w[seen].mean().item() - 1.0) < 1e-5      # scale preserved over SEEN buckets


def test_unseen_buckets_get_zero_weight_not_the_highest():
    """A bucket with no samples must not be handed the rarest-bucket weight."""
    targets = torch.tensor([0] * 900 + [29] * 10)
    w = inverse_frequency_weights(targets)
    assert w[5].item() == 0.0                  # never observed
    assert w[29] == w.max()                    # genuinely rare bucket is the top weight


def test_weight_cap_bounds_the_upweighting():
    targets = torch.tensor([0] * 9999 + [29])
    uncapped = inverse_frequency_weights(targets)
    capped = inverse_frequency_weights(targets, cap=10.0)
    assert capped.max() < uncapped.max()
    seen = capped > 0
    assert capped[seen].max() / capped[seen].min() <= 10.0 + 1e-4


def test_sample_weights_indexes_per_sample():
    targets = torch.tensor([0] * 900 + [29] * 10)
    bw = inverse_frequency_weights(targets)
    sw = sample_weights(targets, bw)
    assert sw.shape == targets.shape
    assert sw[0] == bw[0] and sw[-1] == bw[29]


def test_balanced_loss_moves_the_optimum_toward_the_rare_bucket():
    """End-to-end: with the same data, balancing must change what the loss prefers."""
    targets = torch.cat([torch.zeros(200, dtype=torch.long), torch.full((5,), 29)])
    clock = torch.full((205,), 180.0)
    bw = inverse_frequency_weights(targets, cap=20.0)
    sw = sample_weights(targets, bw)

    majority = torch.full((205, N_BUCKETS), -5.0); majority[:, 0] = 5.0   # predict snap always
    hedged = torch.full((205, N_BUCKETS), -5.0)
    hedged[:, 0] = 3.0; hedged[:, 29] = 3.0                               # covers the tail too

    # unweighted, always-snap wins because the tail is only 5 of 205 samples
    assert rps_loss(majority, targets, clock) < rps_loss(hedged, targets, clock)
    # weighted, ignoring the tail is punished
    assert rps_loss(hedged, targets, clock, sw) < rps_loss(majority, targets, clock, sw)


def test_expected_time_is_a_weighted_average_of_centers():
    p = torch.zeros(1, N_BUCKETS); p[0, 0] = 0.5; p[0, 10] = 0.5
    got = expected_time(p).item()
    assert abs(got - (CENTERS[0] + CENTERS[10]) / 2) < 1e-5
