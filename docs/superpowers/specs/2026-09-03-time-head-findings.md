# Findings: what actually matches human think-time

**Date:** 2026-09-03
**Status:** Measured, committed, reproducible
**Bears on:** `2026-08-26-bigger-clone-base-time-elo-design.md` (sections 3, 4.2, 5, 9)

Overnight bake-off of every candidate think-time head. Three of the conclusions overturn things
we believed going in, including two of my own recommendations from earlier the same day.

---

## Setup

- **80 chess.com players**, 317k train / 86k test positions, all 3+0 with clocks.
- **Split by PLAYER**: 48 train / 12 val / 20 test, disjoint humans. Positions inside a game are
  not independent - the temporal vector carries that player's own last-5 think-times - so a
  position- or game-level split would let a head memorise a person's tempo.
- Every head trains on **identical frozen features** from `base_300k_best.pt`, so differences are
  the head and nothing else.
- Hyperparameters chosen on **val**, never on test. Each method judged by its own native loss
  during selection: a shared learning rate would rig the comparison, because RPS gradients are far
  smaller in magnitude than cross-entropy's.
- **Null baselines throughout.** Any head that cannot beat "sample from a histogram" has learned
  nothing, however good its distribution match looks.

Reproduce: `scripts/timehead_features.py` -> `timehead_bakeoff.py` -> `timehead_readout.py` ->
`timehead_calibrate.py` -> `timehead_perplayer.py`.

---

## 1. The readout dominates. The head barely matters.

Real: **snap(<1s) 17.2%, tail(>10s) 5.54%**

| readout | W1 per player | snap% | tail% | r |
|---|---|---|---|---|
| **sampled** | 0.375 - 0.402 | 19-21% | 4.9-6.1% | ~0.31 |
| **E[t]** | 0.962 - 1.050 | 6.8-9.9% | 0.8-1.5% | ~0.55 |

Sampling versus E[t] changes distribution match by **2.6x**. Head family and loss change it by
~7%, and the sampled 95% CIs overlap across every head, so that ranking is not significant.

**This overturns section 3 of the 08-26 spec.** That spec adopts a bucket head because the MDN
"hedges into the middle, under-snaps and kills the tail". But the bake-off behind that decision
compared the MDN's **E[t]** against the bucket head's **sampled** draws - two changes at once,
credited to one of them. Sampled, the MDN ties the best bucket head (W1 0.375). And under E[t]
*every* bucket head collapses exactly the same way: 7-10% snap against a real 17.2%, ~1% tail
against a real 5.5%.

The hedge is a property of taking a conditional mean, not of the mixture-of-log-normals.

ChessMimic (arXiv 2606.04473) uses E[t] at inference and reports r=0.41, below Allie's plain
scalar head at r=0.70. This is very likely the same effect in the published work.

## 2. Two of my own recommendations were wrong

Both came from the literature reading earlier that day, and both are empirically inert or harmful.

- **RPS instead of CE/Brier: no benefit.** CE 1.2632, Brier 1.2650, RPS 1.2677. RPS does not even
  win its own metric. The ordinal argument is correct in principle - CE and Brier really do score
  a 3s prediction identically to a 40s one when the truth is 4s - and it makes no measurable
  difference here.
- **Inverse-frequency balancing (Balanced DRPS): actively harmful.** W1 0.655-0.690 against 0.375
  unbalanced, consistent across every learning rate and both caps on validation.

Keep cross-entropy. It is the simplest and marginally the best.

## 3. Distribution-match metrics are gameable by a lookup table

The population head under-disperses badly. It ranks people well (per-player snap rate r=0.77,
surviving a time-pressure control at clock>120s: r=0.69) but captures only **39% of the spread** -
real 5.5-33.8%, predicted 12.4-26.4%. Everyone's clone drifts toward the average human, which is
the opposite of what clonefish sells.

Fitting **two scalars per player** (snap-logit bias + temperature) on their earlier games and
scoring on their later games fixes the scale:

| | W1 | snap slope |
|---|---|---|
| population head | 0.492 | 0.32 |
| + 2 per-player scalars | **0.315** (-36%) | **0.90** |

Better for 15/18 players. Slope 0.90 means it now tracks the person rather than the mean.

**And then the null deflates it.** Against simply resampling that player's own past think-times,
ignoring the position entirely:

| | W1 | r |
|---|---|---|
| calibrated model | 0.315 | **0.323** |
| per-player marginal null | 0.336 | 0.006 |
| model wins W1 for | 9/18 players | - |

On distribution match the model beats a lookup table for **half the players** - a coin flip. The
36% gain is almost entirely "learn the person's marginal histogram", which needs no model.

What the model actually contributes is the **conditional** part: knowing which positions deserve
time. The null cannot do that at all (r=0.006).

## 4. What this means

**Change the readout, not the head.** Sampling is the entire measured win. That is one line at
inference and does not require the architecture change in section 4.2.

**Section 9.2's acceptance criteria cannot stand alone.** Judging the time head by "snap-rate
within +-5pp and tail-rate within +-2pp" is passable by a lookup table with zero positional
understanding. Report conditional skill alongside: **r at E[t]** for comparability with Allie and
ChessMimic, and distribution match from **sampled** draws, with the per-player marginal null
printed next to both.

**For clonefish, timing splits in two.** Distribution realism is cheap - calibrate to the user's
own histogram, two scalars, no training. Position-aware timing is what the model buys, and it is
the part worth improving.

## 5. Answered: the base is the ceiling

Same head, fixed test humans, growing number of training humans:

| players | positions | r@E[t] | change |
|---|---|---|---|
| 12 | 77,493 | 0.5425 | |
| 24 | 144,014 | 0.5611 | +0.0186 |
| 48 | 289,741 | 0.5697 | +0.0086 |
| 96 | 572,639 | 0.5744 | +0.0047 |
| 192 | 1,161,161 | 0.5748 | +0.0004 |
| 288 | 1,781,465 | 0.5753 | +0.0005 |

Tripling the humans past 96 moves conditional skill by **+0.0009**. RPS is flat too. The time
head saturates at roughly 100 players / 0.5M positions.

So the ceiling is the **frozen 19.5M base representation** - not the head, not the loss, not the
amount of timing data. Three consequences:

- **More corpus does not buy better timing.** It may still buy move-matching, which is a separate
  question, but the time head stops learning at ~0.5M positions.
- **No time-head architecture can help either**, consistent with all eight heads landing within 7%
  of each other.
- The only remaining lever for position-aware timing is **a better backbone**.

For reference this ceiling (r 0.575) sits between ChessMimic's 0.41 and Allie's 0.70 - and
Allie's backbone is far larger than ours.


## 6. Clock overrun: the one real defect, and a free fix

Chasing an apparent "+2.5s/game systematic drift" found that the drift itself was an artifact -
the drift scripts re-derived their test set with `permutation(len(players))` after the cache grew
80 -> 384, so 4 of 20 "held-out" players were training players. On the correct held-out set the
drift is **+0.1s/game**. The tell was a human snap rate of 24.9% where this document reports 17.2%.
The split is now pinned by name in `cache/timehead_split.json`.

What survives on clean data is more serious:

| mode | total/game | drift | **FLAG %** | snap% | tail% | W1 |
|---|---|---|---|---|---|---|
| human | 121.8s | | 0.00% | 17.2% | 5.54% | |
| teacher-forced | 123.2s | +1.4s | **12.78%** | 18.7% | 5.61% | 0.068 |
| self-clocked | 120.5s | -1.3s | **0.08%** | 20.9% | 5.53% | 0.073 |
| self+pace | 113.0s | -8.8s | 0.00% | 18.9% | 4.00% | 0.278 |

**In 12.8% of games the model's think-times sum past the 180s it has.** The per-move clock mask
stops any single move exceeding the clock, but nothing tracks cumulative spend, so independent
draws from a heavy tail occasionally run away.

**Humans budget; the model does not.** Controlling for ply count and player identity,
corr(first-half spend, second-half spend) is **-0.170** for humans and **+0.004** for the model. A
human who spends lavishly early spends 8.1s less later; the model pulls back only 4.1s, and that
is borrowed from reading the human's depleted clock rather than learned.

**The fix is free:** mask against the model's own remaining clock instead of the human's. Flag rate
12.78% -> 0.08%, drift -1.3s, distribution match unchanged and the tail lands closer to human. Same
category as "sample, don't average" - a readout change, not an architecture change. The explicit
pace prior overcorrects and kills the tail; do not use it.

This makes **two** readout fixes that together are worth more than any head change measured here:
sample rather than average, and clock yourself rather than trusting the opponent's clock.

## 7. Honest limits

- 80 players, 20 in test, one site (chess.com), one time control (3+0).
- All heads sit on frozen features from the existing 19.5M base. A different backbone could
  reorder the head comparison, though it would not change the readout finding, which is a
  property of conditional means rather than of any particular model.
- The per-player calibration used ~70% of each player's games. Whether two scalars suffice for a
  user with 50 games is untested.
- The clock-overrun fix is measured with teacher-forced FEATURES: only the mask is
  self-consistent, since the cached pooled vectors still encode the human's real clock. A
  full free-running simulation needs the base model re-run per ply.
- Sampled correlation (~0.31) is necessarily below E[t] correlation (~0.55): a single draw is
  noisier than a conditional mean by construction. That is arithmetic, not a defect, and it is why
  both must be reported.
