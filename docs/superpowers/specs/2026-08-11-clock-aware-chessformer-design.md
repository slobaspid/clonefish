# Clock-Aware Chessformer for 3+0 Blitz — Design

**Date:** 2026-08-11
**Status:** Approved design, pre-implementation

## One-line summary

A clock-aware extension of Maia-3 / Chessformer that models the **average human of a
given Elo** playing **3+0 blitz** — predicting not just the move, but also **how long a
human would think** — by injecting temporal context through **FiLM + time-conditioned
Geometric Attention Bias (GAB)**.

## Goal & success criteria

**Primary goal: human *feel*, not benchmark accuracy.** The model should manage the clock
like a real 3+0 player — blitz openings, spend time on hard positions, degrade under time
pressure, premove in known lines, and scramble on the flag.

Move-matching accuracy is a **guard rail, not the objective**: the clock-aware model must
not get *dramatically* worse than a plain baseline, but small accuracy differences are
acceptable — behavioral and temporal realism is what we optimize for.

Non-goals:
- Not modeling a specific individual across games — we model an **Elo cohort**, like Maia.
- Not other time controls — **3+0 only** (fixed 180s base, 0 increment) for the first model.
- Not chasing SOTA leaderboard numbers.

## Background: what Maia-3 / Chessformer is

From the Chessformer paper (ICLR 2026) and the CSSLab/maia3 repo:

- **Board-as-tokens**: 64 squares = 64 tokens, each a 12-dim piece one-hot, plus **7
  historical positions** stacked in the channel dimension (gives move-sequence context).
- **Geometric Attention Bias (GAB)**: a learned, board-state-dependent bias added to
  attention logits (per-head 64×64 matrices). The paper's core novelty.
- **Rating conditioning**: two 128-dim skill embeddings (one per player), interpolated
  between a learnable weak (Elo 0) and strong (Elo 5000) endpoint, prepended as tokens.
- **Heads**: source→destination attention policy (64×64 + promotion bias) and a
  mean-pooled Win/Draw/Loss value head.
- **Sizes**: 5M / 23M / 79M — move-match **55.4 / 56.6 / 57.1%** on the Allie test set (paper Table 1).
  Trained on rating-balanced Lichess **blitz games from Jan 2023 to Jul 2025** (31 months), resampled so
  every skill level is equally represented, 32 positions sampled per game. The paper states no total.
  **CORRECTION (2026-09-02):** this line previously read "trained on ~884k rating-balanced Lichess blitz
  positions". That was wrong — 884,049 is the **Allie TEST set** (Zhang et al. 2025: Lichess blitz 2022,
  first 10 moves and time-pressure positions removed), the shared eval benchmark, not training data. The
  error made our corpus look ~90x larger than Maia-3's when it is very likely smaller.
- **Crucially for us**: Maia-3 **deliberately discards time** — it removed time-pressure
  positions and only carried a time-pressure flag "for infrastructure compatibility without
  performance impact." **No one has modeled clock-aware human play in this architecture.**
  That gap is this project.

## Compute plan

- **Train** the **5M** model on a **Colab free T4 (~16GB)**. Small is fine — Maia-3 shows 5M
  is respectable. Checkpoint to Google Drive constantly (Colab disconnects/time-outs).
- **Serve/play/analyze** on the user's Xeon box (32GB RAM) — even 79M runs fine on CPU for
  *inference* (~320MB fp32 weights). Never *train* 79M on CPU (compute-bound, weeks/months).
- Later: rent a bigger GPU for a day to scale 5M → 23M/79M if the idea proves out.

---

## Section 1 — Data pipeline

- **Source**: Lichess open database, filtered to **3+0 rated blitz** with `%clk` clock
  annotations. Data is abundant (tens of millions of games/month) — compute/disk is the
  bottleneck, not data.
- **Per-position features:**
  - Board + 7 history planes (Maia-3 style).
  - Both players' Elo → skill embeddings (unchanged from Maia-3).
  - **Temporal feature block** (raw, in-game observations only):
    - log clocks for both sides (`log(t+1)`, normalized), plus clock **differential**
    - **last-k think-times** (k≈3–5) of the current game, raw — lets the model derive
      in-game rhythm itself
    - **premove flags** on recent moves (think-time < ~100ms)
    - **time-pressure buckets / RBF features** near zero (`<2s`, `<5s`, `<10s`, `>30s`) so
      the net can represent nonlinear panic thresholds
    - burn rate (how fast each side spends time)
- **Targets per position:** (a) move played, (b) game result W/D/L, (c) **this move's
  think-time** stored as a **raw float (seconds)** — the MDN head consumes it directly in
  log-space; no bucketing.
- **Key difference from Maia-3**: we **KEEP** time-pressure and flag-scramble positions —
  that's the whole point.
- **Balancing**: equalize 22 Elo bins (Maia-3 style). Start ~1M positions for a Colab run.

### Design principle: get the most signal out of time

Raw "seconds left" is the weakest use of time. Higher-signal ideas, folded into the block above:

1. **The Elo embedding owns the pace baseline.** A 1200 and a 2200 have different natural
   tempos, panic thresholds, and premove habits. We do NOT hand-engineer a per-individual
   baseline (no persistent identity). Instead we feed **absolute in-game observations** and
   let the model compare them against its **Elo-conditioned expectation** internally — the
   relativity is emergent, not hand-coded.
2. **Rhythm over snapshots** — feed the recent *sequence* of think-times, not just the last.
3. **Nonlinear near zero** — log + buckets/RBF, because human play has hard cliffs (<3s ≈ reflex).
4. **Premove detection** — sub-100ms is qualitatively different (decided before opponent moved).
5. **No leakage** — this move's think-time is a **target**; only think-times of moves 1…n−1
   may be inputs.

---

## Section 2 — Model architecture

**Decision (2026-08-11):** after reading Maia-3's actual source (`maia3/model_registry.py`,
`models.py`, `dataset.py`), we build the backbone **faithful to their real 5M architecture**,
and keep **only the time layer as ours**. (Their code already ships basic time inputs +
a scalar "ponder" head; our time handling is deliberately richer and different.) We reimplement
from understanding — their code is **AGPL-3.0**, so no verbatim porting.

### Faithful Maia-3 5M backbone (their exact config)

From `model_registry.py` BASE + 5M spec:
- **Input**: 64 tokens. Each token's channels = `12 × history(8) + 2 × dim_emb(128)` = **352**:
  the 12-plane board for the current + 7 past positions (earliest repeated if missing),
  **concatenated with** the two interpolated skill embeddings (self + opponent), one 128-vec
  each, broadcast to every square. Then `Linear(352 → dim_vit=256)`.
- **`use_absolute_pe = False`** — GAB is the only positional signal.
- **Trunk**: `num_blocks=8`, `dim_vit=256`, `num_heads=8`, `mlp_ratio=2` (FFN 512),
  **RMSNorm, post-norm**, GELU, `omit_qkv_biases=True`, dropout 0.
- **GAB, per-layer, inside each attention block** (5M variant, `gab_per_square_dim=0` → avg-pool):
  mean-pool the layer's 64 tokens → `Linear(256→64)`+GELU+LN → `Linear(64→heads·64)`+GELU+LN →
  reshape `(B, heads, gab_gen_size=64)` → a **shared-across-layers** `gab_weight (4096, 64)`
  einsum → `(B, heads, 64, 64)` added to that layer's attention logits.
- **Skill interpolation**: `e_k = γ·e_weak + (1−γ)·e_strong`, `γ = (5000−k)/5000`, dim_emb 128.
- **Policy head**: `proj_from`, `proj_to` = `Linear(256 → head_hid_dim=256, bias=False)`;
  bilinear `einsum/√256 → (B,64,64)` → 4096 move logits, **plus 256 promotion logits**
  (8 files × 8 target files × 4 pieces) = **4352 total**.
- **Value head**: norm → mean-pool → `Linear(256→256)`+ReLU → `Linear(256→3)` (W/D/L).

The clock-blind version of this (no time layer) is the **ablation baseline**.

### Our time layer (kept different — this is the contribution)

1. **Temporal context** `t`: the Section-1 21-dim block → `TemporalEncoder` MLP → 128-dim.
   (Richer than their 4 raw scalars.)
2. **Time-conditioned GAB**: concatenate `t` into each layer's GAB generator input, so the
   geometric attention becomes clock-aware — under pressure it can narrow onto forcing moves.
3. **FiLM** on each block: `t` → per-layer per-channel `(γ, β)`, near-identity at init.
4. **MDN think-time head** (replaces their scalar ponder head): mean-pool (+ `t`) → MLP →
   parameters of a mixture of **log-normals** (M=3: weights `π`, means `μ`, spreads `σ` in
   log-time space). Continuous, multimodal, samplable — represents "60% premove ~0.05s,
   40% real think ~4s" instead of collapsing to a meaningless mean.
5. We **train on time-pressure positions** (Maia-3 discards them).

New parameters vs. their backbone: temporal MLP, per-layer time-conditioning of GAB, FiLM
generator, and the MDN head.

---

## Section 3 — Losses & training

**Combined loss:**
- **Policy** — cross-entropy on move, weight **1.0** (main objective).
- **Value** — cross-entropy on W/D/L, weight **~0.1** (Maia-3 default).
- **Think-time** — **negative log-likelihood** of the observed think-time under the predicted
  log-normal mixture (MDN loss), weight **~0.2–0.3** (tuned: high enough to learn time, low
  enough not to hurt move accuracy). Numerical stability: compute NLL in log-space with a
  log-sum-exp over components; floor `σ` with a softplus + epsilon.

Move-accuracy is watched as a guard metric; if the time loss tanks it vs. baseline, turn the
weight down.

**Optimizer / schedule** (mirror Maia-3, scaled for Colab):
- AdamW, cyclic cosine annealing, batch 128 (→64 if T4 is tight), **fp16 mixed precision**.
- Not 1M steps — as many as fit a Colab session, **checkpoint to Drive** so runs resume
  across disconnects.

**Staged curriculum (always leaves a working checkpoint):**
1. Plain 5M → roughly reproduce Maia-3-ish move accuracy. **The baseline.**
2. + think-time head → predicts time above chance?
3. + FiLM → move accuracy in time-pressure positions improves?
4. + time-conditioned GAB → the full model.

If stage 4 gets finicky, stages 2/3 still yield a publishable clock-aware model.

---

## Section 4 — Evaluation

**Success = human feel.** Ordered by importance:

1. **Behavioral realism (headline).** Simulate full 3+0 games using both move and time heads.
   Check emergent human patterns: openings blitzed, hard middlegames eat time, flag scrambles
   produce fast/degraded moves. Compare model clock trajectories against real 3+0 games.
2. **Time-pressure slices.** Bucket test positions by clock (`>60s`, `10–60s`, `3–10s`, `<3s`)
   and report move-match per bucket. Hypothesis: clock-awareness helps most in **low-time**
   buckets — exactly where Maia-3 is blind.
3. **Think-time head quality.** Bucket accuracy / top-3 vs. a median-predicting baseline;
   calibration (predicted "long think" ↔ humans actually think long); sanity (time rises in
   tactical positions, falls in forced recaptures).
4. **Interpretability (pays off FiLM/GAB).** Visualize GAB attention for the *same position*
   at high vs. low clock — attention should visibly **narrow onto forcing moves** under time
   pressure. Concrete, publishable picture of the model "playing faster."
5. **Accuracy ablation (guard rail, demoted).** Clock-aware vs. no-time baseline on overall
   move-match — only to confirm we didn't get dramatically worse. Small deltas are fine.

---

## Open questions / to settle during planning

- Exact k for the think-time window, number of MDN components M, and normalization constants.
- Whether to reimplement Chessformer from scratch or adapt CSSLab/maia3's open model code.
- Precise Lichess month(s) to download and target dataset size vs. Colab epoch time.
