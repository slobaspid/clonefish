# Design: Bigger clone base with a native bucket time-head and first-class Elo

**Date:** 2026-08-26
**Status:** Approved for planning
**Author:** session (sahformer-agent) + user

---

## 1. Motivation

The clonefish product ("play against a clone built from someone's games") rests on a base model that
imitates a human's moves **and** their clock behaviour. Two problems with the current 19.5M base block
that goal, and one opportunity was surfaced this session:

1. **The time head predicts "always more."** The MDN think-time head (mixture of 3 log-normals, NLL in
   log-space) systematically over-predicts. Measured on latebloomer's held-out games: real median
   **1.0s / 13% snap (<0.5s)**, but the head plods at ~1.3–1.9s and commits only **6%** weight to the
   snap component. Root cause (verified, not guessed): human blitz timing is a **spike-at-near-zero +
   fat right tail** ("snap instantly OR think"), which a continuous log-normal mixture cannot represent —
   it hedges into the middle. The rare 20–30s tanks widen the fit further. This is a *shape* problem, not
   a data-volume problem (Allie hit the same hedge with 90M games).

2. **Elo conditioning is lazy.** HANDOFF-08-16 §3 found the Elo dial mostly changes move *confidence*,
   not *which* moves — the model took a shortcut, so `elo` in → play works, but the representation does
   not genuinely encode strength. For clones built from a whole history we need the model to track a
   player's **rating trajectory** and their **opponent-relative style** (how they play up vs down).

3. **A better time head does NOT help move-matching** (tested this session, see §3). So the time head's
   job is *feel*, and we stop trying to route it into move accuracy.

The fix template comes from **ChessMimic** (arXiv 2606.04473), the one public model that models human
clock well: a **discretised-bucket classifier** (K≈30, 1s bins to 27s + wide tail to 40+), **masked
Brier / cross-entropy**, and **clock-masking** so it never predicts an impossible time. We adopt this,
plus a bigger backbone and first-class Elo, in **one fresh training run**.

## 2. Goals / Non-goals

**Goals**
- A new base model, ~30M params, trained on the HF corpus, that:
  - predicts think-time as a **30-bucket distribution** with human snap/tank shape;
  - treats **Elo as a first-class signal** — conditioned on (self, opponent, gap) *and* predicted (an
    auxiliary rating head that forces real strength encoding);
  - keeps the existing move (policy) and value heads.
- Plays as a real engine (self-play + UCI) spending the clock with human feel.
- Slots under clonefish as the upgraded base.

**Non-goals**
- No routing of time into move prediction (proven dead end this session).
- No change to the move/value head design, GAB, FiLM backbone (only the time head is replaced and Elo
  gains a head + a gap feature).
- No pondering / adaptive-computation (killed earlier; out of scope).
- Not chasing Maia-3-79M scale in this iteration (possible later; ~3–4× compute for uncertain gains).

## 3. Evidence from this session (justifies the design)

**Head bake-off** (same frozen features, same player, held-out; MDN vs bucket):

| metric | BASE-MDN | fresh-MDN | BUCKET-brier | BUCKET-ce |
|---|---|---|---|---|
| NLL↓ (latebloomer) | 1.64 | 1.81 | 2.34 | 2.21 |
| Brier↓ | .78 | .80 | **.67** | .67 |
| top-bucket acc↑ | 38% | 15% | **52%** | 51% |

- Metric-dependent on calibration (each head wins its own loss), **but the bucket head is much sharper**
  (top-bucket acc ~1.5–2× across 3 players: latebloomer, OKENITE, couli).
- **Distribution shape (the point):** the MDN's expected value under-snaps (1–3% vs real 5–13%) and kills
  the tail (0.1–0.4% vs real 1–3.3%). The **bucket head, sampled**, reproduces both: couli tail 3.3% vs
  3.3% real, OKENITE snap 9% vs 6%, latebloomer snap 12% vs 13%. It is the only head that snaps *and*
  tanks like a human. Weakness: its central tendency can drift high — so **play must sample, not use E[t]**.

**Time → moves (oracle upper bound):** a personalized move re-ranker given the *real* think-time bucket
gained **~0pp** over a time-blind one (latebloomer 58.5 vs 58.7; OKENITE 53.8 vs 53.0). Move difficulty is
already readable from the board, so time is redundant for move *choice*. → time head is for feel only.

**Reference landscape:** Maia-3's public weights ship with `include_time_info=False` (move-only; its
scalar `fc_ponder` head is off). Allie uses a scalar MSE time head and reports the same hedge (predicts
*low*). ChessMimic uses the bucket classifier we adopt.

**Decision — train our own (not reuse).** All open weights with a usable time head are
**license-encumbered for a product**: ChessMimic (github.com/thomasj02/1e4_ai; move + bucket-clock +
outcome, per-rating, ~9M/band) is **PolyForm Noncommercial, and the restriction extends to the trained
artifacts**; Nova (99M, move-only) is custom-NC; Allie/Maia-3 are research-licensed. Crucially, **none of
them personalize / clone an individual** — all are population or per-band. So the population base is not
novel, but reusing it would bind clonefish to non-commercial terms. We therefore **train our own base**
(unencumbered ownership) and use the open models only as **external benchmarks**:
- **ChessMimic clock model** — target for our bucket time head's snap/tail calibration.
- **Maia-3 (23M) move-match** (~57%) — the move-accuracy bar for our bigger backbone.
The personalization layer (per-player fine-tune, Elo trajectory, opponent-relative style) is the part
that is genuinely ours and has no open equivalent.

## 4. Architecture

### 4.1 Backbone
`ClockAwareChessformer` (`sahformer/model/clockaware.py`), scaled up:
- `dim_vit=512, num_blocks=12` (~30M), heads/mlp_ratio scaled consistently (`num_heads=8`, `mlp_ratio=2`).
- Everything else (InputEmbedding, GAB, FiLM, TemporalEncoder) unchanged.

### 4.2 Time head — MDN → bucket classifier
New `BucketTimeHead` in `sahformer/model/heads.py`, replacing `ThinkTimeMDNHead`:
- Input: the pooled position summary **only** (drop `think_extra` — the policy-difficulty shortcut).
- `LayerNorm(dim_vit) → Linear(dim_vit, head_hid) → ReLU → Linear(head_hid, K)`, `K=30` bucket logits.
- **Buckets:** edges `[0,1,2,…,27, 32, 40, ∞]` → 27 one-second bins + 3 tail bins = 30. Well-matched to
  3+0 (observed think-times max ~30s). Defined once in a small `sahformer/model/timebuckets.py`
  (`EDGES`, `CENTERS`, `to_bucket(t)`), shared by training and play.
- **Sampler** (`sample_think_time`): mask buckets whose lower edge exceeds the remaining clock (never
  self-flags), sample a bucket by probability, then draw a time *within* the bucket (uniform in the bin;
  a capped draw for the 40+ tail). This is the sampled readout the bake-off validated.

### 4.3 Elo — first-class signal
**Conditioning (inputs):** keep per-game `elo_self`, `elo_opp` (already interpolated into skill
embeddings). Add the explicit **Elo gap** `(elo_self − elo_opp)`, normalised, into the temporal/context
vector so "vs stronger/weaker" is directly available to FiLM/GAB, not only implicit.

**Prediction (head):** new `EloHead` — `LayerNorm → Linear → ReLU → Linear(→1)` on pooled, predicting
normalised `elo_self` (regression, smooth-L1). This is the HANDOFF-08-16 §3 fix: it forces the
representation to encode strength beyond a confidence knob, which (a) makes the Elo dial track rating and
(b) gives clonefish a strength meter for free. Optionally also predict `elo_opp` (cheap; decide in
planning). Small loss weight so it regularises without dominating.

### 4.4 Heads summary
`move_logits` (4352, unchanged) · `value_logits` (3, unchanged) · `time_logits` (30, NEW) ·
`elo_pred` (1–2, NEW). `pooled` still exposed for clone adapters / fine-tune.

## 5. Loss (`sahformer/training/losses.py`)
Replace the MDN term; add the Elo term. `compute_losses` becomes:
```
policy = CE(move_logits, move_target)                      # w_policy = 1.0
value  = CE(value_logits, result)                          # w_value  = 0.1
time   = CE(time_logits, to_bucket(think_time))            # w_time   ≈ 0.2   (clock-mask optional in train)
elo    = smooth_l1(elo_pred, norm(elo_self))               # w_elo    ≈ 0.05
total  = w_policy*policy + w_value*value + w_time*time + w_elo*elo
```
Notes:
- CE chosen for the time head (co-trains cleanly with the other CE losses; Brier is a drop-in alt if we
  want max calibration — revisit after the smoke run).
- Training targets are bucketed with `to_bucket`; no clock-masking needed in training (a played move's
  think-time is ≤ the clock by construction). Masking is applied at inference.
- `per_sample_loss` / `ponder_loss` (PonderNet path) are dead code for this run; leave them but they are
  not exercised (mode = `full`, not `ponder`).

## 6. Data & targets
- Corpus: HF `slobaspeed/chesscom-balanced-shards` (311 shards, ~77.5M positions), 3+0 with `%clk`.
- `records.py` already yields `think_time = max(prev_clock − clock_after, 0)` and per-game `elo_self /
  elo_opp`. No dataset rebuild needed for time (bucketing is at loss time) or Elo (already present).
- Add the normalised Elo-gap feature into the temporal builder (`encoding.build_temporal`) or the batch
  assembly — decide in planning (prefer adding one channel to the 21-dim temporal vector; bump
  `temporal_dim`).

## 7. Training recipe & de-risk
- Recipe (Maia-3, from prior runs): AdamW `lr 5e-5, wd 1e-6, grad-clip 3.5, warmup 1000, batch 512, AMP,
  stream`. Bigger model → consider `lr 4e-5` and ~**400k steps**. Rolling `best.pt`/`last.pt`, resumable.
- **Smoke run first (local, CPU/1060):** tiny config, a few hundred steps on one shard — assert the four
  losses go down, the time head produces a non-degenerate bucket distribution, and a self-play game spends
  the clock without flagging. Only then rent the 3090.
- Full run: rented 3090 (Vast) per HANDOFF logistics (held-open ssh; container kills orphaned procs).
  User handles instance start/stop; agent handles setup + launch + monitoring.

## 8. Play & clonefish integration
- `sahformer/play.py`: `self_play` uses `BucketTimeHead.sample_think_time` when the model exposes
  `time_logits` (fall back to `_sample_think_time` MDN if an old checkpoint). Keep the clock/flag logic.
- `scripts/uci_engine.py` and `scripts/clone_uci.py`: same sampler; MimicClock unchanged in behaviour.
- Fine-tune (`scripts/finetune_clone.py`): swap the MDN loss for bucket CE; keep per-game elos so a clone
  learns trajectory + opponent-relative style. Save time head + elo head with the clone.
- clonefish: point its engine at the new base once trained (separate repo; not in this plan's scope
  beyond the hand-off note).

## 9. Acceptance criteria
1. Smoke run: all four losses decrease; time-head bucket entropy is sane (not one-hot-collapsed, not
   uniform); a self-play game completes without self-flagging.
2. Full base: on 3 held-out players, the **sampled** time distribution matches real within tolerance —
   snap-rate (<1s) within ±5pp and tail-rate (>10s) within ±2pp (the metrics the MDN failed).
3. Move-match top-1 not worse than the 19.5M base (bigger model + dropped shortcut should be ≥).
4. Elo dial: predicted Elo correlates with true rating on held-out (r materially > the old confidence-knob
   ~0), and setting the dial visibly shifts play (opening choice/quality), including vs stronger/weaker
   opponents via the gap feature.
5. Plays as a UCI engine in a GUI, spending the clock with visible snap-and-tank behaviour.

## 10. Testing
- Unit (TDD) for `timebuckets` + sampler: `to_bucket` boundaries; sampler never returns > remaining
  clock; snap bucket and tail bucket both reachable; distribution over many samples matches given logits.
- Unit for `compute_losses`: shapes, finite losses, bucketing correctness on a toy batch.
- Integration: build_model('full') forward returns `time_logits (B,30)` + `elo_pred`; a 5-ply self-play
  runs and spends time.

## 11. Risks & open questions
- **Dropping `think_extra`** may slightly hurt raw time fit; mitigated by the pooled rep + bigger model.
  Revisit if smoke-run time-loss stalls.
- **CE vs Brier** for the time head — start CE; switch to masked Brier if calibration (snap/tail match)
  is off after the full run.
- **Elo-predict head** could still be gamed if too weak; keep `w_elo` small but non-trivial, verify with
  the held-out rating correlation (criterion 4).
- **Bigger model speed/cost** on the 3090 — the smoke run + a short timed 3090 burn-in confirm steps/s and
  ETA before committing the full run.
- **Elo trajectory at inference** needs a control (set the clone's current Elo). The input already exists;
  clonefish must expose it — noted for the product, not this plan.
