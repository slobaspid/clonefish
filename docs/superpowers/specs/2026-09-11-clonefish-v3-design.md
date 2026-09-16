# clonefish v3 — "your games in, an engine that plays like you out"

Status: **r3, supersedes v2** (2026-09-11). v2 + a full subagent design review + this session's
re-verification of the raw result JSONs. Changes from v2 are marked **[CHANGED]**.

Evidence: `results/FINDINGS_2026-09-11.md` (numbers recomputed from raw JSONs this session),
`results/book_blend/*.json`, `lit/NOTES*.md`, project memory.
**Both controls have since REPORTED** (see `results/FINDINGS_2026-09-11.md`):
`[POP]` n=20 — personal book +3.61 pp vs cohort book +0.63 pp, cohort tier adds +0.05 pp ⇒ the book
gain is genuinely personal, and the cohort tier is dropped.
`[INHEAD]` 6/12 — in-head players average +2.67 pp vs strangers +2.44 pp ⇒ **no in-head advantage**;
the "head diversity is the bottleneck" story is dead.

---

## 0. The answer in one paragraph

A clone is **a strong population human model + a personal opening book + personal clock habits,
sampled, never argmaxed** — and that is *almost all of it*. Measured on 17 held-out strangers, every
personal layer we have is an **opening** layer: post-ply-10 the entire stack is worth **+0.55 pp**.
So base quality is the dominant lever on move-match, the book is the dominant lever on
"that's my repertoire", and the clock is the only place we differentiate from every published
system. Everything else — style codes, low-rank updates, search — is either a small NLL/realism
gain or an open research bet, and must be gated by an experiment that tests *the premise*, not the
method. Per-user fitting is counting + two scalars + one grid search: seconds on a laptop CPU.

## 1. What the evidence actually says (verified this session)

1. **All measured personal signal is in the opening.** 17 strangers, Δtop-1 vs base:
   book +2.91 (all) / **+0.26 (ply≥10)**; code +2.44 / +0.33; both **+3.49 / +0.55**.
   By phase (both layers): open **+11.4**, early-mid +0.02, mid **+0.001**, end −0.05.
2. **The midgame null is real, not a harness artifact.** Checked explicitly: the pooled code *does*
   change midgame predictions for **17/17** players (max |Δtop-1| 2.2e-2, mid Δnll −0.0028); the
   per-player changes **cancel to +0.001 pp**. The residual is applied everywhere and buys nothing.
3. **Layers are redundant, not additive.** book 2.91 + code 2.44 would be 5.35; measured 3.49.
   **[CHANGED]** L2's marginal value over L1 is **+0.58 pp [0.18, 0.98] all**, **+0.28 pp
   [−0.09, 0.66] at ply≥10 — CI crosses zero**, worse for 4/17 players. v2 claimed "+4–5 pp alone";
   that figure came from players *inside* the shared head's training pool.
4. **Temperature is not a component.** `base+T` is identical to `base` in top-1 at every phase
   (T cannot move an argmax); in the midgame the fitted T makes NLL slightly *worse* (+0.0025).
5. **The in-head vs stranger gap is confounded.** v2's implicit "head diversity is the bottleneck"
   rests on n=1 (latebloomer, base top-1 45.8 vs strangers 52.1 — a known low-floor outlier, memory
   `opening-gain-is-player-headroom`). **[RESOLVED]** `[INHEAD]`, same pool/harness/splits, 6
   in-head players: **+2.67 pp mean vs +2.44 pp for strangers — no in-head advantage.** The
   diversity story is dead; a 3,000-player head retrain cannot be justified on L2's account.
6. **Mistakes are position difficulty × clock — not difficulty alone. [CHANGED]** v2 said "mistakes
   follow position difficulty, not the clock" and used it to license independent sampling. Anderson
   1606.04956 actually says the blunder rate is flat above ~10 s and **rises steeply below**.
7. **Within a player, slower moves are worse moves** (Sunde PNAS 2026, player-game fixed effects;
   strongest in blitz). Players also think longer where computation pays (Russek 2025).
8. **Top-1 is a weak guide to feel** (DAgger; Codevilla: equal offline error, very different
   behaviour). Automated judges fail at *fine* ranking — which is why recognizer-as-reward failed 3×.
9. **~Half of a person's "thinking style" isn't stable.** van Opheusden: per-person parameters
   correlate 0.93 on refits but **0.53–0.58 across sessions**. Part of the midgame ceiling is the
   person drifting, not a model defect.

## 2. Architecture

### 2.1 Offline, once (GPU, by us)
- **Population model** = the clock-aware base. **Base quality is the biggest single lever on
  move-match** (§1.1). A new base is adopted only if it wins the frozen 22-player harness.
- **[CHANGED] No shared-head retrain is scheduled on L2's account.** It is justified only if
  `[INHEAD]` shows a real in-head advantage, and even then it buys NLL/realism, not top-1. If that
  GPU run ever happens it must do triple duty (§6 Phase 2).

### 2.2 Per user (laptop CPU, seconds)

| Layer | What | Fit | Verified value |
|---|---|---|---|
| **L0** rating + real clocks | `elo_self` from game headers; true clock features | none | conditioning. **Take Elo from headers — do NOT fit it by likelihood** (logP rises monotonically with Elo for everyone → saturates 1900–2200; memory `fingerprint-result-and-calibration`) |
| **L1 opening book** | counts of *their* moves per exact position (EPD, transpositions merge), own-move < 20 | counting + one α | **+2.91 pp alone, +9.5–13.5 pp in the opening, NLL better 20/20.** Best value/cost in the design |
| **L3 clock habits** | log-time shift + spread on the sampled base think-time | 2 scalars on tune | distribution realism. **The only thing here no published system does** |
| L2 style code *(optional)* | 512-d code vs a frozen shared head | GD on their positions | **+0.58 pp marginal (ply≥10 CI crosses zero)**; NLL −0.021, better 15/17. **[CHANGED] demoted to optional**; ship only if it also earns its keep on NLL |
| L4 midgame | — | — | **[CHANGED] deferred; E6 is a no-go as specified** (§5) |

**[CHANGED] Every per-user scalar uses closed-form partial pooling**, not hand-tuned shrinkage:
`θ_user·n/(n+k) + θ_pop·k/(n+k)`, with `k` estimated once from between-player variance on the
existing pool. This replaces v2's three separate hand-set knobs (code weight decay, α default,
λ∝1/games). Gains are largest exactly at the 50-game end (HDDM).

**Book blend at a position** (hierarchical, so it degrades gracefully and subsumes the cohort case):
```
r(m) = (n_cohort(m) + β·p(m)) / (N_cohort + β)      # cohort/population book, β→∞ ⇒ r = p
q(m) = (n_user(m)   + α·r(m)) / (N_user   + α)      # personal book on top
```
Out of book `q = p` — no cliff. **[RESOLVED — `[POP]`, n=20]** the cohort tier `r(m)` is **dropped**
and the blend is personal-only, `q(m) = (n_user(m) + α·p(m)) / (N_user + α)`: cohort alone is worth
+0.63 pp against personal +3.61 pp, and stacking them adds **+0.05 pp**. Not worth the machinery.

### 2.3 Play time (UCI)

- **Sample** the move from `q`. `Strict` mode = argmax for users who want max top-1.
- **[CHANGED] Couple the move to the sampled time (the v2 bug).** v2 sampled move and think-time
  independently, which contradicts §1.7. Between-position coupling already exists (the time head
  reads `policy_difficulty`, `clockaware.py:32`); what's missing is *within-position*. Fix, one
  scalar, no extra compute:
  ```
  t ~ time_head ;  z = (log t − E[log t | position]) / sd ;  T_move = 1 + γ·z ;  sample from q^(1/T)
  ```
  Fit `γ` once on population data to reproduce the observed within-player time→quality slope.
  **Measure the slope before fixing** — the fix is only justified if our clone currently fails it.
- **[CHANGED] No hard clamp to the clock budget.** Clamping truncates the long-think tail exactly
  in scrambles, where Anderson's curve is steepest and where players notice most. Use an explicit
  low-clock time policy calibrated to budget-use fraction (Sigman: 77.9 % strong vs 74.7 % weak).
- Write the **sampled** time into the engine's own history (constant fake times break the model).
- **[CHANGED] Anchored search: dropped from Phase 0/1.** +0.2–0.8 pp at blitz, needs a value head,
  cpuct tuning and a playout budget, and argmax-of-search costs −18.7 pp.

## 3. Fitting pipeline (`clone_fit`)

1. Fetch rated 3+0 games with clocks; de-dup by game id; sort by date.
2. **Time split:** fit = all but last 80, tune = −80..−40, scorecard = newest 40.
   **[CHANGED]** Under ~150 games: tune = newest 20 %, scorecard skipped with a warning, and α is
   **still fitted on the tune split** (v2 said "defaults α=2", which left the tune split with no
   consumer and L3 with no fitting split — ambiguous enough to produce two different products).
3. One base forward over all positions (cache pooled + logits).
4. Build L1 from fit; fit α, L3 and (if enabled) L2 on tune; print the scorecard.
5. Refit L1 on all games. **[CHANGED]** α is re-checked after the refit (the book densifies, so the
   optimum moves); the printed scorecard is labelled as pre-refit.
6. Save `{format_version, base_sha, book, α, time_cal, elo, scorecard}`.

## 4. Evaluation

Null = base at T=1. Ceiling = the person's own held-out games. Per-move metrics on real held-out
positions, mean over players with per-player CIs, **by phase** (a whole-game mean hides that
everything is opening).

- **Target:** NLL overall and on book-unseen positions — must not be worse than base on either.
- **Report:** top-1 by phase; **marginal** value of each layer (not standalone); calibration gap.
- **[CHANGED] Promote whole-game branch-game bias into the acceptance set.** v2 pinned ~everything
  to offline per-move metrics while §1.8 argues offline metrics are untrustworthy; the rollout check
  cannot stay optional.
- **Feel:** think-time median/snap/tank shares; blunder rate vs seconds left (must be flat >10 s,
  steep below); **the within-player time→quality slope** (§2.3); opening-distribution distance vs
  the same/different-player gap.
- **[CHANGED] Human pair test must be powered to fail.** v2 cited 0.77–0.84 judge accuracy, but that
  is human-vs-RL-agent; the same source says *fine* ranking is unsolved. State N, pairs/judge, and a
  bar **relative to the control**: judges ≤55 % on clone pairs while ≥70 % on the real-vs-base control.
- **[CHANGED] The non-circular recognizer is dropped.** Fourth recognizer signal after three
  documented failures; circular through L1 (it reads openings, and we hand the clone its own book);
  and "time zeroed" is an artifact, not a control — base features carry clock info through
  `pooled = enc.mean + t_to_d(t)` and FiLM, and faking times destroys even real-game ID (0.33→0.03).

**Regression harness:** the 22 players/splits from E1/E2 are frozen as the acceptance test.

## 5. The midgame: what to try, in order **[CHANGED]**

v2's single bet (E6: anchored rank-4–8 LoRA + policy head) is **no-go as specified**:
its extrapolation base is a 2-player mean spanning +1.1 to +6.3 pp (one a documented failure mode);
everything between a code and a full fine-tune measures ≤0.4 pp midgame; full FT *loses* 3 pp at 1k
games while the median user is far below 10k; ~half the target signal isn't stable across sessions;
and its success criterion (+0.5–1.5 pp midgame top-1) is a metric this document argues is a weak
guide to feel and which the human gate cannot detect.

**M0 — test the premise first (hours, decisive).** Fine-tune the *same recipe, same game count* on a
**different** player in the same rating band, then evaluate on latebloomer's held-out games. If the
cross-player fine-tune recovers most of the +6.3 pp, the gain is rating/domain adaptation, **not
identity**, and no anchored variant can deliver a personal midgame. This tests the premise, not the
method — run it before any midgame engineering.

**M1 — adapter bank + routing** (2502.14998; Per-Pcs). GPU once to build the bank, then **10–50
mixture weights per user** on CPU. The right parameter count for a 50–500 game user, versus E6's
tens of thousands of free parameters — the same geometry that produced v1's −6.2 pp.

**M2 — Maia4All two-stage** (2507.21488): individual modeling from ~20 games; stage (a) is
essentially the diverse-player retrain, and stage (b) is initialised by a **discriminative**
nearest-prototype task. **We already own that discriminator** (91 % top-1 of 90 players from 10
games; GE2E 0.993@10). The shelved identification work is an asset the design was not using.

**M3 — amortized encoder / CNP** (1807.01613; P2P is 33× faster at deployment): train a
games→code encoder *during* whatever diverse-player GPU run happens, deleting the per-user gradient
step entirely. Counterweight to cite: 2306.13554 finds plain fine-tuning competitive with
meta-learning — so build the simple encoder, not a meta-learning protocol.

## 6. Plan

- **Phase 0 — ship what is verified.** Replace v1's adapter (a measured **−6.2 pp** regression) with
  **L0 + L1 + L3 + sampling**, the time/move coupling, partial-pooled scalars, and a scorecard.
  No L2, no search, no recognizer. This is the whole product for most users.
- **Phase 1 — cheap, decisive controls.** `[POP]`, `[INHEAD]`, **M0**, the time→quality slope
  measurement, and L2/book thinning **in the 50–180 game band** (never tested; `clone_stack_eval.py`
  skips players under 180 games, yet 50 games is the product target and v1's −6.2 pp happened at ~120).
- **Phase 2 — only if Phase 1 earns it.** One diverse-player GPU run doing triple duty: shared head
  (L2), adapter bank (M1/M2 stage-a), and games→code encoder (M3). Plus match-dependent book weight
  (kNN-LM) for the coverage cliff (cov1 0.381 → 0.005 at early-middlegame).

## 7. Explicitly rejected
Recognizer-as-reward (3× dead end) and the non-circular recognizer (§4) · per-user timing residual
(over-thinks) · full fine-tune as default · recency-weighted book (no gain) · fitted temperature as a
component (§1.4) · fitting Elo by likelihood (§2.2 L0) · prompting a model with raw games ·
argmax-of-search · **E6 as specified** (§5).
