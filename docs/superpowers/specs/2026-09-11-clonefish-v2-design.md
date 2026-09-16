# clonefish v2 — proposed architecture: "your games in, an engine that plays like you out"

Status: PROPOSAL r2 (2026-09-11), after a three-way literature dive (A: games AI, B: cognitive science,
C: ML personalization) and this session's experiments. For subagent review. Slots marked [B] are filled once
dive B reports; slots marked [E2]/[POP] once the running experiments finish.

Evidence files: `results/book_blend/` (experiments), `lit/NOTES_dive_A_games.md`, `lit/NOTES_dive_B_cogsci.md`,
`lit/NOTES_dive_C_ml.md`, `lit/NOTES_priorart.md`, `lit/NOTES.md`, project memory.

---

## 0. The answer in one paragraph

A clone is **a strong population human model, plus four cheap personal layers, sampled — never argmaxed**.
The population model decides most of how human the engine is (it already predicts ~50–58% of a club
player's moves). On top of it, in order of measured value: (1) the person's **own opening book**, blended
with the model as a Bayesian posterior; (2) a small **personal style code** against a shared head trained on
many diverse players; (3) their **clock habits**, as two numbers applied to a sampled think time; (4) their
**rating**, taken from their games. A fifth layer — an **anchored per-user weight update** to reach the
middlegame — is the one open research bet, gated by an experiment. Everything is fitted on a laptop CPU.
Success is judged by held-out likelihood and a human pair test, not by top-1 alone.

## 1. What the problem actually is (the facts that force the design)

1. **Personalization is small next to the population model.** Base top-1 for club players 46–60%; every
   personal method we or the literature have found adds single-digit points. So *base quality is the biggest
   lever on move-match*, and the personal layers are what make it feel like *you*. (E1; Maia-individual
   Table 2: +2.8 pp at 10k games, +5.2 at 40k.)
2. **Personal signal is concentrated in the opening.** Book +12.8 pp in own moves 0–11, exactly 0 after;
   pooled code likewise almost all opening. (E1; memory.)
3. **The middlegame has personal signal, but a single code can't reach it.** A full fine-tune on ~10k games
   gave +3.6 pp midgame; codes, retrieval, reweighting all ≤0.4 pp. Image/LLM personalization shows the same
   pattern: embedding-only saturates ("paltry improvement" with more data, Textual Inversion 2208.01618; "can
   be limiting", 2210.03505); anchored weight updates get past it (DreamBooth 2208.12242, Ditto 2012.04221).
4. **Naive per-user fitting overfits.** v1's single-player adapter is −6.2 to −6.6 pp vs base; Maia
   fine-tuning loses 3 pp at 1k games. Every personal parameter needs a pull back to the population.
5. **Top-1 is a weak guide to feel.** Errors compound over a game (DAgger 1011.0686); equal offline error can
   mean very different behaviour (Codevilla 1809.04843); automated judges can't rank close candidates (NTT
   2105.09637) — which is exactly how our recognizer-as-reward attempts failed 3×. Humans judge
   human-likeness reliably (0.77–0.84, n=30).
6. **Mistakes follow position difficulty, not the clock** (Anderson 1606.04956). Realistic blunders come from
   a calibrated policy that is *sampled*; think time should follow the model's own uncertainty (Russek 2025;
   Sunde 2201.10808). Search barely moves blitz move-match (+0.2–0.8 pp, piKL 2112.07544) and argmax-of-search
   destroys it (−18.7 pp, 2605.11893).
7. **Nobody models an individual's think time** (dive A). Our base already has a think-time head; sampling it
   (never its mean) is what makes timing realistic (memory `time-head-ceiling`).

## 2. Architecture

### 2.1 Offline, once (GPU, by us)

- **Population model** = the clock-aware base (currently `base_300k_best`, 19.5M params, 8 blocks). Any new
  base (v3, or one with engine features) is swapped in only if it wins on the fixed 22-player harness below.
- **Shared style head**, retrained on **many diverse players** (the 3,000-player `lichess_scale` pool, not the
  current 120). Diversity of training users is the stated bottleneck for anything that generalises to new
  users (P2P 2510.16282: "training user diversity is more critical than sheer quantity"). Published to HF with
  the base hash it was trained against.

### 2.2 Per user (laptop CPU, minutes)

| Layer | What it is | Fit | Measured value |
|---|---|---|---|
| L0 conditioning | their recent rating (from game headers) as `elo_self`; real clocks | none | — |
| L1 opening book | counts of *their* moves per exact position (EPD, transpositions merge) | counting | +3.6 pp top-1 alone, NLL better 20/20; 25 games already give ~½–¾ of it (E1, E5) |
| L2 style code | 512-d code vs the frozen shared head, weight-decayed toward zero | gradient descent on their positions | +4–5 pp alone; stacks with L1 [E2] |
| L3 clock habits | two scalars (log-time shift, spread) on the sampled base think time | on the tune split | distribution realism; no per-user time model exists anywhere |
| L4 middlegame (gated) | rank-4–8 low-rank update on the top block(s) + policy head, with (a) L2 anchor to base weights, strength ∝ 1/games (Ditto), (b) KL to the base on generic same-rating positions (prior preservation) | on their positions | unmeasured; expected +0.5–1.5 pp midgame at 1k–5k games (extrapolated) → **experiment E6 decides** |

Combination at a position: `p = softmax(base_logits + code_residual [+ L4])`,
`q(m) = (n_m + α·p(m)) / (N + α)`. Out of book `q = p` — no cliff. `α` fitted per user (typically 1–2).

### 2.3 Play time (UCI)

- **Sample** the move from `q`. `Strict` mode = argmax, for people who want maximum top-1.
- `Randomness` = temperature on `p`, default 1 (fitted T is ≈1.0 once the code is in; E2).
- Think time: sample the base time head, apply L3, clamp to the clock budget; write the *sampled* time into
  the engine's own history (constant fake times break the model — audit finding).
- Optional, off by default: a light think-time-scaled, policy-anchored search (Allie + piKL) only for
  "play at my strength but cleaner" or strong users; sample from the visit policy, never argmax.

### 2.4 Cognitive-science layer [B]

[B — fill from dive B: whether a per-person bounded-rationality parameter (planning depth, lapse rate,
uncertainty-coupled time) adds anything the layers above don't.]

## 3. Fitting pipeline (clonefish `clone_fit`)

1. Fetch rated 3+0 games with clocks; de-duplicate by game id; sort by date.
2. Split by time: tune = games −80..−40, scorecard = newest 40, fit = the rest (under ~150 games: tune on the
   newest 20%, scorecard skipped with a warning, defaults α = 2).
3. One base forward over all positions (cache pooled features + logits).
4. Fit L2 on fit; build L1 from fit; fit α and L3 on tune; print the scorecard on the scorecard split.
5. Refit L1/L2 on all games with frozen hyper-parameters; save
   `{format_version, base_sha, head_sha, code, book, α, time_cal, elo, scorecard}`.

## 4. Evaluation

Null = base at T = 1. Ceiling = the person's own held-out games. All per-move metrics on real held-out
positions (no self-play), mean over players with per-player CIs, by phase.

- **Target:** NLL overall and on book-unseen positions (N = 0) — must not be worse than base on either.
- **Report:** strict top-1 (overall, ply ≥ 10, by phase); sampled match ÷ self-agreement; calibration gap.
- **Feel (research harness):** mistake rate of sampled moves by position difficulty and clock (value head,
  spot-checked vs Stockfish once); think-time median / snap / tank shares; opening-distribution distance vs
  the same-player/different-player gap (UCLouvain scale: <0.069 same player, ≥0.101 different); systematic
  whole-game biases from branch games (clone vs base from real branch points, control base vs base).
- **Recognizer**, non-circular only: a new encoder on (before, after) pairs, time zeroed, own move ≥ 12, same
  positions scored three ways (real / clone-sampled / base-sampled), normalised; > 1 is a red flag.
- **Human pair test** (the gate): real game vs clone continuation branched at own move 12, with real-vs-base
  and self-vs-stranger controls; randomised and pre-registered.

**Regression harness:** the 22 players and splits from E1/E2 are frozen as the acceptance test for every
change (base swaps, head retrains, L4).

## 5. Explicitly rejected

Recognizer-as-reward (3× dead end) · per-user timing residual (over-thinks) · full fine-tune as default (GPU,
~10k games, loses at 1k) · recency-weighted book (E3: no gain) · style-twin borrowing (E5 thin users already
get most of the book; NLP prior art ~1 perplexity point) · fitted temperature as a model component (≈1.0 with
the code) · prompting a model with raw games (VALL-E needed 60K hours; USER-LLM's encoder beats raw history) ·
argmax search.

## 6. Plan

- **Phase 0 — ship what is proven (clonefish):** replace v1's adapter with L0–L3 + sampling; scorecard in
  `clone_fit`; format/compat checks. This alone removes a known regression and adds the measured gains.
- **Phase 1 — raise the ceiling (research repo):** E6 anchored low-rank update (the middlegame bet); retrain
  the shared head on ~1,000–3,000 diverse players and re-run the stranger test; population-book control [POP];
  base v3 acceptance on the harness; fix the triage harness bugs; build the non-circular recognizer.
- **Phase 2 — only if Phase 1 earns it:** amortized clone encoder (instant clones; P2P); match-dependent book
  weight / opening-family backoff; optional anchored search for strength calibration.
