# Streak Model + Revealed-Strength Meter — design

> A clock-aware "intrinsic performance rating" that reads a player's *revealed* strength from their
> moves **and** timing, profiles how that strength swings at each rating, and models how it drifts
> across win/loss streaks. Captured 2026-08-14. Depends on a trained, calibrated base model.

## 1. The idea in one paragraph

Humans don't play at a single fixed Elo — within a game some moves read like a 2300, some like a
3000, and across a session their strength *drifts* with momentum (tilt after losses, confidence
after wins). This design adds two nested pieces on top of the existing skill-conditioned base model:
a **strength-meter** that reads a player's revealed-Elo *distribution* from their actual play, and a
**streak model** that learns how that distribution moves with win/loss streaks and steers the live
Elo dial so the bot's session behavior is human — not just its individual moves.

## 2. The strength-meter (measurement foundation)

The base model is Elo-**conditioned**: feed it `(position, Elo=X)` and it predicts a move
distribution (policy head) and a think-time distribution (MDN head) for a player of strength X. It
has **no Elo output** — so we read strength by *inverting* the conditioning:

- For each move in a real game, sweep the Elo dial (e.g. 2000, 2100, … 3200) and score how probable
  the player's **actual move AND actual think-time** are under each setting. The best-fit dial
  position is that move's **revealed Elo**.
- Score = joint likelihood: `P(move | pos, Elo) · P(observed think_time | pos, Elo)`
  (policy head × MDN head). Match on **both** move and timing — this is the novel part.
- Aggregate per-move reads into a per-game (and per-session) **revealed-strength distribution**
  (mean, spread, skew), not a single number.

**Why the joint (move + time) fit matters:** the move alone is often ambiguous — obvious/forced
moves are played the same at every rating and carry ~no signal. Timing sharpens the read: a hard,
only-move found *instantly* reveals a much stronger player than the same move after a long, nervous
think. No existing strength-meter (e.g. Regan's Intrinsic Performance Rating) uses the clock; folding
timing in is our extension.

**Why NOT a discriminative "guess-the-Elo" model:** training a model to output a rating directly
would learn the player's *stable account rating* (same label for all their games that month) and
average away the per-game tilt variation — exactly the signal we want. The inverse/sweep method
reads *this game's* actual play, so it catches "this game played like a 2200 even though the account
says 2500." That gap = the tilt signal.

## 3. Per-Elo swing profiles (+ free calibration check)

Run the meter across the whole dataset → build a **swing profile for every rating band**: the normal
mean, spread, and shape of revealed strength at 2200 vs 2500 vs 2800. Open empirical question
(genuinely novel with timing included): **does swing shrink as rating climbs?** (Hypothesis: lower
ratings are more volatile, higher ratings tighter/more consistent — data decides.)

- **Calibration check (bonus):** run the meter on players of *known* rating; if the 2500-band's mean
  revealed Elo ≈ 2500, the meter (and base model) is calibrated. A large offset = a bug found for
  free.
- The per-Elo table has three uses: (a) the **baseline envelope** the streak model swings around,
  (b) a **calibration/trust check** on the base model, (c) a **human-fingerprint reference** (what a
  normal swing at each level looks like; anomalies stand out — the anti-cheat/defensive angle).

## 4. The streak model (the payoff)

A session-level layer on top of the base model:

- **Offline (learn the tilt curve):** from real players' *chronological* game sequences (the
  Chess.com archives are per-player and in time order — the raw material is already harvested),
  measure each game's revealed strength (via the meter) and fit
  `effective_Elo_distribution = f(baseline, recent win/loss streak, session length, gap between
  games, …)`. Tilt likely shifts the distribution **down and wider** after losses; a heater lifts
  and tightens it.
- **Live (apply the curve):** the streak model tracks the bot's own running W/L and applies the
  learned curve to pick the effective Elo (or full distribution) to feed the base model each game.
  **No engine, no measurement needed live** — it just remembers the streak and turns the dial.

Result: snap moves stay snap, tilted sessions genuinely play weaker and rushed, hot streaks play
sharper — a bot with a human *emotional arc across games*, which no current human-imitation model
(Maia, ALLIE) has.

## 5. Honest caveats

- **Per-move Elo is noisy** — most moves are uninformative (obvious moves look identical across
  ratings). The signal concentrates at critical/branching moves. Read the *distribution*, not
  individual moves.
- **Tilt magnitude is empirical** — a 300-point swing (2500→2200) may be larger than reality
  (documented tilt is often ~50–150 pts); direction and existence are real, size TBD by data.
- **Data-thin at the extremes** — swing stats are robust in dense bands (2400–2800); noisier below
  ~2200 (~1.7% of data) and above ~3000. Trust the middle of the curve.
- **Depends on a calibrated base model** — the whole thing is only as good as the base model's
  Elo-conditioning; build the calibration check first.

## 6. Research grounding

Kenneth Regan's **Intrinsic Performance Rating** / per-move skill modeling is the established
statistical backbone (also the core of serious cheat detection). Our contributions: (a) folding
**timing** into the per-move strength read, (b) modeling **cross-game streak/tilt dynamics** rather
than single-game strength, (c) producing per-Elo **swing profiles**. See project memory
`pondering-outcome-and-timing-direction` and the clock-aware design spec.

## 7. Dependencies & next steps

1. Finish training the base ~19.5M clock-aware model (in progress: shards building).
2. Build the strength-meter (inverse likelihood sweep) + run the per-Elo calibration check first —
   it validates everything downstream.
3. Build per-Elo swing profiles.
4. Fit the streak/tilt curve on chronological player sequences; wire the live streak layer to steer
   the Elo dial.
5. Evaluate: does streak-driven play read as more human across a session?

---

# Addendum — Player Fingerprint + Re-Identification Experiment (captured 2026-08-15)

> A second Stage-2 branch, sibling to the streak model. Same dependency: a well-calibrated base.
> Idea developed in conversation while the base model trained; recorded here so it isn't lost.

## 8. The player fingerprint (residual / style-embedding)

Same core move as the strength-meter: **model a player *relative to their rating baseline*, not from
scratch.** "Base = how your rating *should* play; your deviation from it is your signature."

- **Method:** freeze the base model, train a **tiny** per-player adapter (LoRA / player-embedding,
  ~a few thousand params) on that player's own games. The small capacity is deliberate — it *can't*
  memorize noise, so it's forced to capture only the consistent deviation (the signature).
- **Why relative-to-rank wins (background subtraction):** most of what any player does is just their
  *level* (a 1500 plays 1500-ish moves, misses 1500-ish tactics, uses the clock 1500-ish ways).
  Subtracting the rank baseline removes the generic part so the personal part "pops." Modeled in
  isolation, two 1500s mostly just look like "two 1500s"; against the baseline, their opposite habits
  (grabs space + tanks quiet moves  vs  blitzes book + only thinks on tactics) light up.
- **Why it needs little data:** the base did ~99% of the work; the fingerprint only learns the ~1%
  delta → a few thousand of the player's games is plenty.

**Style-space (the elegant form):** the per-player embeddings place every player as a **point in a
learned coordinate system**. Nearby points = similar players; distance = stylistic difference. Axes
emerge (aggressive↔solid, blitz-book↔think-early, tactical↔positional). Uses: compression (whole
signature ≈ 16 numbers), "who plays like me?" = nearest neighbor, **interpolation** (blend two
players), clustering. **Timing gives axes move-only models are blind to** (fast-and-loose↔slow-and-
careful): two players at the same *move* coordinates can be pulled apart on the *clock* axis.
Direction matches the "rating-conditioned residual" / Maia-Individual literature.

## 9. Validation — the re-identification experiment (the falsifiable proof)

The clean, falsifiable test that the fingerprints are real signal, not noise. **Identification ≠ full
description** — we only need enough *distinguishing* bits to separate a player from the others in the
pool (like recognizing a friend by their walk).

**Setup:**
1. Build fingerprints for a set of blitz players from their game history.
2. **Hold out** games from each player that the fingerprint never trained on (mandatory — otherwise
   we measure memorization, not generalization).
3. Pool the held-out games (target ~10k for a statistically solid benchmark), shuffle, hide labels.
4. For each mystery game, score it under **every** player's fingerprint; guess = argmax. Check vs
   truth.

**Metrics / artifacts:**
- **Top-1 accuracy vs chance** (1/N). Above chance = fingerprints carry real signal.
- **Ranked shortlist / top-k** — even when top-1 misses, "it's one of these 8" beats chance hugely.
- **Confusion matrix (the best artifact):** *who* gets mistaken for *whom*. Mistakes aren't random —
  the model confuses **stylistic twins**, so the off-diagonal is a free "who-plays-alike" map =
  the style-space clusters made visible.
- **Per-player accuracy distribution:** distinctive players (weird repertoire + odd clock habits)
  nailed every time; generic players fuzzy. The spread is itself a finding.
- **Difficulty-scaling curve:** identify among 10 → 100 → 1000 candidates; plot where it breaks down.
- **Games-per-query curve:** 1 game = thin (good ranked shortlist at best) → 3–5 games = confidence
  climbs steeply. The 1→5 climb is a clean plot.

**Timing edge (where we most beat prior work):** existing player-ID uses moves only. A single game is
thin on move-signal but **dense with clock-signal on every move** — so one game carries far more
identifying info for us. Single-game ID (esp. among small pools, and for distinctive players) is more
feasible here than for move-only approaches.

**Honest caveats:** single-game top-1 among large pools (500+) is hard — one game just lacks the
signal to beat 499 others outright (ranked shortlist still useful). Per-game noise (opponent, opening,
luck, mood) fights identification; multiple games average it out. Needs enough held-out games per
player to both build a sharp fingerprint and test fairly.
