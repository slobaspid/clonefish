# clonefish — final summary (2026-09-16)

Given one person's Lichess/chess.com games, produce a UCI chess engine that plays like them: their moves, their
clock habits, their resignations. `PLAN.md` is the full chronological log; this file is the conclusion.

## Build a clone for anyone

```
python scripts/clonefish_build.py --lichess  THEIR_USERNAME
python scripts/clonefish_build.py --chesscom THEIR_USERNAME
python scripts/clonefish_build.py --pgn their_games.pgn --name THEIR_USERNAME   # a PGN you already have
```

Downloads their rated 3+0 games with clock times from the site's public API (`scripts/clonefish_fetch.py`;
single-threaded, honours HTTP 429, public data only), detects Lichess (whole-second) vs chess.com (tenth-second)
clocks, fine-tunes on up to their 5,000 most recent games
(~45 min on a GTX 1060), builds their opening book, learns their resignation habit, writes
`engines/clonefish_THEIR_USERNAME.bat`. Everything is learned from their games; nothing is hand-set per player.

## Recommended configuration

Full fine-tune with `--time-loss bucket` · personal opening book · sampled move (top-p 0.9) · sampled think-time
really spent · `--resign` (v3 mate-danger model) · per-player field opponent for evaluation.
**Every extra knob stays at 0: `NormPush`, `IdentityPush`, `PaceSigma`, `ResignBias`.** Each was tested and rejected
(see "Settled" below).

## How well it matches

**Game level — 13 of 15 bars** (3 players, 150 simulated games per arm, `eval11_v3resign.json`):
resign 3/3 · lost on time 3/3 · game length 3/3 · think-time 2/3 · mistake profile 2/3.

- Think-time distribution W1 0.018-0.030 vs the base model's 0.13-0.15 (5-10x closer).
- Clone-vs-real average centipawn loss: **+2.0, +1.0, -4.9** (Stockfish itself drifts ~2 points between identical
  runs, so read smaller gaps as noise).
- Generality: on three players never used in development (~1,120 games each), resign 3/3, lost on time 3/3,
  length 3/3 against whole-history targets.

**Move level — finished work**: top-1 58.7 / 58.9 / 62.9 %, top-3 85.5 / 85.7 / 88.8 %, against a base model at
51.8 / 80.1 %. The fine-tune's ~+7pp IS the personalisation; nothing added on top of it moves the number.

**Always quote a bar with its target window** (see "Measurement lessons"). The two bars that miss are kpowe52's
timing (his clone plays 11.2 % instant moves vs his ~8.3 % — a state effect, not a time-head defect: teacher-forced
his head is conservative at 6.6 % vs his 8.5 %) and VEGETAL's mistake bar, which fails on blunder rate 4.5 % vs the
base model's 4.4 % against his 3.8 % — inside Stockfish noise, and his clone's ACPL is the closer one.

## Settled — do not re-tread

| idea | verdict |
|---|---|
| Amplify the clone away from the norm (`clone + w*(clone - stock)`) | **Dead, 5 ways.** Scalar push, learned identity head, discriminator guidance, top-3-selected sweep, and a family signal learned from self-play disagreements all choose w = 0 (or slightly negative, i.e. TOWARD the norm). |
| Is there even a clone-vs-norm signal? | **Yes, and it is strong** — a 10k-param head reads the clone's pick vs the stock's pick in the same position at **81-88 %** on all three players. It is real, systematic, legible — and already correctly scaled, which is why amplifying it does nothing. |
| Identity as a learned head on a FROZEN trunk | Works as *personalisation* (+5.04pp of the fine-tune's +6.92pp with 34.8k params) and is legible (castles +0.79, checks -0.54 …). Not an improvement on the fine-tune; useful as a cheap per-player option or a fingerprint. |
| Opening book for MOVE match | Contributes nothing (+-0.4pp, slightly negative). **But keep it on — its value is TIMING** (the player's real opening think readings = premoves). |
| kNN retrieval | Closed by inference. Its headline lifted a BASE model to 65.2 % opening top-1; our clones sit at 60.7 / 71.8 / 74.0 % unaided. |
| Per-game pace variance (`PaceSigma`) | Failed its pre-registered bar at every sigma. Informative failure: sigma 0.70 matched the opponent's flag RATE (28.3 vs 28.8 %) while cutting games to 59 plies (real 90) — causality runs length -> flags, not flags -> length. |
| Opponent `ResignBias` | An 80-game sweep liked -50; the 150-game 3-player test rejected it (helped one player, broke two). |
| Per-player move temperature | Clones are already calibrated (max-likelihood T = 1.0-1.1). |
| v2/v3 resign features | v3 (mate-danger) raised AUC 0.914 -> 0.922 but barely moved the per-GAME firing rate; kept, but it was not the lever for game length. |

## Measurement lessons (these cost the most time)

1. **Never judge a correction with the objective that trained the model.** Scoring "should we amplify?" by move
   likelihood asks "is the MLE not the MLE" — no, by construction. Use generated games, or top-k, or game statistics.
2. **80-game targets are too noisy for the bars.** Median game length reads 99 on the newest 80 and 90 on every
   larger window; that alone flipped the length bar from 1/3 to 3/3. Estimate game-level targets from the player's
   whole history. Contamination only matters for move-match and the think-time reference.
3. **Adjudicate the target window per statistic**: use the long window unless the statistic shows a MONOTONE trend in
   time-ordered blocks (VEGETAL's instant % really does rise 5.1 -> 7.1 across his history; his game length does not).
4. **"Closer to REAL than BASE" is a bad bar** — between two runs it went 0/3 -> 2/3 while the clone barely moved and
   the BASE arm swung 7-10 points. Report the clone-vs-real gap directly.
5. **Never tune on one seed / <80 games.** ACPL swings ~7 points; ResignBias -50 looked good at 80 games and was
   overturned at 150.
6. **Watch for input leaks in discriminators.** Real games carry each game's true Elo, generated games one average —
   equalise them or AUC 0.83 collapses to 0.61.
7. Stockfish drifts ~2 ACPL between identical runs (hash state carries across sequential `analyse()` calls).

## What is actually left

**The policies are faithful; the STATE DISTRIBUTION is what differs.** Four independent measurements agree — a
discriminator (move AUC minus position-only AUC is <= +0.010 across 3 players x 2 sides), the top-k amplification
test, the game-length decomposition, and the teacher-forced pace check. Which move and how long to think are right;
what differs is the distribution of *situations* the games produce, set by the simulated opponent and the self-play
loop.

Useful metric that came out of this: **state drift = position-only AUC minus the real-vs-real floor**, per player and
per side, needing no Stockfish and no pre-registered bars (~3 min/player). It localises the problem:
VEGETAL +0.004 own / +0.080 opponent (his clone is faithful, his OPPONENT is wrong); OKENITE +0.120 own /
+0.053 opponent (his own clone drifts, U-shaped by phase: +0.187 opening, +0.099 middlegame, +0.193 endgame);
kpowe52 +0.054 / +0.091. OKENITE drifts while passing every bar, so drift sees something the bars do not.
State-matching is not a simple knob — the one intervention tried on it (ResignBias) helped one player and broke two.

Also open: a blind "is this you?" test with people, other time controls, a strength ladder.
