# clonefish engine — "working clone" check (pass bars written BEFORE results, 2026-09-14)

Engine: `scripts/clonefish_uci.py` (fine-tuned clone + personal book + sampled think-time, whole-second
timing inputs like the Lichess training data). Launchers: `engines/clonefish_<player>.bat`.
Harness: `scripts/clonefish_eval.py` (simulated 3+0 games vs a base-model opponent at the player's typical
opponent rating; REAL = player's newest 80 games, never used for the clone or its book).

## Gate 1 — it is a real engine
- A real-time UCI game vs Stockfish finishes with no illegal move, crash, or flag.
- 60 simulated games per arm per player complete with no errors.

## Gate 2 — timing is fixed (the measured give-away)
For >= 2 of 3 players:
- CLONE log-think W1 vs REAL is lower than BOTH CLONE_OLD (exact-float timing) and BASE.
- CLONE instant-move % and 10 s+ % each within 2 points of REAL.
- CLONE median clock left at own move 30 within 15 s of REAL.

## Gate 3 — it makes mistakes like the player
For >= 2 of 3 players (Stockfish depth 10, own moves after the first 8):
- CLONE blunder % (>= 300 cp) within 2 points of REAL, and closer to REAL than BASE.
- CLONE average centipawn loss closer to REAL than BASE.

## Gate 4 — timing no longer gives it away (independent recognizer)
Line-up on the ENGINE's games: target rank with timing visible no worse than with timing hidden
(+/- 1.5 ranks) for >= 2 of 3 players, with the real-games positive control passing.

## Results
- **Gate 1a PASS (2026-09-14):** real-time UCI game, VEGETAL clone vs Stockfish UCI_Elo 1886, 3+0 clocks,
  30 plies, no illegal move/crash/flag (`results/clonefish/smoke.pgn`). Book moves ~1 s (1.e4 from 2,954
  games), out-of-book thinks 2-16 s; clone lost to a king hunt (Stockfish "1886" is tactically far stronger
  than a 1886 human — not a strength measure). Launcher `engines/clonefish_kpowe52.bat` answers the UCI
  handshake and plays the player's book reply (1.e4 e6, from 1,785 games).

- **Eval 1 (never-flag engine, top-p 0.9; results/clonefish/eval.json):**
  - Gate 1b PASS: 540 simulated games, no errors.
  - Gate 2 FAIL (instant moves only). W1 clone < old < base for VEGETAL (0.053/0.087/0.211) and kpowe52
    (0.134/0.149/0.205); OKENITE tie (0.080 vs 0.078; its clone predates the whole-second loss). 10 s+ % and
    clock@30 within bars for 3/3. Instant % off for 3/3 (7.1 vs 10.7, 10.8 vs 20.4, 5.6 vs 11.5).
    Cause found: clock < 10 s instant rate real 16-22 % vs clone 73-80 %; clone also plays far more moves
    under 10 s (kpowe52 18 % vs 3.9 %). The never-flag rule makes scramble moves instant and keeps the game
    going; real 3+0 games end on time in 35 / 48 / 41 % of games (player lost on time 14 / 30 / 12 %).
  - Gate 3 FAIL. ACPL real/clone/base: 66.7/65.1/67.4, 74.9/64.2/66.3, 73.1/65.4/71.9; blunder %
    4.0/4.0/5.5, 5.8/3.5/4.4, 5.0/4.5/5.0. Clones cluster at ~65 ACPL = cleaner than the player.
    Suspect: nucleus top-p 0.9 removes the low-probability moves that human blunders come from.
- **Eval 2 (pre-registered):** AllowFlag on (human-like), arms CLONE (top-p 0.9), CLONE_P100 (no filter),
  BASE. Same Gate 2/3 bars, plus: player-lost-on-time % within 10 points of REAL for >= 2 of 3.

- **Eval 2 RESULT (AllowFlag on; results/clonefish/eval2.json):**
  | | VEGETAL real / clone / base | kpowe52 | OKENITE |
  |---|---|---|---|
  | W1 log-think vs real | — / **0.029** / 0.101 | — / **0.025** / 0.132 | — / **0.032** / 0.167 |
  | instant % | 7.1 / 6.4 / 15.0 | 10.8 / 8.8 / 14.4 | 5.6 / 8.4 / 15.2 |
  | 10 s+ % | 4.8 / 3.8 / 5.5 | 6.2 / 5.8 / 5.7 | 2.4 / 2.6 / 4.3 |
  | clock @ own move 30 | 82 / 89 / 93 | 77 / 73 / 75 | 98 / 93 / 98 |
  | player lost on time % | 15.0 / 21.7 / 11.7 | 16.2 / 25.0 / 16.7 | 11.2 / 8.3 / 13.3 |
  | ACPL | 66.7 / **66.5** / 72.4 | 74.8 / **77.3** / 71.6 | 71.7 / **70.6** / 57.5 |
  | blunder % | 4.0 / **4.0** / 4.7 | 5.0 / **5.3** / 5.2 | 4.6 / **4.6** / 3.9 |
  - **Gate 2 PASS** (2/3: VEGETAL, kpowe52 all bars; OKENITE misses instant % by 0.8 — its clone predates the
    whole-second loss). Allowing flags fixed the instant surplus (W1 halved to 5x lower than eval 1).
  - **Gate 3 PASS** (2/3: VEGETAL, OKENITE; kpowe52 blunder is 0.3 off real but base is 0.2 off — inside the
    ~0.8-point Stockfish run-to-run noise seen on identical REAL games). The eval-1 "too clean" result was the
    never-flag rule, not the nucleus filter.
  - Lost-on-time within 10 points: 3/3.
  - Nucleus filter: top-p 0.9 matches VEGETAL/OKENITE better than no filter (P100 ACPL 77.4 / 67.5); keep 0.9.
  - Open oddity: vs the same base-model opponent, clones score 27.5 / 30.0 / 20.8 % while BASE scores
    52.5 / 32.5 / 46.7 %, despite matching mistake profiles. The BASE arm is effectively playing a copy of its
    opponent; not a strength measure. A Stockfish-ladder rating test would settle it.

- **Gate 4 RESULT (engine games, leak-free enrollment; results/clonefish/lineup_engine.json).** Mean rank
  of the target (30 players), 5-game queries; S2 = rating neutral + timing VISIBLE, S3 = timing hidden.
  | | VEGETAL real / base / clone | kpowe52 | OKENITE |
  |---|---|---|---|
  | S0 all clues | 5.5 / 22.8 / **6.1** | 1.0 / 2.6 / **1.5** | 4.0 / 21.8 / 10.8 |
  | S2 timing visible | 7.6 / 18.3 / **5.2** | 2.0 / 7.8 / **4.6** | 3.3 / 26.6 / 18.2 |
  | S3 timing hidden | 8.5 / 14.2 / 13.1 | 2.1 / 13.1 / 2.6 | 5.0 / 16.8 / 13.1 |
  - VEGETAL PASS (timing now HELPS: 5.2 visible vs 13.1 hidden; weak real control P@1 0.25).
  - kpowe52 narrow FAIL (timing costs 2.0 ranks, bar 1.5) — vs ~10 ranks before the engine fixes; with timing
    visible the clone now beats base (4.6 vs 7.8), where the old generator had it worse than base (13.5 vs 10.8).
  - OKENITE FAIL (timing costs 5.1 ranks) — the only clone trained WITHOUT the whole-second loss, and the one
    still over-producing instant moves. **Gate 4 = 1/3 -> FAIL.** Next: bucket-loss fine-tune for OKENITE, rerun.
  - S4 moves-only recognizer fails the real-games control for all 3 again (unusable).

- **OKENITE retrained with the whole-second loss** (`clones/ftval_OKENITE_bucket.pt`, launcher updated).
  Post-opening top-1 +3.55 (old loss +4.44; inside the ~1.2 pp noise). Engine eval (eval3_OKENITE.json): think W1
  **0.0095** (old clone 0.032, base 0.167); instant 5.8 vs real 5.6; 10 s+ 2.4 vs 2.4; clock@30 97 vs 98;
  lost on time 11.7 vs 11.2; ACPL 70.2 vs 71.7; blunder 4.5 vs 4.6. **Gate 2 now 3/3, Gate 3 still passes.**
  Line-up (lineup_engine_OKENITE_bucket.json): S0 rank 6.2 (old clone 10.8, real 4.0, base 22.0);
  S2 timing visible 12.4 vs S3 hidden 7.9 -> timing still costs 4.5 ranks (was 5.1). **Gate 4 stays 1/3.**
- **What drives think-time (own moves after 8, clock >= 10 s): correlation of log think with ...**
  | | legal moves | in check | capture | move # | clock | lag-1 |
  |---|---|---|---|---|---|---|
  | VEGETAL real / clone / base | .27/.26/.28 | -.09/-.10/-.13 | -.12/-.08/-.09 | -.31/-.33/-.20 | .26/.27/.11 | .16/.15/.23 |
  | kpowe52 | .25/.33/.34 | -.09/-.13/-.15 | -.15/-.11/-.07 | -.26/-.32/-.28 | **-.08/.21**/.12 | .19/.19/.27 |
  | OKENITE | .30/.32/.30 | -.16/-.17/-.14 | **-.18/-.12**/-.09 | -.31/-.29/-.25 | .22/.20/.14 | .21/.21/.24 |
  The clone matches how each player's think-time responds to the position on nearly every driver (and beats
  the base on clock and move-number coupling). No systematic timing flaw left; residual gaps are player-specific
  (kpowe52 clock coupling, OKENITE captures). Whatever CosFace still detects is finer than these, and each line-up
  rank rests on 12 five-game queries.

## VERDICT (2026-09-14)
**Working clone engine: YES by Gates 1-3; the strictest check (Gate 4, recognizer) is 1/3.**
Gate 1 real engine PASS · Gate 2 timing PASS 3/3 · Gate 3 mistakes PASS 2/3 (3rd inside measurement noise) ·
lost-on-time rate PASS 3/3 · Gate 4 CosFace timing-visible vs hidden 1/3 (gaps 2.0 and 4.5 ranks vs ~10 before).
Not pursuing Gate 4 further by tuning against the recognizer (that failed 3x before, see memory). Next real
gates: resignation behaviour (engine never resigns), a strength ladder, and a blind human "is this you?" test.

## Play-in-browser prototype (2026-09-15)
`play_clonefish.bat` -> `scripts/clonefish_play.py` (Flask, CPU) + `scripts/clonefish_play.html` at
http://localhost:5001. Server owns real-time 3+0 clocks (first move of each side free; flag loses unless the
opponent has insufficient material), clone samples move + think and really waits it, resign, PGN with [%clk].
Verified:
- API game vs VEGETAL: clone 1.e4 in 0.1 s (book), clone clock charged exactly its think (1.2 s), my 3 s wait
  charged 3.0 s, first moves free, illegal move rejected.
- Losing on time (test server, 8 s per side): clone flagged after 21 plies ("Black lost on time", clock 0.0);
  a player who stops moving lost exactly when their clock ran out ("White lost on time").
- Page: all 3 clones listed (fixed a load-order bug), pieces load, New game from the page vs kpowe52,
  1.e4 answered 1...e6 in 0.1 s, move list shows clone think times, clocks tick correctly.
- NOT verified: the mouse drag gesture itself (the app window was hidden, so the browser pane could not take
  pointer input); the page's move code path was exercised directly instead.

## Phase 2 — resignation + scramble timing (pre-registered 2026-09-15, before any result)
Real behaviour (2,000 recent training games each): VEGETAL resigns 23 % of games (median -8 material,
48 s left), **kpowe52 never resigns (0 of 443 non-flag losses; all checkmated)**, OKENITE resigns 12 %
(-11 material, never under 10 s). Scramble gap (eval 2 games): at 5-10 s real instant 21 / 13 / 15 % vs clone
6 / 7 / 3 %, real mean think 1.17 / 1.28 / 1.16 s vs clone 1.48 / 1.46 / 1.43 s; clones lose on time more
(22 / 25 / 12 % vs 15 / 16 / 11 %).
Design: per-player logistic resign hazard on (value-head loss/win prob, material, clocks, move #), fitted on
training games (`clonefish_resign_fit.py`); opponent gets a "field" hazard fitted on the players' real opponents.
Then check whether the scramble gap is the time head (teacher-forced, `clonefish_scramble_check.py`) before
changing anything.
**Pass bars (>= 2 of 3 players, simulated games vs the base opponent):**
- R1 resign model val AUC >= 0.80 and predicted resignations/game within 30 % of real (on held-out training games).
- R2 player-resigned % of games within 8 points of REAL (newest 80); kpowe52 must resign in <= 3 % of games.
- R3 player-lost-on-time % within 7 points of REAL; median game length within 12 plies of REAL.
- R4 no regression: timing Gate 2 and mistakes Gate 3 still pass.

### Phase 2 results so far
- **R1 resign models** (val = 10 % of training games held out of the fit):
  VEGETAL player AUC 0.939, resignations/game predicted 0.223 vs real 0.212 (PASS); field AUC 0.887, 0.137 vs 0.116.
  OKENITE player AUC 0.957, 0.158 vs 0.112 (count 41 % high -> FAIL the 30 % bar; ~28 val resignations, noisy);
  field AUC 0.803, 0.133 vs 0.148. kpowe52: 0 resignations in 2,500 games -> the regression cannot be fitted;
  script now saves a flat smoothed rate (~0.0002 resignations/game). R2 (simulated resign rate) is the real test.
- **Scramble check (teacher-forced, training games): the time head is calibrated in scrambles for all 3.**
  | instant % real / model | <5 s | 5-10 s | 10-30 s |
  |---|---|---|---|
  | VEGETAL | 20.9 / 20.7 | 12.4 / 11.8 | 7.7 / 8.1 |
  | kpowe52 | 29.2 / 26.8 | 13.3 / 12.9 | 6.4 / 6.9 |
  | OKENITE | 19.1 / 19.9 | 11.4 / 10.9 | 6.7 / 7.3 |
  Mean thinks match to within 0.04 s. So the free-play gap (clone 5-10 s instant 6 / 7 / 3 %) is NOT the head.
  Note the eval "REAL" scramble numbers came from only ~100 moves per bucket in the newest 80 games (VEGETAL
  5-10 s: 21 % there vs 12.4 % over 2,314 training moves) — use the larger training sample as the reference.
  The engine sampler `_sample_think_time` uses the same mixture parameterisation (checked). Next:
  `clonefish_engine_replay.py` — real positions through the ENGINE path, plus an opponent +60 s variant.

### Eval 4 — resignation ON (results/clonefish/eval4_resign.json), % of games
| | resigned | mated | lost on time | won: opp resigned / mate / time | draw | median plies |
|---|---|---|---|---|---|---|
| VEGETAL real | 20.0 | 21.2 | 15.0 | 7.5 / 2.5 / **28.8** | 5.0 | 99 |
| VEGETAL clone | **31.7** | 11.7 | 16.7 | 15.0 / 13.3 / **3.3** | 8.3 | 86 |
| kpowe52 real | 0.0 | 25.0 | 15.0 | 17.5 / 21.2 / **16.2** | 5.0 | 80 |
| kpowe52 clone | **0.0** | 28.3 | **31.7** | 16.7 / 8.3 / **5.0** | 10.0 | 92 |
| OKENITE real | 7.5 | 21.2 | 10.0 | 12.5 / 7.5 / **31.2** | 10.0 | 99 |
| OKENITE clone | 16.7 | 36.7 | 15.0 | 5.0 / 10.0 / **3.3** | 13.3 | 90 |
- R2 FAIL (1/3: kpowe52 0 vs 0; VEGETAL +11.7, OKENITE +9.2). R3 lost-on-time 2/3 (kpowe52 31.7 vs 15.0);
  median plies 2/3 (VEGETAL -13).
- Scrambles now match: 5-10 s instant clone 15.9 / 11.5 / 15.2 vs real (training) 12.4 / 13.3 / 11.4.
  <5 s still high for VEGETAL / OKENITE (40 / 27 vs 21 / 19). Think W1 0.029 / 0.031 / 0.025 (base 0.12-0.15).
- **Diagnosis: the simulated OPPONENT is unrealistic.** Real opponents lose on time in 16-31 % of games;
  the base-model opponent (chess.com-trained) in 3-5 %. **CORRECTED after measuring the opponents directly**
  (1,500 training games each): it is NOT scramble speed — <5 s instant is similar (real opponents 43-60 %,
  base opponent 45-74 %). It is PACE: real opponents spend more time earlier, so more of their moves are played
  with 10-30 s left (8.4-10.2 % vs 6.1-6.5 %) and they flag 19-23 % of games vs 3-5 %. So clones win less (score ~36 / 35 / 25 % vs real 41 / 58 / 56 %) and
  sit in lost positions longer against an opponent slow to finish them: VEGETAL clone resigns where the real
  VEGETAL got mated (resign + mate 43.4 % vs real 41.2 %).
- R4 mistakes with resignation on (ACPL / blunder %, real / clone / base): VEGETAL 66.7/4.0 · 75.2/5.0 · 75.2/5.0
  (FAIL: clone no closer than base — likely more time spent in lost positions vs this opponent); kpowe52
  74.7/5.3 · 72.7/4.9 · 80.4/6.3 (PASS); OKENITE 71.7/4.6 · 74.7/4.8 · 67.7/3.9 (PASS). Gate 3 = 2/3.
- **Engine replay (real training games through the ENGINE code path; engine_replay.json): no pipeline bug.**
  instant % real / engine — VEGETAL <5 s 22.9/21.8, 5-10 s 14.5/15.1, 10-30 s 9.7/8.7; kpowe52 25.8/26.6,
  10.0/14.7, 6.6/6.8; OKENITE 23.7/20.3, 9.9/9.9, 7.1/6.4; mean thinks within ~0.1 s. Giving the opponent +60 s
  lowers the engine's 5-10 s instant rate by 3-5 points (15.1->11.8, 14.7->10.0): the opponent's clock matters a
  little. So free-play timing gaps are situational (the opponent), not the engine.
  Next: a Lichess "field" opponent fine-tuned on these
  players' real opponents (excluding the test games), replacing the base-model opponent.

### Eval 5 — Lichess field opponent (results/clonefish/eval5_field.json; summary: clonefish_eval_summary.py)
- Opponent flag rate improved: won-on-time VEGETAL 10.0 (base opp 3.3, real 28.8), kpowe52 13.3 (5.0, real 16.2),
  OKENITE 6.7 (3.3, real 31.2).
- Bars: R2 resign 2/3 (was 1/3) · R3 lost on time 1/3 · R3 length 2/3 · G2 timing 2/3.
- VEGETAL still resigns 33.3 % vs 20.0 (same as eval 4 -> not the opponent); games 82 vs 99 plies.
- kpowe52 instant 8.1 vs 10.8 in every eval so far (8.8 / 8.5 / 8.1): real kpowe52 plays 31 % of own moves 2-8
  instantly (book premoves) vs clone 16 % -> the clone copies book MOVES but not book TIMING.
- Fix in progress (general, data-driven): the book cache (v2) also stores each player's real think readings per
  opening position (first 40 plies, >= 3 samples); in a known position the engine samples its think from those,
  blended with the model by the same n/(n+alpha) rule as the moves.

- **Resignation diagnosis (training games vs eval-5 clone games): the resign HABIT matches; the clones lose more.**
  VEGETAL resigns in 42 % of losses (real, 857 losses) vs 49 % (clone, 41 losses, +/- ~8); at resignation ply 81 vs 81,
  own clock 48 s vs 48 s, material -8 vs -5. OKENITE 22 % vs 28 % of losses (32 losses), material -11 vs -13.
  Unconditional resigned % is high because the clone LOSES more to the simulated opponent (VEGETAL 68 % of games vs
  real ~56 %; the base model also loses 67 % there -> the field opponent is relatively strong at these Elo inputs).
  Summary script now also prints endings conditional on the result (diagnostic, not a pre-registered bar).
- Stockfish measurement noise on identical REAL games across runs: ACPL moves ~1.5-2 (OKENITE 71.7 vs 73.0, kpowe52
  74.7 vs 76.4), blunder % ~0.5 — so strict "clone closer than base" on sub-2-point gaps is a coin flip.

- **Eval 5 final bars:** R2 resign 2/3 · R3 lost on time 1/3 · R3 length 2/3 · G2 timing 2/3 · G3 mistakes 0/3
  (every clone within ~3 ACPL / ~1 blunder point of the real player; strict "closer than base" is noise-level).
  Endings conditional on the result: real opponents flag in 74 % / 30 % / 61 % of the player's wins vs the pooled
  field opponent 40 % / 27 % / 25 % -> VEGETAL's and OKENITE's real opponents are far more time-troubled than the
  pooled field. Next: **per-player opponent models** (each trained only on that player's own real opponents) -> eval 7.

### Eval 6 — + player's real think times in known opening positions (eval6_booktime.json, pooled field opponent)
- **G2 timing 3/3 (was 2/3).** Instant % real / clone: VEGETAL 7.1 / 8.0, **kpowe52 10.8 / 10.4 (was 8.1)**, OKENITE
  5.6 / 7.1. Think W1 vs real: **0.017 / 0.018** / 0.031 (eval 5: 0.048 / 0.032 / 0.017). Clock at own move 30:
  82 / 80, 77 / 71, 98 / 99 s. The book-timing change closed kpowe52's premove gap as predicted.
- R3 length 3/3 · R3 lost on time 2/3 · R2 resign 1/3 (VEGETAL 28.3 vs 20.0; OKENITE 16.7 vs 7.5 — OKENITE's clone
  loses 63 % of games vs real 39 %, so the opponent-strength confound again). Next: eval 7 (per-player opponents).
- **Eval 6 final: R2 1/3 · R3 lost on time 2/3 · R3 length 3/3 · G2 timing 3/3 · G3 mistakes 2/3** — best run so far.
  ACPL / blunder % real / clone / base: VEGETAL 66.7/4.0 · 71.6/4.3 · 75.4/5.0 (PASS); kpowe52 75.2/5.3 · 73.8/4.8 ·
  65.3/4.4 (PASS — base far too clean, clone reproduces kpowe52's mistakes); OKENITE 71.6/5.1 · 76.6/5.6 · 72.7/4.3
  (fail on ACPL). Resignation is the only bar below 2/3, tied to opponent strength.

### Eval 7 — per-player opponents (each trained only on that player's own real opponents; eval7_perfield.json)
- **R2 resign 3/3 · R3 lost on time 3/3 · G2 timing 3/3** · R3 length 1/3 · G3 mistakes 1/3.
  Resigned % real / clone: VEGETAL 20.0 / 20.0, kpowe52 0 / 0, OKENITE 7.5 / 13.3. Lost on time 15.0 / 15.0,
  15.0 / 16.7, 10.0 / 10.0. Opponent flag (won on time) 28.8 / 11.7, 16.2 / 21.7, 31.2 / 21.7 — much closer than the
  pooled field. Length: VEGETAL 76 vs 99, kpowe52 95 vs 80, OKENITE 92 vs 99. ACPL/blunder real vs clone: VEGETAL
  66.7/4.0 vs 69.3/3.8, kpowe52 74.2/5.2 vs 75.4/4.6, OKENITE 74.3/5.1 vs 67.4/3.9.
- Per-player opponents make the whole-game endings match; use them as the evaluation environment from now on.

### DEAD END — time-head-only refinement for low-data players (phase3/eval_fresh_th.json, pace_check_th.json)
- `--init-ft clone --epochs 0 --time-head-epochs 2`: held-out pace got slightly WORSE (first-30 sum real / before / after:
  ARQZITRO 70 / 79.0 / 79.6, AlexDC 82 / 88.2 / 91.4, Alexei_Soroka 110 / 107.9 / 113.9); fresh-player G2 1/3 -> 0/3.
  Don't retry. (The option stays in finetune_clone.py, off by default.)

### Phase 4 — push the clone away from the norm (user's idea, 2026-09-16)
The clone plays the stock model at the player's rating. Compare, after the opening, how often the clone's move is NOT the
stock model's first choice (and how surprising its moves are to the stock model) with the same numbers for the REAL player
on their training games. If the clone is "too normal", push: move scores = clone + w * (clone - stock); pick w where the
clone's off-norm numbers match the real player's, then re-run the full engine eval (per-player opponents) and keep the push
only if timing / mistakes / endings / length get closer to the real games. Nothing is trained.
Script: `clonefish_norm_push.py`; engine option `NormPush`; eval option `--norm-push NAME=w`.

**Round 1 — clone vs stock self-play, 40 games per strength (results/clonefish/norm_push.json).**
Off-norm % = own moves >= 12 that are not the stock model's first choice (+/- ~1.4); surprise = mean -log p_stock.
| | REAL off-norm / surprise | w=0 | w=0.5 | w=1.0 | w=1.5 | w=2.0 | best |
|---|---|---|---|---|---|---|---|
| VEGETAL | 44.4 / 1.325 | 44.4 / 1.209 | 43.0 / 1.199 | **44.6 / 1.362** | 48.5 / 1.417 | 48.3 / 1.510 | 1.0 |
| kpowe52 | 48.6 / 1.500 | **41.7** / 1.220 | 46.7 / 1.319 | **48.8 / 1.580** | 56.3 / 1.903 | 55.3 / 1.911 | 1.0 |
| OKENITE | 42.5 / 1.256 | 39.7 / 1.094 | **44.0 / 1.286** | 45.3 / 1.357 | 43.8 / 1.426 | 46.0 / 1.408 | 0.5 |
| ARQZITRO (1.1k) | 45.6 / 1.401 | 40.8 / 1.143 | 40.9 / 1.176 | 41.7 / 1.222 | 44.7 / 1.346 | 46.6 / 1.451 | 2.0 (~1.5) |
| AlexDC (1.1k) | 41.6 / 1.238 | 35.1 / 1.046 | 37.5 / 1.066 | **42.9 / 1.206** | 37.9 / 1.180 | 44.7 / 1.348 | 1.0 |
| Alexei_Soroka (1.1k) | 44.8 / 1.343 | 37.5 / 1.060 | 43.9 / 1.180 | 40.9 / 1.197 | **43.1 / 1.295** | 47.7 / 1.425 | 1.5 |
- All 6 unpushed clones are more "normal" than their player (off-norm short by 0-7 points, surprise short for all 6).
  Low-data clones (1.1k games) need stronger pushes (1.0-2.0) than 5k-game clones (0.5-1.0). Adjacent strengths are
  sometimes non-monotone (AlexDC 1.0 -> 1.5, Soroka 0.5 -> 1.0): each strength plays different games, so the real noise is
  larger than the +/- 1.4 binomial figure; treat neighbouring strengths as ties.
- Unpushed clones are "too normal": off-norm short by 0 / 7 / 3 points and surprise low for all three (their deviations
  from the stock model are milder than the real player's). A push of 0.5-1.0 matches both numbers; 1.5+ overshoots into a
  caricature (kpowe52 56 % vs 49 %). The right strength differs per player and is picked automatically.
**Round 5 — STABILITY (80 games per strength, seeds 11 and 22; push_stability.log). The decisive run.**
| | push 0, seed 11 / 22 | push 0.5, seed 11 / 22 |
|---|---|---|
| Alexei_Soroka (real ACPL 82.9, blunder 5.8 %) | 70.5 / 69.4 (gap **-12.4 / -13.5**), blunder 3.8 / 3.8 | **77.9 / 78.5** (gap -5.0 / -4.4), blunder **5.4 / 5.4** |
| kpowe52 (real 85.5, 6.1 %) | 83.6 / 76.4 (gap **-1.9 / -9.1** — flips) | 82.1 / 82.6 (gap -3.4 / -2.9) |
**Verdict (the rule that survives):**
1. **Detecting a too-clean clone is reliable only when the gap is large and repeats across seeds** (Soroka -9 to -13.5
   in three runs; kpowe52 flipped -1.9 vs -9.1 with only the seed changed).
2. **Fitting an exact push strength is NOT measurable** at 30-80 games: the seed decides the pick (kpowe52 chose 0 on
   seed 11 and 0.5 on seed 22). Never fit w; use a fixed modest 0.5.
3. **A modest push 0.5 on a genuinely too-clean clone reproducibly helps**: Soroka gap -13 -> -4.7 and blunder 3.8 ->
   5.4 % (real 5.8) in BOTH seeds.
So: default NormPush 0; apply 0.5 only when the clone's ACPL is cleaner than the player's by > ~8 in two seeds
(>= 80 games each).
**Confirmation (full engine eval, Alexei_Soroka, push 0.5; phase3/eval_soroka_push05.json). Real / no push / push 0.5 / base:**
ACPL 78.7 / 66.3 / **76.7** / 74.3 · blunder 5.2 / 4.3 / **5.7** / 4.7 · think W1 — / 0.074 / **0.053** / 0.193 ·
lost on time 23.8 / 13.3 / **16.7** / 6.7 · resigned 6.2 / 13.3 / **10.0** · plies 84 / 72 / **82**.
Everything moved toward the player. Two bars still read fail by a hair (instant moves 10.3 vs 8.2, limit 2.0; lost on
time gap 7.1, limit 7.0) and the mistake bar fails only on a tie-breaker (clone closer on ACPL 2.0 vs base 4.4, but
exactly tied on blunder distance 0.5 vs 0.5, and the rule demands strictly closer).
**Phase 4 conclusion: the user's idea works as a REPAIR, not a general improvement** — clone-vs-stock games reliably
DETECT a too-tame clone; a fixed modest push fixes it; fitting the strength per player is not measurable.

**Round 4 — pick the push by matching the player's real MISTAKES, and only if the clone is too clean**
(`clonefish_push_fit_acpl.py`, 30 games per strength, Stockfish depth 9 — absolute ACPL is higher than the depth-10
tables, but internally consistent). Real ACPL / unpushed clone / chosen push:
VEGETAL 75.3 / 77.1 / **0** · kpowe52 85.5 / 78.7 / 1.0 · OKENITE 75.9 / 76.7 / **0** · ARQZITRO 82.3 / 82.9 / **0** ·
AlexDC 78.6 / 78.8 / **0** · Alexei_Soroka 83.6 / 74.4 / 0.5.
- **4 of 6 now get no push**, including ARQZITRO and AlexDC, which the off-norm rule had pushed (2.0 and 1.0) and broken.
  The "only if the unpushed clone is cleaner than the player by > 2 ACPL" guard is doing the work.
- **But the two fitted pushes rest on noisy evidence.** ACPL vs push is non-monotone for both: Soroka 74.4 / 83.4 / 75.0 /
  78.2 and kpowe52 78.7 / 76.7 / 85.7 / 83.3 (real 83.6 / 85.5). With ~1,000 scored moves per strength the swing is ~+/-5
  ACPL, larger than the effect. Stability check running: push 0 vs 0.5, 80 games, seeds 11 and 22, both players.
  If the ordering flips between seeds, per-player push fitting is not measurable at this sample size — report that.

**Round 3 — FRESH PLAYERS (1.1k games): the push helps exactly the clone that was TOO CLEAN, and hurts the others.**
(phase3/eval_fresh_np.json vs phase3/eval_fresh.json; real / no push / push)
| | ACPL | blunder % | think W1 | lost on time % | mated % |
|---|---|---|---|---|---|
| **Alexei_Soroka** (w 1.5) | 78.0 / 66.3 / **77.6** | 5.4 / 4.3 / **5.7** | — / 0.074 / **0.052** | 23.8 / 13.3 / **18.3** | 20.0 / 20.0 / 21.7 |
| AlexDC (w 1.0) | 71.7 / 69.2 / 77.3 | 4.9 / 4.6 / 4.7 | — / **0.060** / 0.081 | 6.2 / **10.0** / 18.3 | 23.8 / **16.7** / 26.7 |
| ARQZITRO (w 2.0) | 75.1 / **75.3** / 82.9 | 4.7 / **4.7** / 5.8 | — / 0.111 / **0.070** | 5.0 / 8.3 / 6.7 | 41.2 / **48.3** / 61.7 |
Bars: no push R2 3/3 · time 2/3 · length 3/3 · timing 1/3 · mistakes 2/3; push 2/3 · 2/3 · 3/3 · 2/3 · 1/3.
**Verdict: the idea is sound but the SELECTION RULE was wrong.** Matching the off-norm rate picked w=2.0 for ARQZITRO
(whose clone already matched its player's mistakes) and broke it; it picked 1.5 for Soroka (clone 12 ACPL too clean) and
fixed it almost exactly. Next: pick w by matching the player's real ACPL, only when the unpushed clone is too clean by
more than the ~2-point Stockfish noise, and cap w at ~1.5.

**Round 2 — does the push make whole games closer? DEV PLAYERS: NO (eval8_normpush.json vs eval7_perfield.json).**
Same clones, same per-player opponents, only the push differs (VEGETAL 1.0, kpowe52 1.0, OKENITE 0.5). Real / no push / push:
| | resigned % | lost on time % | mated % | think W1 |
|---|---|---|---|---|
| VEGETAL | 20.0 / **20.0** / 26.7 | 15.0 / **15.0** / 10.0 | 21.2 / 26.7 / **15.0** | — / **0.038** / 0.039 |
| kpowe52 | 0 / 0 / 0 | 15.0 / **16.7** / 31.7 | 25.0 / **25.0** / 36.7 | — / **0.033** / 0.039 |
| OKENITE | 7.5 / **13.3** / 16.7 | 10.0 / **10.0** / 8.3 | 21.2 / 30.0 / **25.0** | — / **0.015** / 0.045 |
**Final bars (with Stockfish): eval 7 no push R2 3/3 · time 3/3 · length 1/3 · timing 3/3 · mistakes 1/3;
eval 8 push R2 2/3 · time 2/3 · length 1/3 · timing 3/3 · mistakes 0/3.** Nothing improved, three bars got worse.
Mistakes real / no push / push: VEGETAL 66.7/4.0 · 69.3/3.8 · (pending in table); kpowe52 74.2/5.2 · 75.4/4.6 ·
**71.5/3.8**; OKENITE 74.3/5.1 · 67.4/3.9 · **65.6/3.7**. The push makes clones play CLEANER, not sloppier: the moves it
prefers over the stock model's favourite are unusual but not mistakes — so it cannot fix a clone that is too clean.
Bars: R2 resign 3/3 -> 2/3; R3 lost on time 3/3 -> 2/3; length 1/3 both; G2 timing 3/3 both but every W1 got worse.
So matching the off-norm statistic did NOT transfer to whole-game behaviour for 5k-game clones — it made it slightly
worse (kpowe52 flags twice as often as the real player once pushed). Mistake counts pending; the fresh players
(1.1k games, clones 5-7 points too normal) are the case where a push could still pay off.

## Phase 3 — does it work for ANYONE? Fresh players (pre-registered 2026-09-15, before any result)
The method was developed and checked on VEGETAL / kpowe52 / OKENITE only. To test generality, apply the exact
same pipeline, with no changes and no per-player settings, to players never used in development.
Selection rule, fixed in advance: `data/lichess_1k`, sorted by name, skipping the 5k-set players
(MALOUMNJAK, OKENITE), take the first three: **ARQZITRO, AlexDC, Alexei_Soroka** (~1,200 3+0 games each).
Pipeline per player: fine-tune (newest 80 held out: 40 validation + 40 test, whole-second time loss) ->
book (newest 80 excluded) -> resign fit (newest 80 excluded) -> clonefish_eval vs the Lichess field opponent.
Pass bars: the same R2 / R3 / G2 / G3 as above, >= 2 of 3 fresh players. Caveat known in advance: ~1,200
games per player (not 5,000), so moves / timing may personalise less; that is part of what is being tested.
**Phase 3 progress (fine-tunes on ~1,120 games, newest 80 held out):**
- ARQZITRO post-opening top-1 54.68 -> 58.14 (+3.46); by phase top-1/top-3 opening +40.2/+34.4, early-mid +9.4/+10.4,
  mid -0.2/+2.9, end +1.7/+1.2. Resign: rare resigner (14 in 1,120 games); player AUC 0.991, 0.014 vs 0.018 per game;
  field AUC 0.865, 0.191 vs 0.161.
- AlexDC 58.33 -> 61.20 (+2.87); opening +37.6/+24.1, early-mid +6.1/+1.5, mid +1.1/+1.5, end +0.4/-1.2.
  Resign: player AUC 0.906, **0.144 vs 0.092 per game (+56 %)**; field AUC 0.851, 0.294 vs 0.220 (+34 %).
- Resign models over-predict on the single 10 % held-out split in most fits (OKENITE +41 %, AlexDC +56 %, fields +13..+34 %
  in 5 of 6). Each split has only ~10-45 resignations, so it may be noise — but the direction repeats. The fit script now
  reports 5-fold grouped cross-validation calibration (all resignations out-of-sample) and saves its features
  (`clones/<name>_resign_features.npz`) so calibration can be re-checked without a GPU.
- Alexei_Soroka 55.74 -> 57.10 (+1.36); opening +12.7/+14.4 (small: already-predictable book), early-mid +2.3/+2.5,
  mid +0.2/+1.6, end +1.9/+2.7. **First 5-fold CV resign fit: calibrated.** Player AUC 0.928, out-of-sample predicted /
  real resignations 142.2 / 137 = 1.04 (lost positions 115.3 vs 112); field AUC 0.830, 225.6 / 221 = 1.02.
  -> the earlier single-split over-predictions look like small-sample noise, not a model bias.

**Phase 3 engine eval (phase3/eval_fresh.json, Lichess field opponent; mistakes pending):**
R2 resign **3/3** · R3 lost on time **2/3** · R3 length **3/3** · G2 timing **1/3** (only AlexDC).
- Endings conditional on losing (real vs clone): ARQZITRO resign 5 / 0 %, mate 85 / 85, flag 10 / 15; AlexDC 35 / 30,
  52 / 43, 13 / 26; Alexei_Soroka 12 / 29, 40 / 43, 48 / 29.
- Timing failure = PACE, in both directions: clock at own move 30 real / clone / base — ARQZITRO (very fast)
  118 / 95 / 89 s, Alexei_Soroka (slow) 72 / 89 / 98 s. The clone moved only ~20-35 % of the way from base pace to the
  player's with ~1,120 games (the 5k-game players matched pace). ARQZITRO think W1 clone 0.111 vs base 0.108.
  Checking next: teacher-forced pace on held-out games (head under-personalised vs situational).
- **Phase 3 FINAL scorecard (fresh players, incl. Stockfish):** R2 resign 3/3 · R3 lost on time 2/3 · R3 length 3/3 ·
  G2 timing 1/3 · **G3 mistakes 2/3** — so 4 of 5 bars pass on players never used in development, with ~1,120 games each.
  ACPL / blunder % real / clone / base: ARQZITRO 75.1/4.7 · 75.3/4.7 · 77.7/5.4 (near-exact); AlexDC 69.9/4.2 · 69.2/4.6 ·
  66.1/5.0; Alexei_Soroka 77.8/5.3 · **66.3/4.3** · 66.9/4.2 (clone as clean as base: Soroka's sloppiness not learned from
  1,120 games; also the smallest move gain, +1.36). Checking: per-player maximum-likelihood move temperature.
- **DEAD END — per-player move temperature.** Max-likelihood T on validation games (post-opening, never trained on):
  VEGETAL 1.0, kpowe52 1.1, OKENITE 1.0, ARQZITRO 1.1, AlexDC 1.1, Alexei_Soroka 1.0; NLL gain vs T=1 at most 0.0034.
  The clones' move probabilities are already calibrated, so a temperature dial cannot make Soroka's clone sloppier.
  Soroka's free-play ACPL gap is situational or the top-p 0.9 tail cut (untested per player; eval 2 P100 was mixed).
- **Pace diagnosis (teacher-forced on the 80 held-out real games; sum of think over own moves 1-29, s):**
  ARQZITRO real 70 / clone 79 / base 70; Alexei_Soroka 111 / 108 / 101; AlexDC 82 / 88 / 96. So on real positions the
  head is mostly personalised (Soroka matches, AlexDC closer than base) except ARQZITRO (clone ~13 % slow, worse than base).
  In free play the gaps are larger and go both ways (ARQZITRO slower still, Soroka faster: +17 s clock at move 30), and
  the base model shifts by ~20 s between teacher-forced and free play -> mostly situational (opponent / positions), as the
  engine replay found for the dev players. Next (queued after eval 7): time-head-only refinement (`--init-ft` clone,
  `--epochs 0 --time-head-epochs 2`, moves unchanged) for the 3 fresh players, pace re-check, fresh-player eval rerun.

Also added: `scripts/clonefish_build.py` (one command: PGN + username -> engine; detects Lichess whole-second vs
chess.com tenth-second clocks — verified on VEGETAL / ARQZITRO = whole, chess.com mischuk_d = exact).

Known limits: 3 players; engine never resigns or flags (humans do); opponent is the base model, not the
real field; REAL games are vs humans.

## Round 4 — the definitive low-noise scorecard (eval9_final, 150 games/arm, Stockfish on 100)

Why: 30-80 game runs swing ~5-7 ACPL points on the seed alone, so several earlier verdicts were noise-limited.
Recommended config: full fine-tune (`--time-loss bucket`) + book + sampled think + `--resign` + per-player field opponent,
NormPush 0.

| bar | result |
|---|---|
| R2 resign | **3/3** |
| R3 lost on time | **3/3** |
| G2 timing (think W1) | **3/3** (0.018 / 0.028 / 0.026 vs base 0.13-0.15) |
| R3 game length | 1/3 (VEGETAL 78 vs 99, kpowe52 89 vs 80, OKENITE 86 vs 99 plies) |
| G3 mistakes | 0/3 — but see below |

ACPL real / clone / base: VEGETAL 68.4 / 73.3 / 69.0 · kpowe52 79.5 / 74.4 / 77.8 · OKENITE 70.3 / 71.4 / 67.7.
The clone is within ~5 points of the player every time, and the two misses go in OPPOSITE directions (VEGETAL too
sloppy, kpowe52 too clean), so this is not a systematic bias to push on. The bar fails because it asks the clone to
beat the BASE model's distance, and the base — a Maia-like model at the same Elo — already lands near a human's ACPL.
That makes G3 a weak test of personalisation; the honest reading is "clone matches the player to within noise".

**Game length is a resignation-TIMING problem, not a play-length problem.** Mean plies by how the game ended
(real held-out vs clone, VEGETAL): mated 80.4 / 81.7, own flag 135.8 / 110.2, opponent flags 105.7 / 97.9,
draw 95.8 / 105.3 — but **he resigns at 94.5 / clone 70.1**, and **his opponents resign at 73.3 / simulated 49.6**.
OKENITE the same (88.3 / 77.9 and 68.9 / 55.1). Both sides quit ~20-25 plies too early, and the simulated opponent
also resigns far too often (24 % of games vs 7.5 % real, 12.6 % fitted).
Cause: a per-turn logistic hazard fit on all turns is right about WHETHER (VEGETAL 23.6 % fitted, 21.3 % in the clone)
but too flat across the lost stretch — real players resign right after things collapse, not evenly while losing.
Fix under test: v2 resign features = + (jump in p_loss since my last turn, material just dropped, how many turns in a
row I have been lost, p_loss x that streak), with a new out-of-sample diagnostic reporting the move number the hazard
fires at vs the move the player really resigned. `clonefish_resign_fit.py` now fits v1 and v2 and keeps the closer one;
the engine picks the version from the coefficient count, so old `_resign.json` files still work.

**Negative result — v2 resign history features do nothing** (`resign_v2.log`, 3 players, 5-fold CV):
adding (jump in p_loss since my last turn, material just dropped, lost-streak length, p_loss x streak) moved AUC by
+0.002 at most (VEGETAL 0.914 -> 0.916, OKENITE 0.950 -> 0.949 so v1 was kept) and left the firing move unchanged
(VEGETAL -9.6 both, OKENITE -7.9 / -8.1). Whether a losing turn is the LAST one is close to unpredictable from these.
The hazard SHAPE is in fact right: per-turn resignation rate by how many turns the side has been lost, real vs fitted —
VEGETAL 1.35/1.76/2.20/3.75/5.18 % vs 1.18/1.69/2.45/3.50/5.07 % (streak 0 / 1-2 / 3-5 / 6-10 / 11-20), OKENITE
0.23/0.38/0.56/0.45/1.02 vs 0.21/0.30/0.43/0.53/1.09. The feature set is fine; the model is memoryless, and the
"-10 moves" number came from conditioning on games that ended in resignation.

**Where the resignation really differs (Stockfish depth 12 on the final position, real vs clone):**
VEGETAL real median **-838 cp** (p25 -2995, so a quarter are mate scores), clone **-649** (p25 -775);
OKENITE real **-2997** (over half his resignations have a forced mate on the board), clone **-852** (p25 -990).
Material at resignation matches (-8 / -6 and -11 / -10) and so does p_loss (real median 0.936 / 0.973), because the
value head SATURATES: it reads "down a rook, will lose" and "mate in 3" both as p_loss ~ 0.95. The clone therefore
resigns in merely-lost positions and almost never in a mating net, ~10 plies before the player would.
Unconditional first-passage on the player's own games confirms the size of the miss: the fitted hazard fires in 16.8 %
of VEGETAL's games (real 23.6 %) and 8.5 % of OKENITE's (real 11.4 %) at mean move 35.7 / 42.2 vs real 39.5 / 45.2 —
right shape, mass spread too thin across a long lost stretch instead of concentrated where mate is forced.
Next: cheap mate-danger features (in check, legal-move count, king escape squares, opponent's checking replies) that
the value head cannot express, computable identically in the fit and in the engine (no search).

**Median-length gap decomposed (clone arm of eval9, reweighted to the player's real ending mix, then shifted to the
real within-class medians):**

| player | clone as-is | + real ending mix | + within-class shift | real |
|---|---|---|---|---|
| VEGETAL | 78 | **92** | 102 | 99 |
| OKENITE | 86 | 87 | **95** | 99 |
| kpowe52 | 89 | 84 | 79 | 80 |

So the two failures have different causes. VEGETAL's is mostly the ENDING MIX, and the mix is set by the simulated
opponent, not the clone: the field opponent resigns in 24.0 % of games (real opponents 7.5 %) at median 42 plies
(real 78), and flags in only 9.3 % (real 28.8 %) — its shortest games flood the pool. Fixing the mix alone already
clears the +-12 bar (92 vs 99). OKENITE's is mostly WITHIN-class: his clone's own resignations at median 79 vs his 88,
plus mated 79 vs 82 and flags 114 vs 124; the mate-danger refit is aimed exactly there.
Note the bar reads the MEDIAN while the earlier per-class diagnosis used means; both point the same way.

**v3 mate-danger resign fit (`resign_v3.log`, 5-fold CV, picked by AUC):** small but consistent gain, not a fix.
AUC v1 -> v2 -> v3: VEGETAL player 0.914 / 0.916 / **0.922**, field 0.853 / 0.856 / **0.858**; OKENITE player
0.950 / 0.949 / **0.952**, field 0.843 / 0.844 / **0.844**; kpowe52 field 0.824 / 0.840 / **0.840** (his player model
stays the constant rate — 0 resignations in 2,500 games). All six now carry the danger features except kpowe52's own.
But the per-GAME firing rate barely moved and is still ~30 % low everywhere: VEGETAL 16.9 % vs his real 23.6 %,
OKENITE 8.6 % vs 11.4 %, fields 10.4 / 14.9 / 11.5 % vs 12.6 / 18.1 / 13.9 %, and the mean firing move went
35.7 -> 35.7 (VEGETAL) and 42.0 -> 42.6 (OKENITE) against real 39.5 / 45.2. So "which losing turn is the last one"
stays mostly unpredictable; the engine's resign RATE only comes out right because simulated games present more
hazard than the player's real games did.

**Why the simulated opponent never flags — it is variance, not pace.** Opponent clock at own move 20/30/40, real
opponents vs the field model: VEGETAL 124/78/44 vs 119/75/42, OKENITE 120/79/59 vs 115/70/40, kpowe52 120/75/44 vs
119/70/38 — the medians MATCH. But the share of games where the opponent ever drops under 10 s is VEGETAL 48.8 %
real vs **23.0 %** simulated, OKENITE 47.5 vs 38.7 (kpowe52 23.8 vs 30.7, the other way). Same median clock, half the
time-trouble: the think-time model produces games that are too alike, missing the across-GAME spread of a human who
burns three minutes in one game and premoves the next. That across-game variance is what turns into opponent flags
(real 28.8 % of VEGETAL's games vs 9.3 % simulated) and it is the single biggest lever on his game-length bar.

## Round 5 — identity as a learned head, not a scalar and not a feature list

Why the norm push was the wrong shape: `clone + w*(clone - stock)` is linear extrapolation in log-space with ONE
scalar for every position, phase and clock (borrowed from classifier-free guidance; nothing about the player
justified it), and it was fitted against self-play ACPL, which swings ~7 points on the seed alone.

First, fitting the push against the player's REAL moves instead (`clonefish_push_fit_moves.py`, 5-fold CV by game on
the 80 never-trained games, families: linear w / affine a*lc+b*lb / gap-power / per-phase affine). VEGETAL, all moves:
clone as-is NLL 1.2394, top1 58.68 %, top3 85.48 %; the BEST of every family is NLL 1.2311, top1 59.03 % — i.e.
+0.35pp top1 for the whole family, and the fitted w is **negative** (-0.038), meaning the likelihood wants the clone
pulled slightly TOWARD the norm, not away from it. A scalar push cannot express this person.

So: `clonefish_identity_head.py` — freeze the model, hook the trunk's 64 square encodings, and train a small
policy-shaped head (from/to bilinear + promotion, over dim_vit 512, hid 32, 34,820 params) whose only job is how this player deviates
from the frozen model underneath. Initialised as a no-op, so it must earn every deviation; no move features are
named by hand — whatever pattern it needs it has to find in the trunk's own representation. Identity = the head's
weights; `lambda` scales how pronounced it is: `score = base_logits + lambda * head(enc)`.
Smoke test (only 150 training games, VEGETAL): on the STOCK model NLL 1.5155 -> 1.4172 and top1 51.76 -> **54.24 %**,
so the head really does extract the person from the norm. On the CLONE it hurts (58.68 -> 56.95 %) — the clone
already carries that personalisation and the head double-counts it. Full runs (1,200 games, both bases, lambda
sweep) queued; next is the self-play half of the idea — the clone playing the stock, so the personal patterns get
more pronounced where the two disagree.

**Identity head result (VEGETAL, 1,200 training games, 4 epochs, tested on the 80 never-trained games):**

| lambda | on frozen STOCK | on CLONE |
|---|---|---|
| 0 (plain) | 1.5155 / 51.76 % / 80.11 % | 1.2394 / 58.68 % / 85.48 % |
| 0.5 | 1.3668 / 55.16 / 82.44 | 1.2616 / 57.97 / 84.65 |
| **1.0** | **1.3216 / 56.80 / 83.10** | 1.3418 / 56.04 / 83.96 |
| 1.5 | 1.3651 / 55.97 / 82.29 | 1.4737 / 53.41 / 81.88 |
| 2.0 | 1.4786 / 53.23 / 81.10 | 1.6481 / 50.67 / 79.66 |
| 3.0 | 1.8662 / 47.63 / 77.00 | 2.0933 / 46.87 / 74.74 |

(NLL / top-1 / top-3.) Two findings:
1. **A 34.8k-param head on a FROZEN model recovers +5.04pp of the +6.92pp that a full 19.5M-param fine-tune buys**
   (51.76 -> 56.80 vs the clone's 58.68). Identity is compact and learnable without touching the trunk — a candidate
   cheap per-player personalisation, and a candidate fingerprint.
2. **Amplification does not work.** lambda = 1 (maximum likelihood) is the optimum on stock and every lambda > 0 hurts
   on the clone, monotonically. Making the personal signal "more pronounced" than the person's own moves support
   makes a WORSE model of them. This is the same wall the scalar push hit (the real-move fit chose w = -0.038,
   i.e. slightly toward the norm) — now reproduced by a nonlinear learned model that was free to find any direction.
   Two independent parameterisations agreeing => it is a property of the problem, not of the linear form.
Caveat kept open: top-1 is not the goal. A push can hurt likelihood and still help GAME-level realism (NormPush 0.5
repaired Soroka's too-clean clone). VEGETAL needs no such repair (clone ACPL 73.3 vs his 68.4 — already sloppier),
so amplification must be judged on a clone that is measurably off, at game level, not on move match.
Next: `--cv` (5-fold inside the never-trained window) for base=clone — the only contamination-free test of whether
identity REMAINS beyond the clone, since the head's normal training data is inside the clone's fine-tuning set.

**Engine integration.** `IdentityHead` / `hook_enc` / `load_identity` now live in `clonefish_uci.py` (one definition,
imported by the trainer), `CloneEngine(..., identity=head)` hooks the trunk and applies `lambda * head(enc)` to the
legal-move scores, UCI option **IdentityPush** (/100, default 0 = off), CLI `--identity`. Verified on CPU: the head
round-trips (dim_vit 512, hid 32, 34,820 params), IdentityPush changes the move probabilities and IdentityPush 0
reproduces the plain clone exactly. Visible in that check: a stock-trained head applied to the CLONE at lambda 1
SHARPENS hard (start position p 0.85 -> 1.00) and in one position swung the move to f2f3 at p 0.91 — the same
double-counting the lambda sweep measured (58.68 -> 56.04 % top-1). So IdentityPush stays default 0, and the open
question is only whether a small lambda helps GAME-level stats on a clone that is measurably off (Alexei_Soroka).

**What the head actually learned (read-out, not input).** Applying VEGETAL's stock-trained head to 300 held-out
positions and averaging its logit boost per legal move, position-centred, then DESCRIBING the result with the
move_habits families (the head never saw these categories — it found the patterns in the trunk's own square
representations; the families are only a read-out):

    castle +0.785 (17 pos) · rook +0.283 (199) · retreat +0.225 (233) · bishop +0.197 (177) · recapture +0.112 (78)
    queen -0.008 · pawn -0.113 · edge-pawn push -0.121 · king non-castle -0.156 · into enemy half -0.245
    knight -0.269 (199) · capture -0.315 (214) · check -0.535 (101)

A coherent, recognisable style versus the norm at his rating: castles, plays rooks and bishops, retreats instead of
lunging, and declines the forcing moves (checks, captures, advances) the average player reaches for. The castle
number rests on 17 positions and should not be leaned on; rook / retreat / knight / capture / check have 100-233.
So identity IS extractable as a learned model, compact (34,820 params on a frozen trunk) and legible after the fact.

**Clean CV verdict (5-fold inside the 80 never-trained games).** On the CLONE, every lambda > 0 still hurts
(58.68 -> 58.42 at 0.5, 57.64 at 1.0, 56.35 at 1.5) — no identity remains beyond the fine-tune. Caveat being
controlled for: a CV head trains on only ~3,100 moves vs ~54,000 in the full run, so "no residual" could be "too
little clean data"; the base=stock CV (same tiny training set, model with a known large residual) is the control.

**Control settles it: no identity remains beyond the clone.** 5-fold CV inside the never-trained window, so BOTH
heads train on the same ~3,100 moves — the only difference is which frozen model they correct:

| lambda | head on STOCK (control) | head on CLONE |
|---|---|---|
| 0 | 1.5155 / 51.76 % | 1.2394 / 58.68 % |
| 0.5 | 1.4236 / 53.08 | 1.2619 / 57.87 |
| **1.0** | **1.4102 / 54.29** | 1.3384 / 56.45 |
| 1.5 | 1.4819 / 51.58 | 1.4685 / 53.10 |

Same data size, opposite outcomes: **+2.53pp on stock, negative on the clone.** So "too little clean data" is ruled
out — the full fine-tune has genuinely absorbed the identity, and a head of this class finds nothing left. Note the
stock head still peaks at lambda = 1 and falls off hard by 1.5, reproducing the no-amplification result on clean data.

## Round 6 — the amplification tests were CIRCULAR; the honest test is generated-vs-real

Correction to rounds 4-5. Every push/identity measurement so far was scored by **move likelihood on the player's
real positions** — which is exactly the objective the clone was fine-tuned with. So "is lambda > 1 better than
lambda = 1" reduces to "is the MLE not the MLE", and the answer is no BY CONSTRUCTION, for any parameterisation.
That is why the linear scalar (best w = -0.038) and the nonlinear learned head (peak at lambda = 1, monotone
decline after) agreed so exactly: both were graded by the loss that produced the thing they were correcting.
It was never a test of the self-play idea.

What the clone actually gets wrong is NOT per-move likelihood — it is the generated games: 21 plies short
(VEGETAL 78 vs 99), resignations at -649 cp where the player waits for -838 / -2997, opponent-flag share 9.3 % vs
28.8 %. None of that is visible to move match, and self-play data is precisely the data maximum likelihood never sees.

So `clonefish_discriminator.py`: train D(position, move) -> "player or clone" on the clone's GENERATED games vs the
player's real ones (same frozen-trunk + small policy-shaped head, but the label is PROVENANCE, not the move).
- AUC ~ 0.5 => the clone's moves are already indistinguishable and there is nothing to amplify; the idea closes
  empirically instead of by a circular argument.
- AUC >> control => D has located where the clone stops looking like the player, and `clone + beta * D_logit` is a
  learned, position-DEPENDENT correction — unlike `clone - stock`, which only points away from a norm the fine-tune
  has already moved away from. beta then gets fitted on GAME-level stats (ACPL, W1, length, endings), not likelihood.
**Control (required):** real games are played against real opponents and generated ones against the field model, so D
could be reading the OPPONENT rather than the player. The real-vs-real control (the player's own games split in half
and labelled as two sources) is the floor any claimed signal must beat; a gap > 0.03 AUC counts as signal.
Runs on data we already have: the eval9 CLONE arms (150 games per player) + their real games. No new generation.

**Game-level test of the identity head (Alexei_Soroka, eval10_ident.json, 100 games/arm, SF on 80).**
IDENT = frozen base + his identity head at lambda 1 (the arm that could ADD mistakes rather than sharpen them away).

| arm | ACPL / blunder % | think W1 | instant % | lost on time | plies |
|---|---|---|---|---|---|
| REAL | 76.1 / 5.3 | — | 8.2 | 23.8 % | 84 |
| CLONE | 67.2 / 4.0 | **0.0521** | 10.4 | 14.0 % | 74 |
| IDENT | 63.9 / 3.9 | 0.2063 | 17.4 | 5.0 % | 83 |
| BASE | 74.6 / 5.2 | 0.2108 | 16.7 | 3.0 % | 74 |

**The plain BASE model matches his mistake profile best (74.6 vs his 76.1); his fine-tuned clone is 9 points too
clean, and adding the identity head makes it CLEANER STILL (63.9, gap -12.2).** Every step of fitting his real moves
made the engine play BETTER than he actually plays. That is structural, not a bug: likelihood training reproduces a
player's MODAL move in positions where they were doing fine, not the errors they make under pressure in the states
the engine's own play wanders into. Same circularity as round 6, now visible at game level.
IDENT also fails timing by construction (W1 0.21 vs clone 0.05, instant 17.4 vs his 8.2): the head corrects the move
policy only and inherits the base think-time head. So stock+head is NOT a replacement engine — its value is compact
per-player personalisation on a frozen trunk (34.8k params) and the legible style read-out.
Eval harness fixed: the Stockfish phase and the summary both iterate ("REAL","CLONE","IDENT","BASE") now — the IDENT
arm was generated but silently unscored on the first run.

**Discriminator, first pass (UNCONTROLLED — do not trust these numbers yet).** D(position, move) -> player or clone,
trained on the eval9 CLONE arms (150 generated games) vs 400 real games, balanced, split by game, 3 epochs:

| player | real-vs-generated AUC | real-vs-real control | moves/side |
|---|---|---|---|
| VEGETAL | 0.570 (rose 0.528 -> 0.559 -> 0.570) | 0.484-0.520 | 5,688 |
| kpowe52 | 0.656 (0.610 -> 0.633 -> 0.656, not plateaued) | 0.493-0.515 | 6,218 |
| OKENITE | **0.829** (0.801 -> 0.820 -> 0.829) | 0.457-0.475 | 6,416 |

The real-vs-generated AUC rises monotonically while the control wanders with no trend, so SOMETHING is separable.
But 0.83 is too good, and two confounds can produce it without any difference in move CHOICE:
1. **Elo leak.** `rows_of` feeds elo_self/elo_opp into the trunk (Elo embedding -> enc). Real games carry each game's
   true ratings; generated games all use one fixed average. D can read "Elo == the mean" and win.
2. **State drift, not move choice.** D scores a per-move logit but sees the whole position: the clone reaches
   different states (different opponent, games 21 plies shorter), which is separable with zero knowledge of moves.
Both now controlled: Elo inputs forced to the same median on BOTH classes, plus a **position-only** discriminator
(`PosDisc`, mean-pooled enc -> 1, never sees the played move). The usable quantity is **move AUC - position AUC**;
only that part can steer a move policy. If the gap is ~0, the separability is state drift — which would point the fix
at the resign model / time head / opponent model (where the game-level failures already live), not at move scores.

**Discriminator, CONTROLLED (Elo equalised on both classes + a position-only control). The leak was most of it.**

| player | move AUC | position-only | real-vs-real floor | usable (move - position) |
|---|---|---|---|---|
| VEGETAL | 0.499 (was 0.570) | 0.508 | 0.504 | **-0.009** |
| kpowe52 | 0.574 (was 0.656) | 0.579 | 0.510 | **-0.004** |
| OKENITE | 0.610 (was 0.829) | 0.644 | 0.477 | **-0.034** |

**No move-choice direction exists.** In all three, knowing WHICH MOVE was played adds nothing over the position
alone, so there is nothing to steer a move policy with — `clone + beta * D_logit` has no signal to carry. The
self-play idea is now tested in its strongest, non-circular form and closed at move level.

**But the same instrument found the real gap: STATE DRIFT, and it is per-player.** Position-only AUC above the
real-vs-real floor measures how different the SITUATIONS the clone gets into are, independent of its moves:
VEGETAL +0.004 (his clone's states are statistically his), kpowe52 +0.069, OKENITE **+0.167**.
This cross-checks the game-level failures exactly: VEGETAL's length bar fails because of the OPPONENT (it resigns in
24 % of games vs real opponents' 7.5 %) and the discriminator only sees HIS moves/positions — so his own states test
clean, correctly. OKENITE's clone drifts on its own account, and his resignations fire at -852 cp where he waits for
-2997. => `position-only AUC - real-vs-real floor` is a cheap single-number per-player diagnostic of clone realism
that needs no Stockfish and no pre-registered bars. Fix the states (timing / resign / opponent), not the move scores.

## Round 7 — amplification judged by TOP-K (the non-circular version), with self-play over many openings

The round-6 circularity argument kills likelihood-based tests, but it does NOT kill the idea, because **top-k
accuracy is a different objective from log-loss** (Lapin 2015). Every earlier sweep FITTED by NLL and only reported
top-k afterwards; a push can improve top-3 while hurting NLL. `clonefish_amplify_topk.py` selects w BY TOP-3.

Design, with each half doing only what it can:
- **self-play (clone vs stock, forced through 10 different opening lines** via the new `simulate(..., start=)`):
  no ground-truth player move, so it cannot score anything. What it CAN say is where the two models disagree and how
  often each kind of position actually arises — and amplification can only ever change a move where they disagree,
  so that share is the ceiling on how often this can matter at all.
- **the player's real held-out games**: the only place top-1/top-3 exists.
Segmented by phase (<10 / 10-25 / >25), by whether clone and stock agree, and by opening (1.e4 / 1.d4), because a
signal that helps in sharp positions can vanish when averaged over quiet ones.
w is selected **per fold, 5-fold CV by game**, so the reported gain is what you would get picking w on other games
rather than the best cell in the table. Grid spans negative w (toward the norm) as well as positive.

**v3 resign models at game level (eval11_v3resign, 150 games/arm) — no improvement, as predicted.**

| | median plies (real) | resign % (real) | lost on time % (real) | think W1 |
|---|---|---|---|---|
| VEGETAL | 78 (99) — unchanged from eval9 | 20.7 (20.0) | 10.0 (15.0) | 0.0184 |
| kpowe52 | 88 (80) PASS | 0.0 (0.0) | 24.7 (15.0) | 0.029 |
| OKENITE | 84 (99) — was 86 | 10.7 (7.5) | 8.7 (10.0) | 0.0296 |

Bars: resign **3/3**, timing **3/3**, length **1/3** (unchanged), lost on time 2/3 (kpowe52 slipped from 22.0 to 24.7 %
against his 15.0 % — a borderline bar moving inside noise at 150 games).
Why this was predictable: the mate-danger features raised AUC (0.914 -> 0.922) but left the per-GAME firing rate
almost untouched (16.9 % vs his real 23.6 %), and game LENGTH depends on the firing rate, not on ranking quality.
**The remaining length gap is the OPPONENT model, not the player's resign model.** VEGETAL's clone wins by opponent
resignation 22.7 % of games (real opponents 7.5 %) and by opponent flag 13.3 % (real 28.8 %) — the simulated
opponent quits early and almost never runs out of time, so the pool is full of short games. Three independent
measurements now agree on this: the median-length decomposition (mix alone moves 78 -> 92), the opponent-pace
variance result (reaches <10 s in 23.0 % of games vs real opponents' 48.8 %), and the discriminator's state drift
(VEGETAL's OWN states test at the floor, 0.508 vs 0.504 — his side is fine, the opponent's is not).
Bug found and fixed: adding "IDENT" to the Stockfish arm loop without a presence guard crashed this run with
KeyError after phase 2 (`clonefish_eval.py` now skips absent arms). Phase 2 had already written timing/endings and
all 900 games were saved, so only the Stockfish pass needed redoing — from the saved PGNs, no regeneration.

**Final v3 scorecard (eval11, 150 games/arm, Stockfish on 80-100).** ACPL / blunder %, real / clone / base:
VEGETAL 68.4/3.8 · 70.4/4.5 · 70.5/4.4 — OKENITE 70.3/4.7 · **71.3/4.7** · 74.6/5.5 — kpowe52 80.9/5.6 · 76.0/4.8 · 67.6/4.5.
Bars: resign **3/3** · timing **3/3** · mistakes **2/3** · lost on time 2/3 · length **1/3**.

**Caveat on G3 — the bar moved because BASE moved, not the clone.** vs eval9, mistakes went 0/3 -> 2/3 while the
CLONE arms barely changed (kpowe52 74.4 -> 76.0, OKENITE 71.4 -> 71.3); it was the reference arm that swung
(kpowe52 BASE 77.8 -> 67.6, OKENITE BASE 67.7 -> 74.6). G3 is defined as "closer to REAL than BASE", so a 7-10 point
swing in BASE flips the verdict by itself. **G3 measures the reference arm's variance as much as the clone's
fidelity** — report the clone-vs-real gap directly instead (VEGETAL +2.0, OKENITE +1.0, kpowe52 -4.9).
Also re-confirmed: Stockfish drifts ~1.4 ACPL on the SAME 80 games between runs (kpowe52 REAL 79.5 -> 80.9) because
hash state carries across sequential analyse() calls — the known ~2-point floor on any mistake comparison.

**Round 7 RESULT — amplification selected by TOP-3 fails too, on all three players.**
Self-play: clone vs stock over 10 opening lines; they pick DIFFERENT moves in 26.2 % (VEGETAL) / 32.0 % (kpowe52) /
32.0 % (OKENITE) of the clone's own positions — that share is the ceiling on how often any w could change anything.
Held-out real games, w chosen per fold by top-3 (5-fold CV by game):

| player | all moves top1 / top3 | clone!=stock segment | w the CV picked |
|---|---|---|---|
| VEGETAL | +0.33 / **-0.15** | top1 +0.09, top3 **-0.45** | 0 / -0.25 |
| kpowe52 | -0.73 / -0.06 | **exactly 0.00 / 0.00** | **0 in all 5 folds** |
| OKENITE | +0.00 / +0.00 | **exactly 0.00 / 0.00** | **0 in all 5 folds** |

Given a free choice on held-out games the criterion picks **w = 0**, and when it strays it goes NEGATIVE (toward the
norm). Top-3 falls in nearly every VEGETAL segment; kpowe52's small top-1 "gains" come with top-1 losses overall and
sign-flipping per-fold picks = fitting noise. **This was the non-circular test** (top-k is not the training loss, and
the strength was cross-validated), run exactly as specified — many openings, self-play to locate the disagreements —
and the amplification direction does not exist at any strength, in any segment, for any player.
Useful by-product: where clone and stock AGREE the clone's top-1 is 62-67 %; where they DISAGREE it is 43-54 %.
Model disagreement tracks genuine position difficulty — a point in the clone's favour, but nothing to amplify.

**Opponent-side discriminator — the two length failures have OPPOSITE causes.** Same method, now judging the
simulated opponent's moves against the player's REAL opponents (`--side opp`). State drift = position-only AUC
above the real-vs-real floor:

| player | own-side drift | opponent-side drift | length bar |
|---|---|---|---|
| VEGETAL | **+0.004** (0.508 vs 0.504) | **+0.080** (0.576 vs 0.496) | fails 78 vs 99 |
| kpowe52 | +0.069 (0.579 vs 0.510) | +0.091 (0.567 vs 0.476) | passes 88 vs 80 |
| OKENITE | **+0.167** (0.644 vs 0.477) | +0.053 (0.573 vs 0.520) | fails 84 vs 99 |

VEGETAL's clone is faithful and his OPPONENT is wrong; OKENITE's OWN clone drifts 3x more than his opponent does
(it resigns at -852 cp where he waits for -2997, and is mated in 27.3 % of games vs his 21.2 %). kpowe52 is middling
on both and correspondingly passes length while failing lost-on-time. So **one cheap number per side localises which
half of the simulation to fix** — no Stockfish, no pre-registered bars, ~3 min per player.
In all SIX measurements (3 players x 2 sides) move AUC - position AUC is <= +0.010: there is never a move-choice
signal, only state drift. That is the same verdict the top-k amplification test reached by a completely different route.

**Next, per player:** VEGETAL -> the opponent model (it resigns 22.7 % of games vs real opponents' 7.5 % and flags
13.3 % vs 28.8 %; its clock MEDIANS already match, so what is missing is across-GAME pace variance — real opponents
reach <10 s in 48.8 % of games, the field model in 23.0 %). OKENITE -> his own clone's state generation.

## Round 8 — per-game pace spread (PaceSigma), bar written BEFORE the result

Hypothesis, from three converging measurements: the simulated opponent's clock MEDIANS already match real opponents
(VEGETAL 119/75/42 at moves 20/30/40 vs real 124/78/44) but it reaches under 10 s in **23.0 % of games vs real 48.8 %**.
Same median, half the time-trouble => the time head has the right WITHIN-game spread and too little BETWEEN-game
spread. Humans burn three minutes in one game and premove the next. That missing tail is what produces flags (real
opponents flag in 28.8 % of VEGETAL's games, the simulated one in 13.3 %), and flags are what keep games long — which
is the whole of his length gap (78 plies vs 99; reweighting to his real ending mix alone moves it to 92).

Mechanism: draw ONE pace factor per game, `exp(sigma * N(0,1))`, and multiply that game's sampled think times by it
(applied after the book readings too, so a slow game is slow throughout and a 0 s premove stays instant).
Engine option `PaceSigma` (/100), default 0 — verified that 0 gives pace exactly 1.0, so nothing already passing can
regress. `scripts/clonefish_pace_fit.py` sweeps it on the OPPONENT engine.

**Pre-registered bar. The fix counts only if BOTH hold:**
1. the opponent's under-10 s share moves from 23 % toward the real 48.8 % (within ~10 points), and
2. **the player's think-time W1 does not get worse** (currently 0.018; it is already the strongest result we have).
Rationale for (2): if the TOTAL think-time variance is already correct and merely misallocated between within- and
between-game, then adding between-game spread will widen the marginal and damage W1. A sigma that buys flags by
wrecking the timing distribution is not a fix, and I will report it as a failure rather than trade one bar for another.

**Round 8 RESULT — PaceSigma FAILS its pre-registered bar. Default stays 0.** (60 games per sigma, VEGETAL.)

| sigma | opp under10s (real 48.8) | opp flags (real 28.8) | player W1 (0 -> 0.0187) | median plies (real 99) |
|---|---|---|---|---|
| 0.00 | 30.0 | 13.3 | **0.0187** | 81 |
| 0.20 | 21.7 | 13.3 | 0.0306 | 76 |
| 0.35 | 13.6 | 6.8 | 0.0658 | 59 |
| 0.50 | 28.3 | 25.0 | 0.029 | 70 |
| 0.70 | 28.3 | **28.3** | 0.029 | **59** |

Both conditions violated at every sigma: time trouble never moves toward 48.8 %, and W1 is worse everywhere.
**The informative cell is sigma 0.70: it matches the opponent's flag rate almost exactly (28.3 vs 28.8) and makes
game length WORSE (59 plies vs 81 at sigma 0, real 99).** So you can match the flag RATE and still lose on length —
real opponents flag at the end of long grinding games, while a per-game pace multiplier makes them flag EARLY, which
ends the game sooner. Causality runs length -> flags, not flags -> length, and a symmetric multiplier also makes half
the games faster, so those never reach a scramble at all. The under-10s series (30.0 / 21.7 / 13.6 / 28.3 / 28.3)
bounces like noise at n=60 — consistent with the standing "never tune on < 80 games" rule.

**So the lever is the opponent's ENDING MIX, not its pace.** The simulated opponent resigns in 22.7 % of VEGETAL's
games where his real opponents resign in 7.5 % — conditional on losing, 52 % vs 19 %, i.e. 2.7x too often — and that
is what ends games before anyone reaches a time scramble. The length decomposition already said fixing the mix alone
moves his median 78 -> 92 (bar needs >= 87). Next: calibrate the FIELD resign hazard in simulation (a single logit
offset, `ResignBias`) so the simulated opponent's resign-given-loss matches real opponents', then re-measure length.

## Round 9 — the 80-game TARGETS are too noisy to carry the bars (this undercuts rounds 4-8)

Measuring the same quantities on the held-out 80 vs a large window of the player's own games:

| VEGETAL, opponent stats | held-out 80 | 1,000 games |
|---|---|---|
| opponent resigns | 7.5 % of games (19.4 % of losses) | **13.8 % (36.8 %)** |
| opponent flags | 28.8 % | **18.2 %** |
| opponent under 10 s | **48.8 %** | **30.1 %** |

The simulated opponent's under-10 s rate is 26-30 %, i.e. it already MATCHES the 1,000-game figure (30.1 %). The
"opponent never gets into time trouble" problem that drove all of round 8 was mostly an artifact of an 80-game
sample. That is also why PaceSigma could not fix it — there was far less to fix than the target claimed.

**Median game length, the bar that has failed all session:**

| player | held-out 80 | last 500 | last 1,500 | all 5,920 | clone | verdict vs 99 | vs 90 |
|---|---|---|---|---|---|---|---|
| VEGETAL | **99** | 91 | 90 | 90 | 78 | fail (21) | **pass (12)** |
| OKENITE | **99** | 92 | 91 | 90 | 84 | fail (15) | **pass (6)** |
| kpowe52 | 80 | 86 | 82 | 83 | 88 | pass | pass |

**Against a well-estimated target the length bar is 3/3, not 1/3.** Honest caveat: the bootstrap 95 % CI on the
held-out median is [94, 110] for VEGETAL, which excludes 91, so a genuine recent drift toward longer games cannot be
ruled out — but a target that moves 9 plies with the window is too fragile for a +-12 bar either way.

**Consequence for method:** game-level TARGETS (median length, ending mix, opponent behaviour) should be estimated
from the player's whole history, not from the 80 held-out games. Contamination is not an issue here: the clone
training on those games matters for MOVE-MATCH, but the target distribution for "do the generated games look like
this person's games" simply IS the person's game distribution. Keep the held-out 80 only where contamination bites
(top-k, and the think-time reference).
This does not retract round 8's result — PaceSigma still failed on its own terms (W1 worse everywhere, length worse)
— but it does mean the problem it was attacking was largely measurement noise.

**Re-scored: with whole-history targets the current config passes every game-level bar.** Same clone arms, same
generated games, only the TARGET re-estimated (1,500 games instead of the held-out 80):

| bar | eval9: 80-game -> whole-history | eval11 (v3): 80-game -> whole-history |
|---|---|---|
| R2 resign | 3/3 -> 3/3 | 3/3 -> 3/3 |
| R3 lost on time | 3/3 -> 3/3 | 2/3 -> **3/3** |
| R3 length | 1/3 -> 2/3 | 1/3 -> **3/3** |

Per player (eval11, clone vs whole-history target): VEGETAL resign 20.7 vs 23.7, flag 10.0 vs 14.4, plies 78 vs 90;
kpowe52 0.0 vs 0.0, 24.7 vs 28.9, 88 vs 82; OKENITE 10.7 vs 10.5, 8.7 vs 11.9, 84 vs 91.
Note kpowe52's lost-on-time target is 28.9 % over 1,500 games vs 15.0 % on the held-out 80 — his clone's 24.7 % was
called a FAILURE against the noisy target and is a PASS against the real one. Four rounds of work (v2/v3 resign
features, PaceSigma, the whole opponent-pace investigation) were chasing a gap that was mostly measurement error.

**ResignBias -50 is a genuine improvement on top of that** (80 games/setting): opponent resign-given-loss
51.4 % -> 29.0 % (real 36.8 %, so -50 slightly overshoots; the optimum is between 0 and -50), median plies
**78 -> 87** against a 90 target, and the player's think W1 IMPROVES 0.0217 -> 0.0124 — longer games sample more of
the clone's time distribution in the phases where it is accurate. Caveat before banking it: an 80-game median
carries ~+-8 plies, so 78 -> 87 is about one CI width; monotonicity across the rest of the grid is the check.

**ResignBias sweep on VEGETAL's opponent (80 games/setting), against WHOLE-HISTORY targets:**

| bias | resign-given-loss (real 36.8) | opp flag (18.2) | under10s (30.1) | plies (90) | player W1 (0.0217) |
|---|---|---|---|---|---|
| 0 | 51.4 | 13.8 | 27.5 | 78 | 0.0217 |
| **-50** | 29.0 | 13.8 | **31.2** | **87** | 0.0124 |
| -100 | 30.4 | 13.9 | 31.6 | 94 | 0.0233 |
| -150 | 13.8 | 17.5 | 42.5 | 96 | 0.0316 |

Length is cleanly monotone (78 -> 87 -> 94 -> 96), so the effect is real, not a lucky draw. **But no single bias
matches everything:** rate-matching wants ~-25, length-matching ~-70, flag-matching -150. The script's own pick
(-150, closest on length) is rejected by its own pre-registered rule — W1 degraded 0.0217 -> 0.0316.
-50 is the defensible compromise (length 87 vs 90, time trouble 31.2 vs 30.1, W1 no worse), but it UNDER-resigns
(29.0 vs 36.8), i.e. it buys length by making the opponent stubborner than real opponents are. W1 is non-monotone
(0.0217 / 0.0124 / 0.0233 / 0.0316), so the apparent W1 *gain* at -50 is inside noise at n=80.
**Not adopting it off this sweep** — the standing rule is never tune on one seed at 80 games. The test that decides
it is a full 150-game eval at -50 against eval11 (bias 0), which is simultaneously the verification and the
deliverable. Added `--opp-resign-bias` for that. Remember: with corrected targets VEGETAL already PASSES length at
bias 0 (78 vs 90 = the 12-ply bar edge), so this is centring, not rescue.

## VERDICT (2026-09-16) — ResignBias REJECTED (the 14/15 figure in this section was later corrected to 13/15)

150 games/arm, three players, whole-history targets:

| bar | eval11 (opponent bias 0) | eval12 (bias -50) |
|---|---|---|
| R2 resign | 3/3 | 3/3 |
| R3 lost on time | **3/3** | 2/3 |
| R3 length | 3/3 | 3/3 |
| G2 timing | **3/3** | 2/3 |
| G3 mistakes | **2/3** | 1/3 |

**-50 helped VEGETAL** (plies 78 -> 87, W1 0.0184 -> 0.0122, ACPL gap +2.0 -> +1.7) **and hurt the other two**:
OKENITE W1 0.0296 -> 0.0373 (timing FAIL) and ACPL gap +1.0 -> +4.4 (suppressing early opponent resignations drags
games into endgame/scramble phases where his clone errs more); kpowe52 lost-on-time 4.2 -> 8.9 points off target.
It was fixing a problem that had already dissolved — length was ALREADY 3/3 at bias 0 once targets were estimated
from the player's whole history — so it could only trade other bars away. The 80-game sweep that picked -50 was
precisely the one-seed tuning the standing rule forbids, and the 150-game test caught it. **Default stays 0.**

**Recommended config (unchanged): full fine-tune `--time-loss bucket` + personal book + sampled think-time +
`--resign` (v3 mate-danger models) + per-player field opponent, NormPush 0, IdentityPush 0, PaceSigma 0,
ResignBias 0.** Scorecard: resign 3/3 · lost on time 3/3 · length 3/3 · timing 3/3 · mistakes 2/3 — **superseded: timing is 2/3 once
the target window is adjudicated per statistic, so the config scores 13/15. See the ADJUDICATED section below.**
The single miss is VEGETAL's G3, which fails on blunder rate 4.5 % vs BASE's 4.4 % against his 3.8 % — a 0.1-point
difference inside Stockfish's ~0.8-point noise, and his clone's ACPL is the closer one (70.4 vs 70.5 against 68.4).
Clone-vs-real ACPL gaps, the number to report instead of the BASE-relative bar: VEGETAL +2.0, OKENITE +1.0,
kpowe52 -4.9.

**Generality: the correction holds on the fresh players too.** Re-scoring `phase3/eval_fresh.json` (ARQZITRO,
AlexDC, Alexei_Soroka — never used in development, ~1,120 games each) against whole-history targets instead of the
held-out 80:

| player | clone plies | held-80 target -> bars | whole-history target -> bars |
|---|---|---|---|
| ARQZITRO | 85 | 92 -> resign/time/length all pass | 87 -> all pass |
| AlexDC | 82 | 78 -> all pass | 75 -> all pass |
| Alexei_Soroka | 72 | 84 -> lost-on-time **fail** | 83 -> **all pass** |

Totals: resign 3/3, lost on time **2/3 -> 3/3**, length 3/3. So on players never touched during development the
corrected verdict is 3/3 on all three game-level bars, matching the dev players.

## CORRECTION — the "14 of 15" figure used INCONSISTENT targets

I re-scored the ENDINGS against whole-history targets but left the TIMING bar scored against the held-out 80. Applied
consistently, the timing bar moves the OTHER way:

| | clone instant % | real held-80 | real 1,500 games | W1 vs held-80 | W1 vs 1,500 |
|---|---|---|---|---|---|
| VEGETAL | 8.3 | 7.1 | **6.3** | 0.0184 | 0.0267 |
| kpowe52 | 11.2 | 10.8 | **8.3** | 0.029 | 0.0549 |
| OKENITE | 7.2 | 5.6 | 5.5 | 0.0296 | 0.0238 |

G2 timing: **3/3 against the held-80 target, 1/3 against the whole-history target.** So the honest consistent
scorecard is **12 of 15**, not 14: resign 3/3 · lost on time 3/3 · length 3/3 · timing **1/3** · mistakes 2/3.

**Which window is right is genuinely unclear for timing, and that differs from the length case.** Both players' recent
games are FASTER than their historical average in the same direction (instant % 7.1 vs 6.3, 10.8 vs 8.3) — that looks
like drift, and if a player really has sped up then the recent window is the correct target and the clone (fine-tuned
on the recent 5,000 games, Elo-conditioned) is right to match it. Length was not like this: 99 on the newest 80
against 91/90/90 on EVERY larger window is an outlier, not a trend.
Resolution being measured: the same statistics in time-ordered blocks. A monotone trend => drift => score timing
against the recent window; a flat series with the last 80 sticking out => noise => score against the long window.
Until that is settled, **do not quote a single headline number** — quote the bar and the target window together.

## ADJUDICATED — drift vs noise settled by time-ordered blocks; final scorecard 13/15

Same statistics in consecutive blocks of 500 games, oldest -> newest, with the held-out 80 last:

| instant % | 0-500 | 500-1k | 1k-1.5k | 1.5k-2k | 2k-2.5k | 2.5k-3k | held-80 | verdict |
|---|---|---|---|---|---|---|---|---|
| VEGETAL | 5.1 | 5.0 | 5.5 | 6.0 | 6.3 | 6.7 | **7.1** | monotone => **real drift** |
| kpowe52 | 8.6 | 8.2 | 8.8 | 8.8 | 7.8 | 8.1 | **10.8** | flat, last sample jumps => outlier |
| OKENITE | 6.4 | 5.3 | 5.2 | 5.1 | 5.8 | 5.6 | 5.6 | flat, windows agree |

**Rule: score against the long window unless the statistic shows a monotone trend, then score against the recent one.**
- VEGETAL timing: he is genuinely speeding up, so the recent target (7.1) is fair; clone 8.3 -> gap 1.2 **PASS**.
- kpowe52 timing: no trend, so the long target (8.3) is fair; clone 11.2 -> gap 2.9 **FAIL** (too many instant moves).
  Caveat: with move-level clustering that final jump is only ~1.5-2 SE, so this one is not airtight either way.
- OKENITE timing: 5.5/5.6 either way; clone 7.2 -> gap 1.6 **PASS**.

The same blocks independently confirm the LENGTH correction: plies are flat (VEGETAL 93/91/83/88/89/91,
OKENITE 88/87/94/90/91/92) with the held-out 99s as clear outliers — so the long target (90/91) was right there.

**FINAL: 13 of 15.** resign 3/3 · lost on time 3/3 · length 3/3 · timing **2/3** · mistakes 2/3.
(I reported 14/15 earlier from mixed targets, then 12/15 from uniformly-long targets; 13/15 is the adjudicated
number and the one to quote — with the window rule above, never as a bare headline.)
Remaining real defects: kpowe52's clone plays too many instant moves (11.2 % vs ~8.3 %), and OKENITE's own-side
state drift (+0.167) is still unexplained given that he now passes every bar.

## The time heads are faithful — the instant-move excess is STATE drift, not timing

Teacher-forced on each player's 80 held-out real games (`clonefish_pace_check.py`, sampled MDN readings; absolute
rates are not comparable to the free-play table, only real vs clone vs base within a row):

| player | instant % real / clone / base | first-30 pace sum real / clone |
|---|---|---|
| VEGETAL | 5.2 / **4.1** / 10.6 | 101.4 / 99.6 |
| kpowe52 | 8.5 / **6.6** / 10.9 | 109.4 / 113.1 |
| OKENITE | 3.5 / **3.9** / 10.1 | 90.9 / 89.1 |

Every clone tracks its player closely and, if anything, plays slightly SLOWER than they do; the base model sits at
10.1-10.9 % instant for all three. **So kpowe52's free-play excess (11.2 % vs his ~8.3 %) is not a time-head defect:
teacher-forced his head is conservative at 6.6 % vs his 8.5 %.** The excess appears only in self-play, i.e. his
clone's games reach more low-clock states than his real games do.

**This closes the loop across four independent measurements this session**, all reaching the same conclusion by
different routes: the discriminator (move AUC - position AUC <= +0.010 on 3 players x 2 sides — only state drift),
the top-k amplification test (no move-level direction at any strength, in any segment), the length decomposition
(the ending MIX, not within-class play), and now the pace check (heads faithful, free-play excess situational).
**The clones' POLICIES — which move, how long to think — are faithful. What differs is the distribution of
SITUATIONS their games produce**, which is governed by the simulated opponent and the self-play feedback loop.
And state-matching is not a simple knob: the one intervention tried on that side (ResignBias) helped one player and
broke two.

**State drift re-measured on the v3 games** (`discriminator_v3.json`; drift = position-only AUC - real-vs-real floor):

| player | eval9 games | v3 games | passes all 5 bars? |
|---|---|---|---|
| VEGETAL | +0.004 | +0.012 | yes |
| kpowe52 | +0.069 | +0.054 | yes (timing fails on the long window) |
| OKENITE | **+0.167** | **+0.120** | **yes** |

Not attributing the drops to the v3 resign models: different generated games and a freshly trained discriminator
each time, so changes this size are not separable from run-to-run variation. What IS solid is that the **ranking is
stable across two independent sets of games** (OKENITE >> kpowe52 > VEGETAL), and that OKENITE's +0.120 persists
while he passes every pre-registered bar — so drift sees something the bars do not. Move signal stayed <= +0.009 on
all three, as in every previous run.
Next probe: segment the drift by game phase (`--phase`) to find WHERE his games diverge — if it is concentrated in
the endgame, his clone is playing out decided positions differently, which no current bar checks.

**OKENITE's state drift by game phase** (v3 games, `--phase`; drift = position-only AUC - real-vs-real floor):

| phase | move AUC | position-only | floor | drift |
|---|---|---|---|---|
| opening (own moves 0-10) | 0.670 | 0.674 | 0.487 | **+0.187** |
| middlegame (10-25) | 0.563 | 0.621 | 0.522 | +0.099 |
| endgame (25+) | 0.627 | 0.685 | 0.492 | **+0.193** |

**U-shaped, not endgame-only as I guessed**: his clone's games diverge most at BOTH ends and least in the middlegame
(overall was +0.120). Caveat: each phase has its own control on a smaller sample and those wander ~+-0.04, so read
these at the +-0.05 level — the U-shape is suggestive, the exact ordering is not.
Plausible causes, both properties of the simulation rather than of his move policy: in the OPENING he faces one field
model with a narrow repertoire where his real games face many opponents with varied ones; in the ENDGAME his clone
reaches and plays out decided positions differently (it is mated in 29.3 % of games vs his 21.2 %). The middlegame —
where the move policy does most of the work — is where his clone is most faithful.
Move signal stayed <= +0.009 in every phase, as in all previous runs.

## Round 10 — family signal from clone-vs-stock self-play: SIGNAL CONFIRMED, amplification still does nothing

The mechanism as specified: clone and stock play each other across 10 openings; wherever they would pick DIFFERENT
moves, that pair (same position, clone's pick vs stock's pick) is the signal; learn it with a deliberately
LOW-CAPACITY head (hid 8, ~10k params) so it can only represent the systematic "family" component and cannot
memorise positions; amplify it; judge by top-1/top-3 on the player's real held-out games, 5-fold CV by game.
Guard: both candidate moves come from the SAME position, so the head cannot win by learning whose turn it is or
which kind of position each engine reaches. `scripts/clonefish_family_amplify.py`.

**Stage 1 — the signal is REAL.** VEGETAL: 60 self-play games -> 816 disagreement positions (~34 % of the clone's
moves); held-out **pair accuracy 81.1 %** from ~650 training pairs. The clone's divergence from the norm is strongly
systematic, exactly as hypothesised — a small head reads it four times out of five.

**Stage 2 — amplifying it does nothing.** CV picks **w = 0 in all five folds**: top1 58.68 -> 58.68, top3
85.48 -> 85.48. The raw `clone - stock` push on the same folds and grid: identical, w = 0 everywhere.

**Why both can be true:** the head can READ the preference clearly (81 %), which proves it exists; the clone is
already EXPRESSING it at the magnitude the player's own moves support, so scaling it up only overshoots. That would
make it the fifth parameterisation to land here — scalar push, learned identity head, discriminator guidance,
top-3-selected sweep, and now a family signal from self-play disagreements — **but stage 2 is VEGETAL only so far;
kpowe52 and OKENITE are running, and this project's own rule is not to conclude from one player.** Note this grid was non-negative (it tests
amplification specifically); earlier grids that included negative w found the optimum drifting slightly BELOW zero.
Limit: 816 pairs is thin, so a bigger self-play run could sharpen the head — but the failure is not at the head
(81 % accuracy), it is that the direction is already correctly scaled.

**Round 10 REPLICATED on all three players — signal 81-88 %, amplification exactly nothing.**

| player | disagreements (60 games) | held-out pair accuracy | learned-family top1 / top3 | w per fold |
|---|---|---|---|---|
| VEGETAL | 816 | 81.1 % | +0.00 / +0.00 | 0,0,0,0,0 |
| kpowe52 | 866 | **87.9 %** | -0.16 / -0.09 | 0.25,0,0,0,0 |
| OKENITE | 914 | 85.2 % | +0.00 / +0.00 | 0,0,0,0,0 |

The raw `clone - stock` push on identical folds: VEGETAL +0.00/+0.00, kpowe52 -0.22/-0.13, OKENITE +0.00/+0.00.
**14 of 15 fold-selections chose w = 0**, and the one that chose 0.25 lost on its test fold.

**Settled, and worth stating precisely because the two halves point opposite ways:** the clone's deviation from the
norm is strongly SYSTEMATIC and LEGIBLE — a 10k-param head reads it at 81-88 % from under a thousand self-play
disagreements, on every player — and it is ALREADY CORRECTLY SCALED. Readable but not amplifiable. Five
parameterisations now agree (scalar push, learned identity head, discriminator guidance, top-3-selected sweep,
family signal from self-play), the last three chosen specifically to escape the circularity of the first two.
**The move policy is finished work.** What remains is the state distribution (see the drift results above).

## Round 11 — the opening book adds NOTHING to move match, and that closes the retrieval question too

Every top-k number measured this session used the raw policy; the engine also blends the personal book
q = (n + 2*p)/(N + 2). Measured on the held-out 80 games, by phase (`clonefish_book_effect.py`, board/row alignment
verified exactly: 3,947 items, zero legal-index disagreements):

| player | opening book coverage | opening top1 clone -> +book | ALL top1 clone -> +book | ALL top3 |
|---|---|---|---|---|
| VEGETAL | 44.2 % | 60.7 -> **60.3** | 58.7 -> 58.6 | 85.5 -> 85.5 |
| kpowe52 | 54.2 % | 71.8 -> **71.4** | 58.9 -> 58.8 | 85.7 -> 85.6 |
| OKENITE | 58.3 % | 74.0 -> **73.9** | 62.9 -> 62.8 | 88.8 -> 88.9 |

Coverage is ~0 % after move 10 for everyone, so the book can only act in the opening — and there it is within
+-0.4pp and consistently slightly NEGATIVE on top-1. The 5,000-game fine-tune has already absorbed the player's
repertoire into its weights; the book's exact-position counts add no information and the alpha=2 blend mildly
distorts an already better-calibrated policy.

**This also closes RETRIEVAL without running it.** Retrieval's headline (latebloomer: opening top-1 32.1 % -> 65.2 %,
+8.4pp overall, and it HURT midgame/endgame) was lifting a BASE model. Our fine-tuned clones sit at **60.7 / 71.8 /
74.0 %** in the opening unaided — at or above retrieval's post-boost figure. Both the book and kNN retrieval are
memory mechanisms that a large fine-tune has already internalised; there is no opening headroom left to chase.

**WARNING — do not switch the book off.** Its value is TIMING, not moves: it supplies the player's real opening think
readings, which is how the clone reproduces premoves (kpowe52 premoves ~50 % of his second moves). Removing it
because it does not help move match would silently break the timing behaviour that currently passes 3/3.

**Move matching is finished work** at 58.7 / 58.9 / 62.9 % top-1 and 85.5 / 85.7 / 88.8 % top-3. Nothing tried on top
of the fine-tune moves it: five amplification parameterisations, the opening book, and (by inference) retrieval.

---

# PROJECT WRAPPED (2026-09-16)

This file is the raw chronological log — eleven rounds with corrections layered on corrections, and several
conclusions here were later overturned by entries further down (the "14 of 15" scorecard, the length-bar failure,
the ResignBias fit). **Read `FINAL.md` instead**: it is the self-contained summary of what was built, what it
achieves, what is settled and must not be re-tried, the measurement lessons, and what is genuinely left open.
