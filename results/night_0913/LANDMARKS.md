# Landmarks — night of 2026-09-12

## Literature (verified by agents from abstracts; notes in lit/NOTES_night_{A,B,C,D}*.md)
- Partial personalization gets most of full personalization with few personal params (Pillutla 2204.03809).
- WiSE-FT weight interpolation improves fine-tunes (2109.01903); L2-SP anchors to pretrained weights (1802.01483).
- LoRA learns less, forgets less (2405.09673) — explains adapters ~80% of full FT but never negative.
- Softmax CE is already competitive for all top-k (Lapin 1512.00486) — no special top-3 loss needed.
- No non-chess system clones a specific individual's decisions with a match-rate number (Go/poker/SC2/Dota/RL searched).
- Lognormal is the standard family for human movement timing (Sigma-Lognormal) — keep our MDN family.

## Design verdicts (one think-through agent per candidate; results/night_0913/CANDIDATES.md)
- K1 weight dial on full FT: TEST (expected tie with adapters on top-3).
- K2 identity-aware re-trained base: DROP at 5k games (only helps few-game users); needs a no-vector control if ever run.
- K3 style-twin cohort then self: DROP (matched vs anti-matched +0.7 pp, n=3, inside noise).
- K4 3-scalar timing calibration with whole-second likelihood: TEST, fit on validation games only.

## Found tonight
- Lichess clocks are whole seconds; mdn_nll clamps 0 s moves to log(1e-6) ≈ -13.8. Every Lichess
  fine-tune's time loss (w_time 0.2) has trained on that. Possible silent damage to the time head — unmeasured.
- ~~The CosFace identifier was never saved~~ WRONG: `checkpoints/recognizer_film.pt` IS a CosFace recognizer
  (clone_judge log prints "(cosface)"). It was trained on the 90-player titled cache_film, not the 3,000-player set.
- Only clones/ft5k_VEGETAL.pt was saved among the 5k fine-tunes.

## Measured tonight (scripts/night_dial_eval.py; JSON in results/night_0913/)
- Stage 0 smoke (VEGETAL, ft5k checkpoint): dial 0.75 ~= full FT on top-3 (cross-fit halves +0.4 / +0.0).
- Stage 1 VEGETAL (val-picked): FT post-opening top-1 +4.96; dial a=0.75 vs FT top-3 -0.07 [CI -0.78, 0.58].
  Timing bucket NLL: base 1.890, base+K4 1.850, FT 1.980, FT+K4 1.858, lookup null 2.103.
  FT damages the time head; K4 repairs it but base+K4 is still best. Snap bias fitted -4..-6.
  K4 lowers position-time correlation r 0.40 -> 0.35 (trade-off to watch).
- Stage 1 kpowe52: FT top-1 +4.50, top-3 79.74 -> 83.81 (+4.07). Dial a=0.75 vs FT top-3 +0.26 [-0.79, 1.31].
  Timing: FT NLL worse (2.027 -> 2.177) BUT W1 better (0.080 -> 0.035) and r better (0.47 -> 0.53).
  K4 on FT: NLL 1.991 but W1 0.101 and r 0.42, snap 6.8% vs real 10.3% — the snap bias over-corrects.
  => Bucket NLL and sampled-distribution metrics DISAGREE; do not pick timing by NLL alone.
- Stage 1 OKENITE (after power-cut restart, 2026-09-13): FT top-1 +4.44, top-3 86.06 -> 89.29 (+3.23).
  Dial picked a=1.0 on validation (= plain FT). Timing NLL base 1.689 / base+K4 1.628 / FT 1.720 /
  FT+K4 1.606 / null 1.918; W1 base 0.114 / base+K4 0.050 / FT 0.050 / FT+K4 0.031 / null 0.329.
  FT+K4 snap 3.7% vs real 5.8% (over-corrects again), r 0.43 -> 0.36.
- No-snap K4 (VEGETAL): shift+scale only makes W1 WORSE (base 0.059 -> 0.070, FT 0.064 -> 0.097).
  The snap bias carries K4's W1 gain, and it is also what over-corrects snap rate.

## Stage 1 verdicts, n=3 (pass bars from PLAN.md, written before results)
- Full FT @5k, post-opening: top-1 +4.96 / +4.50 / +4.44 (mean +4.63); top-3 +2.38 / +4.07 / +3.23 (mean +3.23).
- K1 weight dial: FAILS the "clearly better" bar (dial - FT top-3: -0.07, +0.26, 0.00; all CIs span 0).
  Not a component. Ship plain FT (or adapters).
- K4 timing calibration: PASSES the written bar (NLL better on 3/3, W1 better on 2/3 for base+K4 and FT+K4,
  lookup null loses on everything), BUT with a caveat the bar did not anticipate: its snap term under-shoots
  real snap rate and lowers position-time correlation r. Treat as "helps the distribution, costs the
  position-awareness a little" — needs a snap-rate-matched variant before shipping.
- Lookup-table null loses to the model on NLL and W1 for all 3 players (unlike chess.com, where it won
  for half) — on these Lichess players the model's timing is the better starting point.
- FT makes time NLL worse on 3/3 (+0.09, +0.15, +0.03) while W1/r are equal or better — stage 2 tests
  whether the whole-second loss bug causes this.

## Stage 2 — whole-second (bucket) time loss in the fine-tune, same split
- VEGETAL (n=1): moves unchanged (top-1 59.51 both; top-3 85.26 -> 85.60, noise).
  FT time NLL 1.980 (point) -> 1.949 (bucket) vs base 1.890: the bug explains ~1/3 of the damage, not all.
  FT W1 0.064 -> 0.039 and r 0.403 -> 0.414 (both better); snap 5.7% vs real 6.9% (was 8.2%).
  FT+K4: NLL 1.742 (best of any arm) and r 0.441 (K4 no longer lowers r), but W1 0.080 and tank 6.5% vs 4.9%.
  NLL and W1 disagree again. Waiting on kpowe52 before any verdict.
- kpowe52: moves top-1 53.25 -> 53.51, top-3 83.81 -> 83.03 (inside noise).
  FT time NLL 2.177 (point) -> 1.965 (bucket), now BETTER than base 2.027 (the bug explained all the damage here).
  FT W1 0.035 -> 0.031, r 0.531 -> 0.561. FT+K4: NLL 1.892, W1 0.025, snap 10.0% vs real 10.3%,
  tank 6.0% vs 6.5%, r 0.536 — best arm on every timing metric.

## Stage 2 verdict (n=2)
- **Use the whole-second (bucket) time loss for every Lichess fine-tune.** vs the old point loss: time NLL
  better 2/2 (-0.03, -0.21), W1 better 2/2, position-time r better 2/2; moves unchanged (top-3 +0.34, -0.78).
- K4 on top of a bucket fine-tune: NLL better 2/2, W1 split (VEGETAL worse 0.039 -> 0.080, kpowe52 better
  0.031 -> 0.025). Keep K4 optional until n>=3 with a snap-rate check.

## K4 with vs without the snap term (n=3, on the OLD point-loss fine-tunes and on base)
- W1: snap term better 4/6 arms, worse 1 (kpowe52 base 0.082 vs 0.060), tie 1.
- Snap rate matches the real rate in neither version: with the snap term it undershoots
  (FT arms 6.8/6.8/3.7% vs real 6.9/10.3/5.8%); without it, it overshoots (11.1/12.4/9.4%).
- Position-time r is higher without the snap term in 5/6 arms.
- => The snap term is a blunt tool: it helps the overall distribution but misses the target snap
  rate and costs some position-awareness. On the bucket-loss kpowe52 fine-tune, K4 hit snap 10.0% vs
  10.3% with r 0.536 — the bucket loss may remove most of the need for a snap correction.
  Next test: K4 only on bucket fine-tunes, n>=3.

## Line-up test (scripts/clone_lineup.py) — pass bars written BEFORE running (2026-09-13)
Question from the user: does the CosFace recognizer recognise the engine's games as the player's own?
Panel ~30 Lichess players (6 lichess_5k + 24 lichess_1k), 60 enrollment games each; chance P@1 ~0.033.
- Positive control: REAL held-out 5-game P@1 >= 0.5 on average. If it fails, the recognizer (trained on 90
  titled players) can't separate these players and NO clone number is interpretable.
- "Yes, recognised": CLONE_VSB 5-game P@1 >= 0.5 AND above BASE for all 3 targets.
- "Partly": CLONE above BASE on mean rank for all 3 but P@1 < 0.5.
- "No": CLONE not above BASE.
Caveats known in advance: 8 five-game queries per arm (very noisy); clone plays at a fixed average Elo;
recognizer embeddings see timing (via base features + logt), so timing realism counts too.

### Line-up RESULT (results/night_0913/lineup.json, log lineup3.log; 5-game queries, n=8 each, chance 0.033)
| target (clone ckpt) | REAL P@1 / rank | BASE P@1 / rank | CLONE_SELF rank | CLONE_VSB P@1 / top3 / rank |
|---|---|---|---|---|
| VEGETAL (bucket FT) | 0.75 / 1.38 | 0.125 / 2.38 | 3.88 | **0.50 / 1.00 / 1.62** |
| kpowe52 (bucket FT) | 0.875 / 1.25 | 0.00 / 4.00 | 6.50 | 0.00 / 0.00 / **5.50 (worse than base)** |
| OKENITE (point FT) | 0.43 / 2.57 | 0.00 / 13.38 | 7.00 | 0.00 / 0.00 / 7.88 |
- Positive control passes on average (REAL mean 0.685; OKENITE alone 0.43).
- **Pre-registered verdict: NO.** CLONE_VSB beats BASE on rank for 2/3 (VEGETAL, OKENITE), is worse for kpowe52.
- kpowe52 has the LARGEST top-3 gain (+4.07) yet the clone reads LESS like them than the base: per-move match
  and recognisability are different things (again).
- Clone-vs-clone self-play reads less like the player than clone-vs-base for VEGETAL/kpowe52 (self-play confound).
- CONFOUND: base features take Elo as input and all generated games use the target's mean Elo, so "same rating"
  pulls BASE and CLONE toward the target (BASE top-3 0.875 for VEGETAL). Fix: neutral/real-distribution Elo.
- Untested explanations for kpowe52: 80-ply cap on generated games (real games longer / time scrambles),
  think-time patterns (recognizer reads logt), fixed Elo.

### Line-up LADDER — what gives the clone away? (pre-registered before running, 2026-09-13)
Same generated games (60 per arm) scored under S0 orig -> S1 length-matched -> S2 +neutral Elo 1700 ->
S3 +neutral think 2.0 s -> S4 S3 with moves-only recognizer. 5-game queries.
Reading rules (on CLONE_VSB mean rank vs BASE, per target):
- A rung that moves kpowe52's CLONE_VSB from worse-than-BASE to better-than-BASE names the give-away.
- S1 fixes it -> game length/resignation. S2 -> the rating shortcut. S3/S4 -> think-time.
- If REAL P@1 collapses at a rung, that rung removed real identity signal too (not just a confound) —
  report it, don't count it as a fix.
- If no rung fixes kpowe52, the give-away is in the MOVES of the generated games (clone style drift in
  its own positions), not a featurization artefact.

### Ladder RESULT (lineup_ladder.json) — mean rank of the target, 5-game queries (lower = more like them)
| rung | VEGETAL real / base / clone_vsb | kpowe52 real / base / clone_vsb | OKENITE real / base / clone_vsb |
|---|---|---|---|
| S0 orig | 1.4 / 2.1 / 1.7 | 1.2 / 4.1 / **5.2** | 2.6 / 13.3 / 8.8 |
| S1 +length | 1.4 / 2.2 / 1.7 | 1.2 / 3.8 / **4.3** | 2.6 / 13.8 / 10.6 |
| S2 +Elo | 3.8 / 7.2 / 2.2 | 2.2 / 9.6 / **11.3** | 2.4 / 18.3 / 7.8 |
| S3 +time | 4.1 / 5.5 / 5.4 | 2.1 / 14.1 / **5.5** | 5.7 / 19.9 / 12.1 |
| S4 moves-only rec | 6.8 / 15.2 / 4.4 | 6.4 / 15.1 / 3.6 | 5.6 / 16.4 / 6.8 |
- **kpowe52's give-away = THINK-TIME** (pre-registered reading): worse than base through S0-S2, far better
  than base once timing is neutral (S3: 5.5 vs 14.1, clone_vsb P@1 0.50). Length explains a little (S1).
- VEGETAL is the reverse: her clone's closeness was mostly timing (S3 clone 5.4 ≈ base 5.5).
- The rating input carries a lot of REAL identity too (VEGETAL real P@1 0.75 -> 0.38 at S2).
- S4 moves-only recognizer barely identifies kpowe52/OKENITE's real games (P@1 0.12/0.14) -> uninformative
  for them by the pre-registered rule.
- **LEAK found**: target enrollment games [-140,-80) sat INSIDE the clone's training window (6000 games,
  train = [-5080,-80)). This favours every clone. Fixed: enroll targets on validation games [-80,-40),
  which --train-skip-recent 40 kept out of training. Rerun -> lineup_ladder_clean.json.

### CLEAN ladder (leak fixed; lineup_ladder_clean.json) — mean rank real / base / clone_vsb
| rung | VEGETAL | kpowe52 | OKENITE |
|---|---|---|---|
| S0 orig | 5.5 / 4.5 / 2.8 | 1.0 / 5.1 / **5.9** | 4.0 / 14.8 / 10.1 |
| S1 +length | 5.5 / 4.2 / 2.7 | 1.0 / 4.4 / **5.2** | 4.0 / 14.7 / 10.8 |
| S2 +Elo | 7.6 / 5.4 / 5.5 | 2.0 / 10.8 / **13.5** | 3.3 / 18.4 / 5.9 |
| S3 +time | 8.5 / 14.2 / 16.6 | 2.1 / 8.6 / **3.6 (P@1 0.67)** | 5.0 / 18.9 / 5.7 |
| S4 moves-only | 11.0 / 17.2 / 10.5 | 7.1 / 15.2 / 7.1 | 8.0 / 18.5 / 5.1 |
**Clean verdicts (supersede the leaky ladder):**
- **kpowe52: think-time is the give-away — CONFIRMED leak-free.** Worse than base S0-S2, then with timing
  hidden the clone is named kpowe52 67% of the time vs 62% for their real games.
- **OKENITE: on moves the clone reads close to the player** (S2/S3 rank 5.9/5.7 vs real 3.3/5.0, base 18-19).
- **VEGETAL: inconclusive.** Clone beats base only with timing+rating visible (S0/S1 2.8 vs 4.5); once they
  are hidden the recognizer can't identify her REAL games (P@1 0.25, rank 7.6-8.5). The earlier
  "VEGETAL passes" was inflated by the enrollment leak.
- S4 moves-only recognizer fails the positive control for all 3 (real P@1 0.00-0.14) — unusable here.
- Take-away: where the recognizer can identify the player, the clone's MOVES read as the player (2/2);
  the TIMING layer is what exposes clones. Timing is the next lever, and this line-up (S2 vs S3 gap) is
  its independent check.

### Move-family habits (scripts/move_habits.py, move_habits.json, 2026-09-14)
Teacher-forced on each player's 80 newest (never-trained) games: REAL rate of each move family vs the
probability mass BASE and CLONE put on it. 2 standard errors ~1-2 pp.
- **BASE vs real: clear gaps.** Captures over-predicted for all 3 (kpowe52 -5.1 all / -6.0 post-opening;
  VEGETAL -2.2 post). Knights over-predicted (VEGETAL -5.6 all, OKENITE -3.7 all — mostly opening).
  "Into enemy half" over-predicted for VEGETAL/kpowe52 (-2.3..-3.8). kpowe52 edge-pawn pushes +3.8.
- **CLONE vs real: essentially no gap.** Every family within 1.3 pp, all inside the error bars. The full
  fine-tune already learned these habits — part of its top-3 gain is exactly this.
- => A per-player move-type bias has almost nothing to fix on a fine-tuned clone. It could be a cheap
  "light clone" on the BASE (for users with few games). Consistent with the line-up: moves pass, timing fails.

ALL RUNS COMPLETE 2026-09-13 09:08 (after two power cuts; see memory pc-power-cuts-under-load).

## Dead ends (don't retry)
- (see K2, K3 above)
