# Night plan — the architecture and tonight's little test

## Proposed architecture ("clonefish v4"), per user with ~5k games
1. **Frozen population base** (existing 19.5M clock-aware model).
2. **Identity inside the trunk**: full fine-tune on the user's games (or rank-16 adapters after every
   block when storage matters), followed by a **weight dial** a in [0,1]:
   theta = base + a*(ft - base), a picked on the user's own validation games (K1, WiSE-FT).
   Why: full FT is the best measured post-opening gain but over-specialises; the dial should keep the
   base's broad ranking (top-3) while keeping personal peaks (top-1).
3. **Personal opening book** (verified, unchanged).
4. **Timing layer (K4)**: keep the time head, fit 3 per-user scalars on log-time — shift, sigma
   scale, snap-component bias — with a **whole-second bucket likelihood** (Lichess clocks are integer
   seconds; the current point-density loss clamps 0 s moves to log(1e-6)). Always SAMPLE.
5. Dropped after review: identity-aware re-trained base (K2: ~0 expected at 5k games), cohort
   retrieval (K3: matched-vs-anti +0.7 pp n=3, inside noise), top-k loss (CE already near top-k optimal).

## Tonight's little test (GTX 1060, no user interaction)
**Stage 0 (minutes, free):** existing `clones/ft5k_VEGETAL.pt`.
- Dial grid a in {0, .25, .5, .75, 1}; no validation slice exists for this checkpoint, so **cross-fit
  on the 40 test games**: choose a on odd games, report on even games, and vice versa. Report
  post-opening top-1 / top-3 per arm.
- Timing: fraction of 0-second own moves; for base and ft: bucketed NLL, sampled-time W1 on log(t+1),
  snap rate (0 s), tank rate (>=10 s), vs a lookup-table null (empirical think-time distribution per
  clock bucket x move-number bucket from the user's training games).
- K4: fit shift / sigma-scale / snap-bias on the last 500 training games (bucket likelihood), evaluate
  on test.

**Stage 1 (overnight, background):** honest validation split for n=3 (VEGETAL, kpowe52, OKENITE):
full FT with `--train-skip-recent 40 --max-train-games 5000 --save-ft`, so games [-80,-40) are
validation and [-40:] test. Then the same Stage-0 evaluation with a picked on validation only.

**Fixes after plan review (FIX FIRST verdict, all applied):**
- K4 scalars are fitted on VALIDATION games (never on games the fine-tune saw). Stage 0 has no K4.
- Dial a is picked by post-opening TOP-3 on validation (ties -> top-1); a fixed a=0.5 is also reported.
- Model samples are rounded through a simulated whole-second clock: reading = ceil(t - u), u~U[0,1);
  bucket likelihood uses the matching triangular kernel (reading k covers true t in (k-1, k+1)).
- Timing is teacher-forced on the real clock at every move; it does not test clock drain over a game.
- Stage 0 is a SMOKE TEST only (checkpoint trained right up to the test games; 20-game halves).
- Validation games are rebuilt with the fine-tune script's exact sort/filter (reuse its functions).
- n=3 K1 bar replaced: dial ships only if not worse than full FT on top-3 for all 3 players, with
  game-bootstrap CIs reported. The +1.0 pp mean below is kept as the "clearly better" bar.

**Pass bars (written before any result):**
- K1 dial: post-opening top-3 of dial-picked >= full FT + 1.0 pp on average AND top-1 no more than
  0.5 pp below full FT AND sampled W1 not worse. Else: ship plain FT/adapters, dial is not a component.
- K4: bucketed NLL and W1 better than both the uncalibrated head and the lookup-table null on >= 2 of 3
  players (n=1 at stage 0 is only a smoke test). Else: timing stays "base head, sampled".
- Everything reported next to the base arm and run-to-run noise (0.13 pp) / paired sd (1.2 pp).
  n=3 cannot detect < ~2 pp; say so.
