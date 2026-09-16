# clonefish v4 — recommended architecture (night search 2026-09-12/13)

Target: top-3 move match + the person's think-time distribution, ~5,000 games per person.
Evidence: `results/night_0913/LANDMARKS.md` (all numbers), `PLAN.md` (pass bars written before results),
`lit/NOTES_night_{A,B,C,D}*.md` (literature), `CANDIDATES.md` (design reviews).

## The recommendation

| layer | what | status |
|---|---|---|
| 1 | Population base (existing 19.5M clock-aware model) | unchanged |
| 2 | **Full fine-tune on the person's games** (or rank-16 adapters when storage matters) | measured, n=3 |
| 3 | **Whole-second (bucket) time loss** in that fine-tune for Lichess clocks (`--time-loss bucket`) | measured, n=2 |
| 4 | Personal opening book | verified earlier (n=20) |
| 5 | Sample think-time from the head, never its mean | verified earlier |
| 6 | Per-person 3-scalar timing calibration (K4) | optional; mixed at n=2-3 — its snap term misses the real snap rate either way and lowers position-time correlation; the bucket loss may make it unnecessary |

## Why, in numbers (post-opening = own move >= 12, test = newest 40 games)

- Full fine-tune @5k: top-1 **+4.96 / +4.50 / +4.44**, top-3 **+2.38 / +4.07 / +3.23** (mean +3.2 pp top-3).
- Bucket vs old time loss (n=2): time NLL -0.03 / -0.21, W1 0.064→0.039 / 0.035→0.031, position-time
  correlation 0.40→0.41 / 0.53→0.56; moves unchanged (top-3 +0.34 / -0.78, inside noise). On kpowe52 the old
  loss was the whole reason the fine-tuned time head scored worse than the base.
- Timing model vs a per-player lookup table: the model wins NLL and W1 on 3/3 Lichess players.

## Tested tonight and dropped

- **Weight dial (WiSE-FT)**: dial − full FT top-3 = −0.07 / +0.26 / 0.00, all CIs span 0. Not a component.
- **Identity-aware re-trained base** (AdaSpeech/Maia4All style): reviewer estimate ≈ 0 at 5k games —
  it helps users with few games. Revisit only for a "clone from 50 games" product.
- **Style-twin cohort then self**: matched vs anti-matched cohort +0.7 pp (n=3), inside the 1.2 pp paired noise.
- **Special top-k loss**: plain cross-entropy is already competitive for every k (Lapin 1512.00486).

## Honest limits

- n=3 players (n=2 for the time loss). Paired between-method sd is ~1.2 pp, so effects under ~2 pp are
  invisible here. The top-3 gain is well above that; the timing choices are not yet.
- Timing is scored teacher-forced on the real clock at every move. Clock drain across a whole game is untested.
- Lichess clocks are whole seconds; model samples are rounded through a simulated clock before comparison.
- Top-3 gains are real but smaller than top-1 gains, because base top-3 is already 80–86%.

## Next, in order

1. Bucket time loss + K4 on a 3rd–5th player, with a K4 variant that matches the real snap rate.
2. Adapter (rank 16) with the bucket loss at 5k games vs full FT, on top-3 (is 0.7% of the weights enough?).
3. Whole-game self-play timing check (clock drain), then the human "is this you?" test as the ship gate.
