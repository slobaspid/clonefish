# Night 2026-09-12 — candidate architectures (synthesis of lit/NOTES_night_{A,B,C,D} + BRIEF.md)

Target: top-3 move match + think-time DISTRIBUTION, ~5k games/user. Always paired with the personal
opening book and sampling (both already verified).

## K1 — "Anchored fine-tune + weight dial" (no new pretraining)
Full fine-tune on the user's games with an L2-SP pull back toward base weights (Li 1802.01483, Ditto),
then WiSE-FT interpolation theta = (1-a)*base + a*ft (2109.01903), a chosen on a validation slice of the
user's own games (games just before the test set). Rationale: full FT is our best number (+3.41 pp @5k)
but over-specialises (worse on strangers, endgame top-3 ~0 or negative); interpolation keeps the base's
broad ranking (helps top-3) while keeping personal peaks (top-1). Per-user artefact: 19.5M weights
(or store the delta). Cost: ~1 GPU-hour/user or a few CPU hours.

## K2 — "Identity-aware base" (AdaSpeech / speaker-adaptive training + Maia4All)
One-time: re-train/enrich the base over thousands of players with a per-player embedding z feeding the
FiLM generator (next to the clock) and the input Elo slot, with z-dropout so the base stays a good
population model. Optionally Reptile-style meta steps so the base is "one adaptation away" from any
player (Meta-TTS 2111.04040). New user: init z from CosFace nearest neighbours (prototype init), fit z,
then optionally a small adapter. Cost: ~1 GPU-day once, minutes per user.

## K3 — "Retrieve-a-cohort, then personalise" (uses the CosFace identifier)
CosFace embeds the user's games; retrieve the K nearest players from the 3,000-player pool (style
twins, Welch ACL'22 / ARMED unseen-cluster prior). Stage 1: fine-tune on cohort games (fact 6: a
trait cohort alone gives ~half the self gain). Stage 2: fine-tune on the user's own games from that
starting point (data interpolation, Mansour 2002.10619). Rationale: more on-style data than the user
alone has, reducing overfit in rare positions (middlegame/endgame top-3). Cost: 2 fine-tunes/user.

## K4 — Timing: "population head + shrunk personal calibration, sampled"
Keep the base MDN (position-aware). Per user fit a few scalars on log-time: shift and scale of the
component means/sigmas, maybe a snap-probability bump, partial-pooled toward 0 by n/(n+k)
(hierarchical RT models). Evaluate with CRPS, Wasserstein on log-time, KS, snap-rate and tank-rate,
always next to the per-player lookup-table null. Composes with K1-K3 (a fine-tuned trunk also moves the
time head; K4 then fixes the residual calibration).

## Rejected tonight (and why)
- Smooth top-k loss (Berrada): Lapin 1512.00486 says softmax CE is already competitive for all k; a
  cheap ablation at most.
- LoRA bank + routing (MHR): already designed in docs; needs a bank pre-train like K2; same test gate.
- Human survey: evaluation, not architecture; stays the ship gate.
