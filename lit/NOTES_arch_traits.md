# Trait-dial / personalized-eval-function architectures (2026-09-12)

Angle: person = fitted evaluation/preference function or a handful of
interpretable parameters, not a dense embedding. Combined with our existing
policy (candidate move logits) as a reranker, not a trunk edit.

## Verified sources

- **MMBM** — Wang, Sun, Zheng, "Beyond Winning and Losing: Modeling Human
  Motivations and Behaviors Using Inverse Reinforcement Learning" (arXiv
  1807.00366). VERIFIED p.1 text: "Multi-Motivation Behavior Modeling (MMBM)
  ... extends IRL to uncover the complex, multi-dimensional reward
  mechanism. Our model first quantifies each dimension of the reward signal
  individually... decomposition of the full reward signal is reduced into a
  linear program, which is solved efficiently." Off-policy only (no env
  access needed). Not chess; game-behavior domain.
- **Behavior-fingerprint identification** — Yuda et al., "Identification of
  Play Styles in Universal Fighting Engine" (arXiv 2108.03599). VERIFIED
  p.1: player = vector of action-category probabilities ("behavior
  fingerprints"), compared "as vectors using cosine similarity." Built by
  counting, no gradient training.
- **Mixed logit / random-coefficients discrete choice** — standard
  econometric family (Hensher & Greene; Wikipedia "Mixed logit"): "different
  coefficients for each person... approximate to any degree of accuracy any
  true random utility model." No chess-specific instance found after
  targeted search — general statistical technique, not previously applied
  here.
- **Engine style knobs** — Stockfish `Contempt` UCI option (-100..100,
  documented default 0, positive = more "risky"/optimistic play); Fritz
  personality sliders and Rodent/Fruit/Toga exposed eval-term weights
  (aggressiveness, king-safety weight, etc.) — established production
  pattern, confirmed via search results, not a research paper.
- Already checked and rejected as *not* on this angle: 2504.05425
  (behavior-based features, not per-player) and 2507.21488 (Maia4All,
  identity-in-trunk, already logged elsewhere) — both pre-existing in
  lit/pdf, confirmed via NOTES_priorart.md / NOTES.md, not re-cited below.

## Candidate architectures (ranked, see final answer for full writeup)

1. Per-player mixed-logit reranker over engine features (utility = w_p ·
   phi(move, position); phi = material delta, mobility, king-safety,
   central control, our own value-head eval, capture/check flags).
2. MMBM-style IRL trait vector: decompose reward into ~5-8 named
   dimensions (material, king safety, tempo, risk/variance via own
   clock use, piece activity), fit per-player weights via linear
   program on move choices, rerank logits.
3. Behavior-frequency-vector reranker: per-player histogram over
   ~15-20 tactical/positional move categories, Bayes-rerank top-k.
4. Classical style-knob layer: hand-picked scalar knobs (contempt-like
   risk bias, king-safety weight) fit per-player by black-box search
   (CMA-ES/grid) on held-out move log-likelihood, feeding a shallow
   search on top of the value head.

Full mechanism / params / cost / failure mode given in the final chat
answer to avoid duplicating content here (line budget).
