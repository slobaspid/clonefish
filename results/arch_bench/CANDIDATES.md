# Candidate architectures for "user's games in, engine out" — and how each is being tested

Sources: `lit/NOTES_arch_trunk.md` (identity inside the trunk), `lit/NOTES_arch_traits.md` (person as an
evaluation function), `lit/NOTES_arch_dynamics.md` (within-person state), plus the earlier dives
(`lit/NOTES_dive_{A,B,C}*.md`) and our own measurements.

Reference points (measured, this project): personal opening book **+2.9 pp** overall but **opening only**;
dense style code on a frozen trunk **+0.001 pp** midgame; **full fine-tune @5k games +3.41 pp post-opening**
(n=7, range +0.8…+5.8), midgame +3.03. So the open question is: *how few parameters, in the right place,
recover the full fine-tune's post-opening gain?*

## A. Identity inside the trunk — STAGE 1, running now
One player (VEGETAL), 2,000 games, 3 epochs, learning rates pre-registered before any result was seen.

| # | Candidate | Params/user | Status |
|---|---|---|---|
| A1 | Full fine-tune (in-design reference at 2k games) | 19.5M | running |
| A2 | LHUC — one learned gain per hidden unit, all 8 blocks | 0.004M | running |
| A3 | Bottleneck adapters, rank 16, after every block | 0.135M | running |
| A4 | BitFit — biases only | 0.032M | running |
| A5 | LayerNorm affines only | ~0.01M | running |
| A6 | FiLM generator only (re-purposes an existing conditioning slot) | 1.06M | running |
| A7 | Move head only | 0.26M | running |
| A8 | Last encoder block + head | 2.4M | running |
| A9 | Hypernetwork → adapters from a user embedding | embedding only | queued (needs a one-off GPU pre-train over many players) |
| A10 | Scale+bias code injected at a single mid-trunk layer | ~0.1k | queued (cheap; run if A2/A5 look alive) |

## B. Person as an evaluation function / trait dials — queued, cheap
Tests on cached logits, no trunk training. Dive A's own caveat: these have *less* capacity than the dense
code that already gave +0.001 midgame, so treat as cheap falsification, not a favourite.

| # | Candidate | Fitted per user | Status |
|---|---|---|---|
| B1 | Mixed-logit reranker over interpretable features (material, mobility, king safety, our value head, capture/check flags) | 10–20 coefficients | queued |
| B2 | IRL trait vector (~5–8 named axes incl. risk), linear-program fit, used to rerank logits | 5–8 numbers | queued |
| B3 | Behaviour-frequency histogram over move categories | counts only | queued (weakest; likely re-encodes rating) |
| B4 | Classical engine style knobs (contempt-like) | 2–4 scalars | rejected — no classical eval terms to hook into in a transformer |

## C. Within-person dynamics — mostly ruled out
| # | Candidate | Status |
|---|---|---|
| C1 | Session-state GRU feeding a time-varying conditioning vector | queued (the one live idea here) |
| C2 | Latent skill/mood random walk across games (Kalman-style), fed to FiLM | queued behind C1; our Elo dial is already a weak knob |
| C3 | Post-loss "tilt" trigger | **rejected** — 57,421-game analysis finds ~0 autocorrelation in excess score once rating is removed |
| C4 | Per-session re-fit of book/temperature | rejected — redundant with the book; source not verifiable |

## D. Already settled (do not re-test)
Recognizer-as-training-reward (failed 3×) · per-user timing residual (over-thinks) · recency-weighted book
(no gain) · style-twin borrowing (thin users already get most of the book) · retrieval clone (+0.4 midgame) ·
bigger shared head / more head players (no in-head advantage) · prompting with raw games · argmax search
(−18.7 pp) · search for midgame accuracy (+0.2–0.8 pp at blitz, a strength/feel tool instead).

## Protocol
Stage 1 picks the best trunk modes on one player. Stage 2 re-runs the best two on a high-gain player
(kpowe52, +5.81 full-FT) and a low-gain one (MALOUMNJAK, +0.79), because the gain varies ~5 pp across people.
Headline metric: post-opening (own ply ≥ 12) top-1 vs base, with the midgame split reported separately.
Anything that beats the base by more than ~1 pp midgame goes to the independent recognizer metrics
(1-NN, MMD) before it is believed.
