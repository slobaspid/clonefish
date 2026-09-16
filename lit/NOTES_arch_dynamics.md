# Within-person dynamics/state architectures — verified sources

1. Rosenthal, "Probabilities of Streaks in Online Chess" (Aug 2024, HDSR 7.2 2025 version).
   PDF verified: probability.ca/jeff/ftpdir/chessstreaks.pdf, p1 text confirmed.
   57,421-game autocorrelation analysis of one player's excess-score time series;
   lag-1 and higher autocorrelations ~0 -> no hot-hand/tilt signal in win/loss outcome
   sequence once Elo expectation is subtracted out.

2. Duffield, Power, Rimella, "A State-Space Perspective on Modelling and Inference
   for Online Skill Rating" (arXiv:2308.02414). PDF verified, p1 title/authors match.
   Frames rating systems (Elo/Glicko/TrueSkill) as state-space models: latent skill
   evolves each match via a transition (random-walk/AR) kernel, observed indirectly
   through match outcomes; supports filtering/smoothing for time-varying skill.

3. Hidasi et al., "Session-based Recommendations with Recurrent Neural Networks"
   (ICLR 2016, arXiv:1511.06939) and Quadrana et al., "Personalizing Session-based
   Recommendations with Hierarchical Recurrent Neural Networks" (arXiv:1706.04148).
   Both PDFs verified, p1 text matches. GRU4Rec: single GRU carries within-session
   state, reset each session. HGRU4Rec adds a second, slower GRU that carries a
   summary across sessions and re-initializes the fast GRU's hidden state at the
   start of each new session (cross-session "user state" -> within-session "mood").

4. Bennett, Fulton, Forbes, "Chasing emotional losses: negative subjective affect
   is linked to increased risk-seeking behavior both within and between
   individuals" (JDM 2024, 19:e31). PDF verified, p1 text matches.
   Within-person emotion fluctuations shift loss-aversion/utility curvature trial
   to trial; negative affect after a loss -> more risk-seeking on the next choice,
   an effect estimated both within- and between-subject.

5. Leone, Slezak, Golombek, Sigman, "Time to decide: diurnal variations on the
   speed and quality of human decisions" (Cognition 158, 2017) — chess-specific,
   internet fast-chess move data. NOT independently PDF-verified (paywalled,
   ResearchGate only offers "request PDF"); reported here only via corroborating
   secondary summaries (BPS Research Digest, APS), so treated as suggestive, not
   a hard-verified number. Reported pattern: decisions get faster/less accurate
   over the course of the day/session with total performance roughly flat.

# Candidate architectures (ranked)

1. **Session-state GRU head (HGRU4Rec-style)** — a small recurrent state vector
   updated move-by-move within a game and carried, at reduced update rate,
   across a user's games in a sitting; feeds the existing transformer as an
   extra conditioning vector (like the style code, but time-varying not static).
   Extra input: nothing beyond what's already used — move times, results,
   ply — but needs game *order and timestamps* per user to define "session"
   boundaries (gaps > N minutes = new session). Expected effect: gives the
   model a slot for warm-up/fatigue/momentum that a static per-user code
   cannot express; direct architectural fix for the 0.53-0.58 cross-session
   correlation finding. CPU cost: trivial (one GRU cell, tens of thousands of
   params, fits laptop CPU easily). Failure mode: with only 50-5000 games per
   user, the recurrent state may just learn noise; needs a strong prior (e.g.
   initialize near zero / small state dim) and eval against the ceiling
   already established in time-head-ceiling.md.

2. **Latent state-space skill/mood model (Duffield et al. state-space framing)**
   — replace the static style code with a 1-3 dim continuous latent that
   random-walks game-to-game (like a Kalman filter on top of Elo), inferred by
   smoothing over a user's game sequence, then fed into FiLM/GAB as a
   time-varying instead of fixed conditioning signal. Extra input: per-game
   result, opponent rating, timestamp (all in PGN). Expected effect: captures
   session-to-session drift already known to be large (0.53-0.58 corr);
   cheap Bayesian filter, no deep net needed for the state estimation itself.
   CPU cost: negligible (closed-form/EM filtering, laptop-trivial). Failure
   mode: same risk as (1) — the transformer's move/time output may not be
   sensitive to this scalar unless explicitly trained to be, and the earlier
   "Elo dial is a weak confidence-knob" result (fingerprint-result file) is a
   warning sign that a single scalar conditioning signal may again do little.

3. **Post-loss affect trigger (Bennett-style state variable)** — a hand-built
   or learned binary/scalar "just lost / losing streak length" feature, reset
   each game, that modulates a risk/time-pressure term (feeds the existing
   clock-conditioning path) rather than the move head directly. Extra input:
   prior game(s) result and time since previous game (session gap) — both in
   PGN metadata. Expected effect: could move clock-usage/blunder-rate
   realism after losses, NOT necessarily move choice — consistent with
   "human feel over accuracy" priority. CPU cost: trivial (feature
   engineering, no new compute). Failure mode: Rosenthal's own-data result
   (item 1) shows no detectable outcome-autocorrelation effect at all once
   Elo is subtracted — so this is directly contradicted by chess-specific
   evidence for *win/loss outcome*; it might still hold for *move quality or
   clock behavior* (untested), but should be treated as a likely dead end
   for move prediction and worth testing cheaply before investing more.

4. **Session-boundary re-embedding, no within-game recurrence** — simplest
   version of (1): keep the style code static within a session but let the
   user's opening book / temperature params be refit or blended per session
   using only that session's early moves (few-shot within-session
   adaptation), justified by the warm-up/diurnal pattern in Leone et al.
   (unverified numerically, treat as weak prior only). Extra input: none
   beyond existing per-session move stream. Expected effect: small,
   opening-adjacent (consistent with "opening gain is player headroom"
   finding) — likely does not touch midgame move accuracy. CPU cost:
   trivial. Failure mode: probably redundant with the existing opening book;
   may just re-derive the already-known opening-only effect rather than
   anything new.

# Bottom line
Chess-specific outcome evidence (item 1, verified) argues tilt-as-win/loss-
streak is a dead end for the move/outcome level. The more promising angle is
architectural: give the model an explicit *time-varying* state slot
(session-level GRU or state-space latent, items 1-2) rather than another
static per-user parameter, since that's what the 0.53-0.58 correlation
finding says is missing — but two prior negative results in this project
(Elo dial weak, judge/DPO circularity) suggest test cheaply on independent
metrics before committing.
