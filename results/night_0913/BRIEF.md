# Brief for research / review agents — clonefish architecture search (night of 2026-09-12)

## The product
"clonefish": give it one person's Lichess / chess.com games (3+0 blitz), get back a chess engine that
plays like them. Targets set by the user tonight:
- **top-3 move match** on the person's held-out games (not top-1),
- **the think-time distribution** (how long they spend per move, sampled, not averaged),
- typical person has **~5,000 games** (~40 own moves/game => ~200k positions).

## What we have
- **Base model**: 19.5M-param clock-aware Chessformer (Maia-3-like encoder, 64 square tokens, 8 blocks,
  geometric attention bias, FiLM per block driven by clock/temporal features, Elo embedding at input).
  Heads: policy (4352 moves), value, think-time mixture-of-lognormals (MDN). Trained on a large
  balanced chess.com 3+0 corpus. Checkpoint `checkpoints/base_300k_best.pt`.
- **CosFace identifier**: game encoder that identifies a player among 3,000 from 5 games at 0.956 P@1,
  10 games 0.992. Single game among 3000: 0.39.
- GPU: one GTX 1060 6GB locally (rentable 3090 possible). CPU Xeon 32GB.
- Data: 6 Lichess players with ~5k 3+0 games each (`data/lichess_5k/`), 100 players ~1.2k games
  (`data/lichess_1k/`), 3,000-player Lichess scale set, big chess.com corpus.

## What has been MEASURED (do not re-propose these as new; build on them)
Held-out = the person's newest 40 games. "Post-opening" = own move >= 12.
1. Residual/code bolted on AFTER the frozen trunk (the user's current approach): opening-only.
   +0.55 pp top-1 after move 10, **+0.001 pp middlegame**. Bigger shared head / more players: no help.
2. Personal count-based opening book (Dirichlet blend with base): +3.6 pp overall, opening only,
   verified personal (other players' book +0.63).
3. **Full fine-tune on 5k games**: post-opening top-1 **+3.41 pp mean** (n=7, range +0.8..+5.8),
   middlegame +3.0. Gain anti-correlates with base strength (r = -0.58): predictable players gain little.
   Fine-tuned model is WORSE on a different player (mostly -0.7..-5.8 pp) => identity, not domain.
4. Parameter placement bench (1.2k games, n=3 players), post-opening top-1 vs base:
   full FT +2.72 | **rank-16 bottleneck adapters after every block (0.135M) +2.19, best midgame, never
   negative** | last block+head +1.62 | BitFit +0.99 (goes negative) | FiLM-generator only +2.11 (n=1) |
   LHUC +1.70 (n=1) | head only +1.83 (n=1).
5. Top-3 gains are much smaller than top-1 gains past the opening (VEGETAL full FT: middlegame top-1
   +5.5 but top-3 +2.5; endgame top-3 ~0). Base top-3 is already ~82-90%.
6. Trait-matched cohort of 5 other same-Elo players recovers ~half the self fine-tune gain (n=1-3).
7. Time head: 8 different head designs tie within 7%; **sampling vs taking the mean** is the whole
   win for distribution realism. Per-user timing residual over-thinks. A per-player lookup table
   beats the model on snap-rate/tail-rate for half of players. Time ceiling = base representation.
8. Distribution-level DPO / GAN with the recognizer as reward: failed 3x on independent metrics.
   Never put a discriminative identity loss on the generator.
9. In-context prompting with raw games: rejected (identity is weak per game). Retrieval clone +0.4
   midgame. Search on top of policy: +0.2-0.8 pp at blitz.
10. Paired sd between methods ~1.2 pp per player => need ~11 players to detect a 1 pp difference.
11. Run-to-run noise of one fine-tune: 0.13 pp.

## Literature ALREADY read (in lit/, don't re-summarise; cite if relevant)
Maia-individual (2008.10086), Maia4All (2507.21488, prototype enrichment), MHR / LoRA bank + routing
(2502.14998), Carlson Elo-disentangled style embeddings (2606.25176v1), UCLouvain player-specific
(2605.11893), stylometry (2208.01366), ChessMimic (2606.04473), Chessformer (2605.19091), Maia-2
(2409.20553), Allie (2410.03893), Mixture of Masters (2602.04447), LHUC, residual adapters TTS,
HyperTTS, BitFit, scaling/bias codes, VALL-E, Prompt-DT, ICRT, OPPU, Per-Pcs, P2P hypernetwork
(2510.16282), USER-LLM, low-rank+sparse personalization (2210.03505), DreamBooth, textual inversion,
Ditto, three approaches for personalization, kNN-LM adaptive, DAgger, offline driving eval, Navigation
Turing test, Welch similar users, CNPs, fine-tune vs meta-learning for few-shot imitation (2306.13554),
LANs (likelihood approximation networks), response-time preference identification (2507.20403),
OT imitation, IRL multi-motivation, GRU4Rec/HGRU4Rec, chess streaks, state-space skill rating.

## Rules for every agent
- Verify every paper from its primary source (arXiv abstract page / PDF). Quote numbers only if you
  saw them. Mark anything unverified as UNVERIFIED. Never invent a citation.
- Plain English. Short sentences. Explain any jargon the first time.
- Always say how an idea maps onto OUR model and OUR measured facts above, and what would falsify it.
