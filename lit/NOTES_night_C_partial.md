# Night C — which part of the network should be personal, how to shrink it

Verified from arXiv/PMLR abstract pages only.

| Paper | ID | Verified claim | Maps to our facts |
|---|---|---|---|
| Pillutla et al., Federated Learning with Partial Model Personalization | 2204.03809 | Partial personalization gets "most of the benefits of full model personalization with a small fraction of personal parameters." | Matches our adapters-after-every-block: 0.135M params, +2.19pp vs full-FT +2.72pp — small fraction, most of the gain, and never negative (full-FT was worse cross-player). |
| Collins et al., FedRep | 2102.07078 | Shared low-dim body (representation) + personalized head/classifier per client. | Contradicted by our bench: head-only +1.83, last-block+head +1.62 — both worse than distributing adapters through every block (+2.19). Head-only personalization is not enough for us. |
| Oh et al., FedBABU | 2106.06042 | Body trained/shared; head left random during training, only personalized (fine-tuned) at test time. | Same "personalize only the top" family as FedRep. Same contradiction: our data says the personal signal is spread through the trunk, not concentrated at the head. |
| Liang et al., LG-FedAvg | 2001.01523 | Opposite split: personalize the *local representation/encoder*, share the global classifier on top. | Also a "personalize one end only" design (early instead of late). Neither end-only split beats every-block adapters in our bench — supports "distribute personalization across depth," not "pick one end." |
| Simchoni & Rosset, LMMNN | 2206.03314 | Random effects get their own branch; trained via Gaussian NLL so effect size is shrunk by estimated variance, not a fixed penalty. | We currently use a fixed rank-16 adapter + implicit L2 regardless of a player's game count (1.2k vs 5k). No variance-based shrinkage yet. |
| Nguyen et al., ARMED | 2202.11783 | Adversarial classifier forces trunk to be cluster-invariant; separate random-effects subnet carries cluster-specific signal; explicit mechanism to assign random effects to *unseen* clusters. | Matches our trunk+adapter split directly. Their unseen-cluster mechanism maps to fact 6 (trait-matched cohort recovers ~half the self-FT gain for a new player) — a natural empirical-Bayes prior. |
| Li, Grandvalet & Davoine, L2-SP | 1802.01483 | Penalize distance to the *pretrained* weights (not zero) during fine-tuning, as an explicit inductive-bias anchor. | Could explain/fix fact 3's failure mode: full-FT gets the biggest gain (+2.72) but goes -0.7..-5.8pp on a different player. An L2-SP anchor, not a parameter-subset switch, may be the actual fix. |

## Top 2 ideas

**1. Empirical-Bayes shrinkage of the adapter, not a fixed rank/L2.** Scale each player's adapter output by a data-driven shrinkage factor (e.g. n_games/(n_games+k), LMMNN/ARMED-style) instead of the current fixed rank-16 + flat L2, so a 1.2k-game player and a 5k-game player aren't regularized identically.
*Falsify:* fit k on held-out players; if shrunk adapters don't beat the flat +2.19pp post-opening baseline, or don't shrink the negative-transfer tail for low-data players, drop it.

**2. L2-SP anchor instead of switching to adapters.** Keep full fine-tuning (best single number, +2.72pp) but penalize ||θ−θ₀||² with strength swept by game count, aiming to keep the gain while killing the cross-player brittleness (fact 3).
*Falsify:* if anchored full-FT on player A still degrades player B's held-out set by a similar margin as vanilla full-FT, or its gain isn't distinguishable from plain adapters given our ~1.2pp/11-player noise floor (fact 10), it buys nothing.
