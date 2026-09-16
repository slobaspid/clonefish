# Night A — pretraining a base that adapts best per-person (verified from arXiv abstracts)

| Paper | arXiv | Verified claim | Maps to our Chessformer (8 blocks, FiLM, Elo emb) |
|---|---|---|---|
| AdaSpeech | 2103.00993 | Conditional LayerNorm in decoder; adapts with ~20 sentences (~1 min), only ~5K params/speaker | Our FiLM is already conditional-norm-like, but driven by clock, not by identity. Add a per-player slot to the FiLM generator so adaptation = tuning a tiny embedding, not adapters bolted on after. |
| Sample-Efficient Adaptive TTS (SAT-style) | 1809.10460 | Shared WaveNet core + small per-speaker embedding learned fast from few samples | Same shape as our base + Elo embedding; suggests giving each player their own embedding slot *during pretraining*, not just at fine-tune time. |
| Meta-TTS (MAML for personalization) | 2111.04040 | MAML finds an init that adapts to a new speaker in far fewer steps than plain fine-tune-from-scratch adaptation | Directly answers the question: don't just pretrain on volume, pretrain the *init* to be one adapt-step away from any player. |
| Reptile | 1803.02999 | First-order meta-learning: repeatedly sample task, train, move init toward trained weights | Cheap way to run the Meta-TTS idea without MAML's 2nd-order cost — sample random player slices of the corpus as "tasks" during pretraining. |
| (IA)^3 / T-Few | 2205.05638 | Learned rescaling vectors, tiny param count, beat SOTA by 6% absolute on RAFT | Cheaper than our rank-16 adapters (+2.19pp, best midgame so far) — worth a placement-bench entry, but doesn't change pretraining. |
| LoRA learns less, forgets less | 2405.09673 | Full FT's needed weight-perturbation rank is 10-100x a typical LoRA rank; LoRA forgets less | Explains why our full-FT (+3.41pp, but hurts other players) beats rank-16 adapters (+2.19pp, never negative) — adapters are capacity-capped, not just placement-capped. |
| WiSE-FT | 2109.01903 | Interpolating base and fine-tuned weights gives 2-23pp robustness gain, 0.8-3.3pp accuracy over plain fine-tune | Gives a dial between our two extremes (safe adapters vs risky-but-bigger full FT) instead of picking one. |
| HyperFormer | 2106.04489 | Shared hypernetwork generates per-task adapters, +0.29% params/task, helps few-shot generalization | Already close to MHR (2502.14998, already read) — not new, skip re-proposing. |

## Top 2 ideas + falsification

1. **Meta-learn the init (Reptile-style), not just pretrain on volume.** During base pretraining, periodically sample a random player's slice, take one adapt step (rank-16 adapter or FiLM slot), and nudge the init toward that adapted point — literally what Meta-TTS/Reptile do for speakers. *Falsify:* fine-tune both bases (plain vs Reptile-pretrained) identically on held-out players; if post-opening top-1 gain isn't >1.2pp better than today's +2.19pp adapter result (our measured per-player noise floor), kill it.

2. **Bake a per-player FiLM/embedding slot in from pretraining (AdaSpeech/SAT-style)**, fed a large pool of dummy player IDs during pretraining so the backbone learns to route personalization there, then adapt only that small embedding on the real person. *Falsify:* if tuning only the embedding doesn't beat the current +2.19pp adapter baseline, it adds nothing over what we already have.
