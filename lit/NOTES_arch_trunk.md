# In-trunk per-user conditioning: candidate architectures

Angle: speaker-adaptation literature (ASR/TTS) on how to inject a person's
identity INSIDE a frozen trunk cheaply, not append a residual after it.

## Sources verified (page-1 + grep, arxiv ids in lit/pdf/)

1. **LHUC** — Swietojanski, Li, Renals, "Learning Hidden Unit Contributions
   for Unsupervised Acoustic Model Adaptation" (1601.02828, TASLP).
   Per-speaker scaling vector r^l (one scalar per hidden unit) per layer,
   diagonal transform, learned by backprop with everything else frozen.
   WER reductions **10-32%** relative depending on task/data, using only
   a **few minutes** of unsupervised adaptation data. Extended to SAT-LHUC
   for joint speaker-independent/dependent training.

2. **Residual Adapters for TTS** — Morioka et al., Google (2210.15868,
   ICASSP). Small bottleneck adapter (rank r) inserted per Tacotron/FastSpeech
   layer, backbone frozen. r=16 -> **0.12% of backbone params**, MOS
   4.46 vs 4.51 reference (near-parity); r=128 -> 0.89% params, no real
   extra gain (diminishing returns above ~0.1-0.2%). Trained on **30 min**
   of target-speaker data; degrades gracefully at 5 min / 1 min (shown in
   text but not re-quoted numerically here).

3. **HyperTTS** — Li, Bhardwaj, Mehrish et al. (2404.04645, LREC 2024).
   A hypernetwork (<1% of TTS backbone size) *generates* the adapter
   weights per layer from a speaker embedding, instead of learning
   separate adapter weights per speaker. Reported adapter budgets of
   **0.1-0.56% trainable params** across variants; competitive with full
   fine-tune and with static (non-hyper) adapters at matched budget, and
   scales better as more speakers are added (one hypernetwork amortizes
   across all speakers instead of one adapter set per speaker).

4. **BitFit** — Ben-Zaken, Ravfogel, Goldberg (2106.10199, ACL 2022).
   Train only bias terms (all of them, or just query+FFN-mid biases).
   **0.08-0.09% of BERT params**; competitive with full fine-tune
   specifically in the **small-to-medium training-data regime** (their
   selling point, not huge corpora) on GLUE.

5. **Scaling & bias codes** — Luong & Yamagishi, NII/Edinburgh
   (1807.11632, Interspeech). Per-speaker scaling code (elementwise gain,
   sizes swept 1-128 dims) and/or bias code injected at a chosen hidden
   layer of a DNN speech-synthesis acoustic model, unifying i-vector /
   speaker-code / LHUC-style approaches into one framework. Scaling code
   alone gets most of the MCD improvement; bias code alone is
   consistently worse; combining is best. Confirms **gain saturates fast
   with code size** (small vectors, order tens of dims, already capture
   most of the benefit) — same diminishing-returns shape as adapters.

## Common quantitative pattern across all 5
- Per-user budget that works: **0.05-0.2% of trunk params** (LHUC: one
  scalar/hidden-unit; adapters/codes: low-rank or small vector at one or
  few layers). Going 10x bigger does not buy much more (residual-adapter
  and scaling-code papers both show explicit saturation).
- Data needed: minutes, not hours (LHUC: few minutes unsupervised; TTS
  adapters: 30 min, still working at 1-5 min degraded).
- All are gradient-fit against a **frozen** backbone — cheap, CPU-feasible
  fits since only a tiny parameter set gets gradients.

## Mapping to our 19.5M model (8 blocks x 2.18M + 0.26M head)
Candidates ranked below by (expected gain x cheapness to test).
