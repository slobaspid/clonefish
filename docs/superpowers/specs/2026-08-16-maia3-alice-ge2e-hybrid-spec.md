# Maia-3 + Alice + GE2E — hybrid style-fingerprint (spec)

**Date:** 2026-08-16
**Status:** design, not started
**Context:** follows the fingerprint result in `HANDOFF_2026-08-16.md` §4/§6. This is the "v2 hybrid" that
was sketched but not built.

---

## 0. One sentence

Fingerprint a player by their **deviation from two frozen "average human" models** — Maia-3 (generic moves)
and Alice (generic clock) — pooled into one **style embedding**, trained with **GE2E** so it also identifies
people cheaply at scale. One model, **two read-outs**: the interpretable/playable deviation-likelihood
(small N) and cheap nearest-centroid in the embedding space (huge N).

**Core principle — pure deviation.** Both channels are anchored to a frozen generic: moves =
`Maia_logits + player_residual`, timing = `Alice_generic + player_deviation`. The generic score is identical
for every candidate player, so it **drops out of the "who is this?" comparison** — identification is driven
*entirely* by deviation from the norm, on both channels. That's the strength-control baked into the model:
Alice/Maia absorb the shared, strength-level behavior; the fingerprint carries only personal style.

## 1. The claim this is built to test

> In a **strength-controlled** setting, adding a **timing channel** beats the moves-only state of the art
> (McIlroy-Young et al. 2021: **54% P@1 at 41,184 players**, 100 games, moves only) — or, honestly, map the
> N at which timing washes out.

Our current generative fingerprint (90 players, 0.91@10-games) **cannot** run this — it costs one forward
pass per candidate player. GE2E replaces that with one embedding + nearest-neighbor. That is the whole
reason to build this.

## 2. What we reuse (important: no re-caching)

`cache_film.pt` already stores, **per ply**, in game order:

| tensor | meaning | use here |
|---|---|---|
| `pooled` [N,512] | Maia-3 position summary | the per-move token sequence for the game encoder |
| `think` [N] | actual think-time (s) | timing input + Alice target |
| `llog` [N,72] | base legal-move logits | aux generative move head |
| `legal`,`acol` | legal move ids, played column | aux generative move head |
| `pid`,`game`,`test` | player, game id, split | grouping + GE2E labels + held-out eval |

Rows for one game are **contiguous and in ply order** (parallel parse keeps a player's records together;
cache concat preserves it). So grouping by `game` and keeping cache order reconstructs the move sequence.
**We do not need `enc` (the 64× cache).** The game encoder aggregates the per-ply `pooled` vectors itself.

## 3. The three components

### 3a. Maia-3 (base) — the *what move* channel
Frozen. Supplies `pooled` (512) per move. Stays frozen for v2 (cheap, reuses cache); fine-tune only if
Stage 3 says the backbone is the bottleneck.

### 3b. Alice — the *how they use the clock* channel (frozen anchor, like Maia)
A **population** think-time MDN over `think` conditioned on `pooled` (+ the base temporal features), trained
on **all** reference plies with **no identity**, then **frozen**. It is the "generic human clock" — the exact
timing counterpart to Maia's generic move logits. It plays two roles:

1. **Deviation anchor (primary).** The per-player timing head outputs small **offsets** on Alice's frozen
   MDN params — shift the "tank" mode longer, reweight toward snapping, widen/narrow spread — starting at
   zero (untrained fingerprint == Alice exactly). Final clock model = `Alice + player_offset`; the fingerprint
   learns **pure deviation**, and Alice's shared score cancels in the who-is-this argmax (§0).
2. **Encoder feature.** Per ply, `surprise = -logp(think | Alice)` (+ raw `log_think`) is fed to the game
   encoder as the timing-deviation signal it pools into the embedding.

Alice is the population version of the timing MDN already in the fingerprint — a ~10-line head, trained once,
frozen, reused everywhere.

### 3c. GE2E — the *scale* mechanism
Metric-learning loss from speaker verification. Pulls a player's game vectors toward their own centroid,
pushes them from other players' centroids. Gives cheap nearest-centroid ID over huge N.

## 4. Architecture (concrete shapes)

Per ply token (built from cache, no base forward pass needed):
```
tok_i = Linear512→256(pooled_i)  +  Linear2→256([log_think_i, surprise_i])   # d = 256
```

Game encoder (McIlroy-Young style, kept small first):
```
seq   = [tok_0 … tok_L]  + learned positional emb over ply index   (cap L = 60 own-moves)
enc   = Transformer(d=256, layers=4, heads=4, mlp×4, pre-norm)      # masked for padding
g_vec = L2norm( mean_pool(enc) )                                    # one 256-vec per game
```
Player embedding:
```
p_vec = L2norm( mean over that player's game vectors )             # 256-vec
```

Sizes: ~17k games (90 players × ~190), ≤60 tokens × d256 — trivial on one GPU.

## 5. Losses

1. **GE2E (primary).** Batch = **M players × K games** (e.g. 32×6). For each game vec, cosine-sim to every
   in-batch player centroid (centroid = mean of that player's *other* games), scaled `w·sim+b`, softmax CE
   toward its own player. Standard GE2E.
2. **Aux generative — the deviation likelihood (keep it).** Conditioned on the player/game embedding, two
   small heads predict the player's actual games as **deviation from the frozen anchors**: move =
   `Maia_logits + residual`, timing = `Alice_params + offset` (§3b). This is the interpretable/playable
   read-out *and* it regularizes the embedding toward "real deviation."
   `loss = GE2E + λ_aux·(move_nll + λ_time·time_nll)`, start `λ_aux = 0.3`.

**Two identification read-outs from the one model:**
- **Cheap / scale:** nearest-centroid in the GE2E embedding space (works at 41K).
- **Rich / small-N:** which fingerprint's deviations best explain the games (the generative score) —
  interpretable and playable. Both come free from the same trained model.

## 6. Eval protocol

- **Closed-set P@1** at N ∈ {10, 30, 60, 90, …scale}: enroll each player from their reference games
  (centroid), classify each held-out **query** (Q games averaged) by nearest centroid. Report Q ∈ {1,5,10,100}.
- **Timing ablation at every N:** train/eval three ways — moves-only tokens, timing-only tokens, both — so
  we see the timing gap *as a function of N* (the actual research question).
- **Baselines to beat:** our per-ply generative FiLM/concat numbers (single-game, 10-game) and, at scale,
  McIlroy-Young 54%@41,184.

## 7. Staged plan

| stage | what | data | output |
|---|---|---|---|
| 0 | Train + freeze **Alice** (population clock model); dump `log_think`+`surprise` per ply | existing `cache_film` | frozen clock anchor + timing features |
| 1 | **Individual deviation fingerprint**: keep the current per-player embedding table, but make timing a **deviation-from-Alice** (moves already deviate from Maia). Same eval. | existing 90-player cache | clean before/after vs current **0.91@10-games** — does the Alice anchor sharpen it? |
| 2 | **Add GE2E on top**: swap the per-player table for a **game encoder** (inductive), train GE2E + keep Stage-1 deviation heads as the aux. Now dual read-out. | same cache | P@1-vs-N curve + timing gap; embedding scales to unseen players |
| 3 | **Scale test** — thousands of players, both read-outs, timing ablation at each N | **NEW harvest** (see below) | the headline claim vs 54%@41K |

Stage 1 validates the *deviation idea* cheaply (small change to the working notebook). Stage 2 is where GE2E
earns its place — only worth it once Stage 1 shows the Alice-anchored deviation helps. Don't build the
encoder before that number is in.

**Stage 3 needs new data.** The 90-player cache is the strength-*controlled* set (keep it — it's the clean
experiment). Scaling to thousands means a fresh harvest: lower the ≥1000-game threshold, or widen the Elo
band and add a strength control as a covariate. This is a separate data-build task, flagged not scoped here.

## 8. Open risks / decisions (keep attached to any result)

- **Does timing survive scale?** More players → more clock-rhythm collisions; timing may partially wash out.
  That's the experiment, not a bug.
- **Freeze vs fine-tune base** — freeze for v2; revisit only if Stage 3 stalls.
- **GE2E may edge the generative head on raw accuracy but lose interpretability** — the aux head is the hedge;
  if λ_aux hurts accuracy, drop it and accept opacity.
- **Leakage** — base trained on these players; a *fully* clean scale claim wants players/games the base never
  saw. Note it; don't over-invest until the method works.
- **Pooling choice** — mean-pool first; try attention-pool only if mean underperforms.

## 9. Rough cost

Stages 0–2 reuse the cache → minutes-to-an-hour of single-GPU training each, ~Kaggle-free tier. Stage 3 cost
is dominated by the **new harvest + re-cache**, not training.

## 10. First concrete step

Add **Stage 0 + Stage 1** as cells to the existing Kaggle notebook (it already loads `cache_film` and groups
by `game`). Moves-only GE2E first, get the P@1-vs-N curve, then switch timing on and read the gap. Everything
else is downstream of that number.
