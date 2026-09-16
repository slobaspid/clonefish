# sah-transformer — Project Status & Handoff

> A clock-aware, human-imitating chess model. This doc is a comprehensive snapshot of what's
> built, why, how to use it, and what's next — written as a handoff so any fresh session can
> continue without re-deriving context. Date of this snapshot: **2026-08-12**.
> Repo: https://github.com/slobaspid/clonefish

---

## 0. Current status (what's happening right now)

- A **pondering experiment** is training on **Kaggle** (`notebooks/train_kaggle_ponder.ipynb`,
  ~14M model, 20M positions, resumable). The thing to watch: the **avg-ponder-steps** plot — does
  it rise above ~1.5 as loss drops (pondering engaging) or stay pinned (didn't pay off)? Plus the
  **poke cell** at the end (does it ponder deeper on hard positions / less under low clock).
- **96 tests pass.** Everything below is committed and pushed to `master`.
- Best trained model so far (the reliable one): a **14M-param clock-aware model** trained on 5M
  positions — the user has `best.pt` locally at `C:\Users\sloba\Downloads\model_full (1)\best.pt`
  (also copied to `model_full\best.pt` for the engine's auto-discovery). It plays a strong Najdorf,
  reacts to the clock, and works in the viewer + Cutechess.

---

## 1. The project in one paragraph

Not a chess *engine* (we don't chase the best move) — a **human-imitation** model for **3+0
blitz**. North star: **human realism over strength** (see `human-feel-over-accuracy` memory). It's
a faithful re-implementation of **Maia-3's 5M board-transformer architecture** (reimplemented from
their papers, **our own weights**, no borrowed weights) + **our own contributions**: a
**clock-aware time layer** (reads the remaining clock and modulates play), an **MDN think-time
head** (predicts *how long a human would think*, as a mixture of log-normals), a
**difficulty-into-timing** signal, an **ALLIE-style time-adaptive search**, and an experimental
**clock-gated pondering** (adaptive computation).

---

## 2. Architecture (the model)

Board-input transformer (Maia-style), skill-conditioned, with heads for move + value + think-time.

- **Input** (`sahformer/model/embedding.py`): 64 square tokens, each = `12×history(8)` board planes
  + both players' interpolated **skill embeddings** (128-d each) concatenated per square → 352 →
  `Linear(→256)`. No absolute positional encoding (GAB provides position).
- **Backbone** (`encoder.py`, `gab.py`, `norm.py`): 8 blocks, `dim_vit=256`, 8 heads, RMSNorm
  post-norm, per-layer avg-pool **Geometric Attention Bias (GAB)** with a shared `gab_weight`.
- **Heads** (`heads.py`): **Policy** (from-to bilinear 4096 + 256 promotion = **4352**),
  **Value** (W/D/L, order is `[P(loss),P(draw),P(win)]`), **MDN think-time** (mixture of 3
  log-normals). `policy_difficulty()` feeds the move-spread (entropy + top-prob) into the time head.
- **Model variants** (`build_model(mode, cfg)` in `training/build.py`, modes = `baseline`,
  `film_only`, `gab_only`, `full`, `ponder`):
  - `FaithfulChessformer` (`chessformer.py`) — clock-blind baseline (= plain Maia-3).
  - `ClockAwareChessformer` (`clockaware.py`) — adds the time layer: `TemporalEncoder` (21 clock
    features → context `t`), **FiLM** per block, **time-conditioned GAB**, `t` fused into the MDN.
    Flags `use_film` / `use_time_gab` express the ablation.
  - `PonderChessformer` (`ponder.py`) — clock-aware backbone + **adaptive computation**: a shared
    `PonderBlock` refines the read up to `max_ponder` steps, a clock-fed `HaltHead` decides when to
    stop (PonderNet halting). `forward()` is a drop-in returning the usual dict + `ponder_steps`;
    `ponder_train()` returns per-step outputs + `p_halt` for the PonderNet loss.
- **Sizes:** default ~5.2M (baseline) / ~5.9M (full). Bigger runs used `dim_vit=384,num_blocks=10`
  (~14M) and `512/12` (~31M, too slow on a T4 — reverted to 14M).

## 3. Training pipeline (`sahformer/training/`, `sahformer/dataset*.py`)

- **Data**: Lichess `.pgn.zst`, filtered to `180+0` with `%clk` clocks (only **2017-04+** has
  clocks). `download.py` streams from a URL (`stream_games_from_url`) or local file. `records.py`
  turns games → `PositionRecord` per ply; `shards.py` packs to `.npz`, `balance_indices` equalizes
  22 Elo bins (very aggressive — off by default).
- **Chunked build** (`dataset_build.py::build_shards`): streams records → many `.npz` shards,
  never holding all in RAM (fixes the memory wall). `records_from_games` filters+expands.
- **Datasets** (`dataset.py`): `ShardDataset` (loads all shards, ~1x RAM) and
  `StreamingShardDataset` (loads a few shards at a time — flat RAM for 5M+ positions; use
  `TrainConfig.stream=True`).
- **Loop** (`training/loop.py::train`): AdamW, **manual warmup→cosine LR** (resume-safe), optional
  **AMP**, **gradient clipping (`grad_clip=1.0`, prevents NaN divergence)**, **NaN-batch skip**,
  best/last checkpointing with **self-describing checkpoints** (store `model_cfg` + optimizer
  state). **Resumable** (`TrainConfig.resume=path`, `max_steps` = total target). `load_model(path)`
  rebuilds any-size model automatically.
- **Losses** (`training/losses.py`): `compute_losses` (policy CE + value CE + MDN NLL + `move_acc`
  sanity metric), `per_sample_loss`, and `ponder_loss` (**PonderNet**: halt-weighted per-step loss
  + KL to geometric prior). Pondering failsafes: `ponder_warmup` (uniform-weight all steps early),
  `ponder_min_steps` + `ponder_floor_beta` (floor vs collapse), `ponder_beta` (KL weight).

## 4. Play / inference

- **Self-play viewer** (`sahformer/play.py` + `scripts/watch_selfplay.py`): plays a game vs itself,
  renders a self-contained HTML that replays **paced by the predicted think-time**. Knobs:
  `--elo`, `--temperature`, `--think-temp` (timing calmness), `--search`, `--pace`. Run:
  `.\.venv\Scripts\python.exe scripts\watch_selfplay.py --ckpt "<best.pt>"`.
- **UCI engine** (`scripts/uci_engine.py` + `sahformer_engine.bat`): plays in any chess GUI. Auto-
  finds `best.pt` (env `SAHFORMER_CKPT` / `--ckpt` / `~/Downloads/model_full/best.pt`). UCI options:
  `UCI_Elo`, `Temperature`, `Pace` (play at human tempo — sleep by predicted think), `ThinkTemp`,
  `Search`. **Cutechess** (not En Croissant) is the tool for watching engine-vs-engine; point it at
  `python.exe "<repo>\scripts\uci_engine.py"` (Cutechess dislikes `.bat`). For real blitz feel: set
  time control **Moves=0 (sudden death)** and `Pace` (smaller = slower).
- **Time-adaptive search** (`sahformer/search.py`): budgeted **PUCT MCTS**, budget from predicted
  think-time (`time_to_sims`), policy head = prior, value head = leaf eval. Human-realism capped
  (`max_sims`). Drop-in via `--search` (viewer) or `Search` UCI option.

## 5. Research grounding (verified)

- **Maia / Maia-2** — board-transformer human move-matching, per-skill. We copied the *architecture
  design*, not weights.
- **ALLIE** ("Human-Aligned Chess with a Bit of Search", 2024) — the model with policy+time+value
  heads + **time-adaptive MCTS** (search budget ∝ predicted human think-time). Trained on 91M games
  (~6.6B positions). Weights are **open** (HuggingFace `yimingzhang/allie-models`,
  `github.com/ippolito-cmu/allie`), playable on Lichess. Our search is our own build of this idea;
  our model is board-based (Maia) not move-sequence (ALLIE), and adds live clock-awareness neither
  has. To get ALLIE's quality into our architecture you'd need **distillation** (open item), not
  weight-copying (incompatible architectures).
- **PonderNet** (Banino 2021) / **ACT** (Graves 2016) — our pondering is a faithful PonderNet:
  `L = Σ p_n·L(ŷ_n) + β·KL(p ‖ geometric prior)`, halting `p_n = λ_n·Π(1-λ_m)`. Our novel twist:
  the halting is **clock-gated**. Known risk: halting collapse — mitigated by the failsafes.
- Human-timing facts we relied on: chess response times are **heavy-tailed/power-law** (justifies
  the MDN); **surprisal ∝ reaction time** (justifies the "opponent-move-surprise" idea); think-time
  prediction is hard (ALLIE r≈0.70, ChessMimic r≈0.41 — expect a ceiling).

## 6. Specs & plans (in `docs/superpowers/`)

- `specs/2026-08-11-clock-aware-chessformer-design.md` — the parent design.
- `plans/2026-08-11-model-backbone-v2-faithful.md`, `plans/2026-08-11-time-modulation-v2.md` — built.
- `specs/2026-08-11-training-pipeline-design.md` + `plans/2026-08-11-training-pipeline.md` — built.
- `specs/2026-08-12-time-adaptive-search-design.md` + `plans/2026-08-12-time-adaptive-search.md` — built.
- `specs/2026-08-12-clock-gated-pondering-design.md` + `plans/2026-08-12-clock-gated-pondering.md` — built.

## 7. Key decisions & rationale

- **Faithful Maia backbone + only the time layer as ours** → the clock-blind version is a clean
  ablation baseline; results are comparably interpretable.
- **Human realism > strength** (memory) → we deliberately don't optimize move-accuracy (it's a
  logged sanity metric; best checkpoint chosen by total loss). Search budgets are small/human-capped
  so it doesn't become an engine.
- **Our own weights, not ALLIE's** → the project's value is building/understanding/extending; ALLIE
  is a benchmark/teacher option, not a shortcut we took.
- **Communication: plain English** (memory `plain-english-communication`) — the user is learning
  the domain; explain simply, no jargon dumps, until told otherwise.

## 8. Known risks / gotchas

- **Pondering may not engage** — on a barely-trained model it sits at the floor (~1.5 steps). Watch
  the avg-steps plot on the real run; if it never rises above the floor, pondering didn't pay off
  (informative, not a bug).
- **NaN divergence** — was caused by *missing gradient clipping* on a big fp16 model; **fixed**
  (`grad_clip=1.0` + NaN-batch skip). Can't be fully tested locally (CPU has no AMP), but stability
  verified for 120 steps.
- **Kaggle limits**: working disk ~20 GB → **~100M positions won't fit** (~20-30M is the ceiling).
  GPU is time-limited → big runs must be **resumable** across sessions (they are). Enable
  **Persistence: Files only** so shards + checkpoints survive.
- **Local machine is CPU-only** (`torch 2.4.0+cpu`) → training happens on Kaggle/Colab; local is for
  the viewer/engine/tests only. Colab was rate-limited/blocked earlier → Kaggle is the workhorse.
- **Env**: Windows, Git Bash + PowerShell. `.venv` at repo root (`./.venv/Scripts/python.exe`).
  `git rm`/`git commit` sometimes need retries (auto-mode classifier); permissions allow-listed in
  `.claude/settings.local.json`. LF→CRLF warnings on commit are harmless.

## 9. How to run things (quick reference)

```bash
# tests
./.venv/Scripts/python.exe -m pytest -q

# watch it play itself (local, needs a best.pt)
.\.venv\Scripts\python.exe scripts\watch_selfplay.py --ckpt "C:\Users\sloba\Downloads\model_full\best.pt" --elo 2000

# build a shard from a local .pgn.zst (chunked)
PYTHONPATH=. ./.venv/Scripts/python.exe scripts/build_dataset.py "<file>.pgn.zst" data/ --max-positions 300000

# train (CLI; notebooks are the usual path)
PYTHONPATH=. ./.venv/Scripts/python.exe scripts/train.py "data/*.npz" --mode full --max-steps 8000 --stream --resume checkpoints/full/last.pt
```

- **Kaggle notebooks**: `notebooks/train_kaggle.ipynb` (full model) and
  `notebooks/train_kaggle_ponder.ipynb` (pondering). Import from GitHub → GPU + Internet +
  Persistence: Files only → Save & Run All (Commit) → download `model_*.zip` from Output.
- **Cutechess self-play**: add engine `python.exe "<repo>\scripts\uci_engine.py"`, working dir the
  repo; init strings `setoption name Pace value 0.5` / `UCI_Elo 2000` / `Temperature 0.2`; New Game
  Sahformer vs Sahformer, sudden-death time control.

## 10. Open items / next steps

- **The eval plan (not yet built)** — the honest measurement stage:
  - `full` vs `baseline` on **unseen games**: does the clock help *move-match* and especially the
    **time-pressure / sacrifice subset**?
  - **Think-time calibration** vs real humans; **thought-depth vs Elo** (using search sims-to-
    converge and ponder-steps — the user's question about measuring "thought depth" by rating).
  - Sampled / non-deterministic **play** measurement (deferred stochasticity discussion).
- **Pondering outcome** — read the current run's avg-steps + poke cell; tune floor/beta or drop it.
- **More data** — closeable gap vs ALLIE (we've used <1% of one month); streaming + resume make it
  feasible in chunks. More real data is the surest upgrade.
- **Distillation from ALLIE** (optional, ambitious) — the only way to move ALLIE's weight-knowledge
  into our architecture; makes us partly a mirror of theirs.
- **"Opponent-move surprise" difficulty signal** — a cheap, principled feature (surprisal→RT) that
  catches the hard-but-forced (sacrifice-recapture) case the move-spread misses.

## 11. The through-line (what got built this session)

Empty folder → faithful Maia backbone → clock-aware time layer → training pipeline → trained a 14M
model on 5M real games → self-play viewer → UCI engine (Cutechess) → the user's difficulty-into-
timing idea → ALLIE-style time-adaptive search → resumable + streaming training → clock-gated
pondering (with anti-collapse failsafes + NaN-safe training). All grounded in real research
(Maia/ALLIE/PonderNet), all the user's own model.
