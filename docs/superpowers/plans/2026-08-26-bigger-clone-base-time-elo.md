# Bigger Clone Base — Bucket Time-Head + First-Class Elo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native 30-bucket think-time head (fixes the MDN "always more" snap/tank hedge) and first-class Elo (gap conditioning + an auxiliary rating head), all gated behind config so the existing base still loads, then train a ~30M base and wire it into play + fine-tune.

**Architecture:** All changes are additive and gated by two new `ModelConfig` fields (`time_head`, `predict_elo`). The model gains a `BucketTimeHead` and `EloHead` in the `time_head="bucket"` path; the MDN path is untouched (backward compat). No corpus rebuild — think-time is bucketed at loss time, elo targets already exist in shards, and the elo-gap is computed inside the model from batch elos.

**Tech Stack:** PyTorch 2.4, python-chess, numpy, pytest. Windows dev (1060 6GB) for smoke tests; rented 3090 for the full run.

**Spec:** `docs/superpowers/specs/2026-08-26-bigger-clone-base-time-elo-design.md`

**Conventions:** run tests with `PYTHONPATH=. python -m pytest <path> -v`. Commit after each task. Keep tensors CPU-safe (dev box is 6GB).

---

## File structure

- Create `sahformer/model/timebuckets.py` — bucket edges/centers, `to_bucket`, `sample_think_time`. One responsibility: the discretisation + sampling of think-time.
- Modify `sahformer/model/heads.py` — add `BucketTimeHead`, `EloHead` (alongside existing heads).
- Modify `sahformer/model/config.py` — add `time_head`, `time_buckets`, `predict_elo` fields.
- Modify `sahformer/model/clockaware.py` — build the new heads when configured; branch `forward`; add elo-gap conditioning.
- Modify `sahformer/training/losses.py` — branch time loss (bucket CE) + add elo loss.
- Modify `sahformer/training/loop.py` — thread `w_elo` through `TrainConfig` + `compute_losses`.
- Modify `sahformer/play.py` — use the bucket sampler when the model exposes `time_logits`.
- Modify `scripts/uci_engine.py`, `scripts/clone_uci.py` — same sampler at play.
- Modify `scripts/finetune_clone.py` — bucket CE loss for the time head when the clone base is bucket-headed.
- Create `scripts/smoke_bucket.py` — tiny local run asserting the new heads train + a self-play spends the clock.
- Create `tests/test_timebuckets.py`, `tests/test_bucket_head.py`, `tests/test_losses_bucket.py`, `tests/test_model_bucket_forward.py`.

---

## Task 1: `timebuckets` module (edges, buckets, sampler)

**Files:**
- Create: `sahformer/model/timebuckets.py`
- Test: `tests/test_timebuckets.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_timebuckets.py
import numpy as np
import torch
from sahformer.model.timebuckets import EDGES, CENTERS, N_BUCKETS, to_bucket, sample_think_time


def test_bucket_layout():
    assert N_BUCKETS == 30
    assert len(EDGES) == N_BUCKETS + 1
    assert EDGES[0] == 0.0 and EDGES[-1] == float("inf")
    # 1-second bins up to 27s
    assert to_bucket(0.2) == 0        # snap bin [0,1)
    assert to_bucket(0.99) == 0
    assert to_bucket(1.0) == 1
    assert to_bucket(26.5) == 26
    assert to_bucket(27.5) == 27      # first wide bin [27,32)
    assert to_bucket(50.0) == N_BUCKETS - 1   # tail 40+


def test_sampler_never_exceeds_clock():
    # logits favouring the tail, but only 3s left -> must sample <= 3s
    logits = torch.zeros(N_BUCKETS); logits[-1] = 10.0
    rng = np.random.default_rng(0)
    for _ in range(200):
        t = sample_think_time(logits, remaining_clock=3.0, rng=rng)
        assert 0.0 <= t <= 3.0


def test_sampler_can_snap_and_tank():
    rng = np.random.default_rng(0)
    snap = torch.full((N_BUCKETS,), -10.0); snap[0] = 10.0
    tank = torch.full((N_BUCKETS,), -10.0); tank[-1] = 10.0
    s = [sample_think_time(snap, 180.0, rng) for _ in range(100)]
    t = [sample_think_time(tank, 180.0, rng) for _ in range(100)]
    assert np.mean(s) < 1.0                       # snaps land in [0,1)
    assert np.mean(t) > 20.0                       # tank lands in 40+ (capped)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/test_timebuckets.py -v`
Expected: FAIL with `ModuleNotFoundError: sahformer.model.timebuckets`

- [ ] **Step 3: Write minimal implementation**

```python
# sahformer/model/timebuckets.py
"""Discretised think-time buckets (ChessMimic-style) + a clock-masked sampler.

1-second bins for [0,27)s (covers the vast majority of 3+0 blitz moves), then wider
tail bins to a half-open 40+ bucket. 30 buckets total. Play samples a bucket by
probability (masking buckets beyond the remaining clock so it never self-flags), then
draws a time within the bucket.
"""
import math
import numpy as np
import torch

# edges: 0,1,...,27, 32, 40, inf  -> 27 one-second bins + 3 tail bins = 30 buckets
EDGES = [float(i) for i in range(0, 28)] + [32.0, 40.0, float("inf")]
N_BUCKETS = len(EDGES) - 1                       # 30

# representative time per bucket (for expected-value readouts / diagnostics)
CENTERS = []
for i in range(N_BUCKETS):
    lo, hi = EDGES[i], EDGES[i + 1]
    CENTERS.append(0.3 if i == 0 else (50.0 if math.isinf(hi) else (lo + hi) / 2.0))
CENTERS = np.array(CENTERS, dtype=np.float32)


def to_bucket(t: float) -> int:
    """Bucket index for a think-time in seconds."""
    t = max(float(t), 0.0)
    for i in range(N_BUCKETS):
        if EDGES[i] <= t < EDGES[i + 1]:
            return i
    return N_BUCKETS - 1


def sample_think_time(logits, remaining_clock: float, rng, tail_cap: float = 30.0) -> float:
    """Sample a think-time (seconds) from bucket `logits` (1-D tensor, length N_BUCKETS).
    Buckets whose lower edge exceeds `remaining_clock` are masked out (never self-flag).
    Draw a bucket by probability, then a time uniformly within it (tail bin: [40, tail_cap])."""
    logits = logits.detach().float().cpu()
    lo = torch.tensor(EDGES[:-1])
    mask = (lo <= float(remaining_clock)).float()
    if mask.sum() == 0:
        return 0.0
    p = torch.softmax(logits, dim=-1) * mask
    p = (p / p.sum()).numpy()
    k = int(rng.choice(N_BUCKETS, p=p))
    a, b = EDGES[k], EDGES[k + 1]
    b = min(b, remaining_clock, tail_cap if math.isinf(b) else b)
    a = min(a, b)
    return float(rng.uniform(a, b))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. python -m pytest tests/test_timebuckets.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add sahformer/model/timebuckets.py tests/test_timebuckets.py
git commit -m "feat: think-time bucket discretisation + clock-masked sampler"
```

---

## Task 2: `BucketTimeHead` + `EloHead` modules

**Files:**
- Modify: `sahformer/model/heads.py` (append two classes)
- Test: `tests/test_bucket_head.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bucket_head.py
import torch
from sahformer.model.config import ModelConfig
from sahformer.model.heads import BucketTimeHead, EloHead
from sahformer.model.timebuckets import N_BUCKETS


def test_bucket_time_head_shape():
    cfg = ModelConfig()
    head = BucketTimeHead(cfg)
    pooled = torch.randn(4, cfg.dim_vit)
    out = head(pooled)
    assert out.shape == (4, N_BUCKETS)


def test_elo_head_shape():
    cfg = ModelConfig()
    head = EloHead(cfg)
    pooled = torch.randn(4, cfg.dim_vit)
    out = head(pooled)
    assert out.shape == (4, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/test_bucket_head.py -v`
Expected: FAIL with `ImportError: cannot import name 'BucketTimeHead'`

- [ ] **Step 3: Write minimal implementation** — append to `sahformer/model/heads.py`

```python
from sahformer.model.timebuckets import N_BUCKETS   # add near the top imports


class BucketTimeHead(nn.Module):
    """Think-time as a distribution over N_BUCKETS discrete buckets. Reads the pooled
    position summary ONLY (no policy-difficulty shortcut)."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(cfg.dim_vit),
            nn.Linear(cfg.dim_vit, cfg.head_hid_dim), nn.ReLU(),
            nn.Linear(cfg.head_hid_dim, N_BUCKETS),
        )

    def forward(self, pooled):
        return self.net(pooled)                    # (B, N_BUCKETS) logits


class EloHead(nn.Module):
    """Auxiliary rating predictor: forces the representation to encode strength
    (not just move confidence). Predicts normalised elo_self in [0,1]."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(cfg.dim_vit),
            nn.Linear(cfg.dim_vit, cfg.head_hid_dim), nn.ReLU(),
            nn.Linear(cfg.head_hid_dim, 1),
        )

    def forward(self, pooled):
        return self.net(pooled)                    # (B, 1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. python -m pytest tests/test_bucket_head.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add sahformer/model/heads.py tests/test_bucket_head.py
git commit -m "feat: BucketTimeHead + EloHead modules"
```

---

## Task 3: Config fields (`time_head`, `time_buckets`, `predict_elo`)

**Files:**
- Modify: `sahformer/model/config.py:15-21`

- [ ] **Step 1: Add fields** — insert after the existing `think_extra` line in `ModelConfig`

```python
    time_head: str = "mdn"        # "mdn" (legacy) or "bucket" (ChessMimic-style)
    time_buckets: int = 30        # only used when time_head == "bucket"
    predict_elo: bool = False     # add an auxiliary elo-prediction head + elo-gap conditioning
```

- [ ] **Step 2: Verify it imports and defaults preserve the old model**

Run: `PYTHONPATH=. python -c "from sahformer.model.config import ModelConfig; c=ModelConfig(); print(c.time_head, c.time_buckets, c.predict_elo)"`
Expected: `mdn 30 False`

- [ ] **Step 3: Commit**

```bash
git add sahformer/model/config.py
git commit -m "feat: config fields for bucket time-head + elo prediction (default off)"
```

---

## Task 4: Wire heads + elo conditioning into `ClockAwareChessformer`

**Files:**
- Modify: `sahformer/model/clockaware.py`
- Test: `tests/test_model_bucket_forward.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model_bucket_forward.py
import torch
from sahformer.model.config import ModelConfig
from sahformer.model.clockaware import ClockAwareChessformer
from sahformer.model.timebuckets import N_BUCKETS


def _batch(cfg, b=2):
    return {
        "board": torch.zeros(b, 8, 8, 12),
        "history": torch.zeros(b, 7, 8, 8, 12),
        "elo_self": torch.tensor([1500, 1800][:b]),
        "elo_opp": torch.tensor([1600, 1700][:b]),
        "temporal": torch.zeros(b, cfg.temporal_dim),
    }


def test_mdn_path_unchanged():
    cfg = ModelConfig()                       # time_head defaults to "mdn"
    m = ClockAwareChessformer(cfg).eval()
    out = m(_batch(cfg))
    assert "mdn" in out and "time_logits" not in out


def test_bucket_path_returns_time_logits_and_elo():
    cfg = ModelConfig(time_head="bucket", predict_elo=True)
    m = ClockAwareChessformer(cfg).eval()
    out = m(_batch(cfg))
    assert out["time_logits"].shape == (2, N_BUCKETS)
    assert out["elo_pred"].shape == (2, 1)
    assert "pooled" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/test_model_bucket_forward.py -v`
Expected: FAIL (`test_bucket_path...` KeyError `time_logits`)

- [ ] **Step 3: Edit `sahformer/model/clockaware.py`**

Replace the imports line and `__init__` head section and `forward`:

```python
# imports (add to existing line)
from sahformer.model.heads import (PolicyHead, ValueHead, ThinkTimeMDNHead,
                                    BucketTimeHead, EloHead, policy_difficulty)
```

In `__init__`, replace the `self.think = ThinkTimeMDNHead(cfg)` line with:

```python
        self.time_head = cfg.time_head
        if cfg.time_head == "bucket":
            self.time = BucketTimeHead(cfg)
        else:
            self.think = ThinkTimeMDNHead(cfg)
        self.predict_elo = cfg.predict_elo
        if cfg.predict_elo:
            self.elo = EloHead(cfg)
            self.elo_cond = nn.Linear(3, cfg.t_ctx)     # (elo_self, elo_opp, gap) -> context
            nn.init.zeros_(self.elo_cond.weight); nn.init.zeros_(self.elo_cond.bias)
```

Replace `forward` with:

```python
    def forward(self, batch: dict) -> dict:
        tok = self.input_emb(batch["board"].float(), batch["history"].float(),
                             batch["elo_self"], batch["elo_opp"])
        t = self.temporal_enc(batch["temporal"])
        if self.predict_elo:
            es = batch["elo_self"].float() / 3000.0
            eo = batch["elo_opp"].float() / 3000.0
            elo_feats = torch.stack([es, eo, es - eo], dim=-1)      # (B, 3)
            t = t + self.elo_cond(elo_feats)
        film = self.film_gen(t) if self.use_film else None
        enc = self.encoder(tok, t=(t if self.use_time_gab else None), film=film)
        pooled = enc.mean(dim=1) + self.t_to_d(t)
        move_logits = self.policy(enc)
        out = {"move_logits": move_logits, "value_logits": self.value(enc), "pooled": pooled}
        if self.time_head == "bucket":
            out["time_logits"] = self.time(pooled)
        else:
            think_in = torch.cat([pooled, policy_difficulty(move_logits)], dim=-1)
            out["mdn"] = self.think(think_in)
        if self.predict_elo:
            out["elo_pred"] = self.elo(pooled)
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. python -m pytest tests/test_model_bucket_forward.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full model test suite to confirm no regression**

Run: `PYTHONPATH=. python -m pytest tests/ -q`
Expected: PASS (existing tests still green — MDN path untouched)

- [ ] **Step 6: Commit**

```bash
git add sahformer/model/clockaware.py tests/test_model_bucket_forward.py
git commit -m "feat: bucket time-head + elo conditioning/head wired into ClockAwareChessformer (gated)"
```

---

## Task 5: Loss — bucket CE + elo, branched by output keys

**Files:**
- Modify: `sahformer/training/losses.py:19-28`
- Test: `tests/test_losses_bucket.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_losses_bucket.py
import torch
from sahformer.training.losses import compute_losses
from sahformer.model.timebuckets import N_BUCKETS


def _batch(b=3):
    return {
        "move_from": torch.zeros(b, dtype=torch.long),
        "move_to": torch.ones(b, dtype=torch.long),
        "promo": torch.zeros(b, dtype=torch.long),
        "result": torch.zeros(b, dtype=torch.long),
        "think_time": torch.tensor([0.3, 2.0, 15.0]),
        "elo_self": torch.tensor([1500, 1800, 2000]),
    }


def test_bucket_time_loss_and_elo():
    b = 3
    out = {
        "move_logits": torch.randn(b, 4352, requires_grad=True),
        "value_logits": torch.randn(b, 3, requires_grad=True),
        "time_logits": torch.randn(b, N_BUCKETS, requires_grad=True),
        "elo_pred": torch.randn(b, 1, requires_grad=True),
    }
    losses = compute_losses(out, _batch(b), w_elo=0.05)
    assert torch.isfinite(losses["total"])
    assert torch.isfinite(losses["time"])
    assert torch.isfinite(losses["elo"])
    losses["total"].backward()                      # gradients flow


def test_mdn_path_still_works():
    b = 3
    k = 3
    out = {
        "move_logits": torch.randn(b, 4352),
        "value_logits": torch.randn(b, 3),
        "mdn": (torch.randn(b, k), torch.randn(b, k), torch.randn(b, k)),
    }
    losses = compute_losses(out, _batch(b))
    assert torch.isfinite(losses["total"])
    assert losses["elo"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. python -m pytest tests/test_losses_bucket.py -v`
Expected: FAIL (`compute_losses` has no `w_elo` / doesn't handle `time_logits`)

- [ ] **Step 3: Replace `compute_losses` in `sahformer/training/losses.py`**

```python
from sahformer.model.timebuckets import to_bucket   # add near top imports


def compute_losses(out, batch, w_policy: float = 1.0, w_value: float = 0.1,
                   w_time: float = 0.2, w_elo: float = 0.05):
    target = move_target_index(batch["move_from"], batch["move_to"], batch["promo"])
    policy = F.cross_entropy(out["move_logits"], target)
    value = F.cross_entropy(out["value_logits"], batch["result"].long())

    if "time_logits" in out:                          # bucket head
        think = batch["think_time"].float()
        tb = torch.tensor([to_bucket(float(t)) for t in think],
                          device=out["time_logits"].device)
        time = F.cross_entropy(out["time_logits"], tb)
    else:                                             # legacy MDN head
        pi, mu, sigma = out["mdn"]
        time = mdn_nll(pi, mu, sigma, batch["think_time"].float())

    if "elo_pred" in out:
        elo_target = (batch["elo_self"].float() / 3000.0).unsqueeze(-1)
        elo = F.smooth_l1_loss(out["elo_pred"], elo_target)
    else:
        elo = torch.zeros((), device=out["move_logits"].device)

    total = w_policy * policy + w_value * value + w_time * time + w_elo * elo
    return {"policy": policy, "value": value, "time": time, "elo": elo, "total": total,
            "move_acc": move_accuracy(out["move_logits"], target)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. python -m pytest tests/test_losses_bucket.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add sahformer/training/losses.py tests/test_losses_bucket.py
git commit -m "feat: bucket cross-entropy time loss + auxiliary elo loss (branched by output keys)"
```

---

## Task 6: Thread `w_elo` through the training loop

**Files:**
- Modify: `sahformer/training/loop.py:19-34` (TrainConfig), `:129` (compute_losses call), `:142-148` (logging)

- [ ] **Step 1: Add `w_elo` to `TrainConfig`** — insert after the `w_time` field:

```python
    w_elo: float = 0.05
```

- [ ] **Step 2: Pass it into `compute_losses`** — change loop.py:129 from

```python
                losses = compute_losses(out, batch, cfg.w_policy, cfg.w_value, cfg.w_time)
```
to
```python
                losses = compute_losses(out, batch, cfg.w_policy, cfg.w_value, cfg.w_time, cfg.w_elo)
```

- [ ] **Step 3: Log the elo loss** — change the log line (loop.py:147-148) to include elo:

```python
            print(f"step {step+1}/{cfg.max_steps} total={total:.4f} "
                  f"policy={rec['policy']:.4f} time={rec['time']:.4f} "
                  f"elo={losses['elo'].item():.4f} acc={rec['move_acc']:.3f}")
```

- [ ] **Step 4: Verify the loop imports/constructs**

Run: `PYTHONPATH=. python -c "from sahformer.training.loop import TrainConfig; print(TrainConfig().w_elo)"`
Expected: `0.05`

- [ ] **Step 5: Commit**

```bash
git add sahformer/training/loop.py
git commit -m "feat: thread w_elo through TrainConfig + loop"
```

---

## Task 7: Play integration — bucket sampler in `self_play`

**Files:**
- Modify: `sahformer/play.py:34-105` (the `self_play` think-time step)

- [ ] **Step 1: Add a helper + branch in `self_play`.** Near the top of `play.py` add:

```python
from sahformer.model.timebuckets import sample_think_time as _sample_bucket
```

In `self_play`, where it currently computes `think = _sample_think_time(out["mdn"], rng, think_temp=think_temp)`, replace with:

```python
            if "time_logits" in out:
                think = _sample_bucket(out["time_logits"][0], remaining_clock=clock[mover], rng=rng)
            else:
                think = _sample_think_time(out["mdn"], rng, think_temp=think_temp)
```

- [ ] **Step 2: Write a smoke test**

```python
# tests/test_play_bucket.py
import chess
from sahformer.model.config import ModelConfig
from sahformer.model.clockaware import ClockAwareChessformer
from sahformer.play import self_play


def test_selfplay_bucket_spends_clock():
    cfg = ModelConfig(time_head="bucket", predict_elo=True)
    m = ClockAwareChessformer(cfg).eval()
    recs = list(self_play(m, max_plies=6, seed=1))
    assert len(recs) >= 1
    for r in recs:
        assert 0.0 <= r["think"] <= 180.0
        assert r["white_clock"] <= 180.0 and r["black_clock"] <= 180.0
```

- [ ] **Step 3: Run test to verify it passes**

Run: `PYTHONPATH=. python -m pytest tests/test_play_bucket.py -v`
Expected: PASS (untrained model, but the sampler must run + respect the clock)

- [ ] **Step 4: Commit**

```bash
git add sahformer/play.py tests/test_play_bucket.py
git commit -m "feat: self_play uses the bucket sampler when the model has a time-logits head"
```

---

## Task 8: UCI engines — bucket sampler

**Files:**
- Modify: `scripts/uci_engine.py:118-137` (the `pace` block), `scripts/clone_uci.py:116-124`

- [ ] **Step 1: `uci_engine.py`** — replace the pace/think block so it uses the bucket head when present:

```python
        if opts["pace"] > 0:
            if "time_logits" in out:
                from sahformer.model.timebuckets import sample_think_time
                think = sample_think_time(out["time_logits"][0], remaining_clock=my, rng=rng)
            else:
                think = _sample_think_time(out["mdn"], rng, think_temp=opts["think_temp"])
            sleep = min(think, 30.0, max(0.3, my * 0.4))
            time.sleep(sleep)
```

- [ ] **Step 2: `clone_uci.py`** — replace the `_sample_think_time` call in `choose` with:

```python
        if "time_logits" in o:
            from sahformer.model.timebuckets import sample_think_time
            think = sample_think_time(o["time_logits"][0], remaining_clock=my_clock, rng=self.rng)
        else:
            think = _sample_think_time(o["mdn"], self.rng, think_temp=1.0)
```

- [ ] **Step 3: Manual smoke — run the UCI loop against a fresh bucket model.** (No pytest; UCI is stdio.) Verify import path only:

Run: `PYTHONPATH=. python -c "import scripts.uci_engine, scripts.clone_uci; print('import ok')"`
Expected: `import ok`

- [ ] **Step 4: Commit**

```bash
git add scripts/uci_engine.py scripts/clone_uci.py
git commit -m "feat: UCI engines spend the clock via the bucket sampler when available"
```

---

## Task 9: Fine-tune — bucket time loss for clones

**Files:**
- Modify: `scripts/finetune_clone.py:114-119` (the loss) and the eval

- [ ] **Step 1: Branch the time loss in the training step.** Replace the MDN time-loss lines with:

```python
            out = model(batch)
            move = torch.tensor([r[5] for r in ch]).to(DEV)
            pol = F.cross_entropy(out["move_logits"], move)
            think = torch.tensor([r[7] for r in ch]).float().to(DEV)
            if "time_logits" in out:
                from sahformer.model.timebuckets import to_bucket
                tb = torch.tensor([to_bucket(float(t)) for t in think]).to(DEV)
                tl = F.cross_entropy(out["time_logits"], tb)
            else:
                pi, mu, sg = out["mdn"]; tl = mdn_nll(pi, mu, sg, think)
            loss = pol + args.w_time * tl
```

- [ ] **Step 2: Verify import + arg parse still work**

Run: `PYTHONPATH=. python scripts/finetune_clone.py --help`
Expected: usage text prints (no import error)

- [ ] **Step 3: Commit**

```bash
git add scripts/finetune_clone.py
git commit -m "feat: finetune_clone trains the bucket time-head when the base has one"
```

---

## Task 10: Local smoke run (de-risk before the GPU-day)

**Files:**
- Create: `scripts/smoke_bucket.py`

- [ ] **Step 1: Write the smoke script**

```python
# scripts/smoke_bucket.py
"""Tiny local sanity run for the bucket time-head + elo head: train a few hundred steps on
ONE shard, assert every loss decreases and a self-play game spends the clock without flagging.

    PYTHONPATH=. python scripts/smoke_bucket.py --shard data/shard0.npz
"""
import argparse, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from sahformer.model.config import ModelConfig
from sahformer.training.loop import TrainConfig, train
from sahformer.model.clockaware import ClockAwareChessformer
from sahformer.play import self_play


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="data/shard0.npz")
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    shards = glob.glob(args.shard)
    assert shards, f"no shard at {args.shard}"

    mc = ModelConfig(dim_vit=128, num_blocks=2, num_heads=4,
                     time_head="bucket", predict_elo=True)
    tc = TrainConfig(mode="full", max_steps=args.steps, batch_size=64, lr=3e-4,
                     warmup_steps=20, w_time=0.2, w_elo=0.05, grad_clip=1.0,
                     device="cuda" if torch.cuda.is_available() else "cpu",
                     out_dir="checkpoints/_smoke_bucket", log_every=50, ckpt_every=300)
    res = train(tc, shards, mc)
    h = res["history"]
    first = sum(r["time"] for r in h[:20]) / 20
    last = sum(r["time"] for r in h[-20:]) / 20
    print(f"time loss {first:.3f} -> {last:.3f}")
    assert last < first, "bucket time loss did not decrease"

    model = res["model"].eval().cpu()
    recs = list(self_play(model, max_plies=20, seed=0))
    flagged = any(r["flagged"] for r in recs)
    print(f"self-play plies={len(recs)} flagged={flagged}")
    assert not flagged, "model flagged itself in a 20-ply game"
    print("SMOKE OK")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the smoke script**

Run: `PYTHONPATH=. python scripts/smoke_bucket.py --shard data/shard0.npz`
Expected: prints `time loss X -> Y` (Y < X), `self-play ... flagged=False`, `SMOKE OK`

- [ ] **Step 3: If it fails,** diagnose with the systematic-debugging skill (do NOT tune blindly). Common causes: `w_elo` too high (elo loss dominates) → lower to 0.02; time loss flat → check `to_bucket` targets are not all one class on shard0.

- [ ] **Step 4: Commit**

```bash
git add scripts/smoke_bucket.py
git commit -m "test: local smoke run for bucket time-head + elo head"
```

---

## Task 10b: `scripts/train.py` CLI flags for the new config

**Files:**
- Modify: `scripts/train.py:18-48`

- [ ] **Step 1: Add the three arguments** — insert after the `--num-blocks` line (train.py:22):

```python
    ap.add_argument("--time-head", default="mdn", choices=["mdn", "bucket"])
    ap.add_argument("--predict-elo", action="store_true")
    ap.add_argument("--w-elo", type=float, default=0.05)
```

- [ ] **Step 2: Pass them into the configs** — replace the `model_cfg` / `cfg` construction (train.py:43-48):

```python
    model_cfg = ModelConfig(dim_vit=args.dim_vit, num_blocks=args.num_blocks,
                            time_head=args.time_head, predict_elo=args.predict_elo)
    cfg = TrainConfig(mode=args.mode, max_steps=args.max_steps, batch_size=args.batch_size,
                      lr=args.lr, weight_decay=args.weight_decay, grad_clip=args.grad_clip,
                      warmup_steps=args.warmup_steps, out_dir=args.out, amp=args.amp,
                      stream=args.stream, resume=args.resume, device=args.device,
                      log_every=args.log_every, ckpt_every=args.ckpt_every, w_elo=args.w_elo)
```

- [ ] **Step 3: Verify the flags parse and build a model**

Run: `PYTHONPATH=. python -c "import sys; sys.argv=['t','x','--time-head','bucket','--predict-elo']; import argparse" && PYTHONPATH=. python scripts/train.py --help | grep -E "time-head|predict-elo|w-elo"`
Expected: the three new flags appear in the help.

- [ ] **Step 4: Commit**

```bash
git add scripts/train.py
git commit -m "feat: train.py flags --time-head/--predict-elo/--w-elo"
```

---

## Task 11: Full training run (~30M base) on a rented 3090

**Not a code task — operational. The user starts/stops the Vast instance; the agent runs setup + launch + monitoring per HANDOFF logistics (held-open ssh; container kills orphaned procs).**

- [ ] **Step 1: Pull the corpus** on the instance from HF `slobaspeed/chesscom-balanced-shards` to `/workspace/shards/`.

- [ ] **Step 2: Burn-in (confirm steps/s + ETA before committing the full run).** Launch 500 steps at the target size and read throughput:

```bash
PYTHONPATH=. python -u scripts/train.py '/workspace/shards/*.npz' \
  --mode full --dim-vit 512 --num-blocks 12 --time-head bucket --predict-elo \
  --lr 4e-5 --weight-decay 1e-6 --grad-clip 3.5 --warmup-steps 1000 \
  --batch-size 512 --max-steps 500 --amp --stream --device cuda \
  --out /workspace/ckpt_smoke --ckpt-every 500 --log-every 50 2>&1 | tee burnin.log
```
Expected: all four losses print and fall; note s/step to compute the ETA. (Requires Task 10b's flags.)

- [ ] **Step 3: Full run** (~400k steps), resumable, rolling best/last:

```bash
PYTHONPATH=. python -u scripts/train.py '/workspace/shards/*.npz' \
  --mode full --dim-vit 512 --num-blocks 12 --time-head bucket --predict-elo \
  --lr 4e-5 --weight-decay 1e-6 --grad-clip 3.5 --warmup-steps 1000 \
  --batch-size 512 --max-steps 400000 --amp --stream --device cuda \
  --out /workspace/ckpt --ckpt-every 2000 --log-every 50 \
  --resume /workspace/ckpt/last.pt 2>&1 | tee -a train.log
```

- [ ] **Step 4: Pull `best.pt` back** to `checkpoints/base_bucket_best.pt` periodically (not just at the end) so an instance loss never costs more than one interval.

- [ ] **Step 5: Destroy the instance** in the Vast dashboard when done.

---

## Task 12: Acceptance evaluation

**Files:**
- Reuse the session's scratch tools, promoted into `scripts/`: create `scripts/eval_timehead.py` (adapt `scratchpad/head_bakeoff.py`'s distribution check to load a bucket-headed base directly and report snap/tail vs real for 3 held-out players).

- [ ] **Step 1: Time-head acceptance** — on latebloomer, OKENITE, couli held-out games, assert the **sampled** think-time matches real: snap-rate (<1s) within ±5pp, tail-rate (>10s) within ±2pp. (Criterion 2 in the spec.)

Run: `PYTHONPATH=. python scripts/eval_timehead.py --ckpt checkpoints/base_bucket_best.pt --players latebloomer,OKENITE,couli`
Expected: per-player snap/tail within tolerance.

- [ ] **Step 2: Move-match not worse** — run `scripts/maia3_phase.py`-style top-1 for the new base vs the old 19.5M base on the same held-out sets; assert new ≥ old, and compare to Maia-3-23M (~57%) as the external bar. (Criterion 3.)

- [ ] **Step 3: Elo dial** — held-out correlation of `elo_pred` vs true rating materially > 0 (the old confidence-knob was ~0); and setting the dial shifts opening choice. (Criterion 4.)

- [ ] **Step 4: Plays in a GUI** — load `base_bucket_best.pt` via `sahformer_engine.bat` in a chess GUI; confirm visible snap-and-tank clock behaviour. (Criterion 5.)

- [ ] **Step 5: Write a short results note** to `docs/superpowers/specs/` and update memory (timing conclusions + the trained base location). Commit.

```bash
git add scripts/eval_timehead.py docs/superpowers/
git commit -m "eval: bucket time-head acceptance (snap/tail match, move-match, elo dial)"
```

---

## Notes for the executor
- **No corpus rebuild** anywhere — think-time is bucketed at loss time, elos are already in shards, elo-gap is computed in-model.
- **Backward compatibility** is load-bearing: the old base (`base_300k_best.pt`) must keep loading (it uses `time_head="mdn"`). Never make `"bucket"` the config default.
- If a step's test is red, use `superpowers:systematic-debugging` — do not guess-patch.
- Keep the dev box safe: smoke configs are tiny; only the 3090 runs the full size.
