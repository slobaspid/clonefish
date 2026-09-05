# Clone DPO — judge → localizer → preference update (spec)

**Date:** 2026-08-21
**Status:** design, not started
**Context:** this is NEXT-2 from `HANDOFF_2026-08-20.md` §2, made buildable. The pooled move-residual clone
works (~+4pp per-move, clean). Per-move imitation cannot see *distributional* style (opening variety, tempo,
where-you-blunder). This spec closes that gap by nudging the per-player code with a **DPO preference update**,
where the preferences are localized to the exact positions where the clone sticks out.

---

## 0. One sentence

Measure where the clone's game-distribution drifts from the real player (**judge**), turn that global gap
into a list of specific positions + a better-move contrast at each (**localizer**), then move the per-player
256-D code toward the human side of those contrasts with a KL-leashed **DPO** step — head frozen, base frozen,
only the code moves.

## 1. The claim this is built to test

> The distributional gap the judge sees is real and closeable **without** wrecking per-move imitation — i.e.
> a preference update on localized tell-points improves the judge's distribution score while per-move top-1 /
> log-lik stays flat or better. If the judge score improves but imitation collapses, the update learned a
> caricature and the spec has failed its own guardrail.

Honest null result is a fine outcome: if the judge finds **no** meaningful gap (§3 report card), we stop —
NEXT-2 was cheap and the clone was already good enough distributionally.

## 2. What moves and what stays frozen (non-negotiable)

| component | role | frozen? |
|---|---|---|
| base (`checkpoints/base_300k_best.pt`) | position feature `pooled` + `move_logits` | **frozen** |
| shared move-residual head (`PooledResidual`, `clone_sweep.py`) | maps `[pooled ; code] → move-residual` | **frozen** |
| **per-player code** (256–1024-D embedding) | the *only* trainable thing | **TRAINS** |
| recognizer / `GameEncoder` (`run_scale.py`) | the judge's measuring tape | frozen, never trained here |

Nobody hand-edits the code's numbers. The dims are an entangled style code, **not** labeled knobs. The judge's
discrepancy becomes a scalar loss; gradient descent decides which combination of the 256+ numbers to move.

## 3. Stage 1 — the JUDGE (build this FIRST, alone)

Pure measurement, no learning. Two read-outs:

1. **Recognizer clouds.** Embed the real player's games and a library of clone-generated games with the frozen
   `GameEncoder`. Compare the two point-clouds in that space:
   - **MMD** (two-sample distance) — one number, "how far apart are the piles."
   - **discriminator AUC** — train a throwaway real-vs-clone classifier; ≈0.5 = indistinguishable = good.
2. **Interpretable stats dashboard** (the trustworthy one — human-auditable, no black box): opening frequency,
   think-time shape, blunder rate + *where* it happens, move-time-vs-eval. Real vs clone, side by side.
   Partial scaffold already exists: `scripts/clone_compare.py`.

**Deliverable:** `scripts/clone_judge.py` producing (a) MMD + AUC and (b) the stats table, real vs clone, for a
player. **STOP HERE and look at the output before building anything downstream.** If the gap is small, done.

## 4. Stage 2 — the LOCALIZER (the piece that makes DPO possible)

DPO's unit is a **single decision point**, not a whole game — you cannot hand it "these games feel off." The
localizer is the missing middle: it turns the judge's *global* verdict into a list of

> `(position s, preferred move a⁺, dispreferred move a⁻)`

triples. **Decision locked: simple-honest + offline.**

- **Pairs come from the real games.** At each position the player actually faced, their real move = `a⁺` (a
  true human anchor). Sample / read the clone's distribution at that same position; its over-produced move
  where it disagrees = `a⁻`. This automatically lands on divergence points and needs no generator loop.
- **The judge selects which pairs to keep.** If we kept *every* disagreement we'd just re-derive behavior
  cloning (which imitation already does — adds ~nothing). So filter/weight pairs to the position-categories
  the stats flagged (e.g. "sharp middlegames," "time-pressure"). **Real moves make the pairs; the judge says
  which pairs matter.** That is what makes DPO fix a *distributional* tell instead of average play.
- **Auditable by construction.** The output is a concrete list of positions you can eyeball ("clone goes quiet
  in every attacking position where the real guy storms"). No mystery gradients.

**Deliverable:** `scripts/clone_localize.py` → a pair file (positions + a⁺/a⁻ + which off-stat tagged it).

*Deferred second source (not v1):* recognizer attribution (which ply shifts a game's embedding toward
"clone") as an automatic tell-finder. Skip for now — simple-honest first.

## 5. Stage 3 — the DPO update

Standard DPO, per-move, over the localized pairs. Clone move-prob is
`softmax(base_logits + residual(pooled, code))` over legal moves — **differentiable in `code`**, so there is
**no sampling-gradient problem**: DPO contrasts likelihoods of *given* moves, never sampled ones. That is
exactly why DPO fits here where a naive "reward the good game" would fight the discrete dice-roll.

For each pair `(s, a⁺, a⁻)`:

```
Δ⁺ = log π_code(a⁺|s) − log π_ref(a⁺|s)
Δ⁻ = log π_code(a⁻|s) − log π_ref(a⁻|s)
loss = − log σ( β · (Δ⁺ − Δ⁻) )
```

- `π_ref` = the **pre-DPO clone** (base + frozen head + imitation-fitted code). Standard DPO reference; its
  KL leash (via `β`) is the thing stopping drift.
- Only `code` gets gradient (head + base frozen).
- **Two leashes, both required (the anti-caricature guardrail):**
  1. DPO's built-in KL-to-reference (`β`) — bounds how far the code moves from the imitation-trained start.
  2. `a⁺` is always the *real human move*, so the update can only pull *toward* the person, never toward a
     judge-gaming exaggeration.
  3. (Optional belt-and-suspenders) add a small imitation-loss term `λ·NLL(real move)` — the "referee" from
     §2 of the handoff. Turn on only if per-move imitation degrades.

**Deliverable:** `scripts/clone_dpo.py` — loads pairs, fits `code`, re-runs the Stage-1 judge before/after.

## 6. Success / failure metrics (report all three, every run)

| metric | source | want |
|---|---|---|
| MMD ↓ / discriminator AUC → 0.5 | Stage-1 judge | distribution gap closes |
| stats table matches real | Stage-1 dashboard | opening/tempo/blunder shapes line up |
| per-move top-1 & log-lik | `clone_sweep.py` eval | **flat or better** (guardrail — if this drops, caricature) |

The guardrail row is the whole point: judge-score up **and** imitation flat = real win. Judge-score up but
imitation down = failure, roll back.

## 7. Data / reuse (no re-caching)

- Players: `data/lichess_scale/` (base-novel → clean move eval). Same clean split as `clone_sweep.py`
  (train = all-but-last-40 games, eval = held-out last 40).
- Base features: reuse `sweep_cache_{N}[_full].pt` where possible; the clone is the pooled recipe from
  `clone_sweep.py` §1.2 (emb 512, hidden 512, wd 2e-2, dropout 0.6, res_l2 0.05).
- Judge encoder: `GameEncoder` from `run_scale.py` (CosFace-trained recognizer).

## 8. Build order (cheap-first, no overengineering)

1. **Stage 1 only** — `clone_judge.py`. Look at the numbers. **Gate: is there a gap worth closing?**
2. If yes → **Stage 2** `clone_localize.py`. Eyeball the tell-points; sanity-check they match the off-stats.
3. → **Stage 3** `clone_dpo.py`, offline pairs, one pass. Re-judge. Read the guardrail row.
4. Only if step 3 underdelivers: consider iterative pairs (regen clone samples per round) or the
   recognizer-attribution localizer. Not before.

## 9. Open questions to resolve at build time

- Game-level vs set-level recognizer embedding for the clouds (single game is noisy; a set of games is the
  fingerprint) — probably embed **sets**, matching how identification actually works.
- Circularity caveat (handoff §2): clone and recognizer share base DNA, so overlap partly rewards shared
  features. Using **real-X** as the target (not "fool the classifier") mutes it; keep the interpretable stats
  as the tie-breaker the human trusts.
- `β` and pair-count sweep — small, this is a 256-D fit, not a full model.
