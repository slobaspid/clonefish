# clonefish — project method document

2026-09-12. The *method*, not the architecture. Written after four iterations of critique, and after
the n=7 fine-tune programme exposed that the earlier work measured the wrong things in the wrong order.
Companions: `2026-09-12-clonefish-architecture.md` (the what), `results/FINDINGS_2026-09-11.md` (numbers).

---

## A. Target spec — "plays like me" is three separable claims

Conflating these is the root cause of every overclaim in this project.

| # | claim | mechanism | status |
|---|---|---|---|
| **T1 Repertoire** | reproduces the openings I actually play | memorised position lookup | **SOLVED** (+3.6 pp, verified personal: a cohort book gets only +0.63) |
| **T2 Novel choice** | picks what I would pick in positions neither of us has seen | generalisation | **CONTESTED** — the whole scientific question |
| **T3 Whole-game feel** | my pace, my mistake pattern, how my games go wrong | traits + sampling | partial (clock shape matches; error pattern never tested) |

Every rung below is labelled with the claim it tests. A result that cannot be assigned to T1/T2/T3 is
not a result.

## B. The central question, and the two experiments that settle it

Our data says a full fine-tune buys **+3.03 pp in the middlegame** (n=7). The cross-domain survey says
**no field reliably clones decisions — only surfaces and low-dimensional traits.** Both cannot be
straightforwardly true. Three explanations, each implying a *different product*:

- **(a) Real T2 personalisation** → the identity-conditioned trunk is worth building.
- **(b) Book / repeat leakage past ply 12** → the gain is an artefact; ship base + book + clock.
- **(c) Trait effects surfacing as move choice** (the player is just faster / sharper / trade-happier)
  → build **4–6 dials**, skip the entire enrichment programme.

**Experiment B1 — separates (b). "Book-unseen split."**
Split held-out positions by whether the position appears in the player's own book (built from their
training games only). Re-measure the fine-tune gain separately on *seen* and *unseen* positions.
Cost: one forward pass over data already on disk. **If the gain on unseen positions is
indistinguishable from zero, the midgame finding is memorisation and (b) wins.**

**Experiment B2 — separates (a) from (c). "Trait-matched stranger cohort."**
Compute 4–6 identity-free traits per player (pace; accuracy / centipawn loss; trade rate; check-and-attack
rate; castling timing; opening entropy). For a target player, assemble a cohort of *other* players whose
trait vector matches theirs, fine-tune on **the cohort's** games, and evaluate on the target. We already
know *random* strangers give −1.4 to +1.0 pp. **If trait-matched strangers recover most of the +3 pp, the
effect is traits (c), not identity (a)** — and the cheap dial product is the correct one.
Cost: cohort selection plus one fine-tune per target. No architecture work.

B2 is the highest-value experiment in this document and it did not exist before this method was written.

## C. Measurement ladder — nothing is believed until the rung below passes

Two rules do most of the work:

1. **Everything paired within-player.** Report per-player deltas with CIs; never a cross-player mean.
   Between-player spread is 5 pp, so unpaired comparisons cannot see a 1 pp effect.
2. **Every rung is calibrated against a measured noise floor before it may gate anything.**

| rung | tests | statistic | pass bar | noise floor |
|---|---|---|---|---|
| **0 sanity** | — | held-out NLL vs base | paired improvement, CI excludes 0 | re-run noise **0.13 pp** (measured) |
| **1 identity** | T2 | self-ft vs **stranger**-ft on the *same* eval set | delta > 2x paired sd | paired sd (to measure, §D) |
| **2 non-memorisation** | T2 | gain on **book-unseen** positions | > 0 with CI excluding 0 | as rung 0 |
| **3 distribution** | T3 | MMD² and 1-NN vs **two** floors (real-vs-real, base) | both move toward real by > 2x base-vs-base noise | **base-vs-base, not yet measured** |
| **4 behaviour** | T3 | branched rollout: KS distance between clone and real per-move centipawn-loss distributions, plus the blunder-rate-vs-seconds-left curve | clone closer to real than base is, beyond the floor | base-vs-base branch control |
| **5 human** | T1+T2+T3 | pair test, pre-registered | judges <=55 % on clone pairs **while** >=70 % on the real-vs-base control | the control arm *is* the floor |

Rung 3 was run once (VEGETAL): MMD² base 0.0433 -> clone 0.0328; 1-NN 0.883 -> 0.825. Directionally a
pass, but **its noise floor was never measured**, so it is provisional.

## D. Power — measure first, then size

Known: run-to-run noise **0.13 pp**; between-player sd **1.77 pp**. The quantity that matters is the
**paired** sd (same player, method A vs B), which has never been measured. It is bounded below by about
0.18 pp (two runs of noise) and inflated by method x player interaction, which is likely non-trivial
because headroom varies (corr(base, gain) = −0.58).

**Plan:** a 3-player pilot running two methods each, compute paired sd, then size every later experiment
from it. If paired sd is about 0.3 pp, detecting a 1 pp method difference needs roughly 3–5 players; if
about 1 pp, it needs 10–15. **Choosing n before this pilot is guesswork and is forbidden by this document.**

## E. Hypothesis space — one factorial, most cells still empty

*where identity enters* x *parameter budget* x *data per player*

- **placement:** frozen residual · **trait dials** · trunk-conditioned · full fine-tune
- **budget:** 4–6 scalars · routing row over shared adapters · LoRA · full
- **data:** 50 · 500 · 5 000

Settled: the placement extremes — frozen residual **+0.001 pp** midgame versus full fine-tune
**+3.03 pp**. Open and product-critical: **budget at 5 k games** (can a routing row buy the full-FT gain?
the published claim is within 1 % of full FT at 1 % of the compute) and **the trait-dial cell** (B2).
Held fixed: base checkpoint, 5 000 games, 3 epochs, date-sorted honest split, newest 40 games as test.

## F. Programme, in order, with kill criteria stated in advance

1. **B1 book-unseen split.** *Kill:* if the unseen-position gain is about 0, the midgame result is
   memorisation — abandon the enrichment programme, ship base + book + clock, and correct the
   architecture doc.
2. **Paired-sd pilot** (§D). *Kill:* if paired sd > 1.5 pp, per-player experiments cannot resolve 1 pp
   effects at feasible n — switch to pooled multi-player training as the unit of analysis.
3. **B2 trait-matched cohort.** *Kill:* if trait-matched strangers recover more than 70 % of the gain,
   drop identity conditioning entirely and build 4–6 dials.
4. **Budget ablation at 5 k.** *Kill:* if a routing row matches full FT, never ship full FT (it
   specialises hard: −5.78 pp on a stranger).
5. **Rung 3 with its floor, then rung 4.** *Kill:* if gap closure is below noise, the move-match gain
   does not change behaviour and is cosmetic.
6. **Rung 5 once**, at the end, as the ship gate.

## G. Product spec — what may be promised

- Honest claim: **"noticeably more like you than a generic human model."** Never "indistinguishable" —
  1-NN is 0.825 against an ideal of 0.5.
- **Tell strong, predictable players they will gain little.** corr(base, gain) = −0.58; the strongest
  base in our sample gained +0.79 pp, the weakest +5.81 pp.
- Personalisation does **not** need recent games (a 3-month shift costs 0.48 pp), so there is no
  "upload your latest games" requirement.

## H. Risk register

| risk | why it is live | mitigation |
|---|---|---|
| midgame gain is leakage | book coverage bleeds past ply 12 | **B1**, rung 2 |
| recognizer metric is circular | it reads openings and clocks, and we hand the clone both | book-only arm; neutral temporal input; prefer rung 4 |
| per-player variance swamps effects | spread 5 pp versus effects of 1–3 pp | within-player pairing; §D pilot |
| we build apparatus, not a product | the ladder has six rungs | rungs 0–2 are cheap and reuse existing data; 4–5 only at decision points |
| garden of forking paths | the DPO overclaim, and four claims that died in one night | pass bars and kill criteria written **before** each run |

## I. Definition of done

The project is finished when: (1) we know which of (a)/(b)/(c) is true; (2) we have the *cheapest*
mechanism that delivers the verified effect; (3) the product states an honest promise; (4) rung 5 has
passed once with its controls.

### B2 design, refined 2026-09-12 (three confounds found before launching it)

**Arms** (target = VEGETAL, eval = her newest 40 games, post-opening own ply >= 12, n = 1472, all
book-unseen per B1):

| arm | what it is | status |
|---|---|---|
| (i) self | fine-tune on her own 5000 games | **+5.23 pp** (have) |
| (ii) random stranger | fine-tune on bigbadbo55 | +1.02 pp (have, but Elo-mismatched: 1609 vs 1815) |
| (iii) trait-MATCHED cohort | 5 players nearest her trait vector, Elo-matched, 1000 games each | to run |
| (iv) trait-ANTI-matched cohort | 5 players farthest on traits, same Elo band, same pooling | to run |

**Confound 1 - strength.** Matching on traits that include Elo, then comparing against arm (ii),
mixes trait distance with strength distance. Fix: restrict candidates to within 75 Elo of the target,
then rank by the NON-Elo traits only. Strength is held fixed; only traits vary.

**Confound 2 - diversity (this is why arm (iv) is mandatory).** A 5-player pool differs from one
player in two ways at once: trait similarity AND data diversity. Pooling alone changes how much the
fine-tune overfits. Comparing (iii) against (ii) therefore cannot isolate traits. The decisive
contrast is **(iii) versus (iv)**: identical pool size, identical pooling structure, identical Elo
band, opposite trait distance. Without (iv), B2 proves nothing.

**Confound 3 - transferability ceiling.** A cohort cannot carry the target repertoire. That is fine:
B1 showed every post-opening test position is book-unseen, so repertoire is not what is being measured.

**Read-out.** If (iii) recovers most of the gap between (iv) and (i), the effect is TRAITS (c) and the
product is 4 to 6 dials. If (iii) is indistinguishable from (iv), traits do not carry it and the
effect is identity (a), justifying trunk conditioning. Anything in between is a mixture and should be
reported as a fraction, not a verdict.

**Traits used** (all identity-free, all computable from PGN plus clocks, no engine needed): median
think time (pace), snap rate below 0.5 s, tank rate above 10 s, capture rate, check rate, fraction
castling by own move 10, entropy of the first own move, mean own-ply count. Computed on TRAIN games
only for the target, so no test leakage.

### D-RESOLVED 2026-09-12 - the pilot was unnecessary; paired sd is already on disk

The six clean same-eval-set contrasts (self-ft minus stranger-ft, per player) ARE a paired contrast:
+3.00, +4.08, +4.35, +4.57, +5.51, +6.50 -> **paired sd = 1.21 pp** (n=6).

Sizing (alpha 0.05, power 0.8, paired):

| effect to detect | players needed |
|---|---|
| 0.5 pp | ~46 |
| 1.0 pp | ~11 |
| 2.0 pp | ~3 |
| 3.0 pp | ~1 |

**Consequence, and it is retroactive:** the n=7 programme could never have resolved a 1 pp difference
between methods. Every small delta quoted from it (the +0.48 pp drift cost, the 1.36 pp spread between
two gap-0 players) is inside noise and must not be interpreted. Only effects of about 2 pp and larger
were ever detectable at n=7.

**Caveat in our favour:** 1.21 pp is an UPPER bound for comparing two *personalisation* methods,
because per-player headroom partly cancels when both arms personalise. The self-vs-stranger contrast
includes the full headroom spread; a routing-row-vs-full-FT contrast should be tighter. Measure it on
the first three players of any method comparison rather than assuming.

**No separate pilot is required. §D is closed.**
