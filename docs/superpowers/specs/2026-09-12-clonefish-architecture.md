# The most plausible architecture for clonefish

2026-09-12. Supersedes the v3 *layer stack* for anything beyond the opening. Written after the
v2 design review, this session's re-verification of all raw results, and primary-source reading of
Maia-individual, Maia4All, MHR/LoRA-bank and the voice-cloning literature.

Evidence levels: **[M]** measured by us · **[P]** published, primary source read · **[H]** hypothesis.

---

## 1. The one-paragraph answer

Our current design bolts a per-player residual onto a **frozen** base that was never trained to use
identity. That is why it saturates: **[M]** post-ply-10 our whole stack is worth **+0.55 pp** and the
midgame is **+0.001 pp**, and **[M]** giving the shared head 120 players' worth of coverage buys an
in-head player *nothing* (+2.55 vs +2.44 for strangers). The frontier does the opposite: it makes
identity a **first-class conditioning signal inside the trunk** and trains the base itself to use it.
Maia4All **[P]** gets **+2.5 pp post-ply-10** that way, and states plainly that fine-tuning directly
on a low-resource player *doesn't work* — the enrichment stage is what makes it work.

**But the honest top-line answer changed on 2026-09-12 (see §4), and this is the part to read.**
A cross-domain survey found **no field reliably clones an individual's decisions** — only output
surfaces and a few low-dimensional trait knobs — and our +0.55 pp has an **external replication**:
the same machinery is worth +5.28 pp over a *weak* baseline but **+0.70 pp over a strong one**
(SynthesizeMe, ACL 2025). We are at the field's ceiling, not below our own method's.

**So the most plausible architecture is, in priority order:**
1. **Ship what is verified and is most of the product:** the best base we can train, the personal
   count-based **opening book** (+3.6 pp, verified personal), the **clock model** (ours alone), and
   **sampling rather than argmax**. This is not a consolation prize — the book is the same mechanism
   that carries "personalization" in recommenders.
2. **The identity-conditioned trunk (§3) is now a JUSTIFIED bet, not a speculative one — the gate passed
   at n=7.** Full fine-tunes at 5,000 games give **+3.41 pp mean post-opening** (range +0.79…+5.81) and
   **+3.03 pp mean in the middlegame**, against **+0.001 pp** for residuals on a frozen base. The
   properly controlled identity effect (self-ft minus stranger-ft on the same eval set, n=6) is
   **+4.67 pp**, above Maia-individual's +2.8 pp at *twice* the data. So the earlier "~+1–2.5 pp"
   estimate was too pessimistic, and "the middlegame is unpersonalizable" was a fact about our
   **method**, not about chess.
3. **Expect headroom-shaped returns, and say so to users.** corr(base strength, gain) = **−0.582**:
   the strongest base gained +0.79 pp, the weakest +5.81 pp. Cloning pays most for weak/idiosyncratic
   players — the same population where the opening book pays most.
4. **If the bet is built, prefer TRAIT-shaped conditioning over a high-dimensional embedding** (§4b),
   and buy the gain with a **routing row over shared adapters** rather than private dense weights — a
   private fine-tune specialises hard (it cost a stranger −5.78 pp).

**The one thing still owed before acting: the midgame result is on probation.** It contradicts the
cross-domain finding that no field clones *decisions*, so per this document's own rule it must survive
independent metrics (1-NN, MMD) before being banked. **[M] That check has now RUN — WEAK PASS (VEGETAL, n=1).** Recognizer-space metrics, never part of the
training objective: MMD² **base 0.0433 → clone 0.0328** (real-vs-real floor −0.0003, so **24 % of the
gap closed**); 1-NN same-class **base 0.883 → clone 0.825** (ideal 0.5, **15 % closed**). Both arms are
self-play so drift cancels in the contrast. **So the midgame finding survives an independent metric and
is no longer merely a move-match artefact.**
**But the pass is weak and bounds the claim:** 1-NN 0.825 vs 0.5 means the clone is still easily
separable from the human — personalization *moves* the distribution without closing it. n=1, one seed.
A second player is queued. **Read this as: the §3 bet is justified and its ceiling is modest** — do not
promise "indistinguishable from you"; promise "noticeably more like you than a generic human model".
§3 below is the design; §5 is the order to build it in.

## 2. Why our current architecture cannot get there (the diagnosis)

| | our design | what the frontier does |
|---|---|---|
| where identity enters | a residual added to `pooled` **after** a frozen trunk | an **embedding that modulates the trunk**, trained jointly |
| who learns to use identity | only the small head | the **base weights themselves** (enrichment) |
| new-player init | random / zero | **nearest prototype** via a discriminative net |
| result post-ply-10 | **+0.55 pp [M]** | **+2.5 pp [P]** |

**[M]** The decisive internal evidence: our `pooled` residual *does* change midgame predictions for
17/17 players — and the changes **cancel to +0.001 pp**. The signal has nowhere useful to act.
**[M]** And the INHEAD control rules out the obvious alternative explanation (that the head just
needs more players): head membership is worth nothing. This is a *representational* ceiling, not a
coverage one.

## 3. The architecture

### Stage A — identity-conditioned enrichment (one-time, GPU). **The missing piece.**
Add a per-player embedding `z` (d = 128–256) and inject it where conditioning already works in our
model, **not** as an output residual. Our backbone already has two proven modulation paths:
1. the **skill embedding** concatenated per square at the input (`InputEmbedding`), and
2. **FiLM per block** + time-conditioned GAB (`FiLMGenerator`, `clockaware.py`).

Inject `z` through both: concatenate alongside the Elo embedding, and add `z` to the FiLM context so
it modulates every block. Initialise the `z`-path at zero so an untrained clone *is* the base.
Train jointly over many players with rich histories, with per-player `z` as free parameters, Elo
conditioning retained, and **embedding dropout** (randomly zero `z`) so the model stays a good
population model and `z` carries only deviations.
*Why this and not a bigger residual:* **[P]** Maia-2/Maia4All modulate prediction through embeddings
inside a unified parameter space; Maia4All's whole contribution is extending those population
embeddings to individual ones and moving the base weights to individual-level modelling.

### Stage B — prototype matcher (one-time, GPU). **Reuse what we already own.**
Train/reuse a discriminative net mapping a player's games → nearest prototype player, and initialise
a new user's `z` from that prototype. **[M]** We already have this model: GE2E/CosFace, 0.99 top-1
identification from 10 games, validated toward 2,844 players. This is the single biggest asset we
have been leaving on the table.
**Critical design caution [P] — CONFIRMED and refined.** Use the recognizer to *retrieve, initialise
and sanity-check*, **never** as the conditioning vector, and **never** as a loss on the generator's
output. Cai et al. (2005.04587, Interspeech 2020) confirm both halves: the x-vector with the *best*
verification EER produced *worse* TTS audio, and GE2E embeddings leak non-identity factors (emotion).
They also report the converse failure — purely *jointly-trained* embeddings generalise badly to unseen
speakers because synthesis corpora have too few speakers, **which is an independent second explanation
for our INHEAD null (we have ~3k players, not enough)**. The resolution is the hybrid: pretrained
**plus** learnable (Chien et al., 2103.04088). And ID-Booth (2504.07392) documents that identity
objectives on the generator overfit identity while **collapsing diversity** — fatal for a clone that
must reproduce a human's variety. So: fit the conditioning embedding under the **generative** loss
(move + time likelihood), recognizer frozen and out of the loop at sampling time.

### Stage C — per user (your ~5k-game regime)
1. **Match → initialise `z`** from the nearest prototype (instant, no gradients).
2. **Fit `z`** on their own positions — a few hundred parameters, minutes on CPU, partial-pooled
   toward the prototype by `n/(n+k)`.
3. **Per-user weight update — prefer a routing row over a private adapter.**
   **[P]** MHR/LoRA-bank is the right geometry: the LoRA adapters are **shared/population**
   parameters (chess config: an **inventory of 32 adapters, 8-head routing**, rank 16, on 24 MHR
   layers), and an individual is nothing but **their row of the routing matrix** — a softmax mixture
   over shared "latent skills". Per-user cost is therefore **thousands of weights at most, not tens
   of thousands** — which is exactly the geometry our −6.2 pp v1 adapter violated.
   **[P]** Fitting is *routing-only* fine-tuning and **"100 games is sufficient to learn the style
   vector of an unseen player"** — your ~5,000-game regime has ~50× that.
   **[P]** It lands "within 1% accuracy of individual model fine-tuning … at roughly 1% of the
   compute", so a full fine-tune is a *cost* choice, not an accuracy one.
   **[P]** A full private fine-tune remains the fallback, and Maia-individual only starts winning at
   ≥5,000 games (it *loses* 3 pp at 1k) — your regime is exactly that threshold.
   Whichever is used, anchor it: L2 to base weights + KL prior-preservation, λ **fitted** on a tune
   split rather than a hand-set schedule.
4. **Opening book** — count-based, Dirichlet-blended, α fitted. **[M]** +3.61 pp (n=20), and its
   gain is genuinely personal (cohort book only +0.63 pp). Note the literature *discards plies 0–9*,
   so this is gain we get **on top of** their reported numbers, not overlapping it.
5. **Clock** — sample the time head (never its mean), two partial-pooled scalars, coupled to the move
   (long think ⇒ harder position ⇒ flatter move distribution). **[P]** Nobody models an individual's
   think-time; this stays our only true differentiator.

## 4. Honest expected value — REVISED DOWN after the cross-domain dive

**[P] No domain reliably clones an individual's *decisions*.** What clones reliably is (a) the output
*surface* (timbre, letterforms, tone — for us, the opening book) and (b) a handful of **low-dimensional
trait knobs** (how fast / risky / aggressive someone is — for us, the clock and the Elo dial).

**[P] Our +0.55 pp has been independently replicated in another field.** SynthesizeMe (ACL 2025,
2506.05598) is worth **+5.28 pp over a weak LLM-judge baseline (56.69→61.97) but only +0.70 pp over a
strong fine-tuned reward model (71.48→72.18)**, and +0.53 pp on PRISM. Same machinery, same collapse
against a strong baseline. **We are at the field's ceiling, not below our own method's.**

**[P] And the optimistic numbers in this literature do not survive audit.** Recommenders: HRNN claimed
up to +28 % recall from per-user history, but independent re-benchmarking found plain session-based
methods beat session-aware personalized ones "even though they do not consider the available long-term
preference information". That is precisely what happened to our own distribution-DPO result.

**The detect/generate asymmetry is therefore expected, not a paradox.** Identification aggregates a
weak per-decision signal over hundreds of decisions; generation must win *one decision at a time*.
**[M]** 0.99 top-1 ID from 10 games is fully consistent with a per-move edge well under 1 pp.
**Never again read identifiability as generative headroom.**

**Revised target.** Stage A+B+C is now a **~+1–2.5 pp post-opening bet, not +2–3**, and the only
contrary data point is Maia4All's +2.5 pp — whose baseline should be treated with the same suspicion
the audits above earned. The reliable value remains the **+3.6 pp opening book** (which is *not* a
cheap trick — it is the same mechanism that carries "personalization" in recsys), the clock, and
**base quality**. **[M]** Our own full fine-tune is consistent and unstable: +6.3 pp (latebloomer,
9.2k) but +1.1 pp (dtchess, 15.2k, rated 2794).

**[M] 2026-09-12 — the bet is now measured, once.** A full fine-tune on **exactly 5,000 games**
(latebloomer) buys **+1.70 pp post-opening** on her own held-out games (early-mid +3.6, mid +1.3,
end −0.3) — inside the ~+1–2.5 pp band above, and below Maia-individual's +2.8 pp at 10k, which is the
right direction for half the data. Crucially the **cross-player control came back −5.78 pp**, so this
is identity, not domain adaptation (§5 step 0). Two consequences: the post-opening prize is **real but
small**, and a *private full fine-tune specialises hard* — which is exactly why Stage C should buy that
gain with a routing row over shared adapters rather than a private dense update.

**[M] 2026-09-12, replication 1 — bigger than expected, and a TEMPORAL-DRIFT confound surfaced.**
VEGETAL at 5,000 games: **54.55 → 59.65 = +5.10 pp** post-opening (middlegame **+5.5**), cross-control
on bigbadbo55 **−1.36 pp**, and both are Lichess so there is no domain confound whatsoever. Leakage was
checked and excluded (0 duplicate game URLs, 0 duplicate keys, train/test overlap 0). The 3.4 pp spread
vs latebloomer tracks **drift**: VEGETAL's train→test gap is **0 days** with a flat rating
(1820→1828), while latebloomer's train slice spans **7+ years**, she gains **+130 Elo** across it, and
her test sits **189 days** later — i.e. that fine-tune partly clones a younger, weaker player.
**VEGETAL's condition is the deployment-relevant one** (a user uploads games and plays now).
So the honest post-opening range is currently **+1.7 to +5.1 pp at 5k games (n=2)**, and the §4 estimate
above may be too pessimistic — but do not revise it upward until the three queued replications
(all gap ≈ 0) report. Prediction: they land near +5 if drift is the driver, near +1.7 if VEGETAL is
just unusually clonable.

**[M] RESOLVED — the drift hypothesis above is REFUTED; the spread is per-player variance.**
n=5 own-player post-opening gains: +1.70 (latebloomer) and **+5.10 / +3.74 / +3.09 / +3.68** for the
four gap-0 players (mean **+3.90**). The decisive test was *within* one player: VEGETAL re-fine-tuned
with training shifted ~960 games (~3 months) earlier, same game count, same test set — **+4.62 pp vs
+5.10 pp, i.e. a 3-month shift costs 0.48 pp**, far too little to explain a 2.2 pp spread. So recency
of the training window barely matters, and latebloomer is simply a harder player to clone.
**Two consequences:** (1) personalization does **not** require the user's most recent games — a
deployment constraint we can drop; (2) expect **per-player variance of ±1 pp** around a post-opening
mean near **+3.9 pp** at 5k games, and quote the range, never one player.
**Properly controlled identity effect** (same eval set, self-fine-tune minus stranger-fine-tune):
**+4.08 pp** (VEGETAL) and **+4.45 pp** (bigbadbo55) — *above* Maia-individual's +2.8 pp at 10k games.
**Midgame across n=5: +1.3 / +5.5 / +2.5 / +2.7 / +4.4 (mean ≈ +3.3) vs +0.001 for residuals on a frozen
base** — the empirical case for conditioning identity inside the trunk, still on probation until
1-NN/MMD move.

**[M] FINAL, n=7 — all runs complete. Supersedes the n=5 figures above.**
Own-player post-opening gain vs base: **mean +3.41 pp, range +0.79 … +5.81, sd 1.77** (MALOUMNJAK +0.79,
latebloomer +1.69, bigbadbo55 +3.09, OKENITE +3.67, Yespapa +3.74, VEGETAL +5.10, kpowe52 +5.81).
Midgame: **mean +3.03 pp, range +0.4 … +5.5** — against **+0.001 pp** for residuals on a frozen base.
Clean identity contrast (self-ft − stranger-ft, same eval set, n=6): **mean +4.67 pp, sd 1.21**.
**The "±1 pp per-player variance" claim above is WITHDRAWN — the real spread is 5 pp wide.**
**And the spread is explained: corr(base strength, gain) = −0.582.** Personalization is largely
**headroom** — it buys back what a player's own unpredictability costs the population model. The
strongest base gains least (+0.79), the weakest gains most (+5.81).
**Two product consequences:** (1) quote **+3.4 pp mean post-opening at 5k games**, never a single
player; (2) **tell strong, predictable players to expect little from cloning** — the value is
concentrated in weaker/idiosyncratic users, which is also where the opening book pays most.

### 4b. If we take one more shot at the middlegame, make it TRAIT-shaped, not taste-shaped
The only parameterizations anyone has shown to transfer to genuinely novel situations are
**low-dimensional trait parameters**: EIDT individuality transfer (eLife 107163) generalizes to
unseen conditions but what transfers is a few scalars (learning rate, reward sensitivity,
speed–accuracy); personalized driving (2308.07439) gets 8.5→43.5 % RMSE from 30 minutes but clones a
*habitual speed/headway setpoint*; mixed-logit conditions a couple of taste weights shrunk toward the
population. **High-dimensional per-situation preference transfers nowhere.**
So: prefer **a handful of interpretable dials** (risk, trade-happiness, king-safety tolerance,
speed–accuracy) fit per player and conditioned on — over a 128–256-d embedding. This is cheaper than
Stage A, it is the shape the evidence supports, and it composes with the clock model we already have.

## 5. Build order, cost, and what kills each step

| step | cost | falsification test (run BEFORE building the next) |
|---|---|---|
| **0. FT@5k + cross-player control** — **PASSED [M] 2026-09-12** | hours, 1 GPU | Fine-tune on latebloomer's 5,000 games: her own held-out post-opening **52.75 → 54.44 (+1.70 pp)**; by phase early-mid +3.6, mid +1.3, end −0.3. **Cross-player control: the same fine-tune evaluated on chess711 goes 57.09 → 51.32 (−5.78 pp).** Both are Lichess players and the base is chess.com-trained, so domain adaptation would have *helped* chess711; instead it hurt badly. **The gain is identity-specific, not domain.** The premise of §3 survives. **Now replicated to n=7: own-player post-opening mean +3.41 pp (range +0.79…+5.81, sd 1.77), midgame mean +3.03 pp, and a clean self-vs-stranger identity contrast of +4.67 pp (n=6). Cross-controls: −5.78, −3.56, −1.84, −1.36, −0.83, −0.69, +1.02 — mostly protective but NOT uniformly, so one of them is partly domain adaptation.** Caveat: a full fine-tune is evidently *specialising* — the −5.78 pp is the cost of a large private weight change, which is the argument for the routing-row geometry in Stage C. |
| **1. Stage A enrichment** | ~1 GPU-day | Do in-head players now beat strangers? Our INHEAD control is the exact harness, and currently says **no** for a frozen base. If enrichment doesn't flip that, `z` still has nowhere to act. |
| **2. Stage B matcher** | hours (model exists) | Does prototype-init beat zero-init at 50/500/5,000 games? |
| **3. Stage C LoRA** | minutes/user | Must beat `z`-only at 5k games on **post-opening NLL**, with λ fitted, or it is not worth shipping. |

## 6. What this replaces
Drops from v3: the 3,000-player *shared-head* retrain (**[M]** INHEAD killed its justification — but
note the same GPU budget now goes to Stage A, which is a different and better-evidenced thing), the
cohort book tier (**[M]** +0.05 pp), and E6's anchored-LoRA-as-first-bet (it is now step 3, *after*
enrichment, which is the part Maia4All says is load-bearing).

## 7. Open, honestly
- Whether input-concat vs FiLM injection of `z` matters **[H]** — cheap ablation in Stage A.
- ~~MHR's exact per-player parameter count~~ **RESOLVED [P]**: shared inventory of 32 rank-16
  adapters + 8-head routing over 24 layers; the player *is* one routing row (a mixture over shared
  skills), fitted routing-only, and 100 games suffices for an unseen player. This is the recommended
  Stage C mechanism and it makes the 50-game user safe as well as the 5,000-game one.
- Whether our clock head should also be `z`-conditioned **[H]** — likely yes, and no one has done it.
- ~~the cross-domain question~~ **RESOLVED [P] — and it is the discouraging answer.** No domain
  reliably clones decisions; only surfaces and low-dimensional trait knobs. Our number has an external
  replication (SynthesizeMe: +5.28 pp over a weak baseline → **+0.70 pp over a strong one**). Practical
  consequence: **stop funding middlegame personalization**, invest in base + book + clock, and treat
  any future middlegame win above ~+1 pp as presumed baseline-weakness or book/repeat leakage until
  independent metrics (1-NN, MMD) move. See §4 and §4b.
- Still genuinely open: the in-context/amortized and PEFT dives never completed (rate limit). Given
  §4, they are now lower priority than the step-0 gate.


---

# FINAL RECOMMENDATION (2026-09-12, evidence-backed)

Supersedes every earlier recommendation in this document. Each line cites the experiment behind it.

## What to build

1. **Population base + personal opening book + clock head, sampled never argmaxed.** The book is
   +3.61 pp and verified personal (cohort book only +0.63, n=20). Sampling not E[t] is the whole
   timing win. This is most of the product.
2. **A rank-16 bottleneck adapter fitted on the user's own games** - NOT a full fine-tune.
   `arch_bench`: adapter recovers **83-84 %** of full fine-tuning on both players with signal
   (VEGETAL +3.94 vs +4.76; kpowe52 +2.47 vs +2.94). A full private fine-tune specialises hard
   (-5.78 pp on a stranger) and costs far more.
3. **Ask for ~1,200 games, not 5,000.** Same player, same test set: **+4.76 pp at 1,160 games vs
   +5.23 pp at 5,000** - about 91 % of the gain at a quarter of the data. (Maia-individual *loses*
   3 pp at this data volume; our base is far more data-efficient.)
4. **Cold start via trait-matched cohort initialisation.** A cohort of same-Elo, trait-similar players
   gives **+2.38 pp on a target whose own games were never used** (B2). For a user with 20 games this
   is roughly half the achievable benefit, available immediately. This is Maia4All prototype
   initialisation, validated on our model.
5. **Do NOT build trait dials as a separate runtime mechanism.** Fitting the adapter on the user's own
   games already absorbs their traits. B2's value is explanatory and cold-start, not architectural.

## Why the old design failed, precisely

The frozen-base residual was not wrong in *kind*, it was **starved**. `arch_bench` gives a
dose-response curve, consistent across players: full (+4.76) > adapter (+3.94) > lastblock (+3.67) >
bitfit/norm (+2.65) > film (+2.11) > **head-only (+1.83)** > lhuc (+1.70). Head-only is structurally
closest to the old clone adapter. Capacity and placement, not concept, were the binding constraint.

## What the gain actually is (B2 decomposition, n=1, VEGETAL)

| component | value | share |
|---|---|---|
| generic same-Elo adaptation | +0.95 pp | 20 % |
| trait similarity | +1.43 pp | 30 % |
| identity proper | +2.38 pp | 50 % |

Traits carry **at least** 1.43 pp (clean contrast); identity **up to** 2.38 pp (entangled with
single-player concentration - see FINDINGS). n=1, paired sd 1.21 pp: suggestive, not established.

## Honest promise to users

- "Noticeably more like you than a generic human model." **Never "indistinguishable"** - 1-NN is
  0.825 against an ideal of 0.5.
- **Strong, predictable players gain little** (corr(base, gain) = -0.58; strongest base gained +0.79 pp,
  weakest +5.81 pp). Say so up front rather than disappointing them.
- Recent games are not required (a 3-month training shift costs 0.48 pp).

## What is still NOT done (project is not finished by its own definition)

- **Rung 3 has no measured noise floor** (base-vs-base MMD/1-NN never run), so its weak pass is provisional.
- **Rung 4 (behaviour under the engine's own distribution)** never run: mistake-rate-vs-clock curve and
  per-move loss distribution, with a base-vs-base branch control.
- **Rung 5 (human pair test)** never run. This is the ship gate.
- **B2 at n>=3 targets with headroom**, and the concentration confound separated.
