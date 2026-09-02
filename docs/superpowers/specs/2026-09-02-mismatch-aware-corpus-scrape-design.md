# Design: Mismatch-aware corpus scrape (player-identified, Elo-balanced, gap-quota'd)

**Date:** 2026-09-02
**Status:** Approved for planning
**Author:** session (sahformer-agent) + user
**Companion spec:** `2026-08-26-bigger-clone-base-time-elo-design.md` (the model). This spec supplies the
data that run trains on; the shard rebuild here is a hard dependency for it.

---

## 1. Motivation

clonefish's product is: **a person drops their chess.com username and gets a clone of themselves**, with
the fine-tune running **locally on their own GPU**. Three facts follow, and the current corpus violates
all three.

**1. The corpus is aimed at the wrong people.** The original sieve seeded from the blitz leaderboard and
titled-player lists, so the corpus we have is a *high-Elo* corpus. Recorded per-band game counts
(`data/old_db_band_counts.json` for the 5,000-player `data/chesscom_corpus`, plus the
`data/chesscom_lowmid` crawl state):

| Band | Games held | vs 100k target |
|---|---|---|
| 1000-1900 | 6.2k - 22.3k each | **~899k short** |
| 2000-2200 | 23.5k / 41.0k / 60.4k | ~175k short |
| 2300-2800 | 107.7k - 190.9k each | **already at target** |
| 2900 | 81.3k | ~19k short |
| 3000+ | 39.3k, 12.9k, 3.2k, 207 | supply-limited; will not fill |

Six bands are finished and need no crawling. The entire hole is **1000-2200** - precisely where the
people who will actually use the product live. Nobody dropping a username is 2600.

**2. Elo mismatches are being thrown away.** `chesscom_sieve.harvest_user` discards every game where the
two players differ by more than `--max-gap` (the low-mid crawl ran at 150). That deletes the signal the
user explicitly asked for: *"this guy is playing a lower rated opponent / a higher one"*. The 08-26 model
spec adds a normalised Elo-gap input feature; today there is almost no data in the corpus for it to learn
from.

**3. Positions carry no identity.** `PositionRecord` (`sahformer/records.py:9`) stores `elo_self` and
`elo_opp` but **no player and no date**. A person's games therefore cannot be pulled back out of the
corpus, and their games cannot be ordered in time - which blocks the rating-trajectory conditioning that
section 4.3 of the 08-26 spec depends on. The information exists in the PGN headers and is dropped at
record build time.

## 2. Goals / Non-goals

**Goals**
- An Elo-balanced chess.com corpus of **100k games per 100-Elo band across 800-3000**, weighted toward
  where the user base is, not where the crawler started.
- Elo mismatches **kept and quota'd**, so "plays up / plays down" is learnable rather than a thin tail.
- A cleaning system that removes games whose **Elo label is false**, while keeping games whose *behaviour*
  is merely unusual.
- **`player_id`, `date`, `site`** in every record, so a person's games are recoverable and orderable.
- A **whale holdout set** (~30 deep archives, excluded from training) to measure clone quality against
  number-of-games.

**Non-goals**
- No re-crawl of bands 2300-2800; they are done.
- No behavioural filtering (blowouts, premove flurries, time scrambles are kept - see 6.6).
- No Lichess harvest in this iteration. The `site` field exists so a later one does not require another
  rebuild.
- No change to the model architecture; that is the 08-26 spec.
- No per-user server-side fine-tuning infrastructure. The fine-tune runs on the user's machine, and a
  discrete GPU is a stated requirement.

## 3. Corpus target shape

- **Bands:** 100 Elo wide, floor **800**, ceiling **3000** -> 23 bands.
  The floor drops from 1000 to 800 because a large share of chess.com's active blitz population is below
  1000, and "anyone can clone themselves" has to include them.
- **Target:** 100,000 games per band -> ~2.3M games, ~184M positions, **~33GB** of shards at the observed
  ~180 bytes/position. Roughly 2.4x the current corpus.
- **Remaining work:** ~1.15M games (~899k of it below 2000), plus ~200k if the two new 800/900 bands are
  filled. Bands 2300-2800 contribute ~1.03M games already on disk at zero crawl cost.
- **Band assignment is per position, by the mover's own rating.** `chesscom_fill_bands.py:142` currently
  files a game under `band_of((wr + br) // 2)`, which is only meaningful for peer-matched games - a
  1300-vs-1900 game files as 1600, a band neither player is in. Since records are per-position and carry
  `elo_self`, band by that. One mismatched game then contributes positions to two bands, which is correct.
- **3000 will not fill.** Take what exists and record the true count. A band padded out by oversampling
  four accounts is worse than an honestly thin band.

### 3.1 Gap quotas (2-D steering)

Simply ceasing to discard mismatches lands them at an estimated 5-10% of the corpus, in the thin tail of
the gap feature - and the model will learn to ignore it, exactly as the Elo dial collapsed into a
confidence knob (HANDOFF-08-16 section 3). Volume must be guaranteed, not hoped for.

Extend the crawler's existing band steering to two dimensions: **band x gap bucket**, with buckets

    |gap| < 100    |    100 <= |gap| < 250    |    |gap| >= 250

signed by whether the mover is the stronger or the weaker side (5 buckets, since the small bucket is
unsigned). Per-band quota: **>= 20% of positions in the two >=100 buckets, >= 7% in the two >=250
buckets.** The crawler already walks the opponent graph toward the least-filled needy band; it steers
toward the least-filled needy *cell* by the same mechanism.

## 4. Record format change

Add three fields to `PositionRecord` (`sahformer/records.py`), populated in `game_to_records` from PGN
headers:

| Field | Type | Source | Why |
|---|---|---|---|
| `player_id` | int64 | stable hash of the **mover's** username | pull one person's games back out |
| `date` | int32 | game date as `YYYYMMDD` | order a player's games -> rating trajectory |
| `site` | int8 | 0 = chess.com, 1 = lichess | 1800 does not mean the same thing on both |

The Elo gap needs no field - `elo_self - elo_opp` already gives it, and the 08-26 spec computes it inside
the model from batch elos.

**Consequences:**
- A full shard rebuild is required. It is **not** a re-crawl: the 5,000 per-player `.pgn.zst` files in
  `data/chesscom_corpus` still hold the headers, so identity and date are recoverable for free.
- The HF dataset `slobaspeed/chesscom-balanced-shards` becomes stale and needs a v2 upload.
- Username hashing is one-way over the lowercased username with a **fixed salt committed as a module
  constant** - it must be identical across runs, machines and rebuilds, or a player's games scatter across
  several ids. Usernames themselves are never written into shards; the raw PGN archives retain them and
  remain local.

## 5. Reuse plan - do not re-crawl what we have

| Asset | State | Action |
|---|---|---|
| `data/chesscom_corpus` (5,000 player files, ~1.17M games, bands 2000-3300) | complete for 2300-2800 | **re-process only** - new fields, no crawl |
| `data/chesscom_lowmid` + `_trial` (650 players crawled, ~124k games) | far short of target | keep, re-process, continue the crawl from its state file |
| `data/chesscom_balanced_shards` (11GB) | no identity, no date | superseded; rebuild from the PGNs |

## 6. The cleaning system

**Principle: drop a game when the Elo label is a lie; keep it when the behaviour is merely unusual.**
A "1200" who is really a 2000-rated smurf poisons the conditioning - the model learns that 1200s play like
2000s. A 2000 casually crushing a 1200 is genuine human data and is exactly the "playing down" behaviour
we want. The first is deleted; the second is kept and labelled.

Layers, cheapest first. Layers 1-2 run at crawl/build time on API data; 3-5 are pure arithmetic over
archives we already hold.

### 6.1 Rated games only
`chesscom_sieve.harvest_user` checks `time_control` and `rules` but never `rated`. Casual games carry the
players' live ratings in the header while being played nothing like a rated game, and they cluster in
mismatches - a friendly against a much stronger friend is the canonical case. One-line filter.

### 6.2 Account status (fair-play closures)
Drop players whose chess.com profile `status` indicates a fair-play closure, and every game against them.
Mismatches are where engine assistance concentrates by construction - a 1300 playing 2600-level moves *is*
a mismatch.

Run this **at build time, not crawl time**: closures land months after the games. The player list can be
re-checked later at one request each and the corpus purged retroactively. The exact `status` values must
be verified against a live API response before the filter is relied upon.

### 6.3 Rating-stability windows
Because whole archives are pulled, each player's own rating curve is available at one-game resolution - so
a provisional or fraudulent label can be detected without any external flag. **Slide a 50-game window over
the player's games in date order; if the rating moves more than 150 points across that window, drop the 50
games in it and keep the rest of the player.** One rule catches new accounts, smurf ramps, and comeback
accounts, at zero extra API cost.

This is a second reason to raise `--months` past its default of 6: more games *and* a longer curve to
judge each game's label against.

### 6.4 Performance-rating check
Per player, compare actual score against the Elo-expected score across their games. A genuine 1200 scores
roughly 5% against 2000s; one scoring 40% is not 1200. Drop players whose performance rating diverges from
their nominal rating by more than ~200 over a sample of at least 50 games. This catches the smurfs whose
ramp predates the archive window - the case 6.3 misses.

### 6.5 Reciprocal check
The snowball crawl over the opponent graph means both players' archives are often held. Where they are,
validate each side's rating against their own independent history rather than trusting a single game
header. Partial coverage, free where it applies.

### 6.6 What is deliberately NOT filtered
No dropping of blowouts, resignations, premove flurries, or low-clock scrambles. This follows the call
already made in `build_corpus.iter_corpus`, which keeps timeout games on purpose because the scramble
before a flag-fall is real human timing signal. A rout in which the winner blitzes out twenty moves is
true data about how humans play when winning easily. If it skews the snap bucket, the answer is a feature
the model can condition on - not a deletion.

Existing cleaning is retained: disconnect/abandoned games dropped, miniatures under 10 plies dropped,
games deduped by uuid.

### 6.7 Audit - how we know the cleaning worked
The 08-26 spec adds an `EloHead` that predicts a player's rating. After training, compare its held-out
error on **mismatched** games against **matched** games. Comparable error means the labels are honest;
materially worse error on mismatches means something is still lying and 6.3-6.4 need tightening. This is
an acceptance criterion, not a vibe check.

## 7. Whale holdout set

~30 players with 10k+ 3+0 games, full archives pulled, **excluded from all training shards**.

Purpose: plot clone quality against number of games - 100, 300, 1k, 3k, 10k - on the *same* human. This
answers the first question every user hits ("do I have enough games?"), decides whether clonefish should
refuse to run below some threshold, and tells the product what to promise: at ~300 games you get your
openings; at ~3k you start getting your midgame. Today that curve exists for exactly one player
(latebloomer, HANDOFF-08-25 section 1).

Selection: spread across bands where 10k-game accounts actually exist (roughly 1200-2200), subject to the
same cleaning layers. Membership recorded in a manifest so the disjointness is verifiable, not assumed.

## 8. Crawler changes (`scripts/chesscom_fill_bands.py`, `scripts/chesscom_sieve.py`)

The directed band-capped snowball crawl already does most of this and is resumable. Changes:

1. `harvest_user`: add the `rated` filter; make `max_gap` default to unlimited; keep the yielded game's
   both-sides ratings (already present).
2. Band accounting: replace `band_of((wr + br) // 2)` with per-side banding by each mover's own rating.
3. Quota state: `band_count` becomes a 2-D `{band: {gap_bucket: n}}`; `pick()` steers to the least-filled
   needy **cell**. The JSON state file schema changes - old state files are migrated by treating existing
   counts as the `<100` bucket.
4. `--floor` default 800; `--ceiling` 3000.
5. Raise `--months` substantially (per-player depth + rating curve for 6.3).
6. Record per-band and per-cell **actual** counts at the end of the crawl, including shortfalls.

## 9. Acceptance criteria

1. Every record carries `player_id`, `date`, `site`; a named player's positions can be recovered from the
   shards and sorted into their true chronological order.
2. Bands 800-2900 reach >=100k games each, or the shortfall is recorded with the reason. Band 3000 is
   allowed to be short and is not padded by oversampling.
3. Per band, >=20% of positions have |gap| >= 100 and >=7% have |gap| >= 250, with both signs represented.
4. Cleaning layers 1-5 are individually toggleable and each reports how many games it removed, so their
   cost is measurable rather than assumed.
5. The whale set is verifiably disjoint from training shards (manifest + an automated check).
6. Post-training audit (6.7): `EloHead` held-out error on mismatched games is within tolerance of its
   error on matched games.
7. Total corpus ~2.3M games / ~184M positions, and the HF v2 dataset is uploaded.

## 10. Testing

- Unit: `game_to_records` populates `player_id`/`date`/`site` correctly, including which side is the mover;
  hashing is stable across runs and platforms.
- Unit: gap-bucket assignment at boundaries (99/100/249/250) and sign correctness.
- Unit: per-side band assignment on a mismatched game puts positions in two different bands.
- Unit: rating-stability detector flags a synthetic 1000->1800 ramp and leaves a flat curve alone.
- Unit: performance-rating check flags a synthetic over-performer at the stated thresholds.
- Integration: crawl-state migration from the old 1-D schema; a resumed crawl does not double-count.
- Integration: build a small corpus end-to-end, then recover one known player's games from the shards and
  confirm the count and date order match the source PGN.

## 11. Risks & open questions

- **Mismatch games shift what the corpus is.** More resignations, more blowouts, different clock
  behaviour. This is intended and was explicitly requested, but the old and new bases will not be cleanly
  comparable - any move-match or timing regression against the 19.5M base is confounded. Note it in the
  training run rather than explaining it away later.
- **The cleaning layers can over-fire.** The 150-points-over-50-games threshold in 6.3 is a first guess.
  It must be tuned by measuring how many games it removes (criterion 4) before the full rebuild, not
  after.
- **Low bands may be harder to fill than the arithmetic suggests.** Sub-1000 accounts play fewer games and
  churn more, so games-per-player will be lower than the ~190 the low-mid crawl saw. The crawl may need
  materially more players per band down there.
- **`status` field semantics unverified.** 6.2 rests on chess.com exposing fair-play closures in the
  public profile. If it does not, that layer is dropped and 6.3-6.4 carry the load; the audit in 6.7 will
  show whether that is sufficient.
- **Whale supply above 2200.** 10k-game 3+0 accounts thin out at higher ratings, so the games-vs-quality
  curve may only be measurable in the mid bands. Acceptable - that is where the users are.
- **Rebuild cost.** ~184M positions re-encoded from PGN. This is CPU-bound and single-threaded today;
  worth a parallel build pass, sized in planning.
