"""Build Elo-even .npz training shards from one or more per-player PGN corpora.

Differs from build_corpus.py in three ways (design 2026-09-02):

1. **Elo-even by construction.** Every band gets the same POSITION budget, so the model cannot
   ignore its Elo input and default to the majority band. build_corpus.py's `--balance` cannot
   do this: it balances each 150k chunk independently, and since PGNs are read per player the
   chunks cluster by band, so it downsamples to whatever the rarest band in that chunk happens
   to be with no global guarantee.
2. **Whole games only.** See sahformer/bandcap.py - truncating a game deletes its flag scramble.
3. **Admission is decided from the PGN HEADERS**, before encoding. Once a band fills, its games
   are rejected for the cost of a header read instead of a full board encode.

Player files are shuffled (seeded) so a band fills from a random cross-section of players
rather than from whoever sorts first.

    PYTHONPATH=. python -u scripts/build_corpus_v2.py \
        data/chesscom_bands_v2 data/chesscom_corpus data/chesscom_lowmid \
        --out data/shards_v2 --budget 7300000
"""
import argparse
import glob
import multiprocessing as mp
import os
import random
import sys
from collections import Counter

from sahformer.bandcap import BandBudget, bands_touched
from sahformer.dataset_build import build_shards
from sahformer.download import iter_games_from_zst
from sahformer.records import game_to_records


def player_files(dirs, seed=0):
    files = []
    for d in dirs:
        files.extend(glob.glob(os.path.join(d, "*.pgn.zst")))
    random.Random(seed).shuffle(files)
    return files


def _dedup_key(h):
    return h.get("Link") or (h.get("White"), h.get("Black"), h.get("UTCDate"), h.get("UTCTime"))


def admitted_records(dirs, budget, seed=0, min_plies=10, drop_abandoned=True,
                     stats=None, progress_every=1_000_000):
    """Stream PositionRecords for games admitted under the per-band position budget.

    Cleaning matches build_corpus.iter_corpus: dedup across files, drop disconnect/abandoned
    games (their last move carries a fake 'think'), drop miniatures. Timeout games are KEPT -
    the low-clock scramble before a flag-fall is the human timing signal we want.
    """
    stats = stats if stats is not None else Counter()
    seen = set()
    files = player_files(dirs, seed)
    print(f"{len(files):,} player files across {len(dirs)} dir(s); budget {budget.budget:,} "
          f"positions/band x {len(budget.bands)} bands", file=sys.stderr)

    for fi, path in enumerate(files):
        if budget.full():
            print("all bands at budget - stopping early", file=sys.stderr)
            break
        for g in iter_games_from_zst(path):
            if budget.full():
                break
            h = g.headers
            key = _dedup_key(h)
            if key in seen:
                stats["dup"] += 1
                continue
            seen.add(key)
            if drop_abandoned and "abandon" in h.get("Termination", "").lower():
                stats["abandoned"] += 1
                continue
            try:
                we = int(h.get("WhiteElo", 0) or 0)
                be = int(h.get("BlackElo", 0) or 0)
            except ValueError:
                stats["bad_elo"] += 1
                continue
            touched = bands_touched(we, be)
            if not touched:
                stats["out_of_range"] += 1
                continue
            if not budget.wants(touched):          # rejected for a header read, not an encode
                stats["band_full"] += 1
                continue

            recs = list(game_to_records(g))
            if len(recs) < min_plies:
                stats["short"] += 1
                continue
            budget.admit(recs)
            yield from recs

            if progress_every and budget.positions % progress_every < len(recs):
                print(f"  [{fi}/{len(files)} files] {budget.positions:,} positions, "
                      f"{budget.games:,} games, shortfall {budget.shortfall():,}", file=sys.stderr)



def _worker(job):
    """One worker: encode its slice of player files against its own share of the band budget.

    Splitting the budget per worker rather than sharing one counter keeps this lock-free. Bands
    come out approximately even by the law of large numbers over thousands of files, and the
    parent reports the ACTUAL per-band counts rather than assuming they landed on target.
    """
    wid, files, budget_each, out, seed, min_plies, drop_abandoned = job
    from sahformer.bandcap import BandBudget
    from sahformer.dataset_build import build_shards
    budget = BandBudget(budget_each)
    stats = Counter()
    paths = build_shards(
        _admit_from_files(files, budget, seed=seed, min_plies=min_plies,
                          drop_abandoned=drop_abandoned, stats=stats, quiet=True),
        out, chunk_positions=200000, balance=False, prefix=f"w{wid}_shard")
    return wid, dict(budget.counts), budget.games, budget.positions, budget.spill, dict(stats), len(paths)


def _admit_from_files(files, budget, seed=0, min_plies=10, drop_abandoned=True,
                      stats=None, quiet=False):
    """Same admission rule as admitted_records, over an explicit file list."""
    stats = stats if stats is not None else Counter()
    seen = set()
    for path in files:
        if budget.full():
            break
        for g in iter_games_from_zst(path):
            if budget.full():
                break
            h = g.headers
            key = _dedup_key(h)
            if key in seen:
                stats["dup"] += 1
                continue
            seen.add(key)
            if drop_abandoned and "abandon" in h.get("Termination", "").lower():
                stats["abandoned"] += 1
                continue
            try:
                we = int(h.get("WhiteElo", 0) or 0); be = int(h.get("BlackElo", 0) or 0)
            except ValueError:
                stats["bad_elo"] += 1
                continue
            touched = bands_touched(we, be)
            if not touched:
                stats["out_of_range"] += 1
                continue
            if not budget.wants(touched):
                stats["band_full"] += 1
                continue
            recs = list(game_to_records(g))
            if len(recs) < min_plies:
                stats["short"] += 1
                continue
            budget.admit(recs)
            yield from recs


def main():
    ap = argparse.ArgumentParser(description="Elo-even shard build with whole-game admission")
    ap.add_argument("dirs", nargs="+", help="per-player .pgn.zst corpus directories")
    ap.add_argument("--out", required=True, help="output shard directory")
    ap.add_argument("--budget", type=int, default=7_300_000,
                    help="positions per Elo band (measured: ~72.8 positions/game, so 100k "
                         "games/band ~= 7.3M)")
    ap.add_argument("--chunk", type=int, default=200000, help="positions per .npz shard")
    ap.add_argument("--seed", type=int, default=0, help="shuffles the player-file order")
    ap.add_argument("--min-plies", type=int, default=10)
    ap.add_argument("--keep-abandoned", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel encoders; each gets budget/workers per band")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.workers <= 1:
        budget = BandBudget(args.budget)
        stats = Counter()
        paths = build_shards(
            admitted_records(args.dirs, budget, seed=args.seed, min_plies=args.min_plies,
                             drop_abandoned=not args.keep_abandoned, stats=stats),
            args.out, chunk_positions=args.chunk, balance=False)
        print("\n=== band budget ===")
        print(budget.report())
        print(f"\nrejected: {dict(stats)}")
        print(f"wrote {len(paths)} shard(s) to {args.out}")
        return

    files = player_files(args.dirs, args.seed)
    chunks = [files[i::args.workers] for i in range(args.workers)]
    per = max(args.budget // args.workers, 1)
    print(f"{len(files):,} files across {args.workers} workers | {per:,} positions/band each "
          f"({args.budget:,} total)", flush=True)
    jobs = [(i, chunks[i], per, args.out, args.seed + i, args.min_plies,
             not args.keep_abandoned) for i in range(args.workers)]

    total = Counter()
    games = pos = spill = nshards = 0
    rej = Counter()
    with mp.Pool(args.workers) as pool:
        for wid, counts, g, p, sp, st, ns in pool.imap_unordered(_worker, jobs):
            for b, n in counts.items():
                total[b] += n
            games += g; pos += p; spill += sp; nshards += ns
            for k, v in st.items():
                rej[k] += v
            print(f"  worker {wid} done: {g:,} games, {p:,} positions, {ns} shards", flush=True)

    print("\n=== combined bands (ACTUAL counts, not assumed) ===")
    for b in sorted(total):
        print(f"  {b}-{b + 99}: {total[b]:>10,}")
    print(f"\ntotal {games:,} games | {pos:,} positions | {nshards} shards | spill {spill:,}")
    print(f"rejected: {dict(rej)}")


if __name__ == "__main__":
    main()
