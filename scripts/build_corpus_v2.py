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
    args = ap.parse_args()

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


if __name__ == "__main__":
    main()
