"""Build BALANCED training shards: cap every 100-Elo band to `--per-band` games, drawing from many
corpus dirs (old DB high bands + crawled low-mid bands). Bins each game by the average of the two
player ratings, stops feeding a band once it hits the cap, dedups across dirs, drops
abandoned/miniature games, then streams the kept games through the standard records->shard pipeline.

Usage:
    PYTHONPATH=. python scripts/build_balanced.py data/chesscom_balanced_shards \
        --per-band 50000 --floor 1000 --ceiling 3000 \
        --dirs data/chesscom_corpus data/chesscom_lowmid data/chesscom_lowmid_trial \
               data/chesscom_p1 data/chesscom_p2 data/chesscom_p3
"""
import argparse
import collections
import glob
import os

from sahformer.download import iter_games_from_zst
from sahformer.dataset_build import build_shards, records_from_games


def balanced_games(dirs, per_band, floor, ceiling, min_plies=10):
    """Yield up to `per_band` clean games for each 100-Elo band in [floor, ceiling)."""
    bands = set(range(floor, ceiling, 100))
    band_count = collections.Counter()
    seen = set()
    files = []
    for d in dirs:
        files += sorted(glob.glob(os.path.join(d, "*.pgn.zst")))
    print(f"scanning {len(files)} player files across {len(dirs)} dirs...")

    for path in files:
        for g in iter_games_from_zst(path):
            we = g.headers.get("WhiteElo")
            be = g.headers.get("BlackElo")
            if not we or not be:
                continue
            try:
                b = ((int(we) + int(be)) // 2) // 100 * 100         # band by average rating
            except ValueError:
                continue
            if b not in bands or band_count[b] >= per_band:
                continue
            key = g.headers.get("Link") or (g.headers.get("White"), g.headers.get("Black"),
                                            g.headers.get("UTCDate"), g.headers.get("UTCTime"))
            if key in seen:
                continue
            seen.add(key)
            if "abandon" in g.headers.get("Termination", "").lower():
                continue
            if sum(1 for _ in g.mainline_moves()) < min_plies:
                continue
            band_count[b] += 1
            yield g
        if all(band_count[b] >= per_band for b in bands):
            print("all bands capped — stopping scan early.")
            break

    print("\nper-band games kept:")
    for b in sorted(bands):
        c = band_count[b]
        print(f"  {b}-{b+99}: {c:>6,} {'' if c >= per_band else '(under cap)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--per-band", type=int, default=50000)
    ap.add_argument("--floor", type=int, default=1000)
    ap.add_argument("--ceiling", type=int, default=3000)
    ap.add_argument("--chunk", type=int, default=250000)
    args = ap.parse_args()

    paths = build_shards(
        records_from_games(balanced_games(args.dirs, args.per_band, args.floor, args.ceiling)),
        args.outdir, chunk_positions=args.chunk, progress_every=100000)
    print(f"\nwrote {len(paths)} shard(s) to {args.outdir}")


if __name__ == "__main__":
    main()
