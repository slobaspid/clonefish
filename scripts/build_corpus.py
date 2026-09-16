"""Build .npz training shards from a Chess.com sieve corpus (a directory of per-player
.pgn.zst files written by scripts/chesscom_sieve.py).

Streams every game through the SAME records -> shard pipeline used for Lichess data, so the
output shards are drop-in for training. Dedups games across files (a safety net for the rare
duplicate that can appear across a resumed harvest).

Usage:
    python scripts/build_corpus.py data/chesscom_corpus data/chesscom_shards --max-positions 50000000
"""
import argparse
import glob
import os

from sahformer.download import iter_games_from_zst
from sahformer.dataset_build import build_shards, records_from_games


def iter_corpus(corpus_dir, dedup=True, drop_abandoned=True, min_plies=10):
    """Yield games from every <corpus_dir>/*.pgn.zst.

    Cleaning (on by default): drop disconnect/abandoned games (their last move carries a fake
    'think' where a player just left) and drop miniatures under `min_plies`. Timeout games are
    KEPT on purpose — the low-clock scramble before a flag-fall is exactly the human timing signal
    we want. Also dedups by game link/identity across files (resume-boundary safety net).
    """
    seen = set()
    kept = dropped = 0
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.pgn.zst"))):
        for g in iter_games_from_zst(path):
            if dedup:
                key = g.headers.get("Link") or (
                    g.headers.get("White"), g.headers.get("Black"),
                    g.headers.get("UTCDate"), g.headers.get("UTCTime"))
                if key in seen:
                    continue
                seen.add(key)
            if drop_abandoned and "abandon" in g.headers.get("Termination", "").lower():
                dropped += 1
                continue
            if min_plies and sum(1 for _ in g.mainline_moves()) < min_plies:
                dropped += 1
                continue
            kept += 1
            yield g
    print(f"cleaning: kept {kept:,} games, dropped {dropped:,} (abandoned/miniatures)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_dir", help="directory of per-player .pgn.zst files")
    ap.add_argument("outdir", help="where to write shard*.npz")
    ap.add_argument("--max-positions", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=200000)
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--keep-abandoned", action="store_true", help="don't drop disconnect games")
    ap.add_argument("--min-plies", type=int, default=10, help="drop games shorter than this")
    args = ap.parse_args()

    n_files = len(glob.glob(os.path.join(args.corpus_dir, "*.pgn.zst")))
    print(f"corpus: {n_files} player files in {args.corpus_dir}")
    paths = build_shards(
        records_from_games(iter_corpus(args.corpus_dir, dedup=not args.no_dedup,
                                       drop_abandoned=not args.keep_abandoned,
                                       min_plies=args.min_plies)),
        args.outdir, chunk_positions=args.chunk, max_positions=args.max_positions,
        progress_every=50000)
    print(f"wrote {len(paths)} shard(s) to {args.outdir}")


if __name__ == "__main__":
    main()
