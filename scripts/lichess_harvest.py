"""Harvest per-player games from a Lichess monthly DB dump (for the scale experiments).

    PYTHONPATH=. python scripts/lichess_harvest.py --dump lichess_db_standard_rated_2018-05.pgn.zst \
        --out data/lichess_scale --time-control 180+0 --min-rating 1000 --max-rating 2000 \
        --min-games 140 --max-players 3000 --cap 400

Get a dump from https://database.lichess.org/  (standard, rated). Clocks (%clk) are present in
real-time games since April 2017, at WHOLE-SECOND resolution — coarser than chess.com's 0.1s, so
treat any "timing doesn't help" result here as resolution-limited, not a verdict. A ~2017-2019
month is a good size/coverage tradeoff (recent months are 30GB+ compressed).

Two passes over the dump: (1) count games/player for the target time control + rating band,
(2) write the top players' games to <out>/<user>.pgn.zst. Same format build_cache_scale.py eats.
"""
import argparse
import io
import os
import re
import sys
import time
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import zstandard

RE_WHITE = re.compile(r'\[White "([^"]*)"\]')
RE_BLACK = re.compile(r'\[Black "([^"]*)"\]')
RE_TC = re.compile(r'\[TimeControl "([^"]*)"\]')
RE_WELO = re.compile(r'\[WhiteElo "(\d+)"\]')
RE_BELO = re.compile(r'\[BlackElo "(\d+)"\]')


def iter_blocks(path):
    """Yield each game's full PGN text. Every Lichess game starts with a [Event line."""
    dctx = zstandard.ZstdDecompressor()
    with open(path, "rb") as fh:
        text = io.TextIOWrapper(dctx.stream_reader(fh), encoding="utf-8", errors="ignore")
        lines = []
        for line in text:
            if line.startswith("[Event ") and lines:
                yield "".join(lines)
                lines = [line]
            else:
                lines.append(line)
        if lines:
            yield "".join(lines)


def headers(block):
    w = RE_WHITE.search(block); b = RE_BLACK.search(block); tc = RE_TC.search(block)
    we = RE_WELO.search(block); be = RE_BELO.search(block)
    return (w.group(1) if w else None, b.group(1) if b else None,
            tc.group(1) if tc else None,
            int(we.group(1)) if we else 0, int(be.group(1)) if be else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="lichess_db_standard_rated_YYYY-MM.pgn.zst")
    ap.add_argument("--out", default="data/lichess_scale")
    ap.add_argument("--time-control", default="180+0", help="exact TimeControl header to keep (3+0 = 180+0)")
    ap.add_argument("--min-rating", type=int, default=1000)
    ap.add_argument("--max-rating", type=int, default=2000)
    ap.add_argument("--min-games", type=int, default=140, help="keep players with >= this many games")
    ap.add_argument("--max-players", type=int, default=3000)
    ap.add_argument("--cap", type=int, default=400, help="max games written per player (memory bound)")
    ap.add_argument("--keep-users-from-dir", default=None,
                    help="only harvest players whose <user>.pgn.zst already exists in this dir "
                         "(for cross-month: pass the earlier window's corpus to match the SAME players)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    TC = args.time_control

    keep = None
    if args.keep_users_from_dir:
        import glob as _g
        keep = {os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0].lower()
                for f in _g.glob(os.path.join(args.keep_users_from_dir, "*.pgn.zst"))}
        print(f"restricting to {len(keep)} known users from {args.keep_users_from_dir}")

    def in_band(elo):
        return args.min_rating <= elo <= args.max_rating

    def wanted(user):
        return keep is None or user.lower() in keep

    # ---- pass 1: count games per player (target TC + player in band) ----
    print(f"pass 1: counting players in {args.dump} (TC={TC}, rating {args.min_rating}-{args.max_rating})")
    cnt = Counter(); t0 = time.time(); n = 0
    for block in iter_blocks(args.dump):
        n += 1
        w, b, tc, we, be = headers(block)
        if tc != TC or w is None:
            continue
        if in_band(we) and wanted(w):
            cnt[w] += 1
        if in_band(be) and wanted(b):
            cnt[b] += 1
        if n % 2_000_000 == 0:
            print(f"  scanned {n:,} games, {len(cnt):,} players, {time.time()-t0:.0f}s")
    picked = [u for u, c in cnt.most_common() if c >= args.min_games][:args.max_players]
    picked_set = set(picked)
    print(f"pass 1 done: {n:,} games scanned; {len(picked)} players with >= {args.min_games} games kept")
    if not picked:
        print("no players met the threshold — lower --min-games or --min-rating band."); return

    # ---- pass 2: buffer the picked players' games, write per-player files ----
    print("pass 2: collecting games for picked players")
    buf = defaultdict(list); t0 = time.time(); n = 0
    for block in iter_blocks(args.dump):
        n += 1
        w, b, tc, we, be = headers(block)
        if tc != TC or w is None:
            continue
        if "%clk" not in block:                       # need clock annotations
            continue
        for user, elo in ((w, we), (b, be)):
            if user in picked_set and in_band(elo) and len(buf[user]) < args.cap:
                buf[user].append(block)
        if n % 2_000_000 == 0:
            print(f"  scanned {n:,} games, buffered {sum(len(v) for v in buf.values()):,} games, {time.time()-t0:.0f}s")

    written = 0
    cctx = zstandard.ZstdCompressor(level=10)
    for user, blocks in buf.items():
        if len(blocks) < args.min_games:              # some games dropped for missing %clk
            continue
        path = os.path.join(args.out, f"{user}.pgn.zst")
        with open(path, "wb") as fh:
            with cctx.stream_writer(fh) as z:
                z.write("\n".join(blocks).encode("utf-8"))
        written += 1
    print(f"\nwrote {written} player files to {args.out}/  "
          f"(next: build_cache_scale.py --corpus {args.out})")


if __name__ == "__main__":
    main()
