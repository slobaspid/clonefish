"""Directed, band-capped snowball crawl.

Fills every 100-Elo band in [floor, ceiling) to `target` peer-matched 3+0 games from Chess.com,
then stops feeding that band. Steers toward under-filled bands: the frontier is bucketed by band,
and we always crawl a player from the *least-filled* needy band, walking the opponent graph up/down
the ladder toward whatever is short. Resumable via a JSON state file in the output dir — kill it and
re-run the same command and it continues.

Usage:
    PYTHONPATH=. python -u scripts/chesscom_fill_bands.py --out data/chesscom_lowmid \
        --floor 1000 --ceiling 2200 --target 50000 --max-gap 150
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict, deque

import zstandard as zstd
import chesscom_sieve as cs   # scripts/ is on sys.path when run as a script


def band_of(rating):
    return (int(rating) // 100) * 100


# Elo-gap buckets, from the MOVER's point of view. Sign says who is stronger:
#   0 = even (<100)   +1 = playing DOWN 100-249   -1 = playing UP 100-249
#                     +2 = playing DOWN 250+      -2 = playing UP 250+
# Fractions are the per-band share each bucket must reach (design 2026-09-02 s3.1:
# >=20% of a band at |gap|>=100, >=7% at |gap|>=250). Anything above target is fine;
# these are floors, and bucket 0 soaks up the remainder.
BUCKET_FRAC = {0: 0.80, 1: 0.065, -1: 0.065, 2: 0.035, -2: 0.035}
BUCKETS = tuple(BUCKET_FRAC)


def gap_bucket(own, opp):
    """Which gap bucket a position by `own` against `opp` belongs to."""
    d = int(own) - int(opp)
    a = abs(d)
    if a < 100:
        return 0
    if a < 250:
        return 1 if d > 0 else -1
    return 2 if d > 0 else -2


def _as_cells(raw):
    """Read a {band: n} (old 1-D schema) or {band: {bucket: n}} counter into the 2-D form.

    Old counts were harvested peer-matched (--max-gap 150), so they migrate into bucket 0.
    """
    out = defaultdict(lambda: defaultdict(int))
    for b, v in (raw or {}).items():
        if isinstance(v, dict):
            for k, n in v.items():
                out[int(b)][int(k)] += int(n)
        else:
            out[int(b)][0] += int(v)
    return out


def seed_from_corpus(corpus_dir, floor, ceiling):
    """Mine (username, rating) pairs out of an existing corpus dir's PGN headers.

    The high-band corpus was harvested at --max-gap 150, so those accounts' MISMATCH games were
    discarded — they are exactly the players the high bands still need. Their ratings are already in
    the headers, so seeding from them costs no API calls, and unlike the country-list seed it can
    actually reach 2300+ (the opponent graph from a low-band frontier never walks up that far).
    """
    import collections
    import glob
    import io as _io

    hdr = re.compile(r'\[(White|Black|WhiteElo|BlackElo) "([^"]*)"\]')
    rating = {}
    dctx = zstd.ZstdDecompressor()
    for p in glob.glob(os.path.join(corpus_dir, "*.pgn.zst")):
        try:
            with open(p, "rb") as fh:
                text = _io.TextIOWrapper(dctx.stream_reader(fh), encoding="utf-8", errors="ignore")
                cur = {}
                for line in text:
                    m = hdr.match(line)
                    if m:
                        cur[m.group(1)] = m.group(2)
                    elif line.startswith("1.") and cur:       # end of a game's headers
                        for side in ("White", "Black"):
                            u = cur.get(side, "")
                            try:
                                r = int(cur.get(side + "Elo", 0))
                            except ValueError:
                                r = 0
                            if u and floor <= r < ceiling:
                                rating[u] = r                  # last rating seen for that account wins
                        cur = {}
        except Exception:                                      # a truncated archive must not stop seeding
            continue
    return sorted(rating.items(), key=lambda kv: -kv[1])        # strongest first


def load_state(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"band_count": {}, "crawled": [], "frontier": []}


def save_state(path, band_count, crawled, buckets):
    frontier = [[u, r] for dq in buckets.values() for (u, r) in dq]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"band_count": {str(b): {str(k): n for k, n in cells.items()}
                                  for b, cells in band_count.items()},
                   "crawled": list(crawled), "frontier": frontier}, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/chesscom_lowmid")
    ap.add_argument("--floor", type=int, default=800)
    ap.add_argument("--ceiling", type=int, default=3000)
    ap.add_argument("--target", type=int, default=100000, help="games per 100-Elo band")
    ap.add_argument("--max-gap", type=int, default=10000,
                    help="max Elo diff between sides; the default keeps mismatches (they are quota'd)")
    ap.add_argument("--months", type=int, default=60)
    ap.add_argument("--sleep", type=float, default=0.25)
    ap.add_argument("--countries",
                    default="RS,US,IN,DE,BR,ES,FR,RU,PL,NL,GB,UA,AR,TR,IT,PH,CA,MX,ID,VN,CN,EG,IR")
    ap.add_argument("--seed-users", type=int, default=4000, help="initial country seeds (bootstrap)")
    ap.add_argument("--seed-titled", type=int, default=0, help="also seed N titled players (high-band entry)")
    ap.add_argument("--seed-dir", default="",
                    help="mine (user, rating) from an existing corpus dir's PGN headers and add them to "
                         "the frontier (no API calls). The way to reach the high bands.")
    ap.add_argument("--prior-counts", default="",
                    help="JSON {band: existing_game_count} to count toward target (e.g. the old DB)")
    ap.add_argument("--max-crawl", type=int, default=300000, help="safety cap on players crawled")
    ap.add_argument("--save-every", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    state_path = os.path.join(args.out, "_crawl_state.json")
    st = load_state(state_path)
    band_count = _as_cells(st["band_count"])                        # this crawl's own harvested games
    crawled = set(st["crawled"])
    bands = list(range(args.floor, args.ceiling, 100))

    prior = defaultdict(lambda: defaultdict(int))                    # external games already in hand (old DB)
    if args.prior_counts and os.path.exists(args.prior_counts):
        with open(args.prior_counts, encoding="utf-8") as f:
            prior = _as_cells(json.load(f))
        print(f"loaded prior counts for {len(prior)} bands (old DB head-start)", file=sys.stderr)

    # frontier bucketed by band for O(bands) steering
    buckets = defaultdict(deque)
    frontier_set = set()
    for u, r in st["frontier"]:
        if u not in crawled and u not in frontier_set:
            buckets[band_of(r)].append((u, r))
            frontier_set.add(u)

    def have(b, k=None):
        if k is not None:
            return prior[b][k] + band_count[b][k]
        return sum(prior[b][j] + band_count[b][j] for j in BUCKETS)

    def cell_target(k):
        return int(args.target * BUCKET_FRAC[k])

    def cell_needs(b, k):
        return b in bands and have(b, k) < cell_target(k)

    def needs(b):
        """A band still wants games if ANY of its gap buckets is under its floor.

        Bands crawled peer-matched (the old DB) look full on totals but hold zero
        mismatches, so they stay needy until their +/-1 and +/-2 cells fill.
        """
        return b in bands and any(cell_needs(b, k) for k in BUCKETS)

    def shortfall(b):
        return sum(max(0, cell_target(k) - have(b, k)) for k in BUCKETS)

    def all_full():
        return not any(needs(b) for b in bands)

    # bootstrap the frontier from country lists (low entry) + optional titled (high entry)
    if not frontier_set:
        print("seeding frontier from country lists...", file=sys.stderr)
        for u in cs.seed_country([c.strip() for c in args.countries.split(",")], args.seed_users):
            if u not in crawled and u not in frontier_set:
                buckets[args.floor].append((u, args.floor))   # unknown rating: crawl to discover
                frontier_set.add(u)
        if args.seed_titled > 0:
            print("seeding frontier from titled players (high-band entry)...", file=sys.stderr)
            hi = max(b for b in bands)
            for u in cs.seed_titled_multi(["GM", "IM", "FM", "NM", "WGM", "WIM", "WFM"], args.seed_titled):
                if u not in crawled and u not in frontier_set:
                    buckets[hi].append((u, hi))               # crawl these when high bands need filling
                    frontier_set.add(u)

    # Corpus seed runs on EVERY start, not just the bootstrap: an existing frontier is no help if it
    # holds nothing above ~2000, which is the state a country-list seed leaves it in.
    if args.seed_dir and os.path.isdir(args.seed_dir):
        print(f"seeding frontier from corpus {args.seed_dir} ...", file=sys.stderr)
        added = 0
        for u, r in seed_from_corpus(args.seed_dir, args.floor, args.ceiling):
            if u not in crawled and u not in frontier_set:
                buckets[band_of(r)].append((u, r))
                frontier_set.add(u)
                added += 1
        print(f"  seeded {added:,} players (frontier now {len(frontier_set):,})", file=sys.stderr)

    cctx = zstd.ZstdCompressor(level=10)
    seen_uuid = set()
    processed = 0
    t0 = time.time()

    def pick():
        # least-filled needy band that has candidates; else any non-empty bucket (to bridge)
        cand = [b for b in bands if needs(b) and buckets.get(b)]
        if cand:
            b = max(cand, key=shortfall)          # neediest band across all its gap cells
            return buckets[b].popleft()
        for b, dq in buckets.items():
            if dq:
                return dq.popleft()
        return None

    while not all_full() and processed < args.max_crawl:
        item = pick()
        if item is None:
            print("frontier exhausted.", file=sys.stderr)
            break
        user, _ = item
        frontier_set.discard(user)
        if user in crawled:
            continue
        crawled.add(user)

        blob = []
        try:
            for g in cs.harvest_user(user, args.months, args.floor, args.sleep, seen_uuid,
                                     max_rating=args.ceiling, max_gap=args.max_gap):
                wr = g["white"]["rating"]
                br = g["black"]["rating"]
                # A game is TWO sets of positions, one per mover. Band each side by its OWN
                # rating (the pair average files a 1300-vs-1900 game under 1600, a band neither
                # player is in) and count it into that side's gap cell. Keep the game if either
                # side lands somewhere still needy.
                # Dedup by band so units stay GAMES, matching the old per-game prior counts:
                # peer-matched sides share a band (gap <100 cannot straddle a 100-wide band) and
                # collapse to one entry; a mismatch lands in two distinct bands, one each.
                cells = {}
                for own, opp in ((wr, br), (br, wr)):
                    cells[band_of(own)] = gap_bucket(own, opp)
                cells = list(cells.items())
                if any(cell_needs(b, k) for b, k in cells):
                    pgn = g["pgn"].replace('[TimeControl "180"]', '[TimeControl "180+0"]')
                    blob.append(pgn.strip())
                    for b, k in cells:
                        if b in bands:
                            band_count[b][k] += 1
                for side in ("white", "black"):
                    opp = g[side]["username"]
                    oppr = g[side]["rating"]
                    if (opp != user and opp not in crawled and opp not in frontier_set
                            and args.floor <= oppr < args.ceiling and needs(band_of(oppr))):
                        buckets[band_of(oppr)].append((opp, oppr))
                        frontier_set.add(opp)
        except Exception as e:                                  # never let one bad account stop the crawl
            print(f"  err {user}: {e}", file=sys.stderr)

        if blob:
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", user)
            path = os.path.join(args.out, f"{safe}.pgn.zst")
            with open(path, "wb") as fh, cctx.stream_writer(fh) as comp:
                comp.write(("\n\n".join(blob) + "\n\n").encode("utf-8"))
        processed += 1

        if processed % args.save_every == 0:
            save_state(state_path, band_count, crawled, buckets)
            full = sum(1 for b in bands if not needs(b))
            tot = sum(have(b) for b in bands)
            mism = sum(have(b, k) for b in bands for k in BUCKETS if k)
            front = sum(len(dq) for dq in buckets.values())
            print(f"[{processed}] bands {full}/{len(bands)} full | {tot:,} sides "
                  f"({mism:,} mismatch) | frontier {front:,} | {time.time()-t0:.0f}s", file=sys.stderr)

    save_state(state_path, band_count, crawled, buckets)
    print("\n=== crawl checkpoint ===  (per band: total / target, then the gap cells)")
    for b in bands:
        cells = "  ".join(f"{k:+d}:{have(b,k):,}/{cell_target(k):,}" for k in (-2, -1, 0, 1, 2))
        print(f"  {b}-{b+99}: {have(b):>7,}/{args.target:,}  {cells}"
              f" {'FULL' if not needs(b) else ''}")
    print(f"players crawled this run: {processed}   frontier left: "
          f"{sum(len(dq) for dq in buckets.values()):,}")


if __name__ == "__main__":
    main()
