"""Chess.com high-Elo 3+0 sieve.

Harvests standard 3+0 (time_control "180") blitz games with clock annotations from the
Chess.com *public* API (no auth). Writes one compressed PGN per player into an output
directory; the existing records.py -> shard pipeline (via scripts/build_corpus.py) consumes
them exactly like Lichess data.

Access model note: Chess.com has no bulk dump. Games are fetched per player, per month
(GET /pub/player/{user}/games/{YYYY}/{MM}). We seed strong usernames from the blitz
leaderboard and/or titled-player lists, then pull their recent archives. A game between two
seeded players appears in BOTH archives, so we dedup by game uuid within a run.

Resumable: each player's games go to <outdir>/<user>.pgn.zst. On restart, players whose file
already exists are skipped, so an interrupted multi-hour harvest picks up where it left off.

Etiquette: single-threaded, descriptive User-Agent, polite sleep, backoff on HTTP 429.
"""
import argparse
import io
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error

import zstandard as zstd

UA = "sah-transformer-research/1.0 (contact: dusan.rakic007@gmail.com)"
API = "https://api.chess.com/pub"


def api_get(url, retries=4):
    """GET JSON with a real User-Agent, backing off on rate limits / transient errors."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8", "ignore"))
        except urllib.error.HTTPError as e:
            if e.code == 429:                      # rate limited: back off and retry
                wait = 2 ** attempt
                print(f"  429 rate-limited, waiting {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            if e.code == 404:                      # no such archive/player
                return None
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 ** attempt)
    return None


def seed_blitz_leaderboard(limit):
    """Top live-blitz players by rating — the densest high-Elo 3+0 pool."""
    data = api_get(f"{API}/leaderboards") or {}
    return [p["username"] for p in data.get("live_blitz", [])][:limit]


def seed_titled(title, limit):
    """All players with a given title (GM, IM, FM, ...)."""
    data = api_get(f"{API}/titled/{title}") or {}
    return data.get("players", [])[:limit]


def seed_titled_multi(titles, limit):
    """Merge several title lists, dedup, then shuffle so a --max-users cap samples across the
    whole strength range (GMs..NMs) instead of taking all GMs first."""
    import random
    names, seen = [], set()
    for t in titles:
        for u in seed_titled(t, limit=10 ** 9):
            if u not in seen:
                seen.add(u)
                names.append(u)
    random.Random(0).shuffle(names)
    return names[:limit]


def seed_opponents_from_dir(corpus_dir, min_rating, limit):
    """Snowball wave 2: mine every strong player (>= min_rating in some game) out of the games
    already harvested into corpus_dir, EXCLUDING players we've already got a file for. Ranked by
    how often they appear (frequent = active = high future yield). Finds the strong UNTITLED
    accounts that titled players faced but that were never in a titled list."""
    import collections
    import glob
    import re

    have = {os.path.basename(p)[:-8].lower() for p in glob.glob(os.path.join(corpus_dir, "*.pgn.zst"))}
    hdr = re.compile(r'\[(White|Black|WhiteElo|BlackElo) "([^"]*)"\]')
    freq = collections.Counter()
    dctx = zstd.ZstdDecompressor()
    for p in glob.glob(os.path.join(corpus_dir, "*.pgn.zst")):
        with open(p, "rb") as fh:
            text = io.TextIOWrapper(dctx.stream_reader(fh), encoding="utf-8", errors="ignore")
            cur = {}
            for line in text:
                m = hdr.match(line)
                if m:
                    cur[m.group(1)] = m.group(2)
                elif line.startswith("1.") and cur:          # end of a game's headers
                    for side in ("White", "Black"):
                        try:
                            if int(cur.get(side + "Elo", 0)) >= min_rating:
                                u = cur.get(side, "")
                                if u and re.sub(r"[^A-Za-z0-9_.-]", "_", u).lower() not in have:
                                    freq[u] += 1
                        except ValueError:
                            pass
                    cur = {}
    ranked = [u for u, _ in freq.most_common(limit)]
    print(f"opponents mined: {len(freq):,} distinct new strong players; taking top {len(ranked):,}",
          file=sys.stderr)
    return ranked


def seed_country(countries, limit):
    """Usernames from country player lists (all rating levels) — a broad low-to-high seed.
    /pub/country/{ISO}/players returns up to ~10k usernames spanning the whole rating range,
    which is how we reach the low/mid bands the titled lists can't."""
    import random
    names, seen = [], set()
    for iso in countries:
        data = api_get(f"{API}/country/{iso}/players") or {}
        for u in data.get("players", []):
            if u not in seen:
                seen.add(u)
                names.append(u)
    random.Random(0).shuffle(names)
    return names[:limit]


def recent_archives(user, months):
    """The last `months` monthly-archive URLs for a user (most recent last)."""
    data = api_get(f"{API}/player/{user}/games/archives")
    if not data:
        return []
    return data.get("archives", [])[-months:]


def harvest_user(user, months, min_rating, sleep, seen, max_rating=10000, max_gap=10000):
    """Yield unique standard-3+0 game dicts for ONE user: both sides in [min_rating, max_rating]
    AND within `max_gap` Elo of each other (peer-matched / competitive games only)."""
    for url in recent_archives(user, months):
        month = api_get(url) or {}
        for g in month.get("games", []):
            if g.get("time_control") != "180" or g.get("rules") != "chess":
                continue
            if not g.get("rated", False):        # casual games carry live ratings but aren't played like rated ones
                continue
            uuid = g.get("uuid")
            if uuid in seen:
                continue
            wr = g.get("white", {}).get("rating", 0)
            br = g.get("black", {}).get("rating", 0)
            if min(wr, br) < min_rating or max(wr, br) > max_rating:   # both sides in target range
                continue
            if abs(wr - br) > max_gap:                                 # peer-matched only (no mismatches)
                continue
            seen.add(uuid)
            yield g
        time.sleep(sleep)                           # be polite between archive fetches


def main():
    ap = argparse.ArgumentParser(description="Chess.com high-Elo 3+0 sieve (resumable)")
    ap.add_argument("--out", default="data/chesscom_corpus", help="output DIR (one .pgn.zst/user)")
    ap.add_argument("--seed", choices=["blitz", "titled", "opponents", "country", "GM", "IM", "FM"],
                    default="titled")
    ap.add_argument("--titles", default="GM,IM,FM,NM,WGM,WIM,WFM",
                    help="title lists to merge when --seed titled")
    ap.add_argument("--countries", default="RS,US,IN,DE,BR,ES,FR,RU,PL,NL,GB,UA,AR,TR,IT",
                    help="ISO country codes to seed from when --seed country (all rating levels)")
    ap.add_argument("--from-dir", default=None,
                    help="corpus dir to mine strong opponents from (--seed opponents); defaults to --out")
    ap.add_argument("--max-users", type=int, default=5000)
    ap.add_argument("--months", type=int, default=6, help="recent months per user")
    ap.add_argument("--min-rating", type=int, default=2000, help="both sides must be >= this")
    ap.add_argument("--max-rating", type=int, default=10000, help="both sides must be <= this")
    ap.add_argument("--max-gap", type=int, default=10000,
                    help="max Elo difference between the two players (peer-matched games only)")
    ap.add_argument("--sleep", type=float, default=0.3, help="seconds between API calls")
    args = ap.parse_args()

    if args.seed == "blitz":
        users = seed_blitz_leaderboard(args.max_users)
    elif args.seed == "titled":
        users = seed_titled_multi([t.strip() for t in args.titles.split(",")], args.max_users)
    elif args.seed == "opponents":
        users = seed_opponents_from_dir(args.from_dir or args.out, args.min_rating, args.max_users)
    elif args.seed == "country":
        users = seed_country([c.strip() for c in args.countries.split(",")], args.max_users)
    else:
        users = seed_titled(args.seed, args.max_users)
    print(f"seed={args.seed}: {len(users)} users (showing 8) -> {users[:8]}", file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)
    cctx = zstd.ZstdCompressor(level=10)
    seen = set()
    tot_games = tot_plies = skipped = 0
    ratings = []

    for i, user in enumerate(users, 1):
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", user)
        path = os.path.join(args.out, f"{safe}.pgn.zst")
        if os.path.exists(path):                    # resume: already harvested this player
            skipped += 1
            continue

        blob, ug, up = [], 0, 0
        for g in harvest_user(user, args.months, args.min_rating, args.sleep, seen,
                              max_rating=args.max_rating, max_gap=args.max_gap):
            pgn = g["pgn"].replace('[TimeControl "180"]', '[TimeControl "180+0"]')  # canonicalize
            blob.append(pgn.strip())
            ug += 1
            up += pgn.count("%clk")
            if len(ratings) < 40000:
                ratings += [g["white"]["rating"], g["black"]["rating"]]

        text = ("\n\n".join(blob) + "\n\n") if blob else ""   # empty file = "checked, none" marker
        with open(path, "wb") as fh, cctx.stream_writer(fh) as comp:
            comp.write(text.encode("utf-8"))
        tot_games += ug
        tot_plies += up
        print(f"[{i}/{len(users)}] {user}: {ug} games, {up} pos", file=sys.stderr)
        if i % 50 == 0:
            print(f"  == running totals: {tot_games:,} games, {tot_plies:,} positions, "
                  f"{skipped} resumed-skips ==", file=sys.stderr)

    print("\n=== harvest complete ===")
    print(f"players processed : {len(users) - skipped:,} (+{skipped} skipped/resumed)")
    print(f"unique 3+0 games  : {tot_games:,}")
    print(f"positions (plies) : {tot_plies:,}")
    if ratings:
        ratings.sort()
        print(f"rating spread     : {ratings[0]}-{ratings[-1]} (median {ratings[len(ratings)//2]})")
    print(f"corpus dir        : {args.out}")


if __name__ == "__main__":
    main()
