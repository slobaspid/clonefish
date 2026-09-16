"""Harvest ~1000+ 3+0 blitz games per player via the Lichess public API, filtered to an Elo band.

The lichess_scale filenames are real usernames -> reuse them as candidates. For each: check blitz
rating (Elo band), then stream their blitz games (moves+clocks), keep only 180+0, cap, save.

    PYTHONPATH=. python scripts/lichess_api_harvest.py --out data/lichess_1k \
        --n-players 100 --min-games 1000 --elo-min 1500 --elo-max 1900

API-polite: sequential, User-Agent, honors 429 Retry-After. Resumable (skips existing outputs).
"""
import argparse, os, sys, io, time, glob, json, urllib.request, urllib.error

UA = "sahformer-research (personal-style clone study; contact via lichess)"
def stem(f): return os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0]


def _get(url, stream=False, tries=5):
    for t in range(tries):
        req = urllib.request.Request(url, headers={"User-Agent": UA,
                                                   "Accept": "application/x-ndjson" if "api/user" in url else "application/x-chess-pgn"})
        try:
            r = urllib.request.urlopen(req, timeout=120)
            return r if stream else r.read().decode("utf-8", "ignore")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = int(e.headers.get("Retry-After", 60)) + 5
                print(f"    429 rate-limited, sleeping {wait}s", flush=True); time.sleep(wait)
            else:
                print(f"    HTTP {e.code} on {url[:80]}", flush=True); return None
        except Exception as ex:
            print(f"    err {ex}; retry", flush=True); time.sleep(10)
    return None


def blitz_rating(user):
    txt = _get(f"https://lichess.org/api/user/{user}")
    if not txt:
        return None, None
    try:
        j = json.loads(txt)
        b = j.get("perfs", {}).get("blitz", {})
        return b.get("rating"), b.get("games")
    except Exception:
        return None, None


def pull_3plus0(user, cap_games, max_pull):
    """Stream the user's blitz games, keep TimeControl 180+0, return a PGN string of up to cap_games."""
    url = (f"https://lichess.org/api/games/user/{user}?perfType=blitz&rated=true"
           f"&moves=true&clocks=true&tags=true&max={max_pull}")
    r = _get(url, stream=True)
    if r is None:
        return None, 0
    kept = []; buf = []; is_180 = False; n_kept = 0
    for raw in r:
        line = raw.decode("utf-8", "ignore")
        if line.startswith("[Event ") and buf:
            if is_180:
                kept.append("".join(buf)); n_kept += 1
                if n_kept >= cap_games:
                    break
            buf = []; is_180 = False
        buf.append(line)
        if line.startswith('[TimeControl "180+0"'):
            is_180 = True
    if is_180 and buf and n_kept < cap_games:
        kept.append("".join(buf)); n_kept += 1
    return "".join(kept), n_kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/lichess_1k")
    ap.add_argument("--candidates", default="data/lichess_scale", help="dir of candidate usernames")
    ap.add_argument("--n-players", type=int, default=100)
    ap.add_argument("--min-games", type=int, default=1000, help="keep player only if >= this many 3+0 games")
    ap.add_argument("--cap-games", type=int, default=1200, help="max 3+0 games to save per player")
    ap.add_argument("--max-pull", type=int, default=2500, help="max blitz games to stream per player")
    ap.add_argument("--elo-min", type=int, default=1500)
    ap.add_argument("--elo-max", type=int, default=1900)
    ap.add_argument("--sleep", type=float, default=2.0, help="polite delay between players (s)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import zstandard

    cands = [stem(f) for f in sorted(glob.glob(f"{args.candidates}/*.pgn.zst"), key=os.path.getsize, reverse=True)]
    have = {stem(f) for f in glob.glob(f"{args.out}/*.pgn.zst")}
    kept = len(have)
    print(f"{len(cands)} candidates | already have {kept} | target {args.n_players} "
          f"| band {args.elo_min}-{args.elo_max} | >= {args.min_games} 3+0 games", flush=True)
    t0 = time.time()
    for i, user in enumerate(cands):
        if kept >= args.n_players:
            break
        if user in have:
            continue
        rating, ngames = blitz_rating(user)
        time.sleep(args.sleep)
        if rating is None or not (args.elo_min <= rating <= args.elo_max):
            print(f"[{i}] {user:22s} blitz {rating} -> skip (band)", flush=True); continue
        pgn, n = pull_3plus0(user, args.cap_games, args.max_pull)
        time.sleep(args.sleep)
        if pgn is None or n < args.min_games:
            print(f"[{i}] {user:22s} blitz {rating} -> {n} 3+0 games (< {args.min_games}, skip)", flush=True); continue
        out = f"{args.out}/{user}.pgn.zst"
        with open(out, "wb") as fh:
            fh.write(zstandard.ZstdCompressor().compress(pgn.encode("utf-8")))
        kept += 1
        print(f"[{i}] {user:22s} blitz {rating} -> SAVED {n} 3+0 games  "
              f"({kept}/{args.n_players}, {time.time()-t0:.0f}s)", flush=True)
    print(f"\ndone: {kept} players in {args.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
