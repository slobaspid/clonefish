"""Step 0 of the pipeline: a username in, their 3+0 games out as one PGN.

    python scripts/clonefish_fetch.py --site lichess  --user THEIR_USERNAME
    python scripts/clonefish_fetch.py --site chesscom --user THEIR_USERNAME

Writes <out>/<user>.pgn.zst, which `clonefish_build.py` consumes directly. Both sites are read through the
per-user helpers that already exist for corpus harvesting, so the API etiquette is theirs: single-threaded,
descriptive User-Agent, honours HTTP 429 Retry-After, polite sleep between requests. Public data only, read-only.

Why a wrapper and not those scripts directly: their `main()`s are built for building a CORPUS — they seed players
from leaderboards or an Elo band and drop anyone outside it. For one named person those filters have to be off, or
their games get silently discarded.

clonefish needs 3+0 games WITH clock times, so both paths keep only TimeControl 180+0 and report how many of the
games actually carry [%clk] comments.
"""
import argparse, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import zstandard as zstd


def fetch_lichess(user, games, max_pull):
    from lichess_api_harvest import blitz_rating, pull_3plus0
    rating, n_blitz = blitz_rating(user)
    if rating is None:
        print(f"  could not read a blitz profile for '{user}' — check the spelling", flush=True)
    else:
        print(f"  lichess profile: blitz {rating}, {n_blitz} blitz games played", flush=True)
    print(f"  streaming up to {max_pull} blitz games, keeping 3+0 (this is the slow part)", flush=True)
    text, kept = pull_3plus0(user, games, max_pull)
    return (text or ""), kept


def fetch_chesscom(user, games, months):
    from chesscom_sieve import harvest_user
    blob, kept = [], 0
    # the rating/gap filters exist for corpus building; for one named player they must not exclude anything
    for g in harvest_user(user, months, 0, 0.3, set(), max_rating=10000, max_gap=10000):
        blob.append(g["pgn"].replace('[TimeControl "180"]', '[TimeControl "180+0"]').strip())
        kept += 1
        if kept >= games:
            break
    return (("\n\n".join(blob) + "\n\n") if blob else ""), kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, choices=["lichess", "chesscom"])
    ap.add_argument("--user", required=True)
    ap.add_argument("--games", type=int, default=5000, help="most recent 3+0 games to keep")
    ap.add_argument("--months", type=int, default=24, help="chess.com: how many monthly archives back to read")
    ap.add_argument("--max-pull", type=int, default=0,
                    help="lichess: blitz games to stream before filtering to 3+0 (default 3x --games)")
    ap.add_argument("--out", default=os.path.join("data", "clonefish_in"))
    ap.add_argument("--force", action="store_true", help="re-download even if the file exists")
    a = ap.parse_args()

    out_dir = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    os.makedirs(out_dir, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", a.user)
    path = os.path.join(out_dir, f"{safe}.pgn.zst")
    if os.path.exists(path) and not a.force:
        print(f"{path} already exists (use --force to re-download)")
        print(f"PGN: {path}")
        return

    print(f"fetching {a.user} from {a.site}", flush=True)
    if a.site == "lichess":
        text, kept = fetch_lichess(a.user, a.games, a.max_pull or max(3 * a.games, 3000))
    else:
        text, kept = fetch_chesscom(a.user, a.games, a.months)

    if kept == 0:
        raise SystemExit(f"no rated 3+0 games found for '{a.user}' on {a.site}. Check the username, or the player "
                         f"may not play 3+0 blitz — clonefish needs 3+0 games with clock times.")
    with_clk = text.count("%clk")
    with open(path, "wb") as fh, zstd.ZstdCompressor(level=10).stream_writer(fh) as comp:
        comp.write(text.encode("utf-8"))
    print(f"  {kept} games, {with_clk} clock stamps -> {os.path.relpath(path, ROOT)}", flush=True)
    if with_clk == 0:
        raise SystemExit("those games carry no [%clk] comments, so think-times cannot be learned from them.")
    print(f"PGN: {path}")


if __name__ == "__main__":
    main()
