"""Stage 1 of the time-head bake-off: freeze the base and cache its features per player.

Every head variant then trains on the SAME frozen features, so any difference between them is
the head's loss/architecture and nothing else.

Two things this does deliberately:

* **Keeps only the crawled account's own positions.** A player file holds that person's games,
  but each game's positions are split between them and their opponent. Training a "player's
  timing" model on both halves would mix hundreds of different humans under one label. The
  crawled account is found as the modal player_id in the file (robust to the filename
  sanitisation the crawler applies).
* **Records game_id and ply.** Positions inside one game are NOT independent - the temporal
  vector carries that player's own last-5 think-times - so any later split must be by game or
  by player, never by position. Storing game_id makes that possible; storing ply makes it
  checkable.

Resumable: one .pt per player, existing files are skipped.

    PYTHONPATH=. python -u scripts/timehead_features.py --out cache/timehead \
        --n-players 60 --games-per-player 150
"""
import argparse
import collections
import glob
import os
import random
import sys
import time

import numpy as np
import torch

from sahformer.download import iter_games_from_zst
from sahformer.records import game_to_records, player_id_of
from sahformer.shards import records_to_arrays
from sahformer.training.loop import load_model

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 256


def modal_player(path, probe_games=40):
    """The crawled account for this file = the player_id appearing in the most games."""
    freq = collections.Counter()
    for i, g in enumerate(iter_games_from_zst(path)):
        for side in ("White", "Black"):
            pid = player_id_of(g.headers.get(side, ""))
            if pid:
                freq[pid] += 1
        if i + 1 >= probe_games:
            break
    return freq.most_common(1)[0][0] if freq else 0


@torch.no_grad()
def features_for_player(model, path, max_games, min_plies=10):
    """Yield (arrays, meta) for the crawled account's own positions in this file."""
    target = modal_player(path)
    if not target:
        return None
    recs, game_ids = [], []
    gi = 0
    for g in iter_games_from_zst(path):
        if gi >= max_games:
            break
        if "abandon" in g.headers.get("Termination", "").lower():
            continue
        mine = [r for r in game_to_records(g) if r.player_id == target]
        if len(mine) < min_plies // 2:
            continue
        recs.extend(mine)
        game_ids.extend([gi] * len(mine))
        gi += 1
    if not recs:
        return None

    arr = records_to_arrays(recs)
    n = len(recs)
    pooled = np.zeros((n, 512), np.float16)
    diff = np.zeros((n, 2), np.float16)
    for s in range(0, n, BATCH):
        e = min(s + BATCH, n)
        batch = {k: torch.from_numpy(np.ascontiguousarray(arr[k][s:e])).float().to(DEV)
                 for k in ("board", "history", "elo_self", "elo_opp", "temporal")}
        with torch.autocast("cuda", enabled=(DEV == "cuda")):
            out = model(batch)
            from sahformer.model.heads import policy_difficulty
            d = policy_difficulty(out["move_logits"])
        pooled[s:e] = out["pooled"].float().cpu().numpy().astype(np.float16)
        diff[s:e] = d.float().cpu().numpy().astype(np.float16)

    return {
        "pooled": pooled,
        "diff": diff,
        "think": arr["think_time"].astype(np.float32),
        # temporal[0] is my_clock/180 -> seconds left at decision time (for clock-masking)
        "my_clock": (arr["temporal"][:, 0] * 180.0).astype(np.float32),
        "elo_self": arr["elo_self"].astype(np.int16),
        "elo_opp": arr["elo_opp"].astype(np.int16),
        "ply": (arr["temporal"][:, 20] * 80.0).astype(np.float32),
        "game_id": np.asarray(game_ids, np.int32),
        "player_id": np.int64(target),
        "games": gi,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["data/chesscom_bands_v2"])
    ap.add_argument("--out", default="cache/timehead")
    ap.add_argument("--ckpt", default="checkpoints/base_300k_best.pt")
    ap.add_argument("--n-players", type=int, default=60)
    ap.add_argument("--games-per-player", type=int, default=150)
    ap.add_argument("--min-games", type=int, default=30, help="skip thin accounts")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = []
    for d in args.dirs:
        files.extend(glob.glob(os.path.join(d, "*.pgn.zst")))
    random.Random(args.seed).shuffle(files)
    print(f"{len(files):,} candidate player files", file=sys.stderr)

    model, _ = load_model(args.ckpt)
    model.to(DEV).eval()

    done = kept = 0
    t0 = time.time()
    for path in files:
        if kept >= args.n_players:
            break
        stem = os.path.basename(path)[:-8]
        outp = os.path.join(args.out, f"{stem}.pt")
        if os.path.exists(outp):
            kept += 1
            continue
        try:
            feat = features_for_player(model, path, args.games_per_player)
        except Exception as e:
            print(f"  skip {stem}: {e}", file=sys.stderr)
            continue
        done += 1
        if feat is None or feat["games"] < args.min_games:
            continue
        torch.save(feat, outp)
        kept += 1
        print(f"[{kept}/{args.n_players}] {stem}: {feat['games']} games, "
              f"{len(feat['think']):,} positions ({time.time()-t0:.0f}s)", file=sys.stderr)

    print(f"\ncached {kept} players into {args.out} (scanned {done})", file=sys.stderr)


if __name__ == "__main__":
    main()
