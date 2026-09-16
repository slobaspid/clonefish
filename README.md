# clonefish

Turns someone's online chess history into a chess engine that plays like them — their moves, how long they think,
when they resign, when they lose on time. Built for **3+0 blitz**.

## Build a clone

```bash
python scripts/clonefish_build.py --lichess  THEIR_USERNAME
python scripts/clonefish_build.py --chesscom THEIR_USERNAME
```

Downloads their rated 3+0 games from the site's public API, fine-tunes the base model on up to 5,000 of them, builds
their opening book, learns their resignation habit, and writes `engines/clonefish_<name>.bat`.

Takes about 45 minutes for 5,000 games on a GTX 1060. Don't run other heavy jobs while it works.

Already have a PGN? Skip the download:

```bash
python scripts/clonefish_build.py --pgn their_games.pgn --name THEIR_USERNAME
```

Useful flags: `--games N` (default 5000), `--months N` (chess.com, how far back to read), `--force` (rebuild from
scratch).

## Play against it

- **In your browser** — double-click `play_clonefish.bat`, then open http://localhost:5001. Real 3+0 clocks, either
  side can lose on time, PGN with clock times at the end.
- **In a chess GUI** — add `engines/clonefish_<name>.bat` as a UCI engine (Cute Chess, Arena, Nibbler,
  En Croissant).
- **On Lichess** — point a lichess-bot at the same `.bat`. Set `resign: enabled: true, score: -9000, moves: 1` so it
  resigns when the clone decides to; UCI has no resign command, so it reports a hopeless score instead.

## What you need

```bash
pip install -r requirements.txt     # python-chess, numpy, zstandard, torch 2.4
```

- A **GPU** to build a clone. Playing one only needs a CPU.
- **The base model**, `checkpoints/base_300k_best.pt` (234 MB). Every clone is a fine-tune of it, and it is **not in
  this repo and not published** — so you need it from the author, or you train it yourself with `scripts/train.py`
  (see `PROJECT_STATUS.md`). Nothing else here will run without it.
- **Stockfish** only if you want to run the evaluation harness — not needed to build or play. Path is set at the top
  of `scripts/clonefish_eval.py`.

## Engine options

Set these in your GUI like any other UCI option.

| option | default | what it does |
|---|---|---|
| `Elo` / `OppElo` | the player's own / their usual opponents' | strength it plays at, and what it expects to face |
| `UseBook` | true | use the player's opening book — also where their opening think-times come from |
| `Temperature` | 100 | move randomness; lower is cleaner and less human |
| `TopP` | 90 | ignore the least likely 10 % of moves |
| `MimicClock` | true | actually spend the think-time it sampled |
| `AllowFlag` | true | may lose on time, like a person. Set false for an engine that never flags |
| `Resign` | on if a resign model exists | resign the way that player does |
| `MoveOverheadMs` | 100 | lag allowance for online play |

`NormPush`, `IdentityPush`, `PaceSigma` and `ResignBias` exist but default to 0 — all were tested and made the clone
worse. Leave them alone.

## Limits

- 3+0 blitz only.
- Needs games that carry clock times (Lichess and chess.com both do).
- Clones a person's *style*, not a strength dial — it plays at roughly their level because that's how they play.
- One player per clone.

## More

`results/clonefish/FINAL.md` — how closely clones match their players, and a list of approaches that were tried and
didn't work, so you don't repeat them. `PROJECT_STATUS.md` covers the underlying model.
