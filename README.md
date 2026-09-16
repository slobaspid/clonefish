# sah-transformer — play against a clone of yourself

Give it a Lichess or chess.com username and it produces a **chess engine that plays like that person** — their
moves, how long they think, when they resign, when they lose on time.

```bash
python scripts/clonefish_build.py --lichess  THEIR_USERNAME
python scripts/clonefish_build.py --chesscom THEIR_USERNAME
```

That downloads their rated 3+0 games with clock times from the site's public API, fine-tunes the base model on up to
5,000 of them, builds their opening book, learns their resignation habit, and writes
`engines/clonefish_<name>.bat` — a UCI engine you can add to any chess GUI (Cute Chess, Arena, Nibbler, En Croissant)
or point a lichess-bot at. Nothing is set by hand per player; everything is learned from their games.

Then either add that `.bat` to your GUI, or double-click `play_clonefish.bat` to play them in your browser.

## How well does it actually match?

Measured against each player's own held-out games — games the clone never trained on — across 150 simulated games
per player. It passes **13 of 15** whole-game checks:

| what | result |
|---|---|
| think-time distribution | W1 **0.018–0.030** vs the base model's 0.13–0.15 (5–10× closer) |
| mistakes (Stockfish) | average centipawn loss within **+2.0 / +1.0 / −4.9** of the player |
| move match | top-3 **85.5 / 85.7 / 88.8 %** vs a base model at 80.1 % |
| resigning, losing on time, game length | matches on all three players |

It holds up on players never used while developing it. The full write-up, including the two checks it misses and
why, is in **[`results/clonefish/FINAL.md`](results/clonefish/FINAL.md)** — along with a list of things that were
tried and *rejected*, so nobody re-treads them.

Scope: **3+0 blitz**, and it needs games that have clock times in them.

## The model underneath

Not an engine that chases the best move — a human-imitation model where realism beats strength. A Maia-3-style
board transformer (re-implemented from their papers, our own weights) with three additions of our own:

- a **clock-aware layer** that reads the remaining time and modulates play,
- an **MDN think-time head** that predicts *how long a human would think*, as a mixture of log-normals,
- a difficulty-into-timing signal.

A clone is then a fine-tune of that base on one person's games. `PROJECT_STATUS.md` covers the training side.

## Setup

```bash
pip install -r requirements.txt     # python-chess, numpy, zstandard, torch 2.4
```

A GPU is needed to *build* a clone (~45 min for 5,000 games on a GTX 1060; don't run other heavy jobs alongside).
Playing one needs only a CPU. **Stockfish is needed only by the evaluation harness**, not for building or playing —
see FINAL.md for where to put it.

Model weights, encoded caches, game data and the Stockfish binary are not in git (several GB); the code that
produces them is.

## Layout

| path | what |
|---|---|
| `scripts/clonefish_build.py` | the one command: username → engine |
| `scripts/clonefish_fetch.py` | downloads a player's 3+0 games (public API, rate-limit friendly) |
| `scripts/clonefish_uci.py` | the engine itself |
| `scripts/clonefish_eval.py` | the evaluation harness behind every number above |
| `results/clonefish/FINAL.md` | **start here** — results, what was rejected, what's still open |
| `sahformer/` | the model: trunk, heads, encoding, training loop |
| `engines/` | generated launchers + how to use them |
