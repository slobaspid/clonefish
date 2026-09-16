# clonefish engines — play against a player's clone

## Quickest way: play in your browser
Double-click `play_clonefish.bat` in the project folder (or run `python scripts/clonefish_play.py`) and open
http://localhost:5001. Pick a clone, pick your colour, enter your rating, press **New 3+0 game**.
- Real 3+0 clocks. Each side's first move is free (Lichess rule).
- The clone plays the player's moves and openings, and really waits its sampled think time (shown in blue).
- **Either side can lose on time** — the clone flags like the player does (tested: the clone ran out on a short
  test clock after 21 plies; a player who stops moving loses exactly when their clock hits zero). If the side
  with time left can't possibly checkmate, it's a draw instead.
- Resign button; after the game the PGN with clock times appears for download/analysis.
- The clones load in about 20 s after starting. Everything runs on the CPU, so nothing heavy should run alongside.

Each `.bat` here is a real UCI chess engine: a player's fine-tuned clone + their opening book + human-like
think time. Add the `.bat` as an engine in any UCI chess GUI (Cute Chess, Arena, BanksiaGUI, En Croissant,
Nibbler) or point a lichess-bot at it. Built for **3+0 blitz**.

| engine | player | rating it plays at |
|---|---|---|
| `clonefish_VEGETAL.bat` | VEGETAL | ~1840 |
| `clonefish_kpowe52.bat` | kpowe52 | ~1440 |
| `clonefish_OKENITE.bat` | OKENITE | ~1540 |

## What it does each move
1. The player's fine-tuned model scores every legal move.
2. If the position is in the player's own games, their real move counts are blended in (opening book).
3. The move is **sampled** (not always the top move), and a think time is **sampled** from the clone's time
   model and really spent — so it plays fast in book, thinks in hard spots, and can lose on time like a human.

## UCI options
| option | default | meaning |
|---|---|---|
| Elo / OppElo | player's / opponents' average | strength it plays at / strength it expects to face (a GUI sending `UCI_Opponent` sets OppElo) |
| UseBook | true | use the player's opening book |
| BookAlpha | 200 (=2.0) | lower = follow the book more strictly |
| Temperature | 100 | move randomness; lower = cleaner, less human |
| TopP | 90 | ignore the least likely 10% of moves (matched the players' mistake rates best) |
| MimicClock | true | actually spend the sampled think time |
| AllowFlag | true | human-like: may lose on time. Set false for a "never flags" engine |
| MoveOverheadMs | 100 | lag allowance for online play |
| Resign | true if a resign model exists | resign lost positions the way the player does (learned from their games) |
| IdentityPush | 0 (off) | experimental: scale a learned per-player head (`--identity`). 100 = the strength the player's own moves support; higher makes the clone worse, not more distinctive |

### If a clone plays too cleanly (NormPush)
`NormPush` (default 0 = off) pushes the clone away from the stock model: move scores = clone + w*(clone - stock).
Use it **only** as a repair for a clone that is measurably too clean, and never fit the strength:
1. Measure the player's average centipawn loss on their held-out games and the clone's over **>= 80 self-play games,
   with two different seeds** (`scripts/clonefish_push_fit_acpl.py --games 80 --seed N`).
2. If the clone is cleaner than the player by more than ~8 centipawns in BOTH seeds, set `NormPush 50` (w = 0.5).
   Otherwise leave it at 0.
Why so cautious: with 30-80 games the measured gap swings ~7 points on the seed alone, so fitting a per-player
strength is not measurable — one player's fit flipped between 0 and 0.5 with only the seed changed. A fixed 0.5 on a
genuinely too-clean clone reproducibly helped (gap -13 -> -4.7, blunders 3.8 % -> 5.4 % vs the player's 5.8 %).

### Resigning
Each move the engine estimates how likely **this player** would be to resign here, from the model's own
win/loss estimate, the material balance, both clocks and the move number, and resigns with that probability.
The habit is learned per player by `scripts/clonefish_resign_fit.py` (saved as `clones/<player>_resign.json`).
Players differ a lot: VEGETAL resigns about 1 game in 5, OKENITE about 1 in 8, kpowe52 never resigns (plays on
to mate or the flag).
UCI has no resign command, so when the clone resigns it reports `info score cp -9999` and `info string resign`
before its move. In **lichess-bot** set `resign: enabled: true, score: -9000, moves: 1` so it resigns right away;
most GUIs can adjudicate a loss on that score too.

## How well it matches (3 players, 5,000 games each, 150 simulated games per arm; `results/clonefish/PLAN.md`)
Each check compares the clone's games with the player's own. Which slice of their games you compare against matters:
a target taken from only 80 games can move ~9 plies just by resampling, so we compare against their whole history
unless a habit is genuinely changing over time (one of the three players really has been speeding up), in which case
their recent games are the fair comparison.
- **Passes 13 of 15 checks**: how often they resign 3/3 · how often they lose on time 3/3 · game length 3/3 ·
  mistake profile 2/3 · think-time distribution 2/3.
- Think-time distribution is 5-10x closer to the real player than the base model (W1 0.018-0.030 vs 0.13-0.15).
- Average centipawn loss within 1-5 points of the player (+2.0, +1.0, -4.9); Stockfish itself drifts ~2 points
  between identical runs, so treat smaller gaps as noise.
- Known gap: one clone plays a few too many instant moves (11.2 % vs the player's ~8.3 %).

Full write-up — what was built, how it is measured, what was tried and rejected, and what is still open:
`results/clonefish/FINAL.md`.
- Average centipawn loss within 1-5 points of the player (+2.0, +1.0, -4.9); Stockfish itself drifts ~2 points
  between identical runs, so treat smaller gaps as noise.
- Top-3 move match after the opening: +2.4 to +4.1 points over the base model.
- Resigning: learned per player, including *where* they resign (the model reads mate danger, not just material).
- Not yet: other time controls, a blind "is this you?" test with people.

## Make a clone for someone new (one command)
1. Run one command with their username:
   ```
   python scripts/clonefish_build.py --lichess  THEIR_USERNAME
   python scripts/clonefish_build.py --chesscom THEIR_USERNAME
   ```
   It downloads their rated 3+0 games (with clock times) from the site's public API first. If you already have a
   PGN, skip the download with `--pgn their_games.pgn --name THEIR_USERNAME`.
   It detects whether the clocks are Lichess (whole seconds) or chess.com (tenths), fine-tunes on up to their
   5,000 most recent games (~45 min on a GTX 1060 for 5,000 games; run nothing else heavy on this PC), builds
   their opening book, learns their resignation habit, and writes `engines/clonefish_THEIR_USERNAME.bat`.
   Everything is learned from their games; nothing is set by hand per player. Logs go to `results/builds/`.
3. Add the new `.bat` to your chess GUI or lichess-bot.
