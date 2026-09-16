"""One command: a username in, their clone engine out.

    python scripts/clonefish_build.py --lichess  their_username
    python scripts/clonefish_build.py --chesscom their_username
    python scripts/clonefish_build.py --pgn their_games.pgn --name their_username   # a PGN you already have

Steps (each skipped if its output already exists, unless --force):
  0. download their rated 3+0 games with clocks from the site's public API (skipped when --pgn is given)
  1. detect clock precision from the PGN: Lichess = whole seconds, chess.com = tenths
  2. fine-tune the base model on up to --games of their games (whole-second time loss for Lichess clocks)
  3. build their opening book (cached next to the clone)
  4. learn their resignation habit (a flat near-zero rate if they never resign)
  5. write engines/clonefish_<name>.bat — a UCI engine for any GUI or lichess-bot
Everything is learned from their games only; nothing is hand-set per player. Built for 3+0 games with clocks.
Needs a GPU for step 2 (~45 min for 5,000 games on a GTX 1060); run nothing else heavy alongside.
"""
import argparse, os, re, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from finetune_clone import open_pgn

PY = sys.executable
CLK = re.compile(r"\[%clk (\d+):(\d+):(\d+(?:\.\d+)?)\]")


def clock_precision(pgn, chars=2_000_000):
    txt = open_pgn(pgn).read(chars)
    vals = [m.group(3) for m in CLK.finditer(txt)]
    if not vals:
        raise SystemExit("no [%clk] clock comments found: clonefish needs games with clock times")
    frac = sum(1 for v in vals if "." in v and float(v) % 1 != 0) / len(vals)
    return ("exact" if frac > 0.05 else "whole"), len(vals)


def run(cmd, log):
    print("  $ " + " ".join(cmd), flush=True)
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONPATH=ROOT))
    if r.returncode != 0:
        raise SystemExit(f"step failed (exit {r.returncode}), see {log}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lichess", help="their Lichess username — downloads their 3+0 games first")
    ap.add_argument("--chesscom", help="their chess.com username — downloads their 3+0 games first")
    ap.add_argument("--pgn", help="or give a PGN you already have (skips the download)")
    ap.add_argument("--name", help="their username as it appears in the PGN (default: the --lichess/--chesscom name)")
    ap.add_argument("--games", type=int, default=5000, help="most recent games to fine-tune on")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--months", type=int, default=24, help="chess.com: how many monthly archives back to read")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if sum(bool(x) for x in (a.lichess, a.chesscom, a.pgn)) != 1:
        raise SystemExit("give exactly one of --lichess USER, --chesscom USER, or --pgn FILE")

    site, user = ("lichess", a.lichess) if a.lichess else (("chesscom", a.chesscom) if a.chesscom else (None, None))
    if site:
        name = a.name or user
        print(f"[0/5] download {user}'s 3+0 games from {site}")
        cmd = [PY, "scripts/clonefish_fetch.py", "--site", site, "--user", user, "--games", str(a.games)]
        if site == "chesscom":
            cmd += ["--months", str(a.months)]
        if a.force:
            cmd += ["--force"]
        log = os.path.join(ROOT, "results", "builds", f"{name}_fetch.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        run(cmd, log)
        pgn = next((l.split("PGN: ", 1)[1].strip() for l in reversed(open(log, encoding="utf-8").read().splitlines())
                    if l.startswith("PGN: ")), None)
        if not pgn or not os.path.exists(pgn):
            raise SystemExit(f"download did not produce a PGN, see {log}")
        print(f"      -> {os.path.relpath(pgn, ROOT)}")
    else:
        pgn = os.path.abspath(a.pgn)
        name = a.name or os.path.basename(pgn).split(".")[0]
    clones, engines, logs = (os.path.join(ROOT, d) for d in ("clones", "engines", os.path.join("results", "builds")))
    for d in (clones, engines, logs): os.makedirs(d, exist_ok=True)

    clock, n = clock_precision(pgn)
    print(f"[1/5] clock precision: {clock} ({'Lichess-style whole seconds' if clock == 'whole' else 'chess.com-style tenths'}; {n} clock readings sampled)")

    ckpt = os.path.join(clones, f"{name}.pt")
    print(f"[2/5] fine-tune -> {ckpt}")
    if a.force or not os.path.exists(ckpt):
        run([PY, "scripts/finetune_clone.py", "--pgn", pgn, "--name", name, "--test-games", "0",
             "--max-train-games", str(a.games), "--epochs", str(a.epochs),
             "--time-loss", "bucket" if clock == "whole" else "point", "--save-ft", ckpt],
            os.path.join(logs, f"{name}_finetune.log"))

    print("[3/5] opening book")
    from clonefish_uci import player_data
    data = player_data(pgn, name, 0)
    print(f"      {data['games']} games, {len(data['counts'])} positions, plays at ~{data['elo']}, opponents ~{data['opp_elo']}")

    rj = os.path.join(clones, f"{name}_resign.json")
    print(f"[4/5] resignation habit -> {rj}")
    if a.force or not os.path.exists(rj):
        run([PY, "scripts/clonefish_resign_fit.py", "--player", name, "--clone", ckpt, "--pgn", pgn, "--exclude-recent", "0"],
            os.path.join(logs, f"{name}_resign.log"))

    bat = os.path.join(engines, f"clonefish_{name}.bat")
    print(f"[5/5] engine launcher -> {bat}")
    with open(bat, "w", encoding="utf-8") as f:
        f.write("@echo off\r\n"
                f"REM clonefish UCI engine: {name}'s clone. Add this file as a UCI engine in any chess GUI or lichess-bot.\r\n"
                "set ROOT=%~dp0..\r\nset PYTHONPATH=%ROOT%\r\n"
                f"\"{PY}\" \"%ROOT%\\scripts\\clonefish_uci.py\" --player {name} --pgn \"{pgn}\" "
                f"--clone \"%ROOT%\\clones\\{name}.pt\" --clock {clock}\r\n")
    print(f"done: add {bat} to your chess GUI or lichess-bot (resign: score -9000, moves 1)")


if __name__ == "__main__":
    main()
