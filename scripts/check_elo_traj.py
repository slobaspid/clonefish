"""Chronological Elo trajectory per harvested player: do they climb/drop over their 1k games?
Tells us whether the data supports the Elo-adaptability feature."""
import os, sys, io, glob, re
import zstandard, numpy as np

DIR = sys.argv[1] if len(sys.argv) > 1 else "data/lichess_1k"


def stem(f): return os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0]


def games_elos(path, me):
    """Return list of (utc_datetime_str, my_elo) in file order."""
    me = me.lower()
    text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(path, "rb")),
                            encoding="utf-8", errors="ignore")
    out = []
    w = b = we = be = date = time = None
    for line in text:
        if line.startswith("[White "): w = line.split('"')[1].lower()
        elif line.startswith("[Black "): b = line.split('"')[1].lower()
        elif line.startswith("[WhiteElo "): we = line.split('"')[1]
        elif line.startswith("[BlackElo "): be = line.split('"')[1]
        elif line.startswith("[UTCDate "): date = line.split('"')[1]
        elif line.startswith("[UTCTime "): time = line.split('"')[1]
        elif line.startswith("1.") or line.startswith("1 "):     # movetext = end of a game's headers
            if w is not None:
                elo = we if me == w else (be if me == b else None)
                try:
                    elo = int(elo)
                    out.append((f"{date} {time}", elo))
                except (TypeError, ValueError):
                    pass
            w = b = we = be = date = time = None
    return out


rows = []
for f in sorted(glob.glob(f"{DIR}/*.pgn.zst")):
    me = stem(f)
    g = games_elos(f, me)
    if len(g) < 50:
        continue
    g.sort(key=lambda x: x[0])                                    # chronological
    elos = np.array([e for _, e in g])
    # smooth first/last with 50-game windows to avoid single-game noise
    first = elos[:50].mean(); last = elos[-50:].mean()
    rows.append((me, len(elos), int(elos.min()), int(elos.max()), int(elos.max() - elos.min()),
                 int(round(last - first)), round(elos.std(), 0)))

rows.sort(key=lambda r: -abs(r[5]))
print(f"{'player':22s}{'n':>6}{'min':>6}{'max':>6}{'range':>7}{'drift(last-first)':>18}{'std':>6}")
for r in rows:
    print(f"{r[0]:22s}{r[1]:>6}{r[2]:>6}{r[3]:>6}{r[4]:>7}{r[5]:>18}{int(r[6]):>6}")

if rows:
    drift = np.array([abs(r[5]) for r in rows]); rng = np.array([r[4] for r in rows])
    print(f"\n{len(rows)} players | drift>=150: {(drift>=150).sum()}  drift>=300: {(drift>=300).sum()}  "
          f"| range>=300: {(rng>=300).sum()}  range>=500: {(rng>=500).sum()}")
    print(f"median |drift| {int(np.median(drift))}  median range {int(np.median(rng))}")
