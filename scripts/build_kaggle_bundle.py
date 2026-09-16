"""Assemble the Kaggle upload bundle: the 90 chosen players' games + the base model
+ the sahformer code, into one folder to zip and upload as a Kaggle Dataset.

    python scripts/build_kaggle_bundle.py
Produces:  kaggle_bundle/{sahformer/, best.pt, players/*.pgn.zst}
"""
import glob, os, io, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import zstandard, chess.pgn
import numpy as np

CORPUS = "data/chesscom_corpus"
MODEL = "checkpoints/base_300k_best.pt"
BAND = (2397, 2547)
MIN_GAMES = 1000
TARGET = 90
OUT = "kaggle_bundle"

def scan_file(f):
    stem = os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0].lower()
    n = 0; ratings = []
    try:
        with open(f, "rb") as fh:
            text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh),
                                    encoding="utf-8", errors="ignore")
            while True:
                h = chess.pgn.read_headers(text)
                if h is None: break
                n += 1
                w = (h.get("White","") or "").lower(); b = (h.get("Black","") or "").lower()
                if stem == w: ratings.append(int(h.get("WhiteElo",0) or 0))
                elif stem == b: ratings.append(int(h.get("BlackElo",0) or 0))
    except Exception:
        return None
    rr = [x for x in ratings if x > 0]
    return (f, n, int(np.median(rr)) if rr else 0)

def main():
    files = sorted(glob.glob(os.path.join(CORPUS, "*.pgn.zst")))
    print(f"scanning {len(files)} files to select {TARGET} players in {BAND} with >={MIN_GAMES} games...")
    chosen = []
    for i, f in enumerate(files):
        r = scan_file(f)
        if r and r[1] >= MIN_GAMES and BAND[0] <= r[2] <= BAND[1]:
            chosen.append(r)
        if len(chosen) >= TARGET: break
        if (i+1) % 500 == 0: print(f"  ...{i+1} scanned, {len(chosen)} found")
    print(f"selected {len(chosen)} players (ratings {min(c[2] for c in chosen)}-{max(c[2] for c in chosen)})")

    if os.path.exists(OUT): shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "players"))
    for f, n, rt in chosen:
        shutil.copy(f, os.path.join(OUT, "players", os.path.basename(f)))
    shutil.copy(MODEL, os.path.join(OUT, "best.pt"))
    shutil.copytree("sahformer", os.path.join(OUT, "sahformer"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    sz = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fs in os.walk(OUT) for fn in fs)
    print(f"\nbundle ready at ./{OUT}/  ({sz/1e6:.0f} MB)")
    print(f"  players/: {len(chosen)} files   best.pt: {os.path.getsize(MODEL)/1e6:.0f} MB   sahformer/: code")
    print("\nNext: zip the folder and upload it to Kaggle as a new Dataset.")

if __name__ == "__main__":
    main()
