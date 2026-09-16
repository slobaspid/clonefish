"""Build a LARGE, SLIM cache from a corpus dir (many players), for the scale experiments.

Single-corpus (same-month split):
    PYTHONPATH=. python scripts/build_cache_scale.py --corpus data/lichess_scale \
        --out cache_scale.pt --n-ref 60 --n-test 40 --max-players 3000

Cross-month split (HONEST — reference and test from different time windows, same players):
    PYTHONPATH=. python scripts/build_cache_scale.py \
        --ref-corpus data/lichess_scale --test-corpus data/lichess_2017_09 \
        --out cache_xmonth.pt --n-ref 60 --n-test 40 --max-players 3000

SLIM = stores only pooled features + think + player/game/split ids (what run_scale.py needs).
"""
import argparse
import os
import sys
import io
import time
import glob
import random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import zstandard
import chess
import chess.pgn

from sahformer.encoding import encode_board, build_temporal
from sahformer.records import _stack_history, BASE_SECONDS
from multiprocessing import Pool


def extract_games(path, me, cap):
    """Return up to `cap` games (each a list of the player's own ply-rows) from one file."""
    me = me.lower()
    games = []
    with open(path, "rb") as fh:
        text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh),
                                encoding="utf-8", errors="ignore")
        while True:
            g = chess.pgn.read_game(text)
            if g is None:
                break
            w = (g.headers.get("White", "") or "").lower()
            b = (g.headers.get("Black", "") or "").lower()
            if me == w:
                me_white = True
            elif me == b:
                me_white = False
            else:
                continue
            we = int(g.headers.get("WhiteElo", 0) or 0)
            be = int(g.headers.get("BlackElo", 0) or 0)
            board = g.board()
            prev = {chess.WHITE: BASE_SECONDS, chess.BLACK: BASE_SECONDS}
            thist = {chess.WHITE: [], chess.BLACK: []}
            ph, node, ply, rows = [], g, 0, []
            while node.variations:
                node = node.variation(0)
                mv = node.move
                mover = board.turn
                ca = node.clock()
                if ca is None:
                    board.push(mv); continue
                think = max(prev[mover] - ca, 0.0)
                cur = encode_board(board)
                if (mover == chess.WHITE) == me_white:
                    h = _stack_history(ph, cur)
                    rows.append((cur, h,
                                 build_temporal(my_clock=prev[mover], opp_clock=prev[not mover],
                                                own_think_history=thist[mover], ply=ply),
                                 (we if me_white else be), (be if me_white else we), think))
                prev[mover] = ca
                thist[mover] = [think] + thist[mover]
                ph.append(cur); board.push(mv); ply += 1
            if len(rows) >= 8:
                games.append(rows)
            if len(games) >= cap:
                break
    return games


def parse_player(args):           # single-corpus: random split later
    path, me, need = args
    games = extract_games(path, me, need)
    return (me, games if len(games) >= need else None)


def parse_xmonth(args):           # cross-month: ref games + test games from two files
    ref_path, test_path, me, n_ref, n_test = args
    rg = extract_games(ref_path, me, n_ref)
    if len(rg) < n_ref:
        return (me, None)
    tg = extract_games(test_path, me, n_test)
    if len(tg) < n_test:
        return (me, None)
    return (me, (rg, tg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", help="single corpus dir (same-month random split)")
    ap.add_argument("--ref-corpus", help="reference-window corpus (cross-month)")
    ap.add_argument("--test-corpus", help="test-window corpus (cross-month)")
    ap.add_argument("--out", default="cache_scale.pt")
    ap.add_argument("--n-ref", type=int, default=60)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--max-players", type=int, default=3000)
    ap.add_argument("--chunk-players", type=int, default=100)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--chrono", action="store_true",
                    help="single-corpus: split by TIME (older games = reference, newer = test) "
                         "instead of a random shuffle. Files are stored newest-first.")
    args = ap.parse_args()
    xmonth = bool(args.ref_corpus and args.test_corpus)
    need = args.n_ref + args.n_test

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, "|", torch.cuda.get_device_name(0) if dev == "cuda" else "CPU",
          "| mode:", "CROSS-MONTH" if xmonth else "same-month")

    # build the job list
    def stem(f):
        return os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0]
    if xmonth:
        ref = {stem(f): f for f in glob.glob(f"{args.ref_corpus}/*.pgn.zst")}
        tst = {stem(f): f for f in glob.glob(f"{args.test_corpus}/*.pgn.zst")}
        common = sorted(set(ref) & set(tst), key=lambda u: -os.path.getsize(ref[u]))
        print(f"ref {len(ref)} players, test {len(tst)}, common {len(common)}; "
              f"need {args.n_ref} ref + {args.n_test} test each")
        jobs = [(ref[u], tst[u], u, args.n_ref, args.n_test) for u in common]
        worker = parse_xmonth
    else:
        files = sorted(glob.glob(f"{args.corpus}/*.pgn.zst"), key=os.path.getsize, reverse=True)
        print(f"{len(files)} player files; need >= {need} games")
        jobs = [(f, stem(f), need) for f in files]
        worker = parse_player

    from sahformer.training.loop import load_model
    ckpt = os.environ.get("BASE_CKPT", "checkpoints/base_300k_best.pt")
    if not os.path.exists(ckpt):
        ckpt = "kaggle_bundle/best.pt"
    model, mcfg = load_model(ckpt); model.eval().to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    print("base:", ckpt)

    @torch.no_grad()
    def base_pooled(board, hist, temporal, es, eo):
        tok = model.input_emb(board.float(), hist.float(), es, eo)
        t = model.temporal_enc(temporal)
        film = model.film_gen(t) if model.use_film else None
        enc = model.encoder(tok, t=(t if model.use_time_gab else None), film=film)
        return enc.mean(1) + model.t_to_d(t)

    cP, cTh, cSp, cGm, cPid = [], [], [], [], []
    names = []
    pid = 0
    span = need + 1
    t0 = time.time()

    def flush(name, ref_rows, test_rows):
        nonlocal pid
        recs = []
        for split, gs in ((0, ref_rows), (1, test_rows)):
            for local, rows in enumerate(gs):
                gidx = local if split == 0 else args.n_ref + local
                guid = pid * span + gidx
                for r in rows:
                    recs.append((*r, split, guid))
        for s in range(0, len(recs), args.batch):
            batch = recs[s:s + args.batch]
            board = torch.from_numpy(np.stack([r[0] for r in batch])).to(dev)
            hist = torch.from_numpy(np.stack([r[1] for r in batch])).to(dev)
            temp = torch.from_numpy(np.stack([r[2] for r in batch])).float().to(dev)
            es = torch.tensor([r[3] for r in batch]).to(dev)
            eo = torch.tensor([r[4] for r in batch]).to(dev)
            pooled_b = base_pooled(board, hist, temp, es, eo)
            cP.append(pooled_b.half().cpu())
            cTh.append(torch.tensor([r[5] for r in batch]))
            cSp.append(torch.tensor([r[6] for r in batch]))
            cGm.append(torch.tensor([r[7] for r in batch]))
            cPid.append(torch.full((len(batch),), pid))
        names.append(name)
        pid += 1

    for cs in range(0, len(jobs), args.chunk_players):
        chunk = jobs[cs:cs + args.chunk_players]
        with Pool(args.workers) as pool:
            for name, res in pool.imap_unordered(worker, chunk):
                if res is None:
                    continue
                if xmonth:
                    flush(name, res[0], res[1])
                else:
                    gl = res[:need]
                    if args.chrono:
                        gl = gl[::-1]                     # newest-first on disk -> oldest-first
                    else:
                        random.Random(pid).shuffle(gl)
                    flush(name, gl[:args.n_ref], gl[args.n_ref:need])
                if pid >= args.max_players:
                    break
        gib = sum(t.numel() * t.element_size() for t in cP) / 1e9
        print(f"  accepted {pid} players | scanned {min(cs+args.chunk_players, len(jobs))} | "
              f"pooled {gib:.1f} GB | {time.time()-t0:.0f}s")
        if pid >= args.max_players:
            break

    n_rows = sum(t.shape[0] for t in cP)
    pooled = torch.empty((n_rows, cP[0].shape[1]), dtype=cP[0].dtype)
    off = 0
    while cP:                                   # copy-and-release: peak = final + one chunk
        t = cP.pop(0)
        pooled[off:off + t.shape[0]] = t
        off += t.shape[0]
    think = torch.cat(cTh).float()
    test = torch.cat(cSp).bool(); game = torch.cat(cGm).long(); pidt = torch.cat(cPid).long()
    torch.save({"pooled": pooled, "think": think, "pid": pidt, "test": test, "game": game,
                "names": names}, args.out)
    sz = os.path.getsize(args.out) / 1e9
    print(f"\nsaved {args.out}: {pooled.shape[0]} plies, {pid} players, {sz:.1f} GB on disk")


if __name__ == "__main__":
    main()
