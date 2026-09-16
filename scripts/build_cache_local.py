"""Build cache_film.pt locally from kaggle_bundle (90 players + best.pt), on the GPU.

    PYTHONPATH=. python scripts/build_cache_local.py

Same cache format as the Kaggle notebooks: per-ply base features (pooled, legal-move
logits, think-time, player/game/split ids). Run once; the GE2E / fingerprint notebooks
then load it and skip parsing.
"""
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
from multiprocessing import Pool

from sahformer.encoding import encode_board, encode_move, build_temporal
from sahformer.records import _stack_history, BASE_SECONDS
from sahformer.model.heads import move_to_index

BUNDLE = "kaggle_bundle"
CACHE = "cache_film.pt"
N_REF, N_TEST = 150, 40
MAXLEG = 72


def player_plies(path, me):
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
                    board.push(mv)
                    continue
                think = max(prev[mover] - ca, 0.0)
                cur = encode_board(board)
                if (mover == chess.WHITE) == me_white:
                    h = _stack_history(ph, cur)
                    frm, to, pr = encode_move(board, mv)
                    legal = [move_to_index(*encode_move(board, m)) for m in board.legal_moves]
                    rows.append((cur, h,
                                 build_temporal(my_clock=prev[mover], opp_clock=prev[not mover],
                                                own_think_history=thist[mover], ply=ply),
                                 (we if me_white else be), (be if me_white else we),
                                 move_to_index(frm, to, pr), legal, think))
                prev[mover] = ca
                thist[mover] = [think] + thist[mover]
                ph.append(cur)
                board.push(mv)
                ply += 1
            if len(rows) >= 8:
                yield rows


def extract_worker(args):
    path, me, pid, nref, ntest = args
    rnd = random.Random(pid)
    games = []
    for rows in player_plies(path, me):
        games.append(rows)
        if len(games) >= nref + ntest:
            break
    rnd.shuffle(games)
    out = []
    span = nref + ntest + 1
    for split, gs in ((0, games[:nref]), (1, games[nref:nref + ntest])):
        for local, rows in enumerate(gs):
            gidx = local if split == 0 else nref + local
            guid = pid * span + gidx
            for r in rows:
                out.append((*r, split, guid, pid))
    return out


def main():
    if os.path.exists(CACHE):
        print(f"{CACHE} already exists — delete it to rebuild.")
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, "|", torch.cuda.get_device_name(0) if dev == "cuda" else "CPU")

    files = sorted(glob.glob(f"{BUNDLE}/players/*.pgn.zst"))
    names = [os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0] for f in files]
    P = len(files)
    print(f"{P} players — parsing in parallel...")

    t0 = time.time()
    allrecs = []
    args = [(files[i], names[i].lower(), i, N_REF, N_TEST) for i in range(P)]
    with Pool(min(4, os.cpu_count() or 2)) as pool:
        for k, recs in enumerate(pool.imap_unordered(extract_worker, args)):
            allrecs.extend(recs)
            if (k + 1) % 10 == 0:
                print(f"  parsed {k+1}/{P} players, {len(allrecs)} plies, {time.time()-t0:.0f}s")
    print(f"parsed {len(allrecs)} plies in {time.time()-t0:.0f}s")

    from sahformer.training.loop import load_model
    model, mcfg = load_model(f"{BUNDLE}/best.pt")
    model.eval().to(dev)
    for p in model.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def base_forward(board, hist, temporal, es, eo):
        tok = model.input_emb(board.float(), hist.float(), es, eo)
        t = model.temporal_enc(temporal)
        film = model.film_gen(t) if model.use_film else None
        enc = model.encoder(tok, t=(t if model.use_time_gab else None), film=film)
        return enc.mean(1) + model.t_to_d(t), model.policy(enc)

    cP, cLL, cLeg, cAc, cTh, cSp, cGm, cPid = ([] for _ in range(8))
    B = 256                      # modest batch for a 6GB card
    t0 = time.time()
    for s in range(0, len(allrecs), B):
        batch = allrecs[s:s + B]
        board = torch.from_numpy(np.stack([r[0] for r in batch])).to(dev)
        hist = torch.from_numpy(np.stack([r[1] for r in batch])).to(dev)
        temp = torch.from_numpy(np.stack([r[2] for r in batch])).float().to(dev)
        es = torch.tensor([r[3] for r in batch]).to(dev)
        eo = torch.tensor([r[4] for r in batch]).to(dev)
        pooled_b, ml = base_forward(board, hist, temp, es, eo)
        legpad = np.zeros((len(batch), MAXLEG), np.int64)
        acols = np.zeros(len(batch), np.int64)
        lens = np.zeros(len(batch), np.int64)
        for j, r in enumerate(batch):
            leg = r[6][:MAXLEG]
            lens[j] = len(leg)
            legpad[j, :len(leg)] = leg
            acols[j] = leg.index(r[5]) if r[5] in leg else 0
        legt = torch.from_numpy(legpad).to(dev)
        llog_b = torch.gather(ml, 1, legt)
        pad = torch.arange(MAXLEG, device=dev)[None, :] >= torch.from_numpy(lens).to(dev)[:, None]
        llog_b = llog_b.masked_fill(pad, -1e4)
        cP.append(pooled_b.half().cpu()); cLL.append(llog_b.half().cpu()); cLeg.append(legt.int().cpu())
        cAc.append(torch.from_numpy(acols)); cTh.append(torch.tensor([r[7] for r in batch]))
        cSp.append(torch.tensor([r[8] for r in batch])); cGm.append(torch.tensor([r[9] for r in batch]))
        cPid.append(torch.tensor([r[10] for r in batch]))
        if (s // B) % 40 == 0:
            print(f"  cached {s+len(batch)}/{len(allrecs)} plies, {time.time()-t0:.0f}s")

    pooled = torch.cat(cP); llog = torch.cat(cLL).float(); legal = torch.cat(cLeg).long()
    acol = torch.cat(cAc).long(); think = torch.cat(cTh).float()
    test = torch.cat(cSp).bool(); game = torch.cat(cGm).long(); pid = torch.cat(cPid).long()
    M = int(legal.max().item()) + 1
    torch.save({"pooled": pooled, "legal": legal, "llog": llog, "acol": acol, "think": think,
                "pid": pid, "test": test, "game": game, "M": M, "names": names}, CACHE)
    print(f"\nsaved {CACHE}: {pooled.shape[0]} plies, {P} players, move-space {M}")


if __name__ == "__main__":
    main()
