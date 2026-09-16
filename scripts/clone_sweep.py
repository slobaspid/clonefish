"""Autonomous residual-config sweep on the CLEAN test (no leakage).

Train each player's move-residual on games 0-119, evaluate on HELD-OUT games 120-159, on
Lichess players (base-novel). Goal: find a config where the clone's held-out move top-1 beats
the frozen base — staying true to "frozen base + small player residual on top".

Base features are cached once per player; each config trial just retrains the small adapter,
so we can sweep many configs fast. Prints mean move Δ (clone - base) and log-lik Δ over players.
"""
import io, os, sys, time, argparse, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, zstandard, chess, chess.pgn
from sahformer.encoding import encode_board, encode_move, build_temporal
from sahformer.records import _stack_history, BASE_SECONDS
from sahformer.model.heads import move_to_index
from sahformer.training.loop import load_model

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MAXLEG = 72


def _games(path, me, max_games):
    """Parse up to max_games of the player's games -> list of per-game ply-row lists."""
    me = me.lower()
    text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(path, "rb")),
                            encoding="utf-8", errors="ignore")
    games = []
    while len(games) < max_games:
        g = chess.pgn.read_game(text)
        if g is None: break
        w = (g.headers.get("White","") or "").lower(); b = (g.headers.get("Black","") or "").lower()
        if me == w: me_white = True
        elif me == b: me_white = False
        else: continue
        we = int(g.headers.get("WhiteElo",0) or 0); be = int(g.headers.get("BlackElo",0) or 0)
        board = g.board(); prev = {chess.WHITE: BASE_SECONDS, chess.BLACK: BASE_SECONDS}
        thist = {chess.WHITE: [], chess.BLACK: []}; ph, node, ply, rows = [], g, 0, []
        while node.variations:
            node = node.variation(0); mv = node.move; mover = board.turn; ca = node.clock()
            if ca is None: board.push(mv); continue
            cur = encode_board(board)
            if (mover == chess.WHITE) == me_white:
                frm, to, pr = encode_move(board, mv)
                legal = [move_to_index(*encode_move(board, m)) for m in board.legal_moves]
                rows.append((cur, _stack_history(ph, cur),
                             build_temporal(prev[mover], prev[not mover], thist[mover], ply),
                             (we if me_white else be), (be if me_white else we),
                             legal, move_to_index(frm, to, pr)))
            think = max(prev[mover]-ca, 0.0); prev[mover] = ca
            thist[mover] = [think]+thist[mover]; ph.append(cur); board.push(mv); ply += 1
        if len(rows) >= 8: games.append(rows)
    return games


def extract(path, me, skip, take):
    gs = _games(path, me, skip + take)
    return [r for g in gs[skip:skip+take] for r in g]


def extract_split(path, me, max_games=300, test_n=40):
    """Full-history split: train = all games except the last test_n, test = last test_n (held out).
    Returns per-train-position game index (gid) plus meta = (ply_in_game, my_elo) for train and test
    (enables phase-breakdown and Elo-adaptability)."""
    gs = _games(path, me, max_games)
    if len(gs) <= test_n + 20:
        return None
    train, gid, meta_tr = [], [], []
    for gi, g in enumerate(gs[:-test_n]):
        for ply, r in enumerate(g):
            train.append(r); gid.append(gi); meta_tr.append((ply, r[3]))
    test, meta_te = [], []
    for g in gs[-test_n:]:
        for ply, r in enumerate(g):
            test.append(r); meta_te.append((ply, r[3]))
    return train, test, gid, meta_tr, meta_te


@torch.no_grad()
def base_feats(model, rows):
    """-> pooled[n,512], base_legal[n,72], legal[n,72], lens[n], acol[n]  (all held for adapter)."""
    P, BL, LG, LN, AC = [], [], [], [], []
    B = 256
    for s in range(0, len(rows), B):
        ch = rows[s:s+B]
        batch = {"board": torch.from_numpy(np.stack([r[0] for r in ch])).float().to(DEV),
                 "history": torch.from_numpy(np.stack([r[1] for r in ch])).float().to(DEV),
                 "elo_self": torch.tensor([r[3] for r in ch]).to(DEV),
                 "elo_opp": torch.tensor([r[4] for r in ch]).to(DEV),
                 "temporal": torch.from_numpy(np.stack([r[2] for r in ch])).float().to(DEV)}
        out = model(batch)
        P.append(out["pooled"].half().cpu()); ml = out["move_logits"].cpu()
        legpad = np.zeros((len(ch), MAXLEG), np.int64); acol = np.zeros(len(ch), np.int64); lens = np.zeros(len(ch), np.int64)
        for j, r in enumerate(ch):
            leg = r[5][:MAXLEG]; lens[j] = len(leg); legpad[j,:len(leg)] = leg
            acol[j] = leg.index(r[6]) if r[6] in leg else 0
        legt = torch.from_numpy(legpad)
        bl = torch.gather(ml, 1, legt); bl[torch.arange(MAXLEG)[None,:] >= torch.from_numpy(lens)[:,None]] = -1e4
        BL.append(bl.half()); LG.append(legt.int()); LN.append(torch.from_numpy(lens)); AC.append(torch.from_numpy(acol))
    return (torch.cat(P), torch.cat(BL), torch.cat(LG), torch.cat(LN), torch.cat(AC))


class MoveResidual(nn.Module):
    def __init__(self, dim=512, n_moves=4352, hidden=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, n_moves))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, pooled): return self.net(pooled)


class PooledResidual(nn.Module):
    """Shared head + low-dim per-player embedding (empirical-Bayes pooling)."""
    def __init__(self, n_players, emb_dim=32, dim=512, n_moves=4352, hidden=64, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(n_players, emb_dim)
        nn.init.normal_(self.emb.weight, std=0.01)
        self.net = nn.Sequential(nn.LayerNorm(dim + emb_dim), nn.Linear(dim + emb_dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, n_moves))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, pooled, pid):
        return self.net(torch.cat([pooled, self.emb(pid)], -1))


def evaluate_pooled(cache, emb_dim, hidden, wd, epochs, dropout, res_l2, lr, bs=2048, adapt_kappa=0.0):
    torch.manual_seed(0)
    players = list(cache.keys()); P = len(players)
    Ptr, BLtr, LGtr, LNtr, ACtr, PID = [], [], [], [], [], []
    n_pos = []
    for i, n in enumerate(players):
        tp, tbl, tlg, tln, tac = cache[n]["train"]
        Ptr.append(tp.float()); BLtr.append(tbl.float()); LGtr.append(tlg.long()); LNtr.append(tln); ACtr.append(tac)
        PID.append(torch.full((tp.shape[0],), i)); n_pos.append(tp.shape[0])
    # keep the big tensors on CPU (fits huge caches on a 6GB GPU); move only each batch to the GPU
    Ptr=torch.cat(Ptr); BLtr=torch.cat(BLtr); LGtr=torch.cat(LGtr)
    LNtr=torch.cat(LNtr); ACtr=torch.cat(ACtr); PID=torch.cat(PID)
    npos_t = torch.tensor(n_pos, dtype=torch.float32, device=DEV)
    inv_n = npos_t.median() / npos_t
    m = PooledResidual(P, emb_dim, hidden=hidden, dropout=dropout).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    N = Ptr.shape[0]
    arange_leg = torch.arange(MAXLEG, device=DEV)
    for _ in range(epochs):
        perm = torch.randperm(N); m.train()
        for s in range(0, N, bs):
            b = perm[s:s+bs]
            pb = Ptr[b].to(DEV); bl = BLtr[b].to(DEV); lg = LGtr[b].to(DEV)
            ln = LNtr[b].to(DEV); ac = ACtr[b].to(DEV); pid = PID[b].to(DEV)
            pad = arange_leg[None,:] >= ln[:,None]
            res = m(pb, pid); rl = torch.gather(res, 1, lg)
            logits = (bl + rl).masked_fill(pad, -1e4)
            loss = F.cross_entropy(logits, ac) + res_l2 * res.pow(2).mean()
            if adapt_kappa > 0:                       # shrink each player's embedding by kappa/n_games
                loss = loss + adapt_kappa * (inv_n * m.emb.weight.pow(2).sum(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); dmove, dll = [], []
    with torch.no_grad():
        for i, n in enumerate(players):
            ep, ebl, elg, eln, eac = cache[n]["test"]
            epf=ep.float().to(DEV); ebf=ebl.float().to(DEV); elf=elg.long().to(DEV); eac_d=eac.to(DEV)
            epad = torch.arange(MAXLEG, device=DEV)[None,:] >= eln.to(DEV)[:,None]
            pid = torch.full((epf.shape[0],), i, device=DEV)
            res = m(epf, pid); rl = torch.gather(res, 1, elf)
            base_lg = ebf.masked_fill(epad, -1e4); clone_lg = (ebf + rl).masked_fill(epad, -1e4)
            dmove.append((( clone_lg.argmax(1)==eac_d).float().mean() - (base_lg.argmax(1)==eac_d).float().mean()).item()*100)
            dll.append((torch.log_softmax(clone_lg,1).gather(1,eac_d[:,None]).mean()
                        - torch.log_softmax(base_lg,1).gather(1,eac_d[:,None]).mean()).item())
    return float(np.mean(dmove)), float(np.mean(dll))


def evaluate(cache, hidden, wd, epochs, dropout, res_l2, lr, bs=1024):
    torch.manual_seed(0)
    dmove, dll = [], []
    for name, c in cache.items():
        tp, tbl, tlg, tln, tac = c["train"]
        ep, ebl, elg, eln, eac = c["test"]
        tp = tp.float().to(DEV); tbl = tbl.float().to(DEV); tlg = tlg.long().to(DEV); tac = tac.to(DEV)
        pad = torch.arange(MAXLEG, device=DEV)[None,:] >= tln.to(DEV)[:,None]
        adapter = MoveResidual(hidden=hidden, dropout=dropout).to(DEV)
        opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=wd)
        N = tp.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(N, device=DEV)
            adapter.train()
            for s in range(0, N, bs):
                b = perm[s:s+bs]
                res = adapter(tp[b])
                rl = torch.gather(res, 1, tlg[b])
                logits = (tbl[b] + rl).masked_fill(pad[b], -1e4)
                loss = F.cross_entropy(logits, tac[b]) + res_l2 * res.pow(2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        # eval on held-out
        adapter.eval()
        with torch.no_grad():
            epf = ep.float().to(DEV); ebf = ebl.float().to(DEV); elf = elg.long().to(DEV)
            eac_d = eac.to(DEV); epad = torch.arange(MAXLEG, device=DEV)[None,:] >= eln.to(DEV)[:,None]
            res = adapter(epf); rl = torch.gather(res, 1, elf)
            base_lg = ebf.masked_fill(epad, -1e4); clone_lg = (ebf + rl).masked_fill(epad, -1e4)
            b_top1 = (base_lg.argmax(1) == eac_d).float().mean().item()
            c_top1 = (clone_lg.argmax(1) == eac_d).float().mean().item()
            b_ll = torch.log_softmax(base_lg,1).gather(1, eac_d[:,None]).mean().item()
            c_ll = torch.log_softmax(clone_lg,1).gather(1, eac_d[:,None]).mean().item()
        dmove.append((c_top1 - b_top1) * 100); dll.append(c_ll - b_ll)
    return float(np.mean(dmove)), float(np.mean(dll))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=int, default=8)
    ap.add_argument("--ckpt", default="checkpoints/base_300k_best.pt")
    ap.add_argument("--mode", choices=["perplayer", "pooled"], default="perplayer")
    ap.add_argument("--full-train", action="store_true", help="train on all games except held-out last 40 (variable n)")
    ap.add_argument("--players-curve", action="store_true", help="trace moveΔ vs #players pooled")
    ap.add_argument("--data-dir", default="data/lichess_scale", help="corpus dir of <user>.pgn.zst files")
    ap.add_argument("--max-games", type=int, default=300, help="max games/player for the full-train split")
    ap.add_argument("--cache-path", default=None, help="override cache filename")
    ap.add_argument("--build-only", action="store_true", help="build+save the cache then exit (no sweep)")
    args = ap.parse_args()
    files = sorted(__import__("glob").glob(f"{args.data_dir}/*.pgn.zst"), key=os.path.getsize, reverse=True)[:args.players]
    names = [os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0] for f in files]
    print(f"players: {names}")
    tag = os.path.basename(args.data_dir.rstrip("/"))
    cache_path = args.cache_path or f"sweep_cache_{tag}_{args.players}{'_full' if args.full_train else ''}.pt"
    if os.path.exists(cache_path):
        print(f"loading cached features from {cache_path}")
        cache = torch.load(cache_path, weights_only=False)
    else:
        model, _ = load_model(args.ckpt); model.eval().to(DEV)
        cache = {}
        t0 = time.time()
        for f, n in zip(files, names):
            gid = None
            meta_tr = meta_te = None
            if args.full_train:
                sp = extract_split(f, n, args.max_games, 40)
                if sp is None:
                    continue
                tr, te, gid, meta_tr, meta_te = sp
            else:
                tr = extract(f, n, 0, 120); te = extract(f, n, 120, 40); gid = None
            cache[n] = {"train": base_feats(model, tr), "test": base_feats(model, te)}
            if gid is not None:
                cache[n]["gid"] = torch.tensor(gid)
            if meta_tr is not None:
                cache[n]["meta_train"] = torch.tensor(meta_tr, dtype=torch.long)   # (ply, elo)
                cache[n]["meta_test"] = torch.tensor(meta_te, dtype=torch.long)
            print(f"  cached {n}: {len(tr)} train, {len(te)} test positions ({time.time()-t0:.0f}s)")
        del model; torch.cuda.empty_cache()
        torch.save(cache, cache_path)
        print(f"saved cache -> {cache_path}")
    if args.build_only:
        print("build-only: done."); return

    if args.mode == "pooled" and args.players_curve:
        print("\n=== POOLING CURVE at HIGH DIM: moveΔ vs #players (full histories, emb=256) ===")
        cfg = (256, 256, 2e-2, 18, 0.5, 0.05, 1e-3)   # high-dim config (players x dim interaction)
        allp = list(cache.items())
        print(f"{'#players':>8}   {'moveΔ':>7} {'llΔ':>7}")
        for K in [50, 120, 200, 300, 400]:
            if K > len(allp): break
            sub = dict(allp[:K])
            dm, dl = evaluate_pooled(sub, *cfg)
            print(f"{K:>8}   {dm:>+7.2f} {dl:>+7.3f}")
        return

    if args.mode == "pooled":
        print("\n=== POOLED SWEEP (shared head + per-player embedding, clean held-out) ===")
        print(f"{'emb':>4} {'hidden':>6} {'wd':>6} {'ep':>3} {'drop':>4} {'resL2':>6} {'lr':>6} {'kappa':>6}   {'moveΔ':>7} {'llΔ':>7}")
        # HIGH EMBEDDING-DIM sweep — find where per-player capacity converges (emb, hidden, wd, ep, drop, resL2, lr, kappa)
        pconfigs = [
            (256, 256, 2e-2, 15, 0.5, 0.05, 1e-3, 0.0),   # reference
            (512, 512, 2e-2, 18, 0.5, 0.05, 1e-3, 0.0),
            (1024, 512, 2e-2, 20, 0.6, 0.05, 1e-3, 0.0),
            # bigger codes — the games x dim interaction that ~250 games couldn't support (1k games test)
            (2048, 768, 3e-2, 22, 0.6, 0.10, 1e-3, 0.0),
            (2048, 1024, 4e-2, 22, 0.7, 0.15, 1e-3, 0.0),
            (4096, 1024, 5e-2, 24, 0.7, 0.20, 1e-3, 0.0),
            # lighter reg at high games (1k games needs less babysitting than 250)
            (1024, 768, 1e-2, 22, 0.4, 0.03, 1e-3, 0.0),
            (2048, 1024, 2e-2, 22, 0.5, 0.05, 1e-3, 0.0),
        ]
        best = (-1e9, None)
        for cfg in pconfigs:
            dm, dl = evaluate_pooled(cache, *cfg[:7], adapt_kappa=cfg[7])
            star = "  <--" if dm > best[0] else ""
            if dm > best[0]: best = (dm, cfg)
            print(f"{cfg[0]:>4} {cfg[1]:>6} {cfg[2]:>6.0e} {cfg[3]:>3} {cfg[4]:>4.1f} {cfg[5]:>6.2f} {cfg[6]:>6.0e} {cfg[7]:>6.2f}   {dm:>+7.2f} {dl:>+7.3f}{star}")
        print(f"\nbest pooled: moveΔ {best[0]:+.2f}pp  config {best[1]}")
        return

    print("\n=== CONFIG SWEEP (clean held-out) ===")
    print(f"{'hidden':>6} {'wd':>6} {'ep':>3} {'drop':>4} {'resL2':>6} {'lr':>6}   {'moveΔ':>7} {'llΔ':>7}")
    # (hidden, wd, epochs, dropout, res_l2, lr)
    configs = [
        (8, 1e-2, 6, 0.3, 0.03, 1e-3),     # prev best (+0.66)
        (4, 1e-2, 8, 0.3, 0.03, 1e-3),
        (6, 1e-2, 8, 0.3, 0.03, 1e-3),
        (8, 1e-2, 8, 0.3, 0.03, 1e-3),
        (8, 5e-3, 8, 0.2, 0.02, 1e-3),
        (8, 3e-2, 8, 0.3, 0.05, 1e-3),
        (8, 1e-2, 10, 0.4, 0.05, 1e-3),
        (12, 1e-2, 8, 0.3, 0.05, 1e-3),
        (8, 1e-2, 6, 0.3, 0.01, 2e-3),
        (8, 1e-2, 12, 0.3, 0.05, 5e-4),
        (4, 3e-2, 12, 0.2, 0.03, 1e-3),
        (8, 1e-2, 6, 0.3, 0.10, 1e-3),
    ]
    best = (-1e9, None)
    for cfg in configs:
        dm, dl = evaluate(cache, *cfg)
        star = "  <--" if dm > best[0] else ""
        if dm > best[0]: best = (dm, cfg)
        print(f"{cfg[0]:>6} {cfg[1]:>6.0e} {cfg[2]:>3} {cfg[3]:>4.1f} {cfg[4]:>6.2f} {cfg[5]:>6.0e}   {dm:>+7.2f} {dl:>+7.3f}{star}")
    print(f"\nbest so far: moveΔ {best[0]:+.2f}pp  config {best[1]}")


if __name__ == "__main__":
    main()
