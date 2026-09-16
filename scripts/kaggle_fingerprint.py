# ============================================================================
# Player-fingerprint identification — Kaggle GPU notebook (v2, careful pass)
#
# Bundle (uploaded as a Kaggle Dataset) must contain:
#   <BUNDLE>/sahformer/  <BUNDLE>/best.pt  <BUNDLE>/players/*.pgn.zst
#
# Flow:  parallel-parse players (CPU, no GPU yet)  ->  load base + cache its
# features once (GPU)  ->  save cache to disk  ->  train joint player-embedding
# + move/timing residual adapters (base frozen)  ->  identify held-out games.
# Re-runs load the cache and skip straight to training.
# ============================================================================

# ==== cell 1: setup =========================================================
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "python-chess", "zstandard"])

BUNDLE = "/kaggle/input/datasets/slobaspeed/sah-fingerprint"   # <-- your dataset path
CACHE  = "/kaggle/working/cache.pt"                             # persists across re-runs
import os, glob, io, math, random, time
sys.path.insert(0, BUNDLE)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import zstandard, chess, chess.pgn
from multiprocessing import Pool
from sahformer.encoding import encode_board, encode_move, build_temporal
from sahformer.records import _stack_history, BASE_SECONDS
from sahformer.model.heads import move_to_index

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); random.seed(0); np.random.seed(0)
N_REF, N_TEST = 150, 40        # games/player: reference (train) / held-out (test)
MAXLEG = 72                    # pad legal moves to this width
EMB_DIM = 48
LAMBDA_TIME = 1.0
MODE = "film"                  # "film" (multiplicative) or "concat" (baseline) — A/B knob
print("device:", DEV, "| fingerprint mode:", MODE)

# ==== cell 2: per-player extraction (CPU only; picklable for the Pool) =======
def player_plies(path, me):
    """Yield the file-owner's own moves (feature dicts), grouped per game."""
    with open(path, "rb") as fh:
        text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh),
                                encoding="utf-8", errors="ignore")
        while True:
            g = chess.pgn.read_game(text)
            if g is None: break
            w = (g.headers.get("White","") or "").lower(); b = (g.headers.get("Black","") or "").lower()
            if me == w: me_white = True
            elif me == b: me_white = False
            else: continue
            we = int(g.headers.get("WhiteElo",0) or 0); be = int(g.headers.get("BlackElo",0) or 0)
            board = g.board(); prev = {chess.WHITE:BASE_SECONDS, chess.BLACK:BASE_SECONDS}
            thist = {chess.WHITE:[], chess.BLACK:[]}; ph = []; node = g; ply = 0; rows = []
            while node.variations:
                node = node.variation(0); mv = node.move; mover = board.turn; ca = node.clock()
                if ca is None: board.push(mv); continue
                think = max(prev[mover]-ca, 0.0)
                cur = encode_board(board)                      # every ply (history)
                if (mover == chess.WHITE) == me_white:
                    h = _stack_history(ph, cur)
                    frm,to,pr = encode_move(board, mv)
                    legal = [move_to_index(*encode_move(board, m)) for m in board.legal_moves]
                    rows.append((cur, h,
                                 build_temporal(my_clock=prev[mover], opp_clock=prev[not mover],
                                                own_think_history=thist[mover], ply=ply),
                                 (we if me_white else be), (be if me_white else we),
                                 move_to_index(frm,to,pr), legal, think))
                prev[mover] = ca; thist[mover] = [think]+thist[mover]; ph.append(cur); board.push(mv); ply += 1
            if len(rows) >= 8:
                yield rows

def extract_worker(args):
    """Parse one player -> list of positions tagged (split, global game id)."""
    path, me, pid, nref, ntest = args
    rnd = random.Random(pid)
    games = []
    for rows in player_plies(path, me):
        games.append(rows)
        if len(games) >= nref + ntest: break
    rnd.shuffle(games)
    out = []
    span = nref + ntest + 1
    for split, gs in ((0, games[:nref]), (1, games[nref:nref+ntest])):
        for local, rows in enumerate(gs):
            gidx = local if split == 0 else nref + local
            guid = pid * span + gidx                            # globally unique
            for r in rows:
                out.append((*r, split, guid, pid))              # (cur,h,temp,es,eo,ai,legal,think,split,guid,pid)
    return out

# ==== cell 3: build-or-load the cache =======================================
# Find the cache either in /kaggle/working (built here) or in an uploaded
# input dataset (glob fallback) so it survives kernel restarts / new sessions.
_hits = [CACHE] if os.path.exists(CACHE) else glob.glob("/kaggle/input/**/cache.pt", recursive=True)
if _hits:
    print("loading cache from", _hits[0])
    c = torch.load(_hits[0])
    pooled,legal,llog,acol,think,pid,test,game = (c[k] for k in
        ["pooled","legal","llog","acol","think","pid","test","game"])
    M, names = c["M"], c["names"]; P = len(names); N = pooled.shape[0]
    print(f"loaded cache: {N} plies, {P} players, move-space {M}")
else:
    files = sorted(glob.glob(f"{BUNDLE}/players/*.pgn.zst"))
    names = [os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0] for f in files]
    P = len(files); print(f"{P} players — parsing in parallel...")

    # ---- parallel parse (CPU, no CUDA yet) ----
    t0 = time.time(); allrecs = []
    args = [(files[i], names[i].lower(), i, N_REF, N_TEST) for i in range(P)]
    with Pool(min(4, os.cpu_count() or 2)) as pool:
        for k, recs in enumerate(pool.imap_unordered(extract_worker, args)):
            allrecs.extend(recs)
            if (k+1) % 10 == 0:
                print(f"  parsed {k+1}/{P} players, {len(allrecs)} plies, {time.time()-t0:.0f}s")
    print(f"parsed {len(allrecs)} plies in {time.time()-t0:.0f}s")

    # ---- NOW load the base on GPU and cache its features ----
    from sahformer.training.loop import load_model
    model, mcfg = load_model(f"{BUNDLE}/best.pt"); model.eval().to(DEV)
    for p in model.parameters(): p.requires_grad_(False)
    DIM = mcfg.dim_vit

    @torch.no_grad()
    def base_forward(board, hist, temporal, es, eo):
        tok = model.input_emb(board.float(), hist.float(), es, eo)
        t = model.temporal_enc(temporal)
        film = model.film_gen(t) if model.use_film else None
        enc = model.encoder(tok, t=(t if model.use_time_gab else None), film=film)
        return enc.mean(1) + model.t_to_d(t), model.policy(enc)

    cP,cLL,cLeg,cAc,cTh,cSp,cGm,cPid = ([] for _ in range(8)); B = 512
    t0 = time.time()
    for s in range(0, len(allrecs), B):
        batch = allrecs[s:s+B]
        board = torch.from_numpy(np.stack([r[0] for r in batch])).to(DEV)
        hist  = torch.from_numpy(np.stack([r[1] for r in batch])).to(DEV)
        temp  = torch.from_numpy(np.stack([r[2] for r in batch])).float().to(DEV)
        es = torch.tensor([r[3] for r in batch]).to(DEV)
        eo = torch.tensor([r[4] for r in batch]).to(DEV)
        pooled_b, ml = base_forward(board, hist, temp, es, eo)          # [b,DIM], [b,Mtrue]
        legpad = np.zeros((len(batch), MAXLEG), np.int64)
        acols  = np.zeros(len(batch), np.int64); lens = np.zeros(len(batch), np.int64)
        for j, r in enumerate(batch):
            leg = r[6][:MAXLEG]; lens[j] = len(leg); legpad[j,:len(leg)] = leg
            acols[j] = leg.index(r[5]) if r[5] in leg else 0
        legt = torch.from_numpy(legpad).to(DEV)
        llog_b = torch.gather(ml, 1, legt)                              # [b,MAXLEG]
        pad = torch.arange(MAXLEG, device=DEV)[None,:] >= torch.from_numpy(lens).to(DEV)[:,None]
        llog_b = llog_b.masked_fill(pad, -1e4)
        cP.append(pooled_b.half().cpu()); cLL.append(llog_b.half().cpu()); cLeg.append(legt.int().cpu())
        cAc.append(torch.from_numpy(acols)); cTh.append(torch.tensor([r[7] for r in batch]))
        cSp.append(torch.tensor([r[8] for r in batch])); cGm.append(torch.tensor([r[9] for r in batch]))
        cPid.append(torch.tensor([r[10] for r in batch]))
        if (s//B) % 20 == 0: print(f"  cached {s+len(batch)}/{len(allrecs)} plies, {time.time()-t0:.0f}s")

    pooled = torch.cat(cP); llog = torch.cat(cLL).float(); legal = torch.cat(cLeg).long()
    acol = torch.cat(cAc).long(); think = torch.cat(cTh).float()
    test = torch.cat(cSp).bool(); game = torch.cat(cGm).long(); pid = torch.cat(cPid).long()
    M = int(legal.max().item()) + 1; N = pooled.shape[0]
    torch.save({"pooled":pooled,"legal":legal,"llog":llog,"acol":acol,"think":think,
                "pid":pid,"test":test,"game":game,"M":M,"names":names}, CACHE)
    del allrecs, model
    print(f"cache built + saved: {N} plies, {P} players, move-space {M}")

# ---- move everything to the compute device (ALL of them) ----
pooled=pooled.to(DEV); legal=legal.to(DEV); llog=llog.to(DEV); acol=acol.to(DEV)
think=think.to(DEV); pid=pid.to(DEV); test=test.to(DEV); game=game.to(DEV)
print(f"{N} plies | {int(test.sum())} test | ready on {DEV}")

# ==== cell 4: the fingerprint model =========================================
class Fingerprint(nn.Module):
    """Conditions the base's position summary on a per-player embedding, then reads
    move + timing residual heads off the conditioned rep. Two modes (A/B knob):
      - "film":   player -> (gamma, beta), h = (1+gamma)*pooled + beta   (multiplicative)
      - "concat": h = MLP([pooled ; e_p])                                (baseline)"""
    def __init__(self, n_players, dim, emb=EMB_DIM, moves=None, mdn_k=4, hid=256, mode="film"):
        super().__init__()
        self.k = mdn_k; self.mode = mode
        self.emb = nn.Embedding(n_players, emb)
        if mode == "film":
            self.film = nn.Linear(emb, 2*dim)             # player -> (gamma, beta)
            nn.init.zeros_(self.film.bias)                # start: scale≈1, shift≈0
        elif mode == "concat":
            self.mix = nn.Sequential(nn.Linear(dim + emb, dim), nn.GELU())
        else:
            raise ValueError(f"unknown mode {mode!r}")
        self.move = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hid), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(hid, moves))
        self.time = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hid), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(hid, mdn_k*3))
        nn.init.zeros_(self.move[-1].weight); nn.init.zeros_(self.move[-1].bias)  # residual starts at 0
    def forward(self, pooled, e):
        if self.mode == "film":
            gamma, beta = self.film(e).chunk(2, -1)
            h = (1 + gamma) * pooled + beta               # FiLM modulation by the player
        else:
            h = self.mix(torch.cat([pooled, e], -1))      # concat baseline
        return self.move(h), self.time(h)

def move_logp(res, llog_b, legal_b, acol_b):
    r_leg = torch.gather(res, 1, legal_b)                 # residual at legal moves
    logits = llog_b + r_leg
    return torch.log_softmax(logits, -1).gather(1, acol_b[:,None]).squeeze(1)

def time_logp(mdn, t):
    k = mdn.shape[1] // 3
    pi = torch.log_softmax(mdn[:, :k], -1)
    mu = mdn[:, k:2*k]; sg = F.softplus(mdn[:, 2*k:3*k]) + 1e-3
    lt = torch.log(t.clamp(min=0.1))[:, None]
    comp = -lt - torch.log(sg) - 0.5*math.log(2*math.pi) - (lt-mu)**2/(2*sg**2)
    return torch.logsumexp(pi + comp, -1)

fp = Fingerprint(P, dim=pooled.shape[1], moves=M, mode=MODE).to(DEV)

# ==== cell 5: train (base frozen; only embeddings + adapters) ================
ref_idx = (~test).nonzero(as_tuple=True)[0]               # on DEV (test is on DEV)
opt = torch.optim.AdamW(fp.parameters(), lr=1e-3, weight_decay=1e-4)
EPOCHS, BS = 15, 4096
for ep in range(EPOCHS):
    perm = ref_idx[torch.randperm(len(ref_idx), device=DEV)]
    fp.train(); tot = n = 0
    for s in range(0, len(perm), BS):
        b = perm[s:s+BS]
        res, mdn = fp(pooled[b].float(), fp.emb(pid[b]))
        loss = -(move_logp(res, llog[b], legal[b], acol[b]).mean()
                 + LAMBDA_TIME * time_logp(mdn, think[b]).mean())
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); n += 1
    print(f"epoch {ep+1}/{EPOCHS}  loss {tot/max(1,n):.4f}")

# ==== cell 6: identification eval ===========================================
@torch.no_grad()
def channel_scores():
    """[n_test_plies, P] log-likelihood under every player, split by channel."""
    fp.eval(); ti = test.nonzero(as_tuple=True)[0]
    Sm = torch.zeros(len(ti), P, device=DEV); St = torch.zeros(len(ti), P, device=DEV)
    allp = torch.arange(P, device=DEV); D = pooled.shape[1]
    CH = 128
    for s in range(0, len(ti), CH):
        b = ti[s:s+CH]; nb = len(b)
        pl = pooled[b].float()[:,None,:].expand(nb,P,D).reshape(nb*P, D)
        e  = fp.emb(allp)[None].expand(nb,P,EMB_DIM).reshape(nb*P, EMB_DIM)
        res, mdn = fp(pl, e)
        Sm[s:s+nb] = move_logp(res, llog[b].repeat_interleave(P,0), legal[b].repeat_interleave(P,0),
                               acol[b].repeat_interleave(P,0)).view(nb, P)
        St[s:s+nb] = time_logp(mdn, think[b].repeat_interleave(P,0)).view(nb, P)
    return Sm, St, game[ti], pid[ti]

def eval_curves():
    Sm, St, tg, tp = channel_scores()
    games = tg.unique()
    gm = torch.zeros(len(games), P, device=DEV); gt = torch.zeros(len(games), P, device=DEV)
    g_true = torch.zeros(len(games), dtype=torch.long, device=DEV)
    for i, gg in enumerate(games):
        m = tg == gg
        gm[i] = Sm[m].sum(0); gt[i] = St[m].sum(0); g_true[i] = tp[m][0]
    gb = gm + LAMBDA_TIME * gt
    def acc(scores, Nsub):
        keep = g_true < Nsub
        return (scores[keep][:, :Nsub].argmax(1) == g_true[keep]).float().mean().item()
    print(f"\n=== mode: {MODE} ===")
    print("players(N)   both    moves   timing   chance")
    for Nsub in [10, 30, 60, P]:
        print(f"{Nsub:>8}   {acc(gb,Nsub):.3f}   {acc(gm,Nsub):.3f}   {acc(gt,Nsub):.3f}   {1/Nsub:.3f}")

eval_curves()
