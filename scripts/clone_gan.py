"""ITERATIVE (label-free) distribution DPO — the "keep spotting discrepancies" loop.

Each round: refit the reward direction to where the clone CURRENTLY differs from real
(w = mu_real - mu_clone, recomputed as the clone changes -> a moving target, not a fixed axis
that goes blind after round one). DPO the code toward w a few epochs, base-relative (triangle:
reward deviation BEYOND base, so it can't win by chasing the self-play confound), imitation-anchored.

HONESTY: trained against recognizer A (moves-only); VALIDATED every round on recognizer B
(moves+time) + a held-out ID panel. If B improves -> genuine. If only A moves -> sucking up, stop.

    PYTHONPATH=. python scripts/clone_gan.py --name latebloomer --rounds 8
"""
import argparse, os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
from sahformer.training.loop import load_model
from sahformer.clone import PooledCloneAdapter, save_pooled_clone
from clone_dpo_pooled import train_shared, gen_clone_traj, base_feats_of_traj, MAXLEG
from clone_judge import real_games, embed_games
from run_scale import GameEncoder

LSCALE = "data/lichess_scale"
def stem(f): return os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0]


def load_reco(path, dev):
    ck = torch.load(path, weights_only=False)
    enc = GameEncoder(ck["dp"], ck["use_time"], d=ck["d_model"], layers=ck["layers"],
                      heads=ck["heads"], L=ck["max_plies"]).to(dev)
    enc.load_state_dict(ck["state_dict"]); enc.eval()
    return enc, ck["tmean"], ck["tstd"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--cache", default="sweep_cache_120_full.pt")
    ap.add_argument("--reco-train", default="checkpoints/recognizer_film_moves.pt")
    ap.add_argument("--reco-val", default="checkpoints/recognizer_film.pt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--games", type=int, default=20, help="clone games generated per round")
    ap.add_argument("--epochs", type=int, default=6, help="DPO grad steps per round")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--lambda-imit", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--panel", type=int, default=12)
    ap.add_argument("--real-games", type=int, default=100)
    ap.add_argument("--emb", type=int, default=512); ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--wd", type=float, default=3e-2); ap.add_argument("--head-epochs", type=int, default=20)
    ap.add_argument("--dropout", type=float, default=0.6); ap.add_argument("--res-l2", type=float, default=0.10)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    elo = 1500

    model, _ = load_model("checkpoints/base_300k_best.pt"); model.eval().to(dev)
    for p in model.parameters(): p.requires_grad_(False)
    cache = torch.load(args.cache, weights_only=False)
    hm, players = train_shared(cache, args.emb, args.hidden, args.wd, args.head_epochs, args.dropout, args.res_l2, 1e-3)
    head = hm.net.to(dev).eval()
    for p in head.parameters(): p.requires_grad_(False)
    idx = players.index(args.name)
    code0 = hm.emb.weight[idx].detach().to(dev)
    code = torch.nn.Parameter(code0.clone())
    opt = torch.optim.Adam([code], lr=args.lr)
    def adapter_for(c): a = PooledCloneAdapter(head, c.detach()); a.to(dev); a.eval(); return a
    def gather_head(pooled, c): return head(torch.cat([pooled, c.expand(pooled.shape[0], -1)], -1))

    encT, tmT, tsT = load_reco(args.reco_train, dev)
    encV, tmV, tsV = load_reco(args.reco_val, dev)

    def embed(rows_list, enc, tm, ts):
        out = []
        for rows in rows_list:
            pooled, _, _, think = base_feats_of_traj(model, rows, elo, dev)
            logt = ((torch.log1p(think.clamp(min=0)) - tm) / ts).unsqueeze(0)
            mask = torch.ones(1, pooled.shape[0], dtype=torch.bool, device=dev)
            out.append(enc(pooled.unsqueeze(0), logt, mask).squeeze(0))
        return torch.stack(out) if out else torch.zeros(0, device=dev)

    def gen_rows(c_or_none, n, seed):
        games = gen_clone_traj(model, c_or_none, elo, n, dev, base_seed=seed)
        feats = []
        for rows in games:
            pooled, base_ml, chosen, think = base_feats_of_traj(model, rows, elo, dev)
            feats.append((pooled, base_ml, chosen, think))
        return games, feats

    # real (both recognizers) + base control + val panel
    realr = real_games(f"{LSCALE}/{args.name}.pgn.zst", args.name, args.real_games, elo)
    XrT = embed_games(realr, model, encT, tmT, tsT, dev); muT_real = XrT.mean(0)
    XrV = embed_games(realr, model, encV, tmV, tsV, dev); muV_real = F.normalize(XrV.mean(0), dim=-1)
    baseg = gen_clone_traj(model, None, elo, args.games, dev, base_seed=7000)
    XbaseT = embed([r for r in baseg], encT, tmT, tsT)  # base rows already full games
    muT_base = XbaseT.mean(0)
    # val panel: target + field players' centroids (val recognizer)
    others = [f for f in sorted(glob.glob(f"{LSCALE}/*.pgn.zst"), key=os.path.getsize, reverse=True)
              if stem(f) != args.name][:args.panel]
    cents = [muV_real]
    for f in others:
        e = embed_games(real_games(f, stem(f), 40, elo), model, encV, tmV, tsV, dev)
        cents.append(F.normalize(e.mean(0), dim=-1) if len(e) else torch.zeros_like(muV_real))
    Cval = torch.stack(cents)
    XbaseV = embed([r for r in baseg], encV, tmV, tsV)
    base_idp = float((XbaseV @ Cval.T).argmax(1).eq(0).float().mean())
    real_idp = float((XrV @ Cval.T).argmax(1).eq(0).float().mean())
    print(f"INDEPENDENT floors (val recognizer): base ID {base_idp:.3f}  real ID {real_idp:.3f}\n")

    imit_tp, imit_tbl, imit_tlg, imit_tln, imit_tac = cache[args.name]["train"]
    imit_tp=imit_tp.float().to(dev); imit_tbl=imit_tbl.float().to(dev); imit_tlg=imit_tlg.long().to(dev)
    imit_tln=imit_tln.to(dev); imit_tac=imit_tac.to(dev)
    imit_pad = torch.arange(MAXLEG, device=dev)[None,:] >= imit_tln[:,None]

    for r in range(args.rounds):
        games, feats = gen_rows(adapter_for(code), args.games, seed=r * 1000 + 1)
        with torch.no_grad():
            Xc = embed(games, encT, tmT, tsT)
            w = F.normalize(muT_real - Xc.mean(0), dim=-1)          # REFIT: current biggest discrepancy
            base_proj = float((XbaseT @ w).mean()); real_proj = float((XrT @ w).mean())
            scores = [float(e @ w) for e in Xc]                     # per clone game
        # DPO a few steps toward w (base-relative via the pref/dispref contrast), imitation-anchored
        order = np.argsort(scores); h = len(order) // 2
        pref, disp = order[h:], order[:h]; npair = min(len(pref), len(disp))
        for _ in range(args.epochs):
            def tlp(fi, c):
                pooled, base_ml, chosen, _ = feats[fi]
                lp = F.log_softmax(base_ml + gather_head(pooled, c), dim=-1)
                return lp[torch.arange(len(chosen), device=dev), chosen].sum()
            dpo = 0.0
            for k in range(npair):
                pi, di = int(pref[-1-k]), int(disp[k])
                with torch.no_grad(): rp, rd = tlp(pi, code0), tlp(di, code0)
                dpo = dpo - F.logsigmoid(args.beta * ((tlp(pi, code) - rp) - (tlp(di, code) - rd)))
            dpo = dpo / max(1, npair)
            res = gather_head(imit_tp, code); rl = torch.gather(res, 1, imit_tlg)
            imit = F.cross_entropy((imit_tbl + rl).masked_fill(imit_pad, -1e4), imit_tac)
            (dpo + args.lambda_imit * imit).backward(); opt.step(); opt.zero_grad()
        # --- validate on the INDEPENDENT recognizer ---
        with torch.no_grad():
            vg = gen_clone_traj(model, adapter_for(code), elo, args.games, dev, base_seed=r*1000 + 9000)
            XcV = embed(vg, encV, tmV, tsV)
            val_id = float((XcV @ Cval.T).argmax(1).eq(0).float().mean())
            train_closed = ((float((Xc @ w).mean()) - base_proj) / (real_proj - base_proj + 1e-9))
        print(f"round {r+1}/{args.rounds}  ||dcode|| {float((code-code0).norm()):.2f}  "
              f"train w-closed {train_closed*100:+.0f}%  |  INDEP val ID {val_id:.3f} "
              f"(base {base_idp:.3f} -> real {real_idp:.3f})", flush=True)

    out = args.out or f"clones/{args.name}_gan.pt"
    save_pooled_clone(out, PooledCloneAdapter(head.cpu(), code.detach().cpu()),
                      {"name": args.name, "avg_elo": elo, "kind": "pooled", "gan_rounds": args.rounds})
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
