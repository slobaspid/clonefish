"""DISTRIBUTION DPO on the pooled per-player CODE (the frontier — closes what per-move can't).

Trajectory-level DPO that sidesteps the sampling-gradient wall (spec §5):
  1. generate a batch of clone self-play games under the current code (sampled),
  2. embed each with the frozen recognizer; score = closeness to the player's REAL game cloud,
  3. pair them (real-looking = preferred, clone-looking = dispreferred) and run DPO on each game's
     TRAJECTORY log-prob (sum of per-move log-probs — differentiable in the code), leashed to the
     frozen start code, anchored by an imitation term on the player's real moves.
Only the 1 player's code vector moves (base + shared head + recognizer all frozen).

    PYTHONPATH=. python scripts/clone_dpo_pooled.py --cache sweep_cache_120_full.pt \
        --name latebloomer --recognizer checkpoints/recognizer_film_moves.pt \
        --out clones/latebloomer_dpo.pt --iters 40
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
from sahformer.encoding import encode_board, encode_move, build_temporal
from sahformer.records import _stack_history, BASE_SECONDS
from sahformer.model.heads import move_to_index
from sahformer.training.loop import load_model
from sahformer.clone import PooledCloneAdapter, save_pooled_clone
from sahformer.play import self_play
from build_pooled_clone import train_shared, MAXLEG
from clone_judge import real_games, embed_games, mmd_rbf, discriminator_auc
from run_scale import GameEncoder


def gen_clone_traj(model, adapter, elo, ngames, dev, base_seed=0, max_plies=80):
    """Generate clone self-play games. For each: the WHITE side's own plies, with per-ply
    (board_enc, hist, temporal, chosen_full_move_idx, think). One base forward later gives
    pooled + base move-logits for the trajectory log-prob."""
    out = []
    for s in range(ngames):
        traj = []
        for rec in self_play(model, max_plies=max_plies, elo=elo, temperature=1.0, top_p=0.9,
                             seed=base_seed + s, adapter=adapter, device=dev):
            traj.append((rec["move"], rec["mover"] == "white", rec["think"]))
        # replay, keep White's own plies with the chosen move index
        board = __import__("chess").Board(); ph = []; rows = []; ply = 0
        prev = {True: BASE_SECONDS, False: BASE_SECONDS}; thist = {True: [], False: []}
        for mv, mover_white, think in traj:
            cur = encode_board(board)
            if mover_white:
                frm, to, pr = encode_move(board, mv)
                rows.append((cur, _stack_history(ph, cur),
                             build_temporal(prev[True], prev[False], thist[True], ply),
                             move_to_index(frm, to, pr), think))
            prev[mover_white] = max(prev[mover_white] - think, 0.0)
            thist[mover_white] = [think] + thist[mover_white]
            ph.append(cur); board.push(mv); ply += 1
        if len(rows) >= 8:
            out.append(rows)
    return out


@torch.no_grad()
def base_feats_of_traj(model, rows, elo, dev):
    """One base forward over a game's own plies -> pooled[L,512], base_ml[L,4352], chosen[L], think[L]."""
    rows = rows[:60]
    board = torch.from_numpy(np.stack([r[0] for r in rows])).float().to(dev)
    hist = torch.from_numpy(np.stack([r[1] for r in rows])).float().to(dev)
    temp = torch.from_numpy(np.stack([r[2] for r in rows])).float().to(dev)
    es = torch.full((len(rows),), elo).to(dev)
    out = model({"board": board, "history": hist, "elo_self": es, "elo_opp": es, "temporal": temp})
    chosen = torch.tensor([r[3] for r in rows], device=dev)
    think = torch.tensor([r[4] for r in rows], device=dev).float()
    return out["pooled"], out["move_logits"], chosen, think


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="sweep_cache_120_full.pt")
    ap.add_argument("--name", required=True)
    ap.add_argument("--recognizer", default="checkpoints/recognizer_film_moves.pt")
    ap.add_argument("--pgn", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--games-per-iter", type=int, default=10)
    ap.add_argument("--real-games", type=int, default=100)
    ap.add_argument("--judge-games", type=int, default=40)
    ap.add_argument("--judge-every", type=int, default=10)
    ap.add_argument("--field-players", type=int, default=6, help="other players for the field centroid")
    ap.add_argument("--field-games", type=int, default=30)
    ap.add_argument("--beta", type=float, default=0.5, help="DPO KL-leash on the code")
    ap.add_argument("--lambda-imit", type=float, default=1.0, help="imitation anchor on real moves")
    ap.add_argument("--lr", type=float, default=5e-2)
    # shared-head config (must match build_pooled_clone default)
    ap.add_argument("--emb", type=int, default=512); ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--wd", type=float, default=3e-2); ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--dropout", type=float, default=0.6); ap.add_argument("--res-l2", type=float, default=0.10)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pgn = args.pgn or f"data/lichess_scale/{args.name}.pgn.zst"

    # frozen base + shared head (+ target's imitation-fit code as the reference)
    model, _ = load_model("checkpoints/base_300k_best.pt"); model.eval().to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    cache = torch.load(args.cache, weights_only=False)
    head_model, players = train_shared(cache, args.emb, args.hidden, args.wd, args.epochs, args.dropout, args.res_l2, 1e-3)
    head = head_model.net.to(dev).eval()
    for p in head.parameters():
        p.requires_grad_(False)
    idx = players.index(args.name)
    code0 = head_model.emb.weight[idx].detach().to(dev)          # frozen reference code
    code = torch.nn.Parameter(code0.clone())                     # the ONLY trainable thing
    elo = 1500
    opt = torch.optim.Adam([code], lr=args.lr)

    def adapter_for(c):
        a = PooledCloneAdapter(head, c.detach()); a.to(dev); a.eval(); return a

    # frozen recognizer + real-game cloud (embedded once)
    rck = torch.load(args.recognizer, weights_only=False)
    enc = GameEncoder(rck["dp"], rck["use_time"], d=rck["d_model"], layers=rck["layers"],
                      heads=rck["heads"], L=rck["max_plies"]).to(dev)
    enc.load_state_dict(rck["state_dict"]); enc.eval()
    tmean, tstd = rck["tmean"], rck["tstd"]
    real = real_games(pgn, args.name, args.real_games, elo)
    Xr = embed_games(real, model, enc, tmean, tstd, dev)         # real cloud [Nr, d]
    mu_real = Xr.mean(0)

    # FIELD: other players' REAL games -> mu_field. The fingerprint direction is what makes THIS
    # player distinctive vs a generic real player (not vs the base's self-play artifact).
    import glob
    others = [f for f in sorted(glob.glob("data/lichess_scale/*.pgn.zst"), key=os.path.getsize, reverse=True)
              if os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0] != args.name][:args.field_players]
    field_embs = []
    for f in others:
        nm = os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0]
        fg = real_games(f, nm, args.field_games, elo)
        if fg:
            field_embs.append(embed_games(fg, model, enc, tmean, tstd, dev))
    mu_field = torch.cat(field_embs).mean(0)
    w = F.normalize(mu_real - mu_field, dim=-1)                  # TRIANGULATED fingerprint direction

    @torch.no_grad()
    def embed_selfplay(adapter, n, seed):
        clg = gen_clone_traj(model, adapter, elo, n, dev, base_seed=seed)
        embs = []
        for rows in clg:
            pooled, _, _, think = base_feats_of_traj(model, rows, elo, dev)
            logt = ((torch.log1p(think.clamp(min=0)) - tmean) / tstd).unsqueeze(0)
            mask = torch.ones(1, pooled.shape[0], dtype=torch.bool, device=dev)
            embs.append(enc(pooled.unsqueeze(0), logt, mask).squeeze(0))
        return torch.stack(embs)

    # TRIANGULATION anchors (all along w). base self-play = the control that carries the self-play
    # artifact; clone should move from base_proj toward real_proj.
    real_proj = float((Xr @ w).mean())
    base_proj = float((embed_selfplay(None, args.judge_games, 8000) @ w).mean())     # base = adapter None
    print(f"fingerprint projections along w:  base(control) {base_proj:+.3f}  real(target) {real_proj:+.3f}  "
          f"gap {real_proj-base_proj:+.3f}")

    # imitation anchor data: the player's real TRAIN positions (cached, legal-gathered)
    tp, tbl, tlg, tln, tac = cache[args.name]["train"]
    tp=tp.float().to(dev); tbl=tbl.float().to(dev); tlg=tlg.long().to(dev); tln=tln.to(dev); tac=tac.to(dev)
    tpad = torch.arange(MAXLEG, device=dev)[None,:] >= tln[:,None]

    @torch.no_grad()
    def judge(c):
        Xc = embed_selfplay(adapter_for(c), args.judge_games, 9000)
        clone_proj = float((Xc @ w).mean())
        closed = (clone_proj - base_proj) / (real_proj - base_proj + 1e-9)   # frac of base->real gap
        return closed, clone_proj, discriminator_auc(Xr.cpu(), Xc.cpu())

    cl0, cp0, nn0 = judge(code0)
    print(f"START  clone_proj {cp0:+.3f}  gap-closed {cl0*100:+.0f}%  1-NN {nn0:.3f}")

    def gather_head(pooled, c):
        cc = c.expand(pooled.shape[0], -1)
        return head(torch.cat([pooled, cc], dim=-1))

    for it in range(args.iters):
        # 1) generate under current code (sampled), collect per-ply features + recognizer score
        games = gen_clone_traj(model, adapter_for(code), elo, args.games_per_iter, dev, base_seed=it * 1000)
        feats, scores = [], []
        with torch.no_grad():
            for rows in games:
                pooled, base_ml, chosen, think = base_feats_of_traj(model, rows, elo, dev)
                logt = ((torch.log1p(think.clamp(min=0)) - tmean) / tstd).unsqueeze(0)
                mask = torch.ones(1, pooled.shape[0], dtype=torch.bool, device=dev)
                emb = enc(pooled.unsqueeze(0), logt, mask).squeeze(0)
                feats.append((pooled, base_ml, chosen))
                scores.append(float((emb @ w)))                   # projection onto fingerprint dir
        order = np.argsort(scores)                                # worst..best
        h = len(order) // 2
        pref_ids = order[h:]; disp_ids = order[:h]                # top half preferred
        npair = min(len(pref_ids), len(disp_ids))
        if npair == 0:
            continue

        # 2) trajectory-DPO on the code (differentiable: fixed trajectories, logprob under code)
        def traj_logp(fi, c):
            pooled, base_ml, chosen = feats[fi]
            logp = F.log_softmax(base_ml + gather_head(pooled, c), dim=-1)
            return logp[torch.arange(len(chosen), device=dev), chosen].sum()

        dpo = 0.0
        for k in range(npair):
            pi, di = int(pref_ids[-1 - k]), int(disp_ids[k])
            with torch.no_grad():
                rp = traj_logp(pi, code0); rd = traj_logp(di, code0)
            lp = traj_logp(pi, code); ld = traj_logp(di, code)
            dpo = dpo - F.logsigmoid(args.beta * ((lp - rp) - (ld - rd)))
        dpo = dpo / npair

        # 3) imitation anchor (real moves stay likely) — keeps it from caricaturing
        res = gather_head(tp, code); rl = torch.gather(res, 1, tlg)
        imit = F.cross_entropy((tbl + rl).masked_fill(tpad, -1e4), tac)

        loss = dpo + args.lambda_imit * imit
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % args.judge_every == 0:
            cl, cp, nn = judge(code)
            print(f"iter {it+1:>3}/{args.iters}  dpo {float(dpo):+.3f}  imit {float(imit):.3f}  "
                  f"meanscore {np.mean(scores):+.3f}  ||dcode|| {float((code-code0).norm()):.3f}  "
                  f"clone_proj {cp:+.3f}  gap-closed {cl*100:+.0f}%  1-NN {nn:.3f}", flush=True)
        else:
            print(f"iter {it+1:>3}/{args.iters}  dpo {float(dpo):+.3f}  imit {float(imit):.3f}  "
                  f"meanscore {np.mean(scores):+.3f}  ||dcode|| {float((code-code0).norm()):.3f}", flush=True)

    cl1, cp1, nn1 = judge(code)
    print(f"\nEND    clone_proj {cp1:+.3f}  gap-closed {cl1*100:+.0f}%  1-NN {nn1:.3f}   "
          f"(start proj {cp0:+.3f}, closed {cl0*100:+.0f}%)")
    save_pooled_clone(args.out, PooledCloneAdapter(head.cpu(), code.detach().cpu()),
                      {"name": args.name, "avg_elo": elo, "kind": "pooled",
                       "dpo": {"beta": args.beta, "iters": args.iters}})
    print(f"saved DPO'd pooled clone -> {args.out}")


if __name__ == "__main__":
    main()
