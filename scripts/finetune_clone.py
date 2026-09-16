"""Copy the Maia-individual approach: FULL fine-tune the base model on one player's games (not a
small residual). Tests whether the field's strongest personalization cracks the MIDGAME where our
thin residual couldn't. Honest split (train = all but last 40 games, test = last 40), phase-split eval.

    PYTHONPATH=. python scripts/finetune_clone.py --pgn data/latebloomer_full.pgn --name latebloomer --epochs 3
"""
import argparse, os, sys, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F, chess, chess.pgn
from sahformer.encoding import encode_board, encode_move, build_temporal
from sahformer.records import _stack_history, BASE_SECONDS
from sahformer.model.heads import move_to_index, mdn_nll
from sahformer.training.loop import load_model

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BUCKETS = [(0, 12, "opening"), (12, 24, "early-mid"), (24, 40, "middlegame"), (40, 9999, "endgame")]


def open_pgn(p):
    if p.endswith(".zst"):
        import zstandard
        return io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(p, "rb")), encoding="utf-8", errors="ignore")
    return open(p, encoding="utf-8", errors="ignore")


def games_of(path, me):
    me = me.lower(); text = open_pgn(path); out = []
    while True:
        g = chess.pgn.read_game(text)
        if g is None: break
        w = (g.headers.get("White","") or "").lower(); b = (g.headers.get("Black","") or "").lower()
        if me in (w, b): out.append(g)
    return out


def rows_of(g, me):
    """per own-ply: (board_enc, hist, temporal, es, eo, played_full_idx, legal_idxs, think, ply)"""
    me_white = (me == (g.headers.get("White","") or "").lower())
    we = int(g.headers.get("WhiteElo",0) or 0); be = int(g.headers.get("BlackElo",0) or 0)
    board = g.board(); prev = {chess.WHITE: BASE_SECONDS, chess.BLACK: BASE_SECONDS}
    thist = {chess.WHITE: [], chess.BLACK: []}; ph, node, ply, rows, own = [], g, 0, [], 0
    while node.variations:
        node = node.variation(0); mv = node.move; mover = board.turn; ca = node.clock()
        if ca is None: board.push(mv); continue
        cur = encode_board(board)
        if (mover == chess.WHITE) == me_white:
            think = max(prev[mover]-ca, 0.0)
            frm, to, pr = encode_move(board, mv)
            legal = [move_to_index(*encode_move(board, m)) for m in board.legal_moves]
            rows.append((cur, _stack_history(ph, cur),
                         build_temporal(prev[mover], prev[not mover], thist[mover], ply),
                         (we if me_white else be), (be if me_white else we),
                         move_to_index(frm, to, pr), legal, think, own)); own += 1
        think2 = max(prev[mover]-ca, 0.0); prev[mover] = ca; thist[mover] = [think2]+thist[mover]
        ph.append(cur); board.push(mv); ply += 1
    return rows


def bucket_nll(pi_logits, mu, sigma_param, think, n_q=16):
    """-log P(whole-second clock reading = k). Lichess clocks are integer seconds, so a reading k means
    true think t in (k-1, k+1) with triangular weight; the point-density mdn_nll clamps k=0 to log(1e-6)."""
    import math
    k = think.round().unsqueeze(-1)
    lo, hi = (k - 1).clamp_min(0), k + 1
    t = lo + (hi - lo) * (torch.arange(n_q, device=k.device, dtype=k.dtype) + 0.5) / n_q
    w = (1 - (t - k).abs()).clamp_min(1e-12) * (hi - lo) / n_q
    lt, sig = t.log().unsqueeze(-1), F.softplus(sigma_param).unsqueeze(1) + 1e-3
    comp = -0.5 * ((lt - mu.unsqueeze(1)) / sig) ** 2 - sig.log() - 0.5 * math.log(2 * math.pi) - lt
    logdens = torch.logsumexp(F.log_softmax(pi_logits, -1).unsqueeze(1) + comp, -1)
    return -torch.logsumexp(logdens + w.log(), -1).mean()


def batch_of(rows, idx, dev):
    ch = [rows[i] for i in idx]
    return {"board": torch.from_numpy(np.stack([r[0] for r in ch])).float().to(dev),
            "history": torch.from_numpy(np.stack([r[1] for r in ch])).float().to(dev),
            "elo_self": torch.tensor([r[3] for r in ch]).to(dev),
            "elo_opp": torch.tensor([r[4] for r in ch]).to(dev),
            "temporal": torch.from_numpy(np.stack([r[2] for r in ch])).float().to(dev)}, ch


@torch.no_grad()
def phase_eval(model, rows, dev, tag):
    agg = {lab: {"n":0, 1:0, 3:0} for _,_,lab in BUCKETS}
    for s in range(0, len(rows), 256):
        batch, ch = batch_of(rows, range(s, min(s+256, len(rows))), dev)
        ml = model(batch)["move_logits"].cpu()
        for j, r in enumerate(ch):
            legal, actual, ply = r[6], r[5], r[8]
            if actual not in legal: continue
            lg = ml[j][legal]; ai = legal.index(actual)
            top = lg.topk(min(3, len(legal))).indices.tolist()
            for lo,hi,lab in BUCKETS:
                if lo <= ply < hi:
                    agg[lab]["n"] += 1
                    agg[lab][1] += int(lg.argmax().item() == ai); agg[lab][3] += int(ai in top); break
    print(f"[{tag}] {'phase':>11}{'n':>7}{'top1':>8}{'top3':>8}")
    for _,_,lab in BUCKETS:
        a = agg[lab]
        if a["n"]: print(f"[{tag}] {lab:>11}{a['n']:>7}{a[1]/a['n']*100:>7.1f}{a[3]/a['n']*100:>7.1f}")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", default="checkpoints/base_300k_best.pt")
    ap.add_argument("--test-games", type=int, default=40)
    ap.add_argument("--max-train-games", type=int, default=0, help="cap train to the N games right before the test split (0 = all)")
    ap.add_argument("--train-skip-recent", type=int, default=0,
                    help="drop the N games immediately before the test split, so training sits FURTHER "
                         "BACK IN TIME. Isolates within-player temporal drift: same player, same game "
                         "count, only the train->test gap changes.")
    ap.add_argument("--eval-pgn", default=None, help="cross-player control: also evaluate on THIS player's held-out games")
    ap.add_argument("--eval-name", default=None)
    ap.add_argument("--save-ft", default=None,
                    help="save the fine-tuned model here. Needed for any post-hoc check (1-NN/MMD, "
                         "self-play, UCI play) - without it the clone is discarded at exit.")
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=256); ap.add_argument("--w-time", type=float, default=0.2)
    ap.add_argument("--adapt", default="full",
                    choices=["full", "head", "lastblock", "bitfit", "norm", "film", "lhuc", "adapter"],
                    help="which parameters are personalised: full model, policy head only, last encoder "
                         "block+head, all biases (BitFit), all LayerNorm affines, the FiLM generator, "
                         "per-unit gains on every block (LHUC), or rank-r bottleneck adapters. "
                         "NOTE: lhuc/adapter add modules, so --save-ft stores the base weights only.")
    ap.add_argument("--adapter-rank", type=int, default=16, help="bottleneck width for --adapt adapter")
    ap.add_argument("--opponents", action="store_true",
                    help="train on the opponents' moves in these players' games (a realistic Lichess 'field' "
                         "opponent for simulations); combine with several --pgn/--name for a pooled field")
    ap.add_argument("--opponents-exclude-recent", type=int, default=80,
                    help="with --opponents: drop each player's newest N games (their test games)")
    ap.add_argument("--time-head-epochs", type=int, default=0,
                    help="after the fine-tune, train ONLY the think-time head this many more epochs (moves unchanged)")
    ap.add_argument("--time-head-lr", type=float, default=3e-4)
    ap.add_argument("--init-ft", default=None, help="start from this fine-tuned clone instead of the base weights")
    ap.add_argument("--time-loss", default="point", choices=["point", "bucket"],
                    help="point = original mdn_nll; bucket = whole-second clock likelihood (Lichess)")
    args = ap.parse_args()
    me = args.name.lower()
    dkey = lambda g: ((g.headers.get("UTCDate") or g.headers.get("Date") or ""),
                      (g.headers.get("UTCTime") or g.headers.get("StartTime") or ""))
    # --pgn/--name accept COMMA-SEPARATED lists to train on a pooled COHORT of players
    # (experiment B2: trait-matched vs trait-anti-matched cohorts, evaluated on a target via --eval-pgn).
    names = [n.strip() for n in args.name.split(",")]
    pgns = [p.strip() for p in args.pgn.split(",")]
    if len(names) != len(pgns):
        raise SystemExit("give exactly one --pgn per --name")
    pooled = len(names) > 1
    per = (args.max_train_games // len(names)) if args.max_train_games else 0
    tr_g, tr_rows, te_rows = [], [], []
    for nm, pg in zip(names, pgns):
        nml = nm.lower()
        gs = games_of(pg, nml); gs.sort(key=dkey)           # chronological: test = newest games
        keep = gs if (pooled or args.test_games == 0) else gs[:-args.test_games]   # --test-games 0 = train on everything
        if args.opponents:                                  # train on the OPPONENTS' side of these games
            keep = gs[:-args.opponents_exclude_recent]      # (never the player's test games)
        if args.train_skip_recent and not pooled:
            keep = keep[:max(0, len(keep) - args.train_skip_recent)]
        if per:
            keep = keep[-per:]
        tr_g += keep
        if args.opponents:
            other = lambda g: ((g.headers.get("Black") if nml == (g.headers.get("White", "") or "").lower()
                                else g.headers.get("White")) or "").lower()
            tr_rows += [r for g in keep for r in rows_of(g, other(g))]
        else:
            tr_rows += [r for g in keep for r in rows_of(g, nml)]
        if not pooled and args.test_games > 0:
            te_rows = [r for g in gs[-args.test_games:] for r in rows_of(g, nml)]
        print(f"{nm}: {len(gs)} games, using {len(keep)}", flush=True)
    x_rows = []
    if args.eval_pgn:
        xme = args.eval_name.lower(); xg = games_of(args.eval_pgn, xme); xg.sort(key=dkey)
        x_rows = [r for g in xg[-args.test_games:] for r in rows_of(g, xme)]
    print(f"train {len(tr_rows)} positions ({len(tr_g)} games), test {len(te_rows)} (last {args.test_games})")

    model, _ = load_model(args.ckpt); model.to(DEV)
    model.eval(); base_agg = phase_eval(model, te_rows, DEV, "BASE")
    if args.init_ft:                                    # continue from an already fine-tuned clone (e.g. --epochs 0 +
        model.load_state_dict(torch.load(args.init_ft, map_location=DEV, weights_only=False)["model_state"])  # time-head stage)
        print(f"initialised from {args.init_ft}", flush=True)
    x_base = phase_eval(model, x_rows, DEV, f"CROSS-BASE on {args.eval_name}") if x_rows else None

    extra = []                                   # per-user modules injected INTO the trunk (LHUC / adapters)
    if args.adapt in ("lhuc", "adapter"):
        for p in model.parameters():
            p.requires_grad_(False)
        dim = next(p.shape[0] for n, p in model.named_parameters() if n.endswith("t_to_d.weight"))
        for blk in model.encoder.blocks:
            if args.adapt == "lhuc":             # one learned gain per hidden unit, 2*sigmoid(0)=1 at start
                r = torch.nn.Parameter(torch.zeros(dim, device=DEV)); extra.append(r)
                blk.register_forward_hook(
                    lambda m, i, o, r=r: (o[0] if isinstance(o, tuple) else o) * (2 * torch.sigmoid(r)))
            else:                                # rank-r bottleneck adapter, zero-init up = identity at start
                d = torch.nn.Linear(dim, args.adapter_rank).to(DEV)
                u = torch.nn.Linear(args.adapter_rank, dim).to(DEV)
                torch.nn.init.zeros_(u.weight); torch.nn.init.zeros_(u.bias)
                extra += list(d.parameters()) + list(u.parameters())
                blk.register_forward_hook(
                    lambda m, i, o, d=d, u=u: (lambda x: x + u(F.gelu(d(x))))(o[0] if isinstance(o, tuple) else o))
    elif args.adapt != "full":                   # personalise only part of the existing net
        last = max(int(n.split(".")[2]) for n, _ in model.named_parameters() if n.startswith("encoder.blocks."))
        def trainable(n):
            if args.adapt == "head":      return n.startswith("policy")
            if args.adapt == "lastblock": return n.startswith(f"encoder.blocks.{last}.") or n.startswith("policy")
            if args.adapt == "bitfit":    return n.endswith(".bias")
            if args.adapt == "norm":      return "norm" in n.lower() or ".ln" in n.lower()
            if args.adapt == "film":      return n.startswith("film_gen")
            return True
        for n, p in model.named_parameters():
            p.requires_grad_(trainable(n))
    train_ps = extra if extra else [p for p in model.parameters() if p.requires_grad]
    print(f"adapt={args.adapt}: training {sum(p.numel() for p in train_ps)/1e6:.3f}M of "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M params", flush=True)
    opt = torch.optim.AdamW(train_ps, lr=args.lr, weight_decay=1e-4)
    N = len(tr_rows)
    for ep in range(args.epochs):
        model.train(); perm = np.random.permutation(N); tot = nb = 0
        for s in range(0, N, args.bs):
            idx = perm[s:s+args.bs]; batch, ch = batch_of(tr_rows, idx, DEV)
            out = model(batch)
            move = torch.tensor([r[5] for r in ch]).to(DEV)
            pol = F.cross_entropy(out["move_logits"], move)
            pi, mu, sg = out["mdn"]; think = torch.tensor([r[7] for r in ch]).float().to(DEV)
            tl = bucket_nll(pi, mu, sg, think) if args.time_loss == "bucket" else mdn_nll(pi, mu, sg, think)
            loss = pol + args.w_time * tl
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step(); tot += loss.item(); nb += 1
        print(f"epoch {ep+1}/{args.epochs} loss {tot/max(1,nb):.4f}", flush=True)

    if args.time_head_epochs > 0:
        # extra stage: only the think-time head keeps learning the player's pace; the trunk and move head are frozen,
        # so move predictions cannot change
        for p in model.parameters():
            p.requires_grad_(False)
        head_ps = [p for n, p in model.named_parameters() if n.startswith("think.")]
        for p in head_ps:
            p.requires_grad_(True)
        opt_t = torch.optim.AdamW(head_ps, lr=args.time_head_lr, weight_decay=1e-4)
        print(f"time-head stage: training {sum(p.numel() for p in head_ps)/1e3:.1f}k params for {args.time_head_epochs} epochs", flush=True)
        for ep in range(args.time_head_epochs):
            model.train(); perm = np.random.permutation(N); tot = nb = 0
            for s in range(0, N, args.bs):
                idx = perm[s:s+args.bs]; batch, ch = batch_of(tr_rows, idx, DEV)
                pi, mu, sg = model(batch)["mdn"]; think = torch.tensor([r[7] for r in ch]).float().to(DEV)
                tl = bucket_nll(pi, mu, sg, think) if args.time_loss == "bucket" else mdn_nll(pi, mu, sg, think)
                opt_t.zero_grad(); tl.backward(); opt_t.step(); tot += tl.item(); nb += 1
            print(f"time-head epoch {ep+1}/{args.time_head_epochs} time loss {tot/max(1,nb):.4f}", flush=True)

    model.eval(); ft_agg = phase_eval(model, te_rows, DEV, "FINETUNED")
    post = lambda a: sum(a[l][1] for _, _, l in BUCKETS[1:]) / max(1, sum(a[l]["n"] for _, _, l in BUCKETS[1:]))
    print(f"\n=== POST-OPENING (own ply>=12) top-1: base {post(base_agg)*100:.2f} -> ft {post(ft_agg)*100:.2f} "
          f"({(post(ft_agg)-post(base_agg))*100:+.2f}pp)  train_games={len(tr_g)} ===")
    if x_rows:
        x_ft = phase_eval(model, x_rows, DEV, f"CROSS-FT on {args.eval_name}")
        print(f"=== CROSS-PLAYER CONTROL: ft-on-{args.name} evaluated on {args.eval_name}: post-opening "
              f"{post(x_base)*100:.2f} -> {post(x_ft)*100:.2f} ({(post(x_ft)-post(x_base))*100:+.2f}pp) ===")
    if args.save_ft:
        os.makedirs(os.path.dirname(args.save_ft) or ".", exist_ok=True)
        torch.save({"model_state": model.state_dict(),
                    "meta": {"name": args.name, "base_ckpt": args.ckpt,
                             "train_games": len(tr_g), "epochs": args.epochs, "lr": args.lr}},
                   args.save_ft)
        print(f"saved fine-tuned clone -> {args.save_ft}")
    print(f"\n=== moveΔ (finetuned - base) by phase, top-1 / top-3 ===")
    for _,_,lab in BUCKETS:
        b, f_ = base_agg[lab], ft_agg[lab]
        if b["n"]:
            print(f"{lab:>12}: top1 {(f_[1]-b[1])/b['n']*100:+.1f}pp   top3 {(f_[3]-b[3])/b['n']*100:+.1f}pp")


if __name__ == "__main__":
    main()
