"""TRIAGE — the honest, independent evaluation harness for clone DISTRIBUTION quality.

Lesson from the audited-down DPO: NEVER report the metric you trained on. This harness reports only
INDEPENDENT metrics, on BIG held-out data, with sanity floors and a 2-seed noise band:

  headline:
    ID P@1   — are the clone's games classified as the target player, vs a panel of K players?
               floor = base self-play games, ceiling = the player's REAL held-out games.
    MMD      — two-sample distance real-vs-clone (lower = closer). floor = real-vs-real.
    1-NN     — real-vs-clone separability (0.5 = indistinguishable).

Generation methods (--method): self (standard self-play) | openings (seed from REAL opening positions).

    PYTHONPATH=. python scripts/clone_triage.py --name latebloomer --clone clones/latebloomer_pooled.pt
"""
import argparse, os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
import chess, chess.pgn, io, zstandard
from sahformer.encoding import encode_board, build_temporal
from sahformer.records import _stack_history, BASE_SECONDS
from sahformer.training.loop import load_model
from sahformer.clone import load_any_clone
from sahformer.play import self_play
from clone_judge import real_games, embed_games, mmd_rbf, discriminator_auc
from clone_dpo_pooled import base_feats_of_traj
from run_scale import GameEncoder

LSCALE = "data/lichess_scale"


def stem(f):
    return os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0]


def embed_selfplay(model, enc, adapter, elo, n, dev, tmean, tstd, seed, method="self", openings=None):
    """Generate n clone/base games; embed each with the recognizer."""
    embs = []
    got = 0; s = seed
    while got < n:
        color = "white" if got % 2 == 0 else "black"    # match real's color mix (both methods)
        op = openings[got % len(openings)] if openings else None
        rows = _one_game(model, adapter, elo, dev, s, method=method, color=color, opening=op)
        s += 1
        if rows is None or len(rows) < 8:
            continue
        pooled, _, _, think = base_feats_of_traj(model, rows, elo, dev)
        logt = ((torch.log1p(think.clamp(min=0)) - tmean) / tstd).unsqueeze(0)
        mask = torch.ones(1, pooled.shape[0], dtype=torch.bool, device=dev)
        embs.append(enc(pooled.unsqueeze(0), logt, mask).squeeze(0))
        got += 1
    return torch.stack(embs)


def load_openings(path, name, n, k=12):
    """Up to n lists of the first k plies (chess.Move) of real games of `name`, with the player's
    color. Used to SEED clone games from real openings so early positions match the real ones."""
    me = name.lower()
    text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(path, "rb")),
                            encoding="utf-8", errors="ignore")
    out = []
    while len(out) < n:
        g = chess.pgn.read_game(text)
        if g is None:
            break
        w = (g.headers.get("White", "") or "").lower(); b = (g.headers.get("Black", "") or "").lower()
        if me not in (w, b):
            continue
        board = g.board(); node = g; moves = []
        while node.variations and len(moves) < k:
            node = node.variation(0); moves.append(node.move); board.push(node.move)
        if len(moves) >= k:
            out.append((moves, me == w))
    return out


def _seeded_traj(model, adapter, elo, dev, seed, opening, max_plies=80):
    """Replay a real opening (real moves), then the CLONE continues both sides. Returns
    (traj, player_is_white). Time is dummy (moves-only recognizer ignores it)."""
    from sahformer.encoding import encode_move
    from sahformer.model.heads import move_to_index
    from sahformer.clone import apply_clone
    from sahformer.play import _sample_think_time
    moves, me_white = opening
    rng = np.random.default_rng(seed)
    board = chess.Board(); plane_hist = []
    think_hist = {chess.WHITE: [], chess.BLACK: []}; clock = {chess.WHITE: 180.0, chess.BLACK: 180.0}
    traj = []; ply = 0
    for mv in moves:                                   # replay REAL opening
        mover = board.turn
        traj.append((mv, mover == chess.WHITE, 2.0))
        think_hist[mover] = [2.0] + think_hist[mover]; plane_hist.append(encode_board(board)); board.push(mv); ply += 1
    while not board.is_game_over() and ply < max_plies:      # CLONE continues both sides
        mover = board.turn
        cur = encode_board(board); hist = _stack_history(plane_hist, cur)
        temporal = build_temporal(clock[mover], clock[not mover], think_hist[mover], ply)
        out = model({"board": torch.from_numpy(cur).float().unsqueeze(0).to(dev),
                     "history": torch.from_numpy(hist).float().unsqueeze(0).to(dev),
                     "elo_self": torch.tensor([elo]).to(dev), "elo_opp": torch.tensor([elo]).to(dev),
                     "temporal": torch.from_numpy(temporal).float().unsqueeze(0).to(dev)})
        if adapter is not None:
            out = apply_clone(out, adapter)
        think = _sample_think_time(out["mdn"], rng)
        legal = list(board.legal_moves)
        idxs = [move_to_index(*encode_move(board, m)) for m in legal]
        probs = F.softmax(out["move_logits"][0][idxs], dim=-1).detach().cpu().numpy(); probs /= probs.sum()
        order = np.argsort(probs)[::-1]; csum = np.cumsum(probs[order])
        keep = order[: int(np.searchsorted(csum, 0.9)) + 1]
        trunc = np.zeros_like(probs); trunc[keep] = probs[keep]; probs = trunc / trunc.sum()
        move = legal[int(rng.choice(len(legal), p=probs))]
        clock[mover] -= think
        traj.append((move, mover == chess.WHITE, think))
        if clock[mover] <= 0.0:
            break
        think_hist[mover] = [think] + think_hist[mover]; plane_hist.append(cur); board.push(move); ply += 1
    return traj, me_white


def _asym_traj(model, adapter, elo, dev, seed, max_plies=80, clone_white=True):
    """ASYMMETRIC playout: the clone plays ONE color, the BASE plays the other (generic opponent,
    closer to real's 'vs the field' than clone-vs-clone). Returns (move, is_white, think) plies."""
    from sahformer.encoding import encode_move
    from sahformer.model.heads import move_to_index
    from sahformer.clone import apply_clone
    from sahformer.play import _sample_think_time
    rng = np.random.default_rng(seed)
    board = chess.Board(); plane_hist = []
    think_hist = {chess.WHITE: [], chess.BLACK: []}
    clock = {chess.WHITE: 180.0, chess.BLACK: 180.0}
    traj = []; ply = 0
    while not board.is_game_over() and ply < max_plies:
        mover = board.turn
        cur = encode_board(board); hist = _stack_history(plane_hist, cur)
        temporal = build_temporal(clock[mover], clock[not mover], think_hist[mover], ply)
        batch = {"board": torch.from_numpy(cur).float().unsqueeze(0).to(dev),
                 "history": torch.from_numpy(hist).float().unsqueeze(0).to(dev),
                 "elo_self": torch.tensor([elo]).to(dev), "elo_opp": torch.tensor([elo]).to(dev),
                 "temporal": torch.from_numpy(temporal).float().unsqueeze(0).to(dev)}
        out = model(batch)
        if adapter is not None and (mover == chess.WHITE) == clone_white:   # clone ONLY on its color
            out = apply_clone(out, adapter)
        think = _sample_think_time(out["mdn"], rng)
        logits = out["move_logits"][0]
        legal = list(board.legal_moves)
        idxs = [move_to_index(*encode_move(board, m)) for m in legal]
        probs = F.softmax(logits[idxs], dim=-1).detach().cpu().numpy(); probs /= probs.sum()
        order = np.argsort(probs)[::-1]; csum = np.cumsum(probs[order])
        keep = order[: int(np.searchsorted(csum, 0.9)) + 1]
        trunc = np.zeros_like(probs); trunc[keep] = probs[keep]; probs = trunc / trunc.sum()
        move = legal[int(rng.choice(len(legal), p=probs))]
        clock[mover] -= think
        traj.append((move, mover == chess.WHITE, think))
        if clock[mover] <= 0.0:
            break
        think_hist[mover] = [think] + think_hist[mover]
        plane_hist.append(cur); board.push(move); ply += 1
    return traj


def _one_game(model, adapter, elo, dev, seed, max_plies=80, method="self", color="white", opening=None):
    """One game -> the chosen color's own-ply rows (board,hist,temporal,chosen_idx,think).
    method=self: clone plays both sides. vs_base: clone one color, base other.
    openings: seed from a real opening, then clone continues (positions match the real ones)."""
    from sahformer.encoding import encode_move
    from sahformer.model.heads import move_to_index
    if method == "openings":
        traj, me_white = _seeded_traj(model, adapter, elo, dev, seed, opening, max_plies)
        color = "white" if me_white else "black"       # extract the player's real color
    elif method == "vs_base":
        traj = _asym_traj(model, adapter, elo, dev, seed, max_plies, clone_white=(color == "white"))
    else:
        traj = [(rec["move"], rec["mover"] == "white", rec["think"])
                for rec in self_play(model, max_plies=max_plies, elo=elo, temperature=1.0, top_p=0.9,
                                     seed=seed, adapter=adapter, device=dev)]
    want_white = (color == "white")
    board = chess.Board(); ph = []; rows = []; ply = 0
    prev = {True: BASE_SECONDS, False: BASE_SECONDS}; thist = {True: [], False: []}
    for mv, mover_white, think in traj:
        cur = encode_board(board)
        if mover_white == want_white:
            frm, to, pr = encode_move(board, mv)
            rows.append((cur, _stack_history(ph, cur),
                         build_temporal(prev[True], prev[False], thist[True], ply),
                         move_to_index(frm, to, pr), think))
        prev[mover_white] = max(prev[mover_white] - think, 0.0)
        thist[mover_white] = [think] + thist[mover_white]
        ph.append(cur); board.push(mv); ply += 1
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--clone", default=None, help="clone .pt; omit to evaluate BASE only")
    ap.add_argument("--recognizer", default="checkpoints/recognizer_film_moves.pt")
    ap.add_argument("--panel", type=int, default=20, help="other players in the ID panel")
    ap.add_argument("--real-games", type=int, default=150)
    ap.add_argument("--panel-games", type=int, default=60)
    ap.add_argument("--clone-games", type=int, default=80)
    ap.add_argument("--elo", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--method", choices=["self", "vs_base", "openings"], default="self",
                    help="clone generation: self | vs_base | openings (seed from real openings, clone continues)")
    ap.add_argument("--open-plies", type=int, default=12, help="how many real opening plies to seed")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model, _ = load_model("checkpoints/base_300k_best.pt"); model.eval().to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    rck = torch.load(args.recognizer, weights_only=False)
    enc = GameEncoder(rck["dp"], rck["use_time"], d=rck["d_model"], layers=rck["layers"],
                      heads=rck["heads"], L=rck["max_plies"]).to(dev)
    enc.load_state_dict(rck["state_dict"]); enc.eval()
    tmean, tstd = rck["tmean"], rck["tstd"]

    def emb_real(path, nm, n):
        g = real_games(path, nm, n, args.elo)
        return embed_games(g, model, enc, tmean, tstd, dev) if g else torch.zeros(0, rck["d_model"], device=dev)

    # cache the expensive, experiment-invariant clouds (real/panel/base); only CLONE regenerates
    ck_path = (f"triage_cache_{args.name}_p{args.panel}_r{args.real_games}_"
               f"g{args.panel_games}_{os.path.splitext(os.path.basename(args.recognizer))[0]}.pt")
    if os.path.exists(ck_path):
        d = torch.load(ck_path, weights_only=False)
        Xr_all, perm, C, Xbase = d["Xr_all"].to(dev), d["perm"], d["C"].to(dev), d["Xbase"].to(dev)
        others = d["others"]
    else:
        Xr_all = emb_real(f"{LSCALE}/{args.name}.pgn.zst", args.name, args.real_games)
        perm = torch.randperm(len(Xr_all))
        Xr_cent = Xr_all[perm[:len(Xr_all)//2]]
        others = [f for f in sorted(glob.glob(f"{LSCALE}/*.pgn.zst"), key=os.path.getsize, reverse=True)
                  if stem(f) != args.name][:args.panel]
        cents = [F.normalize(Xr_cent.mean(0), dim=-1)]            # index 0 = target
        for f in others:
            e = emb_real(f, stem(f), args.panel_games)
            cents.append(F.normalize(e.mean(0), dim=-1) if len(e) else torch.zeros(rck["d_model"], device=dev))
        C = torch.stack(cents)
        Xbase = embed_selfplay(model, enc, None, args.elo, args.clone_games, dev, tmean, tstd, 7000)
        torch.save({"Xr_all": Xr_all.cpu(), "perm": perm, "C": C.cpu(), "Xbase": Xbase.cpu(), "others": others}, ck_path)

    half = len(Xr_all) // 2
    Xr_cent = Xr_all[perm[:half]]; Xr_test = Xr_all[perm[half:]]   # disjoint centroid vs eval-cloud

    def id_p1(X):                                                  # fraction classified as target (row 0)
        return float((X @ C.T).argmax(1).eq(0).float().mean())

    print(f"=== TRIAGE  {args.name}  | recognizer {os.path.basename(args.recognizer)} | "
          f"panel {len(others)+1} players | clone-method {args.method} ===")
    print(f"real games {len(Xr_all)} (cent {len(Xr_cent)} / test {len(Xr_test)}) | clone-games {args.clone_games} x {args.seeds} seeds\n")
    print(f"{'cloud':<16}{'ID P@1':>9}{'MMD':>10}{'1-NN':>8}")
    print(f"{'REAL (ceiling)':<16}{id_p1(Xr_test):>9.3f}{mmd_rbf(Xr_test.cpu(),Xr_cent.cpu()):>10.4f}"
          f"{discriminator_auc(Xr_test.cpu(),Xr_cent.cpu()):>8.3f}   <- floors: 1-NN~0.5, MMD~0")
    print(f"{'BASE (floor)':<16}{id_p1(Xbase):>9.3f}{mmd_rbf(Xr_test.cpu(),Xbase.cpu()):>10.4f}"
          f"{discriminator_auc(Xr_test.cpu(),Xbase.cpu()):>8.3f}")

    if args.clone:
        adapter = load_any_clone(args.clone, device=dev)
        openings = None
        if args.method == "openings":
            openings = load_openings(f"{LSCALE}/{args.name}.pgn.zst", args.name, args.clone_games, args.open_plies)
        p1s, mmds, nns = [], [], []
        for sd in range(args.seeds):
            Xc = embed_selfplay(model, enc, adapter, args.elo, args.clone_games, dev, tmean, tstd, 1000 + sd * 3000, method=args.method, openings=openings)
            p1s.append(id_p1(Xc)); mmds.append(mmd_rbf(Xr_test.cpu(), Xc.cpu())); nns.append(discriminator_auc(Xr_test.cpu(), Xc.cpu()))
        print(f"{'CLONE':<16}{np.mean(p1s):>9.3f}{np.mean(mmds):>10.4f}{np.mean(nns):>8.3f}   "
              f"(±{np.std(p1s):.3f} / ±{np.std(mmds):.4f} / ±{np.std(nns):.3f} over {args.seeds} seeds)")
        print(f"\nread: ID P@1 CLONE should rise toward REAL and above BASE; MMD/1-NN should fall toward REAL.")


if __name__ == "__main__":
    main()
