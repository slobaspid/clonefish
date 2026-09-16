"""Line-up test: does the CosFace recognizer pick a fine-tuned clone's games out as THAT player?

Panel = lichess_5k players + extra lichess_1k players (same 1500-1900 band), each enrolled by the mean
recognizer embedding of 60 real games. Query arms per target:
  REAL        newest 40 real games (positive control)       BASE        base self-play at target Elo
  CLONE_SELF  clone self-play                                CLONE_VSB   clone vs base opponent
Games are generated ONCE and kept as raw trajectories; they are then featurized under a ladder of settings
so every setting scores the identical games:
  S0 original | S1 + generated games cut to real game lengths (engines never resign)
  S2 + one neutral Elo fed to the featurizer for every game | S3 + one neutral think-time on every move
  S4 = S3 scored by the moves-only recognizer
Targets' enrollment games come from BEFORE the newest 80, so real queries are never enrolled.
"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F, chess, chess.pgn
from sahformer.encoding import encode_board, encode_move, build_temporal
from sahformer.records import _stack_history, BASE_SECONDS
from sahformer.training.loop import load_model
from sahformer.model.heads import move_to_index
from sahformer.play import self_play, _sample_think_time
from finetune_clone import games_of
from clone_judge import side_rows, embed_games
from run_scale import GameEncoder

OWN_PLIES = 40
dkey = lambda g: ((g.headers.get("UTCDate") or g.headers.get("Date") or ""),
                  (g.headers.get("UTCTime") or g.headers.get("StartTime") or ""))
SETTINGS = {  # name: (match_length, neutral_elo, neutral_time, recognizer key)
    "S0_orig": (False, None, None, "film"), "S1_len": (True, None, None, "film"),
    "S2_len_elo": (True, 1700, None, "film"), "S3_len_elo_time": (True, 1700, 2.0, "film"),
    "S4_movesonly": (True, 1700, 2.0, "moves"),
}


def game_record(g, me):
    """raw trajectory record: (traj, me_white, my_elo, opp_elo) or None for non-standard/short games."""
    if g.headers.get("FEN") or (g.headers.get("Variant", "Standard") or "Standard") != "Standard" \
            or g.board().fen() != chess.STARTING_FEN:
        return None
    me_white = (me == (g.headers.get("White", "") or "").lower())
    we = int(g.headers.get("WhiteElo", 0) or 1500); be = int(g.headers.get("BlackElo", 0) or 1500)
    board, node, traj = g.board(), g, []
    prev = {chess.WHITE: BASE_SECONDS, chess.BLACK: BASE_SECONDS}
    while node.variations:
        node = node.variation(0); mv = node.move; mover = board.turn; ca = node.clock()
        think = max(prev[mover] - ca, 0.0) if ca is not None else 0.0
        traj.append((mv, mover == chess.WHITE, think))
        if ca is not None: prev[mover] = ca
        board.push(mv)
    own = sum(1 for _, w, _ in traj if w == me_white)
    if own < 8: return None
    return (traj, me_white, we if me_white else be, be if me_white else we)


def load_player(path, name, n_enroll, target):
    gs = games_of(path, name.lower()); gs.sort(key=dkey)
    conv = lambda glist: [r for r in (game_record(g, name.lower()) for g in glist) if r]
    if target:
        # enroll on the VALIDATION games [-80,-40): finetune_clone --train-skip-recent 40 kept them out of
        # the clone's training set. (Earlier [:-80][-60:] was INSIDE the training window = leak toward the clone.)
        return conv(gs[-80:-40]), conv(gs[-40:])
    return conv(gs[-n_enroll:]), []


def featurize(records, setting, lengths=None, seed=0):
    match_len, n_elo, n_time, _ = setting
    rng = np.random.default_rng(seed); out = []
    for traj, me_white, my, opp in records:
        if n_time is not None: traj = [(m, w, n_time) for m, w, _ in traj]
        if n_elo is not None: my = opp = n_elo
        rows = side_rows(traj, me_white, my, opp)[:OWN_PLIES]
        if match_len and lengths: rows = rows[:max(8, int(rng.choice(lengths)))]
        if len(rows) >= 8: out.append(rows)
    return out


@torch.no_grad()
def vs_base_traj(clone, base, elo, opp_elo, dev, seed, clone_white, max_plies=100):
    rng = np.random.default_rng(seed); board = chess.Board(); ph = []
    thist = {chess.WHITE: [], chess.BLACK: []}; clock = {chess.WHITE: 180.0, chess.BLACK: 180.0}
    traj, ply = [], 0
    while not board.is_game_over() and ply < max_plies:
        mover = board.turn; is_clone = (mover == chess.WHITE) == clone_white
        cur = encode_board(board); hist = _stack_history(ph, cur)
        es, eo = (elo, opp_elo) if is_clone else (opp_elo, elo)
        batch = {"board": torch.from_numpy(cur).float()[None].to(dev),
                 "history": torch.from_numpy(hist).float()[None].to(dev),
                 "elo_self": torch.tensor([es]).to(dev), "elo_opp": torch.tensor([eo]).to(dev),
                 "temporal": torch.from_numpy(build_temporal(clock[mover], clock[not mover],
                                                             thist[mover], ply)).float()[None].to(dev)}
        out = (clone if is_clone else base)(batch)
        think = _sample_think_time(out["mdn"], rng)
        legal = list(board.legal_moves)
        idx = [move_to_index(*encode_move(board, m)) for m in legal]
        p = F.softmax(out["move_logits"][0][idx], -1).cpu().numpy().astype(np.float64); p /= p.sum()
        order = np.argsort(p)[::-1]; keep = order[: int(np.searchsorted(np.cumsum(p[order]), 0.9)) + 1]
        q = np.zeros_like(p); q[keep] = p[keep]; q /= q.sum()
        mv = legal[int(rng.choice(len(legal), p=q))]
        clock[mover] -= think; traj.append((mv, mover == chess.WHITE, think))
        if clock[mover] <= 0: break
        thist[mover] = [think] + thist[mover]; ph.append(cur); board.push(mv); ply += 1
    return traj


def gen_records(kind, gen, base, elo, opp_elo, n, dev, seed0):
    out = []
    for s in range(seed0, seed0 + n):
        if kind == "self":
            traj = [(r["move"], r["mover"] == "white", r["think"])
                    for r in self_play(gen, max_plies=100, elo=elo, temperature=1.0, top_p=0.9, seed=s, device=dev)]
            me_white = True
        else:
            me_white = (s % 2 == 0)
            traj = vs_base_traj(gen, base, elo, opp_elo, dev, s, me_white)
        out.append((traj, me_white, elo, opp_elo))
    return out


def score(Q, C, t):
    Qn, Cn = F.normalize(Q, dim=-1), F.normalize(C, dim=-1)
    k5 = torch.stack([Qn[i:i + 5].mean(0) for i in range(0, len(Qn) - 4, 5)])
    sims = F.normalize(k5, dim=-1) @ Cn.T
    rank = (sims > sims[:, t:t + 1]).sum(1) + 1
    return {"n": len(k5), "P@1": round(float((rank == 1).float().mean()), 3),
            "top3": round(float((rank <= 3).float().mean()), 3), "rank": round(float(rank.float().mean()), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, help="NAME:ft_ckpt,NAME:ft_ckpt")
    ap.add_argument("--extra", type=int, default=24)
    ap.add_argument("--enroll", type=int, default=60); ap.add_argument("--gen-games", type=int, default=60)
    ap.add_argument("--rec-film", default="checkpoints/recognizer_film.pt")
    ap.add_argument("--rec-moves", default="checkpoints/recognizer_film_moves.pt")
    ap.add_argument("--ckpt", default="checkpoints/base_300k_best.pt")
    ap.add_argument("--out", default="results/night_0913/lineup_ladder.json")
    ap.add_argument("--games-dir", default=None, help="score saved engine games <dir>/<NAME>_<ARM>.pgn instead of generating")
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    targets = [t.split(":") for t in a.targets.split(",")]; tnames = [t[0] for t in targets]
    base, _ = load_model(a.ckpt); base.eval().to(dev)
    encs = {}
    for key, path in (("film", a.rec_film), ("moves", a.rec_moves)):
        ck = torch.load(path, weights_only=False)
        e = GameEncoder(ck["dp"], ck["use_time"], d=ck["d_model"], layers=ck["layers"], heads=ck["heads"],
                        L=ck["max_plies"]).to(dev)
        e.load_state_dict(ck["state_dict"]); e.eval(); encs[key] = (e, ck["tmean"], ck["tstd"])
        print(f"recognizer {key}: {path} ({ck['loss']}, use_time={ck['use_time']})", flush=True)

    stem = lambda f: os.path.basename(f).split(".")[0]
    five = {stem(f): f for f in glob.glob("data/lichess_5k/*.pgn.zst")}
    ones = sorted(f for f in glob.glob("data/lichess_1k/*.pgn.zst") if stem(f) not in five)[:a.extra]
    panel = list(five.items()) + [(stem(f), f) for f in ones]
    names = [n for n, _ in panel]; enroll, arms = {}, {}
    for nm, path in panel:
        enroll[nm], q = load_player(path, nm, a.enroll, nm in tnames)
        if q: arms[nm] = {"REAL": q}
    print(f"panel {len(names)} players loaded, chance P@1 = {1/len(names):.3f}", flush=True)

    for nm, ftp in targets:
        if a.games_dir:                      # score the clonefish ENGINE's saved games (clonefish_eval.py)
            for arm in ("BASE", "CLONE", "CLONE_P100", "CLONE_OLD"):
                p = os.path.join(a.games_dir, f"{nm}_{arm}.pgn")
                if not os.path.exists(p): continue
                recs, fh = [], open(p, encoding="utf-8")
                while (g := chess.pgn.read_game(fh)) is not None:
                    r = game_record(g, "player")
                    if r: recs.append(r)
                arms[nm][arm] = recs
            print(f"loaded engine games for {nm}: " + ", ".join(f"{k} {len(v)}" for k, v in arms[nm].items()), flush=True)
            continue
        q = arms[nm]["REAL"]
        elo, oelo = int(np.mean([r[2] for r in q])), int(np.mean([r[3] for r in q]))
        gen, _ = load_model(a.ckpt)
        gen.load_state_dict(torch.load(ftp, map_location="cpu", weights_only=False)["model_state"]); gen.eval().to(dev)
        arms[nm]["BASE"] = gen_records("self", base, base, elo, oelo, a.gen_games, dev, 5000)
        arms[nm]["CLONE_SELF"] = gen_records("self", gen, base, elo, oelo, a.gen_games, dev, 5000)
        arms[nm]["CLONE_VSB"] = gen_records("vsb", gen, base, elo, oelo, a.gen_games, dev, 5000)
        del gen; torch.cuda.empty_cache()
        print(f"generated games for {nm} (elo {elo}, opp {oelo})", flush=True)

    res = {"panel": names, "chance": round(1 / len(names), 3), "settings": {}}
    for sname, st in SETTINGS.items():
        enc, tm, ts = encs[st[3]]
        C = torch.stack([embed_games(featurize(enroll[n], st), base, enc, tm, ts, dev).mean(0) for n in names])
        res["settings"][sname] = {}
        for nm, _ in targets:
            lengths = [min(OWN_PLIES, sum(1 for _, w, _ in r[0] if w == r[1])) for r in enroll[nm]]
            row = {}
            for arm, recs in arms[nm].items():
                feats = featurize(recs, st, lengths if arm != "REAL" else None, seed=11)
                row[arm] = score(embed_games(feats, base, enc, tm, ts, dev), C, names.index(nm))
            res["settings"][sname][nm] = row
            print(f"[{sname}] {nm:<8} " + "  ".join(f"{k} P@1 {v['P@1']:.2f} r {v['rank']:.1f}"
                                                   for k, v in row.items()), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1); print("wrote", a.out)


if __name__ == "__main__":
    main()
