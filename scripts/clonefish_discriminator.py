"""Can a model tell the CLONE's own games from the PLAYER's? If so, that difference is the thing to correct.

Why this and not the push / identity head: both of those were scored by move likelihood on the player's real
positions — the exact objective the clone was fine-tuned with — so lambda = 1 was optimal by construction and no
amplification could ever win. That test was circular. Self-play games are the data maximum likelihood never sees,
and they are where the clone demonstrably fails (games 21 plies short, resignations in the wrong positions).

So: train D(position, move) -> "did this come from the player or from the clone", on the clone's GENERATED games
vs the player's real ones. Same trick as the identity head — frozen trunk, small policy-shaped head reading out a
scalar per move — but the label is provenance, not the move itself.

  AUC ~ 0.5  -> the clone's moves are already indistinguishable; there is nothing here to amplify.
  AUC >> control -> D has found where the clone stops looking like the player, and `clone + beta * D_logit` is a
                    learned, position-dependent correction (unlike clone - stock, which only points away from the
                    norm the fine-tune already moved away from).

CONTROL matters: real and generated games are played against different opponents, so D could be reading the
OPPONENT's style rather than the player's. The real-vs-real control (two halves of the player's own games) gives
the floor that any real signal has to beat.

    PYTHONPATH=. python scripts/clonefish_discriminator.py --player VEGETAL \
        --clone clones/ftval_VEGETAL_bucket.pt --gen results/clonefish/VEGETAL_CLONE_final.pgn
"""
import argparse, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np, torch, torch.nn.functional as F, chess, chess.pgn
from sklearn.metrics import roc_auc_score
from finetune_clone import games_of, rows_of, batch_of
from clonefish_uci import IdentityHead, hook_enc
from sahformer.training.loop import load_model

CKPT = os.path.join(ROOT, "checkpoints", "base_300k_best.pt")
dkey = lambda g: ((g.headers.get("UTCDate") or g.headers.get("Date") or ""),
                  (g.headers.get("UTCTime") or g.headers.get("StartTime") or ""))


def read_pgn(path):
    out = []
    with open(path, encoding="utf-8") as f:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                return out
            out.append(g)


def rows_labeled(games, me, label, side="player"):
    """(row, label, game index) for every own move with a clock.
    side='opp' scores the OPPONENT's moves instead — real opponents vs the simulated field model. Three separate
    measurements say the simulated opponent is what distorts game length (it resigns 22.7 % of games vs real
    opponents' 7.5 %, flags 13.3 % vs 28.8 %), and D has only ever looked at the player's own side."""
    R = []
    for i, g in enumerate(games):
        if g.board().fen() != chess.STARTING_FEN:
            continue
        nm = me
        if side == "opp":
            w = (g.headers.get("White", "") or "").lower()
            nm = ((g.headers.get("Black") if w == me else g.headers.get("White")) or "").lower()
        for r in rows_of(g, nm):
            if r[5] in r[6] and len(r[6]) > 1:
                R.append((r, label, i))
    return R


class PosDisc(torch.nn.Module):
    """Control that never sees WHICH MOVE was played — only the position. The clone reaches different states than
    the player (different opponent, shorter games), and a per-move D can separate those without knowing anything
    about move choice. Only (move AUC - position AUC) is a guidance direction; the rest is state drift."""

    def __init__(self, dim, hid=32):
        super().__init__()
        self.l1 = torch.nn.Linear(dim, hid); self.l2 = torch.nn.Linear(hid, 1)

    def forward(self, enc):
        return self.l2(F.relu(self.l1(enc.mean(1)))).squeeze(-1)


def train_eval(pairs, model, box, dim, dev, a, tag, save=None, mode="move"):
    """pairs = [(row, label, game id)]; split BY GAME, train D, return held-out AUC"""
    gids = sorted({(lb, gi) for _, lb, gi in pairs})
    rng = np.random.default_rng(0); fold = {k: rng.random() < 0.25 for k in gids}
    tr = [p for p in pairs if not fold[(p[1], p[2])]]
    te = [p for p in pairs if fold[(p[1], p[2])]]
    if not tr or not te or len({p[1] for p in te}) < 2:
        print(f"  [{tag}] not enough data", flush=True); return None
    D = (IdentityHead(dim, a.hid) if mode == "move" else PosDisc(dim, a.hid)).to(dev)
    opt = torch.optim.AdamW(D.parameters(), lr=a.lr, weight_decay=a.wd)

    def scores(batch_rows):
        rows = [p[0] for p in batch_rows]
        b, ch = batch_of(rows, range(len(rows)), dev)
        with torch.no_grad():
            model(b)
        s = D(box["enc"].float())
        return s if mode == "position" else torch.stack([s[j, r[5]] for j, r in enumerate(ch)])

    best = 0.0
    for ep in range(a.epochs):
        D.train(); perm = np.random.default_rng(ep).permutation(len(tr)); tot = n = 0
        for i in range(0, len(tr) - a.bs + 1, a.bs):
            ch = [tr[j] for j in perm[i:i + a.bs]]
            y = torch.tensor([float(p[1]) for p in ch], device=dev)
            loss = F.binary_cross_entropy_with_logits(scores(ch), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(ch); n += len(ch)
        D.eval()
        with torch.no_grad():
            sc, ys = [], []
            for i in range(0, len(te), a.bs):
                ch = te[i:i + a.bs]
                sc += scores(ch).cpu().tolist(); ys += [p[1] for p in ch]
        auc = roc_auc_score(ys, sc)
        best = max(best, auc)       # compare BEST-over-epochs on both arms: the control wanders by +-0.04 on a
        print(f"  [{tag}] epoch {ep + 1}: train BCE {tot / max(1, n):.4f} | held-out AUC {auc:.3f} "
              f"({sum(ys)} player / {len(ys) - sum(ys)} other moves)", flush=True)
    if save:        # keep D: if it separates, its logit IS the guidance direction (clone + beta * D_logit)
        torch.save({"head": D.state_dict(), "hid": a.hid, "dim": dim, "kind": "discriminator", "auc": best}, save)
        print(f"  [{tag}] saved {os.path.relpath(save, ROOT)}", flush=True)
    return best     # smaller sample, so taking its LAST epoch understates the floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", required=True); ap.add_argument("--clone", required=True)
    ap.add_argument("--gen", required=True, help="PGN of the clone's generated games (eval arm output)")
    ap.add_argument("--data-dir", default=os.path.join("data", "lichess_5k"))
    ap.add_argument("--real-games", type=int, default=400)
    ap.add_argument("--hid", type=int, default=32); ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--keep-elo", action="store_true", help="do NOT equalise the Elo inputs (shows the leak)")
    ap.add_argument("--phase", default=None,
                    help="restrict to own moves in [lo,hi), e.g. 0-10 / 10-25 / 25-999 — finds WHERE the games "
                         "diverge, since a clone can pass every bar and still drift in one phase")
    ap.add_argument("--side", default="player", choices=["player", "opp"],
                    help="whose moves to judge: the clone vs the player, or the simulated opponent vs real opponents")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "clonefish", "discriminator.json"))
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    me = a.player.lower(); t0 = time.time()

    model, mcfg = load_model(CKPT)
    model.load_state_dict(torch.load(a.clone, map_location="cpu", weights_only=False)["model_state"])
    model.eval().to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    box = hook_enc(model)

    gs = games_of(os.path.join(ROOT, a.data_dir, f"{a.player}.pgn.zst"), me); gs.sort(key=dkey)
    real = gs[-a.real_games:]
    gen = read_pgn(a.gen)
    print(f"{a.player}: {len(real)} real games vs {len(gen)} generated games", flush=True)

    real_rows = rows_labeled(real, me, 1, a.side)
    gen_rows = rows_labeled(gen, "player", 0, a.side)
    if a.phase:
        lo, hi = (int(x) for x in a.phase.split("-"))
        keep = lambda R: [(r, lb, gi) for r, lb, gi in R if lo <= r[8] < hi]
        real_rows, gen_rows = keep(real_rows), keep(gen_rows)
        print(f"  phase {a.phase}: {len(real_rows)} real / {len(gen_rows)} generated moves", flush=True)
    if not a.keep_elo:
        # real games carry each game's true ratings, generated games all use one average — and those feed the
        # trunk through the Elo embedding, so D could read "Elo == the mean" instead of anything about the moves
        es = int(np.median([r[0][3] for r in real_rows])); eo = int(np.median([r[0][4] for r in real_rows]))
        fix = lambda R: [((*r[:3], es, eo, *r[5:]), lb, gi) for r, lb, gi in R]
        real_rows, gen_rows = fix(real_rows), fix(gen_rows)
        print(f"  Elo inputs forced to {es}/{eo} on BOTH classes", flush=True)
    n = min(len(real_rows), len(gen_rows))
    rng = np.random.default_rng(1)
    real_rows = [real_rows[i] for i in rng.permutation(len(real_rows))[:n]]
    gen_rows = [gen_rows[i] for i in rng.permutation(len(gen_rows))[:n]]
    print(f"  balanced to {n} moves per side", flush=True)

    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    auc = train_eval(real_rows + gen_rows, model, box, mcfg.dim_vit, dev, a, "real vs clone-generated",
                     save=os.path.join(ROOT, "clones", f"{a.player}_disc_{a.side}.pt"))
    pos = train_eval(real_rows + gen_rows, model, box, mcfg.dim_vit, dev, a, "control: position only", mode="position")

    # control: the player's own games split in half and labelled as if they were two different sources.
    # Anything the real discriminator scores above this is a genuine clone/player difference.
    half = len(real_rows) // 2
    ctrl_rows = [(r, 1, g) for r, _, g in real_rows[:half]] + [(r, 0, g) for r, _, g in real_rows[half:]]
    ctrl = train_eval(ctrl_rows, model, box, mcfg.dim_vit, dev, a, "control: real vs real")

    res[f"{a.player}:{a.side}:{a.phase or 'all'}"] = {"auc_real_vs_generated": auc, "auc_position_only": pos,
                                                      "auc_control": ctrl, "n_per_side": n, "gen": a.gen,
                                                      "side": a.side, "phase": a.phase or "all"}
    json.dump(res, open(a.out, "w"), indent=1)
    floor = max(x for x in (pos, ctrl) if x is not None)
    print(f"\n{a.player}: move AUC {auc:.3f} | position-only {pos:.3f} | real-vs-real {ctrl:.3f}", flush=True)
    print(f"  usable (move - position) = {auc - pos:+.3f} -> "
          f"{'GUIDANCE DIRECTION' if auc - pos > 0.03 else 'state drift only, nothing to steer moves with'}"
          f"  [floor {floor:.3f}]  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
