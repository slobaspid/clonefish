"""Learn the player's identity as a MODEL on top of a frozen network — not a scalar push, not a feature list.

The norm push (`clone + w*(clone - stock)`) is one number for every position, phase and clock, and fitting it by
self-play ACPL was seed-noisy. A hand-written move-family list has the opposite problem: it can only find the
patterns someone thought to name. So instead: freeze the model, hook the trunk's 64 square representations, and
train a small head on top whose only job is to say how THIS player deviates from what the frozen model would do.
Whatever the pattern is, it has to find it in the trunk's own representation of the position.

The head has the same factorised shape as the real policy head (from/to bilinear over the square tokens + promotion),
just small, and it is initialised as a no-op so it starts exactly at the base distribution and must earn every
deviation. Identity is then the head's weights, and `lambda` scales how pronounced it is:

    score = base_logits + lambda * head(enc)

lambda = 1 reproduces the player as measured on their real moves; lambda > 1 is the nonlinear, learned analogue of
pushing away from the norm — the personal patterns get more pronounced instead of everything getting sharper.

    PYTHONPATH=. python scripts/clonefish_identity_head.py --player VEGETAL --clone clones/ftval_VEGETAL_bucket.pt
"""
import argparse, json, math, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, chess
from finetune_clone import games_of, rows_of, batch_of
from clonefish_uci import IdentityHead, hook_enc          # one definition, shared with the engine
from sahformer.training.loop import load_model

CKPT = os.path.join(ROOT, "checkpoints", "base_300k_best.pt")
dkey = lambda g: ((g.headers.get("UTCDate") or g.headers.get("Date") or ""),
                  (g.headers.get("UTCTime") or g.headers.get("StartTime") or ""))


def rows_for(games, me):
    rows = []
    for g in games:
        if g.board().fen() != chess.STARTING_FEN:
            continue
        rows += [r for r in rows_of(g, me) if r[5] in r[6] and len(r[6]) > 1]
    return rows


def pack(chunk, dev):
    """legal-move index matrix, mask and the index of the move the player actually played"""
    L = max(len(r[6]) for r in chunk)
    idx = torch.zeros(len(chunk), L, dtype=torch.long)
    mask = torch.zeros(len(chunk), L, dtype=torch.bool)
    tgt = torch.zeros(len(chunk), dtype=torch.long)
    for j, r in enumerate(chunk):
        n = len(r[6])
        idx[j, :n] = torch.tensor(r[6]); mask[j, :n] = True; tgt[j] = r[6].index(r[5])
    return idx.to(dev), mask.to(dev), tgt.to(dev)


def over_legal(logits, idx, mask):
    return logits.gather(1, idx).masked_fill(~mask, -1e9)


@torch.no_grad()
def base_pass(model, box, batch, want_enc):
    out = model(batch)
    return out["move_logits"].float(), (box["enc"].float() if want_enc else None)


def evaluate(models, boxes, head, rows, dev, lams, bs=256):
    """NLL / top-1 / top-3 of the player's real moves, for each base model and each lambda"""
    acc = {k: {l: [0.0, 0, 0, 0] for l in lams} for k in models}
    for s in range(0, len(rows), bs):
        chunk = rows[s:s + bs]
        batch, ch = batch_of(rows, range(s, min(s + bs, len(rows))), dev)
        idx, mask, tgt = pack(ch, dev)
        encs = {}
        for k, m in models.items():
            lg, en = base_pass(m, boxes[k], batch, True)
            encs[k] = (lg, en)
        with torch.no_grad():
            d = {k: head(encs[k][1]) for k in models}
        for k in models:
            for l in lams:
                s_ = over_legal(encs[k][0] + l * d[k], idx, mask)
                lp = F.log_softmax(s_, -1)
                a = acc[k][l]
                a[0] += -lp.gather(1, tgt[:, None]).sum().item()
                top = lp.topk(3, -1).indices
                a[1] += (top[:, 0] == tgt).sum().item()
                a[2] += (top == tgt[:, None]).any(1).sum().item()
                a[3] += len(ch)
    return {k: {l: (a[0] / a[3], a[1] / a[3], a[2] / a[3]) for l, a in v.items()} for k, v in acc.items()}


def fit_head(train, val, base_m, base_box, dim, a, dev, seed=0):
    """Train one head on `train`, keeping the epoch with the best NLL on `val` (val may be None)."""
    head = IdentityHead(dim, a.hid).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=a.wd)
    best, best_state = float("inf"), None
    for ep in range(a.epochs):
        head.train(); perm = np.random.default_rng(seed + ep).permutation(len(train)); tot = n = 0
        for s in range(0, len(train) - a.bs + 1, a.bs):
            sel = perm[s:s + a.bs].tolist()
            batch, ch = batch_of(train, sel, dev)
            idx, mask, tgt = pack(ch, dev)
            lg, enc = base_pass(base_m, base_box, batch, True)
            loss = F.cross_entropy(over_legal(lg + head(enc), idx, mask), tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(ch); n += len(ch)
        head.eval()
        if val:
            v = evaluate({"b": base_m}, {"b": base_box}, head, val, dev, [1.0])["b"][1.0]
            print(f"  epoch {ep + 1}: train NLL {tot / max(1, n):.4f} | val NLL {v[0]:.4f} top1 {100 * v[1]:.2f}%",
                  flush=True)
            if v[0] < best:
                best, best_state = v[0], {k: t.detach().clone() for k, t in head.state_dict().items()}
        else:
            print(f"  epoch {ep + 1}: train NLL {tot / max(1, n):.4f}", flush=True)
    if best_state:
        head.load_state_dict(best_state)
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", required=True); ap.add_argument("--clone", required=True)
    ap.add_argument("--data-dir", default=os.path.join("data", "lichess_5k"))
    ap.add_argument("--base", default="stock", choices=["stock", "clone"], help="which frozen model the head corrects")
    ap.add_argument("--games", type=int, default=1200, help="training games (before the held-out newest 80)")
    ap.add_argument("--hid", type=int, default=32); ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--lams", default="0,0.5,1,1.5,2,3")
    ap.add_argument("--cv", action="store_true",
                    help="5-fold CV inside the held-out window — the only positions the CLONE has never seen")
    ap.add_argument("--cv-games", type=int, default=80)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "clonefish", "identity_head.json"))
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time(); me = a.player.lower()
    lams = [float(x) for x in a.lams.split(",")]

    stock, mcfg = load_model(CKPT); stock.eval().to(dev)
    clone, _ = load_model(CKPT)
    clone.load_state_dict(torch.load(a.clone, map_location="cpu", weights_only=False)["model_state"])
    clone.eval().to(dev)
    for m in (stock, clone):
        for p in m.parameters():
            p.requires_grad_(False)
    models = {"stock": stock, "clone": clone}
    boxes = {k: hook_enc(m) for k, m in models.items()}

    gs = games_of(os.path.join(ROOT, a.data_dir, f"{a.player}.pgn.zst"), me); gs.sort(key=dkey)
    base_m, base_box = models[a.base], boxes[a.base]
    if a.cv:
        # The clone was fine-tuned on everything EXCEPT the newest games, so those are the only positions it has
        # never seen. Training a clone-based head anywhere else scores the clone on its own training data and
        # makes the remaining identity look like zero. Cross-validate inside the clean window instead.
        per = [rows_for([g], me) for g in gs[-a.cv_games:]]
        fold = np.random.default_rng(0).integers(0, 5, len(per))
        acc = {k: {l: [0.0, 0.0, 0.0, 0] for l in lams} for k in models}
        head = None
        for kf in range(5):
            tr = [r for i, p in enumerate(per) if fold[i] != kf for r in p]
            te = [r for i, p in enumerate(per) if fold[i] == kf for r in p]
            if not tr or not te:
                continue
            print(f" fold {kf + 1}: train {len(tr)} / test {len(te)} own moves", flush=True)
            head = fit_head(tr, None, base_m, base_box, mcfg.dim_vit, a, dev, seed=100 * kf)
            rk = evaluate(models, boxes, head, te, dev, lams)
            for k in models:
                for l in lams:
                    n_, t1, t3 = rk[k][l]; A = acc[k][l]
                    A[0] += n_ * len(te); A[1] += t1 * len(te); A[2] += t3 * len(te); A[3] += len(te)
        r = {k: {l: (A[0] / A[3], A[1] / A[3], A[2] / A[3]) for l, A in v.items()} for k, v in acc.items()}
        print(f"\n{a.player}: 5-fold CV inside the {a.cv_games} never-trained games, head on the {a.base} model",
              flush=True)
    else:
        test = rows_for(gs[-80:], me)
        val = rows_for(gs[-160:-80], me)
        train = rows_for(gs[-(160 + a.games):-160], me)
        print(f"{a.player}: train {len(train)} / val {len(val)} / test {len(test)} own moves "
              f"({a.games} training games, newest 80 never trained on)", flush=True)
        head = fit_head(train, val, base_m, base_box, mcfg.dim_vit, a, dev)
        print(f"\n{a.player}: held-out 80 games ({len(test)} own moves), head trained on the {a.base} model", flush=True)
        r = evaluate(models, boxes, head, test, dev, lams)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    res.setdefault(a.player, {})[a.base] = {k: {str(l): v for l, v in d.items()} for k, d in r.items()}
    for k in ("stock", "clone"):
        print(f"  on the {k} model:", flush=True)
        for l in lams:
            n_, t1, t3 = r[k][l]
            tag = "  <- plain model" if l == 0 else ""
            print(f"    lambda {l:<4} NLL {n_:.4f}  top1 {100 * t1:5.2f}%  top3 {100 * t3:5.2f}%{tag}", flush=True)
    if not a.cv:            # in CV mode `head` is just the last fold's — a measurement, not something to ship
        torch.save({"head": head.state_dict(), "hid": a.hid, "base": a.base, "dim": mcfg.dim_vit},
                   os.path.join(ROOT, "clones", f"{a.player}_identity_{a.base}.pt"))
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"saved clones/{a.player}_identity_{a.base}.pt  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
