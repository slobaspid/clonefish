"""Learn WHICH KINDS OF MOVES the clone prefers over the stock model in real play, then amplify that.

The idea being tested (the user's): let the clone and the stock model play each other across many openings. Wherever
they would pick different moves, the clone's choice reveals a systematic preference — a FAMILY of moves it favours,
the kind that recur in the player's own games. Capture that preference, amplify it, and see whether top-1 / top-3
against the player's real moves improve.

Why this is not the scalar push that already failed. The push amplified the raw per-position log-ratio
`clone - stock`, which mixes a systematic component with per-position noise, and sweeping its strength gained
nothing (best w was NEGATIVE, i.e. toward the norm). Here a deliberately LOW-CAPACITY head is trained to tell the
clone's pick from the stock's pick; it cannot memorise per-position quirks, so it can only represent the systematic
part — the families. Amplifying that may behave differently from amplifying the raw difference. Both are swept side
by side below, which is the actual experiment.

Two guards against fooling ourselves:
  * both candidate moves come from the SAME position (clone's top vs stock's top), so the head cannot win by
    learning whose turn it is, which colour is moving, or what kind of position each engine tends to reach;
  * strength is chosen by TOP-3 on the player's held-out real games, 5-fold CV by game — never by the likelihood
    the clone was trained with, and never on the fold it is scored on.

    PYTHONPATH=. python scripts/clonefish_family_amplify.py --player VEGETAL --clone clones/ftval_VEGETAL_bucket.pt
"""
import argparse, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np, torch, torch.nn.functional as F, chess
from finetune_clone import games_of, rows_of, batch_of
from clonefish_uci import IdentityHead, hook_enc, CloneEngine, player_data
from clonefish_eval import simulate, CKPT
from clonefish_push_fit_moves import cache
from clonefish_amplify_topk import OPENINGS, topk, cv_select
from sahformer.training.loop import load_model

dkey = lambda g: ((g.headers.get("UTCDate") or g.headers.get("Date") or ""),
                  (g.headers.get("UTCTime") or g.headers.get("StartTime") or ""))


def selfplay(clone_path, data, dev, n_games, seed=2):
    """clone vs stock across the opening list; returns the generated games"""
    pl = CloneEngine(CKPT, clone_path, data, device=dev, seed=seed)
    op = CloneEngine(CKPT, None, None, device=dev, seed=seed + 1)
    op.opt.update({"Elo": data["opp_elo"], "OppElo": data["elo"], "UseBook": False})
    out, per = [], max(1, n_games // len(OPENINGS))
    for line in OPENINGS:
        for j in range(per):
            out.append(simulate(pl, op, player_white=(j % 2 == 0), start=line))
    del pl, op
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def disagreements(clone, stock, boxC, games, dev, bs=256):
    """At every position the CLONE faced in self-play, compare its top move with the stock model's top move.
    Where they differ, that pair IS the signal: same position, two candidate moves, one 'clone-ish', one 'norm-ish'."""
    rows = []
    for g in games:
        if g.board().fen() != chess.STARTING_FEN:
            continue
        rows += [r for r in rows_of(g, "player") if len(r[6]) > 1]
    enc, pos, neg = [], [], []
    for s in range(0, len(rows), bs):
        idx = range(s, min(s + bs, len(rows)))
        b, ch = batch_of(rows, idx, dev)
        lc = clone(b)["move_logits"].float()
        e = boxC["enc"].float()
        lb = stock(b)["move_logits"].float()
        for j, r in enumerate(ch):
            legal = r[6]
            ci = legal[int(lc[j][legal].argmax())]
            bi = legal[int(lb[j][legal].argmax())]
            if ci == bi:
                continue                      # no disagreement, no signal
            enc.append(e[j].cpu()); pos.append(ci); neg.append(bi)
    return enc, np.array(pos), np.array(neg)


def train_family(enc, pos, neg, dim, hid, dev, epochs=6, bs=128, lr=1e-3, wd=1e-4, seed=0):
    """Low capacity on purpose: it can only learn WHAT KIND of move the clone prefers, not which position it was."""
    D = IdentityHead(dim, hid).to(dev)
    opt = torch.optim.AdamW(D.parameters(), lr=lr, weight_decay=wd)
    n = len(enc)
    split = int(0.8 * n)
    rng = np.random.default_rng(seed); order = rng.permutation(n)
    tr, te = order[:split], order[split:]
    E = torch.stack(enc)
    P = torch.from_numpy(pos).long(); N = torch.from_numpy(neg).long()
    best, state = -1.0, None
    for ep in range(epochs):
        D.train(); perm = rng.permutation(len(tr))
        for i in range(0, len(tr) - bs + 1, bs):
            sel = tr[perm[i:i + bs]]
            e = E[sel].to(dev)
            s = D(e)
            loss = F.binary_cross_entropy_with_logits(
                torch.cat([s.gather(1, P[sel, None].to(dev)), s.gather(1, N[sel, None].to(dev))]).squeeze(1),
                torch.cat([torch.ones(len(sel)), torch.zeros(len(sel))]).to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        D.eval()
        with torch.no_grad():
            acc = []
            for i in range(0, len(te), 512):
                sel = te[i:i + 512]; e = E[sel].to(dev); s = D(e)
                acc.append((s.gather(1, P[sel, None].to(dev)) > s.gather(1, N[sel, None].to(dev))).float().cpu())
            a = float(torch.cat(acc).mean())
        print(f"  epoch {ep + 1}: held-out pair accuracy {100 * a:.1f}% "
              f"(50 % = the preference is not learnable)", flush=True)
        if a > best:
            best, state = a, {k: v.detach().clone() for k, v in D.state_dict().items()}
    if state:
        D.load_state_dict(state)
    return D, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", required=True); ap.add_argument("--clone", required=True)
    ap.add_argument("--data-dir", default=os.path.join("data", "lichess_5k"))
    ap.add_argument("--selfplay", type=int, default=60)
    ap.add_argument("--hid", type=int, default=8, help="deliberately small: families, not positions")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--grid", default="0,0.25,0.5,1,1.5,2,3")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "clonefish", "family_amplify.json"))
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    grid = [float(x) for x in a.grid.split(",")]
    if 0.0 not in grid:
        grid.append(0.0)
    t0 = time.time(); me = a.player.lower()

    stock, mcfg = load_model(CKPT); stock.eval().to(dev)
    clone, _ = load_model(CKPT)
    clone.load_state_dict(torch.load(a.clone, map_location="cpu", weights_only=False)["model_state"])
    clone.eval().to(dev)
    for m in (stock, clone):
        for p in m.parameters():
            p.requires_grad_(False)
    boxC = hook_enc(clone)

    pgn = os.path.join(ROOT, a.data_dir, f"{a.player}.pgn.zst")
    data = player_data(pgn, a.player, exclude_recent=80)
    games = selfplay(a.clone, data, dev, a.selfplay)
    enc, pos, neg = disagreements(clone, stock, boxC, games, dev)
    print(f"{a.player}: {len(games)} clone-vs-stock games -> {len(enc)} positions where they disagree "
          f"({time.time() - t0:.0f}s)", flush=True)
    if len(enc) < 500:
        print("  too few disagreements to learn from"); return
    D, pair_acc = train_family(enc, pos, neg, mcfg.dim_vit, a.hid, dev, epochs=a.epochs)

    gs = games_of(pgn, me); gs.sort(key=dkey)
    held = gs[-80:]
    C, B, played, mv, gid = cache(clone, stock, held, me, dev)
    mask = C > -1e8
    Cn = np.where(mask, C, -np.inf); Bn = np.where(mask, B, -np.inf)

    rows = []
    for g in held:
        if g.board().fen() != chess.STARTING_FEN:
            continue
        rows += [r for r in rows_of(g, me) if r[5] in r[6] and len(r[6]) > 1]
    Dl = np.full_like(C, 0.0)
    with torch.no_grad():
        for s in range(0, len(rows), 256):
            b, ch = batch_of(rows, range(s, min(s + 256, len(rows))), dev)
            clone(b)
            d = D(boxC["enc"].float()).cpu().numpy()
            for j, r in enumerate(ch):
                legal = r[6]
                Dl[s + j, :len(legal)] = d[j][legal]

    print(f"\n{a.player}: amplification selected BY TOP-3, 5-fold CV by game ({len(played)} held-out moves)",
          flush=True)
    res = {"pair_acc": pair_acc, "n_disagree": len(enc)}
    for tag, extra in (("learned family signal", Dl), ("raw clone-stock push", Cn - Bn)):
        hits = {w: topk(Cn + w * extra, played, mask) for w in grid}
        r = cv_select(hits, gid, grid)
        print(f"  {tag:<22} top1 {r['base1']:5.2f} -> {r['top1']:5.2f} ({r['top1'] - r['base1']:+.2f})  "
              f"top3 {r['base3']:5.2f} -> {r['top3']:5.2f} ({r['top3'] - r['base3']:+.2f})  w={r['picks']}",
              flush=True)
        res[tag] = r
    allr = json.load(open(a.out)) if os.path.exists(a.out) else {}
    allr[a.player] = res
    json.dump(allr, open(a.out, "w"), indent=1, default=float)
    print(f"wrote {os.path.relpath(a.out, ROOT)} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
