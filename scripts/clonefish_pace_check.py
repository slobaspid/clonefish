"""Teacher-forced pace check: on a player's held-out real games (newest 80, never trained on), does the clone's time
head spend time like the player, move band by move band? Compares real / clone(s) / base.

    PYTHONPATH=. python scripts/clonefish_pace_check.py --data-dir data/lichess_1k \
        --players ARQZITRO:clones/ftval_ARQZITRO_bucket.pt+clones/ftval_ARQZITRO_bucket_th.pt,...
"""
import argparse, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np, torch
from finetune_clone import games_of, rows_of, batch_of
from night_dial_eval import mix, sample_readings
from sahformer.training.loop import load_model

CKPT = os.path.join(ROOT, "checkpoints", "base_300k_best.pt")
BANDS = [(1, 10), (10, 20), (20, 30), (30, 40), (40, 999)]
dkey = lambda g: ((g.headers.get("UTCDate") or g.headers.get("Date") or ""),
                  (g.headers.get("UTCTime") or g.headers.get("StartTime") or ""))


@torch.no_grad()
def sampled(model, rows, dev):
    P, M, S = [], [], []
    for s in range(0, len(rows), 512):
        b, _ = batch_of(rows, range(s, min(s + 512, len(rows))), dev)
        pi, mu, sg = [t.float().cpu().double() for t in model(b)["mdn"]]
        P.append(pi); M.append(mu); S.append(sg)
    lp, m, sg = mix(torch.cat(P), torch.cat(M), torch.cat(S)); gen = torch.Generator().manual_seed(0)
    return np.stack([sample_readings(lp, m, sg, gen).numpy() for _ in range(5)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", required=True, help="NAME:clone1+clone2,...")
    ap.add_argument("--data-dir", default=os.path.join("data", "lichess_5k"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "clonefish", "pace_check.json"))
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {}
    for spec in a.players.split(","):
        nm, paths = spec.split(":", 1); paths = paths.split("+")
        gs = games_of(os.path.join(ROOT, a.data_dir, f"{nm}.pgn.zst"), nm.lower()); gs.sort(key=dkey)
        rows = [r for g in gs[-80:] for r in rows_of(g, nm.lower()) if r[8] > 0]
        real = np.array([round(r[7]) for r in rows]); own = np.array([r[8] for r in rows])
        arms = {"real": real[None, :]}
        for tag, ft in [("base", None)] + [(os.path.basename(p).replace(".pt", ""), p) for p in paths]:
            m, _ = load_model(CKPT)
            if ft:
                m.load_state_dict(torch.load(os.path.join(ROOT, ft), map_location="cpu", weights_only=False)["model_state"])
            m.eval().to(dev); arms[tag] = sampled(m, rows, dev); del m; torch.cuda.empty_cache()
        res[nm] = {}
        print(f"\n{nm}: {len(rows)} held-out own moves — mean think (s) per arm, and sum over own moves 1-29")
        for tag, A in arms.items():
            bands = [round(float(A[:, (own >= lo) & (own < hi)].mean()), 2) for lo, hi in BANDS]
            first30 = round(float(A[:, own < 30].mean() * 29), 1)
            res[nm][tag] = {"bands": bands, "first30_sum": first30, "instant%": round(100 * float((A == 0).mean()), 1),
                            "10s+%": round(100 * float((A >= 10).mean()), 1)}
            print(f"  {tag:<28} bands {bands}  first-30 sum {first30:6.1f}  instant {res[nm][tag]['instant%']:4.1f}%  "
                  f"10s+ {res[nm][tag]['10s+%']:4.1f}%", flush=True)
    json.dump(res, open(a.out, "w"), indent=1); print("wrote", a.out)


if __name__ == "__main__":
    main()
