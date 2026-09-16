"""Does the clone's time model predict how the player plays when low on time?

Teacher-forced on the player's REAL positions from training games (never the newest 80): for each clock
bucket, compare the real instant-move rate and mean think with the clone model's sampled readings (lognormal
mixture sample, rounded through a whole-second clock exactly like the evaluation games).
  - model matches real here but free play doesn't  -> the gap comes from free-play dynamics, not the head
  - model misses real here too                     -> the head is miscalibrated in scrambles

    PYTHONPATH=. python scripts/clonefish_scramble_check.py --players VEGETAL:clones/ftval_VEGETAL_bucket.pt,...
"""
import argparse, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np, torch
from finetune_clone import games_of, rows_of, batch_of
from night_dial_eval import mix, sample_readings
from sahformer.training.loop import load_model

CKPT = os.path.join(ROOT, "checkpoints", "base_300k_best.pt")
BUCKETS = [(0, 5, "<5s"), (5, 10, "5-10s"), (10, 30, "10-30s"), (30, 60, "30-60s"), (60, 999, ">60s")]
dkey = lambda g: ((g.headers.get("UTCDate") or g.headers.get("Date") or ""),
                  (g.headers.get("UTCTime") or g.headers.get("StartTime") or ""))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", required=True); ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "clonefish", "scramble_check.json"))
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {}
    for spec in a.players.split(","):
        nm, ftp = spec.split(":")
        gs = games_of(os.path.join(ROOT, "data", "lichess_5k", f"{nm}.pgn.zst"), nm.lower()); gs.sort(key=dkey)
        rows = [r for g in gs[:-80][-a.games:] for r in rows_of(g, nm.lower()) if r[8] > 0]
        clock = np.array([r[2][0] * 180 for r in rows]); think = np.array([round(r[7]) for r in rows])
        keep = np.where(clock < 60)[0]                                   # all scramble-ish rows
        far = np.where(clock >= 60)[0]; keep = np.concatenate([keep, np.random.default_rng(0).choice(far, min(len(far), 6000), replace=False)])
        rows = [rows[i] for i in keep]; clock, think = clock[keep], think[keep]
        model, _ = load_model(CKPT)
        model.load_state_dict(torch.load(ftp, map_location="cpu", weights_only=False)["model_state"]); model.eval().to(dev)
        pis, mus, sgs = [], [], []
        for s in range(0, len(rows), 512):
            b, _ = batch_of(rows, range(s, min(s + 512, len(rows))), dev)
            pi, mu, sg = [t.float().cpu().double() for t in model(b)["mdn"]]
            pis.append(pi); mus.append(mu); sgs.append(sg)
        del model; torch.cuda.empty_cache()
        lp, m, sg = mix(torch.cat(pis), torch.cat(mus), torch.cat(sgs))
        gen = torch.Generator().manual_seed(0)
        samp = np.stack([sample_readings(lp, m, sg, gen).numpy() for _ in range(5)])     # 5 draws per position
        res[nm] = {}
        print(f"\n{nm}: {len(rows)} positions (training games only)")
        print(f"  {'clock':<7}{'n':>6}   {'instant% real / model':>22}   {'mean think real / model':>24}   {'10s+% real / model':>19}")
        for lo, hi, lab in BUCKETS:
            sel = (clock >= lo) & (clock < hi)
            if not sel.any(): continue
            rs, ms = think[sel], samp[:, sel].ravel()
            ms = np.minimum(ms, np.repeat(clock[sel][None, :], 5, 0).ravel())         # can't spend more than the clock
            row = {"n": int(sel.sum()), "instant_real": round(100 * float((rs == 0).mean()), 1),
                   "instant_model": round(100 * float((ms == 0).mean()), 1),
                   "mean_real": round(float(rs.mean()), 2), "mean_model": round(float(ms.mean()), 2),
                   "tank_real": round(100 * float((rs >= 10).mean()), 1), "tank_model": round(100 * float((ms >= 10).mean()), 1)}
            res[nm][lab] = row
            print(f"  {lab:<7}{row['n']:>6}   {row['instant_real']:>9} / {row['instant_model']:<9}   "
                  f"{row['mean_real']:>11} / {row['mean_model']:<10}   {row['tank_real']:>8} / {row['tank_model']:<8}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1); print("\nwrote", a.out)


if __name__ == "__main__":
    main()
