"""Phase breakdown: where does the clone's move-match gain actually live? Bucket held-out moveΔ by
game phase (ply). Hypothesis: openings (consistent repertoire) carry most of it, sharp midgames little.
Needs an ENRICHED cache (with meta_test = (ply, elo)).

    PYTHONPATH=. python scripts/phase_breakdown.py --cache sweep_cache_lichess_1k_100_full.pt
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
from clone_sweep import MAXLEG, DEV
from build_pooled_clone import train_shared

BUCKETS = [(0, 12, "opening"), (12, 24, "early-mid"), (24, 40, "middlegame"), (40, 9999, "late/endgame")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="sweep_cache_lichess_1k_100_full.pt")
    args = ap.parse_args()
    cache = torch.load(args.cache, weights_only=False)
    players = list(cache.keys())
    if "meta_test" not in cache[players[0]]:
        print("ERROR: cache has no meta_test (ply/elo) — rebuild with the enriched clone_sweep."); return
    # train the shared head (memory-safe; needs grad)
    m, players = train_shared(cache, 512, 512, 3e-2, 20, 0.6, 0.10, 1e-3)
    m.eval()
    _eval_phases(m, cache, players)


@torch.no_grad()
def _eval_phases(m, cache, players):
    # per phase: [base_correct, clone_correct, n, base_loglik_sum, clone_loglik_sum]
    agg = {lab: [0, 0, 0, 0.0, 0.0] for _, _, lab in BUCKETS}
    aleg = torch.arange(MAXLEG, device=DEV)
    for i, n in enumerate(players):
        ep, ebl, elg, eln, eac = cache[n]["test"]
        ply = cache[n]["meta_test"][:, 0]
        epf=ep.float().to(DEV); ebf=ebl.float().to(DEV); elf=elg.long().to(DEV); eac_d=eac.to(DEV)
        pad = aleg[None,:] >= eln.to(DEV)[:,None]
        res = m(epf, torch.full((epf.shape[0],), i, device=DEV)); rl = torch.gather(res, 1, elf)
        base_lg = ebf.masked_fill(pad,-1e4); clone_lg = (ebf+rl).masked_fill(pad,-1e4)
        base_ok = (base_lg.argmax(1) == eac_d).cpu(); clone_ok = (clone_lg.argmax(1) == eac_d).cpu()
        base_ll = torch.log_softmax(base_lg,1).gather(1,eac_d[:,None]).squeeze(1).cpu()
        clone_ll = torch.log_softmax(clone_lg,1).gather(1,eac_d[:,None]).squeeze(1).cpu()
        for lo, hi, lab in BUCKETS:
            msk = (ply >= lo) & (ply < hi)
            agg[lab][0]+=int(base_ok[msk].sum()); agg[lab][1]+=int(clone_ok[msk].sum()); agg[lab][2]+=int(msk.sum())
            agg[lab][3]+=float(base_ll[msk].sum()); agg[lab][4]+=float(clone_ll[msk].sum())
    print(f"\n{'phase':>14}{'n':>10}{'base top1':>11}{'clone top1':>12}{'moveΔ':>9}{'base LL':>10}{'clone LL':>10}{'llΔ':>9}")
    for _, _, lab in BUCKETS:
        bc, cc, nn, bll, cll = agg[lab]
        if nn == 0: continue
        print(f"{lab:>14}{nn:>10}{bc/nn*100:>10.1f}%{cc/nn*100:>11.1f}%{(cc-bc)/nn*100:>+8.1f}"
              f"{bll/nn:>10.3f}{cll/nn:>10.3f}{(cll-bll)/nn:>+9.3f}")
    t = [sum(agg[l][k] for _,_,l in BUCKETS) for k in range(5)]
    print(f"{'ALL':>14}{t[2]:>10}{t[0]/t[2]*100:>10.1f}%{t[1]/t[2]*100:>11.1f}%{(t[1]-t[0])/t[2]*100:>+8.1f}"
          f"{t[3]/t[2]:>10.3f}{t[4]/t[2]:>10.3f}{(t[4]-t[3])/t[2]:>+9.3f}")


if __name__ == "__main__":
    main()
