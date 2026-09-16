"""Load the trained model and watch it play itself — a live blitz play-by-play
with think-times and ticking clocks (the clock-aware part on display).

    PYTHONPATH=. python scripts/watch_selfplay.py checkpoints/base_300k_best.pt --elo 1500
"""
import argparse
from sahformer.training.loop import load_model
from sahformer.play import self_play

def mmss(t):
    t = max(0.0, t); return f"{int(t//60)}:{t%60:04.1f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--elo", type=int, default=1500)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--think-temp", type=float, default=1.0)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"loading {args.ckpt} ...")
    model, mcfg = load_model(args.ckpt)
    print(f"loaded. dim={mcfg.dim_vit} blocks={mcfg.num_blocks} | playing at Elo {args.elo}\n")
    print(f"{'move':>7}  {'san':<7} {'think':>6}   {'white':>7} {'black':>7}")
    print("-" * 44)

    last = None
    for rec in self_play(model, max_plies=args.max_plies, elo=args.elo,
                         temperature=args.temperature, think_temp=args.think_temp,
                         seed=args.seed):
        n = rec["ply"] // 2 + 1
        tag = f"{n}." if rec["mover"] == "white" else f"{n}..."
        print(f"{tag:>7}  {rec['san']:<7} {rec['think']:>5.1f}s   "
              f"{mmss(rec['white_clock']):>7} {mmss(rec['black_clock']):>7}"
              + ("  <-- FLAG!" if rec["flagged"] else ""))
        last = rec

    print("-" * 44)
    if last and last["flagged"]:
        print(f"Game over: {last['mover']} flagged (ran out of time).")
    else:
        # replay to get the board result
        import chess
        b = chess.Board()
        for rec in self_play(model, max_plies=args.max_plies, elo=args.elo,
                             temperature=args.temperature, think_temp=args.think_temp,
                             seed=args.seed):
            b.push(rec["move"])
        print(f"Game over: {b.result()} ({'checkmate' if b.is_checkmate() else 'stalemate/draw/cap'}).")

if __name__ == "__main__":
    main()
