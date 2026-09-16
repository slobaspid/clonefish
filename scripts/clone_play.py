"""Watch your clone play a game (moves + its clock rhythm), in the terminal.

    PYTHONPATH=. python scripts/clone_play.py --clone clones/your_username.pt

Add --base to compare: play the same seed with and without the clone to see the deviation.
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess
from sahformer.training.loop import load_model
from sahformer.clone import load_clone
from sahformer.play import self_play


def play(model, adapter, elo, seed, temperature, top_p, max_plies):
    board = chess.Board()
    tag = "BASE" if adapter is None else "CLONE"
    print(f"\n=== {tag} · elo {elo} · seed {seed} ===")
    for rec in self_play(model, max_plies=max_plies, elo=elo, temperature=temperature,
                         top_p=top_p, seed=seed, adapter=adapter):
        n = rec["ply"] // 2 + 1
        dots = "." if rec["mover"] == "white" else "..."
        print(f"{n:>3}{dots} {rec['san']:<7}  {rec['think']:>5.1f}s   "
              f"(W {rec['white_clock']:>5.1f} | B {rec['black_clock']:>5.1f})")
        board.push(rec["move"])
        if rec["flagged"]:
            print(f"  {rec['mover']} flagged on time.")
            return
    print("  " + (board.result() if board.is_game_over() else "* (move cap)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone", required=True, help="path to a clone .pt from clone_fit.py")
    ap.add_argument("--ckpt", default="checkpoints/base_300k_best.pt")
    ap.add_argument("--elo", type=int, default=None, help="default: the clone's average Elo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-plies", type=int, default=140)
    ap.add_argument("--base", action="store_true", help="also play the base (same seed) to compare")
    args = ap.parse_args()

    model, mcfg = load_model(args.ckpt)
    model.eval()
    adapter = load_clone(args.clone)
    elo = args.elo or int(adapter.meta.get("avg_elo", 1500))
    print(f"loaded clone: {args.clone}  (name={adapter.meta.get('name','?')}, elo={elo})")
    if args.base:
        play(model, None, elo, args.seed, args.temperature, args.top_p, args.max_plies)
    play(model, adapter, elo, args.seed, args.temperature, args.top_p, args.max_plies)


if __name__ == "__main__":
    main()
