#!/bin/bash
# Stage 1 of results/night_0913/PLAN.md: honest val split fine-tunes, then dial + timing eval.
cd "/c/Users/sloba/Downloads/sah transformer"
export PYTHONPATH=.
for n in VEGETAL kpowe52 OKENITE; do
  echo "=== $n FT start $(date)"
  python scripts/finetune_clone.py --pgn data/lichess_5k/$n.pgn.zst --name $n \
     --max-train-games 5000 --train-skip-recent 40 --save-ft clones/ftval_$n.pt \
     > results/night_0913/ft_$n.log 2>&1 || echo "FT FAILED $n"
  echo "=== $n EVAL start $(date)"
  python scripts/night_dial_eval.py --pgn data/lichess_5k/$n.pgn.zst --name $n \
     --ft clones/ftval_$n.pt --device cuda > results/night_0913/eval_$n.log 2>&1 || echo "EVAL FAILED $n"
  echo "=== $n done $(date)"
done
echo ALL_DONE
