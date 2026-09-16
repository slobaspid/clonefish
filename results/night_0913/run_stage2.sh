#!/bin/bash
# Stage 2: does the whole-second time loss stop fine-tuning from damaging the time head (and moves)?
# Same player/split as stage 1 VEGETAL; only --time-loss changes. Waits for stage 1 to free the GPU.
cd "/c/Users/sloba/Downloads/sah transformer"
export PYTHONPATH=.
until grep -q ALL_DONE results/night_0913/stage1_driver.log; do sleep 60; done
for n in VEGETAL kpowe52; do
  echo "=== $n bucket FT start $(date)"
  python scripts/finetune_clone.py --pgn data/lichess_5k/$n.pgn.zst --name $n --time-loss bucket \
     --max-train-games 5000 --train-skip-recent 40 --save-ft clones/ftval_${n}_bucket.pt \
     > results/night_0913/ft_${n}_bucket.log 2>&1 || echo "FT FAILED $n"
  mkdir -p results/night_0913/bucket
  python scripts/night_dial_eval.py --pgn data/lichess_5k/$n.pgn.zst --name $n \
     --ft clones/ftval_${n}_bucket.pt --device cuda --out results/night_0913/bucket \
     > results/night_0913/eval_${n}_bucket.log 2>&1 || echo "EVAL FAILED $n"
  echo "=== $n done $(date)"
done
echo ALL_DONE
