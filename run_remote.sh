#!/usr/bin/env bash
# Run on a rented GPU box. Encodes the corpus, then continues training base_v3.
#
# Ships the raw PGN (1.7GB) rather than the shards (22GB) and re-encodes on arrival - a rented
# box has many more cores than the 1060 machine, so the encode is quick and the upload is 13x
# smaller.
#
#   ./run_remote.sh            resume from checkpoints/base_v3/last.pt if present, else fresh
#
# NOTE on the container killing orphans (HANDOFF-08-25): nohup/setsid/tmux were all reported dead
# on Vast. This script therefore runs training in the FOREGROUND - keep the ssh session open. If
# it dies anyway, just re-run: checkpoints land every 1000 steps and the trainer resumes from the
# newest one, so a disconnect costs at most 1000 steps.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=.

SHARDS=${SHARDS:-data/shards_v3}
CKPT=${CKPT:-checkpoints/base_v3}
BUDGET=${BUDGET:-7300000}
WORKERS=${WORKERS:-$(nproc 2>/dev/null || echo 8)}
BS=${BS:-256}          # a 24GB card holds far more than the 1060's 64
ACCUM=${ACCUM:-2}      # BS * ACCUM must stay 512 - the agreed 08-26 recipe
STEPS=${STEPS:-400000}

echo "=== environment ==="
python -c "import torch;print('torch',torch.__version__,'| cuda',torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" 2>/dev/null
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "cores: $WORKERS"

python -c "import chess, zstandard" 2>/dev/null || pip install -q python-chess zstandard

echo
echo "=== STAGE 1  encode ($(date)) ==="
if [ "$(ls -1 "$SHARDS"/*.npz 2>/dev/null | wc -l)" -gt 10 ]; then
  echo "shards already present ($(ls -1 "$SHARDS"/*.npz | wc -l)) - skipping"
else
  python -u scripts/build_corpus_v2.py \
      data/chesscom_bands_v2 data/chesscom_corpus data/chesscom_lowmid \
      data/chesscom_lowmid_trial data/chesscom_p1 data/chesscom_p2 data/chesscom_p3 \
      --out "$SHARDS" --budget "$BUDGET" --workers "$WORKERS" --chunk 200000
fi

N=$(ls -1 "$SHARDS"/*.npz 2>/dev/null | wc -l)
echo "shards: $N  ($(du -sh "$SHARDS" 2>/dev/null | cut -f1))"
[ "$N" -lt 10 ] && { echo "FATAL: only $N shards, refusing to train"; exit 1; }

echo
echo "=== STAGE 2  train ($(date)) ==="
echo "effective batch $((BS*ACCUM))  (bs $BS x accum $ACCUM), lr 4e-5"
python -u scripts/kaggle_train.py \
    --shards "$SHARDS" --out "$CKPT" \
    --dim 512 --blocks 12 --heads 8 \
    --bs "$BS" --accum "$ACCUM" --lr 4e-5 --steps "$STEPS" \
    --save-every 1000 --log-every 200 --eval-every 2000 --keep-every 10000 --val-shards 8
echo "exited: $?  ($(date))"
