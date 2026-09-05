#!/usr/bin/env bash
# Overnight: encode the ENTIRE corpus, then train the scaled base on it.
#
# Stage 1 encodes every player-file directory into Elo-even shards (6 workers).
# Stage 2 only starts if stage 1 actually produced shards - training on an empty or
# half-written directory would burn the night silently.
#
# Both stages are resumable: re-running skips nothing in stage 1 (it rebuilds), but
# stage 2 picks up from its newest checkpoint, so killing and re-running this script
# after the encode is finished just continues training.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=.

SHARDS=data/shards_v3
CKPT=checkpoints/base_v3
BUDGET=${BUDGET:-7300000}      # positions per Elo band (~100k games/band)
WORKERS=${WORKERS:-6}
BS=${BS:-128}

echo "=============================================================="
echo "STAGE 1  encode entire corpus -> $SHARDS   ($(date))"
echo "=============================================================="

if [ -d "$SHARDS" ] && [ "$(ls -1 "$SHARDS"/*.npz 2>/dev/null | wc -l)" -gt 0 ]; then
  echo "shards already present ($(ls -1 "$SHARDS"/*.npz | wc -l)) - skipping encode"
else
  python -u scripts/build_corpus_v2.py \
      data/chesscom_bands_v2 data/chesscom_corpus data/chesscom_lowmid \
      data/chesscom_lowmid_trial data/chesscom_p1 data/chesscom_p2 data/chesscom_p3 \
      --out "$SHARDS" --budget "$BUDGET" --workers "$WORKERS" --chunk 200000
  echo "encode exit: $?"
fi

N=$(ls -1 "$SHARDS"/*.npz 2>/dev/null | wc -l)
echo "shards on disk: $N   ($(du -sh "$SHARDS" 2>/dev/null | cut -f1))"
if [ "$N" -lt 10 ]; then
  echo "FATAL: only $N shards - refusing to start training on this. Stopping."
  exit 1
fi

echo
echo "=============================================================="
echo "STAGE 2  train scaled base (28.7M) -> $CKPT   ($(date))"
echo "=============================================================="
python -u scripts/kaggle_train.py \
    --shards "$SHARDS" --out "$CKPT" \
    --dim 512 --blocks 12 --heads 8 \
    --bs "$BS" --lr 4e-5 --steps 400000 \
    --save-every 1000 --log-every 200

echo "training exited: $?   ($(date))"
