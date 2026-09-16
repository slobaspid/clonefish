#!/usr/bin/env bash
# 3090 full batch: 1k-games experiments on ONE enriched cache. Run from repo root on the GPU box.
# Assumes: data/lichess_1k/ (100 players) and checkpoints/base_300k_best.pt present, venv active.
set -e
export PYTHONPATH=.
CACHE=sweep_cache_lichess_1k_100_full.pt

echo "############ STEP 1: build enriched 1k cache + games x dim sweep (was 250 too little?) ############"
python -u scripts/clone_sweep.py --players 100 --full-train --mode pooled \
    --data-dir data/lichess_1k --max-games 1200 2>&1 | tee sweep_1k.log

echo "############ STEP 2: stranger generalization (head 75, clone 25 disjoint) ############"
python -u scripts/stranger_test.py --cache "$CACHE" --strangers 25 2>&1 | tee stranger_1k.log

echo "############ STEP 3: move-identifier (CosFace-margin vs plain-CE) ############"
python -u scripts/move_identifier.py --cache "$CACHE" 2>&1 | tee moveid_1k.log

echo "############ STEP 4: phase breakdown (opening vs midgame moveΔ) ############"
python -u scripts/phase_breakdown.py --cache "$CACHE" 2>&1 | tee phase_1k.log

echo "############ STEP 5: Elo-adaptability (static vs elo-conditioned, on movers) ############"
python -u scripts/elo_adapt.py --cache "$CACHE" 2>&1 | tee elo_1k.log

echo "############ ALL DONE ############"
