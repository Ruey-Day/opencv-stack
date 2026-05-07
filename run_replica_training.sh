#!/usr/bin/env bash
# run_replica_training.sh — generate Replica dataset then train Sim3-PlueckerNet
# Safe to run in tmux or with nohup; logs to output/generate_replica.log and output/train_replica.log

set -e
cd /home/rueyday/scale-aware-PlueckerNet

# Activate conda env
source /home/rueyday/miniconda3/etc/profile.d/conda.sh
conda activate torch5090

echo "=== GENERATION START $(date) ==="
python scripts/generate_replica_dataset.py \
    --n_train_per_scene 600 \
    --n_valid_per_scene 200 \
    --n_candidate_lines 400

echo "=== TRAINING START $(date) ==="
python train_sim3_replica.py

echo "=== DONE $(date) ==="
