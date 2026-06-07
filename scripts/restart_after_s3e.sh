#!/bin/bash
# Wait for Bob/Carol S3E reprocessing to finish, then restart training with all maps.
# Usage: nohup bash scripts/restart_after_s3e.sh >> output/restart_s3e.log 2>&1 &

BOBCAROL_LOG=output/s3e_processing_bobcarol.log
PYTHON=/home/rueyday/miniconda3/envs/torch5090/bin/python3

echo "[$(date '+%H:%M:%S')] Waiting for Bob/Carol S3E processing to finish (need 2nd Done: line)..."

# The log already has one "Done: 0 succeeded..." from an aborted earlier run.
# Wait until the second "Done:" appears (real completion).
until [ "$(grep -c "^Done:" "$BOBCAROL_LOG" 2>/dev/null)" -ge 2 ]; do sleep 30; done
echo "[$(date '+%H:%M:%S')] Bob/Carol processing done."

# Count usable new maps
N=$($PYTHON -c "
import sys; sys.path.insert(0,'.')
from sim3.pair_generator import load_pool_from_db
import glob
maps = glob.glob('../Structure-PLP-SLAM/s3e_S3E_*_bob_map.db') + glob.glob('../Structure-PLP-SLAM/s3e_S3E_*_carol_map.db')
usable = sum(1 for m in maps if len(load_pool_from_db(m)) >= 30)
print(usable)
" 2>/dev/null)
echo "[$(date '+%H:%M:%S')] Usable Bob/Carol pools: $N"

# Find current best checkpoint
BEST=$(ls -t output/joint/2026-06-07-submap-knn-s3e-v2/best_val_checkpoint.pth 2>/dev/null | head -1)
if [ -z "$BEST" ]; then
    BEST=output/joint/2026-06-04-submap-knn-s3e/best_val_checkpoint.pth
fi
echo "[$(date '+%H:%M:%S')] Using pretrain: $BEST"

# Kill current training
TRAIN_PID=$(ps aux | grep "python.*train.py" | grep -v grep | awk 'NR==1{print $2}')
if [ -n "$TRAIN_PID" ]; then
    echo "[$(date '+%H:%M:%S')] Killing training PID $TRAIN_PID (waiting for epoch end)..."
    # Wait for current epoch to finish before killing
    CURRENT_EPOCH=$(grep "Train Epoch:" output/submap_run6.log 2>/dev/null | tail -1 | grep -o 'Epoch: [0-9]*' | awk '{print $2}')
    until grep -q "Train Epoch: $((CURRENT_EPOCH+1)) \[0/" output/submap_run6.log 2>/dev/null; do sleep 30; done
    kill "$TRAIN_PID" 2>/dev/null
    sleep 5
fi

echo "[$(date '+%H:%M:%S')] Starting final training run with all S3E maps..."

nohup $PYTHON train.py \
    --mode live --submap --model knn \
    --db_train '../Structure-PLP-SLAM/*.db' \
    --db_val \
        ../Structure-PLP-SLAM/tum_rgbd_rgbd_dataset_freiburg1_desk_run1_map.db \
        ../Structure-PLP-SLAM/tum_rgbd_rgbd_dataset_freiburg2_xyz_run1_map.db \
        ../Structure-PLP-SLAM/tum_rgbd_rgbd_dataset_freiburg3_long_office_household_run1_map.db \
        ../Structure-PLP-SLAM/tum_rgbd_dataset_freiburg3_long_office_household_run5_map.db \
    --val_size 400 \
    --lr 5e-6 --gamma 0.995 \
    --epochs 1000 --batch 1 --iter_size 32 --epoch_size 32000 \
    --ransac sim3 \
    --name 2026-06-07-submap-knn-s3e-full \
    --pretrain "$BEST" \
    >> output/submap_run7.log 2>&1 &

echo "[$(date '+%H:%M:%S')] Final training PID: $!"
