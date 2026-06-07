#!/bin/bash
# Wait for current epoch to finish, then restart training with new S3E maps.
# Usage: nohup bash scripts/restart_training.sh >> output/restart.log 2>&1 &

TRAIN_PID=3621344
LOG=output/submap_run4.log
NEW_LOG=output/submap_run5.log
PYTHON=/home/rueyday/miniconda3/envs/torch5090/bin/python3
PRETRAIN=output/joint/2026-06-04-submap-knn/best_val_checkpoint.pth

echo "[$(date '+%H:%M:%S')] Watching for epoch 32 to finish..."

# Wait until validation runs (signalled by "Val " line after step 500, or "Epoch: 33" train start)
while true; do
    if grep -q "Train Epoch: 33 \[0/" "$LOG" 2>/dev/null; then
        echo "[$(date '+%H:%M:%S')] Epoch 32 done. Killing PID $TRAIN_PID..."
        break
    fi
    sleep 10
done

kill "$TRAIN_PID" 2>/dev/null
sleep 5

echo "[$(date '+%H:%M:%S')] Starting new training run..."

nohup $PYTHON train.py \
    --mode live --submap --model knn \
    --db_train '../Structure-PLP-SLAM/*.db' \
    --db_val \
        ../Structure-PLP-SLAM/tum_rgbd_rgbd_dataset_freiburg1_desk_run1_map.db \
        ../Structure-PLP-SLAM/tum_rgbd_rgbd_dataset_freiburg2_xyz_run1_map.db \
        ../Structure-PLP-SLAM/tum_rgbd_rgbd_dataset_freiburg3_long_office_household_run1_map.db \
        ../Structure-PLP-SLAM/tum_rgbd_dataset_freiburg3_long_office_household_run5_map.db \
    --val_size 400 \
    --lr 1e-5 --gamma 0.995 \
    --epochs 1000 --batch 1 --iter_size 32 --epoch_size 32000 \
    --ransac sim3 \
    --name 2026-06-04-submap-knn-s3e \
    --pretrain "$PRETRAIN" \
    >> "$NEW_LOG" 2>&1 &

echo "[$(date '+%H:%M:%S')] New training PID: $!"
