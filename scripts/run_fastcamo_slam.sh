#!/bin/bash
# Run Structure-PLP-SLAM on all FastCaMo sequences sequentially.
# Waits for image conversion to finish before processing each sequence.

SLAM_DIR=/home/rueyday/scale-aware-cross-modal-registration/Structure-PLP-SLAM
FASTCAMO=/mnt/crucial/rueyday/data/fastcamo
VOCAB=$SLAM_DIR/orb_vocab/orb_vocab.dbow2
CFG=$SLAM_DIR/example/tum_rgbd/FastCaMo_rgbd.yaml
LOG_DIR=/home/rueyday/scale-aware-cross-modal-registration/ScalePluckerNet/output

SEQS=(apartment_1 apartment_2 gym lab lounge_1 lounge_2 meeting_room office stairwell studio workshop_1 workshop_2)

for name in "${SEQS[@]}"; do
    seq_dir="$FASTCAMO/$name"
    db_out="$SLAM_DIR/fastcamo_${name}_map.db"

    if [ -f "$db_out" ]; then
        echo "[SKIP] $name — db already exists"
        continue
    fi

    total=$(wc -l < "$seq_dir/associations.txt" 2>/dev/null)

    # Wait until image conversion is done (prep script writes rgb.txt + depth.txt when finished)
    echo "[WAIT] $name — waiting for conversion ..."
    until [ -f "$seq_dir/rgb.txt" ] && [ -f "$seq_dir/depth.txt" ]; do
        sleep 30
    done

    echo "[SLAM] $name — starting ($total frames) ..."
    $SLAM_DIR/build/run_tum_rgbd_slam_with_line \
        -v "$VOCAB" -c "$CFG" \
        -d "$seq_dir" \
        -p "$db_out" \
        --no-sleep --auto-term \
        > "$LOG_DIR/slam_fastcamo_${name}.log" 2>&1

    lines=$(grep "landmarks_line" "$LOG_DIR/slam_fastcamo_${name}.log" | tail -1 || echo "?")
    echo "[DONE] $name → $db_out  ($lines)"
done

echo "All FastCaMo SLAM runs complete."
