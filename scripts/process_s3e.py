#!/usr/bin/env python3
"""
Process S3E multi-robot dataset: extract stereo frames from ROS2 bags and
run Structure-PLP-SLAM to produce .db map files for ScalePluckerNet training.

For each S3E sequence (18 total across v1/v2) × 3 robots = up to 54 map files.

Pipeline per robot-sequence:
  1. Extract stereo frames from .db3 bag into a temp EuRoC-format directory
  2. Run run_euroc_slam_with_line → produces a .db map file
  3. Delete temp frames to reclaim disk space

Requires:
  - sqlite3 (stdlib)
  - cv2, numpy (torch5090 conda env)
  - Structure-PLP-SLAM built at ../Structure-PLP-SLAM/build/

Usage:
  python scripts/process_s3e.py [--dry-run] [--skip-existing] [--frame-skip N]
"""

import os
import sys
import sqlite3
import struct
import subprocess
import shutil
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S', level=logging.INFO)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).parent
REPO_ROOT    = SCRIPT_DIR.parent
SLAM_DIR     = REPO_ROOT.parent / 'Structure-PLP-SLAM'
SLAM_BIN     = SLAM_DIR / 'build' / 'run_euroc_slam_with_line'
VOCAB_FILE   = SLAM_DIR / 'orb_vocab' / 'orb_vocab.dbow2'
# Per-robot calibration YAMLs (Bob and Carol have different intrinsics)
CONFIG_YAML = {
    'Alpha': SCRIPT_DIR / 'S3E_stereo.yaml',
    'Bob':   SCRIPT_DIR / 'S3E_stereo_bob.yaml',
    'Carol': SCRIPT_DIR / 'S3E_stereo_carol.yaml',
}
S3E_ROOT     = Path('/mnt/crucial/slam_benchmark/datasets/S3E')
TEMP_ROOT    = Path('/tmp/s3e_extract')

ROBOTS = ['Alpha', 'Bob', 'Carol']

# S3Ev2 sequences that have no stereo cameras (LiDAR-only bags)
LIDAR_ONLY_SEQUENCES = {'S3E_Tunnel_1', 'S3E_Library_2'}

# ── CDR parsing ────────────────────────────────────────────────────────────────

def extract_jpeg_from_compressed_image_cdr(data: bytes):
    """
    Parse CDR-serialized sensor_msgs/msg/CompressedImage.
    Returns (timestamp_ns, jpeg_bytes) or (None, None) on failure.
    """
    if len(data) < 12:
        return None, None
    # CDR encapsulation header: 4 bytes (byte 1 = endianness flag)
    little_endian = (data[1] & 0x01) == 0
    fmt = '<' if little_endian else '>'

    off = 4
    # Header.stamp.sec (int32) + nanosec (uint32)
    sec    = struct.unpack_from(f'{fmt}i', data, off)[0]; off += 4
    nanosec = struct.unpack_from(f'{fmt}I', data, off)[0]; off += 4
    timestamp_ns = sec * 1_000_000_000 + nanosec

    # Find JPEG magic bytes (most reliable extraction for compressed topics)
    jpeg_start = data.find(b'\xff\xd8\xff', off)
    if jpeg_start < 0:
        return None, None
    return timestamp_ns, bytes(data[jpeg_start:])


# ── Frame extraction ───────────────────────────────────────────────────────────

def extract_stereo_frames(bag_path: Path, robot: str, out_dir: Path,
                          frame_skip: int = 1) -> int:
    """
    Extract left+right stereo frames from a ROS2 .db3 bag for the given robot.
    Saves to EuRoC format:  out_dir/cam0/data/<ts>.jpg  and  cam1/data/<ts>.jpg
                            out_dir/cam0/data.csv
    Returns number of frames extracted.
    """
    topic_left  = f'/{robot}/left_camera/compressed'
    topic_right = f'/{robot}/right_camera/compressed'

    cam0_dir = out_dir / 'cam0' / 'data'
    cam1_dir = out_dir / 'cam1' / 'data'
    cam0_dir.mkdir(parents=True, exist_ok=True)
    cam1_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(bag_path))
    cur  = conn.cursor()

    # Get topic IDs
    cur.execute("SELECT id, name FROM topics")
    topic_map = {name: tid for tid, name in cur.fetchall()}

    if topic_left not in topic_map or topic_right not in topic_map:
        log.warning(f'  Topics not found for {robot} in {bag_path.name}')
        conn.close()
        return 0

    tid_left  = topic_map[topic_left]
    tid_right = topic_map[topic_right]

    # Load all messages for both topics, sorted by timestamp
    cur.execute("SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
                (tid_left,))
    left_msgs = cur.fetchall()

    cur.execute("SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
                (tid_right,))
    right_msgs = cur.fetchall()
    conn.close()

    if not left_msgs or not right_msgs:
        log.warning(f'  No messages for {robot}')
        return 0

    # Match left/right by nearest timestamp (within 50ms)
    MAX_DT = 50_000_000  # 50ms in nanoseconds
    timestamps = []
    ri = 0
    for li, (lt, ldata) in enumerate(left_msgs):
        if li % frame_skip != 0:
            continue
        # Find nearest right frame
        while ri + 1 < len(right_msgs) and abs(right_msgs[ri+1][0] - lt) < abs(right_msgs[ri][0] - lt):
            ri += 1
        rt, rdata = right_msgs[ri]
        if abs(rt - lt) > MAX_DT:
            continue

        ts_ns, left_jpeg  = extract_jpeg_from_compressed_image_cdr(bytes(ldata))
        _,     right_jpeg = extract_jpeg_from_compressed_image_cdr(bytes(rdata))
        if left_jpeg is None or right_jpeg is None:
            continue

        ts = lt  # use left timestamp as canonical
        ts_str = str(ts)
        # Decode JPEG → grayscale → save as PNG (SLAM hardcodes .png + IMREAD_GRAYSCALE)
        left_arr  = cv2.imdecode(np.frombuffer(left_jpeg,  np.uint8), cv2.IMREAD_GRAYSCALE)
        right_arr = cv2.imdecode(np.frombuffer(right_jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
        if left_arr is None or right_arr is None:
            continue
        cv2.imwrite(str(cam0_dir / f'{ts_str}.png'), left_arr)
        cv2.imwrite(str(cam1_dir / f'{ts_str}.png'), right_arr)
        timestamps.append(ts)

    # Write data.csv (EuRoC format: header + "timestamp,filename")
    csv_path = out_dir / 'cam0' / 'data.csv'
    with open(csv_path, 'w') as f:
        f.write('#timestamp [ns],filename\n')
        for ts in timestamps:
            f.write(f'{ts},{ts}.jpg\n')

    log.info(f'  Extracted {len(timestamps)} stereo pairs for {robot}')
    return len(timestamps)


# ── SLAM runner ────────────────────────────────────────────────────────────────

def run_slam(seq_dir: Path, out_db: Path, config_yaml: Path, frame_skip: int = 1) -> bool:
    """Run Structure-PLP-SLAM on the EuRoC-format sequence directory."""
    cmd = [
        str(SLAM_BIN),
        '-v', str(VOCAB_FILE),
        '-c', str(config_yaml),
        '-d', str(seq_dir),
        '-p', str(out_db),
        '--no-sleep',
        '--auto-term',
    ]
    if frame_skip > 1:
        cmd += ['--frame-skip', str(frame_skip)]

    log.info(f'  Running SLAM → {out_db.name}')
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        log.error(f'  SLAM failed (rc={result.returncode}): {result.stderr[-500:]}')
        return False
    if not out_db.exists() or out_db.stat().st_size < 1024:
        log.warning(f'  SLAM produced empty/missing db')
        return False
    log.info(f'  Done → {out_db.stat().st_size // 1024} KB')
    return True


# ── Main ───────────────────────────────────────────────────────────────────────

def find_sequences():
    """Yield (version_name, sequence_name, db3_path) for all S3E sequences."""
    for version_dir in sorted(S3E_ROOT.iterdir()):
        if not version_dir.is_dir() or not version_dir.name.startswith('S3E'):
            continue
        for seq_dir in sorted(version_dir.iterdir()):
            if not seq_dir.is_dir() or seq_dir.name == 'Calibration':
                continue
            db3 = seq_dir / f'{seq_dir.name}.db3'
            if db3.exists():
                yield version_dir.name, seq_dir.name, db3


def main():
    ap = argparse.ArgumentParser(description='Process S3E dataset for ScalePluckerNet')
    ap.add_argument('--dry-run',      action='store_true', help='Print plan without executing')
    ap.add_argument('--skip-existing', action='store_true', help='Skip sequences with existing .db output')
    ap.add_argument('--frame-skip',   type=int, default=3, help='Process every Nth frame (default: 3)')
    ap.add_argument('--robots',       nargs='+', default=ROBOTS,
                    choices=ROBOTS, help='Which robots to process (default: all)')
    ap.add_argument('--sequences',    nargs='+', default=None,
                    help='Specific sequence names to process (default: all)')
    args = ap.parse_args()

    # Sanity checks
    for path, name in [(SLAM_BIN, 'SLAM binary'), (VOCAB_FILE, 'vocab file'),
                       (S3E_ROOT, 'S3E dataset')]:
        if not Path(path).exists():
            log.error(f'{name} not found: {path}')
            sys.exit(1)
    for robot, cfg in CONFIG_YAML.items():
        if not cfg.exists():
            log.error(f'config YAML not found for {robot}: {cfg}')
            sys.exit(1)

    sequences = list(find_sequences())
    log.info(f'Found {len(sequences)} S3E sequences × {len(args.robots)} robots')

    ok_count = fail_count = skip_count = 0

    for version, seq_name, db3_path in sequences:
        if args.sequences and seq_name not in args.sequences:
            continue

        if seq_name in LIDAR_ONLY_SEQUENCES:
            log.info(f'[SKIP] {seq_name} — LiDAR-only bag, no stereo cameras')
            skip_count += len(args.robots)
            continue

        for robot in args.robots:
            out_db = SLAM_DIR / f's3e_{seq_name}_{robot.lower()}_map.db'
            tag    = f'{seq_name}/{robot}'

            if args.skip_existing and out_db.exists():
                log.info(f'[SKIP] {tag} — already exists')
                skip_count += 1
                continue

            log.info(f'[PROCESS] {tag}  ({db3_path.stat().st_size // 1024 // 1024} MB)')

            if args.dry_run:
                log.info(f'  → {out_db}')
                continue

            # Extract frames to temp dir
            temp_dir = TEMP_ROOT / f'{seq_name}_{robot}'
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

            n_frames = extract_stereo_frames(db3_path, robot, temp_dir,
                                             frame_skip=args.frame_skip)
            if n_frames < 20:
                log.warning(f'  Too few frames ({n_frames}), skipping SLAM')
                shutil.rmtree(temp_dir, ignore_errors=True)
                fail_count += 1
                continue

            # Run SLAM with per-robot calibration
            success = run_slam(temp_dir, out_db, CONFIG_YAML[robot], frame_skip=1)

            # Always clean up temp frames
            shutil.rmtree(temp_dir, ignore_errors=True)

            if success:
                ok_count += 1
            else:
                fail_count += 1

    log.info(f'\nDone: {ok_count} succeeded, {fail_count} failed, {skip_count} skipped')
    log.info(f'Map files in: {SLAM_DIR}/')


if __name__ == '__main__':
    main()
