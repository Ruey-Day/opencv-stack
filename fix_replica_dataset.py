#!/usr/bin/env python3
"""
Regenerate the validation split (room2) and add missing room0/room1 training pairs.
Run while training is in progress — edits dataset files on disk.
Training will pick up new val data at the next validation epoch.
"""
import os
import sys
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from generate_replica_dataset import (
    REPLICA_ROOT, TRAIN_SCENES, generate_split, build_replica_cloud,
    extract_lines_from_cloud, generate_scene_from_lines,
)

DATA_DIR = './dataset'

# ── 1. Regenerate validation (room2, thresh=0.60 now in generate_split) ──────
print("\n── VALID (room2) ──")
generate_split(
    ['room2'],
    os.path.join(DATA_DIR, 'replica_valid'),
    n_scenes_per_scene=200,
    n_inliers=100, n_outliers=30, n_candidate_lines=400,
    scale_range=(0.3, 3.0),
    seed=99999,
)

# ── 2. Append room0 + room1 to existing training set ──────────────────────────
print("\n── TRAIN append (room0, room1) ──")
keys = ['matches', 'plucker1', 'plucker2', 'R_gt', 't_gt', 's_gt']
train_dir = os.path.join(DATA_DIR, 'replica_train')

# Load existing training data
existing = {}
for k in keys:
    with open(os.path.join(train_dir, f'{k}.pkl'), 'rb') as f:
        existing[k] = pickle.load(f)
print(f"Existing training pairs: {len(existing['plucker1'])}")

# Generate new pairs for room0 and room1
np.random.seed(42)
new_data = {k: [] for k in keys}
for scene_name in ['room0', 'room1']:
    scene_dir = os.path.join(REPLICA_ROOT, scene_name)
    print(f"  [{scene_name}] building point cloud ...")
    cloud = build_replica_cloud(scene_dir, every_n=100, max_depth=4.5,
                                subsample=3, voxel=0.025)
    print(f"  [{scene_name}] {cloud.shape[0]:,} pts — extracting lines ...")
    pool = extract_lines_from_cloud(cloud, 400, k=20, linearity_thresh=0.60,
                                    seed=hash(scene_name) % 10000)
    if pool is None:
        print(f"  [{scene_name}] WARNING: still not enough lines — skip")
        continue

    n_ok = 0
    for _ in range(600 * 3):
        if n_ok >= 600:
            break
        scene = generate_scene_from_lines(pool, 100, 30, (0.3, 3.0))
        if scene is None:
            continue
        for k in keys:
            new_data[k].append(scene[k])
        n_ok += 1
    print(f"  [{scene_name}] {n_ok} pairs generated")

# Merge and save
for k in keys:
    existing[k].extend(new_data[k])
print(f"\nTotal training pairs after merge: {len(existing['plucker1'])}")
print(f"Saving to {train_dir} ...")
for k in keys:
    with open(os.path.join(train_dir, f'{k}.pkl'), 'wb') as f:
        pickle.dump(existing[k], f)
print("Done.")
