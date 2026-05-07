#!/usr/bin/env python3
"""
generate_replica_dataset.py

Builds real-geometry Sim(3) Plücker line training pairs from the Replica RGBD dataset.

Strategy
--------
For each Replica scene:
  1. Load ~20 uniformly-spaced RGBD frames and build a merged, voxel-downsampled
     world-space point cloud (real geometry, not synthetic).
  2. Extract candidate inlier lines via local PCA on the cloud.
  3. Generate K training pairs by:
       a. Subsampling n_inliers lines from the candidate pool.
       b. Applying a random Sim(3) transform to produce the second cloud.
       c. Appending n_outliers synthetic direction-clustered lines (no correspondence).
  4. Shuffle both line sets independently and store ground-truth indices.

Output format is identical to generate_sim3_dataset.py — the existing
Sim3PluckerData / Sim3Trainer work without any changes.

Usage
-----
    python generate_replica_dataset.py

Output
------
    dataset/replica_train/  (matches.pkl, plucker1.pkl, plucker2.pkl, R_gt.pkl, t_gt.pkl, s_gt.pkl)
    dataset/replica_valid/
"""

import os
import sys
import glob
import pickle
import argparse
import numpy as np
from scipy.spatial import cKDTree

# ── Replica constants ──────────────────────────────────────────────────────────
REPLICA_ROOT = "/home/rueyday/data/Replica"
FX = FY = 600.0
CX, CY = 599.5, 339.5
DEPTH_SCALE = 6553.5

# 7 scenes for training, 1 (room2) held out for validation
TRAIN_SCENES = ["office0", "office1", "office2", "office3", "office4", "room0", "room1"]
VALID_SCENES = ["room2"]

# ── Geometry helpers (copied from generate_sim3_dataset.py) ────────────────────

def random_rotation():
    A = np.random.randn(3, 3).astype(np.float64)
    Q, R_mat = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R_mat))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float32)


def apply_sim3(lines, s, R, t):
    """Apply Sim(3) to (n,6) lines in [m, d] format."""
    m = lines[:, :3]
    d = lines[:, 3:6]
    d_new = (R @ d.T).T
    m_new = s * (R @ m.T).T + np.cross(t.flatten()[None], d_new)
    return np.concatenate([m_new, d_new], axis=1).astype(np.float32)


def make_direction_clustered_lines(n, n_dir_clusters=5, dir_spread=0.15, pos_range=2.0):
    """Synthetic outlier lines with direction clustering (same as training data)."""
    n_per = n // n_dir_clusters
    extras = n - n_per * n_dir_clusters
    anchors = np.random.randn(n_dir_clusters, 3).astype(np.float32)
    anchors /= np.linalg.norm(anchors, axis=1, keepdims=True)
    parts = []
    for i, anchor in enumerate(anchors):
        cnt = n_per + (1 if i < extras else 0)
        noise = np.random.randn(cnt, 3).astype(np.float32) * dir_spread
        d = anchor[None] + noise
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        p = np.random.uniform(-pos_range, pos_range, (cnt, 3)).astype(np.float32)
        m = np.cross(p, d)
        parts.append(np.concatenate([m, d], axis=1))
    lines = np.concatenate(parts, axis=0)
    return lines[np.random.permutation(len(lines))]


# ── Point cloud building ───────────────────────────────────────────────────────

def load_replica_poses(scene_dir):
    poses = []
    with open(os.path.join(scene_dir, "traj.txt")) as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) == 16:
                poses.append(np.array([float(v) for v in vals],
                                      dtype=np.float32).reshape(4, 4))
    return poses


def build_replica_cloud(scene_dir, every_n=100, max_depth=4.5,
                         subsample=4, voxel=0.03):
    """Load Replica depth frames, back-project to world space, voxel downsample."""
    import cv2
    poses       = load_replica_poses(scene_dir)
    depth_files = sorted(glob.glob(os.path.join(scene_dir, "results", "depth*.png")))
    selected    = depth_files[::every_n]

    pts_all = []
    for df in selected:
        idx = int(os.path.splitext(os.path.basename(df))[0].replace("depth", ""))
        if idx >= len(poses):
            continue
        T = poses[idx]  # c2w
        depth = cv2.imread(df, cv2.IMREAD_ANYDEPTH).astype(np.float32) / DEPTH_SCALE

        H, W  = depth.shape
        vi, ui = np.meshgrid(np.arange(0, H, subsample),
                             np.arange(0, W, subsample), indexing="ij")
        vi, ui = vi.ravel(), ui.ravel()
        z  = depth[vi, ui]
        ok = (z > 0.1) & (z < max_depth)
        z, vi, ui = z[ok], vi[ok], ui[ok]
        x  = (ui - CX) * z / FX
        y  = (vi - CY) * z / FY
        cam = np.stack([x, y, z, np.ones_like(z)], 0)
        pts_all.append((T @ cam)[:3].T)

    if not pts_all:
        return np.zeros((0, 3), dtype=np.float32)

    cloud = np.concatenate(pts_all, 0)
    keys  = np.floor(cloud / voxel).astype(np.int32)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return cloud[idx].astype(np.float32)


# ── Line extraction ────────────────────────────────────────────────────────────

def extract_lines_from_cloud(cloud, n_lines, k=25, linearity_thresh=0.75, seed=0):
    """Extract 3D line segments via local PCA. Returns (n_lines, 6) in [m, d] format."""
    if len(cloud) < k + 1:
        return None

    rng  = np.random.default_rng(seed)
    tree = cKDTree(cloud)
    mids, dirs = [], []

    indices = rng.choice(len(cloud), size=min(30000, len(cloud)), replace=False)
    for idx in indices:
        if len(mids) >= n_lines:
            break
        nn  = tree.query(cloud[idx], k=k)[1]
        pts = cloud[nn]
        ctr = pts.mean(0)
        cov = (pts - ctr).T @ (pts - ctr) / k
        ev, evec = np.linalg.eigh(cov)
        lam = ev[::-1]
        if (lam[0] - lam[1]) / (lam[0] + 1e-9) > linearity_thresh:
            mids.append(ctr)
            dirs.append(evec[:, -1])

    if len(mids) < n_lines:
        return None

    mids = np.array(mids, dtype=np.float32)
    dirs = np.array(dirs, dtype=np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    m = np.cross(mids, dirs)
    return np.concatenate([m, dirs], axis=1)   # [m, d]


# ── Scene pair generation ──────────────────────────────────────────────────────

def generate_scene_from_lines(candidate_pool, n_inliers, n_outliers, scale_range):
    """
    Generate one training pair from a pre-extracted candidate line pool.

    Subsamples n_inliers lines, applies a random Sim3, appends synthetic outliers.
    Returns the same dict as generate_sim3_dataset.generate_scene().
    """
    n_cands = len(candidate_pool)
    if n_cands < n_inliers + n_outliers:
        return None

    # Sample inlier lines from the pool
    inlier_idx = np.random.choice(n_cands, n_inliers, replace=False)
    lines1_in  = candidate_pool[inlier_idx].copy()

    # Random Sim3
    log_s = np.random.uniform(np.log(scale_range[0]), np.log(scale_range[1]))
    s     = float(np.exp(log_s))
    R     = random_rotation()
    t     = np.random.uniform(-1.5, 1.5, 3).astype(np.float32)
    lines2_in = apply_sim3(lines1_in, s, R, t)

    # Synthetic outlier lines (direction-clustered, no correspondence)
    lines1_out = make_direction_clustered_lines(n_outliers, n_dir_clusters=3)
    lines2_out = make_direction_clustered_lines(n_outliers, n_dir_clusters=3)

    lines1 = np.concatenate([lines1_in, lines1_out], axis=0)
    lines2 = np.concatenate([lines2_in, lines2_out], axis=0)

    idx1 = np.random.permutation(len(lines1))
    idx2 = np.random.permutation(len(lines2))
    lines1 = lines1[idx1]
    lines2 = lines2[idx2]

    inv1 = np.argsort(idx1)
    inv2 = np.argsort(idx2)
    src_inds = inv1[:n_inliers]
    tgt_inds = inv2[:n_inliers]
    matches  = np.stack([src_inds, tgt_inds], axis=0).astype(np.int32)

    return {
        'plucker1': lines1.astype(np.float32),
        'plucker2': lines2.astype(np.float32),
        'matches':  matches,
        'R_gt':     R,
        't_gt':     t.reshape(3, 1),
        's_gt':     np.float32(s),
    }


# ── Split generation ───────────────────────────────────────────────────────────

def generate_split(scene_names, out_dir, n_scenes_per_scene,
                   n_inliers, n_outliers, n_candidate_lines,
                   scale_range, seed):
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(seed)

    keys = ['matches', 'plucker1', 'plucker2', 'R_gt', 't_gt', 's_gt']
    data = {k: [] for k in keys}
    total = 0

    for scene_name in scene_names:
        scene_dir = os.path.join(REPLICA_ROOT, scene_name)
        print(f"  [{scene_name}] building point cloud ...")
        cloud = build_replica_cloud(scene_dir, every_n=100, max_depth=4.5,
                                     subsample=3, voxel=0.025)
        print(f"  [{scene_name}] {cloud.shape[0]:,} points — extracting lines ...")

        candidate_pool = extract_lines_from_cloud(
            cloud, n_candidate_lines, k=20, linearity_thresh=0.60,
            seed=seed + hash(scene_name) % 10000
        )
        if candidate_pool is None:
            print(f"  [{scene_name}] WARNING: not enough linear segments — skipping")
            continue

        print(f"  [{scene_name}] {len(candidate_pool)} candidate lines — "
              f"generating {n_scenes_per_scene} pairs ...")

        n_ok = 0
        for i in range(n_scenes_per_scene * 3):   # 3× attempts to handle rare failures
            if n_ok >= n_scenes_per_scene:
                break
            scene = generate_scene_from_lines(
                candidate_pool, n_inliers, n_outliers, scale_range
            )
            if scene is None:
                continue
            for k in keys:
                data[k].append(scene[k])
            n_ok += 1

        total += n_ok
        print(f"  [{scene_name}] {n_ok} pairs generated  (total so far: {total})")

    print(f"\nSaving {total} scenes to {out_dir} ...")
    for k, v in data.items():
        with open(os.path.join(out_dir, f'{k}.pkl'), 'wb') as f:
            pickle.dump(v, f)
    print(f"Done — {total} scenes saved.")
    return total


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir',
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset'))
    parser.add_argument('--n_train_per_scene',    type=int, default=600)
    parser.add_argument('--n_valid_per_scene',    type=int, default=200)
    parser.add_argument('--n_inliers',            type=int, default=100)
    parser.add_argument('--n_outliers',           type=int, default=30)
    parser.add_argument('--n_candidate_lines',    type=int, default=400,
                        help='Lines extracted from each scene cloud (pool for subsampling)')
    parser.add_argument('--seed',                 type=int, default=0)
    args = parser.parse_args()

    print("=" * 60)
    print("Replica Sim(3) dataset generation")
    print(f"  Train scenes: {TRAIN_SCENES}  ({args.n_train_per_scene} pairs each)")
    print(f"  Valid scenes: {VALID_SCENES}  ({args.n_valid_per_scene} pairs each)")
    print(f"  Lines per pair: {args.n_inliers} inliers + {args.n_outliers} outliers")
    print("=" * 60)

    print("\n── TRAIN ──")
    generate_split(
        TRAIN_SCENES,
        os.path.join(args.out_dir, 'replica_train'),
        args.n_train_per_scene,
        args.n_inliers, args.n_outliers, args.n_candidate_lines,
        scale_range=(0.3, 3.0),
        seed=args.seed,
    )

    print("\n── VALID ──")
    generate_split(
        VALID_SCENES,
        os.path.join(args.out_dir, 'replica_valid'),
        args.n_valid_per_scene,
        args.n_inliers, args.n_outliers, args.n_candidate_lines,
        scale_range=(0.3, 3.0),
        seed=args.seed + 99999,
    )

    print("\nAll done.")


if __name__ == '__main__':
    main()
