#!/usr/bin/env python3
"""
generate_replica_color_dataset.py

Extends generate_replica_dataset.py with RGB color sampling.

Each extracted line gets an average RGB color from its local neighborhood
in the point cloud (sampled from the corresponding frame images).

Output: 9D lines [m0,m1,m2, d0,d1,d2, r,g,b] where RGB ∈ [0, 1].

Training uses per-line color dropout (~30% of lines have RGB zeroed) so the
model handles both colored and colorless inputs with a single 9D network.

Output directories:
    dataset/replica_color_train/
    dataset/replica_color_valid/

Format is identical to the 6D dataset (same pkl keys) except plucker1/plucker2
have shape (n_lines, 9) instead of (n_lines, 6).
"""

import os
import sys
import glob
import pickle
import argparse
import numpy as np
from scipy.spatial import cKDTree

# Re-use constants and helpers from the 6D generator
sys.path.insert(0, os.path.dirname(__file__))
from generate_replica_dataset import (
    REPLICA_ROOT, FX, FY, CX, CY, DEPTH_SCALE,
    TRAIN_SCENES, VALID_SCENES,
    load_replica_poses, random_rotation,
    make_direction_clustered_lines,
)

COLOR_DROPOUT_PROB = 0.30   # fraction of lines whose RGB is zeroed during training
NEUTRAL_COLOR      = 0.5    # replacement value for dropped-out channels

# ── Colored point cloud ────────────────────────────────────────────────────────

def build_replica_color_cloud(scene_dir, every_n=100, max_depth=4.5,
                               subsample=3, voxel=0.025):
    """Load Replica depth+RGB frames; build voxel-downsampled colored cloud.

    Returns:
        cloud_xyz: (N, 3) float32
        cloud_rgb: (N, 3) float32  values in [0, 1]
    """
    import cv2
    poses       = load_replica_poses(scene_dir)
    depth_files = sorted(glob.glob(os.path.join(scene_dir, "results", "depth*.png")))
    selected    = depth_files[::every_n]

    xyz_all, rgb_all = [], []

    for df in selected:
        idx = int(os.path.splitext(os.path.basename(df))[0].replace("depth", ""))
        if idx >= len(poses):
            continue

        # Corresponding RGB frame: frame000000.jpg etc.
        color_path = os.path.join(
            scene_dir, "results",
            f"frame{idx:06d}.jpg"
        )
        if not os.path.exists(color_path):
            continue

        T     = poses[idx]
        depth = cv2.imread(df, cv2.IMREAD_ANYDEPTH).astype(np.float32) / DEPTH_SCALE
        color = cv2.imread(color_path, cv2.IMREAD_COLOR)  # BGR uint8
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        H, W  = depth.shape
        vi, ui = np.meshgrid(np.arange(0, H, subsample),
                             np.arange(0, W, subsample), indexing="ij")
        vi, ui = vi.ravel(), ui.ravel()
        z      = depth[vi, ui]
        ok     = (z > 0.1) & (z < max_depth)
        z, vi, ui = z[ok], vi[ok], ui[ok]

        x  = (ui - CX) * z / FX
        y  = (vi - CY) * z / FY
        cam = np.stack([x, y, z, np.ones_like(z)], 0)
        pts_world = (T @ cam)[:3].T          # (n, 3)

        rgb = color[vi, ui]                   # (n, 3)

        xyz_all.append(pts_world.astype(np.float32))
        rgb_all.append(rgb.astype(np.float32))

    if not xyz_all:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    xyz = np.concatenate(xyz_all, 0)
    rgb = np.concatenate(rgb_all, 0)

    # Voxel downsample — keep one point per voxel (the first one in each cell)
    keys  = np.floor(xyz / voxel).astype(np.int32)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return xyz[idx].astype(np.float32), rgb[idx].astype(np.float32)


# ── Colored line extraction ────────────────────────────────────────────────────

def extract_colored_lines(cloud_xyz, cloud_rgb, n_lines,
                           k=20, linearity_thresh=0.60, seed=0):
    """Extract Plücker lines with average-neighbor RGB color.

    Returns (n_lines, 9) float32: [m0..m2, d0..d2, r, g, b]
    """
    if len(cloud_xyz) < k + 1:
        return None

    rng  = np.random.default_rng(seed)
    tree = cKDTree(cloud_xyz)
    mids, dirs, colors = [], [], []

    indices = rng.choice(len(cloud_xyz),
                         size=min(30000, len(cloud_xyz)), replace=False)
    for idx in indices:
        if len(mids) >= n_lines:
            break
        nn  = tree.query(cloud_xyz[idx], k=k)[1]
        pts = cloud_xyz[nn]
        ctr = pts.mean(0)
        cov = (pts - ctr).T @ (pts - ctr) / k
        ev, evec = np.linalg.eigh(cov)
        lam = ev[::-1]
        if (lam[0] - lam[1]) / (lam[0] + 1e-9) > linearity_thresh:
            mids.append(ctr)
            dirs.append(evec[:, -1])
            colors.append(cloud_rgb[nn].mean(0))   # average neighbor color

    if len(mids) < n_lines:
        return None

    mids   = np.array(mids,   dtype=np.float32)
    dirs   = np.array(dirs,   dtype=np.float32)
    colors = np.array(colors, dtype=np.float32)
    dirs  /= np.linalg.norm(dirs, axis=1, keepdims=True)
    m      = np.cross(mids, dirs)
    return np.concatenate([m, dirs, colors], axis=1)   # (n, 9)


# ── Sim(3) transform for 9D lines ─────────────────────────────────────────────

def apply_sim3_9d(lines9, s, R, t):
    """Apply Sim(3) to (n, 9) lines; color is invariant."""
    m   = lines9[:, :3]
    d   = lines9[:, 3:6]
    rgb = lines9[:, 6:]
    d_new = (R @ d.T).T
    m_new = s * (R @ m.T).T + np.cross(t.flatten()[None], d_new)
    return np.concatenate([m_new, d_new, rgb], axis=1).astype(np.float32)


def make_colored_outliers(n, n_dir_clusters=3, dir_spread=0.15, pos_range=2.0):
    """Synthetic outlier lines with random RGB colors."""
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
        rgb = np.random.rand(cnt, 3).astype(np.float32)   # random color for outliers
        parts.append(np.concatenate([m, d, rgb], axis=1))
    lines = np.concatenate(parts, axis=0)
    return lines[np.random.permutation(len(lines))]


# ── Per-line color dropout ─────────────────────────────────────────────────────

def apply_color_dropout(lines9, dropout_prob=COLOR_DROPOUT_PROB):
    """Zero RGB channels on a random subset of lines (per-line independently)."""
    lines9 = lines9.copy()
    mask = np.random.rand(len(lines9)) < dropout_prob
    lines9[mask, 6:] = NEUTRAL_COLOR
    return lines9


# ── Scene pair generation ──────────────────────────────────────────────────────

def generate_colored_scene(candidate_pool_9d, n_inliers, n_outliers, scale_range,
                            dropout_prob=COLOR_DROPOUT_PROB):
    n_cands = len(candidate_pool_9d)
    if n_cands < n_inliers + n_outliers:
        return None

    inlier_idx = np.random.choice(n_cands, n_inliers, replace=False)
    lines1_in  = candidate_pool_9d[inlier_idx].copy()

    log_s = np.random.uniform(np.log(scale_range[0]), np.log(scale_range[1]))
    s     = float(np.exp(log_s))
    R     = random_rotation()
    t     = np.random.uniform(-1.5, 1.5, 3).astype(np.float32)
    lines2_in = apply_sim3_9d(lines1_in, s, R, t)

    lines1_out = make_colored_outliers(n_outliers, n_dir_clusters=3)
    lines2_out = make_colored_outliers(n_outliers, n_dir_clusters=3)

    lines1 = np.concatenate([lines1_in, lines1_out], axis=0)
    lines2 = np.concatenate([lines2_in, lines2_out], axis=0)

    # Apply per-line color dropout independently to both clouds
    lines1 = apply_color_dropout(lines1, dropout_prob)
    lines2 = apply_color_dropout(lines2, dropout_prob)

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

def generate_color_split(scene_names, out_dir, n_scenes_per_scene,
                          n_inliers, n_outliers, n_candidate_lines,
                          scale_range, seed):
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(seed)

    keys = ['matches', 'plucker1', 'plucker2', 'R_gt', 't_gt', 's_gt']
    data = {k: [] for k in keys}
    total = 0

    for scene_name in scene_names:
        scene_dir = os.path.join(REPLICA_ROOT, scene_name)
        print(f"  [{scene_name}] building colored point cloud ...")
        cloud_xyz, cloud_rgb = build_replica_color_cloud(
            scene_dir, every_n=100, max_depth=4.5, subsample=3, voxel=0.025
        )
        print(f"  [{scene_name}] {cloud_xyz.shape[0]:,} colored points")

        if cloud_xyz.shape[0] == 0:
            print(f"  [{scene_name}] WARNING: empty cloud — skipping")
            continue

        candidate_pool = extract_colored_lines(
            cloud_xyz, cloud_rgb, n_candidate_lines,
            k=20, linearity_thresh=0.60,
            seed=seed + hash(scene_name) % 10000
        )
        if candidate_pool is None:
            print(f"  [{scene_name}] WARNING: not enough linear segments — skipping")
            continue

        print(f"  [{scene_name}] {len(candidate_pool)} 9D candidate lines — "
              f"generating {n_scenes_per_scene} pairs ...")

        n_ok = 0
        for _ in range(n_scenes_per_scene * 3):
            if n_ok >= n_scenes_per_scene:
                break
            scene = generate_colored_scene(
                candidate_pool, n_inliers, n_outliers, scale_range
            )
            if scene is None:
                continue
            for k in keys:
                data[k].append(scene[k])
            n_ok += 1

        total += n_ok
        print(f"  [{scene_name}] {n_ok} pairs  (total: {total})")

    print(f"\nSaving {total} 9D scenes to {out_dir} ...")
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
    parser.add_argument('--n_train_per_scene', type=int, default=600)
    parser.add_argument('--n_valid_per_scene', type=int, default=200)
    parser.add_argument('--n_inliers',         type=int, default=100)
    parser.add_argument('--n_outliers',        type=int, default=30)
    parser.add_argument('--n_candidate_lines', type=int, default=400)
    parser.add_argument('--seed',              type=int, default=0)
    args = parser.parse_args()

    print("=" * 60)
    print("Replica Sim(3) COLOR dataset generation  (9D Plücker)")
    print(f"  Train: {TRAIN_SCENES}  ({args.n_train_per_scene} pairs each)")
    print(f"  Valid: {VALID_SCENES}  ({args.n_valid_per_scene} pairs each)")
    print(f"  Lines: {args.n_inliers} inliers + {args.n_outliers} outliers  (9D)")
    print(f"  Color dropout: {COLOR_DROPOUT_PROB*100:.0f}% per line")
    print("=" * 60)

    print("\n── TRAIN ──")
    generate_color_split(
        TRAIN_SCENES,
        os.path.join(args.out_dir, 'replica_color_train'),
        args.n_train_per_scene,
        args.n_inliers, args.n_outliers, args.n_candidate_lines,
        scale_range=(0.3, 3.0),
        seed=args.seed,
    )

    print("\n── VALID ──")
    generate_color_split(
        VALID_SCENES,
        os.path.join(args.out_dir, 'replica_color_valid'),
        args.n_valid_per_scene,
        args.n_inliers, args.n_outliers, args.n_candidate_lines,
        scale_range=(0.3, 3.0),
        seed=args.seed + 99999,
    )

    print("\nAll done.")


if __name__ == '__main__':
    main()
