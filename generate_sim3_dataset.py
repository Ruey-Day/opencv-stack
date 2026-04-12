#!/usr/bin/env python3
"""
Generate synthetic Sim(3) Plücker line datasets for training.

Each scene: two line sets related by a random Sim(3) = (s, R, t) transform,
plus outlier lines with no correspondence.

Output format: one pickle file per variable, a list of N arrays:
  matches.pkl   list of (2, n_inliers) int arrays
  plucker1.pkl  list of (n_lines, 6) float32 arrays  [m0 m1 m2 d0 d1 d2]
  plucker2.pkl  same
  R_gt.pkl      list of (3, 3) float32 arrays
  t_gt.pkl      list of (3, 1) float32 arrays
  s_gt.pkl      list of float32 scalars

Usage:
    python generate_sim3_dataset.py --out_dir ./dataset --n_train 5000 --n_valid 500
"""
import argparse
import os
import pickle
import numpy as np


def random_rotation():
    """Uniform random rotation from SO(3) via QR decomposition."""
    A = np.random.randn(3, 3).astype(np.float64)
    Q, R_mat = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R_mat))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float32)


def make_lines(n, pos_range=2.0):
    """Generate n random 3D lines as Plücker coordinates (m, d).

    Lines are parameterised by a random point p on the line and a random
    unit direction d. The moment is m = p × d.
    Format: [m0 m1 m2 d0 d1 d2] per row.
    """
    d = np.random.randn(n, 3).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    p = np.random.uniform(-pos_range, pos_range, (n, 3)).astype(np.float32)
    m = np.cross(p, d)
    return np.concatenate([m, d], axis=1)  # (n, 6)


def make_direction_clustered_lines(n, n_dir_clusters=10, dir_spread=0.15,
                                   pos_range=2.0):
    """Generate n Plücker lines with directions clustered around anchor directions.

    Why direction clustering?
    -------------------------
    The model's KNN path for x[:,3:,:] (directions in [m,d] format) computes
    nearest-neighbours by DIRECTION similarity.  For this path to carry a useful
    signal across both views, the KNN neighbourhoods must stay consistent after
    the Sim(3) transform.  Under d' = R·d, lines that share a similar direction
    d_anchor in view 1 share the similar direction R·d_anchor in view 2 — the
    cluster structure is exactly preserved.

    By contrast, with fully random directions the direction-KNN neighbourhood
    of each line in view 1 has no overlap with its neighbourhood in view 2, so
    the GNN never sees consistent local context and cannot learn to match.

    Parameters
    ----------
    n_dir_clusters : int
        Number of anchor directions.  10 gives ≈10 lines per cluster for n=100.
    dir_spread : float
        Std-dev of the von-Mises-like perturbation on the unit sphere.
        0.15 rad ≈ 8.6°, giving tight but distinct clusters.
    pos_range : float
        Half-extent of the uniform distribution for point positions.
    """
    n_per = n // n_dir_clusters
    extras = n - n_per * n_dir_clusters

    # Random anchor directions, well-spread on the unit sphere
    anchor_dirs = np.random.randn(n_dir_clusters, 3).astype(np.float32)
    anchor_dirs /= np.linalg.norm(anchor_dirs, axis=1, keepdims=True)

    parts = []
    for i, anchor_d in enumerate(anchor_dirs):
        cnt = n_per + (1 if i < extras else 0)
        # Perturb around anchor direction
        noise = np.random.randn(cnt, 3).astype(np.float32) * dir_spread
        d = anchor_d[None] + noise
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        # Random point on the line
        p = np.random.uniform(-pos_range, pos_range, (cnt, 3)).astype(np.float32)
        m = np.cross(p, d)
        parts.append(np.concatenate([m, d], axis=1))

    lines = np.concatenate(parts, axis=0)
    idx = np.random.permutation(len(lines))
    return lines[idx]


def make_clustered_lines(n, n_clusters=10, cluster_spread=0.35, pos_range=2.0):
    """Generate n Plücker lines clustered around spatial anchor points.

    Kept for backward compatibility.  New datasets should prefer
    make_direction_clustered_lines which gives consistent KNN structure
    across the Sim(3) transform (direction clusters are preserved under R).
    """
    n_per = n // n_clusters
    extras = n - n_per * n_clusters
    anchors = np.random.uniform(-pos_range, pos_range,
                                (n_clusters, 3)).astype(np.float32)
    parts = []
    for i, anchor in enumerate(anchors):
        cnt = n_per + (1 if i < extras else 0)
        d = np.random.randn(cnt, 3).astype(np.float32)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        p = anchor + np.random.randn(cnt, 3).astype(np.float32) * cluster_spread
        m = np.cross(p, d)
        parts.append(np.concatenate([m, d], axis=1))
    lines = np.concatenate(parts, axis=0)
    idx = np.random.permutation(len(lines))
    return lines[idx]


def apply_sim3(lines, s, R, t):
    """Apply Sim(3) = (s, R, t) to a set of Plücker lines.

    Transformation law:
        d' = R d
        m' = s R m + t × d'
    """
    m = lines[:, :3]
    d = lines[:, 3:]
    d_new = (R @ d.T).T                         # (n, 3)
    m_new = s * (R @ m.T).T + np.cross(t, d_new)  # (n, 3)
    return np.concatenate([m_new, d_new], axis=1)


def generate_scene(n_inliers, n_outliers, scale_range=(0.3, 3.0), n_dir_clusters=10):
    """Generate one scene pair with a random Sim(3) transform.

    Uses direction-clustered lines so that the KNN structure in x[:,3:,:]
    (direction channels) is consistent across both views.  Under d' = R·d,
    lines that share a direction cluster in view 1 share the same rotated
    cluster in view 2 — giving the GNN a stable local context signal.

    n_dir_clusters=10 with n_inliers=100 gives 10 lines per direction cluster,
    so each line has ~9 within-cluster KNN neighbours (assuming net_knn=10).
    This is the ideal case for the GNN: rich intra-cluster context plus
    inter-cluster cross-attention.
    """
    lines1_in = make_direction_clustered_lines(n_inliers, n_dir_clusters=n_dir_clusters)

    # Random Sim(3) parameters — log-uniform scale for balanced coverage
    log_s = np.random.uniform(np.log(scale_range[0]), np.log(scale_range[1]))
    s = float(np.exp(log_s))
    R = random_rotation()
    t = np.random.uniform(-1.5, 1.5, 3).astype(np.float32)

    lines2_in = apply_sim3(lines1_in, s, R, t)

    # Outlier lines — direction-clustered with fewer clusters so outlier clusters
    # are detectable as "unmatched" by the GNN
    n_out_clusters = max(1, n_dir_clusters // 3)
    lines1_out = make_direction_clustered_lines(n_outliers, n_dir_clusters=n_out_clusters)
    lines2_out = make_direction_clustered_lines(n_outliers, n_dir_clusters=n_out_clusters)

    lines1 = np.concatenate([lines1_in, lines1_out], axis=0)
    lines2 = np.concatenate([lines2_in, lines2_out], axis=0)

    # Shuffle so inliers are not always first
    idx1 = np.random.permutation(len(lines1))
    idx2 = np.random.permutation(len(lines2))
    lines1 = lines1[idx1]
    lines2 = lines2[idx2]

    # Recover where the inlier lines ended up after the shuffle
    inv_idx1 = np.argsort(idx1)
    inv_idx2 = np.argsort(idx2)
    src_inds = inv_idx1[:n_inliers]
    tgt_inds = inv_idx2[:n_inliers]
    matches = np.stack([src_inds, tgt_inds], axis=0).astype(np.int32)  # (2, n_inliers)

    return {
        'plucker1': lines1.astype(np.float32),
        'plucker2': lines2.astype(np.float32),
        'matches':  matches,
        'R_gt':     R,
        't_gt':     t.reshape(3, 1),
        's_gt':     np.float32(s),
    }


def generate_split(n_scenes, out_dir, n_inliers=100, n_outliers=30, n_dir_clusters=10):
    """Generate n_scenes with fixed line counts so PyTorch can batch them.

    All scenes in a split must have the same total number of lines
    (n_inliers + n_outliers) so the default collate_fn can stack the
    correspondence matrices and Plücker arrays into batched tensors.

    n_dir_clusters=10 gives 10 lines per direction cluster for n_inliers=100,
    which matches net_knn=10 so each line's KNN neighbourhood is exactly its
    cluster — consistent and informative across both views.
    """
    os.makedirs(out_dir, exist_ok=True)
    keys = ['matches', 'plucker1', 'plucker2', 'R_gt', 't_gt', 's_gt']
    data = {k: [] for k in keys}

    for i in range(n_scenes):
        scene = generate_scene(n_inliers, n_outliers, n_dir_clusters=n_dir_clusters)
        for k in keys:
            data[k].append(scene[k])
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n_scenes} scenes generated")

    for k, v in data.items():
        with open(os.path.join(out_dir, f'{k}.pkl'), 'wb') as f:
            pickle.dump(v, f)
    print(f"  Saved {n_scenes} scenes to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir',    type=str, default='./dataset')
    parser.add_argument('--dataset',    type=str, default='sim3_synthetic')
    parser.add_argument('--n_train',    type=int, default=5000)
    parser.add_argument('--n_valid',    type=int, default=500)
    parser.add_argument('--n_inliers',  type=int, default=100,
                        help='inlier lines per scene (fixed across all scenes)')
    parser.add_argument('--n_outliers', type=int, default=30,
                        help='outlier lines per scene (fixed across all scenes)')
    parser.add_argument('--n_dir_clusters', type=int, default=10,
                        help='direction clusters per scene; 10 gives 10 lines/cluster '
                             'for n_inliers=100, matching net_knn=10')
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"Generating {args.n_train} training scenes "
          f"({args.n_inliers} inliers + {args.n_outliers} outliers = "
          f"{args.n_inliers + args.n_outliers} lines per set, "
          f"{args.n_dir_clusters} direction clusters)...")
    generate_split(args.n_train,
                   os.path.join(args.out_dir, f'{args.dataset}_train'),
                   args.n_inliers, args.n_outliers, args.n_dir_clusters)

    print(f"Generating {args.n_valid} validation scenes...")
    generate_split(args.n_valid,
                   os.path.join(args.out_dir, f'{args.dataset}_valid'),
                   args.n_inliers, args.n_outliers, args.n_dir_clusters)

    print("Done.")


if __name__ == '__main__':
    main()
