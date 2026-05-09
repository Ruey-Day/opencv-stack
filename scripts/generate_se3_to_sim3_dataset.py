#!/usr/bin/env python3
"""
generate_se3_to_sim3_dataset.py

Convert the original PlueckerNet SE(3) training data (Semantic3D + Structured3D)
into Sim(3) training data by adding a random scale factor.

Strategy
--------
For each SE(3) scene (plucker1, plucker2, R_gt, t_gt, matches):
  1. Keep plucker1 unchanged — real geometry from the 3D scan.
  2. Pick a random scale s ~ log-uniform in [0.3, 3.0].
  3. Recompute the matched lines in plucker2 using the Sim(3) law:
         d2 = R * d1
         m2 = s * R * m1 + t × d2
     This replaces the SE(3) correspondences (s=1) with Sim(3) ones.
  4. Keep the unmatched (outlier) lines in plucker2 as-is — they are
     independent lines from the same 3D scan and remain valid outliers.
  5. Emit s_gt = s alongside the standard 5 pickles.

Both datasets use [m, d] format (confirmed by norm check):
    cols 0:3 = moment m = p × d   (larger magnitude)
    cols 3:6 = direction d        (unit vector)

Usage
-----
    python scripts/generate_se3_to_sim3_dataset.py
    # outputs:
    #   dataset/semantic3d_sim3_train/
    #   dataset/semantic3d_sim3_valid/
    #   dataset/structured3d_sim3_train/
    #   dataset/structured3d_sim3_valid/
    #   dataset/se3real_sim3_train/   (combined, used for training)
    #   dataset/se3real_sim3_valid/   (combined, used for validation)
"""

import argparse
import os
import sys
import pickle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLUECKERNET_DATASET = os.path.join(ROOT, '..', 'PlueckerNet', 'dataset')


def _load(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def _save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def apply_sim3_md(lines, s, R, t):
    """Apply Sim(3) to lines in [m, d] format: cols 0:3 = m, cols 3:6 = d."""
    m = lines[:, :3]
    d = lines[:, 3:6]
    d_new = (R @ d.T).T
    m_new = s * (R @ m.T).T + np.cross(t, d_new)
    return np.concatenate([m_new, d_new], axis=1).astype(np.float32)


def augment_split(src_dir, dst_dir, scale_range=(0.3, 3.0), seed=0,
                  se3_fraction=0.15, verbose=True):
    """
    Load SE(3) split from src_dir, augment with random scale, save to dst_dir.

    Parameters
    ----------
    se3_fraction : float
        Fraction of scenes to keep at s=1.0 (SE3-only).  Ensures the model
        sees some SE3 scenes during training (for generalisation).
    """
    os.makedirs(dst_dir, exist_ok=True)

    p1_all  = _load(os.path.join(src_dir, 'plucker1.pkl'))
    p2_all  = _load(os.path.join(src_dir, 'plucker2.pkl'))
    R_all   = _load(os.path.join(src_dir, 'R_gt.pkl'))
    t_all   = _load(os.path.join(src_dir, 't_gt.pkl'))
    m_all   = _load(os.path.join(src_dir, 'matches.pkl'))

    n_scenes = len(p1_all)
    rng = np.random.default_rng(seed)

    new_p1, new_p2, new_R, new_t, new_m, new_s = [], [], [], [], [], []

    for i in range(n_scenes):
        p1 = p1_all[i].astype(np.float32)    # (N, 6)  [m, d]
        p2 = p2_all[i].astype(np.float32)    # (N, 6)  [m, d]
        R  = R_all[i].astype(np.float32)     # (3, 3)
        t  = t_all[i].astype(np.float32).flatten()  # (3,)
        mi = m_all[i]                         # (2, n_inliers)
        src_idx, tgt_idx = mi[0], mi[1]

        # Draw scale: se3_fraction of scenes keep s=1 (SE3 scenario)
        if rng.random() < se3_fraction:
            s = 1.0
        else:
            log_s = rng.uniform(np.log(scale_range[0]), np.log(scale_range[1]))
            s = float(np.exp(log_s))

        # Regenerate matched lines in plucker2 with Sim(3) transform
        p2_new = p2.copy()
        p2_new[tgt_idx] = apply_sim3_md(p1[src_idx], s, R, t)
        # Unmatched lines in p2 stay untouched (independent outlier geometry)

        new_p1.append(p1)
        new_p2.append(p2_new)
        new_R.append(R)
        new_t.append(t.reshape(3, 1))
        new_m.append(mi)
        new_s.append(np.float32(s))

        if verbose and (i + 1) % 500 == 0:
            print(f"  {i+1}/{n_scenes} scenes augmented")

    _save(new_p1, os.path.join(dst_dir, 'plucker1.pkl'))
    _save(new_p2, os.path.join(dst_dir, 'plucker2.pkl'))
    _save(new_R,  os.path.join(dst_dir, 'R_gt.pkl'))
    _save(new_t,  os.path.join(dst_dir, 't_gt.pkl'))
    _save(new_m,  os.path.join(dst_dir, 'matches.pkl'))
    _save(new_s,  os.path.join(dst_dir, 's_gt.pkl'))
    print(f"  Saved {n_scenes} scenes → {dst_dir}")
    return new_p1, new_p2, new_R, new_t, new_m, new_s


def combine_splits(splits_data, dst_dir, seed=0):
    """Merge multiple augmented splits into one combined directory."""
    os.makedirs(dst_dir, exist_ok=True)

    combined = {k: [] for k in ['plucker1', 'plucker2', 'R_gt', 't_gt', 'matches', 's_gt']}
    for p1, p2, R, t, m, s in splits_data:
        combined['plucker1'].extend(p1)
        combined['plucker2'].extend(p2)
        combined['R_gt'].extend(R)
        combined['t_gt'].extend(t)
        combined['matches'].extend(m)
        combined['s_gt'].extend(s)

    # Shuffle
    rng = np.random.default_rng(seed)
    n = len(combined['plucker1'])
    idx = rng.permutation(n)
    for k in combined:
        combined[k] = [combined[k][i] for i in idx]

    for k, v in combined.items():
        _save(v, os.path.join(dst_dir, f'{k}.pkl'))

    print(f"  Combined {n} scenes → {dst_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plueckernet_dataset', default=PLUECKERNET_DATASET,
                        help='Path to PlueckerNet dataset/ directory')
    parser.add_argument('--out_dir', default=os.path.join(ROOT, 'dataset'),
                        help='Output directory for Sim3 datasets')
    parser.add_argument('--scale_min', type=float, default=0.3)
    parser.add_argument('--scale_max', type=float, default=3.0)
    parser.add_argument('--se3_fraction', type=float, default=0.15,
                        help='Fraction of scenes kept at s=1 (SE3 case)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    scale_range = (args.scale_min, args.scale_max)
    print(f"Scale range: [{args.scale_min}, {args.scale_max}]  "
          f"SE3 fraction: {args.se3_fraction}")

    datasets = ['semantic3D', 'structured3D']
    train_splits, valid_splits = [], []

    for dset in datasets:
        for split in ['train', 'valid']:
            src = os.path.join(args.plueckernet_dataset, f'{dset}_{split}')
            dst_name = dset.lower().replace('3d', '3d') + f'_sim3_{split}'
            dst = os.path.join(args.out_dir, dst_name)

            if not os.path.exists(src):
                print(f"[SKIP] {src} not found")
                continue

            print(f"\nAugmenting {dset}/{split} ...")
            data = augment_split(src, dst, scale_range=scale_range,
                                 seed=args.seed, se3_fraction=args.se3_fraction)
            if split == 'train':
                train_splits.append(data)
            else:
                valid_splits.append(data)

    # Combined dataset for training
    if train_splits:
        print("\nCombining train splits → se3real_sim3_train ...")
        combine_splits(train_splits,
                       os.path.join(args.out_dir, 'se3real_sim3_train'),
                       seed=args.seed)
    if valid_splits:
        print("Combining valid splits → se3real_sim3_valid ...")
        combine_splits(valid_splits,
                       os.path.join(args.out_dir, 'se3real_sim3_valid'),
                       seed=args.seed)

    print("\nDone.")
    print(f"Train on: python scripts/train_sim3_se3real.py  (dataset = se3real_sim3)")


if __name__ == '__main__':
    main()
