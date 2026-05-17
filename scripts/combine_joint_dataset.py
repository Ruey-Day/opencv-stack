"""
    python scripts/combine_joint_dataset.py
    python scripts/combine_joint_dataset.py --data_dir ./dataset --seed 42
"""

import os, pickle, argparse
import numpy as np

KEYS = ['matches', 'plucker1', 'plucker2', 'R_gt', 't_gt', 's_gt']

def load_split(split_dir):
    data = {}
    for k in KEYS:
        path = os.path.join(split_dir, f'{k}.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(f'Missing: {path}')
        with open(path, 'rb') as f:
            data[k] = pickle.load(f, encoding='latin1')
    n  = len(data['plucker1'])
    ch = data['plucker1'][0].shape[1]
    print(f'  Loaded {n:,} scenes from {split_dir}  ({ch}D)')
    return data, n

def filter_scenes(data, min_scale=0.1, min_inliers=5):
    """Remove degenerate scenes: near-zero scale or too few ground-truth inliers."""
    n_before = len(data['plucker1'])
    keep = []
    for i in range(n_before):
        s = float(data['s_gt'][i])
        n_inliers = data['matches'][i].shape[1]
        if s >= min_scale and n_inliers >= min_inliers:
            keep.append(i)
    n_after = len(keep)
    print(f'  Filtered {n_before - n_after} degenerate scenes '
          f'(scale < {min_scale} or inliers < {min_inliers}) → {n_after} remain')
    return {k: [data[k][i] for i in keep] for k in KEYS}, n_after

def combine_and_save(source_dirs, out_dir, seed=42, min_scale=0.1, min_inliers=5):
    os.makedirs(out_dir, exist_ok=True)
    combined = {k: [] for k in KEYS}
    total    = 0

    for sd in source_dirs:
        if not os.path.exists(sd):
            print(f'  [SKIP] {sd} — not found yet')
            continue
        data, n = load_split(sd)
        data, n = filter_scenes(data, min_scale=min_scale, min_inliers=min_inliers)
        for k in KEYS:
            combined[k].extend(data[k])
        total += n

    if total == 0:
        print('ERROR: no data found — run generators first.')
        return 0

    rng  = np.random.default_rng(seed)
    perm = rng.permutation(total)
    for k in KEYS:
        combined[k] = [combined[k][i] for i in perm]

    for k, v in combined.items():
        with open(os.path.join(out_dir, f'{k}.pkl'), 'wb') as f:
            pickle.dump(v, f)

    print(f'  → {out_dir}  ({total:,} scenes total)')
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',    default='./dataset')
    p.add_argument('--out_dir',     default='./dataset')
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--min_scale',   type=float, default=0.1,
                   help='Drop scenes with GT scale below this value (default: 0.1)')
    p.add_argument('--min_inliers', type=int,   default=5,
                   help='Drop scenes with fewer GT inliers than this (default: 5)')
    args = p.parse_args()

    train_sources = [
        os.path.join(args.data_dir, 'replica_gs_train'),
        os.path.join(args.data_dir, '7scenes_gs_train'),
        os.path.join(args.data_dir, 'se3real_sim3_train'),
    ]
    valid_sources = [
        os.path.join(args.data_dir, 'replica_gs_valid'),
        os.path.join(args.data_dir, '7scenes_gs_valid'),
        os.path.join(args.data_dir, 'se3real_sim3_valid'),
    ]

    print('Combining TRAIN splits:')
    n_train = combine_and_save(train_sources, os.path.join(args.out_dir, 'joint_train'),
                               args.seed, args.min_scale, args.min_inliers)

    print('\nCombining VALID splits:')
    n_valid = combine_and_save(valid_sources, os.path.join(args.out_dir, 'joint_valid'),
                               args.seed + 1, args.min_scale, args.min_inliers)

    print(f'\nDone — {n_train:,} train / {n_valid:,} valid scenes.')

if __name__ == '__main__':
    main()
