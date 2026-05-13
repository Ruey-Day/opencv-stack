#!/usr/bin/env python3
"""
combine_joint_dataset.py

Merges replica_gs, 7scenes_gs, and tum_rgbd_gs splits into a single
joint_train / joint_valid dataset.

All sources are 9D (Plücker+LAB) with embedded overlap distribution —
no padding or conversion needed, just shuffle and concatenate.

Usage
-----
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
    n = len(data['plucker1'])
    print(f'  Loaded {n:,} scenes from {split_dir}')
    return data, n


def combine_and_save(source_dirs, out_dir, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    combined = {k: [] for k in KEYS}
    total    = 0

    for sd in source_dirs:
        if not os.path.exists(sd):
            print(f'  [SKIP] {sd} — not found yet')
            continue
        data, n = load_split(sd)
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
    p.add_argument('--data_dir', default='./dataset')
    p.add_argument('--out_dir',  default='./dataset')
    p.add_argument('--seed',     type=int, default=42)
    args = p.parse_args()

    train_sources = [
        os.path.join(args.data_dir, 'replica_gs_train'),
        os.path.join(args.data_dir, '7scenes_gs_train'),
        os.path.join(args.data_dir, 'tum_rgbd_gs_train'),
    ]
    valid_sources = [
        os.path.join(args.data_dir, 'replica_gs_valid'),
        os.path.join(args.data_dir, '7scenes_gs_valid'),
        os.path.join(args.data_dir, 'tum_rgbd_gs_valid'),
    ]

    print('Combining TRAIN splits:')
    n_train = combine_and_save(train_sources, os.path.join(args.out_dir, 'joint_train'), args.seed)

    print('\nCombining VALID splits:')
    n_valid = combine_and_save(valid_sources, os.path.join(args.out_dir, 'joint_valid'), args.seed + 1)

    print(f'\nDone — {n_train:,} train / {n_valid:,} valid scenes.')
    print('Train with: python train.py --dataset joint --dustbin')


if __name__ == '__main__':
    main()
