#!/usr/bin/env python3
"""
Generate synthetic SE3 and SE3→Sim3 datasets from random line pools.

Outputs:
    dataset/se3_train/       dataset/se3_valid/       (s_gt = 1.0 always)
    dataset/se3_sim3_train/  dataset/se3_sim3_valid/  (s_gt random in [0.1, 10])

Usage:
    python scripts/generate_synthetic_dataset.py
    python scripts/generate_synthetic_dataset.py --n_train 5000 --n_valid 500
"""

import os
import sys
import pickle
import argparse
import numpy as np
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from sim3.pair_generator import generate_pair, make_structured_pool

POOL_SIZE_MIN = 50
POOL_SIZE_MAX = 500


def _random_pool(rng: np.random.Generator) -> np.ndarray:
    n = int(rng.integers(POOL_SIZE_MIN, POOL_SIZE_MAX + 1))
    return make_structured_pool(n)


def _generate_split(n: int, force_scale, seed: int, label: str) -> dict:
    rng  = np.random.default_rng(seed)
    np.random.seed(seed)

    keys  = ['matches', 'plucker1', 'plucker2', 'R_gt', 't_gt', 's_gt']
    pairs = {k: [] for k in keys}
    ok, attempts = 0, 0

    while ok < n and attempts < n * 20:
        attempts += 1
        pool = _random_pool(rng)
        pair = generate_pair(pool, force_scale=force_scale)
        if pair is None:
            continue
        for k in keys:
            pairs[k].append(pair[k])
        ok += 1

    s = np.array(pairs['s_gt'])
    print(f'  {label}: {ok} pairs, scale=[{s.min():.3f}, {s.max():.3f}]')
    return pairs


def _save(pairs: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for k, v in pairs.items():
        with open(os.path.join(out_dir, f'{k}.pkl'), 'wb') as f:
            pickle.dump(v, f, protocol=4)
    print(f'  Saved → {out_dir}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n_train', type=int, default=5000)
    ap.add_argument('--n_valid', type=int, default=500)
    ap.add_argument('--out_dir', default=str(_ROOT / 'dataset'))
    ap.add_argument('--seed',    type=int, default=42)
    args = ap.parse_args()

    out = args.out_dir

    print('=== SE3 (s=1) ===')
    _save(_generate_split(args.n_train, force_scale=1.0, seed=args.seed,
                          label='se3_train'),
          os.path.join(out, 'se3_train'))
    _save(_generate_split(args.n_valid, force_scale=1.0, seed=args.seed + 1,
                          label='se3_valid'),
          os.path.join(out, 'se3_valid'))

    print('\n=== SE3→Sim3 (random scale) ===')
    _save(_generate_split(args.n_train, force_scale=None, seed=args.seed + 2,
                          label='se3_sim3_train'),
          os.path.join(out, 'se3_sim3_train'))
    _save(_generate_split(args.n_valid, force_scale=None, seed=args.seed + 3,
                          label='se3_sim3_valid'),
          os.path.join(out, 'se3_sim3_valid'))

    print('\nDone.')


if __name__ == '__main__':
    main()
