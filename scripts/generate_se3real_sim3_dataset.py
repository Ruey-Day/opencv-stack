"""
Combines the original PlueckerNet SE(3) datasets (semantic3D + structured3D,
already converted to [m,d] format by convert_se3_datasets.py) and applies
random Sim(3) scale augmentation.

Output:
  dataset/se3real_sim3_train/   (semantic3D_train + structured3D_train)
  dataset/se3real_sim3_valid/   (semantic3D_valid + structured3D_valid)
"""

import os
import sys
import pickle
import argparse
import numpy as np

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS     = ['matches', 'plucker1', 'plucker2', 'R_gt', 't_gt', 's_gt']
SOURCES  = ['semantic3D', 'structured3D']
SCALE_RANGE = (0.1, 10.0)
SE3_KEEP_PROB = 0.15   # fraction of scenes kept at s=1


def _load(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def _save(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=4)


def apply_scale_aug(plucker2, t_gt, rng):
    """Randomly scale plucker2 moments and t_gt; return (new_plucker2, new_t, s)."""
    if rng.random() < SE3_KEEP_PROB:
        return plucker2, t_gt, np.float32(1.0)
    log_s = rng.uniform(np.log(SCALE_RANGE[0]), np.log(SCALE_RANGE[1]))
    s = np.exp(log_s).astype(np.float32)
    p2 = plucker2.copy()
    p2[:, :3] *= s          # scale moments; directions unchanged
    t = (t_gt * s).astype(np.float32)
    return p2, t, s


def generate_split(sources_base, split, out_dir, seed=42, n_copies=1):
    rng = np.random.default_rng(seed)

    combined = {k: [] for k in KEYS}
    for src in SOURCES:
        folder = os.path.join(sources_base, f'{src}_{split}')
        if not os.path.isdir(folder):
            print(f'  [SKIP] {folder} not found — run convert_se3_datasets.py first')
            continue
        data = {}
        for k in KEYS:
            data[k] = _load(os.path.join(folder, f'{k}.pkl'))
        n = len(data['plucker1'])
        print(f'  {src}_{split}: {n} scenes × {n_copies} copies = {n * n_copies}')
        for _ in range(n_copies):
            for i in range(n):
                p2, t, s = apply_scale_aug(data['plucker2'][i], data['t_gt'][i], rng)
                combined['matches'].append(data['matches'][i])
                combined['plucker1'].append(data['plucker1'][i])
                combined['plucker2'].append(p2)
                combined['R_gt'].append(data['R_gt'][i])
                combined['t_gt'].append(t)
                combined['s_gt'].append(s)

    total = len(combined['plucker1'])
    if total == 0:
        print(f'  ERROR: no data loaded for {split}')
        return 0

    # Shuffle
    perm = rng.permutation(total)
    for k in KEYS:
        combined[k] = [combined[k][i] for i in perm]

    os.makedirs(out_dir, exist_ok=True)
    for k in KEYS:
        _save(os.path.join(out_dir, f'{k}.pkl'), combined[k])

    s_vals = np.array(combined['s_gt'])
    n_se3 = int((s_vals == 1.0).sum())
    print(f'  → {out_dir}')
    print(f'     {total} scenes  |  SE3(s=1): {n_se3} ({100*n_se3/total:.0f}%)  '
          f'|  Sim3: {total-n_se3} ({100*(total-n_se3)/total:.0f}%)')
    print(f'     scale range: [{s_vals.min():.3f}, {s_vals.max():.3f}]')
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',  default=os.path.join(ROOT, 'dataset'))
    p.add_argument('--seed',      type=int, default=42)
    p.add_argument('--n_copies',  type=int, default=3,
                   help='Scale-augmentation copies per scene (default: 3). '
                        'Each copy draws a fresh random scale, keeping the same line correspondences.')
    args = p.parse_args()

    print('Generating se3real_sim3 dataset ...')
    print(f'  Scale range     : {SCALE_RANGE}  (log-uniform)')
    print(f'  SE3 keep prob   : {SE3_KEEP_PROB*100:.0f}%')
    print(f'  Copies per scene: {args.n_copies}')
    print()

    generate_split(args.data_dir, 'train',
                   os.path.join(args.data_dir, 'se3real_sim3_train'),
                   seed=args.seed, n_copies=args.n_copies)
    print()
    generate_split(args.data_dir, 'valid',
                   os.path.join(args.data_dir, 'se3real_sim3_valid'),
                   seed=args.seed + 1, n_copies=1)  # keep valid at 1× — no augmentation inflation

    print('\nDone. Train with:')
    print('  python train.py --dataset se3real_sim3')


if __name__ == '__main__':
    main()
