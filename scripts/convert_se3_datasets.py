"""
Converts the original PlueckerNet SE(3) datasets (semantic3D, structured3D)
from [d, m] Plücker format to our [m, d] format and adds s_gt=1.0.

Input:  ../PlueckerNet/dataset/{semantic3D,structured3D}_{train,valid}/
Output: ./dataset/{semantic3D,structured3D}_{train,valid}/
"""

import os
import sys
import pickle
import argparse
import numpy as np

ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLUECKERNET  = os.path.abspath(os.path.join(ROOT, 'PlueckerNet'))
SE3_DATASETS = ['semantic3D', 'structured3D']
SPLITS       = ['train', 'valid']
SE3_KEYS     = ['matches', 'plucker1', 'plucker2', 'R_gt', 't_gt']

def _load(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')

def _save(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def dm_to_md(arr):
    return np.concatenate([arr[:, 3:], arr[:, :3]], axis=1).astype(np.float32)

def convert_split(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    data = {}
    for k in SE3_KEYS:
        p = os.path.join(src_dir, f'{k}.pkl')
        if not os.path.exists(p):
            print(f'  [SKIP] {p} not found')
            return 0
        data[k] = _load(p)

    n = len(data['plucker1'])
    print(f'  {n} scenes from {src_dir}')

    plucker1_md = [dm_to_md(p) for p in data['plucker1']]
    plucker2_md = [dm_to_md(p) for p in data['plucker2']]
    s_gt = [np.float32(1.0)] * n

    _save(os.path.join(dst_dir, 'matches.pkl'),  data['matches'])
    _save(os.path.join(dst_dir, 'plucker1.pkl'), plucker1_md)
    _save(os.path.join(dst_dir, 'plucker2.pkl'), plucker2_md)
    _save(os.path.join(dst_dir, 'R_gt.pkl'),     data['R_gt'])
    _save(os.path.join(dst_dir, 't_gt.pkl'),     data['t_gt'])
    _save(os.path.join(dst_dir, 's_gt.pkl'),     s_gt)

    print(f'  → {dst_dir}  ({n} scenes, s_gt=1.0, 6D [m,d])')
    return n

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plueckernet_dir', default=PLUECKERNET,
                   help='Path to parent PlueckerNet repo')
    p.add_argument('--data_dir', default=os.path.join(ROOT, 'dataset'),
                   help='Output dataset directory')
    args = p.parse_args()

    src_base = os.path.join(args.plueckernet_dir, 'dataset')
    if not os.path.isdir(src_base):
        print(f'ERROR: PlueckerNet dataset dir not found: {src_base}')
        sys.exit(1)

    total = 0
    for ds in SE3_DATASETS:
        for split in SPLITS:
            src = os.path.join(src_base, f'{ds}_{split}')
            dst = os.path.join(args.data_dir, f'{ds}_{split}')
            print(f'\nConverting {ds}_{split}:')
            total += convert_split(src, dst)

    print(f'\nDone — {total} scenes converted.')

if __name__ == '__main__':
    main()
