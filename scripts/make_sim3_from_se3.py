#!/usr/bin/env python3
"""
Convert SE3 pkl dataset to Sim3 by applying a random scale to each sample.

Given a SE3 pair (plucker1, plucker2, R_gt, t_gt, s_gt=1), produces a valid
Sim3 pair by rescaling plucker1's moments using R_gt and t_gt:

    m1_sim3 = s * (m1 - t×d1) + t×d1

This is exact: if m1_se3 = R*m2 + t×d1, then m1_sim3 = s*(R*m2) + t×d1.
Directions are unchanged (scale does not affect them).
The scale s_gt is added as a new pkl file.

Usage:
    python scripts/make_sim3_from_se3.py --src dataset/structured3D_valid \
                                          --dst dataset/structured3D_sim3_valid
    python scripts/make_sim3_from_se3.py --src dataset/semantic3D_valid \
                                          --dst dataset/semantic3D_sim3_valid
    python scripts/make_sim3_from_se3.py --src dataset/structured3D_train \
                                          --dst dataset/structured3D_sim3_train
    python scripts/make_sim3_from_se3.py --src dataset/semantic3D_train \
                                          --dst dataset/semantic3D_sim3_train
"""

import os
import sys
import pickle
import argparse
import numpy as np
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

SCALE_MIN = 0.1
SCALE_MAX = 10.0


def apply_sim3_scale(plucker1: np.ndarray, t: np.ndarray, s: float) -> np.ndarray:
    """
    Apply Sim3 scale to plucker1 moments given translation t and scale s.

    m1_sim3 = s * (m1 - t×d1) + t×d1

    plucker1: (N, 6) in [m, d] order
    t:        (3,)   translation from the SE3 transform
    """
    out = plucker1.copy()
    m1 = plucker1[:, :3]   # moments
    d1 = plucker1[:, 3:]   # directions (unchanged)
    t_cross_d1 = np.cross(t[None], d1)   # (N, 3)
    out[:, :3] = s * (m1 - t_cross_d1) + t_cross_d1
    return out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src',  required=True, help='Source SE3 dataset directory')
    ap.add_argument('--dst',  required=True, help='Output Sim3 dataset directory')
    ap.add_argument('--scale_min', type=float, default=SCALE_MIN)
    ap.add_argument('--scale_max', type=float, default=SCALE_MAX)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.dst, exist_ok=True)

    def load(name):
        return pickle.load(open(os.path.join(args.src, f'{name}.pkl'), 'rb'),
                           encoding='latin1')

    matches  = load('matches')
    plucker1 = load('plucker1')
    plucker2 = load('plucker2')
    R_gt     = load('R_gt')
    t_gt     = load('t_gt')
    n        = len(t_gt)

    # Generate random scales log-uniformly
    log_s = np.random.uniform(np.log(args.scale_min), np.log(args.scale_max), n)
    scales = np.exp(log_s).astype(np.float32)

    new_p1  = []
    new_s   = []
    for i in range(n):
        t = t_gt[i].flatten().astype(np.float64)
        s = float(scales[i])
        new_p1.append(apply_sim3_scale(plucker1[i].astype(np.float64), t, s).astype(np.float32))
        new_s.append(np.float32(s))

    def save(name, obj):
        with open(os.path.join(args.dst, f'{name}.pkl'), 'wb') as f:
            pickle.dump(obj, f, protocol=4)

    save('matches',  matches)
    save('plucker1', new_p1)
    save('plucker2', plucker2)
    save('R_gt',     R_gt)
    save('t_gt',     t_gt)
    save('s_gt',     new_s)

    s_arr = np.array(new_s)
    print(f'Saved {n} Sim3 pairs to {args.dst}')
    print(f'Scale range: [{s_arr.min():.3f}, {s_arr.max():.3f}]  mean={s_arr.mean():.3f}')


if __name__ == '__main__':
    main()
