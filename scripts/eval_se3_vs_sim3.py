#!/usr/bin/env python3
"""
Evaluate PlueckerNet pretrained weights on SE3 vs Sim3 validation splits.

Shows that the original model (trained on SE3 data) fails when the same
scene geometry is presented with a random scale factor applied.

Usage:
    python scripts/eval_se3_vs_sim3.py
    python scripts/eval_se3_vs_sim3.py --ckpt_semantic path/to/checkpoint.pth
"""

import os
import sys
import pickle
import argparse
import numpy as np
import torch
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'PlueckerNet'))

from lib.utils import load_model


CKPT_SEMANTIC  = str(_ROOT / 'PlueckerNet/output/semantic3D/preTrained/best_val_checkpoint.pth')
CKPT_STRUCTURED = str(_ROOT / 'PlueckerNet/output/structured3D/preTrained/best_val_checkpoint.pth')
DATA_DIR        = str(_ROOT / 'dataset')

SPLITS = [
    # (dataset_name,       ckpt_key,    label)
    ('semantic3D',         'semantic',  'semantic3D  SE3  (s=1)'),
    ('semantic3D_sim3',    'semantic',  'semantic3D  Sim3 (random s)'),
    ('structured3D',       'structured','structured3D SE3  (s=1)'),
    ('structured3D_sim3',  'structured','structured3D Sim3 (random s)'),
]


def _load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def load_split(dataset_name, split='valid'):
    folder = os.path.join(DATA_DIR, f'{dataset_name}_{split}')
    data = {}
    for key in ('matches', 'plucker1', 'plucker2', 'R_gt', 't_gt'):
        data[key] = _load_pkl(os.path.join(folder, f'{key}.pkl'))
    s_path = os.path.join(folder, 's_gt.pkl')
    if os.path.exists(s_path):
        data['s_gt'] = _load_pkl(s_path)
    else:
        data['s_gt'] = [np.float32(1.0)] * len(data['t_gt'])
    n = len(data['t_gt'])
    print(f'  loaded {split}: {n} samples from {folder}')
    return data, n


def eval_split(model, data, n, device, max_samples=-1):
    model.eval()
    inlier_ratios = []

    if max_samples > 0:
        n = min(n, max_samples)

    with torch.no_grad():
        for i in range(n):
            p1 = torch.from_numpy(data['plucker1'][i].astype(np.float32)).unsqueeze(0).to(device)
            p2 = torch.from_numpy(data['plucker2'][i].astype(np.float32)).unsqueeze(0).to(device)

            matches_ind = data['matches'][i]   # (2, n_inliers)
            n1, n2 = p1.shape[1], p2.shape[1]

            matches_gt = np.zeros((n1, n2), dtype=np.float32)
            if matches_ind.shape[1] > 0:
                matches_gt[matches_ind[0], matches_ind[1]] = 1.0

            prob_matrix, _, _ = model(p1, p2)   # (1, n1, n2)

            k = min(100, n1 * n2)
            if k <= 3:
                inlier_ratios.append(0.0)
                continue

            flat    = prob_matrix.flatten(start_dim=-2)   # (1, n1*n2)
            _, topk = torch.topk(flat, k=k, dim=-1, largest=True)
            i1 = (topk // prob_matrix.size(-1)).cpu().numpy()[0]
            i2 = (topk  % prob_matrix.size(-1)).cpu().numpy()[0]

            hits = matches_gt[i1, i2].sum()
            inlier_ratios.append(hits / k * 100.0)

    avg = float(np.mean(inlier_ratios))
    med = float(np.median(inlier_ratios))
    return avg, med


def build_model(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg   = ckpt.get('config', None)

    Model = load_model('PluckerNetKnn')
    if cfg is not None:
        model = Model(cfg)
    else:
        from easydict import EasyDict
        cfg = EasyDict({'net_nchannel': 128, 'in_channel': 6,
                        'sinkhorn_mu': 0.1, 'sinkhorn_iterations': 30,
                        'max_num_points': 1500, 'k': 6})
        model = Model(cfg)

    state = ckpt.get('state_dict', ckpt.get('model', ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'  [warn] missing keys: {missing}')
    if unexpected:
        print(f'  [warn] unexpected keys: {unexpected}')
    model = model.to(device)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt_semantic',   default=CKPT_SEMANTIC)
    ap.add_argument('--ckpt_structured', default=CKPT_STRUCTURED)
    ap.add_argument('--split',           default='valid', choices=['train', 'valid'])
    ap.add_argument('--max_samples',     type=int, default=-1,
                    help='Cap samples per split for quick runs (-1 = all)')
    ap.add_argument('--gpu',             type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    ckpts = {
        'semantic':   args.ckpt_semantic,
        'structured': args.ckpt_structured,
    }
    models = {}
    for key, path in ckpts.items():
        if os.path.exists(path):
            print(f'\nLoading {key} checkpoint: {path}')
            models[key] = build_model(path, device)
        else:
            print(f'[MISSING] {path}')
            models[key] = None

    print()
    print(f'{"Dataset":<40s}  {"avg_ir":>8s}  {"med_ir":>8s}  {"n":>6s}')
    print('-' * 68)

    results = {}
    for dataset_name, ckpt_key, label in SPLITS:
        model = models.get(ckpt_key)
        if model is None:
            print(f'{label:<40s}  [SKIPPED — checkpoint missing]')
            continue
        data, n = load_split(dataset_name, args.split)
        avg_ir, med_ir = eval_split(model, data, n, device, args.max_samples)
        results[dataset_name] = avg_ir
        print(f'{label:<40s}  {avg_ir:7.2f}%  {med_ir:7.2f}%  {n:6d}')

    print()
    for base in ('semantic3D', 'structured3D'):
        se3 = results.get(base)
        sim3 = results.get(f'{base}_sim3')
        if se3 is not None and sim3 is not None:
            drop = se3 - sim3
            print(f'{base}: SE3={se3:.2f}%  Sim3={sim3:.2f}%  drop={drop:+.2f}pp')


if __name__ == '__main__':
    main()
