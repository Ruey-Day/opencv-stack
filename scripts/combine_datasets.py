#!/usr/bin/env python3
"""
Merge multiple pre-generated pkl dataset splits into a single combined split.

Usage:
    python scripts/combine_datasets.py \
        --inputs 7scenes fastcamo \
        --output main \
        --splits train valid
"""
import os
import sys
import pickle
import argparse
from pathlib import Path

_ROOT = Path(__file__).parent.parent


def load_split(data_dir, name, split):
    folder = os.path.join(data_dir, f'{name}_{split}')
    keys = ['matches', 'plucker1', 'plucker2', 'R_gt', 't_gt', 's_gt']
    data = {}
    for k in keys:
        with open(os.path.join(folder, f'{k}.pkl'), 'rb') as f:
            data[k] = pickle.load(f, encoding='latin1')
    print(f'  {name}_{split}: {len(data["t_gt"])} samples')
    return data


def save_split(data, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for k, v in data.items():
        with open(os.path.join(out_dir, f'{k}.pkl'), 'wb') as f:
            pickle.dump(v, f, protocol=4)
    print(f'  → {out_dir}  ({len(data["t_gt"])} samples)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs',   nargs='+', required=True, help='Dataset names to merge')
    ap.add_argument('--output',   required=True,            help='Output dataset name')
    ap.add_argument('--splits',   nargs='+', default=['train', 'valid'])
    ap.add_argument('--data_dir', default=str(_ROOT / 'dataset'))
    args = ap.parse_args()

    for split in args.splits:
        print(f'\n--- {split} ---')
        combined = {k: [] for k in ['matches', 'plucker1', 'plucker2', 'R_gt', 't_gt', 's_gt']}
        for name in args.inputs:
            src = load_split(args.data_dir, name, split)
            for k in combined:
                combined[k].extend(src[k])
        out_dir = os.path.join(args.data_dir, f'{args.output}_{split}')
        save_split(combined, out_dir)

    print('\nDone.')


if __name__ == '__main__':
    main()
