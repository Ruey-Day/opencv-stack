#!/usr/bin/env python3
"""
Generate a large, fully-synthetic dataset for general Sim(3) line correspondence.

No map files are required — all pools come from random geometric primitives
(plane patches, wireframes, line bundles, parallel groups, grid patches) with
no axis-aligned or Manhattan-world bias.

Scenarios covered
-----------------
submap      (30%) — sparse/noisy monocular query vs large clean metric reference
relocalize  (25%) — cross-session, similar sizes, moderate overlap
loop        (20%) — loop-closure, high overlap, both sides similar
dense_sparse(15%) — very different line densities (RGB-D vs monocular)
zero_overlap(10%) — completely disjoint views (hard negatives)

Output
------
dataset/synthetic_train/   dataset/synthetic_valid/

Each split is the standard 6-pickle format:
    matches.pkl  plucker1.pkl  plucker2.pkl  R_gt.pkl  t_gt.pkl  s_gt.pkl

Line counts are variable per pair — NOT padded to any fixed size.
Use batch_size=1 (default) or variable_collate for batch > 1.

Usage
-----
python scripts/generate_synthetic_large.py
python scripts/generate_synthetic_large.py --n_train 500000 --n_valid 5000
python scripts/generate_synthetic_large.py --n_train 200000 --workers 8 --seed 123
"""
import os
import sys
import pickle
import argparse
import time
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


# ── Worker function (runs in child process) ───────────────────────────────────

def _worker(args):
    """Generate n pairs starting from a given random seed."""
    n, seed = args
    # Re-seed each worker independently for reproducibility
    np.random.seed(seed)

    # Import here so the child process gets a fresh module state
    from lib.pair_generator import generate_diverse_pair

    keys = ['matches', 'plucker1', 'plucker2', 'R_gt', 't_gt', 's_gt']
    chunk = {k: [] for k in keys}
    for _ in range(n):
        pair = generate_diverse_pair()
        for k in keys:
            chunk[k].append(pair[k])
    return chunk


# ── Dataset saving ────────────────────────────────────────────────────────────

def _save(data: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for k, v in data.items():
        with open(os.path.join(out_dir, f'{k}.pkl'), 'wb') as f:
            pickle.dump(v, f, protocol=4)
    s = np.array(data['s_gt'])
    n1 = np.array([p.shape[0] for p in data['plucker1']])
    n2 = np.array([p.shape[0] for p in data['plucker2']])
    k  = np.array([m.shape[1] for m in data['matches']])
    print(f'  Saved {len(data["t_gt"])} pairs → {out_dir}')
    print(f'    scale : [{s.min():.3f}, {s.max():.3f}]  median {np.median(s):.3f}')
    print(f'    n1    : [{n1.min()}, {n1.max()}]  median {int(np.median(n1))}')
    print(f'    n2    : [{n2.min()}, {n2.max()}]  median {int(np.median(n2))}')
    print(f'    inliers/pair: [{k.min()}, {k.max()}]  median {int(np.median(k))}  '
          f'zero-overlap: {(k == 0).sum()}')


# ── Main generation loop ──────────────────────────────────────────────────────

def generate_split(n_pairs: int, workers: int, chunk_size: int,
                   seed: int, label: str) -> dict:
    n_chunks  = max(1, (n_pairs + chunk_size - 1) // chunk_size)
    tasks     = []
    remaining = n_pairs
    for i in range(n_chunks):
        n = min(chunk_size, remaining)
        tasks.append((n, seed + i * 7919))   # prime stride avoids seed collisions
        remaining -= n
        if remaining <= 0:
            break

    keys = ['matches', 'plucker1', 'plucker2', 'R_gt', 't_gt', 's_gt']
    combined = {k: [] for k in keys}

    t0 = time.time()
    done = 0

    if workers == 1:
        # Single-threaded path (easier to debug)
        for i, task in enumerate(tasks):
            chunk = _worker(task)
            for k in keys:
                combined[k].extend(chunk[k])
            done += task[0]
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f'  {label}: {done}/{n_pairs}  ({rate:.0f} pairs/s)', end='\r')
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for chunk in ex.map(_worker, tasks):
                for k in keys:
                    combined[k].extend(chunk[k])
                done += chunk_size
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(f'  {label}: {min(done, n_pairs)}/{n_pairs}  '
                      f'({rate:.0f} pairs/s)', end='\r')

    print()
    return combined


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--n_train',    type=int, default=200_000,
                    help='Training pairs (default: 200 000)')
    ap.add_argument('--n_valid',    type=int, default=2_000,
                    help='Validation pairs (default: 2 000)')
    ap.add_argument('--out_dir',    default=str(_ROOT / 'dataset'),
                    help='Root dataset directory')
    ap.add_argument('--chunk_size', type=int, default=2_000,
                    help='Pairs per worker task (default: 2 000)')
    ap.add_argument('--workers',    type=int,
                    default=max(1, (os.cpu_count() or 4) - 1),
                    help='Parallel workers (default: cpu_count - 1)')
    ap.add_argument('--seed',       type=int, default=42)
    args = ap.parse_args()

    print(f'Generating synthetic dataset  '
          f'train={args.n_train:,}  valid={args.n_valid:,}  '
          f'workers={args.workers}  seed={args.seed}')
    print(f'Output: {args.out_dir}')

    print('\n=== Training split ===')
    train_data = generate_split(
        args.n_train, args.workers, args.chunk_size,
        seed=args.seed, label='train',
    )
    _save(train_data, os.path.join(args.out_dir, 'synthetic_train'))

    print('\n=== Validation split ===')
    valid_data = generate_split(
        args.n_valid, args.workers, args.chunk_size,
        seed=args.seed + 999_983, label='valid',   # large prime offset
    )
    _save(valid_data, os.path.join(args.out_dir, 'synthetic_valid'))

    print('\nDone.')


if __name__ == '__main__':
    main()
