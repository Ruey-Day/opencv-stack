"""
python scripts/gen_val.py \\
    --db dataset/maps/replica_room1_mono_map_new.db \\
         dataset/maps/7scenes_chess_seq06_map.db \\
         dataset/maps/7scenes_heads_seq02_map.db \\
         dataset/maps/7scenes_stairs_seq06_map.db

# More pairs, custom name:
python scripts/gen_val.py --db dataset/maps/*.db --n 800 --name slam_map
"""
import os
import sys
import glob
import pickle
import argparse
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from sim3.pair_generator import load_pool_from_db, generate_pair

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db', nargs='+', required=True,
                   help='.db map files to generate validation pairs')
    p.add_argument('--n', type=int, default=800,
                   help='Number of validation pairs (default: 800)')
    p.add_argument('--name', default='slam_map',
                   help='Split name — output goes to dataset/<name>_valid/ (default: slam_map)')
    p.add_argument('--out_dir', default=os.path.join(_ROOT, 'dataset'),
                   help='Root dataset directory (default: ./dataset)')
    p.add_argument('--seed', type=int, default=42,
                   help='RNG seed for reproducibility (default: 42)')
    args = p.parse_args()
    
    db_paths = []
    for pat in args.db:
        hits = sorted(glob.glob(pat))
        db_paths.extend(hits if hits else [pat])

    out_dir = os.path.join(args.out_dir, f'{args.name}_valid')
    os.makedirs(out_dir, exist_ok=True)

    print(f'\nGenerating {args.name}_valid ({args.n} pairs, seed={args.seed})')
    print(f'Output: {out_dir}\n')
    
    pools = []
    for db in db_paths:
        pool = load_pool_from_db(db)
        status = 'ok' if len(pool) >= 6 else 'SKIP'
        print(f'  {os.path.basename(db):55s}  {len(pool):4d} lines  [{status}]')
        if len(pool) >= 6:
            pools.append(pool)

    if not pools:
        sys.exit('[ERROR] No valid pools found.')
    
    np.random.seed(args.seed)
    keys  = ['matches', 'plucker1', 'plucker2', 'R_gt', 't_gt', 's_gt']
    pairs = {k: [] for k in keys}
    n_ok, attempts = 0, 0

    while n_ok < args.n and attempts < args.n * 10:
        attempts += 1
        pair = generate_pair(pools[n_ok % len(pools)])
        if pair is None:
            continue
        for k in keys:
            pairs[k].append(pair[k])
        n_ok += 1
    
    for k, v in pairs.items():
        with open(os.path.join(out_dir, f'{k}.pkl'), 'wb') as f:
            pickle.dump(v, f, protocol=4)

    s_vals = np.array(pairs['s_gt'])
    n_zero = int((s_vals == 0).sum())
    print(f'\n  {n_ok} pairs saved to {out_dir}')
    print(f'  zero-overlap: {n_zero} ({100*n_zero/n_ok:.0f}%)')
    if (s_vals > 0).any():
        print(f'  scale range : [{s_vals[s_vals > 0].min():.3f}, {s_vals.max():.3f}]')

if __name__ == '__main__':
    main()
