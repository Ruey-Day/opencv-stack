#!/usr/bin/env python3
"""
Generate static submap registration datasets from Structure-PLP-SLAM .db map files.

Datasets: Replica and 7-Scenes only.

Outputs:
    dataset/replica_train/   dataset/replica_valid/
    dataset/7scenes_train/   dataset/7scenes_valid/

Val held-out scenes:
    Replica  — room1 (both runs)
    7-Scenes — heads + stairs

Pair types per map:
    60%  intra-map     — submap from pool A vs big map A
    20%  cross-session — submap from pool A vs big map B (same scene)
    20%  cross-scene   — submap from pool A vs big map from a different scene

Usage:
    python scripts/generate_submap_dataset.py
    python scripts/generate_submap_dataset.py --pairs_per_train 400 --pairs_per_val 80
"""

import os
import sys
import pickle
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from sim3.pair_generator import load_pool_from_db, generate_submap_pair, SUBMAP_N_MIN

SLAM_DIR = _ROOT.parent / 'Structure-PLP-SLAM'

# ── Map catalogue ──────────────────────────────────────────────────────────────
# (db_stem, scene_group)

REPLICA_TRAIN = [
    ('replica_office0_gt_map',      'replica_office'),
    ('replica_office1_gt_map',      'replica_office'),
    ('replica_office2_gt_map',      'replica_office'),
    ('replica_office3_gt_map',      'replica_office'),
    ('replica_office4_gt_map',      'replica_office'),
    ('replica_room0_map',           'replica_room0'),
    ('replica_room0_mono_slow',     'replica_room0'),
]

REPLICA_VAL = [
    ('replica_room1_map',           'replica_room1'),
    ('replica_room1_mono_map_new',  'replica_room1'),
]

SCENES7_TRAIN = [
    # chess — all except val
    ('7scenes_chess_seq01_map',      '7s_chess'),
    ('7scenes_chess_seq02_map',      '7s_chess'),
    ('7scenes_chess_seq03_map',      '7s_chess'),
    ('7scenes_chess_seq04_map',      '7s_chess'),
    ('7scenes_chess_seq05_map',      '7s_chess'),
    ('7scenes_chess_seq06_map',      '7s_chess'),
    # fire
    ('7scenes_fire_seq01_map',       '7s_fire'),
    ('7scenes_fire_seq02_map',       '7s_fire'),
    ('7scenes_fire_seq03_map',       '7s_fire'),
    ('7scenes_fire_seq04_map',       '7s_fire'),
    # heads — only seq01 (seq02 is val)
    ('7scenes_heads_seq01_map',      '7s_heads'),
    # office — all
    ('7scenes_office_seq01_map',     '7s_office'),
    ('7scenes_office_seq02_map',     '7s_office'),
    ('7scenes_office_seq03_map',     '7s_office'),
    ('7scenes_office_seq04_map',     '7s_office'),
    ('7scenes_office_seq05_map',     '7s_office'),
    ('7scenes_office_seq06_map',     '7s_office'),
    ('7scenes_office_seq07_map',     '7s_office'),
    ('7scenes_office_seq08_map',     '7s_office'),
    ('7scenes_office_seq09_map',     '7s_office'),
    ('7scenes_office_seq10_map',     '7s_office'),
    # pumpkin — all
    ('7scenes_pumpkin_seq01_map',    '7s_pumpkin'),
    ('7scenes_pumpkin_seq02_map',    '7s_pumpkin'),
    ('7scenes_pumpkin_seq03_map',    '7s_pumpkin'),
    ('7scenes_pumpkin_seq06_map',    '7s_pumpkin'),
    ('7scenes_pumpkin_seq07_map',    '7s_pumpkin'),
    ('7scenes_pumpkin_seq08_map',    '7s_pumpkin'),
    # redkitchen — all available
    ('7scenes_redkitchen_seq01_map', '7s_redkitchen'),
    ('7scenes_redkitchen_seq02_map', '7s_redkitchen'),
    ('7scenes_redkitchen_seq03_map', '7s_redkitchen'),
    ('7scenes_redkitchen_seq04_map', '7s_redkitchen'),
    ('7scenes_redkitchen_seq05_map', '7s_redkitchen'),
    ('7scenes_redkitchen_seq06_map', '7s_redkitchen'),
    ('7scenes_redkitchen_seq07_map', '7s_redkitchen'),
    ('7scenes_redkitchen_seq08_map', '7s_redkitchen'),
    ('7scenes_redkitchen_seq11_map', '7s_redkitchen'),
    ('7scenes_redkitchen_seq12_map', '7s_redkitchen'),
    ('7scenes_redkitchen_seq13_map', '7s_redkitchen'),
    ('7scenes_redkitchen_seq14_map', '7s_redkitchen'),
    # stairs — all except val (seq04)
    ('7scenes_stairs_seq01_map',     '7s_stairs'),
    ('7scenes_stairs_seq02_map',     '7s_stairs'),
    ('7scenes_stairs_seq03_map',     '7s_stairs'),
    ('7scenes_stairs_seq05_map',     '7s_stairs'),
    ('7scenes_stairs_seq06_map',     '7s_stairs'),
]

SCENES7_VAL = [
    ('7scenes_heads_seq02_map',  '7s_heads'),
    ('7scenes_stairs_seq04_map', '7s_stairs'),
]

FASTCAMO_TRAIN = [
    ('fastcamo_apartment_1_map', 'fc_apartment'),
    ('fastcamo_gym_map',         'fc_gym'),
    ('fastcamo_lounge_2_map',    'fc_lounge'),
    ('fastcamo_meeting_room_map','fc_meeting'),
    ('fastcamo_stairwell_map',   'fc_stairwell'),
    ('fastcamo_workshop_1_map',  'fc_workshop'),
]

FASTCAMO_VAL = [
    ('fastcamo_studio_map',   'fc_studio'),
    ('fastcamo_lounge_1_map', 'fc_lounge1'),
]


# ── Pool loader ────────────────────────────────────────────────────────────────

def _load_pools(map_list):
    result = []
    for stem, group in map_list:
        db = SLAM_DIR / f'{stem}.db'
        if not db.exists():
            db = SLAM_DIR / f'{stem.replace("_map", "")}.db'
        if not db.exists():
            print(f'  [MISSING] {stem}')
            continue
        pool = load_pool_from_db(str(db))
        if len(pool) < SUBMAP_N_MIN:
            print(f'  [SKIP   ] {stem}  ({len(pool)} lines)')
            continue
        print(f'  [OK     ] {stem}  ({len(pool)} lines)')
        result.append((stem, group, pool))
    return result


# ── Pair generation ────────────────────────────────────────────────────────────

def _generate(entries, n_per_map, seed, label):
    np.random.seed(seed)

    group_idx = defaultdict(list)
    for i, (_, group, _) in enumerate(entries):
        group_idx[group].append(i)

    keys  = ['matches', 'plucker1', 'plucker2', 'R_gt', 't_gt', 's_gt']
    pairs = {k: [] for k in keys}
    total = 0

    for i, (stem, group, pool) in enumerate(entries):
        n_intra   = int(round(n_per_map * 0.60))
        n_cross_s = int(round(n_per_map * 0.20))
        n_cross_e = n_per_map - n_intra - n_cross_s

        same  = [j for j in group_idx[group] if j != i]
        other = [j for j, (_, g, _) in enumerate(entries) if g != group]

        schedule = [
            (n_intra,   None),
            (n_cross_s, same  if same  else other),
            (n_cross_e, other if other else None),
        ]

        map_n = 0
        for n, ctx_list in schedule:
            done, attempts = 0, 0
            while done < n and attempts < n * 20:
                attempts += 1
                ctx = None
                if ctx_list:
                    ctx = entries[ctx_list[np.random.randint(len(ctx_list))]][2]
                pair = generate_submap_pair(pool, context_pool=ctx)
                if pair is None:
                    continue
                for k in keys:
                    pairs[k].append(pair[k])
                done += 1
                map_n += 1

        total += map_n
        print(f'  {stem:50s}  {map_n} pairs')

    print(f'\n  {label}: {total} pairs total')
    return pairs


def _save(pairs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for k, v in pairs.items():
        with open(os.path.join(out_dir, f'{k}.pkl'), 'wb') as f:
            pickle.dump(v, f, protocol=4)
    s = np.array(pairs['s_gt'])
    print(f'  Saved → {out_dir}  scale=[{s.min():.3f}, {s.max():.3f}]')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pairs_per_train', type=int, default=300)
    ap.add_argument('--pairs_per_val',   type=int, default=50)
    ap.add_argument('--out_dir', default=str(_ROOT / 'dataset'))
    ap.add_argument('--seed',    type=int, default=42)
    ap.add_argument('--dataset', choices=['replica', '7scenes', 'fastcamo', 'all'], default='all')
    args = ap.parse_args()

    do_replica  = args.dataset in ('replica',  'all')
    do_7scenes  = args.dataset in ('7scenes',  'all')
    do_fastcamo = args.dataset in ('fastcamo', 'all')

    if do_replica:
        print('\n=== Replica — loading pools ===')
        r_train = _load_pools(REPLICA_TRAIN)
        r_val   = _load_pools(REPLICA_VAL)
        print(f'\n=== Replica — generating train ({len(r_train)} maps) ===')
        _save(_generate(r_train, args.pairs_per_train, args.seed,     'replica_train'),
              os.path.join(args.out_dir, 'replica_train'))
        print(f'\n=== Replica — generating valid ({len(r_val)} maps) ===')
        _save(_generate(r_val,   args.pairs_per_val,   args.seed + 1, 'replica_valid'),
              os.path.join(args.out_dir, 'replica_valid'))

    if do_7scenes:
        print('\n=== 7-Scenes — loading pools ===')
        s_train = _load_pools(SCENES7_TRAIN)
        s_val   = _load_pools(SCENES7_VAL)
        print(f'\n=== 7-Scenes — generating train ({len(s_train)} maps) ===')
        _save(_generate(s_train, args.pairs_per_train, args.seed + 2, '7scenes_train'),
              os.path.join(args.out_dir, '7scenes_train'))
        print(f'\n=== 7-Scenes — generating valid ({len(s_val)} maps) ===')
        _save(_generate(s_val,   args.pairs_per_val,   args.seed + 3, '7scenes_valid'),
              os.path.join(args.out_dir, '7scenes_valid'))

    if do_fastcamo:
        print('\n=== FastCaMo — loading pools ===')
        fc_train = _load_pools(FASTCAMO_TRAIN)
        fc_val   = _load_pools(FASTCAMO_VAL)
        print(f'\n=== FastCaMo — generating train ({len(fc_train)} maps) ===')
        _save(_generate(fc_train, args.pairs_per_train, args.seed + 4, 'fastcamo_train'),
              os.path.join(args.out_dir, 'fastcamo_train'))
        print(f'\n=== FastCaMo — generating valid ({len(fc_val)} maps) ===')
        _save(_generate(fc_val,   args.pairs_per_val,   args.seed + 5, 'fastcamo_valid'),
              os.path.join(args.out_dir, 'fastcamo_valid'))

    print('\nDone.')


if __name__ == '__main__':
    main()
