#!/usr/bin/env python3
"""
generate_slam_map_dataset.py
============================
Build ScalePluckerNet training data from Structure-PLP-SLAM RGB-D map files.

Pair structure:
  plucker1 = mono SLAM side     (N_P1=200)  sparse + noisy + arbitrary scale/pose
  plucker2 = RGB-D SLAM side    (N_P2=200)  metric reference, padded with outliers

Usage:
  python ScalePluckerNet/scripts/generate_slam_map_dataset.py \\
      --db_train replica_room0_map.db Structure-PLP-SLAM/replica_room1_map.db \\
      --db_valid Structure-PLP-SLAM/replica_room1_mono_map_new.db \\
      --out_dir  ScalePluckerNet/dataset \\
      --dataset  slam_map \\
      --n_pairs_train 16000 \\
      --n_pairs_valid 800 \\
      --inter_map_ratio 0.3
"""

import os
import sys
import argparse
import pickle
import glob
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, ".."))
from sim3.pair_generator import (
    N_P1_TOTAL, N_P2_TOTAL, SCALE_RANGE,
    SLAM_NOISE_MIN, SLAM_NOISE_MAX, SLAM_RATIO_MIN, SLAM_RATIO_MAX,
    load_pool_from_db, generate_pair, generate_inter_map_pair,
)


def build_split(db_paths: list[str], out_dir: str, n_pairs: int, seed: int,
                inter_map_ratio: float = 0.0) -> int:
    """
    Generate `n_pairs` samples from the given .db files and save as pkl.
    inter_map_ratio: fraction of train pairs that are cross-map. Keep 0.0 for val.
    """
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    all_pools = []
    for db in db_paths:
        pool = load_pool_from_db(db)
        print(f"  {os.path.basename(db):50s}  {len(pool):4d} lines")
        if len(pool) >= 6:
            all_pools.append(pool)

    if not all_pools:
        print("[error] no valid pools")
        return 0

    can_inter = len(all_pools) >= 2 and inter_map_ratio > 0.0
    print(f"  {len(all_pools)} scenes, generating {n_pairs} pairs "
          f"({int(inter_map_ratio*100)}% inter-map) …")

    keys  = ["matches", "plucker1", "plucker2", "R_gt", "t_gt", "s_gt"]
    pairs = {k: [] for k in keys}
    n_ok, attempts = 0, 0

    while n_ok < n_pairs and attempts < n_pairs * 8:
        attempts += 1
        a_idx  = n_ok % len(all_pools)
        pool_a = all_pools[a_idx]

        if can_inter and np.random.random() < inter_map_ratio:
            b_idx = np.random.choice([i for i in range(len(all_pools)) if i != a_idx])
            pair  = generate_inter_map_pair(pool_a, all_pools[b_idx])
        else:
            pair = generate_pair(pool_a)

        if pair is None:
            continue
        for k in keys:
            pairs[k].append(pair[k])
        n_ok += 1

    for k, v in pairs.items():
        with open(os.path.join(out_dir, f"{k}.pkl"), "wb") as f:
            pickle.dump(v, f, protocol=4)

    print(f"  → {n_ok} pairs saved to {out_dir}")
    return n_ok


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db_train", nargs="+")
    p.add_argument("--db_valid", nargs="+")
    p.add_argument("--db_dir",  default=None)
    p.add_argument("--db_glob", default="replica_*_map.db")
    p.add_argument("--out_dir",  default="./dataset")
    p.add_argument("--dataset",  default="slam_map")
    p.add_argument("--n_pairs_train", type=int, default=4000)
    p.add_argument("--n_pairs_valid", type=int, default=600)
    p.add_argument("--inter_map_ratio", type=float, default=0.3)
    p.add_argument("--seed",     type=int, default=42)
    args = p.parse_args()

    db_train = args.db_train or []
    db_valid = args.db_valid or []

    if not db_train and args.db_dir:
        all_dbs = sorted(glob.glob(os.path.join(args.db_dir, args.db_glob)))
        if not all_dbs:
            raise FileNotFoundError(f"No .db files matching {args.db_glob} in {args.db_dir}")
        db_train, db_valid = all_dbs[:-1], all_dbs[-1:]

    if not db_train:
        p.error("Provide --db_train or --db_dir")
    if not db_valid:
        db_valid = db_train[-1:]

    out_base = os.path.abspath(args.out_dir)
    print(f"\n{'='*60}")
    print(f"Scale range: {SCALE_RANGE}  Noise σ: {SLAM_NOISE_MIN}–{SLAM_NOISE_MAX} m  "
          f"N={N_P1_TOTAL}×{N_P2_TOTAL}")
    print(f"{'='*60}\n")

    print("── TRAIN ──")
    build_split(db_train, os.path.join(out_base, f"{args.dataset}_train"),
                args.n_pairs_train, args.seed, inter_map_ratio=args.inter_map_ratio)

    print("\n── VALID ──")
    build_split(db_valid, os.path.join(out_base, f"{args.dataset}_valid"),
                args.n_pairs_valid, args.seed + 99999, inter_map_ratio=0.0)

    print("\nDone.")


if __name__ == "__main__":
    main()
