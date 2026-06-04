"""
Sim3PluckerData        — static .pkl files (pre-generated offline)
LiveSim3PluckerData    — online pair generation from .db pools (training)

Expected .pkl layout:
    <data_dir>/<dataset>_train/
        matches.pkl     list of (2, n_inliers)  int32 arrays
        plucker1.pkl    list of (n_lines, 6)    float32 arrays
        plucker2.pkl    list of (n_lines, 6)    float32 arrays
        R_gt.pkl        list of (3, 3)           float32 arrays
        t_gt.pkl        list of (3, 1)           float32 arrays
        s_gt.pkl        list of float32 scalars
"""
import os
import sys
import pickle
import numpy as np
from torch.utils.data import Dataset
from sim3.pair_generator import (
    load_pool_from_db, generate_pair, generate_inter_map_pair,
    OVERLAP_PROBS, get_curriculum_probs,
)

def _load(path):
    with open(path, 'rb') as f:
        if sys.version_info[0] == 3:
            return pickle.load(f, encoding='latin1')
        return pickle.load(f)

def load_sim3_data(config, split):
    var_names = ['matches', 'plucker1', 'plucker2', 'R_gt', 't_gt', 's_gt']
    folder = os.path.join(config.data_dir, f'{config.dataset}_{split}')
    data = {}
    for var in var_names:
        data[var] = _load(os.path.join(folder, f'{var}.pkl'))
    print(f'[Sim3] loaded {split}: {len(data["t_gt"])} scenes from {folder}')
    return data


class Sim3PluckerData(Dataset):
    """Dataset for Sim(3) Plücker line matching.

    Returns per sample:
        matches  (n1, n2) float32 — binary correspondence matrix
        plucker1 (n1, 6)  float32
        plucker2 (n2, 6)  float32
        R_gt     (3, 3)   float32
        t_gt     (3, 1)   float32
        s_gt     ()       float32 scalar
    """

    def __init__(self, phase, config):
        super().__init__()
        self.data        = load_sim3_data(config, phase)
        self.len         = len(self.data['t_gt'])
        self.normalize_n    = getattr(config, 'normalize_n_lines',   None)
        self.normalize_n_in = getattr(config, 'normalize_n_inliers', None)
        self.in_channel     = getattr(config, 'in_channel', None)

    def __getitem__(self, index):
        matches_ind = self.data['matches'][index]     # (2, n_inliers)
        plucker1    = self.data['plucker1'][index]    # (n_lines, 6)
        plucker2    = self.data['plucker2'][index]
        R_gt        = self.data['R_gt'][index]
        t_gt        = self.data['t_gt'][index]
        s_gt        = np.float32(self.data['s_gt'][index])

        # Subsample to a fixed size when mixing datasets of different sizes.
        if self.normalize_n is not None and plucker1.shape[0] > self.normalize_n:
            n_in_want  = self.normalize_n_in or (self.normalize_n * 100 // 130)
            n_out_want = self.normalize_n - n_in_want
            n_in_have  = matches_ind.shape[1]

            in_sel  = np.random.choice(n_in_have, min(n_in_want, n_in_have), replace=False)
            in_idx1 = matches_ind[0, in_sel]
            in_idx2 = matches_ind[1, in_sel]

            out_pool1 = np.setdiff1d(np.arange(plucker1.shape[0]), in_idx1)
            out_pool2 = np.setdiff1d(np.arange(plucker2.shape[0]), in_idx2)
            out_sel1  = np.random.choice(out_pool1, min(n_out_want, len(out_pool1)), replace=False)
            out_sel2  = np.random.choice(out_pool2, min(n_out_want, len(out_pool2)), replace=False)

            n_in_actual = len(in_sel)
            new1  = np.concatenate([plucker1[in_idx1], plucker1[out_sel1]])
            new2  = np.concatenate([plucker2[in_idx2], plucker2[out_sel2]])
            perm1 = np.random.permutation(len(new1))
            perm2 = np.random.permutation(len(new2))
            new1  = new1[perm1];  new2  = new2[perm2]
            inv1  = np.argsort(perm1); inv2 = np.argsort(perm2)
            matches_ind = np.stack([inv1[:n_in_actual], inv2[:n_in_actual]], axis=0)
            plucker1, plucker2 = new1, new2

        if self.in_channel is not None:
            plucker1 = plucker1[:, :self.in_channel]
            plucker2 = plucker2[:, :self.in_channel]

        n1, n2 = plucker1.shape[0], plucker2.shape[0]
        matches = np.zeros([n1, n2], dtype=np.float32)
        matches[matches_ind[0, :], matches_ind[1, :]] = 1.0

        return (
            matches.astype('float32'),
            plucker1.astype('float32'),
            plucker2.astype('float32'),
            R_gt.astype('float32'),
            t_gt.astype('float32'),
            s_gt,
        )

    def __len__(self):
        return self.len


class LiveSim3PluckerData(Dataset):
    """Online dataset: generates a fresh pair from DB pools on every __getitem__.

    The model never sees the same (SIM3, subset, noise) combination twice,
    eliminating memorisation of fixed training pairs. DB pools are loaded once
    at startup; generation is CPU-only and fast (~0.5 ms per pair).

    Args:
        db_paths:        list of .db map files to load pools from
        epoch_size:      number of pairs per epoch (defines __len__)
        config:          training config (in_channel)
        inter_map_ratio: fraction of pairs generated as cross-map (default 0.3)
    """

    def __init__(self, db_paths, epoch_size, config, inter_map_ratio=0.3):
        super().__init__()
        self.pools = []
        for db in db_paths:
            pool = load_pool_from_db(db)
            if len(pool) >= 6:
                self.pools.append(pool)
        if not self.pools:
            raise RuntimeError(f"No valid pools found in {db_paths}")
        print(f"[LiveSim3] loaded {len(self.pools)} pools, epoch_size={epoch_size}, "
              f"inter_map_ratio={inter_map_ratio}")

        self.epoch_size      = epoch_size
        self.inter_map_ratio = inter_map_ratio if len(self.pools) >= 2 else 0.0
        self.in_channel      = getattr(config, 'in_channel', None)
        self.overlap_probs   = OVERLAP_PROBS.copy()

    def set_curriculum_phase(self, phase_frac: float):
        """Update overlap distribution for curriculum training. Call once per epoch."""
        self.overlap_probs = get_curriculum_probs(phase_frac)

    def __getitem__(self, index):
        a_idx  = index % len(self.pools)
        pool_a = self.pools[a_idx]

        pair = None
        while pair is None:
            if self.inter_map_ratio > 0 and np.random.random() < self.inter_map_ratio:
                b_idx = np.random.choice([i for i in range(len(self.pools)) if i != a_idx])
                pair  = generate_inter_map_pair(pool_a, self.pools[b_idx],
                                                overlap_probs=self.overlap_probs)
            else:
                pair = generate_pair(pool_a, overlap_probs=self.overlap_probs)

        plucker1 = pair['plucker1']
        plucker2 = pair['plucker2']
        matches_ind = pair['matches']

        if self.in_channel is not None:
            plucker1 = plucker1[:, :self.in_channel]
            plucker2 = plucker2[:, :self.in_channel]

        n1, n2 = plucker1.shape[0], plucker2.shape[0]
        matches = np.zeros([n1, n2], dtype=np.float32)
        if matches_ind.shape[1] > 0:
            matches[matches_ind[0], matches_ind[1]] = 1.0

        return (
            matches.astype('float32'),
            plucker1.astype('float32'),
            plucker2.astype('float32'),
            pair['R_gt'].astype('float32'),
            pair['t_gt'].astype('float32'),
            np.float32(pair['s_gt']),
        )

    def __len__(self):
        return self.epoch_size
