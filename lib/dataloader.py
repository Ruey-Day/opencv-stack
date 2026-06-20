"""
Sim3PluckerData        — static .pkl files (pre-generated offline)
LiveSim3PluckerData    — online pair generation from .db pools

Variable-length batching
    Pairs have different N lines per side depending on pool size and overlap.
    Use variable_collate as the DataLoader collate_fn (any batch size), or
    set batch_size=1 to use PyTorch's default collate with no changes.

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
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from lib.pair_generator import (
    load_pool_from_db,
    generate_pair,
    generate_inter_map_pair,
    generate_submap_pair,
    OVERLAP_PROBS,
    get_curriculum_probs,
)


# ── Variable-length collate ───────────────────────────────────────────────────

def variable_collate(batch):
    """
    Collate variable-N Plücker samples by zero-padding to the max N in the batch.

    Each item is (matches (N1,N2), plucker1 (N1,C), plucker2 (N2,C), R, t, s).
    Returns stacked tensors with shapes (B,maxN1,maxN2), (B,maxN1,C), (B,maxN2,C),
    (B,3,3), (B,3,1), (B,).

    Padded entries are zero.  The loss ignores padded rows/cols because the
    padded region of the matches matrix is also zero (no GT correspondences).
    """
    matches_l, p1_l, p2_l, R_l, t_l, s_l = zip(*batch)

    B   = len(p1_l)
    C   = p1_l[0].shape[1]
    mn1 = max(p.shape[0] for p in p1_l)
    mn2 = max(p.shape[0] for p in p2_l)

    p1_pad = torch.zeros(B, mn1, C)
    p2_pad = torch.zeros(B, mn2, C)
    m_pad  = torch.zeros(B, mn1, mn2)

    for i, (m, p1, p2) in enumerate(zip(matches_l, p1_l, p2_l)):
        n1, n2 = p1.shape[0], p2.shape[0]
        p1_pad[i, :n1] = torch.as_tensor(p1)
        p2_pad[i, :n2] = torch.as_tensor(p2)
        m_pad[i, :n1, :n2] = torch.as_tensor(m)

    return (
        m_pad,
        p1_pad,
        p2_pad,
        torch.stack([torch.as_tensor(r) for r in R_l]),
        torch.stack([torch.as_tensor(t) for t in t_l]),
        torch.tensor(s_l, dtype=torch.float32),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def load_sim3_data(config, split):
    var_names = ['matches', 'plucker1', 'plucker2', 'R_gt', 't_gt', 's_gt']
    folder = os.path.join(config.data_dir, f'{config.dataset}_{split}')
    data = {}
    for var in var_names:
        data[var] = _load(os.path.join(folder, f'{var}.pkl'))
    print(f'[Sim3] loaded {split}: {len(data["t_gt"])} scenes from {folder}')
    return data


# ── Static pkl dataset ────────────────────────────────────────────────────────

class Sim3PluckerData(Dataset):
    """Dataset for Sim(3) Plücker line matching from pre-generated .pkl files.

    Returns per sample:
        matches  (N1, N2) float32 — binary correspondence matrix
        plucker1 (N1, C)  float32
        plucker2 (N2, C)  float32
        R_gt     (3, 3)   float32
        t_gt     (3, 1)   float32
        s_gt     ()       float32 scalar

    N1 and N2 are variable across samples.  Use variable_collate or batch_size=1.
    """

    def __init__(self, phase, config):
        super().__init__()
        self.data       = load_sim3_data(config, phase)
        self.len        = len(self.data['t_gt'])
        self.in_channel = getattr(config, 'in_channel', None)

    def __getitem__(self, index):
        matches_ind = self.data['matches'][index]     # (2, n_inliers)
        plucker1    = self.data['plucker1'][index]    # (n_lines, 6)
        plucker2    = self.data['plucker2'][index]
        R_gt        = self.data['R_gt'][index]
        t_gt        = self.data['t_gt'][index]
        s_gt        = np.float32(self.data['s_gt'][index])

        if self.in_channel is not None:
            plucker1 = plucker1[:, :self.in_channel]
            plucker2 = plucker2[:, :self.in_channel]

        n1, n2 = plucker1.shape[0], plucker2.shape[0]
        matches = np.zeros([n1, n2], dtype=np.float32)
        if matches_ind.shape[1] > 0:
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


# ── Live dataset ──────────────────────────────────────────────────────────────

class LiveSim3PluckerData(Dataset):
    """Online dataset: generates fresh pairs from DB pools on every __getitem__.

    Args:
        db_paths:        list of .db map files
        epoch_size:      number of pairs per epoch
        config:          training config (in_channel)
        inter_map_ratio: fraction of pairs generated as cross-map (default 0.3)
        mode:            'symmetric' (default) — classic intra/inter-map pairs;
                         'submap'              — asymmetric big-map→small-submap pairs
    """

    def __init__(self, db_paths, epoch_size, config,
                 inter_map_ratio=0.3, mode='symmetric'):
        super().__init__()
        from lib.pair_generator import SUBMAP_N_MIN
        min_pool = SUBMAP_N_MIN if mode == 'submap' else 6
        self.pools = []
        for db in db_paths:
            pool = load_pool_from_db(db)
            if len(pool) >= min_pool:
                self.pools.append(pool)
        if not self.pools:
            raise RuntimeError(f"No valid pools found in {db_paths}")
        print(f"[LiveSim3] loaded {len(self.pools)} pools, epoch_size={epoch_size}, "
              f"inter_map_ratio={inter_map_ratio}, mode={mode}")

        self.epoch_size      = epoch_size
        self.inter_map_ratio = inter_map_ratio if len(self.pools) >= 2 else 0.0
        self.in_channel      = getattr(config, 'in_channel', None)
        self.overlap_probs   = OVERLAP_PROBS.copy()
        self.mode            = mode

    def set_curriculum_phase(self, phase_frac: float):
        self.overlap_probs = get_curriculum_probs(phase_frac)

    def __getitem__(self, index):
        a_idx  = index % len(self.pools)
        pool_a = self.pools[a_idx]

        pair = None
        _tries = 0
        while pair is None:
            if _tries > 0:
                a_idx = np.random.randint(len(self.pools))
                pool_a = self.pools[a_idx]
            _tries += 1
            if self.mode == 'submap':
                context = None
                if self.inter_map_ratio > 0 and np.random.random() < self.inter_map_ratio:
                    b_idx   = np.random.choice([i for i in range(len(self.pools)) if i != a_idx])
                    context = self.pools[b_idx]
                pair = generate_submap_pair(pool_a, context_pool=context)
            else:
                if self.inter_map_ratio > 0 and np.random.random() < self.inter_map_ratio:
                    b_idx = np.random.choice([i for i in range(len(self.pools)) if i != a_idx])
                    pair  = generate_inter_map_pair(pool_a, self.pools[b_idx],
                                                    overlap_probs=self.overlap_probs)
                else:
                    pair = generate_pair(pool_a, overlap_probs=self.overlap_probs)

        plucker1    = pair['plucker1']
        plucker2    = pair['plucker2']
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
