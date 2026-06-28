"""
Sim3PluckerData        — static .pkl files (pre-generated offline)
SyntheticLiveData      — on-the-fly infinite pair generation (no disk needed)
SyntheticValData       — fixed val set generated once at startup with a seed

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
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

# Lazy import so the repo root doesn't need to be on sys.path at import time
def _get_generate_pair():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from scripts.generate_synthetic import _generate_pair
    return _generate_pair


def worker_seed_init(worker_id: int) -> None:
    """Give each DataLoader worker a unique numpy seed derived from PyTorch's seed."""
    seed = (torch.initial_seed() + worker_id * 7919) % (2 ** 32)
    np.random.seed(seed)


def _pair_to_tuple(p: dict):
    """Convert _generate_pair() dict → (matches, p1, p2, R, t, s) numpy tuple."""
    n1, n2 = p['plucker1'].shape[0], p['plucker2'].shape[0]
    matches = np.zeros((n1, n2), np.float32)
    if p['matches'].shape[1] > 0:
        matches[p['matches'][0], p['matches'][1]] = 1.0
    return (
        matches,
        p['plucker1'].astype(np.float32),
        p['plucker2'].astype(np.float32),
        p['R_gt'].astype(np.float32),
        p['t_gt'].astype(np.float32),
        np.float32(p['s_gt']),
    )


class SyntheticLiveData(Dataset):
    """Infinite on-the-fly synthetic dataset.

    Each call to __getitem__ generates a fresh unique pair regardless of index.
    Use with worker_seed_init as worker_init_fn to ensure each worker draws
    from a different random stream.
    """

    def __init__(self, epoch_size: int):
        self._generate = _get_generate_pair()
        self.epoch_size = epoch_size

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        return _pair_to_tuple(self._generate())


class SyntheticValData(Dataset):
    """Fixed validation set generated once at construction with a fixed seed.

    Reproducible across runs; the seed is decoupled from training RNG state
    so training randomness doesn't affect the val set.
    """

    def __init__(self, n_pairs: int = 5000, seed: int = 0):
        generate = _get_generate_pair()
        rng_state = np.random.get_state()
        np.random.seed(seed)
        import time; t0 = time.time()
        self._data = [_pair_to_tuple(generate()) for _ in range(n_pairs)]
        np.random.set_state(rng_state)
        print(f'[SyntheticValData] generated {n_pairs} pairs in {time.time()-t0:.1f}s (seed={seed})')

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

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

def load_sim3_data(config, split):
    var_names = ['matches', 'plucker1', 'plucker2', 'R_gt', 't_gt', 's_gt']
    folder = os.path.join(config.data_dir, f'{config.dataset}_{split}')
    data = {}
    for var in var_names:
        with open(os.path.join(folder, f'{var}.pkl'), 'rb') as f:
            data[var] = pickle.load(f, encoding='latin1')
    print(f'[Sim3] loaded {split}: {len(data["t_gt"])} scenes from {folder}')
    return data

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
        plucker1    = self.data['plucker1'][index].copy()    # (n_lines, 6)
        plucker2    = self.data['plucker2'][index].copy()
        R_gt        = self.data['R_gt'][index]
        t_gt        = self.data['t_gt'][index]
        s_gt        = np.float32(self.data['s_gt'][index])

        if self.in_channel is not None:
            plucker1 = plucker1[:, :self.in_channel]
            plucker2 = plucker2[:, :self.in_channel]

        # Per-cloud moment normalisation: brings outdoor (±100 m) and indoor
        # (±5 m) moments into the same activation range for the encoder.
        # Directions (unit vectors) are unchanged.  The matching matrix is
        # index-based so correctness is unaffected.
        std1 = float(plucker1[:, :3].std()) + 1e-6
        std2 = float(plucker2[:, :3].std()) + 1e-6
        plucker1[:, :3] /= std1
        plucker2[:, :3] /= std2

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
