"""
Sim3PluckerData — static .pkl files (pre-generated offline)

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

# CANON=1 : hemisphere sign-canonicalisation of [m, d] lines (v14 experiment).
# Flip each 6-vector jointly so the largest-|component| of the DIRECTION (chan
# 3:6, see feedback on inverted channel names) is >= 0. Idempotent w.r.t. the
# generator's baked-in 50% random sign flips, so the existing dataset is reused
# unchanged; must be applied identically at inference (segments_to_plucker).
_CANON = bool(int(os.environ.get('CANON', '0')))

def canonicalize_sign(p):
    """p: (N, 6) [m, d]. Flip [m; d] jointly so d's largest-|comp| >= 0."""
    d = p[:, 3:6]
    idx = np.argmax(np.abs(d), axis=1)
    s = np.sign(d[np.arange(len(d)), idx]); s[s == 0] = 1.0
    return p * s[:, None]

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

        if _CANON:                       # v14: remove sign DOF before the encoder
            plucker1 = canonicalize_sign(plucker1)
            plucker2 = canonicalize_sign(plucker2)

        # Moment normalisation: divide BOTH clouds by plucker2 (query) std.
        # This preserves the scale ratio p1_inlier/p2 ≈ s_gt, which is the
        # key matching signal, while keeping moments in a manageable range.
        # Using p2 std (not p1) avoids contamination by background lines in
        # the KITTI reference cloud which would otherwise compress inlier
        # moments to near-zero, destroying the correspondence signal.
        std2 = float(plucker2[:, :3].std()) + 1e-6
        plucker1[:, :3] /= std2
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
