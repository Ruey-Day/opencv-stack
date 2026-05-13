"""
Dataloader for dustbin training (joint dataset with embedded overlap).

Each sample may have zero, partial, or full overlap — encoded via s_gt:
  s_gt == 0.0  →  zero overlap  (no GT pose, empty matches)
  s_gt  > 0.0  →  partial or full overlap

Returns a 6-tuple: (matches, plucker1, plucker2, R_gt, t_gt, s_gt).
"""
import os
import sys
import pickle
import numpy as np
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _load(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')


class PartialOverlapData(Dataset):
    """Loads the unified joint dataset (9D Plücker+LAB, embedded overlap).

    Returns per sample:
        matches   (n1, n2) float32  — binary GT correspondence matrix
        plucker1  (n1, 9)  float32  — [m,d,LAB]
        plucker2  (n2, 9)  float32
        R_gt      (3, 3)   float32
        t_gt      (3, 1)   float32
        s_gt      ()       float32  — 0.0 for zero-overlap samples
    """

    VAR_NAMES = ['matches', 'plucker1', 'plucker2', 'R_gt', 't_gt', 's_gt']

    def __init__(self, phase, config):
        super().__init__()
        folder    = os.path.join(config.data_dir, f'{config.dataset}_{phase}')
        self.data = {v: _load(os.path.join(folder, f'{v}.pkl'))
                     for v in self.VAR_NAMES}
        self.len  = len(self.data['t_gt'])
        print(f'[PartialOverlapData] loaded {phase}: {self.len} samples from {folder}')

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        matches_ind = self.data['matches'][index]   # (2, n_inliers)
        plucker1    = self.data['plucker1'][index]  # (n_lines, 9)
        plucker2    = self.data['plucker2'][index]
        R_gt        = self.data['R_gt'][index]
        t_gt        = self.data['t_gt'][index]
        s_gt        = np.float32(self.data['s_gt'][index])

        n1, n2  = plucker1.shape[0], plucker2.shape[0]
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
