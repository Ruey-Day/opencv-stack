"""
Dataloader for Sim(3) Plücker line datasets.

Extends the original PluckerData3D_precompute (lib/dataloader.py) to also
load the ground-truth scale s_gt that is absent in the SE(3) version.

Expected directory layout:
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
        self.data = load_sim3_data(config, phase)
        self.len  = len(self.data['t_gt'])

    def __getitem__(self, index):
        matches_ind = self.data['matches'][index]
        plucker1    = self.data['plucker1'][index]
        plucker2    = self.data['plucker2'][index]
        R_gt        = self.data['R_gt'][index]
        t_gt        = self.data['t_gt'][index]
        s_gt        = np.float32(self.data['s_gt'][index])

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
