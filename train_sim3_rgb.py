#!/usr/bin/env python3
"""
Training entry point for Sim(3)-aware PlueckerNet with RGB colors (9D Plücker).

Extended from train_sim3.py to use 9D colored Plücker coordinates instead of 6D.
The network's input layer (conv_in_seq_direction_moment_knn) automatically adapts
to the 9D input dimensions via the in_channel config parameter.

Typical workflow:
    # 1. Generate synthetic dataset with colors (once)
    python generate_sim3_dataset.py --out_dir ./dataset --n_train 5000 --n_valid 500 --add_colors

    # 2. Train with RGB
    python train_sim3_rgb.py
"""
import os
import sys
import random
import logging
import json
from datetime import date

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

# ---- make original PlueckerNet importable --------------------------------
PLUECKERNET_DIR = os.path.join(os.path.dirname(__file__), '..', 'PlueckerNet')
sys.path.insert(0, os.path.abspath(PLUECKERNET_DIR))
# --------------------------------------------------------------------------

from config import get_config
from sim3.dataloader import Sim3PluckerData
from sim3.trainer import Sim3Trainer

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch]
)


def main(configs):
    train_loader = DataLoader(
        Sim3PluckerData(phase='train', config=configs),
        batch_size=configs.train_batch_size,
        shuffle=True, drop_last=True, num_workers=4,
    )
    val_loader = DataLoader(
        Sim3PluckerData(phase='valid', config=configs),
        batch_size=1, shuffle=False, drop_last=False, num_workers=1,
    )
    trainer = Sim3Trainer(configs, train_loader, val_loader)
    trainer.train()


if __name__ == '__main__':
    configs = get_config()

    # ---- experiment settings for RGB training ----------------------------
    configs.dataset          = 'sim3_synthetic'
    configs.data_dir         = './dataset'
    configs.gpu_inds         = 0
    configs.model_nb         = str(date.today())
    configs.train_batch_size = 12
    configs.train_lr         = 1e-3
    configs.train_epoches    = 400
    configs.best_val_metric  = 'avg_inlier_ratio'
    configs.resume_dir       = None
    configs.in_channel       = 9  # Key difference: 9D Plücker + RGB vs 6D standard
    # --------------------------------------------------------------------------

    dconfig = vars(configs)
    dconfig['resume'] = None

    logging.info('===> Configurations')
    for k in sorted(dconfig):
        logging.info(f'    {k}: {dconfig[k]}')

    configs = edict(dconfig)

    if configs.train_seed is not None:
        random.seed(configs.train_seed)
        torch.manual_seed(configs.train_seed)
        torch.cuda.manual_seed(configs.train_seed)
        cudnn.deterministic = True

    main(configs)
