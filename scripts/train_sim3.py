#!/usr/bin/env python3
"""
tail -f /home/rueyday/scale-aware-PlueckerNet/output/train_rgb.log
Training entry point for Sim(3)-aware PlueckerNet.

The network architecture (PluckerNetKnn) is reused unchanged from the
original PlueckerNet repo.  Only the training data and validation solver
are extended to Sim(3).

Typical workflow:
    # 1. Generate synthetic dataset (once)
    python generate_sim3_dataset.py --out_dir ./dataset --n_train 5000 --n_valid 500

    # 2. Train
    python train_sim3.py
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

_PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLUECKERNET_DIR = os.path.join(_PROJECT_ROOT, '..', 'PlueckerNet')
sys.path.insert(0, os.path.abspath(PLUECKERNET_DIR))
sys.path.insert(0, _PROJECT_ROOT)

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

    # ---- experiment settings ---------------------------------------------
    configs.dataset          = 'sim3_synthetic'
    configs.data_dir         = os.path.join(_PROJECT_ROOT, 'dataset')
    configs.gpu_inds         = 0
    configs.model_nb         = str(date.today())
    configs.train_batch_size = 12
    configs.train_lr         = 1e-3
    configs.train_epoches    = 400
    configs.best_val_metric  = 'avg_inlier_ratio'
    configs.resume_dir       = None
    # ----------------------------------------------------------------------

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
