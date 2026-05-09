#!/usr/bin/env python3
"""
train_sim3_se3real.py

Train Sim(3)-aware PlueckerNet on the original PlueckerNet training data
(Semantic3D + Structured3D) augmented with random scale.

This is the key verification experiment:
  - Same data distribution as SE3-PlueckerNet training (real 3D scan geometry)
  - Added random scale augmentation (Sim3 extension)
  - Expected result: better than SE3-Net on Sim3 tasks, similar on SE3 tasks

Workflow:
    # 1. Download PlueckerNet dataset (once)
    #    Semantic3D + Structured3D: Google Drive 1bVI0Ny4Ly1M4cBxbgRIjgHr8DtIXZLbb
    #    Extract into ../PlueckerNet/dataset/

    # 2. Generate Sim3 augmented dataset (once, ~2 min)
    python scripts/generate_se3_to_sim3_dataset.py

    # 3. Train
    python train_sim3_se3real.py
"""
import os
import sys
import random
import logging
from datetime import date

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

PLUECKERNET_DIR = os.path.join(os.path.dirname(__file__), '..', 'PlueckerNet')
sys.path.insert(0, os.path.abspath(PLUECKERNET_DIR))
sys.path.insert(0, os.path.dirname(__file__))

from config import get_config
from sim3.dataloader import Sim3PluckerData
from sim3.trainer import Sim3Trainer

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])


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

    configs.dataset          = 'se3real_sim3'
    configs.data_dir         = './dataset'
    configs.gpu_inds         = 0
    configs.model_nb         = str(date.today())
    configs.train_batch_size = 12
    configs.train_lr         = 1e-3
    configs.train_epoches    = 400
    configs.best_val_metric  = 'avg_inlier_ratio'
    configs.resume_dir       = None

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
