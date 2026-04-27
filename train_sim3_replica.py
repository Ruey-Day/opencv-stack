#!/usr/bin/env python3
"""
Training entry point for Sim(3)-aware PlueckerNet on real Replica geometry.

Trains from scratch on real RGBD-derived Plücker lines with random Sim(3)
augmentation — closes the synthetic-to-real gap seen in eval_benchmark.py.

Workflow:
    # 1. Generate dataset (once, ~15 min)
    python generate_replica_dataset.py

    # 2. Train (run in tmux for long sessions)
    python train_sim3_replica.py
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

PLUECKERNET_DIR = os.path.join(os.path.dirname(__file__), '..', 'PlueckerNet')
sys.path.insert(0, os.path.abspath(PLUECKERNET_DIR))

from config import get_config
from sim3.dataloader import Sim3PluckerData
from sim3.trainer import Sim3Trainer

import logging, sys
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

    configs.dataset          = 'replica'
    configs.data_dir         = './dataset'
    configs.gpu_inds         = 0
    configs.model_nb         = str(date.today())
    configs.train_batch_size = 12
    configs.train_lr         = 1e-3
    configs.train_epoches    = 400
    configs.best_val_metric  = 'avg_inlier_ratio'
    configs.resume_dir       = None

    dconfig = vars(configs)
    dconfig['resume'] = './output/replica/2026-04-22/checkpoint.pth'

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
