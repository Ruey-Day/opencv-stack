#!/usr/bin/env python3
"""
train.py — ScalePlueckerNet training entry point.

Extensions over the original PlueckerNet (all opt-in via flags):

  CORE (always active)
    Sim(3) scale recovery: network learns R, t, AND scale s jointly.
    The Sim(3) RANSAC solver is used for validation instead of SE(3).

  --dataset         Controls the training data source.
                      semantic3D / structured3D  — original PlueckerNet indoor data (SE3, s=1)
                      replica_gs                 — Replica RGBD, world-space GlueStick lines
                      7scenes_gs                 — 7-Scenes RGBD, world-space GlueStick lines
                      joint (default)            — all four sources combined

  --dustbin         Adds a learnable dustbin token (SuperGlue-style) so unmatched
                    lines are assigned to the dustbin rather than forced onto wrong
                    correspondences.  Intended for fine-tuning from a joint checkpoint.

  --cosine_lr       Switches from ExponentialLR to CosineAnnealingWarmRestarts
                    (T_0=50, T_mult=2, eta_min=1e-6).

Examples
--------
# Original PlueckerNet data only, Sim(3) scale:
python train.py --dataset semantic3D --batch 12 --lr 1e-3

# All sources, geometry-only 6D (default):
python train.py --dataset joint

# Dustbin fine-tuning from pre-trained joint checkpoint:
python train.py --dataset joint --dustbin \\
    --pretrain output/joint/2026-05-12/best_val_checkpoint.pth --lr 2e-4

# Resume any run:
python train.py --dataset joint --resume output/joint/2026-05-12/checkpoint.pth
"""
import os
import sys
import random
import logging
from datetime import date

import torch
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_sched
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

PLUECKERNET_DIR = os.path.join(os.path.dirname(__file__), '..', 'PlueckerNet')
sys.path.insert(0, os.path.abspath(PLUECKERNET_DIR))
sys.path.insert(0, os.path.dirname(__file__))

from config import get_config
from sim3.dataloader import Sim3PluckerData
from sim3.trainer import Sim3Trainer

logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger().setLevel(logging.INFO)


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='ScalePlueckerNet training')

    # Data
    p.add_argument('--dataset',    default='joint',
                   help='Training data: semantic3D | structured3D | replica_gs | 7scenes_gs | joint')
    p.add_argument('--data_dir',   default='./dataset')
    p.add_argument('--n_lines',    type=int, default=700,
                   help='Lines per scene after subsampling (must match dataset)')
    p.add_argument('--n_inliers',  type=int, default=490,
                   help='GT inliers per scene (must match dataset)')

    # Training
    p.add_argument('--epochs',     type=int,   default=400)
    p.add_argument('--batch',      type=int,   default=32)
    p.add_argument('--lr',         type=float, default=5e-4)
    p.add_argument('--gpu',        type=int,   default=0)
    p.add_argument('--workers',    type=int,   default=8)

    # Extensions (all off by default)
    p.add_argument('--in_channel', type=int, default=6,
                   help='6=geometry only (default)  9=Plücker+LAB color')
    p.add_argument('--dustbin',    action='store_true',
                   help='Enable learnable dustbin token for partial-overlap robustness')
    p.add_argument('--cosine_lr',  action='store_true',
                   help='CosineAnnealingWarmRestarts instead of ExponentialLR')

    # Checkpointing
    p.add_argument('--pretrain',   default=None,
                   help='Warm-start from checkpoint (strict=False, e.g. for dustbin init)')
    p.add_argument('--resume',     default=None,
                   help='Resume training from checkpoint')

    return p.parse_args()


def main():
    args = parse_args()

    configs = get_config()
    configs.dataset             = args.dataset
    configs.data_dir            = args.data_dir
    configs.gpu_inds            = args.gpu
    configs.model_nb            = str(date.today())
    configs.train_batch_size    = args.batch
    configs.train_lr            = args.lr
    configs.train_epoches       = args.epochs
    configs.best_val_metric     = 'avg_inlier_ratio'
    configs.resume_dir          = None
    configs.normalize_n_lines   = args.n_lines
    configs.normalize_n_inliers = args.n_inliers
    configs.in_channel          = args.in_channel

    dconfig = vars(configs)
    dconfig['resume'] = args.resume
    configs = edict(dconfig)

    if configs.train_seed is not None:
        random.seed(configs.train_seed)
        torch.manual_seed(configs.train_seed)
        torch.cuda.manual_seed(configs.train_seed)
        cudnn.deterministic = True

    logging.info('===> ScalePlueckerNet Training')
    logging.info(f'  dataset    : {args.dataset}')
    logging.info(f'  in_channel : {args.in_channel}')
    logging.info(f'  dustbin    : {args.dustbin}')
    logging.info(f'  cosine_lr  : {args.cosine_lr}')

    train_loader = DataLoader(
        Sim3PluckerData(phase='train', config=configs),
        batch_size=configs.train_batch_size,
        shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        Sim3PluckerData(phase='valid', config=configs),
        batch_size=1, shuffle=False, drop_last=False,
        num_workers=2,
    )

    if args.dustbin:
        from sim3.trainer_dustbin import DustbinTrainer
        trainer = DustbinTrainer(configs, train_loader, val_loader)
    else:
        trainer = Sim3Trainer(configs, train_loader, val_loader)

    if args.pretrain and os.path.exists(args.pretrain):
        ckpt = torch.load(args.pretrain, map_location='cpu', weights_only=False)
        state = ckpt.get('model', ckpt.get('state_dict', ckpt))
        missing, unexpected = trainer.model.load_state_dict(state, strict=False)
        logging.info(f'Loaded pretrain: {args.pretrain}')
        if missing:
            logging.info(f'  Missing (re-init): {missing}')
        if unexpected:
            logging.info(f'  Unexpected (skipped): {unexpected}')
    elif args.pretrain:
        logging.warning(f'Pretrain not found: {args.pretrain}')

    if args.cosine_lr:
        trainer.scheduler = lr_sched.CosineAnnealingWarmRestarts(
            trainer.optimizer, T_0=50, T_mult=2, eta_min=1e-6,
        )
        logging.info('Scheduler: CosineAnnealingWarmRestarts(T_0=50, T_mult=2)')

    trainer.train()


if __name__ == '__main__':
    main()
