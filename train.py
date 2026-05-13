#!/usr/bin/env python3
"""
train.py — ScalePlueckerNet unified training entry point.

Extensions over the original PlueckerNet are opt-in via flags:
  --color      9D input (Plücker + LAB color); needs a *_color dataset
  --dustbin    Dustbin token for partial-overlap robustness (Phase 4)
  --cosine_lr  CosineAnnealingWarmRestarts instead of ExponentialLR

With no flags and --dataset se3real_sim3 --n_lines 130, training closely
matches the original PlueckerNet setup (same data distribution, same
architecture, SE(3) data — only difference is Sim(3) scale labels).

Examples
--------
# se3real baseline (≈ original PlueckerNet with Sim(3) scale labels):
python train.py --dataset se3real_sim3 --batch 12 --lr 1e-3 --n_lines 130 --n_inliers 100

# joint 6D — all 4 indoor datasets, 700 lines (best geometry model):
python train.py --dataset joint

# joint 9D color — warm-start from 6D joint checkpoint:
python train.py --dataset joint_color --color --cosine_lr \\
    --pretrain output/joint/2026-05-12/best_val_checkpoint.pth

# Phase 4 dustbin fine-tuning — partial-overlap robustness:
python train.py --dataset partial_overlap --dustbin \\
    --pretrain output/joint/2026-05-12/best_val_checkpoint.pth \\
    --lr 2e-4

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
    p.add_argument('--dataset',   default='joint',
                   help='Dataset name under data_dir (default: joint)')
    p.add_argument('--data_dir',  default='./dataset')
    p.add_argument('--n_lines',   type=int, default=700,
                   help='Max lines per scene (700 = full distribution)')
    p.add_argument('--n_inliers', type=int, default=490,
                   help='Max GT inliers per scene (must match dataset generation)')

    # Training
    p.add_argument('--epochs',    type=int,   default=400)
    p.add_argument('--batch',     type=int,   default=32)
    p.add_argument('--lr',        type=float, default=5e-4)
    p.add_argument('--gpu',       type=int,   default=0)
    p.add_argument('--workers',   type=int,   default=8)

    # Model
    p.add_argument('--in_channel', type=int, default=9,
                   help='Input channels: 9=Plücker+LAB (default), 6=geometry only')
    p.add_argument('--no_dustbin', action='store_true',
                   help='Disable dustbin token (use plain Sim3Trainer instead)')
    p.add_argument('--cosine_lr',  action='store_true',
                   help='CosineAnnealingWarmRestarts(T_0=50, T_mult=2, eta_min=1e-6)')

    # Checkpointing
    p.add_argument('--pretrain',  default=None,
                   help='Load weights strict=False (warm-start; e.g. 6D→9D or dustbin init)')
    p.add_argument('--resume',    default=None,
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
    for k in sorted(dconfig):
        logging.info(f'    {k}: {dconfig[k]}')

    if not args.no_dustbin:
        from sim3.dataloader_phase4 import PartialOverlapData
        from sim3.trainer_dustbin import DustbinTrainer
        train_loader = DataLoader(
            PartialOverlapData(phase='train', config=configs),
            batch_size=configs.train_batch_size,
            shuffle=True, drop_last=True,
            num_workers=args.workers, pin_memory=True,
        )
        val_loader = DataLoader(
            PartialOverlapData(phase='valid', config=configs),
            batch_size=1, shuffle=False, drop_last=False,
            num_workers=2,
        )
        trainer = DustbinTrainer(configs, train_loader, val_loader)
    else:
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
        trainer = Sim3Trainer(configs, train_loader, val_loader)

    if args.pretrain and os.path.exists(args.pretrain):
        ckpt = torch.load(args.pretrain, map_location='cpu', weights_only=False)
        state = ckpt.get('model', ckpt.get('state_dict', ckpt))
        missing, unexpected = trainer.model.load_state_dict(state, strict=False)
        logging.info(f'Loaded pretrain weights: {args.pretrain}')
        if missing:
            logging.info(f'  Missing keys (re-init): {missing}')
        if unexpected:
            logging.info(f'  Unexpected keys (skipped): {unexpected}')
    elif args.pretrain:
        logging.warning(f'Pretrain weights not found: {args.pretrain}')

    if args.cosine_lr:
        trainer.scheduler = lr_sched.CosineAnnealingWarmRestarts(
            trainer.optimizer, T_0=50, T_mult=2, eta_min=1e-6,
        )
        logging.info('Scheduler: CosineAnnealingWarmRestarts(T_0=50, T_mult=2, eta_min=1e-6)')

    trainer.train()


if __name__ == '__main__':
    main()
