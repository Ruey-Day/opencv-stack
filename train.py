#!/usr/bin/env python3
"""
train.py — ScalePluckerNet training entry point.

Trains on pre-generated .pkl datasets.
--dataset selects the split name: joint | slam_map | semantic3D | structured3D | ...

Examples
--------
# Standard training on the joint dataset:
python train.py --dataset joint --batch 32 --lr 5e-4 --gamma 0.99

# Resume a run:
python train.py --dataset joint --resume output/joint/2026-05-12/checkpoint.pth
"""
import os
import sys
import random
import logging
from datetime import date

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import RandomSampler
import torch.optim.lr_scheduler as lr_sched
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

sys.path.insert(0, os.path.dirname(__file__))

from config import get_config
from lib.dataloader import (Sim3PluckerData, variable_collate,
                            SyntheticLiveData, SyntheticValData, worker_seed_init)
from lib.trainer import Sim3Trainer

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
    p.add_argument('--dataset',     default='joint',
                   help='Dataset name (joint | slam_map | semantic3D | structured3D | ...)')
    p.add_argument('--val_dataset', default=None,
                   help='Validation dataset name (default: same as --dataset)')
    p.add_argument('--data_dir',    default='./dataset')
    p.add_argument('--n_lines',     type=int, default=700,
                   help='Lines per scene after subsampling.')
    p.add_argument('--n_inliers',   type=int, default=490,
                   help='GT inliers per scene.')

    # Live on-the-fly generation
    p.add_argument('--live', action='store_true',
                   help='Generate training pairs on-the-fly (infinite unique pairs, no pkl files). '
                        'Validation uses a fixed 5000-pair set generated at startup with seed=0.')
    p.add_argument('--val_pairs', type=int, default=5000,
                   help='Number of validation pairs to generate at startup (--live only).')

    # Training
    p.add_argument('--train_epoch_size', type=int, default=0,
                   help='If > 0, each epoch samples this many pairs with '
                        'replacement from the full dataset (keeps epochs short on large datasets). '
                        '0 = use the full dataset per epoch.')
    p.add_argument('--epochs',     type=int,   default=1000)
    p.add_argument('--batch',      type=int,   default=1,
                   help='Batch size. Default 1 for variable-length pairs; '
                        'use variable_collate (automatic) for batch > 1.')
    p.add_argument('--iter_size',  type=int,   default=32,
                   help='Gradient accumulation steps (effective batch = batch × iter_size).')
    p.add_argument('--lr',         type=float, default=5e-4)
    p.add_argument('--gamma',      type=float, default=0.99,
                   help='ExponentialLR decay per epoch (default: 0.99). Use 1.0 to disable.')
    p.add_argument('--gpu',        type=int,   default=0)
    p.add_argument('--workers',    type=int,   default=8)
    p.add_argument('--name',       default=None,
                   help='Override run name (default: today\'s date)')

    # Extensions
    p.add_argument('--cosine_lr',  action='store_true',
                   help='CosineAnnealingWarmRestarts instead of ExponentialLR')
    p.add_argument('--pose_loss',  type=float, default=0.0,
                   help='Weight for differentiable Sim(3) pose loss (0 = disabled)')

    # Validation RANSAC backend
    p.add_argument('--ransac',  default='grassmannian', choices=['sim3', 'grassmannian'],
                   help='RANSAC solver for validation metrics (default: grassmannian)')
    p.add_argument('--val_max_iter', type=int, default=-1,
                   help='Max validation samples per epoch (-1 = all)')
    p.add_argument('--metric',  default='avg_inlier_ratio',
                   choices=['avg_inlier_ratio'],
                   help='Metric used to pick the best checkpoint')

    # Checkpointing
    p.add_argument('--pretrain', default=None,
                   help='Warm-start from checkpoint (strict=False)')
    p.add_argument('--resume',   default=None,
                   help='Resume training from checkpoint')

    return p.parse_args()


def main():
    args = parse_args()

    # get_config() re-parses sys.argv via PlueckerNet's argparse — strip our
    # custom flags so it doesn't choke on unknown arguments.
    import sys as _sys
    _sys.argv = _sys.argv[:1]

    configs = get_config()

    val_dataset = args.val_dataset or args.dataset

    configs.dataset          = args.dataset
    configs.data_dir         = args.data_dir
    configs.gpu_inds         = args.gpu
    configs.model_nb         = args.name if args.name else str(date.today())
    configs.train_batch_size = args.batch
    configs.iter_size        = args.iter_size
    configs.train_lr         = args.lr
    configs.exp_gamma        = args.gamma
    configs.train_epoches    = args.epochs
    configs.best_val_metric  = args.metric
    configs.ransac_type      = args.ransac
    configs.val_max_iter     = args.val_max_iter
    configs.resume_dir       = None
    configs.in_channel       = 6
    configs.pose_loss_weight = args.pose_loss

    dconfig = vars(configs)
    dconfig['resume'] = args.resume
    configs = edict(dconfig)

    if configs.train_seed is not None:
        random.seed(configs.train_seed)
        torch.manual_seed(configs.train_seed)
        torch.cuda.manual_seed(configs.train_seed)
        cudnn.deterministic = True

    logging.info('===> ScalePlueckerNet Training')
    logging.info(f'  live        : {args.live}')
    logging.info(f'  dataset     : {args.dataset}')
    logging.info(f'  val_dataset : {val_dataset}')
    logging.info(f'  cosine_lr   : {args.cosine_lr}')
    logging.info(f'  ransac      : {args.ransac}')
    logging.info(f'  metric      : {args.metric}')
    logging.info(f'  pose_loss   : {args.pose_loss}')
    logging.info(f'  batch       : {args.batch}  iter_size: {args.iter_size}  '
                 f'(effective batch: {args.batch * args.iter_size})')

    # ── Build data loaders ────────────────────────────────────────────────────
    collate = variable_collate if args.batch > 1 else None

    if args.live:
        epoch_size = args.train_epoch_size if args.train_epoch_size > 0 else 19200
        logging.info(f'  [live] epoch_size={epoch_size}  val_pairs={args.val_pairs}')
        train_dataset = SyntheticLiveData(epoch_size=epoch_size)
        train_loader = DataLoader(
            train_dataset,
            batch_size=configs.train_batch_size,
            shuffle=True, drop_last=True,
            num_workers=args.workers, pin_memory=(args.batch > 1),
            collate_fn=collate,
            worker_init_fn=worker_seed_init,
        )
        logging.info('  [live] Generating validation set...')
        val_dataset_obj = SyntheticValData(n_pairs=args.val_pairs, seed=0)
        val_loader = DataLoader(
            val_dataset_obj,
            batch_size=1, shuffle=False, drop_last=False,
            num_workers=2, worker_init_fn=worker_seed_init,
        )
    else:
        train_dataset = Sim3PluckerData(phase='train', config=configs)

        if args.train_epoch_size > 0:
            train_sampler = RandomSampler(train_dataset,
                                          replacement=True,
                                          num_samples=args.train_epoch_size)
            logging.info(f'  train_epoch_size: {args.train_epoch_size} '
                         f'(dataset has {len(train_dataset)} pairs)')
            train_loader = DataLoader(
                train_dataset,
                batch_size=configs.train_batch_size,
                sampler=train_sampler, drop_last=True,
                num_workers=args.workers, pin_memory=(args.batch > 1),
                collate_fn=collate,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=configs.train_batch_size,
                shuffle=True, drop_last=True,
                num_workers=args.workers, pin_memory=(args.batch > 1),
                collate_fn=collate,
            )

        val_cfg = edict(dict(configs))
        val_cfg.dataset = val_dataset
        val_dataset_obj = Sim3PluckerData(phase='valid', config=val_cfg)
        val_loader = DataLoader(
            val_dataset_obj,
            batch_size=1, shuffle=False, drop_last=False,
            num_workers=2,
        )

    # ── Build trainer ─────────────────────────────────────────────────────────
    trainer = Sim3Trainer(configs, train_loader, val_loader)

    # ── Load pretrained weights (non-strict, for fine-tuning) ─────────────────
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

    # ── Scheduler ─────────────────────────────────────────────────────────────
    for pg in trainer.optimizer.param_groups:
        pg['lr'] = args.lr
    trainer.scheduler = lr_sched.ExponentialLR(trainer.optimizer,
                                               gamma=trainer.config.exp_gamma)

    if args.cosine_lr:
        trainer.scheduler = lr_sched.CosineAnnealingWarmRestarts(
            trainer.optimizer, T_0=50, T_mult=2, eta_min=1e-6,
        )
        if args.resume and os.path.exists(args.resume):
            ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
            if 'scheduler' in ckpt:
                try:
                    trainer.scheduler.load_state_dict(ckpt['scheduler'])
                    logging.info(f'Restored cosine scheduler state (last_epoch={ckpt["scheduler"].get("last_epoch")}, '
                                 f'lr={ckpt["scheduler"].get("_last_lr")})')
                except Exception as e:
                    logging.warning(f'Could not restore scheduler state: {e}')
        logging.info('Scheduler: CosineAnnealingWarmRestarts(T_0=50, T_mult=2)')

    trainer.train()


if __name__ == '__main__':
    main()
