#!/usr/bin/env python3
"""
train.py — unified ScalePluckerNet training entry point.

Modes
-----
standard (default)
    Train on pre-generated .pkl datasets.
    --dataset selects the split name: joint | slam_map | semantic3D | structured3D | ...

live
    Generate training pairs on-the-fly from Structure-PLP-SLAM .db map files.
    Pass --db_train with one or more .db paths (shell globs are expanded).
    Validation always uses a static .pkl split (--dataset / --val_dataset).

Note: train_cross_modal.py and train_slam_maps.py have been merged into this
script.  Cross-modal / SLAM-map fine-tuning is now done via --mode live with
--db_train pointing to your map files plus --pretrain for the starting weights.

Examples
--------
# Standard training on the joint dataset:
python train.py --dataset joint --batch 32 --lr 5e-4 --gamma 0.99

# From scratch on SLAM-map live data (all available maps):
python train.py --mode live \\
    --db_train ../Structure-PLP-SLAM/*.db \\
    --val_dataset slam_map \\
    --lr 5e-4 --gamma 0.99 --batch 32 --epochs 1000

# Fine-tune on live SLAM-map pairs starting from a joint checkpoint:
python train.py --mode live \\
    --db_train /path/to/Structure-PLP-SLAM/*.db \\
    --val_dataset slam_map \\
    --n_lines 200 --n_inliers 60 \\
    --pretrain output/joint/2026-05-17/best_val_checkpoint.pth \\
    --lr 1e-5 --batch 16 --epochs 300

# Resume any run:
python train.py --dataset joint --resume output/joint/2026-05-12/checkpoint.pth
"""
import os
import sys
import glob
import random
import logging
from datetime import date

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

PLUECKERNET_DIR = os.path.join(os.path.dirname(__file__), 'PlueckerNet')
sys.path.insert(0, os.path.abspath(PLUECKERNET_DIR))
sys.path.insert(0, os.path.dirname(__file__))

from config import get_config
from sim3.dataloader import Sim3PluckerData, LiveSim3PluckerData, variable_collate
from sim3.trainer import Sim3Trainer

logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger().setLevel(logging.INFO)


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='ScalePlueckerNet unified training')

    # Mode
    p.add_argument('--mode', default='standard', choices=['standard', 'live'],
                   help='standard: use pre-generated .pkl datasets; '
                        'live: generate pairs on-the-fly from .db map files')

    # Data — standard mode
    p.add_argument('--dataset',     default='joint',
                   help='Dataset name (joint | slam_map | semantic3D | structured3D | ...)')
    p.add_argument('--val_dataset', default=None,
                   help='Validation dataset name (default: same as --dataset)')
    p.add_argument('--data_dir',    default='./dataset')
    p.add_argument('--n_lines',     type=int, default=700,
                   help='Lines per scene after subsampling. Use 200 for SLAM-map mode.')
    p.add_argument('--n_inliers',   type=int, default=490,
                   help='GT inliers per scene. Use 60 for SLAM-map mode.')

    # Data — live mode
    p.add_argument('--db_train',        nargs='+', default=None,
                   help='[live mode] Path(s) to Structure-PLP-SLAM .db map files. '
                        'Shell globs are expanded automatically.')
    p.add_argument('--db_val',          nargs='+', default=None,
                   help='[live mode] .db files for live validation (default: same as --db_train). '
                        'Ignored when a static .pkl val split is found on disk.')
    p.add_argument('--epoch_size',      type=int, default=16000,
                   help='[live mode] Pairs generated per epoch (default: 16000)')
    p.add_argument('--val_size',        type=int, default=400,
                   help='[live mode] Val pairs generated when no .pkl split exists (default: 400)')
    p.add_argument('--inter_map_ratio', type=float, default=0.3,
                   help='[live mode] Fraction of cross-map pairs (default: 0.3)')
    p.add_argument('--submap',          action='store_true',
                   help='[live mode] Use asymmetric submap generator instead of symmetric pairs')

    # Training
    p.add_argument('--epochs',     type=int,   default=1000)
    p.add_argument('--batch',      type=int,   default=1,
                   help='Batch size. Default 1 for variable-length pairs; '
                        'use variable_collate (automatic) for batch > 1.')
    p.add_argument('--iter_size',  type=int,   default=32,
                   help='Gradient accumulation steps (effective batch = batch × iter_size). '
                        'Default 32 gives the same gradient quality as the old batch=32.')
    p.add_argument('--lr',         type=float, default=5e-4)
    p.add_argument('--gamma',      type=float, default=0.99,
                   help='ExponentialLR decay per epoch (default: 0.99). Use 1.0 to disable.')
    p.add_argument('--gpu',        type=int,   default=0)
    p.add_argument('--workers',    type=int,   default=8)
    p.add_argument('--name',       default=None,
                   help='Override run name (default: today\'s date)')

    # Extensions
    p.add_argument('--in_channel', type=int, default=6,
                   help='6=geometry only (default)  9=Plücker+LAB color')
    p.add_argument('--cosine_lr',  action='store_true',
                   help='CosineAnnealingWarmRestarts instead of ExponentialLR')
    p.add_argument('--pose_loss',  type=float, default=0.0,
                   help='Weight for differentiable Sim(3) pose loss (0 = disabled)')

    # Model architecture
    p.add_argument('--model', default='knn', choices=['knn', 'alt'],
                   help='knn: original PluckerNetKnn (default); '
                        'alt: asymmetric alternating attention (FUSER-inspired)')
    p.add_argument('--alt_n_blocks', type=int, default=3,
                   help='[alt model] Number of alternating attention blocks. '
                        'Default 3 → 12 attention modules (same as knn baseline). '
                        'Increase for more expressivity at higher compute cost.')

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


def _expand_globs(paths):
    """Expand shell globs in a list of paths."""
    expanded = []
    for pat in paths:
        matches = sorted(glob.glob(pat))
        expanded.extend(matches if matches else [pat])
    return expanded


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
    configs.in_channel       = args.in_channel
    configs.pose_loss_weight = args.pose_loss
    configs.model_type       = args.model
    configs.alt_n_blocks     = args.alt_n_blocks

    dconfig = vars(configs)
    dconfig['resume'] = args.resume
    configs = edict(dconfig)

    if configs.train_seed is not None:
        random.seed(configs.train_seed)
        torch.manual_seed(configs.train_seed)
        torch.cuda.manual_seed(configs.train_seed)
        cudnn.deterministic = True

    logging.info('===> ScalePlueckerNet Training')
    logging.info(f'  mode        : {args.mode}')
    logging.info(f'  dataset     : {args.dataset}')
    logging.info(f'  val_dataset : {val_dataset}')
    logging.info(f'  in_channel  : {args.in_channel}')
    logging.info(f'  cosine_lr   : {args.cosine_lr}')
    logging.info(f'  ransac      : {args.ransac}')
    logging.info(f'  metric      : {args.metric}')
    logging.info(f'  pose_loss   : {args.pose_loss}')
    logging.info(f'  model       : {args.model}')
    logging.info(f'  batch       : {args.batch}  iter_size: {args.iter_size}  '
                 f'(effective batch: {args.batch * args.iter_size})')
    if args.model == 'alt':
        logging.info(f'  alt_n_blocks: {args.alt_n_blocks}')
    if args.submap:
        logging.info('  pair mode   : submap (asymmetric big-map→small-submap)')

    # ── Build training data loader ────────────────────────────────────────────
    # variable_collate handles batches of different-N pairs; batch_size=1 needs none.
    collate = variable_collate if args.batch > 1 else None

    if args.mode == 'live':
        if not args.db_train:
            raise ValueError('--mode live requires --db_train <path(s)>')
        db_paths = _expand_globs(args.db_train)
        logging.info(f'  live db_train: {len(db_paths)} map files')
        train_dataset = LiveSim3PluckerData(
            db_paths=db_paths,
            epoch_size=args.epoch_size,
            config=configs,
            inter_map_ratio=args.inter_map_ratio,
            mode='submap' if args.submap else 'symmetric',
        )
    else:
        train_dataset = Sim3PluckerData(phase='train', config=configs)

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
    # After trainer.__init__ loads the checkpoint, the scheduler state from the
    # checkpoint is applied to whatever scheduler type was just created. This
    # contaminates the new scheduler (e.g. ExponentialLR gets CAWR's _last_lr).
    # Always reset optimizer param_group LRs to the requested args.lr so that
    # the actual training LR matches what was specified on the command line.
    for pg in trainer.optimizer.param_groups:
        pg['lr'] = args.lr
    # Reinitialise the default ExponentialLR from scratch so it starts clean.
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
