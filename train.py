#!/usr/bin/env python3
"""
train.py — ScalePluckerNet training entry point.

Trains on pre-generated .pkl datasets.
--dataset selects the split name: joint | slam_map | semantic3D | structured3D | ...

Examples
--------
# Standard training (foundation generator, hemisphere canon):
CANON=1 python train.py --dataset synthetic_found6 --geo_edge --bucket_batch \
    --match_w 0.2 --batch 1 --iter_size 32 --lr 5e-4 --gamma 0.99 --name v33_hemi

# Resume a run:
python train.py --dataset synthetic_found6 --name v33_hemi \
    --resume output/synthetic_found6/v33_hemi/checkpoint.pth
"""
import os
import sys
import random
import logging
from datetime import date

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import RandomSampler
import torch.optim.lr_scheduler as lr_sched
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

sys.path.insert(0, os.path.dirname(__file__))

from lib.dataloader import Sim3PluckerData, variable_collate
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
    p.add_argument('--dual_knn', action='store_true',
                   help='moment sub-network aggregates over a MOMENT-space '
                        'KNN graph (adds no params; warm-start compatible)')
    p.add_argument('--sign_inv', action='store_true',
                   help='per-line sign-EVEN input embedding phi(x)+phi(-x): makes '
                        'the matcher invariant to Plucker sign by construction '
                        '(no random-flip aug / canonicalisation needed)')

    p.add_argument('--nchannel', type=int, default=128,
                   help='network channel width (v19=128; 256 ~4x params)')
    p.add_argument('--gnn_pairs', type=int, default=6,
                   help='num self+cross GNN layer pairs (v19=6 -> 12 layers)')

    p.add_argument('--dustbin', action='store_true',
                   help='v24: SuperGlue-style unbalanced Sinkhorn with a '
                        'learnable no-match bin — unmatched lines park mass '
                        'in the bin instead of polluting real pairs (real '
                        'cross-modal overlap is 2-8%%)')
    p.add_argument('--attn_heads', type=int, default=4,
                   help='attention heads in the GNN (default 4 = upstream; '
                        '1 = single-head ablation, same param count)')
    p.add_argument('--geo_knn', action='store_true',
                   help='v29: sign-invariant GEOMETRIC KNN graph (foot points '
                        '+ sqrt(2)sin(theta) direction term) instead of KNN on '
                        'learned/moment channels — restores direction-aware '
                        'neighborhoods without sign fragility')
    p.add_argument('--graff_enc', action='store_true',
                   help='v41: ONE Grassmannian branch replaces the moment and '
                        'direction branches. Nodes = vec(P) (10-D projector, '
                        'SHARED origin+sigma from the query so the two clouds '
                        'are comparable); edges = the two principal angles, '
                        'which subsume geo_edge. Implies the graff graph.')
    p.add_argument('--val_epoch_freq', type=int, default=1,
                   help='run validation every N epochs (default 1). Validation '
                        'measured at only 4.8%% of epoch wall time, so 2 buys '
                        '~2.4%% — it does NOT change the LR schedule or the '
                        'epoch definition, unlike enlarging train_epoch_size.')
    p.add_argument('--batch_budget', type=int, default=4096,
                   help='bucket batch = budget // max(n1,n2), capped by '
                        '--max_bucket (default 4096 = pre-2026-08-22 behaviour)')
    p.add_argument('--max_bucket', type=int, default=16,
                   help='hard cap on per-bucket batch size (default 16)')
    p.add_argument('--mem_budget', type=int, default=0,
                   help='SECOND, QUADRATIC cap: batch <= mem_budget // n^2. '
                        '0 = off. Attention cost is O(B*n^2), so the linear '
                        'budget under-batches small clouds and over-batches '
                        'large ones. 8400000 gives B=32 at n=512 (+52%% '
                        'throughput) while keeping n>=2460 at B=1.')
    p.add_argument('--graff_knn', action='store_true',
                   help='AFFINE-GRASSMANNIAN knn graph: each line -> its 2-plane '
                        'in R^4, chordal distance via the 10-dim vec(YY^T). ONE '
                        'graph unifying direction+position, no weighting knob '
                        '(the manifold + the median-radius scale set it).')
    p.add_argument('--geo_edge', action='store_true',
                   help='v24: Sim(3)-invariant relative-geometry edge features '
                        '(inter-line angle + normalized line-line distance) in '
                        'the input encoder')
    p.add_argument('--init_from', default=None,
                   help='checkpoint to initialize WEIGHTS from (fresh '
                        'optimizer/scheduler/epoch counter — unlike --resume). '
                        'For warm-starting on a new dataset mix.')
    p.add_argument('--match_w', type=float, default=0.0,
                   help='weight of the matchability loss on the Sinkhorn '
                        'marginal heads r/c (v21: 0.2). 0 = off (pre-v21).')
    p.add_argument('--bucket_batch', action='store_true',
                   help='group same-shape (n1,n2) pairs into batches '
                        '(requires a FOUND size-grid dataset). The model is '
                        'latency-bound below ~1.5k lines, so this is ~2-4x '
                        'wall-clock at identical sum-over-samples gradients.')

    return p.parse_args()


class BucketBatchSampler(torch.utils.data.Sampler):
    """Batch sampler grouping same-shape (n1, n2) pairs (FOUND grid data).

    Per-bucket batch size: max(1, min(max_b, budget // max(n1, n2))) — 256-line
    pairs batch 16 deep, 4096-line pairs run alone. Each epoch draws
    epoch_size indices without replacement and shuffles the batch order."""

    def __init__(self, shapes, epoch_size, budget=4096, max_b=16, seed=0,
                 mem_budget=0):
        self.shapes = shapes
        self.epoch_size = (min(epoch_size, len(shapes)) if epoch_size > 0
                           else len(shapes))
        self.budget, self.max_b = budget, max_b
        # mem_budget (0 = off, preserves the pre-2026-08-22 behaviour exactly):
        # a SECOND cap that is QUADRATIC in n, because attention memory/compute
        # is O(B * n^2) while `budget // n` is only linear. Measured on the
        # RTX 5090: n=512 gains +52% going B 8->32 (157->239 samples/s) but
        # n=1094 gains NOTHING (52.4->52.8) and n>=2460 would OOM. So the linear
        # rule under-batches small clouds and over-batches large ones.
        self.mem_budget = mem_budget
        self.seed, self.epoch = seed, 0

    def _batch_size(self, n1, n2):
        n = max(n1, n2)
        b = self.budget // n
        if self.mem_budget:
            b = min(b, self.mem_budget // (n * n))
        return max(1, min(self.max_b, b))

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        idx = rng.choice(len(self.shapes), self.epoch_size, replace=False)
        groups = {}
        for i in idx:
            groups.setdefault(self.shapes[i], []).append(int(i))
        batches = []
        for (n1, n2), ids in groups.items():
            B = self._batch_size(n1, n2)
            batches.extend(ids[j:j + B] for j in range(0, len(ids), B))
        order = rng.permutation(len(batches))
        for j in order:
            yield batches[j]

    def __len__(self):
        # estimate; the trainer paces itself by loader.epoch_samples
        groups = {}
        for sh in self.shapes:
            groups[sh] = groups.get(sh, 0) + 1
        frac = self.epoch_size / max(1, len(self.shapes))
        return max(1, int(sum(
            np.ceil(cnt * frac / self._batch_size(*sh))
            for sh, cnt in groups.items())))


def main():
    args = parse_args()

    # TF32 matmuls: free throughput on Ampere+ GPUs, safe for training
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    val_dataset = args.val_dataset or args.dataset

    configs = edict(
        # Network
        net_nchannel       = args.nchannel,
        GNN_layers         = ['self', 'cross'] * args.gnn_pairs,
        net_lambda         = 0.1,
        net_maxiter        = 30,
        dual_knn           = args.dual_knn,
        sign_inv           = args.sign_inv,
        # record the sign handling explicitly so load_network never infers it
        # from `not sign_inv` (see tools/eval_registration_gt.load_network)
        canon              = bool(int(os.environ.get('CANON', '1'))),
        dustbin            = args.dustbin,
        geo_edge           = args.geo_edge,
        geo_knn            = args.geo_knn,
        graff_knn          = args.graff_knn,
        graff_enc          = args.graff_enc,
        attn_heads         = args.attn_heads,
        # Training
        out_dir            = 'output',
        optimizer          = 'Adam',
        train_start_epoch  = 0,
        train_save_freq_epoch = 1,
        val_epoch_freq     = args.val_epoch_freq,
        # bucket-batch packing (2026-08-22). MUST be listed here: `configs` is
        # an EXPLICIT edict, not vars(args), so a CLI flag that is not copied in
        # silently falls back to the sampler defaults and the run looks
        # IDENTICAL (caught only by the unchanged batches/epoch line).
        batch_budget       = args.batch_budget,
        max_bucket         = args.max_bucket,
        mem_budget         = args.mem_budget,
        use_gpu            = True,
        print_freq         = 10,
        train_seed         = 0,
        # Set from args
        dataset            = args.dataset,
        data_dir           = args.data_dir,
        gpu_inds           = args.gpu,
        model_nb           = args.name if args.name else str(date.today()),
        train_batch_size   = args.batch,
        iter_size          = args.iter_size,
        train_lr           = args.lr,
        exp_gamma          = args.gamma,
        train_epoches      = args.epochs,
        best_val_metric    = args.metric,
        ransac_type        = args.ransac,
        val_max_iter       = args.val_max_iter,
        resume_dir         = None,
        in_channel         = 6,
        pose_loss_weight   = args.pose_loss,
        resume             = args.resume,
        match_w            = args.match_w,
    )

    if configs.train_seed is not None:
        random.seed(configs.train_seed)
        torch.manual_seed(configs.train_seed)
        torch.cuda.manual_seed(configs.train_seed)
        cudnn.deterministic = True

    logging.info('===> ScalePlueckerNet Training')
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

    train_dataset = Sim3PluckerData(phase='train', config=configs)

    if args.bucket_batch:
        shapes = [(len(p1), len(p2)) for p1, p2 in
                  zip(train_dataset.data['plucker1'],
                      train_dataset.data['plucker2'])]
        n_shapes = len(set(shapes))
        epoch_size = args.train_epoch_size if args.train_epoch_size > 0 \
            else len(train_dataset)
        sampler = BucketBatchSampler(shapes, epoch_size,
                                     budget=getattr(configs, 'batch_budget', 4096),
                                     max_b=getattr(configs, 'max_bucket', 16),
                                     mem_budget=getattr(configs, 'mem_budget', 0),
                                     seed=configs.train_seed or 0)
        logging.info(f'  bucket_batch: {n_shapes} distinct shapes, '
                     f'~{len(sampler)} batches/epoch for {epoch_size} samples')
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=args.workers, pin_memory=True,
            persistent_workers=(args.workers > 0),
            collate_fn=variable_collate,
        )
        train_loader.epoch_samples = epoch_size
    elif args.train_epoch_size > 0:
        train_sampler = RandomSampler(train_dataset,
                                      replacement=True,
                                      num_samples=args.train_epoch_size)
        logging.info(f'  train_epoch_size: {args.train_epoch_size} '
                     f'(dataset has {len(train_dataset)} pairs)')
        train_loader = DataLoader(
            train_dataset,
            batch_size=configs.train_batch_size,
            sampler=train_sampler, drop_last=True,
            num_workers=args.workers, pin_memory=True,
            persistent_workers=(args.workers > 0),
            collate_fn=collate,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=configs.train_batch_size,
            shuffle=True, drop_last=True,
            num_workers=args.workers, pin_memory=True,
            persistent_workers=(args.workers > 0),
            collate_fn=collate,
        )

    val_cfg = edict(dict(configs))
    val_cfg.dataset = val_dataset
    val_dataset_obj = Sim3PluckerData(phase='valid', config=val_cfg)
    val_loader = DataLoader(
        val_dataset_obj,
        batch_size=1, shuffle=False, drop_last=False,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )

    # ── Build trainer ─────────────────────────────────────────────────────────
    trainer = Sim3Trainer(configs, train_loader, val_loader)
    if args.init_from:
        state = torch.load(args.init_from, map_location='cpu',
                           weights_only=False)
        trainer.model.load_state_dict(state['state_dict'])
        logging.info(f'initialized weights from {args.init_from} '
                     f'(epoch {state.get("epoch", "?")}, fresh optimizer)')

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
    # On resume, trainer.start_epoch already reflects the checkpoint's epoch
    # (set in Sim3Trainer.__init__), but a freshly constructed ExponentialLR
    # does NOT retroactively recompute lr from `last_epoch` at __init__ time
    # in this torch build (get_lr() returns the current, unmultiplied lr on
    # the initial step) — so the resumed lr must be set explicitly, or every
    # resume silently kicks the LR back up to args.lr instead of continuing
    # to anneal from wherever it had decayed to.
    resume_last_epoch = trainer.start_epoch - 1 if args.resume else -1
    resumed_lr = (args.lr * (trainer.config.exp_gamma ** trainer.start_epoch)
                 if args.resume else args.lr)
    for pg in trainer.optimizer.param_groups:
        pg['lr'] = resumed_lr
        pg['initial_lr'] = args.lr
    trainer.scheduler = lr_sched.ExponentialLR(trainer.optimizer,
                                               gamma=trainer.config.exp_gamma,
                                               last_epoch=resume_last_epoch)
    if resume_last_epoch >= 0:
        logging.info(f'Resumed ExponentialLR at epoch {trainer.start_epoch}: '
                     f'lr={resumed_lr:.3e} (= {args.lr:.1e} * '
                     f'{trainer.config.exp_gamma}^{trainer.start_epoch})')

    if args.cosine_lr:
        # Single cosine anneal over the whole run (T_mult=1, T_0=epochs):
        # lr goes args.lr -> eta_min across epochs 0..args.epochs, landing the
        # anneal exactly at the horizon. On resume, align the cosine phase to the
        # model epoch via last_epoch (the old ckpt's ExponentialLR state is
        # incompatible, so we reconstruct rather than load it). initial_lr was
        # set on every param group above, which cosine needs as its base_lr.
        cos_last_epoch = (trainer.start_epoch - 1) if args.resume else -1
        trainer.scheduler = lr_sched.CosineAnnealingWarmRestarts(
            trainer.optimizer, T_0=args.epochs, T_mult=1, eta_min=1e-6,
            last_epoch=cos_last_epoch,
        )
        logging.info(f'Scheduler: CosineAnnealingWarmRestarts(T_0={args.epochs}, '
                     f'T_mult=1, eta_min=1e-6, last_epoch={cos_last_epoch})')

    trainer.train()


if __name__ == '__main__':
    main()
