"""
train_cross_modal.py
====================
Fine-tune ScalePluckerNet on the cross-modal (SLAM-like ↔ DA3-like) Replica dataset.

v2: SLAM tensor = 200 lines (matches real ~147), DA3 = 700 lines.
    CosineAnnealingLR instead of ExponentialLR to avoid post-peak degradation.

Run from the ScalePluckerNet/ directory:
  cd /home/rueyday/scale-aware-cross-modal-registration/ScalePluckerNet
  /home/rueyday/miniconda3/envs/depth_anything/bin/python train_cross_modal.py
"""

import os, sys, random, json, logging
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
from datetime import date

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'PlueckerNet'))

from lib.dataloader import PluckerData3D_precompute
from config import get_config
from trainer_plucker import PluckerTrainer

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

# Start from the epoch-42 best checkpoint (200×700 fine-tune run)
PRETRAINED = os.path.join(
    SCRIPT_DIR, 'output', 'replica_cross_modal', '2026-05-18', 'best_val_checkpoint.pth'
)
PRETRAINED = os.path.abspath(PRETRAINED)

def main():
    configs = get_config()

    # ── Dataset ───────────────────────────────────────────────────────────────
    configs.dataset      = 'replica_cross_modal'
    configs.data_dir     = os.path.join(SCRIPT_DIR, '..', 'dataset')
    configs.train_phase  = 'train'
    configs.val_phase    = 'valid'

    # ── Model ─────────────────────────────────────────────────────────────────
    configs.model          = 'KNNContextNormNet'
    configs.net_depth      = 12
    configs.net_nchannel   = 128
    configs.GNN_layers     = ['self', 'cross'] * 6
    configs.net_gcnorm     = True
    configs.net_batchnorm  = True
    configs.net_topK       = 2000
    configs.net_lambda     = 0.1
    configs.net_maxiter    = 30
    configs.net_knn        = 10
    configs.in_channel     = 6        # pure Plücker, no LAB
    configs.normalize_n_lines   = 200  # SLAM side
    configs.normalize_n_inliers = 140
    configs.ransac_type    = 'grassmannian'
    configs.best_val_metric = 'recall_rot'

    # ── Training ──────────────────────────────────────────────────────────────
    configs.out_dir            = os.path.join(SCRIPT_DIR, 'output')
    configs.model_nb           = str(date.today()) + '-v2'
    configs.gpu_inds           = 0
    configs.use_gpu            = True
    configs.train_batch_size   = 32
    configs.train_lr           = 1e-4    # lower — already fine-tuned once
    configs.train_weight_decay = 1e-3
    configs.train_momentum     = 0.8
    configs.exp_gamma          = 1.0     # unused (replaced by cosine below)
    configs.train_epoches      = 200
    configs.train_start_epoch  = 0
    configs.train_save_freq_epoch = 1
    configs.val_epoch_freq     = 1
    configs.val_max_iter       = -1
    configs.iter_size          = 1
    configs.optimizer          = 'Adam'
    configs.train_num_thread   = 4
    configs.val_num_thread     = 1
    configs.train_seed         = 42
    configs.print_freq         = 20

    # ── Fine-tune from epoch-42 cross-modal checkpoint ────────────────────────
    configs.weights      = PRETRAINED
    configs.resume_dir   = None
    configs.resume       = None

    dconfig = vars(configs)

    logging.info('===> Cross-modal fine-tuning config (v2, 200×700)')
    for k, v in sorted(dconfig.items()):
        logging.info('    {:30s}: {}'.format(k, v))

    configs = edict(dconfig)

    if configs.train_seed is not None:
        random.seed(configs.train_seed)
        torch.manual_seed(configs.train_seed)
        torch.cuda.manual_seed(configs.train_seed)
        cudnn.deterministic = True

    train_loader = DataLoader(
        PluckerData3D_precompute(phase='train', config=configs),
        batch_size=configs.train_batch_size, shuffle=True,
        drop_last=True, num_workers=configs.train_num_thread,
    )
    val_loader = DataLoader(
        PluckerData3D_precompute(phase='valid', config=configs),
        batch_size=1, shuffle=False,
        drop_last=False, num_workers=configs.val_num_thread,
    )

    trainer = PluckerTrainer(configs, train_loader, val_loader)

    # Replace ExponentialLR with CosineAnnealingLR to prevent post-peak degradation
    trainer.scheduler = optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer,
        T_max=configs.train_epoches,
        eta_min=1e-6,
    )

    trainer.train()


if __name__ == '__main__':
    main()
