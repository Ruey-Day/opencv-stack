"""
DustbinTrainer — Phase 4 trainer for PluckerNetKnnDustbin.

Key differences from Sim3Trainer:
  - Data batches include overlap_type (7th element from PartialOverlapData).
  - Loss uses an augmented GT matrix C_aug (N+1 x M+1): dustbin GT is derived
    automatically from unmatched rows/cols of the existing matches matrix.
  - Validation only runs RANSAC on full/sparse overlap samples (s_gt > 0);
    zero-overlap samples (s_gt == 0) are skipped in metric computation.
  - Correspondence selection for val uses P_aug[:N, :M] (ignores dustbin).

Pretrain weight loading is handled by train.py (strict=False) before training.
"""
import os
import os.path as osp
import sys
import logging
import json
import gc

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

_PLUECKERNET = os.path.join(os.path.dirname(__file__), '..', 'PlueckerNet')
sys.path.insert(0, os.path.abspath(_PLUECKERNET))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from lib.utils import load_model
from lib.file import ensure_dir
from lib.timer import AverageMeter, Timer
from lib.loss import TotalLoss
from sim3.ransac import run_ransac_sim3
from sim3.model_dustbin import PluckerNetKnnDustbin


class DustbinTrainer:

    def __init__(self, config, data_loader, val_data_loader=None):
        self.model = PluckerNetKnnDustbin(config)
        logging.info(self.model)

        self.config          = config
        self.max_epoch       = config.train_epoches
        self.save_freq       = config.train_save_freq_epoch
        self.val_max_iter    = config.val_max_iter
        self.val_epoch_freq  = config.val_epoch_freq
        self.best_val_metric = config.best_val_metric
        self.best_val_epoch  = -np.inf
        self.best_val        = -np.inf

        if config.use_gpu and not torch.cuda.is_available():
            raise ValueError('GPU not available but cuda flag set')

        if config.gpu_inds > -1:
            torch.cuda.set_device(config.gpu_inds)
            self.device = torch.device('cuda', config.gpu_inds)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.optimizer = getattr(optim, config.optimizer)(
            self.model.parameters(), lr=config.train_lr, betas=(0.9, 0.999)
        )
        self.scheduler    = optim.lr_scheduler.ExponentialLR(self.optimizer, config.exp_gamma)
        self.start_epoch  = config.train_start_epoch
        self.checkpoint_dir = os.path.join(config.out_dir, config.dataset, config.model_nb)

        ensure_dir(self.checkpoint_dir)
        json.dump(dict(config), open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
                  indent=4, sort_keys=False)

        self.iter_size       = config.iter_size
        self.batch_size      = data_loader.batch_size
        self.data_loader     = data_loader
        self.val_data_loader = val_data_loader
        self.test_valid      = val_data_loader is not None

        self.model = self.model.to(self.device)
        self.writer = SummaryWriter(logdir=self.checkpoint_dir)

        if getattr(config, 'resume', None):
            if osp.isfile(config.resume):
                logging.info(f"=> resuming checkpoint '{config.resume}'")
                state = torch.load(config.resume, weights_only=False)
                self.start_epoch = state['epoch']
                self.model.load_state_dict(state['state_dict'])
                self.scheduler.load_state_dict(state['scheduler'])
                self.optimizer.load_state_dict(state['optimizer'])
                if 'best_val' in state:
                    self.best_val       = state['best_val']
                    self.best_val_epoch = state['best_val_epoch']
                    self.best_val_metric = state['best_val_metric']
            else:
                raise ValueError(f"No checkpoint found at '{config.resume}'")

    # ------------------------------------------------------------------
    # GT construction for augmented dustbin matrix
    # ------------------------------------------------------------------

    @staticmethod
    def _build_C_aug(matches):
        """Build the (N+1)×(M+1) GT matrix for dustbin loss.

        Args:
            matches: (B, N, M) float32 GT correspondence matrix

        Returns:
            C_aug: (B, N+1, M+1) — matches block, plus dustbin col/row.
        """
        B, N, M = matches.shape
        device  = matches.device

        # Unmatched lines: row or col with no positive
        row_has_match = matches.sum(dim=-1, keepdim=True).clamp(max=1.0)  # (B, N, 1)
        col_has_match = matches.sum(dim=-2, keepdim=True).clamp(max=1.0)  # (B, 1, M)

        dustbin_col = 1.0 - row_has_match   # (B, N, 1)  — 1 if line i is unmatched
        dustbin_row = 1.0 - col_has_match   # (B, 1, M)  — 1 if line j is unmatched
        corner      = torch.zeros(B, 1, 1, device=device)  # dustbin-to-dustbin: not supervised

        C_aug = torch.cat(
            [torch.cat([matches,     dustbin_col], dim=-1),   # (B, N,   M+1)
             torch.cat([dustbin_row, corner      ], dim=-1)], # (B, 1,   M+1)
            dim=-2                                             # (B, N+1, M+1)
        )
        return C_aug

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self):
        if self.test_valid:
            with torch.no_grad():
                val_dict = self._valid_epoch()
            for k, v in val_dict.items():
                self.writer.add_scalar(f'val/{k}', v, 0)

        for epoch in range(self.start_epoch, self.max_epoch + 1):
            lr = self.scheduler.get_last_lr()
            logging.info(f' Epoch: {epoch}, LR: {lr}')
            self._train_epoch(epoch)
            self._save_checkpoint(epoch)
            self.scheduler.step()

            if self.test_valid and epoch % self.val_epoch_freq == 0:
                with torch.no_grad():
                    val_dict = self._valid_epoch()
                for k, v in val_dict.items():
                    self.writer.add_scalar(f'val/{k}', v, epoch)
                if self.best_val < val_dict[self.best_val_metric]:
                    logging.info(
                        f'Saving best val model — '
                        f'{self.best_val_metric}: {val_dict[self.best_val_metric]:.3f}'
                    )
                    self.best_val       = val_dict[self.best_val_metric]
                    self.best_val_epoch = epoch
                    self._save_checkpoint(epoch, 'best_val_checkpoint')
                else:
                    logging.info(
                        f'Current best {self.best_val_metric}: '
                        f'{self.best_val:.3f} at epoch {self.best_val_epoch}'
                    )

    def _save_checkpoint(self, epoch, filename='checkpoint'):
        state = {
            'epoch':           epoch,
            'state_dict':      self.model.state_dict(),
            'optimizer':       self.optimizer.state_dict(),
            'scheduler':       self.scheduler.state_dict(),
            'config':          dict(self.config),
            'best_val':        self.best_val,
            'best_val_epoch':  self.best_val_epoch,
            'best_val_metric': self.best_val_metric,
        }
        path = os.path.join(self.checkpoint_dir, f'{filename}.pth')
        logging.info(f'Saving checkpoint: {path}')
        torch.save(state, path)

    def _train_epoch(self, epoch):
        gc.collect()
        self.model.train()
        total_loss, total_num = 0.0, 0.0

        data_loader_iter = iter(self.data_loader)
        iter_size  = self.iter_size
        start_iter = (epoch - 1) * (len(self.data_loader) // iter_size)
        data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()

        for curr_iter in range(len(self.data_loader) // iter_size):
            self.optimizer.zero_grad()
            batch_total_loss  = 0.0
            batch_prob_loss   = 0.0
            data_time         = 0.0
            total_timer.tic()

            for _ in range(iter_size):
                data_timer.tic()
                matches, plucker1, plucker2, R_gt, t_gt, s_gt = \
                    next(data_loader_iter)
                data_time += data_timer.toc(average=False)

                matches  = matches.to(self.device)
                plucker1 = plucker1.to(self.device)
                plucker2 = plucker2.to(self.device)

                # Forward → (B, N+1, M+1)
                P_aug, prior1, prior2 = self.model(plucker1, plucker2)

                # Build augmented GT
                C_aug = self._build_C_aug(matches)

                MatchLoss = TotalLoss().to(self.device)
                loss = MatchLoss(P_aug, C_aug)

                if not torch.isnan(loss).any():
                    loss.backward()

                batch_total_loss += loss.item()
                # InlierProb: only on the main block (not dustbin row/col)
                B, Np1, Mp1 = P_aug.shape
                batch_prob_loss += (
                    (1.0 - 2.0 * matches) * P_aug[:, :Np1-1, :Mp1-1]
                ).sum(dim=(-2, -1)).mean()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            torch.cuda.empty_cache()

            total_loss += batch_total_loss
            total_num  += 1.0
            total_timer.toc()
            data_meter.update(data_time)

            if curr_iter % self.config.print_freq == 0:
                self.writer.add_scalar('train/total_loss', batch_total_loss, start_iter + curr_iter)
                self.writer.add_scalar('train/prob_loss',  batch_prob_loss,  start_iter + curr_iter)
                # Log dustbin parameters
                bd = float(self.model.bin_dist.abs().item())
                bs = float(torch.sigmoid(self.model.bin_score_logit).item())
                self.writer.add_scalar('train/bin_dist',  bd, start_iter + curr_iter)
                self.writer.add_scalar('train/bin_score', bs, start_iter + curr_iter)
                logging.info(
                    f'Train Epoch: {epoch} [{curr_iter}/{len(self.data_loader) // iter_size}]'
                    f'  Loss: {batch_total_loss:.3e}'
                    f'  InlierProb: {batch_prob_loss:.3f}'
                    f'  bin_dist: {bd:.3f}  bin_score: {bs:.3f}'
                    f'  DataT: {data_meter.avg:.4f}'
                    f'  TrainT: {total_timer.avg - data_meter.avg:.4f}'
                )
                data_meter.reset()
                total_timer.reset()

    # ------------------------------------------------------------------
    # Validation — only runs RANSAC on overlapping samples
    # ------------------------------------------------------------------

    def _valid_epoch(self):
        self.model.eval()
        num_data    = 0
        data_timer  = Timer()
        match_timer = Timer()

        tot_num_data = len(self.val_data_loader.dataset)
        if self.val_max_iter > 0:
            tot_num_data = min(self.val_max_iter, tot_num_data)

        data_loader_iter = iter(self.val_data_loader)

        measure_list = ['err_q', 'err_t', 'err_s', 'inlier_ratio']
        eval_res = {m: [] for m in measure_list}

        for batch_idx in range(tot_num_data):
            data_timer.tic()
            matches, plucker1, plucker2, R_gt, t_gt, s_gt = \
                next(data_loader_iter)
            data_timer.toc()

            # Skip zero-overlap samples (no GT pose to evaluate)
            if float(s_gt[0].item()) == 0.0:
                continue

            nb_plucker = matches.size(1)
            if nb_plucker > 3000 or nb_plucker < 2:
                continue

            matches      = matches.to(self.device)
            plucker1_raw = plucker1.to(self.device)
            plucker2_raw = plucker2.to(self.device)

            match_timer.tic()
            P_aug, prior1, prior2 = self.model(plucker1_raw, plucker2_raw)
            match_timer.toc()

            # Use only the main block (ignore dustbin row/col) for top-k selection
            B, Np1, Mp1 = P_aug.shape
            P_main = P_aug[:, :Np1-1, :Mp1-1]   # (B, N, M)

            k = min(100, round(plucker1.size(1) * plucker2.size(1)))

            _, P_topk_i      = torch.topk(P_main.flatten(start_dim=-2), k=k,
                                           dim=-1, largest=True, sorted=True)
            plucker1_indices = P_topk_i // P_main.size(-1)
            plucker2_indices = P_topk_i  % P_main.size(-1)

            err_q        = np.pi
            err_t        = np.inf
            err_s        = np.inf
            inlier_ratio = 0.0

            if k > 3:
                inlier_inds  = matches[:, plucker1_indices, plucker2_indices].cpu().numpy()
                inlier_ratio = np.sum(inlier_inds) / k * 100.0

                p1K = plucker1_raw[0, plucker1_indices[0, :k], :].cpu().numpy()
                p2K = plucker2_raw[0, plucker2_indices[0, :k], :].cpu().numpy()

                best_s, best_rot, best_trans, best_ic, best_ic_mask = run_ransac_sim3(
                    p1K.T, p2K.T, inlier_threshold=0.1,
                )

                if best_rot is not None and best_trans is not None and best_s is not None:
                    cos_angle = (np.trace(best_rot.T @ R_gt[0].numpy()) - 1.0) / 2.0
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    err_q  = np.arccos(cos_angle)
                    err_t  = np.linalg.norm(best_trans.flatten() - t_gt.numpy().flatten())
                    s_gt_v = float(s_gt[0].item())
                    if best_s > 0 and s_gt_v > 0:
                        err_s = abs(np.log(best_s) - np.log(s_gt_v))

            num_data += 1
            torch.cuda.empty_cache()

            eval_res['err_q'].append(err_q)
            eval_res['err_t'].append(err_t)
            eval_res['err_s'].append(err_s)
            eval_res['inlier_ratio'].append(inlier_ratio)

            logging.info(
                f'Val {num_data}/{tot_num_data} '
                f'DataT: {data_timer.avg:.3f}  MatchT: {match_timer.avg:.3f} '
                f'err_rot: {err_q * 180/np.pi:.2f}°  '
                f'inlier_ratio: {inlier_ratio:.1f}%'
            )
            data_timer.reset()

        if not eval_res['err_q']:
            return {'recall_rot': 0.0, 'med_rot': 180.0,
                    'med_trans': np.inf, 'med_scale_err': np.inf, 'avg_inlier_ratio': 0.0}

        recall = self._recalls(eval_res)

        logging.info(
            f'recall_rot: {recall[0]:.3f}  '
            f'med_rot: {recall[1]:.2f}°  '
            f'med_trans: {recall[2]:.3f}  '
            f'med_scale_err(log): {recall[3]:.3f}  '
            f'avg_inlier_ratio: {recall[4]:.1f}%'
        )

        return {
            'recall_rot':       recall[0],
            'med_rot':          recall[1],
            'med_trans':        recall[2],
            'med_scale_err':    recall[3],
            'avg_inlier_ratio': recall[4],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _recalls(self, eval_res):
        ths = np.arange(7) * 5
        cur_err_q = np.array(eval_res['err_q']) * 180.0 / np.pi

        q_acc_hist, _ = np.histogram(cur_err_q, ths)
        num_pair = float(len(cur_err_q))
        q_acc_hist = q_acc_hist.astype(float) / num_pair
        q_acc = np.cumsum(q_acc_hist)

        recall_rot       = np.mean(q_acc[:4])
        med_rot          = np.median(cur_err_q)
        med_trans        = np.median(eval_res['err_t'])
        finite_s = [v for v in eval_res['err_s'] if np.isfinite(v)]
        med_scale_err    = np.median(finite_s) if finite_s else np.inf
        avg_inlier_ratio = np.mean(eval_res['inlier_ratio'])

        return recall_rot, med_rot, med_trans, med_scale_err, avg_inlier_ratio
