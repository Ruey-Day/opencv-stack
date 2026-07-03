"""
Extends the original PlueckerNet trainer to Sim(3):
"""
import os
import os.path as osp
import logging
import json
import gc

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from lib.utils import load_model, ensure_dir, AverageMeter, Timer
from lib.loss import TotalLoss

class Sim3Trainer:

    def __init__(self, config, data_loader, val_data_loader=None):

        Model = load_model('PluckerNetKnn')
        self.model = Model(config)

        logging.info(self.model)

        self.config          = config
        self.max_epoch       = config.train_epoches
        self.save_freq       = config.train_save_freq_epoch
        self.val_max_iter    = config.val_max_iter
        self.val_epoch_freq  = config.val_epoch_freq
        self.best_val_metric = config.best_val_metric
        self.best_val_epoch  = -np.inf
        self.best_val        = -np.inf
        self.curriculum_ir   = 0.0   # current val avg_inlier_ratio driving curriculum

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
        json.dump(config, open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
                  indent=4, sort_keys=False)

        self.iter_size  = config.iter_size
        self.batch_size = data_loader.batch_size
        self.data_loader     = data_loader
        self.val_data_loader = val_data_loader
        self.test_valid      = val_data_loader is not None

        self.model = self.model.to(self.device)
        self.writer = SummaryWriter(logdir=self.checkpoint_dir)

        if config.resume is not None:
            if osp.isfile(config.resume):
                logging.info(f"=> loading checkpoint '{config.resume}'")
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
    
    def train(self):
        if self.test_valid:
            with torch.no_grad():
                val_dict = self._valid_epoch()
            for k, v in val_dict.items():
                if np.isfinite(v):
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
                self.curriculum_ir = val_dict.get('avg_inlier_ratio', 0.0)
                for k, v in val_dict.items():
                    if np.isfinite(v):
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
            'config':          self.config,
            'best_val':        self.best_val,
            'best_val_epoch':  self.best_val_epoch,
            'best_val_metric': self.best_val_metric,
        }
        path = os.path.join(self.checkpoint_dir, f'{filename}.pth')
        logging.info(f'Saving checkpoint: {path}')
        torch.save(state, path)

    def _train_epoch(self, epoch):
        gc.collect()
        ds = self.data_loader.dataset
        if hasattr(ds, 'set_curriculum_phase'):
            ds.set_curriculum_phase(self.curriculum_ir / 100.0)
        self.model.train()
        total_loss, total_num = 0.0, 0.0

        data_loader_iter = iter(self.data_loader)
        iter_size  = self.iter_size
        start_iter = (epoch - 1) * (len(self.data_loader) // iter_size)
        data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()

        for curr_iter in range(len(self.data_loader) // iter_size):
            self.optimizer.zero_grad()
            batch_total_loss = 0.0
            batch_prob_loss  = 0.0
            data_time = 0.0
            total_timer.tic()

            for _ in range(iter_size):
                data_timer.tic()
                matches, plucker1, plucker2, *_ = next(data_loader_iter)
                data_time += data_timer.toc(average=False)

                matches  = matches.to(self.device)
                plucker1 = plucker1.to(self.device)
                plucker2 = plucker2.to(self.device)

                prob_matrix, prior1, prior2 = self.model(plucker1, plucker2)

                MatchLoss = TotalLoss().to(self.device)
                bce_loss  = MatchLoss(prob_matrix, matches)

                loss = bce_loss

                if not torch.isnan(loss).any():
                    loss.backward()

                batch_total_loss += loss.item()
                batch_prob_loss  += (
                    (1.0 - 2.0 * matches) * prob_matrix
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
                logging.info(
                    f'Train Epoch: {epoch} [{curr_iter}/{len(self.data_loader) // iter_size}]'
                    f'  Loss: {batch_total_loss:.3e}'
                    f'  InlierProb: {batch_prob_loss:.3f}'
                    f'  DataT: {data_meter.avg:.4f}'
                    f'  TrainT: {total_timer.avg - data_meter.avg:.4f}'
                )
                data_meter.reset()
                total_timer.reset()
    
    def _valid_epoch(self):
        self.model.eval()
        num_data   = 0
        data_timer = Timer()
        match_timer = Timer()

        tot_num_data = len(self.val_data_loader.dataset)
        if self.val_max_iter > 0:
            tot_num_data = min(self.val_max_iter, tot_num_data)

        data_loader_iter = iter(self.val_data_loader)

        measure_list = ['err_q', 'err_t', 'err_s', 'inlier_ratio']
        eval_res = {m: np.zeros(tot_num_data) for m in measure_list}

        for batch_idx in range(tot_num_data):
            data_timer.tic()
            matches, plucker1, plucker2, R_gt, t_gt, s_gt = next(data_loader_iter)
            data_timer.toc()

            nb_plucker = matches.size(1)
            if nb_plucker > 3000 or nb_plucker < 2:
                continue

            matches      = matches.to(self.device)
            plucker1_raw = plucker1.to(self.device)
            plucker2_raw = plucker2.to(self.device)

            match_timer.tic()
            prob_matrix, prior1, prior2 = self.model(plucker1_raw, plucker2_raw)
            match_timer.toc()

            k = min(100, round(plucker1.size(1) * plucker2.size(1)))

            _, P_topk_i     = torch.topk(prob_matrix.flatten(start_dim=-2), k=k,
                                          dim=-1, largest=True, sorted=True)
            plucker1_indices = P_topk_i // prob_matrix.size(-1)
            plucker2_indices = P_topk_i  % prob_matrix.size(-1)

            # Defaults for failure cases
            err_q        = np.pi
            err_t        = np.inf
            err_s        = np.inf
            inlier_ratio = 0.0

            nb_inliers_gt = np.where(matches[0, :].cpu().numpy() > 0)[0].shape[0]

            if k > 3:
                inlier_inds  = matches[:, plucker1_indices, plucker2_indices].cpu().numpy()
                inlier_ratio = np.sum(inlier_inds) / k * 100.0

            num_data += 1
            torch.cuda.empty_cache()

            eval_res['err_q'][batch_idx]        = err_q
            eval_res['err_t'][batch_idx]        = err_t
            eval_res['err_s'][batch_idx]        = err_s
            eval_res['inlier_ratio'][batch_idx] = inlier_ratio

            logging.info(
                f'Val {num_data}/{tot_num_data} '
                f'DataT: {data_timer.avg:.3f}  MatchT: {match_timer.avg:.3f} '
                f'err_rot: {err_q * 180/np.pi:.2f}°  '
                f'err_t: {err_t:.3f}  '
                f'err_s(log): {err_s:.3f}  '
                f'inlier_ratio: {inlier_ratio:.1f}%  '
                f'nb_matches: {k}  nb_inliers_gt: {nb_inliers_gt}'
            )
            data_timer.reset()

        stats = self._summarise(eval_res)

        logging.info(
            f'med_rot: {stats["med_rot"]:.2f}°  '
            f'med_trans: {stats["med_trans"]:.3f}  '
            f'med_scale_err(log): {stats["med_scale_err"]:.3f}  '
            f'avg_inlier_ratio: {stats["avg_inlier_ratio"]:.1f}%'
        )

        return stats

    def _summarise(self, eval_res: dict) -> dict:
        """Aggregate per-sample eval results into epoch-level stats."""
        n = max(1, (eval_res['inlier_ratio'] > 0).sum())
        return {
            'med_rot':         float(np.median(eval_res['err_q']) * 180 / np.pi),
            'med_trans':       float(np.median(eval_res['err_t'][np.isfinite(eval_res['err_t'])])
                                     if np.any(np.isfinite(eval_res['err_t'])) else np.inf),
            'med_scale_err':   float(np.median(eval_res['err_s'][np.isfinite(eval_res['err_s'])])
                                     if np.any(np.isfinite(eval_res['err_s'])) else np.inf),
            'avg_inlier_ratio': float(np.mean(eval_res['inlier_ratio'])),
        }
