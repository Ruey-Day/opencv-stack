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

class _DualWriter:
    """SummaryWriter that also mirrors scalars to Weights & Biases.

    Opt-in via WANDB=1. Every existing self.writer.add_scalar(...) call is
    forwarded unchanged to TensorBoard, so a run WITHOUT WANDB behaves exactly
    as before; wandb is a pure add-on. Requires `wandb login` (or WANDB_API_KEY)
    — scalars are uploaded to wandb.ai.
    """

    def __init__(self, sw, wb=None):
        self._sw, self._wb = sw, wb
        self.spe = None          # iterations per epoch; set by the trainer

    def add_scalar(self, tag, value, global_step=None, *a, **k):
        self._sw.add_scalar(tag, value, global_step, *a, **k)
        if self._wb is None:
            return
        try:
            # TensorBoard keeps a separate step series per tag; wandb has ONE
            # global counter. train/* is logged per ITERATION and val/* per
            # EPOCH, so forwarding global_step raw would stack every val point
            # at the far left -- and epoch 0 logs at -n_steps, which wandb
            # rejects outright. Put both on a shared `epoch` axis instead.
            row = {tag: float(value)}
            if global_step is not None:
                if tag.startswith('train/') and self.spe:
                    row['epoch'] = global_step / float(self.spe) + 1.0
                elif not tag.startswith('train/'):
                    row['epoch'] = float(global_step)
            self._wb.log(row)
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._sw, name)


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
        # Compiled handle used for forward passes only (~1.5x on this
        # launch-overhead-bound model). Checkpoints are always saved from
        # self.model so state_dict keys stay free of the _orig_mod. prefix.
        try:
            self.net = torch.compile(self.model, dynamic=True)
        except Exception as e:
            logging.warning(f'torch.compile unavailable, running eager: {e}')
            self.net = self.model
        _wb = None
        if os.environ.get('WANDB', '0') == '1':
            try:
                import wandb as _wandb
                # DETERMINISTIC id: without one, resume='allow' has nothing to
                # resume and every restart creates a NEW run. The GPU drives a
                # display, so the CUDA watchdog kills runs regularly and the
                # supervisor restarts them -- v65 fragmented into two runs
                # before this was fixed. md5(name) keeps all restarts on ONE
                # wandb run, matching tools/wandb_backfill.py.
                _rn = osp.basename(str(self.checkpoint_dir).rstrip('/'))
                _wandb.init(project=os.environ.get('WANDB_PROJECT', 'scalepluckernet'),
                            name=_rn,
                            id=__import__('hashlib').md5(_rn.encode()).hexdigest()[:8],
                            config=dict(config), resume='allow')
                _wandb.define_metric('epoch')
                _wandb.define_metric('train/*', step_metric='epoch')
                _wandb.define_metric('val/*', step_metric='epoch')
                _wb = _wandb
                logging.info('wandb: streaming to project '
                             + os.environ.get('WANDB_PROJECT', 'scalepluckernet'))
            except Exception as e:
                logging.warning('wandb disabled (%s: %s)' % (type(e).__name__, e))
        self.writer = _DualWriter(SummaryWriter(logdir=self.checkpoint_dir), _wb)

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
        start_iter = (epoch - 1) * max(1, (getattr(self.data_loader, 'epoch_samples', None)
                                           or len(self.data_loader)) // iter_size)
        data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()
        loss_fn = TotalLoss(
            getattr(self.config, 'match_w', 0.0),
            getattr(self.config, 'dustbin_w', 0.0) if getattr(self.config, 'dustbin', False) else 0.0
        ).to(self.device)

        # iter_size counts SAMPLES per optimizer step (not loader batches):
        # with a bucketed loader a batch carries B samples, so we accumulate
        # until >= iter_size samples. B == 1 reproduces the old loop exactly
        # (per-sample loss summed, i.e. sum-reduction over the step).
        n_samples_epoch = getattr(self.data_loader, 'epoch_samples', None) \
            or len(self.data_loader)
        n_steps = max(1, n_samples_epoch // iter_size)
        self.writer.spe = n_steps
        for curr_iter in range(n_steps):
            self.optimizer.zero_grad()
            # accumulate logging stats on-device; sync once per iteration
            batch_total_loss = torch.zeros((), device=self.device)
            batch_prob_loss  = torch.zeros((), device=self.device)
            data_time = 0.0
            total_timer.tic()

            acc = 0
            while acc < iter_size:
                data_timer.tic()
                try:
                    matches, plucker1, plucker2, *_ = next(data_loader_iter)
                except StopIteration:
                    data_loader_iter = iter(self.data_loader)
                    matches, plucker1, plucker2, *_ = next(data_loader_iter)
                data_time += data_timer.toc(average=False)

                matches  = matches.to(self.device, non_blocking=True)
                plucker1 = plucker1.to(self.device, non_blocking=True)
                plucker2 = plucker2.to(self.device, non_blocking=True)
                B = matches.shape[0]

                prob_matrix, prior1, prior2 = self.net(plucker1, plucker2)

                # x B: keep sum-over-samples semantics at any batch size
                loss = loss_fn(prob_matrix, matches, prior1, prior2) * B

                if not torch.isnan(loss).any():
                    loss.backward()

                batch_total_loss += loss.detach().sum()
                batch_prob_loss  += (
                    (1.0 - 2.0 * matches) * prob_matrix.detach()
                ).sum(dim=(-2, -1)).mean() * B
                acc += B

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            batch_total_loss = batch_total_loss.item()
            batch_prob_loss  = batch_prob_loss.item()
            total_loss += batch_total_loss
            total_num  += 1.0
            total_timer.toc()
            data_meter.update(data_time)

            if curr_iter % self.config.print_freq == 0:
                self.writer.add_scalar('train/total_loss', batch_total_loss, start_iter + curr_iter)
                self.writer.add_scalar('train/prob_loss',  batch_prob_loss,  start_iter + curr_iter)
                logging.info(
                    f'Train Epoch: {epoch} [{curr_iter}/{n_steps}]'
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
            prob_matrix, prior1, prior2 = self.net(plucker1_raw, plucker2_raw)
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

            if k > 3:
                inlier_inds  = matches[:, plucker1_indices, plucker2_indices].cpu().numpy()
                inlier_ratio = np.sum(inlier_inds) / k * 100.0

            num_data += 1

            eval_res['err_q'][batch_idx]        = err_q
            eval_res['err_t'][batch_idx]        = err_t
            eval_res['err_s'][batch_idx]        = err_s
            eval_res['inlier_ratio'][batch_idx] = inlier_ratio

            if num_data % 100 == 0 or num_data == tot_num_data:
                logging.info(
                    f'Val {num_data}/{tot_num_data} '
                    f'DataT: {data_timer.avg:.3f}  MatchT: {match_timer.avg:.3f} '
                    f'running_avg_inlier_ratio: '
                    f'{eval_res["inlier_ratio"][:batch_idx + 1].mean():.1f}%'
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
