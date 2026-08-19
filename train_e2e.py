"""train_e2e.py — END-TO-END fine-tune (v1): matcher + differentiable Sim(3)
head (lib/sim3_diff.py). STANDALONE entry — deliberately does not modify
train.py/trainer.py (safe to add while another run trains).

Per sample: matcher forward -> P -> BCE(+matchability) as before, PLUS the
top-K weighted pose head -> angle-based pose loss vs GT. Pose loss ramps in
over --pose_warmup epochs. Dataset convention (verified on found6):
(R_gt, t_gt, s_gt) maps plucker2 (query) -> plucker1 (reference); prob rows =
plucker1. Samples with s_gt == 0 (zero-overlap sentinel) skip the pose term.

Judge checkpoints by the REAL benchmarks (cache_matcher + solve_sim3), never
trainer val.

Usage (from third_party/ScalePluckerNet):
  python train_e2e.py --ckpt <warm_start.pth> --dataset synthetic_found6 \
      --name v1_e2e [--pose_w 0.3 --pose_warmup 2 --epochs 30]
"""
import argparse
import logging
import os
import pickle
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.loss import TotalLoss                     # noqa: E402
from lib.model import PluckerNetKnn                # noqa: E402
from lib.sim3_diff import pose_head, pose_loss     # noqa: E402


def load_model(ckpt_path, dev):
    """Build the model from the CHECKPOINT'S OWN config (the §9 loading rule:
    sign_inv/dual_knn/... must come from the checkpoint, never from env)."""
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = PluckerNetKnn(ck['config'])
    model.load_state_dict(ck['state_dict'])
    return model.to(dev), ck['config']


def load_split(base, max_samples=None):
    d = {}
    for k in ('matches', 'plucker1', 'plucker2', 'R_gt', 't_gt', 's_gt'):
        d[k] = pickle.load(open(os.path.join(base, k + '.pkl'), 'rb'))
    n = len(d['plucker1'])
    if max_samples and n > max_samples:
        idx = random.Random(0).sample(range(n), max_samples)
        d = {k: [v[i] for i in idx] for k, v in d.items()}
    return d, len(d['plucker1'])


def prenorm_alpha(l_ref):
    """1 / median foot-point radius of the reference cloud (matches solver)."""
    f = torch.linalg.cross(l_ref[:, 3:], l_ref[:, :3])
    rad = (f - f.median(0).values).norm(dim=1).median()
    return 1.0 / (rad + 1e-9)


def sample_losses(model, loss_fn, d, i, dev, topk, warm, unroll):
    p1 = torch.from_numpy(np.asarray(d['plucker1'][i], np.float32)).to(dev)
    p2 = torch.from_numpy(np.asarray(d['plucker2'][i], np.float32)).to(dev)
    m = np.asarray(d['matches'][i])
    C = torch.zeros(len(p1), len(p2), device=dev)
    C[m[0], m[1]] = 1.0
    P, r, c = model(p1[None], p2[None])
    bce = loss_fn(P, C[None], r, c)[0]
    s_gt = float(d['s_gt'][i])
    if s_gt <= 0 or len(m[0]) < 8:                 # zero-overlap / starved
        return bce, None, None, None
    k = min(topk, P[0].numel())
    flat = P[0].flatten().topk(k).indices
    ir, iq = flat // P.shape[2], flat % P.shape[2]
    w = P[0][ir, iq]
    alpha = prenorm_alpha(p1)
    l_ref, l_qry = p1[ir].clone(), p2[iq].clone()
    l_ref[:, :3] *= alpha
    l_qry[:, :3] *= alpha
    R, t, s = pose_head(l_ref, l_qry, w, warm=warm, unroll=unroll)
    R_gt = torch.from_numpy(np.asarray(d['R_gt'][i], np.float32)).to(dev)
    t_gt = torch.from_numpy(np.asarray(d['t_gt'][i], np.float32)).reshape(3).to(dev) * alpha
    with torch.no_grad():                          # basin gate: flip-locked
        cosang = (((R * R_gt).sum() - 1) / 2).clamp(-1, 1)   # samples give BCE
        if torch.rad2deg(torch.acos(cosang)) > 45:           # only, not wrong-
            return bce, None, None, None                     # basin pose grads
    pl, rot = pose_loss(R, t, s, R_gt, t_gt, torch.tensor(s_gt, device=dev))
    serr = (s / s_gt).log().abs()
    return bce, pl, rot, serr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--dataset', default='synthetic_found6')
    ap.add_argument('--name', default='v1_e2e')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--samples_per_epoch', type=int, default=1000)
    ap.add_argument('--max_samples', type=int, default=60000)
    ap.add_argument('--iter_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--pose_w', type=float, default=0.3)
    ap.add_argument('--pose_warmup', type=float, default=2.0)
    ap.add_argument('--topk', type=int, default=200)
    ap.add_argument('--warm_iters', type=int, default=30)
    ap.add_argument('--unroll', type=int, default=8)
    ap.add_argument('--val_pairs', type=int, default=300)
    a = ap.parse_args()

    out = os.path.join('output', 'e2e', a.name)
    os.makedirs(out, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%m/%d %H:%M:%S',
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(os.path.join(out, 'run.log'))])
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg = load_model(a.ckpt, dev)
    model.train()
    match_w = float(cfg.get('match_w', 0.0))
    loss_fn = TotalLoss(match_w).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    tr, n_tr = load_split(os.path.join('dataset', a.dataset + '_train'), a.max_samples)
    va, n_va = load_split(os.path.join('dataset', a.dataset + '_valid'), a.val_pairs)
    logging.info(f'E2E v1: {n_tr} train / {n_va} val, warm={a.warm_iters} '
                 f'unroll={a.unroll} pose_w={a.pose_w} match_w={match_w}')

    best_val = 1e9
    rng = random.Random(1)
    for ep in range(a.epochs):
        pw = a.pose_w * min(1.0, (ep + 1) / max(a.pose_warmup, 1e-9))
        idxs = [rng.randrange(n_tr) for _ in range(a.samples_per_epoch)]
        opt.zero_grad()
        stats, t0 = [], time.time()
        for j, i in enumerate(idxs):
            try:
                bce, pl, rot, serr = sample_losses(model, loss_fn, tr, i, dev,
                                                   a.topk, a.warm_iters, a.unroll)
            except RuntimeError as e:              # rare SVD failures etc.
                logging.warning(f'sample {i} skipped: {e}')
                continue
            loss = bce if pl is None else bce + pw * pl
            if not torch.isfinite(loss):
                continue
            (loss / a.iter_size).backward()
            stats.append([float(bce), float(pl) if pl is not None else np.nan,
                          float(rot) if rot is not None else np.nan,
                          float(serr) if serr is not None else np.nan])
            if (j + 1) % a.iter_size == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
            if (j + 1) % 100 == 0:
                s_ = np.asarray(stats[-100:])
                logging.info(f'ep {ep} [{j + 1}/{len(idxs)}] pw={pw:.2f} '
                             f'bce={np.nanmean(s_[:, 0]):.4f} '
                             f'pose={np.nanmean(s_[:, 1]):.3f} '
                             f'rot={np.degrees(np.nanmean(s_[:, 2])):.2f}deg '
                             f'slog={np.nanmean(s_[:, 3]):.3f} '
                             f'({(time.time() - t0) / (j + 1):.2f}s/sample)')
        # ── validation: head pose metrics on the val split ──
        model.eval()
        rots, slogs = [], []
        for i in range(n_va):
            try:
                _, pl, rot, serr = sample_losses(model, loss_fn, va, i, dev,
                                                 a.topk, a.warm_iters, 0)
            except RuntimeError:
                continue
            if rot is not None:
                rots.append(float(rot)); slogs.append(float(serr))
        vrot = float(np.degrees(np.median(rots))) if rots else 1e9
        vslog = float(np.median(slogs)) if slogs else 1e9
        vscore = vrot + 50 * vslog
        logging.info(f'== ep {ep} VAL: rot med {vrot:.2f} deg  |log s| med '
                     f'{vslog:.4f}  score {vscore:.2f}')
        state = dict(state_dict=model.state_dict(), config=cfg, epoch=ep)
        torch.save(state, os.path.join(out, 'checkpoint.pth'))
        if vscore < best_val:
            best_val = vscore
            torch.save(state, os.path.join(out, 'best_val_checkpoint.pth'))
            logging.info(f'== saved best (score {vscore:.2f})')
        model.train()
    logging.info(f'== TRAINING COMPLETE: {a.epochs} epochs, best val score '
                 f'{best_val:.2f} -> {os.path.join(out, "best_val_checkpoint.pth")}')


if __name__ == '__main__':
    main()
