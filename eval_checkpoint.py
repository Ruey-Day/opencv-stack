#!/usr/bin/env python3
"""
eval_checkpoint.py
==================
Re-evaluate a saved checkpoint against the val set using the L2 Sim3 RANSAC
(sim3/ransac.py) instead of the Grassmannian RANSAC.

Usage:
    python eval_checkpoint.py \
        --checkpoint output/slam_map/2026-05-19-slam-maps-v3/best_val_checkpoint.pth \
        --threshold 0.5 \
        --n_iter 500

Sweeps over a few thresholds if --threshold is not given.
"""
import os, sys, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "PlueckerNet"))
sys.path.insert(0, SCRIPT_DIR)

from sim3.dataloader import Sim3PluckerData
from sim3.ransac import run_ransac_sim3
from lib.utils import load_model


def rotation_error_deg(R_est, R_gt):
    cos = np.clip((np.trace(R_est.T @ R_gt) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def eval_one(model, val_loader, device, threshold, n_iter, max_pairs=800):
    model.eval()
    errs_q, errs_s, inlier_ratios = [], [], []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_pairs:
                break
            matches, plucker1, plucker2, R_gt, t_gt, s_gt = batch
            matches   = matches.to(device)
            plucker1  = plucker1.to(device)
            plucker2  = plucker2.to(device)

            prob_matrix, _, _ = model(plucker1, plucker2)

            k = min(100, plucker1.size(1) * plucker2.size(1))
            _, topk_flat = torch.topk(prob_matrix.flatten(start_dim=-2), k=k, dim=-1)
            idx1 = (topk_flat // prob_matrix.size(-1))[0].cpu().numpy()
            idx2 = (topk_flat  % prob_matrix.size(-1))[0].cpu().numpy()

            p1 = plucker1[0, idx1, :].cpu().numpy().T   # (6, k)
            p2 = plucker2[0, idx2, :].cpu().numpy().T   # (6, k)

            inlier_gt   = matches[:, idx1, idx2].cpu().numpy()
            inlier_ratio = float(np.sum(inlier_gt)) / k * 100.0
            inlier_ratios.append(inlier_ratio)

            s_est, R_est, t_est, _, _ = run_ransac_sim3(
                p1, p2, max_iterations=n_iter, inlier_threshold=threshold
            )

            s_gt_val = float(s_gt[0].item())
            if R_est is not None and s_est is not None and s_est > 0 and s_gt_val > 0:
                errs_q.append(rotation_error_deg(R_est, R_gt[0].numpy()))
                errs_s.append(abs(np.log(s_est) - np.log(s_gt_val)))
            else:
                errs_q.append(180.0)
                errs_s.append(np.inf)

    errs_q  = np.array(errs_q)
    fin_s   = np.array([e for e in errs_s if np.isfinite(e)])

    ths     = np.arange(7) * 5          # [0, 5, 10, 15, 20, 25, 30]
    hist, _ = np.histogram(errs_q, ths)
    q_acc   = np.cumsum(hist.astype(float) / len(errs_q))
    recall  = float(np.mean(q_acc[:4]))  # mean of <5°, <10°, <15°, <20°

    return {
        'recall_rot':       recall,
        'med_rot':          float(np.median(errs_q)),
        'pct_lt5':          float(q_acc[0]) if len(q_acc) > 0 else 0,
        'pct_lt10':         float(q_acc[1]) if len(q_acc) > 1 else 0,
        'pct_lt20':         float(q_acc[3]) if len(q_acc) > 3 else 0,
        'med_scale_err':    float(np.median(fin_s)) if len(fin_s) else float('inf'),
        'avg_inlier_ratio': float(np.mean(inlier_ratios)),
        'n_pairs':          len(errs_q),
    }


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--threshold', type=float, default=None)
    ap.add_argument('--n_iter',    type=int,   default=500)
    ap.add_argument('--max_pairs', type=int,   default=800)
    args, _ = ap.parse_known_args()

    thresholds = [args.threshold] if args.threshold else [0.1, 0.3, 0.5, 1.0]

    # Build config directly — avoids get_config()'s argparse clash
    configs = edict(dict(
        dataset='slam_map',
        data_dir=os.path.join(SCRIPT_DIR, 'dataset'),
        val_phase='valid',
        model='KNNContextNormNet',
        net_depth=12,
        net_nchannel=128,
        GNN_layers=['self', 'cross'] * 6,
        net_gcnorm=True,
        net_batchnorm=True,
        net_topK=2000,
        net_lambda=0.1,
        net_maxiter=30,
        net_knn=10,
        in_channel=6,
        normalize_n_lines=200,
        normalize_n_inliers=60,
        gpu_inds=0,
        use_gpu=True,
    ))

    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')

    Model = load_model('PluckerNetKnn')
    model = Model(configs).to(device)

    state = torch.load(args.checkpoint, weights_only=False, map_location=device)
    sd = state.get('state_dict', state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:   print(f'Missing keys: {missing[:3]}')
    if unexpected: print(f'Unexpected keys: {unexpected[:3]}')
    print(f'Loaded checkpoint: {args.checkpoint}')

    val_loader = DataLoader(
        Sim3PluckerData(phase='valid', config=configs),
        batch_size=1, shuffle=False, drop_last=False, num_workers=1,
    )

    print(f'\n{"threshold":>10s}  {"recall_rot":>10s}  {"<5°":>6s}  {"<10°":>6s}  {"<20°":>6s}  {"med_rot":>8s}  {"med_s_err":>9s}  {"inlier%":>7s}')
    print('-' * 80)
    for thr in thresholds:
        res = eval_one(model, val_loader, device, thr, args.n_iter, args.max_pairs)
        print(f'{thr:>10.3f}  {res["recall_rot"]:>10.3f}  '
              f'{res["pct_lt5"]:>6.3f}  {res["pct_lt10"]:>6.3f}  {res["pct_lt20"]:>6.3f}  '
              f'{res["med_rot"]:>8.1f}°  {res["med_scale_err"]:>9.3f}  '
              f'{res["avg_inlier_ratio"]:>7.1f}%  (n={res["n_pairs"]})')


if __name__ == '__main__':
    main()
