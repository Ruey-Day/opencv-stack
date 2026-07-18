"""
register.py — register two Structure-PLP-SLAM line maps with SCALAR:
ScalePluckerNet correspondences + the line-based Sim(3) solver
(lib/sim3_solver.py), recovering scale, rotation, and translation.

Typical use: mono map (arbitrary scale) -> RGB-D / LiDAR metric map.

Usage:
    python register.py \
        --db_src  path/to/mono_map.db \
        --db_tgt  path/to/metric_map.db \
        --checkpoint output/synthetic_v6/synthetic_v6/best_val_checkpoint.pth

    # classical max-inlier RANSAC baseline instead of the solver:
    python register.py ... --ransac

Output: estimated scale / rotation / translation (+ % scale error if
--gt_scale is given).
"""
import argparse
import os
import sys

import msgpack
import numpy as np
import torch
from easydict import EasyDict as edict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from lib.sim3_solver import Sim3Solver, ransac_sim3, segments_to_plucker
from lib.utils import load_model


def load_endpoints_from_db(db_path: str):
    """Line landmark endpoints from a Structure-PLP-SLAM map: (N,3),(N,3)."""
    with open(db_path, "rb") as f:
        data = msgpack.unpack(f, raw=False)
    p1s, p2s = [], []
    for lm in data.get("landmarks_line", {}).values():
        pw = lm.get("pos_w") or lm.get("pos")
        if pw is None or len(pw) < 6:
            continue
        p1 = np.asarray(pw[:3], np.float32)
        p2 = np.asarray(pw[3:6], np.float32)
        if np.linalg.norm(p2 - p1) < 0.01:
            continue
        p1s.append(p1)
        p2s.append(p2)
    if not p1s:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    return np.stack(p1s), np.stack(p2s)


def load_network(ckpt_path: str, device: torch.device):
    configs = edict(dict(
        model='KNNContextNormNet', net_depth=12, net_nchannel=128,
        GNN_layers=['self', 'cross'] * 6, net_gcnorm=True, net_batchnorm=True,
        net_topK=2000, net_lambda=0.1, net_maxiter=30, net_knn=10,
        in_channel=6, normalize_n_lines=200, normalize_n_inliers=60,
        gpu_inds=0, use_gpu=True,
    ))
    Model = load_model('PluckerNetKnn')
    model = Model(configs).to(device)
    state = torch.load(ckpt_path, weights_only=False, map_location=device)
    sd = state.get('state_dict', state)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def register(db_src, db_tgt, ckpt_path, topk=200, max_query=400,
             use_ransac=False, device=None):
    """Register src (query, scale-ambiguous) onto tgt (metric reference).
    Returns (s, R, t, info) in the maps' original coordinates."""
    if device is None:
        device = (torch.device('cuda', 0) if torch.cuda.is_available()
                  else torch.device('cpu'))
    q1, q2 = load_endpoints_from_db(db_src)
    r1, r2 = load_endpoints_from_db(db_tgt)
    print(f'query map: {len(q1)} lines   ref map: {len(r1)} lines')
    if len(q1) < 10 or len(r1) < 10:
        raise SystemExit(f'too few lines: query {len(q1)}, ref {len(r1)}')
    if len(q1) > max_query:
        k = np.random.default_rng(1).choice(len(q1), max_query, replace=False)
        q1, q2 = q1[k], q2[k]

    solver = Sim3Solver((q1, q2), (r1, r2), device=device)
    # matcher forward pass on pre-normalized, moment-whitened Plücker input
    # (exactly the benchmark protocol of tools/bench_sim3_solver.py)
    p_q = segments_to_plucker(q1 * solver.alpha, q2 * solver.alpha)
    p_r = segments_to_plucker(r1 * solver.alpha, r2 * solver.alpha)
    std = float(p_q[:, :3].std()) + 1e-6
    nrm = lambda x: np.concatenate([x[:, :3] / std, x[:, 3:]],
                                   1).astype(np.float32)
    model = load_network(ckpt_path, device)
    with torch.no_grad():
        prob, _, _ = model(torch.from_numpy(nrm(p_r)[None]).to(device),
                           torch.from_numpy(nrm(p_q)[None]).to(device))
    prob = prob[0]

    if use_ransac:
        # classical max-inlier baseline over the top-k candidates
        pt = torch.as_tensor(prob)
        _, flat = torch.topk(pt.flatten(), k=min(topk, pt.numel()))
        ir = (flat // pt.size(-1)).cpu().numpy()
        iq = (flat % pt.size(-1)).cpu().numpy()
        L1 = segments_to_plucker(q1, q2)[iq].T.astype(float)
        L2 = segments_to_plucker(r1, r2)[ir].T.astype(float)
        R, t, s, _, n_inl = ransac_sim3(L1, L2)
        return s, R, t, dict(n_inliers=n_inl)

    s, R, t, info = solver.register(prob=prob, topk=topk)
    return s, R, t, info


def main():
    ap = argparse.ArgumentParser(
        description='Register two SLAM line maps with SCALAR.')
    ap.add_argument('--db_src', required=True,
                    help='query map .db (scale-ambiguous, e.g. mono)')
    ap.add_argument('--db_tgt', required=True,
                    help='metric reference map .db (RGB-D / LiDAR)')
    ap.add_argument('--checkpoint', required=True,
                    help='ScalePluckerNet checkpoint .pth')
    ap.add_argument('--topk', type=int, default=200,
                    help='top-K matcher correspondences fed to the solver')
    ap.add_argument('--ransac', action='store_true',
                    help='use the max-inlier RANSAC baseline instead')
    ap.add_argument('--gt_scale', type=float, default=None,
                    help='GT scale for error report')
    args = ap.parse_args()

    s, R, t, info = register(args.db_src, args.db_tgt, args.checkpoint,
                             topk=args.topk, use_ransac=args.ransac)
    if s is None:
        raise SystemExit('registration failed')
    np.set_printoptions(precision=4, suppress=True)
    print(f'scale s = {s:.4f}')
    print(f'R =\n{np.asarray(R)}')
    print(f't = {np.asarray(t).reshape(3)}')
    if args.gt_scale:
        print(f'scale error vs GT: '
              f'{abs(s - args.gt_scale) / args.gt_scale * 100:.1f}%')


if __name__ == '__main__':
    main()
