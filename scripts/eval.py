#!/usr/bin/env python3
"""
eval.py — Unified evaluation for ScalePlueckerNet.

Modes
-----
  (default)      Cross-dataset: evaluate a checkpoint on one or more val splits
  --chess        Chess B1/B2 benchmark (real RGBD + simulated scale ambiguity)
  --hypothesis   Synthetic Sim3/SE3 hypothesis test

Examples
--------
  python scripts/eval.py --weights output/joint/2026-05-09/best_val_checkpoint.pth \
      --dataset replica_gs

  python scripts/eval.py --weights output/joint/2026-05-09/best_val_checkpoint.pth \
      --dataset replica_gs,7scenes_gs,se3real_sim3

  python scripts/eval.py --chess \
      --weights output/joint_color/2026-05-10/best_val_checkpoint.pth

  python scripts/eval.py --hypothesis \
      --weights output/se3real_sim3/2026-05-08/best_val_checkpoint.pth \
      [--se3_weights /path/to/se3/best_val_checkpoint_real.pth]
"""

import os
import sys
import argparse
import json
import time
import warnings
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

SCRIPTS_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT         = os.path.dirname(SCRIPTS_DIR)
PLUECKERNET  = os.path.abspath(os.path.join(ROOT, '..', 'PlueckerNet'))
sys.path.insert(0, PLUECKERNET)
sys.path.insert(0, ROOT)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ══════════════════════════════════════════════════════════════════════════════
# Shared: model wrappers + loading
# ══════════════════════════════════════════════════════════════════════════════

class _DustbinWrapper(nn.Module):
    """Strips dustbin row/col from P_aug so Sim3Trainer._valid_epoch() works unchanged."""
    def __init__(self, model):
        super().__init__()
        self.inner = model

    def forward(self, p1, p2):
        P_aug, r, c = self.inner(p1, p2)
        return P_aug[:, :-1, :-1], r, c


_NEUTRAL_LAB = torch.tensor([50.0, 0.0, 0.0])


class _Pad6Dto9DWrapper(nn.Module):
    """Pads 6D Plücker inputs to 9D with neutral LAB so a 9D model evals on 6D datasets."""
    def __init__(self, model):
        super().__init__()
        self.inner = model

    def forward(self, p1, p2):
        if p1.shape[-1] == 6:
            pad = _NEUTRAL_LAB.to(p1.device).view(1, 1, 3)
            p1 = torch.cat([p1, pad.expand(p1.shape[0], p1.shape[1], 3)], dim=-1)
            p2 = torch.cat([p2, pad.expand(p2.shape[0], p2.shape[1], 3)], dim=-1)
        return self.inner(p1, p2)


def _load_model(configs, weights_path):
    """Load PluckerNetKnn or PluckerNetKnnDustbin from checkpoint."""
    ckpt = torch.load(weights_path, weights_only=False)
    sd   = ckpt.get('state_dict', ckpt.get('model', ckpt))
    is_dustbin = any('bin_dist' in k or 'bin_score' in k for k in sd)

    if is_dustbin:
        from sim3.model_dustbin import PluckerNetKnnDustbin
        model = PluckerNetKnnDustbin(configs)
        model.load_state_dict(sd, strict=True)
        print('  (dustbin checkpoint — wrapping for eval)')
        return _DustbinWrapper(model)
    else:
        from lib.utils import load_model
        Model = load_model('PluckerNetKnn')
        model = Model(configs)
        model.load_state_dict(sd)
        return model


# ══════════════════════════════════════════════════════════════════════════════
# Shared: metrics
# ══════════════════════════════════════════════════════════════════════════════

def rot_err_deg(R_est, R_gt):
    tr = np.clip((np.trace(R_est @ R_gt.T) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(tr)))


def scale_err_log(s_est, s_gt):
    return float(abs(np.log(max(float(s_est), 1e-6)) - np.log(max(float(s_gt), 1e-6))))


def trans_err(t_est, t_gt):
    return float(np.linalg.norm(np.asarray(t_est).flatten() - np.asarray(t_gt).flatten()))


# ══════════════════════════════════════════════════════════════════════════════
# Mode 1: cross-dataset eval
# ══════════════════════════════════════════════════════════════════════════════

def cross_dataset_eval(weights_path, datasets, data_dir, out_dir, label=None):
    from easydict import EasyDict as edict
    from torch.utils.data import DataLoader
    from sim3.dataloader import Sim3PluckerData
    from sim3.trainer import Sim3Trainer

    ckpt = torch.load(weights_path, weights_only=False)
    configs = ckpt.get('config')
    if configs is None:
        raise ValueError('Checkpoint has no "config" key — cannot infer model settings.')
    if not isinstance(configs, edict):
        configs = edict(configs)

    results = {}
    for dataset in datasets:
        run_label = label or f"{os.path.basename(os.path.dirname(weights_path))} → {dataset}"
        print(f"\n{'='*60}")
        print(f"Cross-dataset eval: {run_label}")
        print(f"  weights : {weights_path}")
        print(f"  dataset : {dataset}_valid  ({data_dir})")
        print(f"{'='*60}\n")

        cfg = edict(dict(configs))
        cfg.dataset  = dataset
        cfg.data_dir = data_dir
        cfg.resume   = None
        cfg.weights  = None
        cfg.model_nb = 'eval'

        val_loader = DataLoader(
            Sim3PluckerData(phase='valid', config=cfg),
            batch_size=1, shuffle=False, drop_last=False, num_workers=2,
        )
        if len(val_loader.dataset) == 0:
            print(f"ERROR: {dataset}_valid is empty — skipping.")
            continue

        print(f"Validation set: {len(val_loader.dataset)} scenes")

        model = _load_model(cfg, weights_path)

        checkpoint_channels = getattr(configs, 'in_channel', 6)
        sample = next(iter(val_loader))[1]
        data_channels = min(sample.shape[1], sample.shape[2])
        if checkpoint_channels == 9 and data_channels == 6:
            print('  (9D checkpoint on 6D dataset — padding with neutral LAB)')
            model = _Pad6Dto9DWrapper(model)

        trainer = Sim3Trainer(cfg, data_loader=val_loader, val_data_loader=val_loader)
        trainer.model = model.to(trainer.device)
        trainer.model.eval()
        with torch.no_grad():
            metrics = trainer._valid_epoch()

        os.makedirs(out_dir, exist_ok=True)
        safe = run_label.replace(' ', '_').replace('/', '-').replace('→', 'on')
        out_path = os.path.join(out_dir, f'{safe}.json')
        with open(out_path, 'w') as f:
            json.dump({'label': run_label, 'weights': weights_path,
                       'dataset': dataset, 'metrics': metrics}, f, indent=2)
        results[dataset] = metrics

    # Summary table
    if results:
        print(f"\n{'='*75}")
        print(f"{'Dataset':<20} {'recall_rot':>10} {'med_rot (°)':>12} {'med_trans (m)':>14} {'inlier_ratio':>13}")
        print(f"{'-'*75}")
        for ds, m in results.items():
            print(f"{ds:<20} {m['recall_rot']:>10.3f} {m['med_rot']:>12.2f} "
                  f"{m['med_trans']:>14.3f} {m['avg_inlier_ratio']:>12.1f}%")
        print(f"{'='*75}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Mode 2: Chess B1/B2 benchmark
# ══════════════════════════════════════════════════════════════════════════════

CHESS_FX, CHESS_FY, CHESS_CX, CHESS_CY = 525.0, 525.0, 319.5, 239.5
CHESS_DEPTH_SCALE = 1000.0


def _load_chess_frames(seq_dir, n_frames=25, frame_step=40):
    import cv2, glob
    depth_files = sorted(glob.glob(os.path.join(seq_dir, '*.depth.png')))[::frame_step][:n_frames]
    frames = []
    for df in depth_files:
        pf = df.replace('.depth.png', '.pose.txt')
        cf = df.replace('.depth.png', '.color.png')
        if not os.path.exists(pf) or not os.path.exists(cf):
            continue
        depth = cv2.imread(df, cv2.IMREAD_ANYDEPTH).astype(np.float32) / CHESS_DEPTH_SCALE
        pose  = np.loadtxt(pf)
        color = cv2.cvtColor(cv2.imread(cf), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames.append((depth, color, pose))
    print(f"  {len(frames)} frames from {os.path.basename(seq_dir)}")
    return frames


def _build_colored_cloud(frames, subsample=4, max_depth=3.5, voxel=0.025):
    xyz_all, rgb_all = [], []
    for depth, color, pose in frames:
        H, W = depth.shape
        vi, ui = np.meshgrid(np.arange(0, H, subsample),
                             np.arange(0, W, subsample), indexing='ij')
        vi, ui = vi.ravel(), ui.ravel()
        z = depth[vi, ui]
        ok = (z > 0.1) & (z < max_depth)
        z, vi, ui = z[ok], vi[ok], ui[ok]
        cam = np.stack([(ui - CHESS_CX) * z / CHESS_FX,
                        (vi - CHESS_CY) * z / CHESS_FY,
                        z, np.ones_like(z)], 0)
        xyz = (pose @ cam)[:3].T
        xyz_all.append(xyz)
        rgb_all.append(color[vi, ui])
    xyz = np.concatenate(xyz_all)
    rgb = np.concatenate(rgb_all)
    keys = np.floor(xyz / voxel).astype(np.int32)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return xyz[idx], rgb[idx]


def _extract_lines_9d(xyz, rgb, n_lines=300, k=20, linearity_thresh=0.72, seed=42):
    from scipy.spatial import cKDTree
    rng  = np.random.default_rng(seed)
    tree = cKDTree(xyz)
    mids, dirs, colors = [], [], []
    for idx in rng.choice(len(xyz), size=min(15000, len(xyz)), replace=False):
        if len(mids) >= n_lines:
            break
        nn  = tree.query(xyz[idx], k=k)[1]
        pts = xyz[nn]
        ctr = pts.mean(0)
        cov = (pts - ctr).T @ (pts - ctr) / k
        ev, evec = np.linalg.eigh(cov)
        lam = ev[::-1]
        if (lam[0] - lam[1]) / (lam[0] + 1e-9) > linearity_thresh:
            mids.append(ctr)
            dirs.append(evec[:, -1])
            colors.append(rgb[nn].mean(0))
    if not mids:
        return np.zeros((0, 9), dtype=np.float32)
    mids = np.array(mids, np.float32)
    dirs = np.array(dirs, np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    m = np.cross(mids, dirs)
    return np.concatenate([m, dirs, np.array(colors, np.float32)], axis=1)


@torch.no_grad()
def _topk_match(model, L1, L2, topk):
    t1 = torch.from_numpy(L1).unsqueeze(0).to(DEVICE)
    t2 = torch.from_numpy(L2).unsqueeze(0).to(DEVICE)
    P, _, _ = model(t1, t2)
    k = min(topk, P.shape[1] * P.shape[2])
    _, flat = torch.topk(P.flatten(start_dim=-2), k=k, dim=-1)
    i1 = (flat // P.shape[-1]).squeeze(0).cpu().numpy()
    i2 = (flat  % P.shape[-1]).squeeze(0).cpu().numpy()
    return i1, i2


def _run_chess_match(model, L1, L2, topk=100, threshold=0.15):
    from sim3.ransac import run_ransac_sim3
    t0 = time.perf_counter()
    i1, i2 = _topk_match(model, L1, L2, topk)
    p1 = L1[i1, :6].T
    p2 = L2[i2, :6].T
    s, R, t, ic, _ = run_ransac_sim3(p1, p2, inlier_threshold=threshold)
    ms = (time.perf_counter() - t0) * 1000
    if R is None:
        return dict(R=np.eye(3), t=np.zeros(3), s=1.0, ic=0, ms=ms)
    return dict(R=R, t=np.asarray(t).flatten(), s=float(s), ic=int(ic), ms=ms)


def chess_eval(weights_path, chess_seq1, chess_seq3, out_dir, label=None):
    from easydict import EasyDict as edict

    print(f"\n{'='*60}")
    print(f"Chess B1/B2 benchmark")
    print(f"  weights : {weights_path}")
    print(f"  seq1    : {chess_seq1}")
    print(f"  seq3    : {chess_seq3}")
    print(f"{'='*60}\n")

    ckpt    = torch.load(weights_path, weights_only=False)
    configs = ckpt.get('config', edict(net_nchannel=128,
                                       GNN_layers=['self', 'cross'] * 6,
                                       net_lambda=0.1, net_maxiter=30,
                                       net_topK=200))
    if not isinstance(configs, edict):
        configs = edict(configs)

    model = _load_model(configs, weights_path).to(DEVICE).eval()

    print('Loading Chess seq-01 ...')
    f1 = _load_chess_frames(chess_seq1)
    xyz1, rgb1 = _build_colored_cloud(f1)
    L1 = _extract_lines_9d(xyz1, rgb1)
    print(f"  {len(xyz1):,} pts → {len(L1)} lines")

    print('Loading Chess seq-03 ...')
    f3 = _load_chess_frames(chess_seq3)
    xyz3, rgb3 = _build_colored_cloud(f3)
    L3 = _extract_lines_9d(xyz3, rgb3)
    print(f"  {len(xyz3):,} pts → {len(L3)} lines")

    R_gt = np.eye(3, dtype=np.float32)
    t_gt = np.zeros(3, dtype=np.float32)
    s_b2 = 1.8
    L3_scaled       = L3.copy()
    L3_scaled[:, :3] *= s_b2

    # If model is 6D, strip color from lines
    ckpt_ch = getattr(configs, 'in_channel', 9)
    if ckpt_ch == 6:
        L1_in, L3_in, L3s_in = L1[:, :6], L3[:, :6], L3_scaled[:, :6]
    else:
        L1_in, L3_in, L3s_in = L1, L3, L3_scaled

    print('\n── B1: RGBD (GT s=1.0) ──')
    r1 = _run_chess_match(model, L1_in, L3_in)
    b1_rot = rot_err_deg(r1['R'], R_gt)
    b1_s   = scale_err_log(r1['s'], 1.0)
    print(f"  rot={b1_rot:.2f}°  t={trans_err(r1['t'],t_gt):.4f}m  "
          f"s={r1['s']:.4f}  s_err={b1_s:.3f}  ic={r1['ic']}  {r1['ms']:.0f}ms")

    print(f'\n── B2: RGB-only moments×{s_b2} (GT s={s_b2}) ──')
    r2 = _run_chess_match(model, L1_in, L3s_in)
    b2_rot = rot_err_deg(r2['R'], R_gt)
    b2_s   = scale_err_log(r2['s'], s_b2)
    print(f"  rot={b2_rot:.2f}°  t={trans_err(r2['t'],t_gt):.4f}m  "
          f"s={r2['s']:.4f}  s_err={b2_s:.3f}  ic={r2['ic']}  {r2['ms']:.0f}ms")

    results = {
        'B1_rot': b1_rot, 'B1_s_err': b1_s,
        'B2_rot': b2_rot, 'B2_s_err': b2_s,
    }
    os.makedirs(out_dir, exist_ok=True)
    run_label = label or os.path.basename(os.path.dirname(weights_path))
    safe = run_label.replace(' ', '_').replace('/', '-')
    out_path = os.path.join(out_dir, f'chess_{safe}.json')
    with open(out_path, 'w') as f:
        json.dump({'label': run_label, 'weights': weights_path, 'metrics': results}, f, indent=2)
    print(f"\nSaved → {out_path}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Mode 3: synthetic hypothesis test
# ══════════════════════════════════════════════════════════════════════════════

def _random_rotation(rng):
    A = rng.standard_normal((3, 3))
    Q, R_mat = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R_mat))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float32)


def _make_clustered_lines(rng, n, n_dir_clusters=10, dir_spread=0.15, pos_range=2.0):
    n_per = n // n_dir_clusters
    extras = n - n_per * n_dir_clusters
    anchors = rng.standard_normal((n_dir_clusters, 3)).astype(np.float32)
    anchors /= np.linalg.norm(anchors, axis=1, keepdims=True)
    parts = []
    for i, a in enumerate(anchors):
        cnt  = n_per + (1 if i < extras else 0)
        d    = a + (rng.standard_normal((cnt, 3)) * dir_spread).astype(np.float32)
        d   /= np.linalg.norm(d, axis=1, keepdims=True)
        p    = rng.uniform(-pos_range, pos_range, (cnt, 3)).astype(np.float32)
        parts.append(np.concatenate([np.cross(p, d), d], axis=1))
    lines = np.concatenate(parts)
    return lines[rng.permutation(len(lines))]


def _apply_sim3_synth(lines, s, R, t):
    m, d = lines[:, :3], lines[:, 3:6]
    d_new = (R @ d.T).T
    m_new = s * (R @ m.T).T + np.cross(t, d_new)
    return np.concatenate([m_new, d_new], axis=1).astype(np.float32)


def _gen_scenes(n, scale_fn, seed=0, n_inliers=100, n_outliers=30):
    rng = np.random.default_rng(seed)
    scenes = []
    for _ in range(n):
        L1_in = _make_clustered_lines(rng, n_inliers)
        s = float(scale_fn(rng))
        R = _random_rotation(rng)
        t = rng.uniform(-1.5, 1.5, 3).astype(np.float32)
        L2_in = _apply_sim3_synth(L1_in, s, R, t)
        n_cl  = max(1, 3)
        L1_out = _make_clustered_lines(rng, n_outliers, n_dir_clusters=n_cl)
        L2_out = _make_clustered_lines(rng, n_outliers, n_dir_clusters=n_cl)
        L1 = np.concatenate([L1_in, L1_out])
        L2 = np.concatenate([L2_in, L2_out])
        idx1, idx2 = rng.permutation(len(L1)), rng.permutation(len(L2))
        L1, L2 = L1[idx1], L2[idx2]
        src = np.argsort(idx1)[:n_inliers]
        tgt = np.argsort(idx2)[:n_inliers]
        scenes.append({'plucker1': L1.astype(np.float32), 'plucker2': L2.astype(np.float32),
                       'matches': np.stack([src, tgt], 0).astype(np.int32),
                       'R_gt': R, 't_gt': t, 's_gt': np.float32(s)})
    return scenes


def _run_sim3_net_synth(model, scene, topk=100, threshold=0.1):
    from sim3.ransac import run_ransac_sim3
    L1, L2 = scene['plucker1'], scene['plucker2']
    t0 = time.perf_counter()
    i1, i2 = _topk_match(model, L1, L2, topk)
    s, R, t, ic, _ = run_ransac_sim3(L1[i1].T, L2[i2].T, inlier_threshold=threshold)
    ms = (time.perf_counter() - t0) * 1000
    if R is None:
        return dict(R=np.eye(3), t=np.zeros(3), s=1.0, ic=0, ms=ms)
    return dict(R=R, t=np.asarray(t).flatten(), s=float(s), ic=int(ic), ms=ms)


def _run_se3_net_synth(model, ransac_fn, scene, topk=200, threshold=0.5):
    L1, L2 = scene['plucker1'], scene['plucker2']
    dm1 = np.hstack([L1[:, 3:], L1[:, :3]])
    dm2 = np.hstack([L2[:, 3:], L2[:, :3]])
    t0 = time.perf_counter()
    i1, i2 = _topk_match(model, dm1, dm2, topk)
    R, t, ic, _ = ransac_fn(dm1[i1].T, dm2[i2].T, inlier_threshold=threshold)
    ms = (time.perf_counter() - t0) * 1000
    if R is None:
        return dict(R=np.eye(3), t=np.zeros(3), s=1.0, ic=0, ms=ms)
    return dict(R=R, t=np.asarray(t).flatten(), s=1.0, ic=int(ic), ms=ms)


def _summarise_scenes(scenes, run_fn, label):
    rots, scales = [], []
    for sc in scenes:
        res = run_fn(sc)
        rots.append(rot_err_deg(res['R'], sc['R_gt']))
        scales.append(scale_err_log(res['s'], float(sc['s_gt'])))
    print(f"  {label:<40s}  med_rot={np.median(rots):6.2f}°  "
          f"med_s_err={np.median(scales):.3f}")
    return {'med_rot': float(np.median(rots)), 'mean_rot': float(np.mean(rots)),
            'med_scale': float(np.median(scales)), 'mean_scale': float(np.mean(scales))}


def hypothesis_eval(weights_path, se3_weights_path, n_scenes, out_dir, label=None):
    from easydict import EasyDict as edict

    print(f"\n{'='*60}")
    print(f"Hypothesis test (synthetic)")
    print(f"  Sim3 weights : {weights_path}")
    print(f"  SE3  weights : {se3_weights_path or '(skipped)'}")
    print(f"  n_scenes     : {n_scenes} per condition")
    print(f"{'='*60}\n")

    ckpt    = torch.load(weights_path, weights_only=False)
    configs = ckpt.get('config', edict(net_nchannel=128, GNN_layers=['self','cross']*6,
                                       net_lambda=0.1, net_maxiter=30, net_topK=200))
    if not isinstance(configs, edict):
        configs = edict(configs)

    sim3_model = _load_model(configs, weights_path).to(DEVICE).eval()

    se3_model, se3_ransac = None, None
    if se3_weights_path and os.path.exists(se3_weights_path):
        from easydict import EasyDict as _EasyDict
        from model.model_plucker import PluckerNetKnn
        import lib.ransac_l2l as _rm
        def _skew_fixed(x):
            x = np.asarray(x).flatten()
            return np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]])
        _rm.skew = _skew_fixed
        from lib.ransac_l2l import run_ransac
        cfg = _EasyDict(net_nchannel=128, GNN_layers=['self','cross']*6,
                        net_lambda=0.1, net_maxiter=30, net_topK=200)
        se3_model = PluckerNetKnn(cfg).to(DEVICE)
        se3_ckpt  = torch.load(se3_weights_path, weights_only=False)
        se3_model.load_state_dict(se3_ckpt['state_dict'])
        se3_model.eval()
        se3_ransac = run_ransac

    print(f'Generating {n_scenes} SE3 test scenes (s=1.0) ...')
    se3_scenes = _gen_scenes(n_scenes, scale_fn=lambda rng: 1.0, seed=1000)

    print(f'Generating {n_scenes} Sim3 test scenes (s ∈ [0.3, 3.0]) ...')
    sim3_scenes = _gen_scenes(
        n_scenes,
        scale_fn=lambda rng: float(np.exp(rng.uniform(np.log(0.3), np.log(3.0)))),
        seed=2000)

    print('\n─── C1: SE3 test (scale=1.0) ───')
    run_s = lambda sc: _run_sim3_net_synth(sim3_model, sc)
    sum_sim3_se3  = _summarise_scenes(se3_scenes,  run_s, 'Sim3-Net on SE3 test')
    if se3_model:
        run_e = lambda sc: _run_se3_net_synth(se3_model, se3_ransac, sc)
        sum_se3_se3 = _summarise_scenes(se3_scenes, run_e, 'SE3-Net  on SE3 test')

    print('\n─── C2: Sim3 test (scale ∈ [0.3, 3.0]) ───')
    sum_sim3_sim3 = _summarise_scenes(sim3_scenes, run_s, 'Sim3-Net on Sim3 test')
    if se3_model:
        sum_se3_sim3 = _summarise_scenes(sim3_scenes, run_e, 'SE3-Net  on Sim3 test')

    h1 = sum_sim3_sim3['med_scale'] < 0.10
    h2 = sum_sim3_se3['med_rot'] < 5.0
    print(f'\n  H1 scale recovery (C2):    med_scale_err = {sum_sim3_sim3["med_scale"]:.3f}  → {"PASS ✓" if h1 else "FAIL ✗"}')
    print(f'  H2 rotation accuracy (C1): med_rot       = {sum_sim3_se3["med_rot"]:.2f}°   → {"PASS ✓" if h2 else "FAIL ✗"}')

    results = {
        'sim3_on_se3':  sum_sim3_se3,
        'sim3_on_sim3': sum_sim3_sim3,
        'H1_pass': h1, 'H2_pass': h2,
    }
    if se3_model:
        results['se3_on_se3']  = sum_se3_se3
        results['se3_on_sim3'] = sum_se3_sim3

    os.makedirs(out_dir, exist_ok=True)
    run_label = label or os.path.basename(os.path.dirname(weights_path))
    safe = run_label.replace(' ', '_').replace('/', '-')
    out_path = os.path.join(out_dir, f'hypothesis_{safe}.json')
    with open(out_path, 'w') as f:
        json.dump({'label': run_label, 'weights': weights_path, 'results': results}, f, indent=2)
    print(f'\nSaved → {out_path}')
    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description='ScalePlueckerNet evaluation')
    p.add_argument('--weights',      default=None,
                   help='Checkpoint path (required for cross-dataset and chess modes)')
    p.add_argument('--dataset',      default='semantic3D,structured3D,replica_gs,7scenes_gs',
                   help='Comma-separated val split names (default: all four datasets)')
    p.add_argument('--data_dir',     default=os.path.join(ROOT, 'dataset'))
    p.add_argument('--label',        default=None,  help='Human-readable label')
    p.add_argument('--out_dir',      default=os.path.join(ROOT, 'results', 'eval_cross_dataset'))

    # Chess mode
    p.add_argument('--chess',        action='store_true', help='Run Chess B1/B2 benchmark')
    p.add_argument('--chess_seq1',   default='/home/rueyday/Downloads/chess/seq-01')
    p.add_argument('--chess_seq3',   default='/home/rueyday/Downloads/chess/seq-03')
    p.add_argument('--chess_out_dir', default=os.path.join(ROOT, 'results', 'eval_chess'))

    # Hypothesis mode
    p.add_argument('--hypothesis',   action='store_true', help='Run synthetic hypothesis test')
    p.add_argument('--se3_weights',  default=None,
                   help='SE3-Net weights for hypothesis comparison (optional)')
    p.add_argument('--n_scenes',     type=int, default=200)
    p.add_argument('--hypothesis_out_dir',
                   default=os.path.join(ROOT, 'results', 'eval_hypothesis'))

    args = p.parse_args()

    ran_any = False

    if args.chess:
        if not args.weights:
            p.error('--chess requires --weights')
        chess_eval(args.weights, args.chess_seq1, args.chess_seq3,
                   args.chess_out_dir, args.label)
        ran_any = True

    if args.hypothesis:
        if not args.weights:
            p.error('--hypothesis requires --weights')
        hypothesis_eval(args.weights, args.se3_weights, args.n_scenes,
                        args.hypothesis_out_dir, args.label)
        ran_any = True

    if not args.chess and not args.hypothesis:
        # default: cross-dataset eval
        if not args.weights:
            p.error('cross-dataset eval requires --weights')
        datasets = [d.strip() for d in args.dataset.split(',')]
        cross_dataset_eval(args.weights, datasets, args.data_dir,
                           args.out_dir, args.label)
        ran_any = True

    if not ran_any:
        p.print_help()


if __name__ == '__main__':
    main()
