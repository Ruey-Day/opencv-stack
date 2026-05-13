#!/usr/bin/env python3
"""
generate_tum_rgbd_gs_dataset.py

Builds 9D Plücker+LAB training pairs from TUM RGB-D SLAM using GlueStick
world-space line detection.  See scripts/_pair_gen.py for diversity policy.

GlueStick runs on CPU only.  Subprocess-per-sequence for memory safety.

Output: dataset/tum_rgbd_gs_train/, dataset/tum_rgbd_gs_valid/
"""

import os, sys, pickle, argparse, gc, ctypes, subprocess
import numpy as np
import cv2
import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

_libc = ctypes.CDLL('libc.so.6')

def _trim():
    gc.collect()
    _libc.malloc_trim(0)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR   = os.path.dirname(SCRIPT_DIR)
GLUESTICK  = '/home/rueyday/scale-aware-cross-modal-registration/GlueStick'
TUM_ROOT   = '/mnt/crucial/TUM_RGBD-SLAM/TUM_RGBD-SLAM'

DEPTH_SCALE = 5000.0
DEPTH_MAX   = 4.5

_INTRINSICS = {
    'freiburg1': dict(fx=517.3, fy=516.5, cx=318.6, cy=255.3),
    'freiburg2': dict(fx=520.9, fy=521.0, cx=325.1, cy=249.7),
    'freiburg3': dict(fx=535.4, fy=539.2, cx=320.1, cy=247.6),
}

TRAIN_SEQS = [
    'rgbd_dataset_freiburg1_desk',
    'rgbd_dataset_freiburg1_desk2',
    'rgbd_dataset_freiburg1_room',
    'rgbd_dataset_freiburg2_xyz',
]
VALID_SEQS = [
    'rgbd_dataset_freiburg3_long_office_household',
]


def _get_intr(seq_name):
    for key, intr in _INTRINSICS.items():
        if key in seq_name:
            return intr
    raise ValueError(f'No intrinsics for: {seq_name}')


def _parse_file(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            rows.append((float(parts[0]), parts[1]))
    return rows


def _parse_gt(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            rows.append([float(x) for x in line.split()])
    return np.array(rows, dtype=np.float64)


def _associate(rgb_rows, depth_rows, gt_arr, max_diff=0.02):
    depth_ts = np.array([r[0] for r in depth_rows])
    gt_ts    = gt_arr[:, 0]
    triples  = []
    for rt, rf in rgb_rows:
        di = int(np.argmin(np.abs(depth_ts - rt)))
        gi = int(np.argmin(np.abs(gt_ts    - rt)))
        if abs(depth_ts[di] - rt) > max_diff or abs(gt_ts[gi] - rt) > max_diff:
            continue
        triples.append((rf, depth_rows[di][1], gt_arr[gi, 1:]))
    return triples


def _quat_to_matrix(pose_row):
    tx, ty, tz, qx, qy, qz, qw = pose_row
    n  = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    R  = np.array([
        [1 - 2*(qy**2+qz**2),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2+qz**2),  2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2+qy**2)],
    ], dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R;  T[:3, 3] = [tx, ty, tz]
    return T


# ── Worker ─────────────────────────────────────────────────────────────────────

def _worker(seq_name, tum_root, out_dir, n_per_seq, every_n, max_lines, seed):
    sys.path.insert(0, GLUESTICK)
    sys.path.insert(0, SCRIPT_DIR)
    from gluestick import numpy_image_to_torch
    from gluestick.models.wireframe import SPWireframeDescriptor
    from _pair_gen import generate_pair, sample_line_lab

    np.random.seed(seed)

    intr  = _get_intr(seq_name)
    fx, fy, cx, cy = intr['fx'], intr['fy'], intr['cx'], intr['cy']

    conf  = {'max_n_lines': max_lines}
    model = SPWireframeDescriptor(conf).to('cpu').eval()

    seq_dir    = os.path.join(tum_root, seq_name)
    rgb_rows   = _parse_file(os.path.join(seq_dir, 'rgb.txt'))
    depth_rows = _parse_file(os.path.join(seq_dir, 'depth.txt'))
    gt_arr     = _parse_gt(os.path.join(seq_dir, 'groundtruth.txt'))
    triples    = _associate(rgb_rows, depth_rows, gt_arr)
    selected   = triples[::every_n]

    pool = []
    for fi, (rgb_rel, depth_rel, pose_row) in enumerate(selected):
        if fi % 20 == 0:
            print(f'    [{seq_name}] frame {fi}/{len(selected)} pool={len(pool)}', flush=True)
        rgb_path   = os.path.join(seq_dir, rgb_rel)
        depth_path = os.path.join(seq_dir, depth_rel)
        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            continue

        T_cw  = _quat_to_matrix(pose_row)
        R, t_vec = T_cw[:3, :3], T_cw[:3, 3]

        bgr   = cv2.imread(rgb_path)
        gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / DEPTH_SCALE

        with torch.no_grad():
            lines_2d = model({'image': numpy_image_to_torch(gray)[None]})['lines'][0].numpy()

        for ep in lines_2d:
            (u1, v1), (u2, v2) = ep[0], ep[1]
            u1i = int(np.clip(u1, 0, bgr.shape[1]-1))
            v1i = int(np.clip(v1, 0, bgr.shape[0]-1))
            u2i = int(np.clip(u2, 0, bgr.shape[1]-1))
            v2i = int(np.clip(v2, 0, bgr.shape[0]-1))
            d1, d2 = float(depth[v1i, u1i]), float(depth[v2i, u2i])
            if d1 < 0.15 or d1 > DEPTH_MAX or d2 < 0.15 or d2 > DEPTH_MAX:
                continue
            p1c = np.array([(u1-cx)*d1/fx, (v1-cy)*d1/fy, d1])
            p2c = np.array([(u2-cx)*d2/fx, (v2-cy)*d2/fy, d2])
            p1w, p2w = R @ p1c + t_vec, R @ p2c + t_vec
            diff = p2w - p1w
            ln   = np.linalg.norm(diff)
            if ln < 0.02:
                continue
            dw = diff / ln
            mw = np.cross((p1w + p2w) / 2, dw)
            lab = sample_line_lab(bgr, ep)
            pool.append(np.concatenate([mw, dw, lab]).astype(np.float32))

        del bgr, gray, depth, lines_2d
        _trim()

    if not pool:
        print(f'    [{seq_name}] WARNING: empty pool', flush=True)
        return

    pool = np.array(pool, np.float32)
    print(f'    [{seq_name}] pool: {len(pool):,} lines → generating {n_per_seq} pairs', flush=True)

    pairs = {k: [] for k in ['matches', 'plucker1', 'plucker2', 'R_gt', 't_gt', 's_gt']}
    n_ok  = 0
    for _ in range(n_per_seq * 4):
        if n_ok >= n_per_seq:
            break
        pair = generate_pair(pool)
        if pair is None:
            continue
        for k in pairs:
            pairs[k].append(pair[k])
        n_ok += 1

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'_scene_{seq_name}.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(pairs, f, protocol=4)
    print(f'    [{seq_name}] saved {n_ok} pairs → {out_file}', flush=True)


# ── Orchestrator ───────────────────────────────────────────────────────────────

def generate_split(seq_names, tum_root, out_dir, n_per_seq,
                   every_n, max_lines, seed):
    os.makedirs(out_dir, exist_ok=True)
    keys = ['matches', 'plucker1', 'plucker2', 'R_gt', 't_gt', 's_gt']
    scene_files = []

    for i, name in enumerate(seq_names):
        scene_pkl = os.path.join(out_dir, f'_scene_{name}.pkl')
        if os.path.exists(scene_pkl):
            print(f'  [{name}] already done — skipping', flush=True)
            scene_files.append(scene_pkl)
            continue

        print(f'  [{name}] launching subprocess ...', flush=True)
        cmd = [
            sys.executable, os.path.abspath(__file__),
            '--_worker_seq', name,
            '--tum_root', tum_root,
            '--out_dir', out_dir,
            '--n_per_seq', str(n_per_seq),
            '--every_n', str(every_n),
            '--max_lines', str(max_lines),
            '--seed', str(seed + i),
        ]
        ret = subprocess.run(cmd, check=False)
        if ret.returncode != 0:
            print(f'  [{name}] FAILED (returncode={ret.returncode})', flush=True)
            continue
        if os.path.exists(scene_pkl):
            scene_files.append(scene_pkl)
        else:
            print(f'  [{name}] WARNING: no output file', flush=True)

    if not scene_files:
        print(f'ERROR: no sequences generated for {out_dir}')
        return 0

    combined = {k: [] for k in keys}
    total    = 0
    for sf in scene_files:
        with open(sf, 'rb') as f:
            d = pickle.load(f)
        for k in keys:
            combined[k].extend(d[k])
        total += len(d['t_gt'])

    rng  = np.random.default_rng(seed)
    perm = rng.permutation(total)
    for k in keys:
        combined[k] = [combined[k][i] for i in perm]

    for k, v in combined.items():
        with open(os.path.join(out_dir, f'{k}.pkl'), 'wb') as f:
            pickle.dump(v, f, protocol=4)

    for sf in scene_files:
        os.remove(sf)

    print(f'  → {out_dir}  ({total:,} pairs)', flush=True)
    return total


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tum_root',       default=TUM_ROOT)
    p.add_argument('--out_dir',        default=os.path.join(REPO_DIR, 'dataset'))
    p.add_argument('--n_train_per_seq', type=int, default=2000)
    p.add_argument('--n_valid_per_seq', type=int, default=400)
    p.add_argument('--every_n',        type=int, default=5)
    p.add_argument('--max_lines',      type=int, default=300)
    p.add_argument('--seed',           type=int, default=42)
    p.add_argument('--_worker_seq',    default=None)
    p.add_argument('--n_per_seq',      type=int, default=2000)
    args = p.parse_args()

    if args._worker_seq is not None:
        _worker(args._worker_seq, args.tum_root, args.out_dir,
                args.n_per_seq, args.every_n, args.max_lines, args.seed)
        return

    sys.path.insert(0, SCRIPT_DIR)
    from _pair_gen import SCALE_RANGE, N_TOTAL, N_MAX_INLIERS
    print('=' * 60)
    print('TUM RGB-D GlueStick 9D+overlap dataset')
    print(f'  Scale range : {SCALE_RANGE}')
    print(f'  Lines/pair  : {N_TOTAL}  (max inliers={N_MAX_INLIERS})')
    print('=' * 60)

    print('\n── TRAIN ──')
    generate_split(TRAIN_SEQS, args.tum_root,
                   os.path.join(args.out_dir, 'tum_rgbd_gs_train'),
                   args.n_train_per_seq, args.every_n, args.max_lines, args.seed)

    print('\n── VALID ──')
    generate_split(VALID_SEQS, args.tum_root,
                   os.path.join(args.out_dir, 'tum_rgbd_gs_valid'),
                   args.n_valid_per_seq, args.every_n, args.max_lines, args.seed + 99999)

    print('\nAll done.')


if __name__ == '__main__':
    main()
