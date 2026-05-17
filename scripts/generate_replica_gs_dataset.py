"""
Builds 9D Plücker+LAB training pairs from Replica RGBD using GlueStick
world-space line detection.  Each pair includes a sampled overlap ratio
(0–100%) and scale drawn from (0.1, 10.0) — see scripts/_pair_gen.py.

Memory strategy: each scene runs in a fresh subprocess so PyTorch's allocator
arena is fully released between scenes.  The orchestrator stays at ~0.6 GB.

GlueStick runs on CPU only — GPU causes system crashes.

Output: dataset/replica_gs_train/, dataset/replica_gs_valid/
"""

import os, sys, glob, pickle, argparse, gc, ctypes, subprocess
import numpy as np
import cv2
import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

_libc = ctypes.CDLL('libc.so.6')

def _trim():
    gc.collect()
    _libc.malloc_trim(0)

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_DIR     = os.path.dirname(SCRIPT_DIR)
GLUESTICK    = '/home/rueyday/scale-aware-cross-modal-registration/GlueStick'
REPLICA_ROOT = '/mnt/crucial/rueyday/data/Replica'

FX = FY = 600.0;  CX, CY = 599.5, 339.5
DEPTH_SCALE = 6553.5
DEPTH_MAX   = 4.5

TRAIN_SCENES = ['office0', 'office1', 'office2', 'office3', 'office4', 'room0', 'room1']
VALID_SCENES = ['room2']


# ── Worker (runs inside subprocess) ───────────────────────────────────────────

def _worker(scene_name, replica_root, out_dir, n_per_scene,
            every_n, max_lines, seed):
    """Build pool for one scene, generate pairs, save _scene_<name>.pkl."""
    sys.path.insert(0, GLUESTICK)
    sys.path.insert(0, SCRIPT_DIR)
    from gluestick import numpy_image_to_torch
    from gluestick.models.wireframe import SPWireframeDescriptor
    from _pair_gen import generate_pair, sample_line_lab, dedup_pool

    np.random.seed(seed)

    conf  = {'max_n_lines': max_lines}
    model = SPWireframeDescriptor(conf).to('cpu').eval()

    scene_dir   = os.path.join(replica_root, scene_name)
    depth_files = sorted(glob.glob(os.path.join(scene_dir, 'results', 'depth*.png')))

    poses = []
    with open(os.path.join(scene_dir, 'traj.txt')) as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) == 16:
                poses.append(np.array([float(v) for v in vals], np.float32).reshape(4, 4))

    pool = []
    sampled = depth_files[::every_n]
    for fi, df in enumerate(sampled):
        if fi % 10 == 0:
            print(f'    [{scene_name}] frame {fi}/{len(sampled)} pool={len(pool)}', flush=True)
        idx = int(os.path.splitext(os.path.basename(df))[0].replace('depth', ''))
        if idx >= len(poses):
            continue
        cf = df.replace('depth', 'frame').replace('.png', '.jpg')
        if not os.path.exists(cf):
            continue

        bgr   = cv2.imread(cf)
        gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        depth = cv2.imread(df, cv2.IMREAD_ANYDEPTH).astype(np.float32) / DEPTH_SCALE

        with torch.no_grad():
            lines_2d = model({'image': numpy_image_to_torch(gray)[None]})['lines'][0].numpy()

        R, t = poses[idx][:3, :3], poses[idx][:3, 3]
        for ep in lines_2d:
            (u1, v1), (u2, v2) = ep[0], ep[1]
            u1i = int(np.clip(u1, 0, bgr.shape[1]-1))
            v1i = int(np.clip(v1, 0, bgr.shape[0]-1))
            u2i = int(np.clip(u2, 0, bgr.shape[1]-1))
            v2i = int(np.clip(v2, 0, bgr.shape[0]-1))
            d1, d2 = float(depth[v1i, u1i]), float(depth[v2i, u2i])
            if d1 < 0.15 or d1 > DEPTH_MAX or d2 < 0.15 or d2 > DEPTH_MAX:
                continue
            p1c = np.array([(u1-CX)*d1/FX, (v1-CY)*d1/FY, d1])
            p2c = np.array([(u2-CX)*d2/FX, (v2-CY)*d2/FY, d2])
            p1w, p2w = R @ p1c + t, R @ p2c + t
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
        print(f'    [{scene_name}] WARNING: empty pool', flush=True)
        return

    pool = np.array(pool, np.float32)
    pool = dedup_pool(pool)
    print(f'    [{scene_name}] pool: {len(pool):,} lines (deduped) → generating {n_per_scene} pairs', flush=True)

    pairs = {k: [] for k in ['matches', 'plucker1', 'plucker2', 'R_gt', 't_gt', 's_gt']}
    n_ok  = 0
    for _ in range(n_per_scene * 4):
        if n_ok >= n_per_scene:
            break
        pair = generate_pair(pool)
        if pair is None:
            continue
        for k in pairs:
            pairs[k].append(pair[k])
        n_ok += 1

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'_scene_{scene_name}.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(pairs, f, protocol=4)
    print(f'    [{scene_name}] saved {n_ok} pairs → {out_file}', flush=True)


# ── Orchestrator ───────────────────────────────────────────────────────────────

def generate_split(scene_names, replica_root, out_dir, n_per_scene,
                   every_n, max_lines, seed):
    os.makedirs(out_dir, exist_ok=True)
    keys = ['matches', 'plucker1', 'plucker2', 'R_gt', 't_gt', 's_gt']
    scene_files = []

    for i, name in enumerate(scene_names):
        scene_pkl = os.path.join(out_dir, f'_scene_{name}.pkl')
        if os.path.exists(scene_pkl):
            print(f'  [{name}] already done — skipping', flush=True)
            scene_files.append(scene_pkl)
            continue

        print(f'  [{name}] launching subprocess ...', flush=True)
        cmd = [
            sys.executable, os.path.abspath(__file__),
            '--_worker_scene', name,
            '--replica_root', replica_root,
            '--out_dir', out_dir,
            '--n_per_scene', str(n_per_scene),
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
        print(f'ERROR: no scenes generated for {out_dir}')
        return 0

    combined = {k: [] for k in keys}
    total    = 0
    for sf in scene_files:
        with open(sf, 'rb') as f:
            d = pickle.load(f)
        for k in keys:
            combined[k].extend(d[k])
        total += len(d['t_gt'])

    rng = np.random.default_rng(seed)
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
    p.add_argument('--replica_root',      default=REPLICA_ROOT)
    p.add_argument('--out_dir',           default=os.path.join(REPO_DIR, 'dataset'))
    p.add_argument('--n_train_per_scene', type=int, default=2000)
    p.add_argument('--n_valid_per_scene', type=int, default=400)
    p.add_argument('--every_n',           type=int, default=10)
    p.add_argument('--max_lines',         type=int, default=300)
    p.add_argument('--seed',              type=int, default=42)
    p.add_argument('--_worker_scene',     default=None)
    p.add_argument('--n_per_scene',       type=int, default=2000)
    args = p.parse_args()

    if args._worker_scene is not None:
        _worker(args._worker_scene, args.replica_root, args.out_dir,
                args.n_per_scene, args.every_n, args.max_lines, args.seed)
        return

    sys.path.insert(0, SCRIPT_DIR)
    from _pair_gen import SCALE_RANGE, N_TOTAL, N_MAX_INLIERS, OVERLAP_LEVELS, OVERLAP_PROBS
    print('=' * 60)
    print('Replica GlueStick 9D+overlap dataset')
    print(f'  Scale range : {SCALE_RANGE}')
    print(f'  Lines/pair  : {N_TOTAL}  (max inliers={N_MAX_INLIERS})')
    print(f'  Overlap     : {list(zip(OVERLAP_LEVELS.tolist(), OVERLAP_PROBS.tolist()))}')
    print('=' * 60)

    print('\n── TRAIN ──')
    generate_split(TRAIN_SCENES, args.replica_root,
                   os.path.join(args.out_dir, 'replica_gs_train'),
                   args.n_train_per_scene, args.every_n, args.max_lines, args.seed)

    print('\n── VALID ──')
    generate_split(VALID_SCENES, args.replica_root,
                   os.path.join(args.out_dir, 'replica_gs_valid'),
                   args.n_valid_per_scene, args.every_n, args.max_lines, args.seed + 99999)

    print('\nAll done.')


if __name__ == '__main__':
    main()
