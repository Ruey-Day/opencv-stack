#!/usr/bin/env python3
"""
visualize_lines.py — Interactive 3D viewer for a Plücker line-cloud pair.

Saves an HTML file you can open in any browser.  Headless-safe (no display needed).

Usage
-----
  python scripts/visualize_lines.py                          # replica_gs_valid, sample 0
  python scripts/visualize_lines.py --dataset 7scenes_gs --idx 5
  python scripts/visualize_lines.py --no_matches
  python scripts/visualize_lines.py --aligned   # GT-transform cloud1 to verify alignment
  python scripts/visualize_lines.py --scene_dir /mnt/crucial/rueyday/data/Replica/room2
  python scripts/visualize_lines.py --out results/viz/my_pair.html
"""

import os
import sys
import argparse
import pickle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Plücker helpers ────────────────────────────────────────────────────────────

def apply_sim3_plucker(lines, s, R, t):
    """Apply Sim(3) to Plücker lines [m, d]: d' = R d,  m' = s R m + t × d'."""
    m, d = lines[:, :3].astype(np.float64), lines[:, 3:6].astype(np.float64)
    d_new = (R @ d.T).T
    m_new = s * (R @ m.T).T + np.cross(t, d_new)
    return np.concatenate([m_new, d_new], axis=1).astype(np.float32)


def plucker_to_segments(lines, half_len=0.4):
    """Convert (N, 6+) Plücker lines [m, d] to (N, 2, 3) endpoint array.

    NOTE: Plücker lines are infinite — there are no true endpoints.
    This draws a fixed-length segment centred at the closest point to the world
    origin.  Use clip_lines_to_cloud() for geometrically meaningful extents.
    """
    m = lines[:, :3].astype(np.float64)
    d = lines[:, 3:6].astype(np.float64)
    d_norm = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    p0 = np.cross(d_norm, m)   # closest point on line to world origin
    return np.stack([p0 - half_len * d_norm,
                     p0 + half_len * d_norm], axis=1)  # (N, 2, 3)


def clip_lines_to_cloud(lines, xyz_cloud, threshold=0.08, max_half=1.0, fallback=0.4):
    """Clip infinite Plücker lines to the extent of nearby point cloud points.

    Only considers cloud points within ±max_half metres of p₀ along the line,
    then uses 5th/95th percentile of those projections as endpoints.
    This prevents a line running parallel to a long wall from spanning the room.
    Falls back to ±fallback metres when fewer than 4 points are nearby.
    Returns (N, 2, 3) float32.
    """
    m = lines[:, :3].astype(np.float64)
    d = lines[:, 3:6].astype(np.float64)
    d_norm = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    p0_all = np.cross(d_norm, m)   # (N, 3)
    xyz = xyz_cloud.astype(np.float64)

    endpoints = np.empty((len(lines), 2, 3), dtype=np.float32)
    for i, (p0, dv) in enumerate(zip(p0_all, d_norm)):
        vecs = xyz - p0                             # (M, 3)
        proj = vecs @ dv                            # (M,) along-line projection
        # Only look within ±max_half of p₀ — stops wall-spanning
        in_range = np.abs(proj) < max_half
        perp = vecs - proj[:, None] * dv
        dist = np.linalg.norm(perp, axis=1)
        near = (dist < threshold) & in_range
        if near.sum() < 4:
            endpoints[i, 0] = p0 - fallback * dv
            endpoints[i, 1] = p0 + fallback * dv
        else:
            t = proj[near]
            # Percentile trim: ignore stray points at the extremes
            t_lo, t_hi = np.percentile(t, 5), np.percentile(t, 95)
            if t_hi - t_lo < 0.05:   # degenerate — too short, use fallback
                endpoints[i, 0] = p0 - fallback * dv
                endpoints[i, 1] = p0 + fallback * dv
            else:
                endpoints[i, 0] = p0 + t_lo * dv
                endpoints[i, 1] = p0 + t_hi * dv
    return endpoints


# ── Point cloud loading ────────────────────────────────────────────────────────

def load_replica_cloud(scene_dir, every_n=10, pixel_step=2, voxel=0.04, max_depth=4.5):
    """Unproject Replica depth frames to a voxel-downsampled world-space point cloud."""
    import glob
    from PIL import Image

    FX = FY = 600.0
    CX, CY  = 599.5, 339.5
    DEPTH_SCALE = 6553.5

    depth_files = sorted(glob.glob(os.path.join(scene_dir, 'results', 'depth*.png')))
    if not depth_files:
        print(f'  [cloud] no depth files in {scene_dir}/results/')
        return None, None

    poses = []
    with open(os.path.join(scene_dir, 'traj.txt')) as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) == 16:
                poses.append(np.array([float(v) for v in vals], np.float32).reshape(4, 4))

    xyz_all, rgb_all = [], []
    sampled = depth_files[::every_n]
    print(f'  [cloud] loading {len(sampled)} frames from {os.path.basename(scene_dir)} ...', flush=True)

    for df in sampled:
        fidx = int(os.path.splitext(os.path.basename(df))[0].replace('depth', ''))
        if fidx >= len(poses):
            continue
        depth = np.array(Image.open(df), dtype=np.float32) / DEPTH_SCALE
        H, W  = depth.shape

        vi, ui = np.meshgrid(np.arange(0, H, pixel_step),
                             np.arange(0, W, pixel_step), indexing='ij')
        vi, ui = vi.ravel(), ui.ravel()
        z = depth[vi, ui]
        ok = (z > 0.15) & (z < max_depth)
        z, vi, ui = z[ok], vi[ok], ui[ok]
        if len(z) == 0:
            continue

        cam = np.stack([(ui - CX) * z / FX, (vi - CY) * z / FY, z, np.ones_like(z)])
        xyz = (poses[fidx] @ cam)[:3].T

        # load RGB
        cf = df.replace('depth', 'frame').replace('.png', '.jpg')
        if os.path.exists(cf):
            rgb = np.array(Image.open(cf), dtype=np.float32) / 255.0
            rgb_all.append(rgb[vi, ui])
        else:
            rgb_all.append(np.full((len(xyz), 3), 0.65, dtype=np.float32))

        xyz_all.append(xyz.astype(np.float32))

    if not xyz_all:
        return None, None

    xyz = np.concatenate(xyz_all)
    rgb = np.concatenate(rgb_all)

    # voxel downsample
    keys = np.floor(xyz / voxel).astype(np.int32)
    _, idx = np.unique(keys, axis=0, return_index=True)
    print(f'  [cloud] {len(xyz):,} pts → {len(idx):,} after voxel={voxel}m')
    return xyz[idx], rgb[idx]


# ── Plotly trace builders ──────────────────────────────────────────────────────

def _cloud_trace(xyz, rgb, name, opacity=0.25, size=1):
    import plotly.graph_objects as go
    colors = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b in rgb]
    return go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        mode='markers',
        name=name,
        opacity=opacity,
        marker=dict(size=size, color=colors),
    )


def _segments_to_trace(segs, color_hex, name, opacity=0.8, width=3):
    """All segments as one trace with a single colour — much faster than per-line colours."""
    import plotly.graph_objects as go
    xs, ys, zs = [], [], []
    for p, q in segs:
        xs += [p[0], q[0], None]
        ys += [p[1], q[1], None]
        zs += [p[2], q[2], None]
    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='lines',
        name=name,
        opacity=opacity,
        line=dict(color=color_hex, width=width),
    )


def _match_trace(segs1, segs2, src, tgt, n_show=60):
    import plotly.graph_objects as go
    mid1 = segs1.mean(axis=1)
    mid2 = segs2.mean(axis=1)
    rng  = np.random.default_rng(0)
    idx  = rng.choice(len(src), size=min(n_show, len(src)), replace=False)
    xs, ys, zs = [], [], []
    for i in idx:
        p, q = mid1[src[i]], mid2[tgt[i]]
        xs += [p[0], q[0], None]
        ys += [p[1], q[1], None]
        zs += [p[2], q[2], None]
    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='lines',
        name=f'GT matches (n={len(idx)})',
        opacity=0.3,
        line=dict(color='white', width=1),
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',    default='replica_gs')
    p.add_argument('--split',      default='valid')
    p.add_argument('--data_dir',   default=os.path.join(ROOT, 'dataset'))
    p.add_argument('--idx',        type=int, default=0)
    p.add_argument('--half_len',   type=float, default=0.4,
                   help='Half-length of rendered segments (metres)')
    p.add_argument('--no_matches',   action='store_true')
    p.add_argument('--inliers_only', action='store_true',
                   help='Show only GT inlier lines (strips random outliers)')
    p.add_argument('--aligned',    action='store_true',
                   help='Show cloud1 GT-transformed (verify alignment)')
    p.add_argument('--scene_dir',  default=None,
                   help='Path to a scene directory with results/depth*.png + traj.txt '
                        '(auto-detected for replica_gs_valid → room2)')
    p.add_argument('--no_cloud',   action='store_true',
                   help='Skip point cloud even if scene_dir is available')
    p.add_argument('--voxel',      type=float, default=0.04,
                   help='Voxel size for point cloud downsampling (metres)')
    p.add_argument('--out',        default=None)
    args = p.parse_args()

    import plotly.graph_objects as go

    split_dir = os.path.join(args.data_dir, f'{args.dataset}_{args.split}')
    if not os.path.isdir(split_dir):
        print(f'ERROR: {split_dir} not found.')
        sys.exit(1)

    def _load(name):
        with open(os.path.join(split_dir, f'{name}.pkl'), 'rb') as f:
            return pickle.load(f)

    print(f'Loading sample {args.idx} from {split_dir} ...')
    plucker1 = np.array(_load('plucker1')[args.idx], np.float32)[:, :6]
    plucker2 = np.array(_load('plucker2')[args.idx], np.float32)[:, :6]
    matches  = np.array(_load('matches')[args.idx],  np.int32)
    s_gt     = float(_load('s_gt')[args.idx])
    R_gt     = np.array(_load('R_gt')[args.idx], np.float32)
    t_gt     = np.array(_load('t_gt')[args.idx], np.float32).flatten()

    n1 = len(plucker1)
    print(f'  {n1} lines/cloud  |  s_gt={s_gt:.3f}  |  matches={matches.shape[1]}')

    # Optionally strip to inliers only so the random outliers don't obscure structure
    if args.inliers_only and matches.shape[1] > 0:
        src, tgt = matches[0], matches[1]
        plucker1 = plucker1[src]
        plucker2 = plucker2[tgt]
        matches  = np.stack([np.arange(len(src)), np.arange(len(src))], 0).astype(np.int32)
        print(f'  (inliers only: {len(src)} lines)')

    segs1 = plucker_to_segments(plucker1, args.half_len)

    if args.aligned:
        src, tgt = matches[0], matches[1]
        p1_aligned = apply_sim3_plucker(plucker1[src], s_gt,
                                         R_gt.astype(np.float64), t_gt.astype(np.float64))
        segs2 = plucker_to_segments(p1_aligned, args.half_len)
        cloud2_label = f'Cloud1 inliers → GT Sim3 (n={len(src)})'
    else:
        segs2 = plucker_to_segments(plucker2, args.half_len)
        cloud2_label = 'Cloud 2 (target)'

    traces = []
    xyz_cloud = None

    # ── point cloud ──
    scene_dir = args.scene_dir
    if scene_dir is None and not args.no_cloud:
        replica_root = '/mnt/crucial/rueyday/data/Replica'
        if args.dataset == 'replica_gs' and args.split == 'valid':
            scene_dir = os.path.join(replica_root, 'room2')

    if scene_dir and not args.no_cloud:
        xyz_cloud, rgb_cloud = load_replica_cloud(scene_dir, voxel=args.voxel)
        if xyz_cloud is not None:
            traces.append(_cloud_trace(xyz_cloud, rgb_cloud, name='Scene (room2)',
                                       opacity=0.35, size=2))

    # ── line clouds ──
    # Cloud 1 is in the scene's world frame — clip to actual geometry if available.
    # Cloud 2 is Sim(3)-transformed so it won't align with the point cloud; use fixed length.
    if xyz_cloud is not None and not args.aligned:
        print(f'  Clipping {len(plucker1)} lines to point cloud geometry ...', flush=True)
        segs1 = clip_lines_to_cloud(plucker1, xyz_cloud)
    # segs2 always uses fixed half_len (different coordinate frame)

    traces += [
        _segments_to_trace(segs1, '#4a9eff', name='Cloud 1 (source)'),
        _segments_to_trace(segs2, '#ff5555', name=cloud2_label),
    ]
    if not args.no_matches and not args.aligned and matches.shape[1] > 0 and s_gt != 0.0:
        src, tgt = matches[0], matches[1]
        traces.append(_match_trace(segs1, segs2, src, tgt))

    mode  = 'aligned' if args.aligned else 'raw'
    title = (f'{args.dataset}_{args.split}  idx={args.idx}  '
             f's_gt={s_gt:.3f}  {n1} lines  [{mode}]')

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
            aspectmode='data',
            bgcolor='#111111',
            xaxis=dict(gridcolor='#333', zerolinecolor='#333'),
            yaxis=dict(gridcolor='#333', zerolinecolor='#333'),
            zaxis=dict(gridcolor='#333', zerolinecolor='#333'),
        ),
        paper_bgcolor='#1a1a1a',
        font=dict(color='white'),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    out = args.out or os.path.join(ROOT, 'results', 'viz',
                                   f'{args.dataset}_{args.split}_{args.idx}.html')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.write_html(out)
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()
