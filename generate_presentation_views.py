#!/usr/bin/env python3
"""
generate_presentation_views.py

Generates 4 presentation-quality images:
  1. results/presentation/view_seq01.png  — RGB frame from seq-01
  2. results/presentation/view_seq03.png  — RGB frame from seq-03
  3. results/presentation/lines_seq01.png — 3D Plücker lines from seq-01
  4. results/presentation/lines_seq03.png — 3D Plücker lines from seq-03
"""

import os, sys, glob
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.spatial import cKDTree

ROOT       = os.path.dirname(os.path.abspath(__file__))
CHESS_SEQ1 = "/home/rueyday/Downloads/chess/seq-01"
CHESS_SEQ3 = "/home/rueyday/Downloads/chess/seq-03"
OUT_DIR    = os.path.join(ROOT, "results", "presentation")
os.makedirs(OUT_DIR, exist_ok=True)

FX, FY, CX, CY = 525.0, 525.0, 319.5, 239.5
DEPTH_SCALE = 1000.0
BG = '#0e0e1c'


# ── data loading ───────────────────────────────────────────────────────────────

def load_frames(seq_dir, n_frames=20, frame_step=40):
    depth_files = sorted(glob.glob(os.path.join(seq_dir, "*.depth.png")))[::frame_step][:n_frames]
    frames = []
    for df in depth_files:
        pf = df.replace(".depth.png", ".pose.txt")
        cf = df.replace(".depth.png", ".color.png")
        if not os.path.exists(pf) or not os.path.exists(cf):
            continue
        depth = cv2.imread(df, cv2.IMREAD_ANYDEPTH).astype(np.float32) / DEPTH_SCALE
        color = cv2.cvtColor(cv2.imread(cf), cv2.COLOR_BGR2RGB)
        pose  = np.loadtxt(pf)
        frames.append((depth, color, pose))
    return frames


def build_colored_cloud(frames, subsample=4, max_depth=3.5, voxel=0.025):
    pts_all, col_all = [], []
    for depth, color, pose in frames:
        H, W = depth.shape
        vi, ui = np.meshgrid(np.arange(0, H, subsample),
                             np.arange(0, W, subsample), indexing="ij")
        vi, ui = vi.ravel(), ui.ravel()
        z = depth[vi, ui]
        ok = (z > 0.1) & (z < max_depth)
        z, vi, ui = z[ok], vi[ok], ui[ok]
        x = (ui - CX) * z / FX
        y = (vi - CY) * z / FY
        cam = np.stack([x, y, z, np.ones_like(z)], 0)
        pts = (pose @ cam)[:3].T
        col = color[vi, ui] / 255.0
        pts_all.append(pts)
        col_all.append(col)
    cloud = np.concatenate(pts_all, 0)
    colors = np.concatenate(col_all, 0)
    keys = np.floor(cloud / voxel).astype(np.int32)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return cloud[idx], colors[idx]


def extract_lines(cloud, n_lines=300, k=20, linearity_thresh=0.72, seed=42):
    rng  = np.random.default_rng(seed)
    tree = cKDTree(cloud)
    mids, dirs = [], []
    indices = rng.choice(len(cloud), size=min(20000, len(cloud)), replace=False)
    for idx in indices:
        if len(mids) >= n_lines:
            break
        nn  = tree.query(cloud[idx], k=k)[1]
        pts = cloud[nn]
        ctr = pts.mean(0)
        cov = (pts - ctr).T @ (pts - ctr) / k
        ev, evec = np.linalg.eigh(cov)
        lam = ev[::-1]
        if (lam[0] - lam[1]) / (lam[0] + 1e-9) > linearity_thresh:
            mids.append(ctr)
            dirs.append(evec[:, -1])
    mids = np.array(mids, dtype=np.float32)
    dirs = np.array(dirs, dtype=np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return mids, dirs


# ── image 1 & 2: RGB views ─────────────────────────────────────────────────────

def save_rgb_view(seq_dir, out_path, label, frame_idx=200):
    color_files = sorted(glob.glob(os.path.join(seq_dir, "*.color.png")))
    img = cv2.cvtColor(cv2.imread(color_files[frame_idx]), cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.imshow(img)
    ax.set_title(label, color='white', fontsize=16, fontweight='bold', pad=10)
    ax.axis('off')
    plt.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {os.path.relpath(out_path, ROOT)}")


# ── image 3 & 4: 3D line visualizations ───────────────────────────────────────

def _ax_limits(ax, mids):
    lo, hi = mids.min(0) - 0.25, mids.max(0) + 0.25
    r = (hi - lo).max() / 2
    mid = (lo + hi) / 2
    ax.set_xlim(mid[0] - r, mid[0] + r)
    ax.set_ylim(mid[1] - r, mid[1] + r)
    ax.set_zlim(mid[2] - r, mid[2] + r)


def save_lines_view(cloud, mids, dirs, out_path, label,
                    cloud_color='#2a3a5c', line_color='#00d4ff',
                    n_lines=200, half=0.35, elev=18, azim=-50, seed=0):
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(mids), min(n_lines, len(mids)), replace=False)

    # subsample cloud for background scatter
    cloud_idx = rng.choice(len(cloud), min(8000, len(cloud)), replace=False)

    fig = plt.figure(figsize=(8, 7))
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(BG)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('#1e2a40')

    # background point cloud (faint)
    c = cloud[cloud_idx]
    ax.scatter(c[:, 0], c[:, 1], c[:, 2], s=0.3, c=cloud_color, alpha=0.15, linewidths=0)

    # line segments
    for i in pick:
        p = mids[i]
        d = dirs[i]
        s, e = p - half * d, p + half * d
        ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]],
                color=line_color, lw=1.4, alpha=0.85)

    ax.set_title(label, color='white', fontsize=15, fontweight='bold', pad=6)
    ax.tick_params(colors='#556680', labelsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e2a40')
    ax.set_xlabel('X', color='#556680', fontsize=7)
    ax.set_ylabel('Y', color='#556680', fontsize=7)
    ax.set_zlabel('Z', color='#556680', fontsize=7)
    _ax_limits(ax, mids[pick])
    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {os.path.relpath(out_path, ROOT)}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    # ── RGB views ────────────────────────────────────────────────────────────
    print("Saving RGB views ...")
    save_rgb_view(CHESS_SEQ1, os.path.join(OUT_DIR, "view_seq01.png"),
                  "Chess 7-Scenes — Sequence 01", frame_idx=200)
    save_rgb_view(CHESS_SEQ3, os.path.join(OUT_DIR, "view_seq03.png"),
                  "Chess 7-Scenes — Sequence 03", frame_idx=200)

    # ── 3D line segments ─────────────────────────────────────────────────────
    print("\nLoading seq-01 frames ...")
    frames1 = load_frames(CHESS_SEQ1, n_frames=20, frame_step=40)
    cloud1, _ = build_colored_cloud(frames1)
    mids1, dirs1 = extract_lines(cloud1, n_lines=300)
    print(f"  {len(mids1)} lines from {len(cloud1)} cloud points")
    save_lines_view(cloud1, mids1, dirs1,
                    os.path.join(OUT_DIR, "lines_seq01.png"),
                    label="Extracted Plücker Lines — Sequence 01",
                    line_color='#38b0ff', cloud_color='#2040a0')

    print("\nLoading seq-03 frames ...")
    frames3 = load_frames(CHESS_SEQ3, n_frames=20, frame_step=40)
    cloud3, _ = build_colored_cloud(frames3)
    mids3, dirs3 = extract_lines(cloud3, n_lines=300)
    print(f"  {len(mids3)} lines from {len(cloud3)} cloud points")
    save_lines_view(cloud3, mids3, dirs3,
                    os.path.join(OUT_DIR, "lines_seq03.png"),
                    label="Extracted Plücker Lines — Sequence 03",
                    line_color='#ff6633', cloud_color='#802020')

    print(f"\nAll 4 images saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
