#!/usr/bin/env python3
"""
view_lines_interactive.py — rotate 3D line views, then save.

Usage:
    python view_lines_interactive.py seq01   # seq-01 lines (blue)
    python view_lines_interactive.py seq03   # seq-03 lines (orange)

Rotate to your preferred angle, then close the window to save.
"""

import os, sys, glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.spatial import cKDTree

SEQ = sys.argv[1] if len(sys.argv) > 1 else "seq01"
assert SEQ in ("seq01", "seq03"), "Usage: python view_lines_interactive.py seq01|seq03"

ROOT       = os.path.dirname(os.path.abspath(__file__))
CHESS_DIRS = {"seq01": "/home/rueyday/Downloads/chess/seq-01",
              "seq03": "/home/rueyday/Downloads/chess/seq-03"}
COLORS     = {"seq01": "#38b0ff", "seq03": "#ff6633"}
LABELS     = {"seq01": "Extracted Plücker Lines — Sequence 01",
              "seq03": "Extracted Plücker Lines — Sequence 03"}
OUT_PATH   = os.path.join(ROOT, "results", "presentation", f"lines_{SEQ}.png")

FX, FY, CX, CY = 525.0, 525.0, 319.5, 239.5
DEPTH_SCALE = 1000.0
BG = 'white'


def load_frames(seq_dir, n_frames=20, frame_step=40):
    depth_files = sorted(glob.glob(os.path.join(seq_dir, "*.depth.png")))[::frame_step][:n_frames]
    frames = []
    for df in depth_files:
        pf = df.replace(".depth.png", ".pose.txt")
        if not os.path.exists(pf):
            continue
        depth = cv2.imread(df, cv2.IMREAD_ANYDEPTH).astype(np.float32) / DEPTH_SCALE
        pose  = np.loadtxt(pf)
        frames.append((depth, pose))
    return frames


def build_cloud(frames, subsample=4, max_depth=3.5, voxel=0.025):
    pts_all = []
    for depth, pose in frames:
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
        pts_all.append((pose @ cam)[:3].T)
    cloud = np.concatenate(pts_all, 0)
    keys = np.floor(cloud / voxel).astype(np.int32)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return cloud[idx]


def extract_lines(cloud, n_lines=800, radius=0.18, linearity_thresh=0.68,
                  min_len=0.08, dedup_pos=0.035, dedup_ang=0.20, seed=42):
    """
    Extract 3D line segments with real endpoints.

    - radius search so all neighborhoods have the same spatial scale
    - actual endpoints from projecting neighbors onto principal axis
    - min_len filter removes tiny noise segments
    - bin deduplication: skip candidate if a nearly-parallel segment already
      covers the same spatial bin
    """
    rng  = np.random.default_rng(seed)
    tree = cKDTree(cloud)
    starts, ends = [], []
    seen_bins = set()

    indices = rng.permutation(len(cloud))
    for idx in indices:
        if len(starts) >= n_lines:
            break
        nn_idx = tree.query_ball_point(cloud[idx], r=radius)
        if len(nn_idx) < 6:
            continue
        pts = cloud[nn_idx]
        ctr = pts.mean(0)
        cov = (pts - ctr).T @ (pts - ctr) / len(pts)
        ev, evec = np.linalg.eigh(cov)
        lam = ev[::-1]
        if (lam[0] - lam[1]) / (lam[0] + 1e-9) < linearity_thresh:
            continue

        d = evec[:, -1]
        if d[np.argmax(np.abs(d))] < 0:
            d = -d

        proj = (pts - ctr) @ d
        p0, p1 = ctr + d * proj.min(), ctr + d * proj.max()
        if proj.max() - proj.min() < min_len:
            continue

        mid = (p0 + p1) / 2
        pos_bin = tuple(np.floor(mid / dedup_pos).astype(int))
        ang_bin = tuple(np.floor(d / dedup_ang).astype(int))
        key = pos_bin + ang_bin
        if key in seen_bins:
            continue
        seen_bins.add(key)

        starts.append(p0)
        ends.append(p1)

    return (np.array(starts, dtype=np.float32),
            np.array(ends,   dtype=np.float32))


def on_close(event):
    ax = event.canvas.figure.axes[0]
    elev = ax.elev
    azim = ax.azim
    print(f"\nSaving with elev={elev:.1f}°, azim={azim:.1f}° ...")
    event.canvas.figure.savefig(OUT_PATH, dpi=150, bbox_inches='tight',
                                facecolor=event.canvas.figure.get_facecolor())
    print(f"Saved → {os.path.relpath(OUT_PATH, ROOT)}")


print(f"Loading {SEQ} ...")
frames = load_frames(CHESS_DIRS[SEQ], n_frames=40, frame_step=20)
cloud  = build_cloud(frames)
starts, ends = extract_lines(cloud)
print(f"  {len(starts)} lines extracted")

rng  = np.random.default_rng(0)
pick = rng.choice(len(starts), min(200, len(starts)), replace=False)
cidx = rng.choice(len(cloud), min(8000, len(cloud)), replace=False)

fig = plt.figure(figsize=(9, 8))
fig.patch.set_facecolor(BG)
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor(BG)
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.fill = False
    pane.set_edgecolor('#cccccc')

c = cloud[cidx]
ax.scatter(c[:, 0], c[:, 1], c[:, 2], s=0.3, c='#aaaaaa', alpha=0.25, linewidths=0)

for i in pick:
    s, e = starts[i], ends[i]
    ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]],
            color=COLORS[SEQ], lw=1.6, alpha=0.88)

all_pts = np.vstack([starts[pick], ends[pick]])
lo, hi = all_pts.min(0) - 0.25, all_pts.max(0) + 0.25
r = (hi - lo).max() / 2
mid = (lo + hi) / 2
ax.set_xlim(mid[0]-r, mid[0]+r)
ax.set_ylim(mid[1]-r, mid[1]+r)
ax.set_zlim(mid[2]-r, mid[2]+r)

ax.set_title(LABELS[SEQ], color='black', fontsize=14, fontweight='bold', pad=6)
ax.tick_params(colors='#444444', labelsize=6)
ax.set_xlabel('X', color='#444444', fontsize=7)
ax.set_ylabel('Y', color='#444444', fontsize=7)
ax.set_zlabel('Z', color='#444444', fontsize=7)

fig.canvas.mpl_connect('close_event', on_close)

print("\nRotate to your preferred angle, then close the window to save.")
plt.tight_layout(pad=0.5)
plt.show()
