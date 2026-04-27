#!/usr/bin/env python3
"""
view_lsd_cloud_interactive.py

LSD 2D line segments back-projected to 3D and stitched across frames.
This is the monocular line cloud: lines come from color image edges (LSD),
lifted to 3D via depth + pose. In true monocular mode the depth would be
unknown (moments scale-ambiguous); here depth is used only for placement.

Usage:
    python view_lsd_cloud_interactive.py seq01
    python view_lsd_cloud_interactive.py seq03

Close the window to save.
"""

import os, sys, glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib.collections import LineCollection

SEQ = sys.argv[1] if len(sys.argv) > 1 else "seq01"
assert SEQ in ("seq01", "seq03"), "Usage: ...seq01|seq03"

ROOT       = os.path.dirname(os.path.abspath(__file__))
CHESS_DIRS = {"seq01": "/home/rueyday/Downloads/chess/seq-01",
              "seq03": "/home/rueyday/Downloads/chess/seq-03"}
COLORS     = {"seq01": "#38b0ff", "seq03": "#ff6633"}
LABELS     = {"seq01": "LSD Line Cloud — Sequence 01",
              "seq03": "LSD Line Cloud — Sequence 03"}
OUT_PATH   = os.path.join(ROOT, "results", "presentation", f"lsd_cloud_{SEQ}.png")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

FX, FY, CX, CY = 525.0, 525.0, 319.5, 239.5
DEPTH_SCALE    = 1000.0


# ── LSD on one color frame ────────────────────────────────────────────────────

def detect_lsd(img_gray, min_px=20):
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines, _, _, _ = lsd.detect(img_gray)
    if lines is None:
        return np.zeros((0, 4))
    segs = lines[:, 0, :]
    lengths = np.linalg.norm(segs[:, 2:] - segs[:, :2], axis=1)
    return segs[lengths > min_px]


# ── back-project one 2D segment to 3D ────────────────────────────────────────

def backproject_segment(seg2d, depth, pose, n_samples=9,
                        max_depth=4.0, max_depth_std=0.25):
    x1, y1, x2, y2 = seg2d
    ts  = np.linspace(0, 1, n_samples)
    us  = (x1 + ts * (x2 - x1)).astype(np.float32)
    vs  = (y1 + ts * (y2 - y1)).astype(np.float32)

    H, W = depth.shape
    ui = np.clip(us.astype(int), 0, W - 1)
    vi = np.clip(vs.astype(int), 0, H - 1)

    z = depth[vi, ui]
    valid = (z > 0.1) & (z < max_depth)
    if valid.sum() < 4:
        return None
    # reject segments that cross a depth discontinuity
    if z[valid].std() > max_depth_std:
        return None

    u, v, z = us[valid], vs[valid], z[valid]
    x = (u - CX) * z / FX
    y = (v - CY) * z / FY
    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)   # (N,4)
    pts_w   = (pose @ pts_cam.T)[:3].T                        # (N,3)

    # fit 3D line to the back-projected points
    ctr = pts_w.mean(0)
    cov = (pts_w - ctr).T @ (pts_w - ctr)
    _, evec = np.linalg.eigh(cov)
    d = evec[:, -1]

    proj = (pts_w - ctr) @ d
    length_3d = proj.max() - proj.min()
    if length_3d < 0.05:        # shorter than 5 cm in 3D → skip
        return None

    return ctr + d * proj.min(), ctr + d * proj.max()


# ── load all frames and extract line cloud ────────────────────────────────────

def build_lsd_cloud(seq_dir, n_frames=30, frame_step=25):
    depth_files = sorted(glob.glob(os.path.join(seq_dir, "*.depth.png")))
    depth_files = depth_files[::frame_step][:n_frames]

    starts, ends = [], []
    for df in depth_files:
        pf = df.replace(".depth.png", ".pose.txt")
        cf = df.replace(".depth.png", ".color.png")
        if not os.path.exists(pf) or not os.path.exists(cf):
            continue

        depth = cv2.imread(df, cv2.IMREAD_ANYDEPTH).astype(np.float32) / DEPTH_SCALE
        gray  = cv2.cvtColor(cv2.imread(cf), cv2.COLOR_BGR2GRAY)
        pose  = np.loadtxt(pf)

        segs2d = detect_lsd(gray)
        for seg in segs2d:
            result = backproject_segment(seg, depth, pose)
            if result is not None:
                starts.append(result[0])
                ends.append(result[1])

    print(f"  {len(starts)} 3D segments from {len(depth_files)} frames")
    return np.array(starts, dtype=np.float32), np.array(ends, dtype=np.float32)


# ── interactive viewer ────────────────────────────────────────────────────────

def on_close(event):
    ax   = event.canvas.figure.axes[0]
    elev, azim = ax.elev, ax.azim
    print(f"\nSaving at elev={elev:.1f}° azim={azim:.1f}° ...")
    event.canvas.figure.savefig(OUT_PATH, dpi=150, bbox_inches='tight',
                                facecolor=event.canvas.figure.get_facecolor())
    print(f"Saved → {os.path.relpath(OUT_PATH, ROOT)}")


print(f"Building LSD line cloud for {SEQ} ...")
starts, ends = build_lsd_cloud(CHESS_DIRS[SEQ])

rng  = np.random.default_rng(0)
# show up to 1500 segments (subsample if more)
pick = rng.choice(len(starts), min(1500, len(starts)), replace=False)

fig = plt.figure(figsize=(9, 8))
fig.patch.set_facecolor('white')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('white')
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.fill = False
    pane.set_edgecolor('#cccccc')

col = COLORS[SEQ]
for i in pick:
    s, e = starts[i], ends[i]
    ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]],
            color=col, lw=1.2, alpha=0.70)

all_pts = np.vstack([starts[pick], ends[pick]])
lo, hi  = all_pts.min(0) - 0.2, all_pts.max(0) + 0.2
r       = (hi - lo).max() / 2
mid     = (lo + hi) / 2
ax.set_xlim(mid[0]-r, mid[0]+r)
ax.set_ylim(mid[1]-r, mid[1]+r)
ax.set_zlim(mid[2]-r, mid[2]+r)

ax.set_title(f"{LABELS[SEQ]}  ({len(starts)} segments)",
             color='black', fontsize=13, fontweight='bold', pad=6)
ax.tick_params(colors='#444444', labelsize=6)
ax.set_xlabel('X', color='#444444', fontsize=7)
ax.set_ylabel('Y', color='#444444', fontsize=7)
ax.set_zlabel('Z', color='#444444', fontsize=7)

fig.canvas.mpl_connect('close_event', on_close)
print("Rotate to your preferred angle, then close to save.")
plt.tight_layout(pad=0.5)
plt.show()
