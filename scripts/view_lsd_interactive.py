#!/usr/bin/env python3
"""
view_lsd_interactive.py — show LSD 2D line detection on a Chess color frame.

Usage:
    python view_lsd_interactive.py seq01   # frame from seq-01
    python view_lsd_interactive.py seq03   # frame from seq-03

Close the window to save to results/presentation/lsd_seq0X.png
"""

import os, sys, glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

SEQ = sys.argv[1] if len(sys.argv) > 1 else "seq01"
assert SEQ in ("seq01", "seq03"), "Usage: python view_lsd_interactive.py seq01|seq03"

ROOT       = os.path.dirname(os.path.abspath(__file__))
CHESS_DIRS = {"seq01": "/home/rueyday/Downloads/chess/seq-01",
              "seq03": "/home/rueyday/Downloads/chess/seq-03"}
LABELS     = {"seq01": "LSD Line Segments — Sequence 01",
              "seq03": "LSD Line Segments — Sequence 03"}
OUT_PATH   = os.path.join(ROOT, "results", "presentation", f"lsd_{SEQ}.png")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

FRAME_IDX  = 200   # change if you want a different frame


def detect_lsd(img_gray):
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines, widths, _, _ = lsd.detect(img_gray)
    if lines is None:
        return np.zeros((0, 4))
    segs = lines[:, 0, :]          # (N, 4): x1 y1 x2 y2
    w    = widths[:, 0]
    # filter by minimum length
    lengths = np.linalg.norm(segs[:, 2:] - segs[:, :2], axis=1)
    mask = lengths > 15            # at least 15 px long
    return segs[mask], w[mask], lengths[mask]


def on_close(event):
    print(f"Saving → {os.path.relpath(OUT_PATH, ROOT)}")
    event.canvas.figure.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    print("Done.")


# ── load frame ────────────────────────────────────────────────────────────────
color_files = sorted(glob.glob(os.path.join(CHESS_DIRS[SEQ], "*.color.png")))
img_bgr  = cv2.imread(color_files[FRAME_IDX])
img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

segs, widths, lengths = detect_lsd(img_gray)
print(f"{len(segs)} line segments detected")

# ── colour-code by angle ───────────────────────────────────────────────────────
angles = np.arctan2(segs[:, 3] - segs[:, 1],
                    segs[:, 2] - segs[:, 0])          # -π .. π
norm_a = (angles % np.pi) / np.pi                     # 0 .. 1
colors = plt.cm.hsv(norm_a)

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: original frame
axes[0].imshow(img_rgb)
axes[0].set_title("Color frame", fontsize=12, fontweight='bold')
axes[0].axis('off')

# Right: lines overlaid
axes[1].imshow(img_rgb)
line_segs = [[(s[0], s[1]), (s[2], s[3])] for s in segs]
lc = LineCollection(line_segs, colors=colors, linewidths=np.clip(widths * 0.8, 0.8, 2.5))
axes[1].add_collection(lc)
axes[1].set_xlim(0, img_rgb.shape[1])
axes[1].set_ylim(img_rgb.shape[0], 0)
axes[1].set_title(f"{LABELS[SEQ]}  ({len(segs)} segments)", fontsize=12, fontweight='bold')
axes[1].axis('off')

fig.suptitle("LSD — Line Segment Detector (colour = orientation)",
             fontsize=11, y=1.01)
plt.tight_layout()
fig.canvas.mpl_connect('close_event', on_close)
print("Close the window to save.")
plt.show()
