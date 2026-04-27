#!/usr/bin/env python3
"""
view_registration_interactive.py

Loads LSD line clouds from seq-01 and seq-03, runs the Sim3 model to get
correspondences, runs RANSAC for s/R/t, and shows:
  Left  — both line clouds + inlier correspondence lines
  Right — seq-01 transformed onto seq-03 (registered)

Usage:
    python view_registration_interactive.py
Close the window to save to results/presentation/registration.png
"""

import os, sys, glob
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

ROOT        = os.path.dirname(os.path.abspath(__file__))
PLUECKERNET = os.path.abspath(os.path.join(ROOT, "..", "PlueckerNet"))
SIM3_W      = os.path.join(ROOT, "output", "replica", "2026-04-22", "best_val_checkpoint.pth")
CHESS_SEQ1  = "/home/rueyday/Downloads/chess/seq-01"
CHESS_SEQ3  = "/home/rueyday/Downloads/chess/seq-03"
OUT_PATH    = os.path.join(ROOT, "results", "presentation", "registration.png")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

sys.path.insert(0, PLUECKERNET)
sys.path.insert(0, ROOT)

FX, FY, CX, CY = 525.0, 525.0, 319.5, 239.5
DEPTH_SCALE    = 1000.0
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_LINES        = 400   # lines per sequence fed to the model
TOPK           = 120


# ── model ─────────────────────────────────────────────────────────────────────

def load_model():
    from easydict import EasyDict as edict
    from model.model_plucker import PluckerNetKnn
    cfg   = edict(net_nchannel=128, GNN_layers=["self","cross"]*6,
                  net_lambda=0.1, net_maxiter=30, net_topK=200)
    model = PluckerNetKnn(cfg).to(DEVICE)
    ckpt  = torch.load(SIM3_W, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Model loaded from {os.path.relpath(SIM3_W, ROOT)}")
    return model


# ── LSD line cloud ─────────────────────────────────────────────────────────────

def detect_lsd(img_gray, min_px=20):
    lsd  = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines, _, _, _ = lsd.detect(img_gray)
    if lines is None:
        return np.zeros((0, 4))
    segs    = lines[:, 0, :]
    lengths = np.linalg.norm(segs[:, 2:] - segs[:, :2], axis=1)
    return segs[lengths > min_px]


def backproject_segment(seg2d, depth, pose, n_samples=9,
                        max_depth=4.0, max_depth_std=0.25):
    x1, y1, x2, y2 = seg2d
    ts  = np.linspace(0, 1, n_samples)
    us  = (x1 + ts*(x2-x1))
    vs  = (y1 + ts*(y2-y1))
    H, W = depth.shape
    ui = np.clip(us.astype(int), 0, W-1)
    vi = np.clip(vs.astype(int), 0, H-1)
    z  = depth[vi, ui]
    valid = (z > 0.1) & (z < max_depth)
    if valid.sum() < 4:
        return None
    if z[valid].std() > max_depth_std:
        return None
    u, v, z = us[valid], vs[valid], z[valid]
    x = (u - CX) * z / FX
    y = (v - CY) * z / FY
    pts_cam = np.stack([x, y, z, np.ones_like(z)], 1)
    pts_w   = (pose @ pts_cam.T)[:3].T
    ctr     = pts_w.mean(0)
    cov     = (pts_w - ctr).T @ (pts_w - ctr)
    _, evec = np.linalg.eigh(cov)
    d    = evec[:, -1]
    proj = (pts_w - ctr) @ d
    length = proj.max() - proj.min()
    if length < 0.05:
        return None
    return ctr + d*proj.min(), ctr + d*proj.max()


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
        for seg in detect_lsd(gray):
            r = backproject_segment(seg, depth, pose)
            if r is not None:
                starts.append(r[0]); ends.append(r[1])
    return np.array(starts, np.float32), np.array(ends, np.float32)


# ── Plücker conversion ─────────────────────────────────────────────────────────

def endpoints_to_plucker(starts, ends):
    """Endpoints → [m, d] Plücker (N,6)."""
    d = ends - starts
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    m = np.cross(starts, d)
    return np.concatenate([m, d], axis=1).astype(np.float32)


def apply_sim3(starts, ends, s, R, t):
    """Transform segment endpoints by Sim3(s, R, t)."""
    t = t.flatten()
    s_new = (s * R @ starts.T + t[:, None]).T
    e_new = (s * R @ ends.T   + t[:, None]).T
    return s_new, e_new


# ── network inference ─────────────────────────────────────────────────────────

def run_model(model, L1, L2):
    t1 = torch.from_numpy(L1).unsqueeze(0).to(DEVICE)
    t2 = torch.from_numpy(L2).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        P, _, _ = model(t1, t2)
    k = min(TOPK, P.shape[1] * P.shape[2])
    _, flat = torch.topk(P.flatten(start_dim=-2), k=k, dim=-1)
    i1 = (flat // P.shape[-1]).squeeze(0).cpu().numpy()
    i2 = (flat  % P.shape[-1]).squeeze(0).cpu().numpy()
    return i1, i2


# ── 3D drawing helpers ─────────────────────────────────────────────────────────

def draw_segments(ax, starts, ends, color, alpha=0.55, lw=1.0,
                  n=400, seed=0, label=None):
    rng  = np.random.default_rng(seed)
    pick = rng.choice(len(starts), min(n, len(starts)), replace=False)
    for k, i in enumerate(pick):
        s, e = starts[i], ends[i]
        kw = dict(color=color, lw=lw, alpha=alpha)
        if k == 0 and label:
            kw['label'] = label
        ax.plot([s[0],e[0]], [s[1],e[1]], [s[2],e[2]], **kw)


def draw_correspondences(ax, starts1, ends1, starts2, ends2, i1, i2, mask,
                         max_show=50, color='#44ff88', alpha=0.5, lw=0.7):
    if mask is None or mask.sum() == 0:
        return
    inl1, inl2 = i1[mask], i2[mask]
    rng  = np.random.default_rng(3)
    pick = rng.choice(len(inl1), min(max_show, len(inl1)), replace=False)
    for k in pick:
        p1 = (starts1[inl1[k]] + ends1[inl1[k]]) / 2
        p2 = (starts2[inl2[k]] + ends2[inl2[k]]) / 2
        ax.plot([p1[0],p2[0]], [p1[1],p2[1]], [p1[2],p2[2]],
                color=color, lw=lw, alpha=alpha, linestyle='--')


def set_limits(ax, *pt_sets):
    all_pts = np.vstack(pt_sets)
    lo, hi  = all_pts.min(0) - 0.2, all_pts.max(0) + 0.2
    r   = (hi - lo).max() / 2
    mid = (lo + hi) / 2
    ax.set_xlim(mid[0]-r, mid[0]+r)
    ax.set_ylim(mid[1]-r, mid[1]+r)
    ax.set_zlim(mid[2]-r, mid[2]+r)


def style_ax(ax, title, subtitle=''):
    ax.set_facecolor('white')
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False; pane.set_edgecolor('#cccccc')
    full = title + (f'\n{subtitle}' if subtitle else '')
    ax.set_title(full, fontsize=10, fontweight='bold', pad=4)
    ax.tick_params(colors='#555555', labelsize=5.5)
    ax.set_xlabel('X', fontsize=6); ax.set_ylabel('Y', fontsize=6); ax.set_zlabel('Z', fontsize=6)


def on_close(event):
    print(f"Saving → {os.path.relpath(OUT_PATH, ROOT)}")
    event.canvas.figure.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    print("Done.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    model = load_model()

    print("\nBuilding LSD cloud seq-01 ...")
    s1_all, e1_all = build_lsd_cloud(CHESS_SEQ1)
    print(f"  {len(s1_all)} segments")

    print("Building LSD cloud seq-03 ...")
    s3_all, e3_all = build_lsd_cloud(CHESS_SEQ3)
    print(f"  {len(s3_all)} segments")

    # subsample for model input
    rng  = np.random.default_rng(42)
    idx1 = rng.choice(len(s1_all), min(N_LINES, len(s1_all)), replace=False)
    idx3 = rng.choice(len(s3_all), min(N_LINES, len(s3_all)), replace=False)
    s1, e1 = s1_all[idx1], e1_all[idx1]
    s3, e3 = s3_all[idx3], e3_all[idx3]

    L1 = endpoints_to_plucker(s1, e1)
    L2 = endpoints_to_plucker(s3, e3)

    # run model → correspondences
    print("\nRunning Sim3 model ...")
    i1, i2 = run_model(model, L1, L2)

    # RANSAC → s, R, t
    from sim3.ransac import run_ransac_sim3
    s_est, R_est, t_est, n_inliers, mask = run_ransac_sim3(
        L1[i1].T, L2[i2].T, inlier_threshold=0.25, max_iterations=500)

    if R_est is None:
        print("RANSAC failed — showing correspondences only")
        s_est, R_est, t_est = 1.0, np.eye(3), np.zeros(3)
        mask = np.zeros(len(i1), dtype=bool)
        n_inliers = 0

    t_est = np.asarray(t_est).flatten()
    print(f"  inliers: {n_inliers} / {TOPK}   s={s_est:.3f}")

    # transform seq-01 endpoints for registration panel
    s1_reg, e1_reg = apply_sim3(s1, e1, s_est, R_est, t_est)

    # ── figure ────────────────────────────────────────────────────────────────
    ELEV, AZIM = 20, -50

    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor('white')

    # Left: before registration + correspondences
    ax1 = fig.add_subplot(121, projection='3d')
    draw_segments(ax1, s1, e1, '#2288ff', alpha=0.50, n=400, seed=0, label='Seq-01')
    draw_segments(ax1, s3, e3, '#ff4422', alpha=0.50, n=400, seed=1, label='Seq-03')
    draw_correspondences(ax1, s1, e1, s3, e3, i1, i2, mask, max_show=60)
    style_ax(ax1, 'Correspondences',
             f'{n_inliers} inlier matches  (top-{TOPK} from Sim3 model)')
    set_limits(ax1, s1, e1, s3, e3)
    ax1.view_init(elev=ELEV, azim=AZIM)
    leg = ax1.legend(fontsize=7, loc='upper left', framealpha=0.6)

    # Right: after registration
    ax2 = fig.add_subplot(122, projection='3d')
    draw_segments(ax2, s1_reg, e1_reg, '#2288ff', alpha=0.55, n=400, seed=0,
                  label=f'Seq-01 registered (s={s_est:.2f})')
    draw_segments(ax2, s3,     e3,     '#ff4422', alpha=0.55, n=400, seed=1,
                  label='Seq-03 (target)')
    style_ax(ax2, 'After Registration',
             f's={s_est:.3f}  R≈{"I" if np.allclose(R_est,np.eye(3),atol=0.1) else "…"}')
    set_limits(ax2, s1_reg, e1_reg, s3, e3)
    ax2.view_init(elev=ELEV, azim=AZIM)
    ax2.legend(fontsize=7, loc='upper left', framealpha=0.6)

    fig.suptitle('Chess 7-Scenes: seq-01 ↔ seq-03   |   LSD line cloud + Sim3-Net (Replica)',
                 fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout(pad=1.0)
    fig.canvas.mpl_connect('close_event', on_close)
    print("\nRotate to your preferred angle, then close to save.")
    plt.show()


if __name__ == "__main__":
    main()
