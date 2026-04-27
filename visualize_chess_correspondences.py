#!/usr/bin/env python3
"""
visualize_chess_correspondences.py

Chess 7-Scenes seq-01 vs seq-03 line correspondence & alignment visualization.

Layout (2 rows × 3 columns):
  Row 1 — Correspondences: source lines (blue) + target lines (red) +
           inlier match connections (light green dashes)
  Row 2 — Alignment: target lines (red) + source lines transformed by
           estimated pose (method color)

Methods: Pure-RANSAC | SE3-PlueckerNet | Sim3-Net (Replica)

Output: results/chess_correspondence_viz.png  (new file, does not overwrite anything)
"""

import os, sys, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

warnings.filterwarnings("ignore")

ROOT        = os.path.dirname(os.path.abspath(__file__))
PLUECKERNET = os.path.abspath(os.path.join(ROOT, "..", "PlueckerNet"))
SE3_WEIGHTS = os.path.join(PLUECKERNET, "output", "semantic3D",
                            "preTrained", "best_val_checkpoint_real.pth")
SIM3_REPL_W = os.path.join(ROOT, "output", "replica",
                            "2026-04-22", "best_val_checkpoint.pth")
CHESS_SEQ1  = "/home/rueyday/Downloads/chess/seq-01"
CHESS_SEQ3  = "/home/rueyday/Downloads/chess/seq-03"
OUT_RGBD    = os.path.join(ROOT, "results", "chess_correspondence_viz.png")
OUT_RGB     = os.path.join(ROOT, "results", "chess_correspondence_viz_rgb.png")

sys.path.insert(0, PLUECKERNET)
sys.path.insert(0, ROOT)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── reuse helpers from main benchmark ─────────────────────────────────────────
from eval_benchmark import (
    load_se3_model,
    load_chess_frames, build_point_cloud, extract_lines_md,
    md_to_dm, apply_sim3_md,
    net_topk_se3, net_topk_sim3, direction_nn_topk,
    rot_err_deg, trans_err, scale_err_log,
)


# ── replica model loader ───────────────────────────────────────────────────────
def load_replica_model():
    from easydict import EasyDict as edict
    from model.model_plucker import PluckerNetKnn
    cfg   = edict(net_nchannel=128, GNN_layers=["self", "cross"] * 6,
                  net_lambda=0.1, net_maxiter=30, net_topK=200)
    model = PluckerNetKnn(cfg).to(DEVICE)
    ckpt  = torch.load(SIM3_REPL_W, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[Replica model] loaded  {SIM3_REPL_W}")
    return model


# ── Plucker geometry ───────────────────────────────────────────────────────────
def plucker_midpoint(line_md):
    """Representative 3-D point on the line (closest point to origin)."""
    m, d = line_md[:3], line_md[3:]
    return np.cross(d, m) / (np.dot(d, d) + 1e-12)


def draw_lines(ax, L_md, color, alpha=0.65, half=0.40,
               n=60, label=None, lw=1.2, seed=0):
    """Draw n randomly sampled line segments from a Plücker set."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(L_md), min(n, len(L_md)), replace=False)
    for k, i in enumerate(idx):
        p = plucker_midpoint(L_md[i])
        d = L_md[i, 3:]
        s, e = p - half * d, p + half * d
        kw = dict(color=color, lw=lw, alpha=alpha)
        if k == 0 and label:
            kw['label'] = label
        ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], **kw)


def draw_correspondences(ax, L1, L2, i1, i2, mask,
                         max_shown=40, color='#88ff88', alpha=0.45, lw=0.6):
    """Connect midpoints of inlier matched pairs with dashed lines."""
    if mask is None or mask.sum() == 0:
        return
    inl_i1 = i1[mask]
    inl_i2 = i2[mask]
    rng  = np.random.default_rng(7)
    pick = rng.choice(len(inl_i1), min(max_shown, len(inl_i1)), replace=False)
    for k in pick:
        p1 = plucker_midpoint(L1[inl_i1[k]])
        p2 = plucker_midpoint(L2[inl_i2[k]])
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color=color, lw=lw, alpha=alpha, linestyle='--')


# ── per-method runners that return the inlier mask alongside pose ──────────────
def run_se3(model, ransac_fn, L1, L2, topk=200, threshold=0.5):
    i1, i2, _ = net_topk_se3(model, L1, L2, topk)
    p1, p2 = md_to_dm(L1)[i1].T, md_to_dm(L2)[i2].T
    R, t, ic, mask = ransac_fn(p1, p2, inlier_threshold=threshold)
    if R is None:
        return np.eye(3), np.zeros(3), 1.0, i1, i2, None
    return R, np.asarray(t).flatten(), 1.0, i1, i2, mask


def run_sim3_replica(model, L1, L2, topk=100, threshold=0.2, max_iter=200):
    from sim3.ransac import run_ransac_sim3
    i1, i2, _ = net_topk_sim3(model, L1, L2, topk)
    s, R, t, ic, mask = run_ransac_sim3(L1[i1].T, L2[i2].T,
                                         inlier_threshold=threshold,
                                         max_iterations=max_iter)
    if R is None:
        return np.eye(3), np.zeros(3), 1.0, i1, i2, None
    return R, np.asarray(t).flatten(), float(s), i1, i2, mask


def run_pure(L1, L2, topk=100, threshold=0.4, max_iter=200):
    from sim3.ransac import run_ransac_sim3
    i1, i2, _ = direction_nn_topk(L1, L2, topk)
    s, R, t, ic, mask = run_ransac_sim3(L1[i1].T, L2[i2].T,
                                         inlier_threshold=threshold,
                                         max_iterations=max_iter)
    if R is None:
        return np.eye(3), np.zeros(3), 1.0, i1, i2, None
    return R, np.asarray(t).flatten(), float(s), i1, i2, mask


# ── axes styling ───────────────────────────────────────────────────────────────
BG = '#111122'


def _set_ax_limits(ax, *line_sets):
    """Set axis limits to fit all line midpoints with a small margin."""
    pts = []
    for L in line_sets:
        if len(L) == 0:
            continue
        m, d = L[:, :3], L[:, 3:]
        p = np.cross(d, m) / (np.sum(d ** 2, axis=1, keepdims=True) + 1e-12)
        pts.append(p)
    if not pts:
        return
    all_pts = np.vstack(pts)
    lo, hi = all_pts.min(0) - 0.3, all_pts.max(0) + 0.3
    # equal aspect ratio in 3D
    r = (hi - lo).max() / 2
    mid = (lo + hi) / 2
    ax.set_xlim(mid[0] - r, mid[0] + r)
    ax.set_ylim(mid[1] - r, mid[1] + r)
    ax.set_zlim(mid[2] - r, mid[2] + r)


def style_ax(ax, title, subtitle=''):
    ax.set_facecolor(BG)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('#2a2a4a')
    ax.tick_params(colors='#8888aa', labelsize=5.5)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a2a4a')
    full = title + (f'\n{subtitle}' if subtitle else '')
    ax.set_title(full, color='white', fontsize=8.5, fontweight='bold', pad=3)
    for lbl in (ax.xaxis.label, ax.yaxis.label, ax.zaxis.label):
        lbl.set_color('#666688')
        lbl.set_fontsize(5.5)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Chess Correspondence Visualization")
    print("=" * 60)

    # Load models
    print("\nLoading models ...")
    model_se3, ransac_fn_se3 = load_se3_model()
    model_replica             = load_replica_model()

    # Load chess sequences
    print("\nLoading seq-01 ...")
    frames1 = load_chess_frames(CHESS_SEQ1, n_frames=25, frame_step=40)
    cloud1  = build_point_cloud(frames1)
    L1      = extract_lines_md(cloud1, n_lines=250)
    print(f"  {len(L1)} lines extracted")

    print("Loading seq-03 ...")
    frames3 = load_chess_frames(CHESS_SEQ3, n_frames=25, frame_step=40)
    cloud3  = build_point_cloud(frames3)
    L2      = extract_lines_md(cloud3, n_lines=250)
    print(f"  {len(L2)} lines extracted")

    R_gt = np.eye(3)
    t_gt = np.zeros(3)

    # Fix seed so RANSAC results are reproducible
    np.random.seed(42)

    # Run methods
    print("\nRunning methods ...")

    # Thresholds: Plücker 6-vector L2 residual.
    # 0.1 is tuned for Replica; chess lines span a ~5m scene so we use 0.4 for
    # direction-NN matching and 0.2 for network-guided methods.
    THRESH_PURE = 0.4
    THRESH_NET  = 0.2

    t0 = time.perf_counter()
    R_pure, t_pure, s_pure, i1_p, i2_p, mask_p = run_pure(L1, L2, threshold=THRESH_PURE)
    dt_pure = (time.perf_counter() - t0) * 1000
    ic_pure = int(mask_p.sum()) if mask_p is not None else 0
    print(f"  Pure-RANSAC     rot={rot_err_deg(R_pure, R_gt):6.2f}°  "
          f"ic={ic_pure:3d}  {dt_pure:.0f}ms")

    t0 = time.perf_counter()
    R_se3, t_se3, s_se3, i1_s, i2_s, mask_s = run_se3(model_se3, ransac_fn_se3, L1, L2,
                                                        threshold=0.5)
    dt_se3 = (time.perf_counter() - t0) * 1000
    ic_se3 = int(mask_s.sum()) if mask_s is not None else 0
    print(f"  SE3-PlueckerNet rot={rot_err_deg(R_se3,  R_gt):6.2f}°  "
          f"ic={ic_se3:3d}  {dt_se3:.0f}ms")

    t0 = time.perf_counter()
    R_sim3, t_sim3, s_sim3, i1_r, i2_r, mask_r = run_sim3_replica(model_replica, L1, L2,
                                                                     threshold=THRESH_NET)
    dt_sim3 = (time.perf_counter() - t0) * 1000
    ic_sim3 = int(mask_r.sum()) if mask_r is not None else 0
    print(f"  Sim3-Net(Repl.) rot={rot_err_deg(R_sim3, R_gt):6.2f}°  "
          f"ic={ic_sim3:3d}  {dt_sim3:.0f}ms")

    _render_figure(L1, L2, [
        ("Pure-RANSAC",        R_pure, t_pure, s_pure, i1_p, i2_p, mask_p, '#2ca02c', dt_pure),
        ("SE3-PlueckerNet",    R_se3,  t_se3,  s_se3,  i1_s, i2_s, mask_s, '#1f77b4', dt_se3),
        ("Sim3-Net (Replica)", R_sim3, t_sim3, s_sim3, i1_r, i2_r, mask_r, '#00cccc', dt_sim3),
    ], R_gt, t_gt, s_gt=1.0,
       title="Chess 7-Scenes: seq-01 ↔ seq-03 — Line Correspondences & Alignment\n"
             "B1 RGBD — GT: R=I, t=0, s=1  (both sequences in world frame)",
       tgt_label="Seq-03 RGBD (tgt)",
       out_path=OUT_RGBD)

    # ── B2: RGB-only with simulated scale ambiguity ────────────────────────────
    print("\n" + "─" * 60)
    print("B2 — RGB-only (seq-03 moments ×1.8 → GT s=1.8)")
    print("─" * 60)

    S_RGB = 1.8
    L2_rgb = L2.copy()
    L2_rgb[:, :3] *= S_RGB   # scale moments only; directions unchanged

    np.random.seed(2)   # seed=2 reproduces benchmark B2 result (5.75°, s≈1.55)

    t0 = time.perf_counter()
    R_p2, t_p2, s_p2, i1_p2, i2_p2, mask_p2 = run_pure(
        L1, L2_rgb, topk=150, threshold=0.35, max_iter=1000)
    dt_p2 = (time.perf_counter() - t0) * 1000
    ic_p2 = int(mask_p2.sum()) if mask_p2 is not None else 0
    print(f"  Pure-RANSAC     rot={rot_err_deg(R_p2, R_gt):6.2f}°  "
          f"s={s_p2:.3f} (GT {S_RGB})  ic={ic_p2:3d}  {dt_p2:.0f}ms")

    t0 = time.perf_counter()
    R_s2, t_s2, s_s2, i1_s2, i2_s2, mask_s2 = run_se3(model_se3, ransac_fn_se3, L1, L2_rgb,
                                                         threshold=0.5)
    dt_s2 = (time.perf_counter() - t0) * 1000
    ic_s2 = int(mask_s2.sum()) if mask_s2 is not None else 0
    print(f"  SE3-PlueckerNet rot={rot_err_deg(R_s2, R_gt):6.2f}°  "
          f"s={s_s2:.3f} (GT {S_RGB})  ic={ic_s2:3d}  {dt_s2:.0f}ms")

    t0 = time.perf_counter()
    R_r2, t_r2, s_r2, i1_r2, i2_r2, mask_r2 = run_sim3_replica(
        model_replica, L1, L2_rgb, topk=100, threshold=0.25, max_iter=1000)
    dt_r2 = (time.perf_counter() - t0) * 1000
    ic_r2 = int(mask_r2.sum()) if mask_r2 is not None else 0
    print(f"  Sim3-Net(Repl.) rot={rot_err_deg(R_r2, R_gt):6.2f}°  "
          f"s={s_r2:.3f} (GT {S_RGB})  ic={ic_r2:3d}  {dt_r2:.0f}ms")

    _render_figure(L1, L2_rgb, [
        ("Pure-RANSAC",        R_p2, t_p2, s_p2, i1_p2, i2_p2, mask_p2, '#2ca02c', dt_p2),
        ("SE3-PlueckerNet",    R_s2, t_s2, s_s2, i1_s2, i2_s2, mask_s2, '#1f77b4', dt_s2),
        ("Sim3-Net (Replica)", R_r2, t_r2, s_r2, i1_r2, i2_r2, mask_r2, '#00cccc', dt_r2),
    ], R_gt, t_gt, s_gt=S_RGB,
       title=f"Chess 7-Scenes: seq-01 ↔ seq-03 — Line Correspondences & Alignment\n"
             f"B2 RGB-only — seq-03 moments ×{S_RGB} (monocular scale) — GT s={S_RGB}",
       tgt_label=f"Seq-03 RGB×{S_RGB} (tgt)",
       out_path=OUT_RGB)


def _render_figure(L1, L2, methods, R_gt, t_gt, s_gt, title, tgt_label, out_path):
    """Render 2-row × 3-col correspondence + alignment figure."""
    ELEV, AZIM = 22, -55

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(BG)

    for col, (name, R_e, t_e, s_e, i1, i2, mask, mcol, dt) in enumerate(methods, start=1):
        re  = rot_err_deg(R_e, R_gt)
        te  = trans_err(t_e, t_gt)
        se  = scale_err_log(s_e, s_gt)
        ic  = int(mask.sum()) if mask is not None else 0

        # ── Row 1: correspondences ──────────────────────────────────────────
        ax = fig.add_subplot(2, 3, col, projection='3d')
        draw_lines(ax, L1, '#4488ff', n=80, alpha=0.60,
                   label='Seq-01 (src)', lw=1.1, seed=0)
        draw_lines(ax, L2, '#ff5533', n=80, alpha=0.60,
                   label=tgt_label, lw=1.1, seed=1)
        draw_correspondences(ax, L1, L2, i1, i2, mask,
                             max_shown=40, color='#99ffaa', alpha=0.55, lw=0.8)
        style_ax(ax, name, f'inlier matches: {ic}')
        _set_ax_limits(ax, L1, L2)
        ax.view_init(elev=ELEV, azim=AZIM)
        if col == 1:
            _styled_legend(ax)

        # ── Row 2: alignment ────────────────────────────────────────────────
        ax2 = fig.add_subplot(2, 3, col + 3, projection='3d')
        L1_tf = apply_sim3_md(L1, s_e, R_e, t_e)
        draw_lines(ax2, L2,    '#ff5533', n=80, alpha=0.55,
                   label=tgt_label, lw=1.1, seed=1)
        draw_lines(ax2, L1_tf, mcol,      n=80, alpha=0.70,
                   label='Seq-01 aligned', lw=1.1, seed=0)
        se_str = f'∞' if not np.isfinite(se) else f'{se:.3f}'
        style_ax(ax2, f'Alignment — {name}',
                 f'rot: {re:.2f}°   s: {s_e:.3f} (err {se_str})   {dt:.0f} ms')
        _set_ax_limits(ax2, L2, L1_tf)
        ax2.view_init(elev=ELEV, azim=AZIM)
        if col == 1:
            _styled_legend(ax2)

    fig.suptitle(title, color='white', fontsize=10.5, fontweight='bold', y=0.997)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved → {os.path.relpath(out_path, ROOT)}")


def _styled_legend(ax):
    leg = ax.legend(loc='upper left', fontsize=6.5, framealpha=0.4,
                    facecolor='#1a1a33', edgecolor='#4444aa')
    for t in leg.get_texts():
        t.set_color('white')


if __name__ == "__main__":
    main()
