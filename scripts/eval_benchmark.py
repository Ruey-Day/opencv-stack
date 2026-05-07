#!/usr/bin/env python3
"""
eval_benchmark.py — SE3 vs Sim3 PlueckerNet comparison benchmark.

Experiments
-----------
  A1. Synthetic cube wireframe — SE3 transform (R=45°, t=[0.5,0.3,0.2])
  A2. Synthetic cube wireframe — Sim3 transform (same R,t + scale=1.8)
  B1. Chess cross-sequence (seq-01 ↔ seq-03) — RGBD, metric scale s≈1
  B2. Chess cross-sequence (seq-01 ↔ seq-03) — RGB-only, moments×1.8 (simulated monocular)

Methods compared
----------------
  M1 SE3-PlueckerNet  : original pretrained weights + SE3 RANSAC  (input: [d,m])
  M2 Sim3-PlueckerNet : our 6D weights (2026-04-12) + Sim3 RANSAC (input: [m,d])
  M3 Pure-Sim3-RANSAC : direction cosine-NN matching + Sim3 RANSAC (no network)

Outputs
-------
  results/eval/fig_eval_01_cube_lines.png
  results/eval/fig_eval_02_cube_se3.png
  results/eval/fig_eval_03_cube_sim3.png
  results/eval/fig_eval_04_chess_rgbd.png
  results/eval/fig_eval_05_chess_rgb_scale.png
  results/eval/fig_eval_06_summary.png
"""

import os, sys, glob, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT         = os.path.dirname(os.path.abspath(__file__))
PLUECKERNET  = os.path.abspath(os.path.join(ROOT, "..", "PlueckerNet"))
SE3_WEIGHTS  = os.path.join(PLUECKERNET, "output", "semantic3D",
                             "preTrained", "best_val_checkpoint_real.pth")
SIM3_WEIGHTS = os.path.join(ROOT, "output", "sim3_synthetic",
                             "2026-04-12", "best_val_checkpoint.pth")
CHESS_SEQ1   = "/home/rueyday/Downloads/chess/seq-01"
CHESS_SEQ3   = "/home/rueyday/Downloads/chess/seq-03"
OUT_DIR      = os.path.join(ROOT, "results", "eval")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, PLUECKERNET)
sys.path.insert(0, ROOT)

# ── camera intrinsics (7-Scenes) ───────────────────────────────────────────────
FX, FY, CX, CY = 525.0, 525.0, 319.5, 239.5
DEPTH_SCALE = 1000.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

METHOD_NAMES   = ["SE3-PlueckerNet", "Sim3-PlueckerNet", "Pure-Sim3-RANSAC"]
METHOD_COLORS  = ["#1f77b4", "#d62728", "#2ca02c"]   # blue, red, green

# ═══════════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_se3_model():
    from easydict import EasyDict as edict
    from model.model_plucker import PluckerNetKnn
    import lib.ransac_l2l as _rm

    def _skew_fixed(x):
        x = np.asarray(x).flatten()
        return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    _rm.skew = _skew_fixed
    from lib.ransac_l2l import run_ransac

    cfg = edict(net_nchannel=128, GNN_layers=["self", "cross"] * 6,
                net_lambda=0.1, net_maxiter=30, net_topK=200)
    model = PluckerNetKnn(cfg).to(DEVICE)
    ckpt  = torch.load(SE3_WEIGHTS, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[SE3  model] loaded  {SE3_WEIGHTS}")
    return model, run_ransac


def load_sim3_model():
    from easydict import EasyDict as edict
    from model.model_plucker import PluckerNetKnn
    cfg = edict(net_nchannel=128, GNN_layers=["self", "cross"] * 6,
                net_lambda=0.1, net_maxiter=30, net_topK=200)
    model = PluckerNetKnn(cfg).to(DEVICE)
    ckpt  = torch.load(SIM3_WEIGHTS, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[Sim3 model] loaded  {SIM3_WEIGHTS}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Line generation — Plücker coordinates
# ═══════════════════════════════════════════════════════════════════════════════
# Convention throughout this file: [m, d] format — moment in cols 0:3, direction in cols 3:6.
# The SE3 model (chess demo) uses [d, m] format; we convert before calling it.

def md_to_dm(L):   return np.hstack([L[:, 3:], L[:, :3]])
def dm_to_md(L):   return np.hstack([L[:, 3:], L[:, :3]])


def make_cube_lines_md(n_per_edge=10, cube_scale=1.0, dir_noise=0.015,
                        pos_perturb=0.06, seed=0):
    """
    120 Plücker lines along a cube wireframe (12 edges × n_per_edge).
    3 tight direction clusters (X, Y, Z) — matches our training data structure.
    Returns (120, 6) float32 in [m, d] format.
    """
    rng  = np.random.default_rng(seed)
    c    = cube_scale
    # base directions: X, Y, Z axes
    axes = [np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])]

    # 4 edges per axis: (fixed coord 1, fixed coord 2)
    edge_coords = [(c, c), (c, -c), (-c, c), (-c, -c)]

    lines = []
    for axis_idx, d_base in enumerate(axes):
        for fc1, fc2 in edge_coords:
            for _ in range(n_per_edge):
                t = rng.uniform(-c, c)
                if axis_idx == 0:    p = np.array([t,   fc1, fc2])
                elif axis_idx == 1:  p = np.array([fc1, t,   fc2])
                else:                p = np.array([fc1, fc2, t  ])
                p += rng.normal(0, pos_perturb, 3)
                d  = d_base + rng.normal(0, dir_noise, 3)
                d /= np.linalg.norm(d)
                m  = np.cross(p, d)
                lines.append(np.concatenate([m, d]))

    lines = np.array(lines, dtype=np.float32)
    idx   = rng.permutation(len(lines))
    return lines[idx]


def apply_sim3_md(L_md, s, R, t):
    """Apply Sim3(s, R, t) to lines in [m, d] format."""
    m, d    = L_md[:, :3], L_md[:, 3:]
    d_new   = (R @ d.T).T
    m_new   = s * (R @ m.T).T + np.cross(t.flatten()[None], d_new)
    return np.concatenate([m_new, d_new], axis=1).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Chess dataset loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_chess_frames(seq_dir, n_frames=25, frame_step=40):
    import cv2
    files  = sorted(glob.glob(os.path.join(seq_dir, "*.depth.png")))[::frame_step][:n_frames]
    frames = []
    for df in files:
        pf = df.replace(".depth.png", ".pose.txt")
        if not os.path.exists(pf):
            continue
        depth = cv2.imread(df, cv2.IMREAD_ANYDEPTH).astype(np.float32) / DEPTH_SCALE
        pose  = np.loadtxt(pf)
        frames.append((depth, pose))
    print(f"  [chess] {len(frames)} frames from {os.path.basename(seq_dir)}")
    return frames


def build_point_cloud(frames, subsample=4, max_depth=3.5, voxel=0.025):
    pts_all = []
    for depth, pose in frames:
        H, W  = depth.shape
        vi, ui = np.meshgrid(np.arange(0, H, subsample),
                             np.arange(0, W, subsample), indexing="ij")
        vi, ui = vi.ravel(), ui.ravel()
        z  = depth[vi, ui]
        ok = (z > 0.1) & (z < max_depth)
        z, vi, ui = z[ok], vi[ok], ui[ok]
        x   = (ui - CX) * z / FX
        y   = (vi - CY) * z / FY
        cam = np.stack([x, y, z, np.ones_like(z)], 0)
        pts_all.append((pose @ cam)[:3].T)
    cloud = np.concatenate(pts_all, 0)
    keys  = np.floor(cloud / voxel).astype(np.int32)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return cloud[idx]


def extract_lines_md(cloud, n_lines=250, k=20, linearity_thresh=0.72, seed=42):
    """Extract 3D line segments via local PCA; return [m, d] format."""
    rng  = np.random.default_rng(seed)
    tree = cKDTree(cloud)
    mids, dirs = [], []
    indices = rng.choice(len(cloud), size=min(15000, len(cloud)), replace=False)
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
    if not mids:
        return np.zeros((0, 6), dtype=np.float32)
    mids = np.array(mids, dtype=np.float32)
    dirs = np.array(dirs, dtype=np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    m = np.cross(mids, dirs)
    return np.concatenate([m, dirs], axis=1)    # [m, d]


# ═══════════════════════════════════════════════════════════════════════════════
# Network inference helpers
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _topk_from_prob(model, L1, L2, topk):
    """Run model, return top-K (i1, i2) pairs and actual K used."""
    t1 = torch.from_numpy(L1).unsqueeze(0).to(DEVICE)
    t2 = torch.from_numpy(L2).unsqueeze(0).to(DEVICE)
    P, _, _ = model(t1, t2)
    k = min(topk, P.shape[1] * P.shape[2])
    _, flat = torch.topk(P.flatten(start_dim=-2), k=k, dim=-1)
    i1 = (flat // P.shape[-1]).squeeze(0).cpu().numpy()
    i2 = (flat  % P.shape[-1]).squeeze(0).cpu().numpy()
    return i1, i2, k


def net_topk_se3(model, L1_md, L2_md, topk=200):
    """SE3 model uses [d, m] format input."""
    return _topk_from_prob(model, md_to_dm(L1_md), md_to_dm(L2_md), topk)


def net_topk_sim3(model, L1_md, L2_md, topk=100):
    """Sim3 model uses [m, d] format input."""
    return _topk_from_prob(model, L1_md, L2_md, topk)


def direction_nn_topk(L1_md, L2_md, topk=100):
    """Nearest-neighbour matching by direction cosine similarity (no network)."""
    d1  = L1_md[:, 3:]
    d2  = L2_md[:, 3:]
    sim = np.abs(d1 @ d2.T)          # (N1, N2) — abs handles anti-parallel
    k   = min(topk, sim.size)
    flat = sim.flatten()
    idx  = np.argpartition(flat, -k)[-k:]
    i1   = idx // d2.shape[0]
    i2   = idx  % d2.shape[0]
    return i1, i2, k


# ═══════════════════════════════════════════════════════════════════════════════
# Method runners — return a unified result dict
# ═══════════════════════════════════════════════════════════════════════════════

_FAIL = dict(R=np.eye(3), t=np.zeros(3), s=1.0,
             ic=0, t_net_ms=0.0, t_ransac_ms=0.0, t_total_ms=0.0)


def run_se3_net(model, ransac_fn, L1_md, L2_md, topk=200, threshold=0.5):
    t0 = time.perf_counter()
    i1, i2, _ = net_topk_se3(model, L1_md, L2_md, topk)
    t_net = (time.perf_counter() - t0) * 1000

    # SE3 RANSAC expects [d, m] as (6, K) column arrays
    L1_dm = md_to_dm(L1_md); L2_dm = md_to_dm(L2_md)
    p1, p2 = L1_dm[i1].T, L2_dm[i2].T

    t0 = time.perf_counter()
    R, t, ic, _ = ransac_fn(p1, p2, inlier_threshold=threshold)
    t_ran = (time.perf_counter() - t0) * 1000

    if R is None:
        return {**_FAIL, 't_net_ms': t_net, 't_total_ms': t_net}
    t = np.asarray(t).flatten()
    return dict(R=R, t=t, s=1.0, ic=int(ic),
                t_net_ms=t_net, t_ransac_ms=t_ran, t_total_ms=t_net + t_ran)


def run_sim3_net(model, L1_md, L2_md, topk=100, threshold=0.1):
    from sim3.ransac import run_ransac_sim3

    t0 = time.perf_counter()
    i1, i2, _ = net_topk_sim3(model, L1_md, L2_md, topk)
    t_net = (time.perf_counter() - t0) * 1000

    p1, p2 = L1_md[i1].T, L2_md[i2].T

    t0 = time.perf_counter()
    s, R, t, ic, _ = run_ransac_sim3(p1, p2, inlier_threshold=threshold)
    t_ran = (time.perf_counter() - t0) * 1000

    if R is None:
        return {**_FAIL, 't_net_ms': t_net, 't_total_ms': t_net}
    t = np.asarray(t).flatten()
    return dict(R=R, t=t, s=float(s) if s else 1.0, ic=int(ic),
                t_net_ms=t_net, t_ransac_ms=t_ran, t_total_ms=t_net + t_ran)


def run_pure_ransac(L1_md, L2_md, topk=100, threshold=0.1):
    from sim3.ransac import run_ransac_sim3

    t0 = time.perf_counter()
    i1, i2, _ = direction_nn_topk(L1_md, L2_md, topk)
    p1, p2 = L1_md[i1].T, L2_md[i2].T
    s, R, t, ic, _ = run_ransac_sim3(p1, p2, inlier_threshold=threshold)
    t_total = (time.perf_counter() - t0) * 1000

    if R is None:
        return {**_FAIL, 't_total_ms': t_total}
    t = np.asarray(t).flatten()
    return dict(R=R, t=t, s=float(s) if s else 1.0, ic=int(ic),
                t_net_ms=0.0, t_ransac_ms=t_total, t_total_ms=t_total)


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def rot_err_deg(R_est, R_gt):
    c = np.clip((np.trace(R_est.T @ R_gt) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))

def trans_err(t_est, t_gt):
    return float(np.linalg.norm(np.asarray(t_est).flatten() - np.asarray(t_gt).flatten()))

def scale_err_log(s_est, s_gt):
    if s_est is None or s_est <= 0 or s_gt <= 0:
        return float('inf')
    return abs(float(np.log(s_est)) - float(np.log(s_gt)))

def compute_metrics(res, R_gt, t_gt, s_gt):
    return dict(
        rot   = rot_err_deg(res['R'], R_gt),
        trans = trans_err(res['t'], t_gt),
        scale = scale_err_log(res['s'], s_gt),
        ic    = res['ic'],
        t_net = res['t_net_ms'],
        t_ran = res['t_ransac_ms'],
        t_tot = res['t_total_ms'],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _plucker_point(line_md):
    m, d = line_md[:3], line_md[3:]
    return np.cross(d, m) / (np.dot(d, d) + 1e-12)

def plot_lines_3d(ax, L_md, color, alpha=0.75, half=0.18, n=60, label=None, lw=1.2):
    rng = np.random.default_rng(0)
    idx = rng.choice(len(L_md), min(n, len(L_md)), replace=False)
    first = True
    for i in idx:
        p = _plucker_point(L_md[i])
        d = L_md[i, 3:]
        s, e = p - half * d, p + half * d
        kw = dict(color=color, lw=lw, alpha=alpha)
        if first and label:
            kw['label'] = label; first = False
        ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], **kw)


def bar3(ax, vals, ylabel, title, logy=False):
    x = np.arange(3)
    bars = ax.bar(x, vals, color=METHOD_COLORS, edgecolor='white', alpha=0.88, width=0.55)
    for bar, v in zip(bars, vals):
        if not np.isfinite(v): v_str = '∞'
        elif v < 0.001:        v_str = f'{v:.2e}'
        elif v < 10:           v_str = f'{v:.3f}'
        else:                  v_str = f'{v:.1f}'
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(v for v in vals if np.isfinite(v)) * 0.03,
                v_str, ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(["SE3-Net", "Sim3-Net", "Pure-RANSAC"], fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    finite = [v for v in vals if np.isfinite(v)]
    if finite:
        ax.set_ylim(0, max(finite) * 1.45 + 1e-9)
    if logy and finite and min(finite) > 0:
        ax.set_yscale('log')

def timing_bar3(ax, results, title):
    net_t   = [r['t_net_ms'] for r in results]
    ran_t   = [r['t_ransac_ms'] for r in results]
    x = np.arange(3)
    b1 = ax.bar(x, net_t, color=METHOD_COLORS, edgecolor='white', alpha=0.6, width=0.5, label='Network')
    b2 = ax.bar(x, ran_t, bottom=net_t, color=METHOD_COLORS, edgecolor='white', alpha=0.95, width=0.5, label='RANSAC', hatch='//')
    for bar, v in zip(b2, [n + r for n, r in zip(net_t, ran_t)]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + max(net_t[i] + ran_t[i] for i in range(3)) * 0.03,
                f'{v:.0f}ms', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(["SE3-Net", "Sim3-Net", "Pure-RANSAC"], fontsize=8)
    ax.set_ylabel("Time (ms)", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [saved] {os.path.relpath(path, ROOT)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment A — synthetic cube wireframe
# ═══════════════════════════════════════════════════════════════════════════════

def run_cube_experiment(model_se3, ransac_se3, model_sim3):
    print("\n" + "═" * 60)
    print("Experiment A — Synthetic Cube Wireframe")
    print("═" * 60)

    L_src = make_cube_lines_md(n_per_edge=10, cube_scale=1.0)
    print(f"  {len(L_src)} cube lines generated (3 direction clusters × 40 lines each)")

    # Ground-truth transforms
    R_gt  = Rotation.from_rotvec(np.radians(45) * np.array([0.5, 0.7, 0.4]) /
                                  np.linalg.norm([0.5, 0.7, 0.4])).as_matrix()
    t_gt  = np.array([0.50, 0.30, 0.20])
    s_gt  = 1.8

    L_se3  = apply_sim3_md(L_src, 1.0, R_gt, t_gt)
    L_sim3 = apply_sim3_md(L_src, s_gt, R_gt, t_gt)

    # ── Figure 01: cube line visualization ────────────────────────────────────
    fig = plt.figure(figsize=(15, 5))
    titles = ["Source lines (cube wireframe)",
              f"SE3 transform\n(R=45°, t=[0.5,0.3,0.2])",
              f"Sim3 transform\n(same R,t + scale={s_gt})"]
    datasets  = [L_src, L_se3, L_sim3]
    colors    = ["steelblue", "tomato", "darkorange"]
    for col_i, (L, title, clr) in enumerate(zip(datasets, titles, colors)):
        ax = fig.add_subplot(1, 3, col_i + 1, projection="3d")
        plot_lines_3d(ax, L, color=clr, n=80, half=0.25, label="lines")
        ax.set_title(title, fontsize=10)
        ax.view_init(elev=25, azim=40)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    fig.suptitle("Synthetic Cube Wireframe — Plücker Line Sets", fontsize=13, fontweight='bold')
    plt.tight_layout()
    savefig(fig, "fig_eval_01_cube_lines.png")

    # ── Experiment A1 — SE3 ────────────────────────────────────────────────────
    print("\n  A1  SE3 transform  (s=1.0)")
    res_a1 = []
    for name, fn in [
        ("SE3-Net",     lambda: run_se3_net(model_se3,  ransac_se3, L_src, L_se3,  topk=200, threshold=0.3)),
        ("Sim3-Net",    lambda: run_sim3_net(model_sim3,            L_src, L_se3,  topk=100, threshold=0.1)),
        ("Pure-RANSAC", lambda: run_pure_ransac(                    L_src, L_se3,  topk=100, threshold=0.1)),
    ]:
        r = fn()
        m = compute_metrics(r, R_gt, t_gt, 1.0)
        res_a1.append({**r, **m})
        print(f"    {name:18s}  rot={m['rot']:6.2f}°  t={m['trans']:.4f}m  "
              f"s_err={m['scale']:.3f}  ic={m['ic']}  total={m['t_tot']:.0f}ms")

    # ── Experiment A2 — Sim3 ───────────────────────────────────────────────────
    print(f"\n  A2  Sim3 transform  (s={s_gt})")
    res_a2 = []
    for name, fn in [
        ("SE3-Net",     lambda: run_se3_net(model_se3,  ransac_se3, L_src, L_sim3, topk=200, threshold=0.3)),
        ("Sim3-Net",    lambda: run_sim3_net(model_sim3,            L_src, L_sim3, topk=100, threshold=0.1)),
        ("Pure-RANSAC", lambda: run_pure_ransac(                    L_src, L_sim3, topk=100, threshold=0.1)),
    ]:
        r = fn()
        m = compute_metrics(r, R_gt, t_gt, s_gt)
        res_a2.append({**r, **m})
        print(f"    {name:18s}  rot={m['rot']:6.2f}°  t={m['trans']:.4f}m  "
              f"s_err={m['scale']:.3f}  ic={m['ic']}  total={m['t_tot']:.0f}ms")

    # ── Figure 02: SE3 results ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    bar3(axes[0], [r['rot']   for r in res_a1], "Rotation error (°)", "A1 (SE3): Rotation error")
    bar3(axes[1], [r['trans'] for r in res_a1], "Translation error (m)", "A1 (SE3): Translation error")
    bar3(axes[2], [r['scale'] for r in res_a1], "Scale error |log(ŝ/s)|", "A1 (SE3): Scale error (log-ratio)\n(GT s=1.0 → should be 0)")
    bar3(axes[3], [r['ic']    for r in res_a1], "RANSAC inliers", "A1 (SE3): Inlier count")
    timing_bar3(axes[4], res_a1, "A1 (SE3): Compute time")
    axes[2].set_ylim(0, 0.5)   # scale err for s=1 should be near 0 for all methods

    # draw GT lines after registration for SE3-Net and Sim3-Net
    fig.suptitle("Experiment A1 — Cube Wireframe, SE3 Transform (R=45°, t=[0.5,0.3,0.2])\n"
                 "All methods should solve this; scale error = 0 since s=1",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    savefig(fig, "fig_eval_02_cube_se3.png")

    # ── Figure 03: Sim3 results ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    bar3(axes[0], [r['rot']   for r in res_a2], "Rotation error (°)", "A2 (Sim3): Rotation error")
    bar3(axes[1], [r['trans'] for r in res_a2], "Translation error (m)", "A2 (Sim3): Translation error")
    bar3(axes[2], [r['scale'] for r in res_a2], "Scale error |log(ŝ/s)|", f"A2 (Sim3): Scale error (log-ratio)\n(GT s={s_gt}; SE3-Net always s=1)")
    bar3(axes[3], [r['ic']    for r in res_a2], "RANSAC inliers", "A2 (Sim3): Inlier count")
    timing_bar3(axes[4], res_a2, "A2 (Sim3): Compute time")
    fig.suptitle(f"Experiment A2 — Cube Wireframe, Sim3 Transform (scale={s_gt})\n"
                 "SE3-Net fails on translation+scale; Sim3-Net and Pure-RANSAC recover all 3",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    savefig(fig, "fig_eval_03_cube_sim3.png")

    return res_a1, res_a2, L_src, L_se3, L_sim3, R_gt, t_gt, s_gt


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment B — Chess cross-sequence
# ═══════════════════════════════════════════════════════════════════════════════

def run_chess_experiment(model_se3, ransac_se3, model_sim3):
    print("\n" + "═" * 60)
    print("Experiment B — Chess 7-Scenes cross-sequence (seq-01 ↔ seq-03)")
    print("═" * 60)

    print("  Loading seq-01 ...")
    frames1 = load_chess_frames(CHESS_SEQ1, n_frames=25, frame_step=40)
    cloud1  = build_point_cloud(frames1)
    L1      = extract_lines_md(cloud1, n_lines=300)
    print(f"  {cloud1.shape[0]:,} pts → {len(L1)} lines")

    print("  Loading seq-03 ...")
    frames3 = load_chess_frames(CHESS_SEQ3, n_frames=25, frame_step=40)
    cloud3  = build_point_cloud(frames3)
    L3      = extract_lines_md(cloud3, n_lines=300)
    print(f"  {cloud3.shape[0]:,} pts → {len(L3)} lines")

    # GT between sequences: both are in world coordinates → identity transform
    R_gt_chess = np.eye(3, dtype=np.float32)
    t_gt_chess = np.zeros(3, dtype=np.float32)
    s_rgbd     = 1.0
    s_rgb_sim  = 1.8

    # Scale-ambiguous variant: multiply seq-03 moments by s_rgb_sim
    L3_scaled = L3.copy()
    L3_scaled[:, :3] *= s_rgb_sim   # m' = s·m (simulates monocular up-to-scale)

    # ── Experiment B1 — RGBD (metric scale) ────────────────────────────────────
    print(f"\n  B1  RGBD — metric scale (GT s=1.0)")
    res_b1 = []
    for name, fn in [
        ("SE3-Net",     lambda: run_se3_net(model_se3,  ransac_se3, L1, L3,        topk=200, threshold=0.5)),
        ("Sim3-Net",    lambda: run_sim3_net(model_sim3,            L1, L3,        topk=100, threshold=0.15)),
        ("Pure-RANSAC", lambda: run_pure_ransac(                    L1, L3,        topk=150, threshold=0.15)),
    ]:
        r = fn()
        m = compute_metrics(r, R_gt_chess, t_gt_chess, s_rgbd)
        res_b1.append({**r, **m})
        print(f"    {name:18s}  rot={m['rot']:6.2f}°  t={m['trans']:.4f}m  "
              f"s_est={r['s']:.3f}  s_err={m['scale']:.3f}  ic={m['ic']}  total={m['t_tot']:.0f}ms")

    # ── Experiment B2 — RGB-only (simulated scale ambiguity) ───────────────────
    print(f"\n  B2  RGB-only — moments×{s_rgb_sim} (simulated monocular, GT s={s_rgb_sim})")
    res_b2 = []
    for name, fn in [
        ("SE3-Net",     lambda: run_se3_net(model_se3,  ransac_se3, L1, L3_scaled, topk=200, threshold=0.5)),
        ("Sim3-Net",    lambda: run_sim3_net(model_sim3,            L1, L3_scaled, topk=100, threshold=0.15)),
        ("Pure-RANSAC", lambda: run_pure_ransac(                    L1, L3_scaled, topk=150, threshold=0.15)),
    ]:
        r = fn()
        m = compute_metrics(r, R_gt_chess, t_gt_chess, s_rgb_sim)
        res_b2.append({**r, **m})
        print(f"    {name:18s}  rot={m['rot']:6.2f}°  t={m['trans']:.4f}m  "
              f"s_est={r['s']:.3f}  s_err={m['scale']:.3f}  ic={m['ic']}  total={m['t_tot']:.0f}ms")

    # ── Figure 04: Chess RGBD ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    bar3(axes[0], [r['rot']   for r in res_b1], "Rotation error (°)", "B1 (RGBD): Rotation error")
    bar3(axes[1], [r['trans'] for r in res_b1], "Translation error (m)", "B1 (RGBD): Translation error")
    bar3(axes[2], [r['scale'] for r in res_b1], "Scale error |log(ŝ/s)|", "B1 (RGBD): Scale error\n(GT s=1.0)")
    bar3(axes[3], [r['ic']    for r in res_b1], "RANSAC inliers", "B1 (RGBD): Inlier count")
    timing_bar3(axes[4], res_b1, "B1 (RGBD): Compute time")
    fig.suptitle("Experiment B1 — Chess Cross-Sequence RGBD (seq-01 ↔ seq-03)\n"
                 "Both sequences in world frame → GT: R=I, t=0, s=1",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    savefig(fig, "fig_eval_04_chess_rgbd.png")

    # ── Figure 05: Chess RGB-only scale ambiguity ──────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    bar3(axes[0], [r['rot']   for r in res_b2], "Rotation error (°)", "B2 (RGB-only): Rotation error")
    bar3(axes[1], [r['trans'] for r in res_b2], "Translation error (m)", "B2 (RGB-only): Translation error")
    bar3(axes[2], [r['scale'] for r in res_b2], "Scale error |log(ŝ/s)|", f"B2 (RGB-only): Scale error\n(GT s={s_rgb_sim}; SE3-Net always s=1)")
    bar3(axes[3], [r['ic']    for r in res_b2], "RANSAC inliers", "B2 (RGB-only): Inlier count")
    timing_bar3(axes[4], res_b2, "B2 (RGB-only): Compute time")
    fig.suptitle(f"Experiment B2 — Chess Cross-Sequence RGB-only (scale ambiguous)\n"
                 f"seq-03 moments ×{s_rgb_sim} simulates monocular reconstruction → GT s={s_rgb_sim}\n"
                 f"SE3-Net cannot recover scale; Sim3-Net and Pure-RANSAC can",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    savefig(fig, "fig_eval_05_chess_rgb_scale.png")

    return res_b1, res_b2, L1, L3, L3_scaled, cloud1, cloud3


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 06 — summary comparison across all 4 experiments
# ═══════════════════════════════════════════════════════════════════════════════

def make_summary_figure(res_a1, res_a2, res_b1, res_b2):
    import matplotlib.patches as mpatches

    experiments = ["A1 Cube SE3", "A2 Cube Sim3", "B1 Chess RGBD", "B2 Chess RGB+scale"]
    metrics_rot   = [[r['rot']   for r in res] for res in [res_a1, res_a2, res_b1, res_b2]]
    metrics_trans = [[r['trans'] for r in res] for res in [res_a1, res_a2, res_b1, res_b2]]
    metrics_scale = [[r['scale'] for r in res] for res in [res_a1, res_a2, res_b1, res_b2]]
    metrics_time  = [[r['t_tot'] for r in res] for res in [res_a1, res_a2, res_b1, res_b2]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    def grouped_bar(ax, data_per_exp, ylabel, title, ylim=None):
        n_exp  = len(experiments)
        n_meth = 3
        w      = 0.22
        x      = np.arange(n_exp)
        for mi in range(n_meth):
            vals = [data_per_exp[ei][mi] for ei in range(n_exp)]
            vals_plot = [min(v, 180) if np.isfinite(v) else 180 for v in vals]
            ax.bar(x + (mi - 1) * w, vals_plot, width=w,
                   color=METHOD_COLORS[mi], alpha=0.85, edgecolor='white',
                   label=METHOD_NAMES[mi])
        ax.set_xticks(x)
        ax.set_xticklabels(experiments, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        if ylim:
            ax.set_ylim(0, ylim)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    grouped_bar(axes[0], metrics_rot,   "Rotation error (°)",      "Rotation Error", ylim=90)
    grouped_bar(axes[1], metrics_trans, "Translation error (m)",   "Translation Error", ylim=3)
    grouped_bar(axes[2], metrics_scale, "Scale error |log(ŝ/s)|",  "Scale Error (log-ratio)")
    grouped_bar(axes[3], metrics_time,  "Total compute time (ms)", "Compute Time")

    axes[2].axhline(0.1, color='gray', ls='--', lw=1, label='acceptable threshold')
    axes[2].legend(fontsize=8)

    fig.suptitle("Summary: SE3-PlueckerNet vs Sim3-PlueckerNet vs Pure-Sim3-RANSAC\n"
                 "across 4 experiments (cube synthetic + chess cross-sequence)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    savefig(fig, "fig_eval_06_summary.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Numeric table
# ═══════════════════════════════════════════════════════════════════════════════

def print_table(res_a1, res_a2, res_b1, res_b2):
    COLS = ["rot(°)", "t(m)", "s_err", "inliers", "net(ms)", "ran(ms)", "tot(ms)"]
    EXPS = [
        ("A1 Cube SE3   (s=1.0)", res_a1),
        ("A2 Cube Sim3  (s=1.8)", res_a2),
        ("B1 Chess RGBD (s=1.0)", res_b1),
        ("B2 Chess RGB  (s=1.8)", res_b2),
    ]
    METH = METHOD_NAMES

    hdr = f"{'Experiment':<26} {'Method':<20}" + "".join(f"{c:>10}" for c in COLS)
    print("\n" + "═" * len(hdr))
    print("RESULTS TABLE")
    print("═" * len(hdr))
    print(hdr)
    print("─" * len(hdr))

    for exp_name, results in EXPS:
        for mi, (meth, r) in enumerate(zip(METH, results)):
            row  = f"{'':26} {meth:<20}" if mi > 0 else f"{exp_name:<26} {meth:<20}"
            s_err = r['scale']
            s_str = f"{s_err:.3f}" if np.isfinite(s_err) else "   ∞   "
            row += f"{r['rot']:>10.2f}{r['trans']:>10.4f}{s_str:>10}"
            row += f"{r['ic']:>10d}{r['t_net_ms']:>10.0f}{r['t_ransac_ms']:>10.0f}{r['t_total_ms']:>10.0f}"
            print(row)
        print("─" * len(hdr))


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"Device: {DEVICE}")
    print(f"Output: {OUT_DIR}")

    model_se3, ransac_se3 = load_se3_model()
    model_sim3            = load_sim3_model()

    # Experiment A — synthetic cube
    res_a1, res_a2, L_src, L_se3, L_sim3, R_gt, t_gt, s_gt = \
        run_cube_experiment(model_se3, ransac_se3, model_sim3)

    # Experiment B — chess cross-sequence
    res_b1, res_b2, L1, L3, L3_scaled, cloud1, cloud3 = \
        run_chess_experiment(model_se3, ransac_se3, model_sim3)

    # Summary figure
    make_summary_figure(res_a1, res_a2, res_b1, res_b2)

    # Print table
    print_table(res_a1, res_a2, res_b1, res_b2)

    print("\n" + "═" * 60)
    print(f"All figures saved to {os.path.relpath(OUT_DIR, ROOT)}/")
    print("═" * 60)


if __name__ == "__main__":
    main()
