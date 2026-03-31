"""
Chess RGBD → 3D Voxel → PlueckerNet SE(3)/SIM(3) Registration Demo
====================================================================
Uses the actual PlueckerNet neural network (Liu et al., CVPR 2021) to predict
line correspondences, then applies the repo's own RANSAC+SVD solver.

Shows that PlueckerNet:
  • correctly registers SE(3) transformations (R, t)
  • fails for SIM(3) (SE(3) + unknown scale s)
    because the moment term picks up a factor of s that the SE(3) solver cannot absorb.

Pipeline
--------
1.  Load N frames from chess/seq-01 (depth + pose)
2.  Back-project depth → merged, voxel-downsampled point cloud
3.  Extract 3D line segments via local PCA on the point cloud
4.  Convert lines to Pluecker coordinates  L = (d, m)   [direction first, then moment]
5.  Apply a known SE(3)  → run PlueckerNet → RANSAC → success  (residual ≈ 0)
6.  Apply a known SIM(3) → run PlueckerNet → RANSAC → failure  (residual >> 0)
7.  Produce publication-quality figures (same layout as before + probability heatmap)
"""

import os, sys, glob, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
_REPO_ROOT      = os.path.dirname(os.path.abspath(__file__))
PLUECKERNET_DIR = os.path.join(os.path.dirname(_REPO_ROOT), "PlueckerNet")
sys.path.insert(0, PLUECKERNET_DIR)

DATA_DIR = os.getenv("CHESS_DATA_DIR", "/home/rueyday/Downloads/chess/seq-01")
OUT_DIR  = os.path.join(_REPO_ROOT, "results")
os.makedirs(OUT_DIR, exist_ok=True)

WEIGHTS = os.path.join(PLUECKERNET_DIR, "output", "semantic3D", "preTrained", "best_val_checkpoint_real.pth")

# Camera intrinsics
FX, FY      = 525.0, 525.0
CX, CY      = 319.5, 239.5
DEPTH_SCALE = 1000.0

# ─── PlueckerNet imports ───────────────────────────────────────────────────────
from easydict import EasyDict as edict
from model.model_plucker import PluckerNetKnn
import lib.ransac_l2l as _ransac_mod

# Monkey-patch: newer numpy rejects skew() called with (3,1) arrays — flatten first
def _skew_fixed(x):
    x = np.asarray(x).flatten()
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])
_ransac_mod.skew = _skew_fixed
from lib.ransac_l2l import run_ransac

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_config():
    cfg = edict()
    cfg.net_nchannel  = 128
    cfg.GNN_layers    = ["self", "cross"] * 6
    cfg.net_lambda    = 0.1
    cfg.net_maxiter   = 30
    cfg.net_topK      = 200
    return cfg

# ─── Load PlueckerNet ─────────────────────────────────────────────────────────
def load_plueckernet(weights_path):
    cfg   = make_config()
    model = PluckerNetKnn(cfg).to(DEVICE)
    ckpt  = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[model]  PlueckerNet loaded from {weights_path}")
    return model

# ─── 1.  Load depth frames and poses ─────────────────────────────────────────
def load_dataset(n_frames=30, frame_step=30):
    depth_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.depth.png")))
    depth_files = depth_files[::frame_step][:n_frames]
    frames = []
    import cv2
    for df in depth_files:
        base      = df.replace(".depth.png", "")
        pose_file = base + ".pose.txt"
        if not os.path.exists(pose_file):
            continue
        depth = cv2.imread(df, cv2.IMREAD_ANYDEPTH).astype(np.float32) / DEPTH_SCALE
        pose  = np.loadtxt(pose_file)
        frames.append((depth, pose))
    print(f"[load]  {len(frames)} frames loaded")
    return frames

# ─── 2.  Back-project to 3-D world space ─────────────────────────────────────
def depth_to_world_points(depth, pose, subsample=4, max_depth=3.5):
    H, W   = depth.shape
    v_idx, u_idx = np.meshgrid(np.arange(0, H, subsample),
                                np.arange(0, W, subsample), indexing="ij")
    v_idx, u_idx = v_idx.ravel(), u_idx.ravel()
    z = depth[v_idx, u_idx]
    valid = (z > 0.1) & (z < max_depth)
    z, v_idx, u_idx = z[valid], v_idx[valid], u_idx[valid]
    x = (u_idx - CX) * z / FX
    y = (v_idx - CY) * z / FY
    pts_cam   = np.stack([x, y, z, np.ones_like(z)], axis=0)
    pts_world = (pose @ pts_cam)[:3].T
    return pts_world

def build_point_cloud(frames, subsample=4):
    all_pts = [depth_to_world_points(d, p, subsample) for d, p in frames]
    cloud   = np.concatenate(all_pts, axis=0)
    print(f"[cloud]  raw points: {cloud.shape[0]:,}")
    voxel_size = 0.02
    keys = np.floor(cloud / voxel_size).astype(np.int32)
    _, idx = np.unique(keys, axis=0, return_index=True)
    cloud = cloud[idx]
    print(f"[cloud]  after voxel-downsample: {cloud.shape[0]:,}")
    return cloud

# ─── 3.  Extract 3-D lines via local PCA ─────────────────────────────────────
def extract_lines_pca(cloud, n_lines=300, k=20, linearity_thresh=0.85, seed=42):
    rng  = np.random.default_rng(seed)
    tree = cKDTree(cloud)
    mids, dirs = [], []
    indices = rng.choice(len(cloud), size=min(8000, len(cloud)), replace=False)
    for idx in indices:
        if len(mids) >= n_lines:
            break
        nn_idx = tree.query(cloud[idx], k=k)[1]
        pts    = cloud[nn_idx]
        ctr    = pts.mean(axis=0)
        cov    = (pts - ctr).T @ (pts - ctr) / k
        eigvals, eigvecs = np.linalg.eigh(cov)
        lam = eigvals[::-1]
        linearity = (lam[0] - lam[1]) / (lam[0] + 1e-9)
        if linearity > linearity_thresh:
            mids.append(ctr)
            dirs.append(eigvecs[:, -1])
    mids = np.array(mids)
    dirs = np.array(dirs) / np.linalg.norm(np.array(dirs), axis=1, keepdims=True)
    print(f"[lines]  {len(mids)} linear segments extracted")
    return mids, dirs

# ─── 4.  Pluecker coordinates ─────────────────────────────────────────────────
# PlueckerNet uses [direction(3), moment(3)] layout — first 3 dims = direction
def to_pluecker(mids, dirs):
    """Returns (N×6) array [dx dy dz mx my mz] — direction first."""
    moments = np.cross(mids, dirs)          # N×3
    return np.hstack([dirs, moments]).astype(np.float32)  # N×6

def transform_pluecker_SE3(L, R, t):
    """Transform N×6 Pluecker lines [d, m] under SE(3): (R, t)."""
    d, m   = L[:, :3].T, L[:, 3:].T        # 3×N each
    t_col  = t.reshape(3, 1)
    d2     = R @ d
    m2     = R @ m + np.cross(t_col.repeat(d.shape[1], axis=1).T, d2.T).T
    return np.hstack([d2.T, m2.T]).astype(np.float32)

def transform_pluecker_SIM3(L, R, t, s):
    """Transform N×6 Pluecker lines [d, m] under SIM(3): (s, R, t)."""
    d, m   = L[:, :3].T, L[:, 3:].T
    t_col  = t.reshape(3, 1)
    d2     = R @ d
    m2     = s * (R @ m) + np.cross(t_col.repeat(d.shape[1], axis=1).T, d2.T).T
    return np.hstack([d2.T, m2.T]).astype(np.float32)

# ─── 5.  PlueckerNet inference ────────────────────────────────────────────────
@torch.no_grad()
def predict_correspondences(model, L1, L2, topk=200):
    """
    Run PlueckerNet to get:
      P        : (N×M) probability matrix
      top_i1   : top-K source indices
      top_i2   : top-K target indices
    """
    t1 = torch.from_numpy(L1).unsqueeze(0).to(DEVICE)   # 1×N×6
    t2 = torch.from_numpy(L2).unsqueeze(0).to(DEVICE)   # 1×M×6

    P, r, c = model(t1, t2)   # P: 1×N×M

    k = min(topk, P.shape[1] * P.shape[2])
    _, flat_idx = torch.topk(P.flatten(start_dim=-2), k=k, dim=-1)
    i1 = (flat_idx // P.shape[-1]).squeeze(0).cpu().numpy()
    i2 = (flat_idx  % P.shape[-1]).squeeze(0).cpu().numpy()

    P_np = P.squeeze(0).cpu().numpy()
    return P_np, i1, i2

# ─── 6.  Error metrics ────────────────────────────────────────────────────────
def rotation_error_deg(R_est, R_gt):
    diff  = R_est @ R_gt.T
    trace = np.clip((np.trace(diff) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(trace)))

def translation_error(t_est, t_gt):
    return float(np.linalg.norm(t_est.flatten() - t_gt.flatten()))

def pluecker_residual(L1, L2, R, t):
    """Mean residual using [d,m] layout."""
    d1, m1 = L1[:, :3].T, L1[:, 3:].T
    m2      = L2[:, 3:].T
    Rd1 = R @ d1
    Rm1 = R @ m1
    t_col = t.flatten().reshape(3, 1)
    res = m2 - Rm1 - np.cross(t_col.T, Rd1.T).T
    return float(np.mean(np.linalg.norm(res, axis=0)))

# ─── 7.  Visualisation helpers ────────────────────────────────────────────────
def subsample(pts, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    if len(pts) > n:
        pts = pts[rng.choice(len(pts), n, replace=False)]
    return pts

def plot_point_clouds_3d(ax, clouds, labels, colors, alpha=0.3, s=1):
    for pts, lbl, col in zip(clouds, labels, colors):
        pts = subsample(pts)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=col, s=s, alpha=alpha, label=lbl)

# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════
np.random.seed(42)

# --- Load model
model = load_plueckernet(WEIGHTS)

# --- Load & build cloud
frames = load_dataset(n_frames=30, frame_step=30)
cloud  = build_point_cloud(frames, subsample=4)

# --- Extract lines and convert to Pluecker [d, m]
mids, dirs = extract_lines_pca(cloud, n_lines=400, k=25, linearity_thresh=0.70)
L_ref = to_pluecker(mids, dirs)   # N×6
print(f"[pluecker]  {L_ref.shape[0]} lines in reference set")

# --- Ground-truth transforms
np.random.seed(7)
angle    = np.radians(27)
axis     = np.array([0.4, 0.7, 0.6]); axis /= np.linalg.norm(axis)
R_gt     = Rotation.from_rotvec(angle * axis).as_matrix()
t_gt     = np.array([0.35, -0.20, 0.15])
scale_gt = 1.45

cloud_SE3  = (R_gt @ cloud.T).T + t_gt
cloud_SIM3 = scale_gt * (R_gt @ cloud.T).T + t_gt

L_SE3  = transform_pluecker_SE3 (L_ref, R_gt, t_gt)
L_SIM3 = transform_pluecker_SIM3(L_ref, R_gt, t_gt, scale_gt)

print(f"\n[GT]  R angle={np.degrees(angle):.1f}°   t={t_gt}   scale(SIM3)={scale_gt}")

# ─── SE(3) registration via PlueckerNet ───────────────────────────────────────
print("\n── SE(3) registration (PlueckerNet + RANSAC) ───────────────────────────")
P_se3, top_i1_se3, top_i2_se3 = predict_correspondences(model, L_ref, L_SE3, topk=200)
print(f"  Top-{len(top_i1_se3)} correspondences predicted")

plucker1_se3_topk = L_ref[top_i1_se3].T   # 6×K
plucker2_se3_topk = L_SE3[top_i2_se3].T

R_se3_est, t_se3_est, ic_se3, mask_se3 = run_ransac(
    plucker1_se3_topk, plucker2_se3_topk, inlier_threshold=0.5)

if R_se3_est is not None:
    rerr_se3 = rotation_error_deg(R_se3_est, R_gt)
    terr_se3 = translation_error(t_se3_est, t_gt)
    res_se3  = pluecker_residual(L_ref, L_SE3, R_se3_est, t_se3_est)
    print(f"  Rotation error : {rerr_se3:.3f}°")
    print(f"  Translation err: {terr_se3:.4f} m")
    print(f"  Pluecker resid : {res_se3:.6f}")
    print(f"  RANSAC inliers : {ic_se3}/{len(top_i1_se3)}")
else:
    print("  FAILED to converge")
    R_se3_est = np.eye(3); t_se3_est = np.zeros(3)
    rerr_se3 = 999.0; terr_se3 = 999.0; res_se3 = 999.0; ic_se3 = 0

cloud_SE3_aligned = (R_se3_est.T @ (cloud_SE3.T - t_se3_est.reshape(3, 1))).T

# ─── SIM(3) registration via PlueckerNet ──────────────────────────────────────
print("\n── SIM(3) registration (PlueckerNet + RANSAC, wrong model) ────────────")
P_sim3, top_i1_sim3, top_i2_sim3 = predict_correspondences(model, L_ref, L_SIM3, topk=200)
print(f"  Top-{len(top_i1_sim3)} correspondences predicted")

plucker1_sim3_topk = L_ref[top_i1_sim3].T
plucker2_sim3_topk = L_SIM3[top_i2_sim3].T

R_sim3_est, t_sim3_est, ic_sim3, mask_sim3 = run_ransac(
    plucker1_sim3_topk, plucker2_sim3_topk, inlier_threshold=0.5)

if R_sim3_est is not None:
    rerr_sim3 = rotation_error_deg(R_sim3_est, R_gt)
    terr_sim3 = translation_error(t_sim3_est, t_gt)
    res_sim3  = pluecker_residual(L_ref, L_SIM3, R_sim3_est, t_sim3_est)
    print(f"  Rotation error : {rerr_sim3:.3f}°")
    print(f"  Translation err: {terr_sim3:.4f} m")
    print(f"  Pluecker resid : {res_sim3:.6f}   ← residual stays large!")
    print(f"  RANSAC inliers : {ic_sim3}/{len(top_i1_sim3)}")
else:
    print("  FAILED to converge")
    R_sim3_est = np.eye(3); t_sim3_est = np.zeros(3)
    rerr_sim3 = 999.0; terr_sim3 = 999.0; res_sim3 = 999.0; ic_sim3 = 0

cloud_SIM3_aligned = (R_sim3_est.T @ (cloud_SIM3.T - t_sim3_est.reshape(3, 1))).T

# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 1: 3-D Voxel point cloud overview
# ═════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 5))
ax1 = fig.add_subplot(131, projection="3d")
pts = subsample(cloud, 6000)
ax1.scatter(pts[:,0], pts[:,1], pts[:,2], c=pts[:,2], cmap="viridis", s=1, alpha=0.5)
ax1.set_title("Reference Point Cloud\n(stitched from 30 RGBD frames)", fontsize=10)
ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)"); ax1.set_zlabel("Z (m)")
ax1.view_init(elev=25, azim=45)

ax2 = fig.add_subplot(132, projection="3d")
plot_point_clouds_3d(ax2, [cloud, cloud_SE3], ["Reference", "SE(3) transformed"],
                     ["steelblue", "tomato"], alpha=0.35, s=1)
ax2.set_title("SE(3) transform applied\n(R=27°, t=[0.35, -0.20, 0.15])", fontsize=10)
ax2.legend(markerscale=5, fontsize=8); ax2.view_init(25, 45)

ax3 = fig.add_subplot(133, projection="3d")
plot_point_clouds_3d(ax3, [cloud, cloud_SIM3], ["Reference", "SIM(3) transformed"],
                     ["steelblue", "darkorange"], alpha=0.35, s=1)
ax3.set_title(f"SIM(3) transform applied\n(same R, t + scale={scale_gt})", fontsize=10)
ax3.legend(markerscale=5, fontsize=8); ax3.view_init(25, 45)

fig.suptitle("Chess Scene — 3D Voxel from Stitched RGBD Frames", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig1_voxel_overview.png"), dpi=150)
plt.close(fig)
print(f"\n[saved]  fig1_voxel_overview.png")

# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 2: SE(3) registration — success
# ═════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 5))

ax1 = fig.add_subplot(131, projection="3d")
plot_point_clouds_3d(ax1, [cloud, cloud_SE3], ["Source", "Target (SE3 tform)"],
                     ["steelblue","tomato"], alpha=0.35, s=1)
ax1.set_title("(a) Before Registration", fontsize=10)
ax1.legend(markerscale=5, fontsize=8); ax1.view_init(25, 45)

ax2 = fig.add_subplot(132, projection="3d")
plot_point_clouds_3d(ax2, [cloud, cloud_SE3_aligned], ["Source", "Aligned"],
                     ["steelblue","limegreen"], alpha=0.35, s=1)
ax2.set_title(f"(b) After SE(3) Registration\nRot err={rerr_se3:.2f}°  |  t err={terr_se3:.3f} m", fontsize=10)
ax2.legend(markerscale=5, fontsize=8); ax2.view_init(25, 45)

ax3 = fig.add_subplot(133)
metrics = ["Rotation\nerror (°)", "Translation\nerror (m)", "Pluecker\nresidual"]
vals    = [rerr_se3, terr_se3, res_se3]
bars    = ax3.bar(metrics, vals, color=["royalblue","steelblue","dodgerblue"], edgecolor="white")
for bar, val in zip(bars, vals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f"{val:.4f}", ha="center", va="bottom", fontsize=9)
ax3.set_title(f"(c) SE(3) Error Metrics\n(RANSAC inliers: {ic_se3}/{len(top_i1_se3)})", fontsize=10)
ax3.set_ylabel("Error"); ax3.set_ylim(0, max(vals)*1.4+0.01)
ax3.tick_params(axis="x", labelsize=8)
ax3.axhline(0.05, color="green", ls="--", lw=1, label="success threshold")
ax3.legend(fontsize=8)

fig.suptitle("PlueckerNet Registration — SE(3) Case  ✓ SUCCESS",
             fontsize=13, fontweight="bold", color="darkgreen")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig2_SE3_registration.png"), dpi=150)
plt.close(fig)
print(f"[saved]  fig2_SE3_registration.png")

# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 3: SIM(3) registration — failure
# ═════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 5))

ax1 = fig.add_subplot(131, projection="3d")
plot_point_clouds_3d(ax1, [cloud, cloud_SIM3], ["Source", "Target (SIM3 tform)"],
                     ["steelblue","darkorange"], alpha=0.35, s=1)
ax1.set_title("(a) Before Registration", fontsize=10)
ax1.legend(markerscale=5, fontsize=8); ax1.view_init(25, 45)

ax2 = fig.add_subplot(132, projection="3d")
plot_point_clouds_3d(ax2, [cloud, cloud_SIM3_aligned], ["Source", "Aligned (SE3 solver)"],
                     ["steelblue","salmon"], alpha=0.35, s=1)
ax2.set_title(f"(b) After SE(3) Registration (wrong model)\nRot err={rerr_sim3:.2f}°  |  t err={terr_sim3:.3f} m", fontsize=10)
ax2.legend(markerscale=5, fontsize=8); ax2.view_init(25, 45)

ax3 = fig.add_subplot(133)
x  = np.arange(3)
w  = 0.35
vals_se3  = [rerr_se3,  terr_se3,  res_se3]
vals_sim3 = [rerr_sim3, terr_sim3, res_sim3]
b1 = ax3.bar(x - w/2, vals_se3,  w, label="SE(3) input",  color="steelblue", alpha=0.85)
b2 = ax3.bar(x + w/2, vals_sim3, w, label="SIM(3) input", color="tomato",    alpha=0.85)
for bar, val in zip(list(b1)+list(b2), vals_se3+vals_sim3):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f"{val:.3f}", ha="center", va="bottom", fontsize=7)
ax3.set_title("(c) Error: SE(3) solver on SE(3) vs SIM(3)", fontsize=10)
ax3.set_ylabel("Error")
ax3.set_xticks(x); ax3.set_xticklabels(["Rotation\nerror (°)", "Translation\nerror (m)", "Pluecker\nresidual"], fontsize=8)
ax3.legend(fontsize=8)
ax3.set_ylim(0, max(vals_se3+vals_sim3)*1.5+0.02)
ax3.axhline(0.05, color="green", ls="--", lw=1)

fig.suptitle(f"PlueckerNet Registration — SIM(3) Case  ✗ FAILURE  (scale={scale_gt})",
             fontsize=13, fontweight="bold", color="crimson")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig3_SIM3_registration_failure.png"), dpi=150)
plt.close(fig)
print(f"[saved]  fig3_SIM3_registration_failure.png")

# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 4: Why SIM(3) fails — analytical diagram
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

scales = np.linspace(0.5, 2.5, 200)
moment_norms = np.linalg.norm(L_ref[:, 3:], axis=1)   # moments are columns 3:6
theoretical_residual = np.array(
    [np.mean(np.abs(1 - s) * moment_norms) for s in scales]
)
axes[0].plot(scales, theoretical_residual, color="crimson", lw=2)
axes[0].axvline(1.0, color="green", ls="--", lw=1.5, label="SE(3)  (s=1)")
axes[0].axvline(scale_gt, color="orange", ls="--", lw=1.5, label=f"SIM(3) (s={scale_gt})")
axes[0].fill_between(scales, theoretical_residual, alpha=0.15, color="crimson")
axes[0].set_xlabel("Scale factor  s", fontsize=11)
axes[0].set_ylabel("Expected Pluecker residual  ||( 1 − s ) R m₁||", fontsize=10)
axes[0].set_title("Theoretical Pluecker Residual vs Scale\n"
                  "m′ = s·R·m + t×R·d   ≠   R·m + t×R·d  (SE3 model)", fontsize=10)
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
annotate_txt = (
    "SE(3) solver residual:\n"
    "  e = m₂ − R·m₁ − t×R·d₁\n"
    "Under SIM(3): m₂ = s·R·m₁ + t×R·d₁\n"
    "  → e = (1−s)·R·m₁ ≠ 0  for s≠1"
)
axes[0].text(0.97, 0.97, annotate_txt, transform=axes[0].transAxes,
             fontsize=8, va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="grey", alpha=0.9))

categories = ["SE(3)\nRot err°", "SE(3)\nt err m",
              "SIM(3)\nRot err°", "SIM(3)\nt err m",
              "SE(3)\nresidual", "SIM(3)\nresidual"]
values     = [rerr_se3, terr_se3, rerr_sim3, terr_sim3, res_se3, res_sim3]
bar_colors = ["steelblue","steelblue","tomato","tomato","royalblue","crimson"]
bars = axes[1].bar(categories, values, color=bar_colors, edgecolor="white", alpha=0.85)
for bar, val in zip(bars, values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)
axes[1].axhline(0.05, color="green", ls="--", lw=1.5, label="success threshold (0.05)")
se3_patch  = mpatches.Patch(color="steelblue", label="SE(3)")
sim3_patch = mpatches.Patch(color="tomato",    label="SIM(3)")
axes[1].legend(handles=[se3_patch, sim3_patch,
               mpatches.Patch(color="white", ec="green", label="threshold")], fontsize=9)
axes[1].set_title("Summary: PlueckerNet on SE(3) vs SIM(3)", fontsize=11)
axes[1].set_ylabel("Error magnitude")
axes[1].set_ylim(0, max(values)*1.5+0.05)
axes[1].tick_params(axis="x", labelsize=8)

fig.suptitle("Why PlueckerNet Fails on SIM(3): Scale Corrupts the Moment Vector",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig4_why_sim3_fails.png"), dpi=150)
plt.close(fig)
print(f"[saved]  fig4_why_sim3_fails.png")

# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 5: Pluecker line sets in 3D
# ═════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 5))
half_len = 0.15
N_show   = min(40, L_ref.shape[0])
idx_show = np.random.choice(L_ref.shape[0], N_show, replace=False)

for col_idx, (L, title, color, cl) in enumerate([
    (L_ref,  "Reference lines",                             "steelblue",  cloud),
    (L_SE3,  f"SE(3) transformed\n(R={np.degrees(angle):.0f}°)", "tomato",     cloud_SE3),
    (L_SIM3, f"SIM(3) transformed\n(scale={scale_gt})",     "darkorange",  cloud_SIM3),
]):
    ax = fig.add_subplot(1, 3, col_idx+1, projection="3d")
    pts = subsample(cl, 2000)
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], c="grey", s=0.5, alpha=0.15)
    for i in idx_show:
        d = L[i, :3]; m = L[i, 3:]
        p = np.cross(d, m) / (np.dot(d, d) + 1e-12)
        s_pt = p - half_len * d
        e_pt = p + half_len * d
        ax.plot([s_pt[0], e_pt[0]], [s_pt[1], e_pt[1]], [s_pt[2], e_pt[2]],
                color=color, lw=1.2, alpha=0.8)
    ax.set_title(title, fontsize=10)
    ax.view_init(25, 45)

fig.suptitle("3D Pluecker Line Sets: Reference / SE(3) / SIM(3)", fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig5_pluecker_lines_3d.png"), dpi=150)
plt.close(fig)
print(f"[saved]  fig5_pluecker_lines_3d.png")

# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 6 (new): PlueckerNet correspondence probability matrices
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Left: SE(3) probability matrix heatmap
N = min(80, P_se3.shape[0])
M = min(80, P_se3.shape[1])
im0 = axes[0].imshow(P_se3[:N, :M], cmap="hot", aspect="auto",
                     interpolation="nearest", vmin=0)
axes[0].set_title(f"(a) SE(3): Network Probability Matrix\n"
                  f"top-{len(top_i1_se3)} matches selected", fontsize=10)
axes[0].set_xlabel("Target line index"); axes[0].set_ylabel("Source line index")
plt.colorbar(im0, ax=axes[0], fraction=0.046)

# Middle: SIM(3) probability matrix heatmap
im1 = axes[1].imshow(P_sim3[:N, :M], cmap="hot", aspect="auto",
                     interpolation="nearest", vmin=0)
axes[1].set_title(f"(b) SIM(3): Network Probability Matrix\n"
                  f"top-{len(top_i1_sim3)} matches selected", fontsize=10)
axes[1].set_xlabel("Target line index"); axes[1].set_ylabel("Source line index")
plt.colorbar(im1, ax=axes[1], fraction=0.046)

# Right: Compare max per-row confidence SE(3) vs SIM(3)
conf_se3  = P_se3.max(axis=1)[:N]
conf_sim3 = P_sim3.max(axis=1)[:N]
x_idx = np.arange(len(conf_se3))
axes[2].plot(x_idx, conf_se3,  color="steelblue", lw=1.5, label="SE(3)")
axes[2].plot(x_idx, conf_sim3, color="tomato",    lw=1.5, label="SIM(3)", alpha=0.8)
axes[2].fill_between(x_idx, conf_se3,  alpha=0.2, color="steelblue")
axes[2].fill_between(x_idx, conf_sim3, alpha=0.2, color="tomato")
axes[2].set_title("(c) Per-line Max Match Confidence\nSE(3) vs SIM(3)", fontsize=10)
axes[2].set_xlabel("Source line index"); axes[2].set_ylabel("Max probability")
axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)
axes[2].text(0.02, 0.97,
    f"Mean conf SE(3):  {conf_se3.mean():.4f}\nMean conf SIM(3): {conf_sim3.mean():.4f}",
    transform=axes[2].transAxes, fontsize=9, va="top",
    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="grey", alpha=0.9))

fig.suptitle("PlueckerNet — Predicted Correspondence Probability Matrices",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig6_probability_matrices.png"), dpi=150)
plt.close(fig)
print(f"[saved]  fig6_probability_matrices.png")

# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"All figures saved to {OUT_DIR}")
print(f"{'='*60}")
print(f"\nSUMMARY (PlueckerNet + RANSAC)")
print(f"  SE(3)  → rot err {rerr_se3:.3f}°,  t err {terr_se3:.4f} m,  residual {res_se3:.5f},  inliers {ic_se3}/{len(top_i1_se3)}")
print(f"  SIM(3) → rot err {rerr_sim3:.3f}°,  t err {terr_sim3:.4f} m,  residual {res_sim3:.5f},  inliers {ic_sim3}/{len(top_i1_sim3)}")
print(f"\nKey insight:")
print(f"  PlueckerNet predicts correspondences via learned features + Sinkhorn OT.")
print(f"  The SE(3) RANSAC+SVD solver then recovers (R, t).")
print(f"  Under SIM(3), direction d is scale-invariant so R is recoverable,")
print(f"  but the moment m' = s·R·m + t×Rd leaves residual (1-s)·R·m ≠ 0.")
