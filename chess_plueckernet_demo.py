"""
Chess RGBD → 3D Voxel → Pluecker-line SE(3)/SIM(3) Registration Demo
=====================================================================
Shows that Pluecker-coordinate line registration (as in PlueckerNet)
correctly handles SE(3) transformations but *fails* for SIM(3) (i.e.
SE(3) + unknown scale), because the moment term of a Pluecker line
picks up a factor of s under scaling, which the SE(3) solver cannot
absorb into any translation.

Pipeline
--------
1.  Load N frames from chess/seq-01 (depth + pose)
2.  Back-project depth to 3D world space → merged point cloud
3.  Extract 3D line segments via local PCA on the point cloud
4.  Convert lines to Pluecker coordinates  L = (m, d)
5.  Apply a known SE(3)  and register  → success  (residual ≈ 0)
6.  Apply a known SIM(3) and register  → failure  (residual >> 0)
7.  Produce publication-quality figures
"""

import os, sys, glob, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.getenv("CHESS_DATA_DIR", "/home/rueyday/Downloads/chess/seq-01")
OUT_DIR    = os.path.join(_REPO_ROOT, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# Camera intrinsics from dataset_params.yaml
FX, FY = 525.0, 525.0
CX, CY = 319.5, 239.5
DEPTH_SCALE = 1000.0          # chess depth is stored as uint16 mm → m

# ─── 1.  Load depth frames and poses ─────────────────────────────────────────
def load_dataset(n_frames=30, frame_step=30):
    """Return list of (depth_mm, pose_4x4) tuples."""
    depth_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.depth.png")))
    depth_files = depth_files[::frame_step][:n_frames]

    frames = []
    import cv2
    for df in depth_files:
        base = df.replace(".depth.png", "")
        pose_file = base + ".pose.txt"
        if not os.path.exists(pose_file):
            continue
        depth = cv2.imread(df, cv2.IMREAD_ANYDEPTH).astype(np.float32) / DEPTH_SCALE
        pose  = np.loadtxt(pose_file)                # 4×4 camera-to-world
        frames.append((depth, pose))
    print(f"[load]  {len(frames)} frames loaded")
    return frames

# ─── 2.  Back-project to 3-D world space ─────────────────────────────────────
def depth_to_world_points(depth, pose, subsample=4, max_depth=3.5):
    """Back-project one depth frame to world-space XYZ (N×3)."""
    H, W = depth.shape
    v_idx, u_idx = np.meshgrid(np.arange(0, H, subsample),
                                np.arange(0, W, subsample), indexing="ij")
    v_idx, u_idx = v_idx.ravel(), u_idx.ravel()
    z = depth[v_idx, u_idx]
    valid = (z > 0.1) & (z < max_depth)
    z, v_idx, u_idx = z[valid], v_idx[valid], u_idx[valid]

    x = (u_idx - CX) * z / FX
    y = (v_idx - CY) * z / FY
    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=0)   # 4×N
    pts_world = (pose @ pts_cam)[:3].T                         # N×3
    return pts_world

def build_point_cloud(frames, subsample=4):
    all_pts = [depth_to_world_points(d, p, subsample) for d, p in frames]
    cloud = np.concatenate(all_pts, axis=0)
    print(f"[cloud]  raw points: {cloud.shape[0]:,}")
    # Voxel-downsample (simple grid)
    voxel_size = 0.02
    keys = np.floor(cloud / voxel_size).astype(np.int32)
    _, idx = np.unique(keys, axis=0, return_index=True)
    cloud = cloud[idx]
    print(f"[cloud]  after voxel-downsample ({voxel_size*100:.0f} cm): {cloud.shape[0]:,}")
    return cloud

# ─── 3.  Extract 3-D lines via local PCA ─────────────────────────────────────
def extract_lines_pca(cloud, n_lines=300, k=20, linearity_thresh=0.85, seed=42):
    """
    Sample candidate seed points; keep those whose k-NN neighbourhood
    is strongly linear (PCA linearity  = (λ1-λ2)/λ1 > threshold).
    Returns (midpoints N×3, directions N×3).
    """
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
        eigvals, eigvecs = np.linalg.eigh(cov)        # ascending order
        lam = eigvals[::-1]                            # descending
        linearity = (lam[0] - lam[1]) / (lam[0] + 1e-9)
        if linearity > linearity_thresh:
            mids.append(ctr)
            dirs.append(eigvecs[:, -1])                # first eigenvector

    mids = np.array(mids)
    dirs = np.array(dirs)
    # Normalise directions
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    print(f"[lines]  {len(mids)} linear segments extracted")
    return mids, dirs

# ─── 4.  Pluecker coordinate helpers ─────────────────────────────────────────
def to_pluecker(mids, dirs):
    """
    Pluecker coords: L = (moment m = p × d,  direction d)
    Returns (6×N) array, columns [mx my mz dx dy dz].
    """
    moments = np.cross(mids, dirs)                     # N×3
    return np.hstack([moments, dirs]).T                # 6×N

def transform_pluecker_SE3(L, R, t):
    """Transform Pluecker lines under SE(3): (R,t)."""
    m, d = L[:3], L[3:]                                # 3×N each
    t_col = t.reshape(3, 1)
    d2 = R @ d
    m2 = R @ m + np.cross(t_col.repeat(d.shape[1], axis=1).T, d2.T).T
    return np.vstack([m2, d2])

def transform_pluecker_SIM3(L, R, t, s):
    """Transform Pluecker lines under SIM(3): (s·R, t)."""
    m, d = L[:3], L[3:]
    t_col = t.reshape(3, 1)
    d2 = R @ d                                         # direction unchanged (unit)
    m2 = s * (R @ m) + np.cross(t_col.repeat(d.shape[1], axis=1).T, d2.T).T
    return np.vstack([m2, d2])

# ─── 5.  PlueckerNet-style SE(3) solver (RANSAC + closed-form) ───────────────
def rotation_from_direction_pairs(d1, d2):
    """Kabsch / SVD rotation aligning direction vectors d1 → d2."""
    M  = d2 @ d1.T
    U, _, Vh = np.linalg.svd(M)
    R  = U @ Vh
    if np.linalg.det(R) < 0:
        Vh[-1] *= -1
        R = U @ Vh
    return R

def ransac_rotation(L1, L2, iters=500, ang_thr_deg=2.0):
    d1, d2 = L1[3:], L2[3:]
    cos_thr = np.cos(np.radians(ang_thr_deg)) ** 2
    N = d1.shape[1]
    best_R, best_mask, best_ic = np.eye(3), None, 0
    for _ in range(iters):
        idx  = np.random.choice(N, 2, replace=False)
        R_c  = rotation_from_direction_pairs(d1[:, idx], d2[:, idx])
        d1_r = R_c @ d1
        alignment = np.sum(d1_r * d2, axis=0) ** 2
        mask = alignment > cos_thr
        ic   = mask.sum()
        if ic > best_ic:
            best_ic, best_mask, best_R = ic, mask, R_c
    return best_R, best_mask

def solve_translation(L1, L2, R, mask):
    """
    Given R fixed, solve for t via the Pluecker constraint:
        t × (R d1) = m2 − R m1
    This is a linear system A t = b (3N equations, 3 unknowns).
    """
    inl1 = L1[:, mask]
    inl2 = L2[:, mask]
    m1, d1 = inl1[:3], inl1[3:]
    m2      = inl2[:3]

    Rd1 = R @ d1                   # 3×K
    Rm1 = R @ m1                   # 3×K
    rhs = m2 - Rm1                  # 3×K

    # t × Rd1  =  rhs   →  skew(Rd1)^T t = rhs
    # skew(v) @ t = v × t  →  skew(v)[i] = ...
    K = Rd1.shape[1]
    A = np.zeros((3*K, 3))
    b = np.zeros(3*K)
    for i in range(K):
        v = Rd1[:, i]
        S = np.array([[ 0,  v[2], -v[1]],
                      [-v[2], 0,   v[0]],
                      [ v[1],-v[0],  0]])
        A[3*i:3*i+3] = S
        b[3*i:3*i+3] = rhs[:, i]
    t, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return t

def register_SE3_pluecker(L1, L2, iters=500):
    """Return (R_est, t_est, inlier_mask)."""
    R, mask = ransac_rotation(L1, L2, iters=iters)
    if mask is None or mask.sum() < 3:
        return None, None, None
    t = solve_translation(L1, L2, R, mask)
    # One refinement pass with all current inliers
    R_ref = rotation_from_direction_pairs(
        (R @ L1[3:][:, mask]),
        L2[3:][:, mask]
    )
    t_ref = solve_translation(L1, L2, R_ref @ R, mask)
    return R_ref @ R, t_ref, mask

# ─── 6.  Error metrics ────────────────────────────────────────────────────────
def rotation_error_deg(R_est, R_gt):
    diff  = R_est @ R_gt.T
    trace = np.clip((np.trace(diff) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(trace))

def translation_error(t_est, t_gt):
    return float(np.linalg.norm(t_est - t_gt))

def pluecker_residual(L1, L2, R, t):
    """Mean Pluecker constraint residual ||m2 − R m1 − t×R d1||."""
    m1, d1 = L1[:3], L1[3:]
    m2      = L2[:3]
    Rd1 = R @ d1
    Rm1 = R @ m1
    t_col = t.reshape(3, 1)
    residual = m2 - Rm1 - np.cross(t_col.T, Rd1.T).T
    return float(np.mean(np.linalg.norm(residual, axis=0)))

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

def arrow3d(ax, start, end, color, lw=2):
    ax.plot(*zip(start, end), color=color, lw=lw)
    ax.quiver(*start, *(end - start), color=color, length=0.0, arrow_length_ratio=0.3)

# ─── Main ─────────────────────────────────────────────────────────────────────
np.random.seed(42)

# --- Load & build cloud
frames = load_dataset(n_frames=30, frame_step=30)
cloud  = build_point_cloud(frames, subsample=4)

# --- Extract lines
mids, dirs = extract_lines_pca(cloud, n_lines=400, k=25, linearity_thresh=0.70)
L_ref = to_pluecker(mids, dirs)   # reference Pluecker lines (6×N)
print(f"[pluecker]  {L_ref.shape[1]} lines in reference set")

# --- Define ground-truth SE(3) and SIM(3) transforms
np.random.seed(7)
angle  = np.radians(27)
axis   = np.array([0.4, 0.7, 0.6]); axis /= np.linalg.norm(axis)
R_gt   = Rotation.from_rotvec(angle * axis).as_matrix()
t_gt   = np.array([0.35, -0.20, 0.15])
scale_gt = 1.45                    # SIM(3) scale factor

# Transform the point cloud
cloud_SE3  = (R_gt @ cloud.T).T + t_gt
cloud_SIM3 = scale_gt * (R_gt @ cloud.T).T + t_gt

# Transform the Pluecker lines
L_SE3  = transform_pluecker_SE3 (L_ref, R_gt, t_gt)
L_SIM3 = transform_pluecker_SIM3(L_ref, R_gt, t_gt, scale_gt)

print(f"\n[GT]  R angle={np.degrees(angle):.1f}°   t={t_gt}   scale(SIM3)={scale_gt}")

# --- SE(3) registration
print("\n── SE(3) registration ──────────────────────────────────────────────────")
R_se3_est, t_se3_est, mask_se3 = register_SE3_pluecker(L_ref, L_SE3)
if R_se3_est is not None:
    rerr_se3 = rotation_error_deg(R_se3_est, R_gt)
    terr_se3 = translation_error(t_se3_est, t_gt)
    res_se3  = pluecker_residual(L_ref, L_SE3, R_se3_est, t_se3_est)
    print(f"  Rotation error : {rerr_se3:.3f}°")
    print(f"  Translation err: {terr_se3:.4f} m")
    print(f"  Pluecker resid : {res_se3:.6f}")
    print(f"  Inliers        : {mask_se3.sum()}/{L_ref.shape[1]}")
else:
    print("  FAILED to converge")
    R_se3_est = np.eye(3); t_se3_est = np.zeros(3)
    rerr_se3 = 999.0; terr_se3 = 999.0; res_se3 = 999.0

# Apply estimated SE(3) to align cloud back
cloud_SE3_aligned = (R_se3_est @ cloud_SE3.T).T + (
    -R_se3_est @ t_se3_est  # inverse: R^T (x - t)
)
# actually apply inverse: R_est^T (p - t_est)
cloud_SE3_aligned = (R_se3_est.T @ (cloud_SE3.T - t_se3_est.reshape(3,1))).T

# --- SIM(3) registration (same SE(3) solver, wrong model)
print("\n── SIM(3) registration (SE(3) solver — wrong model) ────────────────────")
R_sim3_est, t_sim3_est, mask_sim3 = register_SE3_pluecker(L_ref, L_SIM3)
if R_sim3_est is not None:
    # Rotation CAN be recovered (directions are scale-invariant)
    rerr_sim3 = rotation_error_deg(R_sim3_est, R_gt)
    # Translation will be wrong because moment is corrupted by scale
    terr_sim3 = translation_error(t_sim3_est, t_gt)
    res_sim3  = pluecker_residual(L_ref, L_SIM3, R_sim3_est, t_sim3_est)
    print(f"  Rotation error : {rerr_sim3:.3f}°")
    print(f"  Translation err: {terr_sim3:.4f} m")
    print(f"  Pluecker resid : {res_sim3:.6f}   ← residual stays large!")
    print(f"  Inliers        : {mask_sim3.sum()}/{L_ref.shape[1]}")
else:
    print("  FAILED to converge")
    R_sim3_est = np.eye(3); t_sim3_est = np.zeros(3)
    rerr_sim3 = 999.0; terr_sim3 = 999.0; res_sim3 = 999.0

cloud_SIM3_aligned = (R_sim3_est.T @ (cloud_SIM3.T - t_sim3_est.reshape(3,1))).T

# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 1: The 3-D Voxel point cloud overview
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 5))

ax1 = fig.add_subplot(131, projection="3d")
pts = subsample(cloud, 6000)
ax1.scatter(pts[:,0], pts[:,1], pts[:,2], c=pts[:,2], cmap="viridis", s=1, alpha=0.5)
ax1.set_title("Reference Point Cloud\n(stitched from 30 RGBD frames)", fontsize=10)
ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)"); ax1.set_zlabel("Z (m)")
ax1.view_init(elev=25, azim=45)

ax2 = fig.add_subplot(132, projection="3d")
plot_point_clouds_3d(ax2,
    [cloud, cloud_SE3],
    ["Reference", "SE(3) transformed"],
    ["steelblue", "tomato"], alpha=0.35, s=1)
ax2.set_title("SE(3) transform applied\n(R=27°, t=[0.35, -0.20, 0.15])", fontsize=10)
ax2.legend(markerscale=5, fontsize=8)
ax2.view_init(elev=25, azim=45)

ax3 = fig.add_subplot(133, projection="3d")
plot_point_clouds_3d(ax3,
    [cloud, cloud_SIM3],
    ["Reference", "SIM(3) transformed"],
    ["steelblue", "darkorange"], alpha=0.35, s=1)
ax3.set_title(f"SIM(3) transform applied\n(same R, t + scale={scale_gt})", fontsize=10)
ax3.legend(markerscale=5, fontsize=8)
ax3.view_init(elev=25, azim=45)

fig.suptitle("Chess Scene — 3D Voxel from Stitched RGBD Frames", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig1_voxel_overview.png"), dpi=150)
plt.close(fig)
print(f"\n[saved]  fig1_voxel_overview.png")

# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 2: SE(3) registration — success
# ─────────────────────────────────────────────────────────────────────────────
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
ax3.set_title("(c) SE(3) Error Metrics", fontsize=10)
ax3.set_ylabel("Error"); ax3.set_ylim(0, max(vals)*1.4+0.01)
ax3.tick_params(axis="x", labelsize=8)
ax3.axhline(0.05, color="green", ls="--", lw=1, label="success threshold")
ax3.legend(fontsize=8)

fig.suptitle("PlueckerNet-style Registration — SE(3) Case  ✓ SUCCESS",
             fontsize=13, fontweight="bold", color="darkgreen")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig2_SE3_registration.png"), dpi=150)
plt.close(fig)
print(f"[saved]  fig2_SE3_registration.png")

# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 3: SIM(3) registration — failure
# ─────────────────────────────────────────────────────────────────────────────
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
# compare SE3 vs SIM3 residuals side by side
x  = np.arange(3)
w  = 0.35
metrics = ["Rotation\nerror (°)", "Translation\nerror (m)", "Pluecker\nresidual"]
vals_se3   = [rerr_se3,  terr_se3,  res_se3]
vals_sim3  = [rerr_sim3, terr_sim3, res_sim3]
b1 = ax3.bar(x - w/2, vals_se3,  w, label="SE(3) input", color="steelblue", alpha=0.85)
b2 = ax3.bar(x + w/2, vals_sim3, w, label="SIM(3) input", color="tomato",   alpha=0.85)
for bar, val in zip(list(b1)+list(b2), vals_se3+vals_sim3):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f"{val:.3f}", ha="center", va="bottom", fontsize=7)
ax3.set_title("(c) Error: SE(3) solver on SE(3) vs SIM(3)", fontsize=10)
ax3.set_ylabel("Error")
ax3.set_xticks(x); ax3.set_xticklabels(metrics, fontsize=8)
ax3.legend(fontsize=8)
ax3.set_ylim(0, max(vals_se3+vals_sim3)*1.5+0.02)
ax3.axhline(0.05, color="green", ls="--", lw=1, label="success threshold")

fig.suptitle(f"PlueckerNet-style Registration — SIM(3) Case  ✗ FAILURE  (scale={scale_gt})",
             fontsize=13, fontweight="bold", color="crimson")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig3_SIM3_registration_failure.png"), dpi=150)
plt.close(fig)
print(f"[saved]  fig3_SIM3_registration_failure.png")

# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 4: Why SIM(3) fails — analytical diagram
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: theoretical residual as a function of scale
scales = np.linspace(0.5, 2.5, 200)
# For a single line with unit moment, residual = ||(1-s)R m1||
# We track the mean residual over all reference lines
moments_norms = np.linalg.norm(L_ref[:3], axis=0)   # N
theoretical_residual = np.array(
    [np.mean(np.abs(1 - s) * moments_norms) for s in scales]
)
axes[0].plot(scales, theoretical_residual, color="crimson", lw=2)
axes[0].axvline(1.0, color="green", ls="--", lw=1.5, label="SE(3)  (s=1)")
axes[0].axvline(scale_gt, color="orange", ls="--", lw=1.5, label=f"SIM(3) (s={scale_gt})")
axes[0].fill_between(scales, theoretical_residual, alpha=0.15, color="crimson")
axes[0].set_xlabel("Scale factor  s", fontsize=11)
axes[0].set_ylabel("Expected Pluecker residual  ||( 1 − s ) R m₁||", fontsize=10)
axes[0].set_title("Theoretical Pluecker Residual vs Scale\n"
                  "m′ = s·R·m + t×R·d   ≠   R·m + t×R·d  (SE3 model)", fontsize=10)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
annotate_txt = (
    "SE(3) solver residual:\n"
    "  e = m₂ − R·m₁ − t×R·d₁\n"
    "Under SIM(3): m₂ = s·R·m₁ + t×R·d₁\n"
    "  → e = (1−s)·R·m₁ ≠ 0  for s≠1"
)
axes[0].text(0.97, 0.97, annotate_txt, transform=axes[0].transAxes,
             fontsize=8, va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="grey", alpha=0.9))

# Right: bar chart summarising all four key numbers
categories   = ["SE(3)\nRot err°", "SE(3)\nt err m",
                 "SIM(3)\nRot err°", "SIM(3)\nt err m",
                 "SE(3)\nresidual", "SIM(3)\nresidual"]
values       = [rerr_se3, terr_se3, rerr_sim3, terr_sim3, res_se3, res_sim3]
bar_colors   = ["steelblue","steelblue","tomato","tomato","royalblue","crimson"]
bars = axes[1].bar(categories, values, color=bar_colors, edgecolor="white", alpha=0.85)
for bar, val in zip(bars, values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)
axes[1].axhline(0.05, color="green", ls="--", lw=1.5, label="success threshold (0.05)")
se3_patch   = mpatches.Patch(color="steelblue", label="SE(3)")
sim3_patch  = mpatches.Patch(color="tomato",    label="SIM(3)")
axes[1].legend(handles=[se3_patch, sim3_patch,
               mpatches.Patch(color="white", ec="green", label="threshold")],
               fontsize=9)
axes[1].set_title("Summary: PlueckerNet on SE(3) vs SIM(3)", fontsize=11)
axes[1].set_ylabel("Error magnitude"); axes[1].set_ylim(0, max(values)*1.5+0.05)
axes[1].tick_params(axis="x", labelsize=8)

fig.suptitle("Why PlueckerNet Fails on SIM(3): Scale Corrupts the Moment Vector",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig4_why_sim3_fails.png"), dpi=150)
plt.close(fig)
print(f"[saved]  fig4_why_sim3_fails.png")

# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 5: Pluecker line visualisation in 3D
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 5))
half_len = 0.15
N_show = min(40, L_ref.shape[1])
idx_show = np.random.choice(L_ref.shape[1], N_show, replace=False)

for col_idx, (L, title, color) in enumerate([
    (L_ref,  "Reference lines", "steelblue"),
    (L_SE3,  f"SE(3) transformed\n(R={np.degrees(angle):.0f}°)", "tomato"),
    (L_SIM3, f"SIM(3) transformed\n(scale={scale_gt}, same R)", "darkorange")
]):
    ax = fig.add_subplot(1, 3, col_idx+1, projection="3d")
    # draw cloud faintly
    cl = [cloud, cloud_SE3, cloud_SIM3][col_idx]
    pts = subsample(cl, 2000)
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], c="grey", s=0.5, alpha=0.15)
    # draw lines
    for i in idx_show:
        # reconstruct point-on-line from Pluecker (m,d):  p = d × m / |d|^2
        d = L[3:, i]; m = L[:3, i]
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

print(f"\n{'='*60}")
print(f"All figures saved to {OUT_DIR}")
print(f"{'='*60}")
print(f"\nSUMMARY")
print(f"  SE(3)  registration → rotation error {rerr_se3:.3f}°, t error {terr_se3:.4f} m, residual {res_se3:.5f}")
print(f"  SIM(3) registration → rotation error {rerr_sim3:.3f}°, t error {terr_sim3:.4f} m, residual {res_sim3:.5f}")
print(f"\nKey insight:")
print(f"  Under SIM(3) the moment transforms as m' = s·R·m + t×R·d")
print(f"  The SE(3) solver assumes m' = R·m + t×R·d,")
print(f"  leaving an irremovable residual of (1−s)·R·m ≠ 0 for s={scale_gt}.")
print(f"  Rotation IS recoverable (unit direction d is scale-invariant),")
print(f"  but translation estimate is badly biased by the corrupted moments.")
