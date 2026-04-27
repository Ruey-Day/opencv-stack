#!/usr/bin/env python3
"""
eval_color_chess.py — Evaluate ScalePlueckerNet+color on Chess B1/B2.

Outputs printed to stdout. Results saved to results/eval_color/.
Existing results/ directories untouched.

B1: RGBD, GT s=1.0 (seq-01 vs seq-03, identity transform)
B2: RGB-only, GT s=1.8 (moments of seq-03 scaled ×1.8)
"""

import os, sys, glob, time, warnings
import numpy as np
from scipy.spatial import cKDTree
import torch

warnings.filterwarnings("ignore")

ROOT        = os.path.dirname(os.path.abspath(__file__))
PLUECKERNET = os.path.abspath(os.path.join(ROOT, "..", "PlueckerNet"))
COLOR_W     = os.path.join(ROOT, "output", "replica_color", "2026-04-23",
                           "best_val_checkpoint.pth")
OUT_DIR     = os.path.join(ROOT, "results", "eval_color")
os.makedirs(OUT_DIR, exist_ok=True)

CHESS_SEQ1 = "/home/rueyday/Downloads/chess/seq-01"
CHESS_SEQ3 = "/home/rueyday/Downloads/chess/seq-03"

sys.path.insert(0, PLUECKERNET)
sys.path.insert(0, ROOT)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FX, FY      = 525.0, 525.0
CX, CY      = 319.5, 239.5
DEPTH_SCALE = 1000.0


# ── model ──────────────────────────────────────────────────────────────────────

def load_color_model():
    from easydict import EasyDict as edict
    from model.model_plucker import PluckerNetKnn
    cfg = edict(net_nchannel=128, GNN_layers=["self", "cross"] * 6,
                net_lambda=0.1, net_maxiter=30, net_topK=200, in_channel=9)
    model = PluckerNetKnn(cfg).to(DEVICE)
    ckpt  = torch.load(COLOR_W, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[Color model] loaded  {COLOR_W}")
    return model


# ── chess data loading with color ──────────────────────────────────────────────

def load_chess_frames_rgb(seq_dir, n_frames=25, frame_step=40):
    import cv2
    depth_files = sorted(glob.glob(os.path.join(seq_dir, "*.depth.png")))
    depth_files = depth_files[::frame_step][:n_frames]
    frames = []
    for df in depth_files:
        pf = df.replace(".depth.png", ".pose.txt")
        cf = df.replace(".depth.png", ".color.png")
        if not os.path.exists(pf) or not os.path.exists(cf):
            continue
        depth = cv2.imread(df, cv2.IMREAD_ANYDEPTH).astype(np.float32) / DEPTH_SCALE
        pose  = np.loadtxt(pf)
        color = cv2.imread(cf, cv2.IMREAD_COLOR)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames.append((depth, color, pose))
    print(f"  [chess rgb] {len(frames)} frames from {os.path.basename(seq_dir)}")
    return frames


def build_colored_cloud(frames, subsample=4, max_depth=3.5, voxel=0.025):
    """Return (N, 3) xyz and (N, 3) rgb after voxel downsampling."""
    xyz_all, rgb_all = [], []
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
        xyz = (pose @ cam)[:3].T
        rgb = color[vi, ui]          # (M, 3) already float32 [0,1]
        xyz_all.append(xyz)
        rgb_all.append(rgb)
    xyz = np.concatenate(xyz_all, 0)
    rgb = np.concatenate(rgb_all, 0)
    keys = np.floor(xyz / voxel).astype(np.int32)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return xyz[idx], rgb[idx]


def extract_lines_9d(xyz, rgb, n_lines=300, k=20, linearity_thresh=0.72, seed=42):
    """Extract lines via local PCA; attach mean RGB from neighborhood.

    Returns (M, 9) float32: [m0,m1,m2, d0,d1,d2, r,g,b]
    """
    rng  = np.random.default_rng(seed)
    tree = cKDTree(xyz)
    mids, dirs, colors = [], [], []
    indices = rng.choice(len(xyz), size=min(15000, len(xyz)), replace=False)
    for idx in indices:
        if len(mids) >= n_lines:
            break
        nn  = tree.query(xyz[idx], k=k)[1]
        pts = xyz[nn]
        ctr = pts.mean(0)
        cov = (pts - ctr).T @ (pts - ctr) / k
        ev, evec = np.linalg.eigh(cov)
        lam = ev[::-1]
        if (lam[0] - lam[1]) / (lam[0] + 1e-9) > linearity_thresh:
            mids.append(ctr)
            dirs.append(evec[:, -1])
            colors.append(rgb[nn].mean(0))   # mean RGB of neighborhood
    if not mids:
        return np.zeros((0, 9), dtype=np.float32)
    mids   = np.array(mids,   dtype=np.float32)
    dirs   = np.array(dirs,   dtype=np.float32)
    colors = np.array(colors, dtype=np.float32)
    dirs  /= np.linalg.norm(dirs, axis=1, keepdims=True)
    m = np.cross(mids, dirs)
    return np.concatenate([m, dirs, colors], axis=1)   # (M, 9)


# ── inference ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def _topk_9d(model, L1, L2, topk):
    t1 = torch.from_numpy(L1).unsqueeze(0).to(DEVICE)
    t2 = torch.from_numpy(L2).unsqueeze(0).to(DEVICE)
    P, _, _ = model(t1, t2)
    k = min(topk, P.shape[1] * P.shape[2])
    _, flat = torch.topk(P.flatten(start_dim=-2), k=k, dim=-1)
    i1 = (flat // P.shape[-1]).squeeze(0).cpu().numpy()
    i2 = (flat  % P.shape[-1]).squeeze(0).cpu().numpy()
    return i1, i2


def run_color_net(model, L1_9d, L2_9d, topk=100, threshold=0.15):
    """Network matching on 9D lines; RANSAC on first 6D (geometry only)."""
    from sim3.ransac import run_ransac_sim3

    t0 = time.perf_counter()
    i1, i2 = _topk_9d(model, L1_9d, L2_9d, topk)
    t_net = (time.perf_counter() - t0) * 1000

    # RANSAC uses geometry only — first 6 columns
    p1 = L1_9d[i1, :6].T
    p2 = L2_9d[i2, :6].T

    t0 = time.perf_counter()
    s, R, t, ic, _ = run_ransac_sim3(p1, p2, inlier_threshold=threshold)
    t_ran = (time.perf_counter() - t0) * 1000

    if R is None:
        return dict(R=np.eye(3), t=np.zeros(3), s=1.0, ic=0,
                    t_net_ms=t_net, t_ransac_ms=t_ran, t_total_ms=t_net + t_ran)
    return dict(R=R, t=np.asarray(t).flatten(), s=float(s), ic=int(ic),
                t_net_ms=t_net, t_ransac_ms=t_ran, t_total_ms=t_net + t_ran)


# ── metrics ────────────────────────────────────────────────────────────────────

def rot_err_deg(R_est, R_gt):
    dR  = R_gt @ R_est.T
    cos = np.clip((np.trace(dR) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cos)))


def scale_err_log(s_est, s_gt):
    if s_est <= 0 or s_gt <= 0:
        return float("inf")
    return float(abs(np.log(s_est / s_gt)))


def trans_err(t_est, t_gt):
    return float(np.linalg.norm(np.asarray(t_est).flatten() -
                                np.asarray(t_gt).flatten()))


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    model = load_color_model()

    print("\n  Loading seq-01 (with color)...")
    frames1  = load_chess_frames_rgb(CHESS_SEQ1, n_frames=25, frame_step=40)
    xyz1, rgb1 = build_colored_cloud(frames1)
    L1_9d    = extract_lines_9d(xyz1, rgb1, n_lines=300)
    print(f"  {len(xyz1):,} pts → {len(L1_9d)} lines")

    print("\n  Loading seq-03 (with color)...")
    frames3  = load_chess_frames_rgb(CHESS_SEQ3, n_frames=25, frame_step=40)
    xyz3, rgb3 = build_colored_cloud(frames3)
    L3_9d    = extract_lines_9d(xyz3, rgb3, n_lines=300)
    print(f"  {len(xyz3):,} pts → {len(L3_9d)} lines")

    R_gt     = np.eye(3, dtype=np.float32)
    t_gt     = np.zeros(3, dtype=np.float32)
    s_rgb    = 1.8

    L3_scaled = L3_9d.copy()
    L3_scaled[:, :3] *= s_rgb   # scale moments (geometry); RGB unchanged

    print("\n" + "═" * 60)
    print("B1  RGBD (GT s=1.0)")
    print("═" * 60)
    r1 = run_color_net(model, L1_9d, L3_9d, topk=100, threshold=0.15)
    print(f"  rot  = {rot_err_deg(r1['R'], R_gt):.2f}°")
    print(f"  t    = {trans_err(r1['t'], t_gt):.4f} m")
    print(f"  s    = {r1['s']:.4f}  (s_err={scale_err_log(r1['s'], 1.0):.3f})")
    print(f"  ic   = {r1['ic']}")
    print(f"  time = {r1['t_total_ms']:.0f} ms")

    print("\n" + "═" * 60)
    print(f"B2  RGB-only, moments×{s_rgb} (GT s={s_rgb})")
    print("═" * 60)
    r2 = run_color_net(model, L1_9d, L3_scaled, topk=100, threshold=0.15)
    print(f"  rot  = {rot_err_deg(r2['R'], R_gt):.2f}°")
    print(f"  t    = {trans_err(r2['t'], t_gt):.4f} m")
    print(f"  s    = {r2['s']:.4f}  (s_err={scale_err_log(r2['s'], s_rgb):.3f})")
    print(f"  ic   = {r2['ic']}")
    print(f"  time = {r2['t_total_ms']:.0f} ms")

    print("\n" + "═" * 60)
    print("SUMMARY — ScalePlueckerNet+color on Chess")
    print("═" * 60)
    print(f"  B1 RGBD  rot={rot_err_deg(r1['R'],R_gt):.2f}°  "
          f"s_err={scale_err_log(r1['s'],1.0):.3f}")
    print(f"  B2 RGB   rot={rot_err_deg(r2['R'],R_gt):.2f}°  "
          f"s_err={scale_err_log(r2['s'],s_rgb):.3f}")


if __name__ == "__main__":
    main()
