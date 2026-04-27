#!/usr/bin/env python3
"""
view_compare_interactive.py

Three-method comparison on the LSD line cloud (seq-01 ↔ seq-03):
  Col 1 — Pure RANSAC (SE3 solver — no scale awareness)
  Col 2 — SE3-PlueckerNet (original — no scale awareness)
  Col 3 — ScalePluckerNet (ours — Sim3 solver)

Two scenarios:
  se3  — RGBD, metric scale s≈1  (all methods should work)
  sim3 — seq-03 moments ×SCALE   (SE3 methods fail; ScalePluckerNet recovers)

Usage:
    python view_compare_interactive.py se3
    python view_compare_interactive.py sim3
Close the window to save to results/presentation/compare_{se3,sim3}.png
"""

import os, sys, glob, time
import numpy as np
import cv2, torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "se3"
assert SCENARIO in ("se3", "sim3"), "Usage: ...se3|sim3"

ROOT        = os.path.dirname(os.path.abspath(__file__))
PLUECKERNET = os.path.abspath(os.path.join(ROOT, "..", "PlueckerNet"))
SE3_W       = os.path.join(PLUECKERNET, "output", "semantic3D",
                           "preTrained", "best_val_checkpoint_real.pth")
SIM3_W      = os.path.join(ROOT, "output", "replica",
                           "2026-04-22", "best_val_checkpoint.pth")
CHESS_SEQ1  = "/home/rueyday/Downloads/chess/seq-01"
CHESS_SEQ3  = "/home/rueyday/Downloads/chess/seq-03"
OUT_PATH    = os.path.join(ROOT, "results", "presentation",
                           f"compare_{SCENARIO}.png")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

sys.path.insert(0, PLUECKERNET)
sys.path.insert(0, ROOT)

FX, FY, CX, CY = 525.0, 525.0, 319.5, 239.5
DEPTH_SCALE    = 1000.0
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_LINES        = 400
SIM3_SCALE     = 3.0   # moment multiplier for sim3 scenario — large enough to be visually obvious


# ── models ─────────────────────────────────────────────────────────────────────

def load_se3_model():
    from easydict import EasyDict as edict
    from model.model_plucker import PluckerNetKnn
    import lib.ransac_l2l as _rm
    def _skew(x):
        x = np.asarray(x).flatten()
        return np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]])
    _rm.skew = _skew
    from lib.ransac_l2l import run_ransac
    cfg   = edict(net_nchannel=128, GNN_layers=["self","cross"]*6,
                  net_lambda=0.1, net_maxiter=30, net_topK=200)
    model = PluckerNetKnn(cfg).to(DEVICE)
    ckpt  = torch.load(SE3_W, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print("SE3  model loaded")
    return model, run_ransac


def load_scaleplucker_model():
    from easydict import EasyDict as edict
    from model.model_plucker import PluckerNetKnn
    cfg   = edict(net_nchannel=128, GNN_layers=["self","cross"]*6,
                  net_lambda=0.1, net_maxiter=30, net_topK=200)
    model = PluckerNetKnn(cfg).to(DEVICE)
    ckpt  = torch.load(SIM3_W, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print("ScalePluckerNet loaded")
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
    us, vs = x1 + ts*(x2-x1), y1 + ts*(y2-y1)
    H, W  = depth.shape
    ui = np.clip(us.astype(int), 0, W-1)
    vi = np.clip(vs.astype(int), 0, H-1)
    z  = depth[vi, ui]
    valid = (z > 0.1) & (z < max_depth)
    if valid.sum() < 4 or z[valid].std() > max_depth_std:
        return None
    u, v, z = us[valid], vs[valid], z[valid]
    x = (u - CX)*z/FX;  y = (v - CY)*z/FY
    pts_w = (pose @ np.stack([x,y,z,np.ones_like(z)],1).T)[:3].T
    ctr   = pts_w.mean(0)
    _, evec = np.linalg.eigh((pts_w-ctr).T @ (pts_w-ctr))
    d    = evec[:, -1]
    proj = (pts_w - ctr) @ d
    if proj.max() - proj.min() < 0.05:
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
    print(f"  {len(starts)} segments from {len(depth_files)} frames")
    return np.array(starts, np.float32), np.array(ends, np.float32)


def build_point_cloud(seq_dir, n_frames=30, frame_step=25,
                      subsample=4, max_depth=3.5, voxel=0.03, n_show=8000):
    """Build a voxel-downsampled 3D point cloud for the alignment panel."""
    depth_files = sorted(glob.glob(os.path.join(seq_dir, "*.depth.png")))
    depth_files = depth_files[::frame_step][:n_frames]
    pts_all = []
    for df in depth_files:
        pf = df.replace(".depth.png", ".pose.txt")
        if not os.path.exists(pf):
            continue
        depth = cv2.imread(df, cv2.IMREAD_ANYDEPTH).astype(np.float32) / DEPTH_SCALE
        pose  = np.loadtxt(pf)
        H, W  = depth.shape
        vi, ui = np.meshgrid(np.arange(0, H, subsample),
                             np.arange(0, W, subsample), indexing='ij')
        vi, ui = vi.ravel(), ui.ravel()
        z = depth[vi, ui]
        ok = (z > 0.1) & (z < max_depth)
        z, vi, ui = z[ok], vi[ok], ui[ok]
        x = (ui - CX) * z / FX;  y = (vi - CY) * z / FY
        cam = np.stack([x, y, z, np.ones_like(z)], 0)
        pts_all.append((pose @ cam)[:3].T)
    cloud = np.concatenate(pts_all, 0)
    keys  = np.floor(cloud / voxel).astype(np.int32)
    _, idx = np.unique(keys, axis=0, return_index=True)
    cloud = cloud[idx]
    # subsample to n_show for fast rendering
    rng = np.random.default_rng(0)
    if len(cloud) > n_show:
        cloud = cloud[rng.choice(len(cloud), n_show, replace=False)]
    print(f"  {len(cloud)} cloud points")
    return cloud.astype(np.float32)


# ── Plücker helpers ────────────────────────────────────────────────────────────

def to_plucker_md(starts, ends):
    """Endpoints → [m, d] format (N, 6)."""
    d = ends - starts
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    return np.concatenate([np.cross(starts, d), d], axis=1).astype(np.float32)

def md_to_dm(L):
    """[m, d] → [d, m] for SE3 model."""
    return np.hstack([L[:, 3:], L[:, :3]])

def apply_sim3_endpoints(starts, ends, s, R, t):
    """Apply Sim3(s,R,t) to segment endpoints."""
    t = np.asarray(t).flatten()
    return (s * R @ starts.T + t[:,None]).T, (s * R @ ends.T + t[:,None]).T


# ── matching ──────────────────────────────────────────────────────────────────

def _net_topk(model, L1, L2, topk):
    with torch.no_grad():
        P, _, _ = model(torch.from_numpy(L1).unsqueeze(0).to(DEVICE),
                        torch.from_numpy(L2).unsqueeze(0).to(DEVICE))
    k    = min(topk, P.shape[1]*P.shape[2])
    _, f = torch.topk(P.flatten(start_dim=-2), k=k, dim=-1)
    return (f // P.shape[-1]).squeeze(0).cpu().numpy(), \
           (f  % P.shape[-1]).squeeze(0).cpu().numpy()

def dir_nn_topk(L1, L2, topk=150):
    """Direction cosine-NN — scale-blind matching for pure RANSAC."""
    sim = np.abs(L1[:, 3:] @ L2[:, 3:].T)
    k   = min(topk, sim.size)
    idx = np.argpartition(sim.flatten(), -k)[-k:]
    return idx // sim.shape[1], idx % sim.shape[1]


# ── per-method runners ─────────────────────────────────────────────────────────

def run_pure_ransac_se3(ransac_fn, L1_md, L2_md, topk=150, threshold=0.5):
    """
    Pure RANSAC with SE3 solver — does NOT know about scale.
    Uses direction-NN matching, then SE3 RANSAC on [d,m] lines.
    When scale ≠ 1, (s-1)·Rm absorbs into a spurious translation → wrong R.
    """
    i1, i2 = dir_nn_topk(L1_md, L2_md, topk)
    p1 = md_to_dm(L1_md)[i1].T
    p2 = md_to_dm(L2_md)[i2].T
    t0 = time.perf_counter()
    R, t, ic, mask = ransac_fn(p1, p2, inlier_threshold=threshold)
    dt = (time.perf_counter()-t0)*1000
    if R is None:
        return np.eye(3), np.zeros(3), 1.0, i1, i2, None, dt
    return R, np.asarray(t).flatten(), 1.0, i1, i2, mask, dt


def run_se3_net(model_se3, ransac_fn, L1_md, L2_md, topk=150, threshold=0.5):
    """
    SE3-PlueckerNet — network matching + SE3 solver.
    Scale is ignored: s is always returned as 1.
    """
    t0 = time.perf_counter()
    i1, i2 = _net_topk(model_se3, md_to_dm(L1_md), md_to_dm(L2_md), topk)
    t_net   = (time.perf_counter()-t0)*1000
    p1 = md_to_dm(L1_md)[i1].T
    p2 = md_to_dm(L2_md)[i2].T
    t0 = time.perf_counter()
    R, t, ic, mask = ransac_fn(p1, p2, inlier_threshold=threshold)
    dt = t_net + (time.perf_counter()-t0)*1000
    if R is None:
        return np.eye(3), np.zeros(3), 1.0, i1, i2, None, dt
    return R, np.asarray(t).flatten(), 1.0, i1, i2, mask, dt


def run_scaleplucker(model_sim3, L1_md, L2_md, topk=120, threshold=0.25):
    """
    ScalePluckerNet — network matching + Sim3 solver.
    Jointly estimates s, R, t.
    """
    from sim3.ransac import run_ransac_sim3
    t0 = time.perf_counter()
    i1, i2 = _net_topk(model_sim3, L1_md, L2_md, topk)
    t_net   = (time.perf_counter()-t0)*1000
    t0 = time.perf_counter()
    s, R, t, ic, mask = run_ransac_sim3(L1_md[i1].T, L2_md[i2].T,
                                        inlier_threshold=threshold,
                                        max_iterations=500)
    dt = t_net + (time.perf_counter()-t0)*1000
    if R is None:
        return np.eye(3), np.zeros(3), 1.0, i1, i2, None, dt
    return R, np.asarray(t).flatten(), float(s), i1, i2, mask, dt


# ── drawing ───────────────────────────────────────────────────────────────────

def draw_segs(ax, starts, ends, color, alpha=0.48, lw=1.0,
              n=350, seed=0, label=None):
    rng  = np.random.default_rng(seed)
    pick = rng.choice(len(starts), min(n, len(starts)), replace=False)
    for k, i in enumerate(pick):
        kw = dict(color=color, lw=lw, alpha=alpha)
        if k == 0 and label:
            kw['label'] = label
        s, e = starts[i], ends[i]
        ax.plot([s[0],e[0]], [s[1],e[1]], [s[2],e[2]], **kw)


def draw_corr(ax, s1, e1, s2, e2, i1, i2, mask, max_show=50):
    if mask is None or mask.sum() == 0:
        return
    inl1, inl2 = i1[mask], i2[mask]
    rng  = np.random.default_rng(7)
    pick = rng.choice(len(inl1), min(max_show, len(inl1)), replace=False)
    for k in pick:
        p1 = (s1[inl1[k]] + e1[inl1[k]]) / 2
        p2 = (s2[inl2[k]] + e2[inl2[k]]) / 2
        ax.plot([p1[0],p2[0]], [p1[1],p2[1]], [p1[2],p2[2]],
                color='#33dd66', lw=0.7, alpha=0.55, linestyle='--')


def set_limits(ax, *pt_sets):
    pts = np.vstack(pt_sets)
    lo, hi = pts.min(0)-0.2, pts.max(0)+0.2
    r   = (hi-lo).max()/2
    mid = (lo+hi)/2
    ax.set_xlim(mid[0]-r, mid[0]+r)
    ax.set_ylim(mid[1]-r, mid[1]+r)
    ax.set_zlim(mid[2]-r, mid[2]+r)


def style_ax(ax, title, subtitle=''):
    ax.set_facecolor('white')
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False; pane.set_edgecolor('#cccccc')
    ax.set_title((title + (f'\n{subtitle}' if subtitle else '')),
                 fontsize=9, fontweight='bold', pad=3)
    ax.tick_params(colors='#555', labelsize=5)
    ax.set_xlabel('X', fontsize=5); ax.set_ylabel('Y', fontsize=5)
    ax.set_zlabel('Z', fontsize=5)


def draw_cloud(ax, pts, color, alpha=0.55, s=3.0, label=None):
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               s=s, c=color, alpha=alpha, linewidths=0, label=label)


def render_figure(s1, e1, s3, e3, pc1, pc3, methods, title, elev=20, azim=-50):
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor('white')

    for col, (name, R, t, s, i1, i2, mask, dt, mcol) in enumerate(methods, 1):
        ic = int(mask.sum()) if mask is not None else 0

        # Row 1 — line correspondences
        ax = fig.add_subplot(2, 3, col, projection='3d')
        draw_segs(ax, s1, e1, '#2266cc', n=350, seed=0, label='Seq-01 (src)')
        draw_segs(ax, s3, e3, '#dd2222', n=350, seed=1, label='Seq-03 (tgt)')
        draw_corr(ax, s1, e1, s3, e3, i1, i2, mask)
        ic_str = f'{ic} inliers' if mask is not None else 'RANSAC failed'
        style_ax(ax, name, f'{ic_str}  {dt:.0f} ms')
        set_limits(ax, s1, e1, s3, e3)
        ax.view_init(elev=elev, azim=azim)
        if col == 1:
            ax.legend(fontsize=6.5, loc='upper left', framealpha=0.6)

        # Row 2 — point cloud alignment
        # Both clouds are in metric 3D space → apply only R,t (not s).
        # In the Sim3 scenario s lives in the Plücker moment representation;
        # a wrong R from SE3 RANSAC (scale absorbed into rotation) shows here.
        t_flat  = np.asarray(t).flatten()
        pc1_reg = (R @ pc1.T + t_flat[:, None]).T

        ax2 = fig.add_subplot(2, 3, col+3, projection='3d')
        draw_cloud(ax2, pc3,     '#dd2222', alpha=0.20, label='Seq-03 (target)')
        draw_cloud(ax2, pc1_reg, mcol,      alpha=0.35, label='Seq-01 registered')
        style_ax(ax2, f'Point Cloud — {name}',
                 f's={s:.3f}  (estimated)   {dt:.0f} ms')
        set_limits(ax2, pc3, pc1_reg)
        ax2.view_init(elev=elev, azim=azim)
        if col == 1:
            ax2.legend(fontsize=6.5, loc='upper left', framealpha=0.6,
                       markerscale=6)

    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.005)
    plt.tight_layout(pad=0.8)
    return fig


def on_close(event):
    print(f"Saving → {os.path.relpath(OUT_PATH, ROOT)}")
    event.canvas.figure.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    print("Done.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    model_se3, ransac_se3 = load_se3_model()
    model_spl             = load_scaleplucker_model()

    print("\nBuilding LSD cloud seq-01 ...")
    s1_all, e1_all = build_lsd_cloud(CHESS_SEQ1)
    print("Building point cloud seq-01 ...")
    pc1 = build_point_cloud(CHESS_SEQ1)
    print("Building LSD cloud seq-03 ...")
    s3_all, e3_all = build_lsd_cloud(CHESS_SEQ3)
    print("Building point cloud seq-03 ...")
    pc3 = build_point_cloud(CHESS_SEQ3)

    rng  = np.random.default_rng(42)
    idx1 = rng.choice(len(s1_all), min(N_LINES, len(s1_all)), replace=False)
    idx3 = rng.choice(len(s3_all), min(N_LINES, len(s3_all)), replace=False)
    s1, e1 = s1_all[idx1], e1_all[idx1]
    s3, e3 = s3_all[idx3], e3_all[idx3]

    L1    = to_plucker_md(s1, e1)   # [m, d]
    L2_se3 = to_plucker_md(s3, e3)  # original metric lines  (SE3 scenario)

    if SCENARIO == "se3":
        L2      = L2_se3
        gt_s    = 1.0
        title   = ("Chess 7-Scenes: seq-01 ↔ seq-03   |   SE3 scenario (RGBD, s=1)\n"
                   "All methods use SE3 RANSAC except ScalePluckerNet (Sim3 RANSAC)")
    else:
        L2           = L2_se3.copy()
        L2[:, :3]   *= SIM3_SCALE   # scale moments only — directions unchanged
        gt_s         = SIM3_SCALE
        title   = (f"Chess 7-Scenes: seq-01 ↔ seq-03   |   Sim3 scenario (moments ×{SIM3_SCALE:.0f})\n"
                   f"SE3 RANSAC absorbs scale into translation → wrong rotation.  "
                   f"ScalePluckerNet recovers s≈{SIM3_SCALE:.0f}.")

    print(f"\nScenario: {SCENARIO.upper()}  (GT s={gt_s})\n")
    np.random.seed(42)

    R_p,  t_p,  s_p,  i1_p,  i2_p,  mask_p,  dt_p  = run_pure_ransac_se3(ransac_se3, L1, L2)
    R_se, t_se, s_se, i1_se, i2_se, mask_se, dt_se  = run_se3_net(model_se3, ransac_se3, L1, L2)
    R_sp, t_sp, s_sp, i1_sp, i2_sp, mask_sp, dt_sp  = run_scaleplucker(model_spl, L1, L2)

    for nm, mask, s, dt in [("Pure RANSAC (SE3)",  mask_p,  s_p,  dt_p),
                             ("SE3-PlueckerNet",    mask_se, s_se, dt_se),
                             ("ScalePluckerNet",    mask_sp, s_sp, dt_sp)]:
        ic = int(mask.sum()) if mask is not None else 0
        print(f"  {nm:22s}  inliers={ic:3d}  s={s:.3f}  {dt:.0f}ms")

    methods = [
        ("Pure RANSAC (SE3)",  R_p,  t_p,  s_p,  i1_p,  i2_p,  mask_p,  dt_p,  '#22aa44'),
        ("SE3-PlueckerNet",    R_se, t_se, s_se, i1_se, i2_se, mask_se, dt_se, '#2266cc'),
        ("ScalePluckerNet",    R_sp, t_sp, s_sp, i1_sp, i2_sp, mask_sp, dt_sp, '#9922cc'),
    ]

    fig = render_figure(s1, e1, s3, e3, pc1, pc3, methods, title)
    fig.canvas.mpl_connect('close_event', on_close)
    print("\nRotate to your preferred angle, then close to save.")
    plt.show()


if __name__ == "__main__":
    main()
