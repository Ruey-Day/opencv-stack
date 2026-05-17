"""
Shared pair-generation utilities for all dataset generators.

Data format per pair:
  plucker1/2 : (N_TOTAL, 9) float32  — [m0,m1,m2, d0,d1,d2, L*,A*,B*]
  matches    : (2, n_inliers) int32  — row-0 = idx in p1, row-1 = idx in p2
  R_gt       : (3, 3) float32
  t_gt       : (3, 1) float32
  s_gt       : float32  — 0.0 signals zero-overlap (no GT pose)
"""

import numpy as np
import cv2

SCALE_RANGE   = (0.1, 10.0)
N_TOTAL       = 700   # lines per pair
N_MAX_INLIERS = 490   # inliers when overlap = 100%

OVERLAP_LEVELS = np.array([0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00])
OVERLAP_PROBS  = np.array([0.12, 0.08, 0.08, 0.09, 0.09, 0.10, 0.10, 0.34])

def random_rotation():
    A = np.random.randn(3, 3).astype(np.float64)
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float32)

def apply_sim3(lines9, s, R, t):
    """Transform 9D lines by Sim(3); add illumination noise to LAB channel."""
    m, d, lab = lines9[:, :3], lines9[:, 3:6], lines9[:, 6:9]
    d_new = (R @ d.T).T
    m_new = s * (R @ m.T).T + np.cross(t.flatten()[None], d_new)
    noise = np.random.randn(*lab.shape).astype(np.float32) * np.array([3., 4., 4.])
    return np.concatenate([m_new, d_new, lab + noise], axis=1).astype(np.float32)

def make_outliers(n, n_clusters=6, spread=0.15, pos_range=3.0):
    """Synthetic outlier 9D Plücker lines with random LAB."""
    n_per = n // n_clusters
    extras = n - n_per * n_clusters
    parts = []
    anchors = np.random.randn(n_clusters, 3).astype(np.float32)
    anchors /= np.linalg.norm(anchors, axis=1, keepdims=True)
    for i, a in enumerate(anchors):
        cnt = n_per + (1 if i < extras else 0)
        d = a[None] + np.random.randn(cnt, 3).astype(np.float32) * spread
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        p = np.random.uniform(-pos_range, pos_range, (cnt, 3)).astype(np.float32)
        plucker = np.concatenate([np.cross(p, d), d], axis=1)
        lab = np.column_stack([
            np.random.uniform(20,  80, cnt),
            np.random.uniform(-60, 60, cnt),
            np.random.uniform(-60, 60, cnt),
        ]).astype(np.float32)
        parts.append(np.concatenate([plucker, lab], axis=1))
    lines = np.concatenate(parts, 0)
    return lines[np.random.permutation(len(lines))]

def sample_line_lab(bgr, ep, n_pts=15):
    """Average LAB color sampled along a 2D line segment.

    Returns (3,) float32 [L*, A*, B*] in OpenCV LAB scale
    (L ∈ [0,255], A/B ∈ [0,255] offset by 128).
    """
    H, W = bgr.shape[:2]
    (u1, v1), (u2, v2) = ep[0], ep[1]
    ts = np.linspace(0.0, 1.0, n_pts)
    us = np.clip(np.round(u1 + ts * (u2 - u1)).astype(int), 0, W - 1)
    vs = np.clip(np.round(v1 + ts * (v2 - v1)).astype(int), 0, H - 1)
    pixels = bgr[vs, us].reshape(1, n_pts, 3)
    lab = cv2.cvtColor(pixels, cv2.COLOR_BGR2Lab).reshape(n_pts, 3)
    return lab.mean(axis=0).astype(np.float32)

def dedup_pool(pool, pos_voxel=0.10, dir_voxel=0.04):
    """NMS-style deduplication of world-space Plücker lines.

    The largest moment norm is kept because it is most geometrically discriminative.

    - pos_voxel: spatial bin size in metres
    - dir_voxel: direction bin on unit sphere
    """
    if len(pool) == 0:
        return pool

    m = pool[:, :3].astype(np.float64)
    d = pool[:, 3:6].astype(np.float64)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9

    # Canonical direction: flip so z ≥ 0 (or y ≥ 0 when z ≈ 0)
    flip = (d[:, 2] < 0) | ((d[:, 2] == 0) & (d[:, 1] < 0))
    d_can = d.copy()
    d_can[flip] *= -1

    p0 = np.cross(d, m)          # (N, 3) closest-to-origin anchor
    m_norm = np.linalg.norm(m, axis=1)  # quality: larger = further off-axis
    
    pk = np.floor(p0    / pos_voxel).astype(np.int32)  # (N, 3)
    dk = np.floor(d_can / dir_voxel).astype(np.int32)  # (N, 3)

    # Keep the highest-quality (largest |m|) line per cell
    best = {}   # key → (m_norm, index)
    for i in range(len(pool)):
        key = (pk[i,0], pk[i,1], pk[i,2], dk[i,0], dk[i,1], dk[i,2])
        if key not in best or m_norm[i] > best[key][0]:
            best[key] = (m_norm[i], i)

    keep = np.array([v[1] for v in best.values()], dtype=np.int32)
    return pool[keep]

def generate_pair(pool):
    """Generate one 9D training pair from a world-space line pool (N, 9).

    Samples overlap ratio from OVERLAP_LEVELS/OVERLAP_PROBS, applies Sim(3)
    with scale from SCALE_RANGE, pads to N_TOTAL lines.

    Returns dict with keys: plucker1, plucker2, matches, R_gt, t_gt, s_gt.
    Returns None if pool is too small.
    """
    overlap = float(np.random.choice(OVERLAP_LEVELS, p=OVERLAP_PROBS))

    if overlap == 0.0:
        # Two independent random batches, no GT pose
        l1 = make_outliers(N_TOTAL)
        l2 = make_outliers(N_TOTAL)
        return dict(
            plucker1=l1.astype(np.float32),
            plucker2=l2.astype(np.float32),
            matches=np.zeros((2, 0), dtype=np.int32),
            R_gt=np.eye(3, dtype=np.float32),
            t_gt=np.zeros((3, 1), dtype=np.float32),
            s_gt=np.float32(0.0),
        )

    n_inliers = max(4, int(round(overlap * N_MAX_INLIERS)))
    if len(pool) < n_inliers:
        return None

    n_outliers = N_TOTAL - n_inliers

    idx   = np.random.choice(len(pool), n_inliers, replace=False)
    l1_in = pool[idx].copy()

    log_s = np.random.uniform(np.log(SCALE_RANGE[0]), np.log(SCALE_RANGE[1]))
    s = float(np.exp(log_s))
    R = random_rotation()
    t = np.random.uniform(-2.0, 2.0, 3).astype(np.float32)

    l2_in = apply_sim3(l1_in, s, R, t)

    l1 = np.concatenate([l1_in, make_outliers(n_outliers)], 0)
    l2 = np.concatenate([l2_in, make_outliers(n_outliers)], 0)

    i1, i2  = np.random.permutation(len(l1)), np.random.permutation(len(l2))
    l1, l2  = l1[i1], l2[i2]
    inv1, inv2 = np.argsort(i1), np.argsort(i2)
    matches = np.stack([inv1[:n_inliers], inv2[:n_inliers]], 0).astype(np.int32)

    return dict(
        plucker1=l1.astype(np.float32),
        plucker2=l2.astype(np.float32),
        matches=matches,
        R_gt=R,
        t_gt=t.reshape(3, 1),
        s_gt=np.float32(s),
    )
