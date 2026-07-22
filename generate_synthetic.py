#!/usr/bin/env python3
"""
Generate a large, fully-synthetic dataset for Sim(3) line correspondence training.

No map files required — all pools come from random geometric primitives
(plane patches, wireframes, line bundles, parallel groups, grid patches,
staircases) with no axis-aligned or Manhattan-world bias.

Scenarios
---------
room        (30%) — indoor room scale, moderate-to-high overlap
submap      (10%) — sparse/noisy monocular query vs large clean metric reference
relocalize  (22%) — cross-session re-localization, moderate overlap
loop        (28%) — loop-closure, high overlap, both sides similar
dense_sparse(10%) — very different line densities (RGB-D vs monocular)

Output
------
dataset/synthetic_train/   dataset/synthetic_valid/

Each split uses the standard 6-pickle format:
    matches.pkl  plucker1.pkl  plucker2.pkl  R_gt.pkl  t_gt.pkl  s_gt.pkl

Usage
-----
python generate_synthetic.py
python generate_synthetic.py --n_train 500000 --n_valid 5000
python generate_synthetic.py --n_train 200000 --workers 8 --seed 123
"""
import os
import sys
import pickle
import argparse
import time
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))


# ── Constants ─────────────────────────────────────────────────────────────────

_SCALE_RANGE      = (0.3, 8.0)   # tightened: avoids s_i > 3.3 to prevent moment explosion
_ROOM_SCALE_RANGE = (0.3, 5.0)
_MIN_INLIERS      = 20

# v4 calibration: match measured GT scale of real mono→metric pairs
# (tools/analyze_real_noise.py 2026-07-03: median 2.1, P25–P95 = 1.7–3.4).
# v5: wider spread/clip — the 33-seq GT benchmark (2026-07-07) observed GT
# scales 0.74–10.65 (heads at the old lower clip 0.7, stairs OUTSIDE the old
# upper clip 7.0); cover them with margin while keeping the same median.
_SCALE_LOG_CENTER = np.log(2.2)
_SCALE_LOG_STD    = 0.60
_SCALE_CLIP       = (0.4, 13.0)

# v4 real-structure noise (calibrated against tools/analyze_real_noise.py):
# physical perturbation of (foot point, direction) with m recomputed, so the
# Plücker constraint m·d = 0 holds — Gaussian noise directly on m violates it
# and real/test lines never do.
_V4_DIR_SIGMA_DEG = 5.2      # Rayleigh → median ~6.1°, P90 ~11.2°
_V4_DIR_CAP_DEG   = 25.0
_V4_PERP_SIGMA_M  = 0.07     # Rayleigh, in the METRIC frame
_V4_SEVERITY      = (0.6, 1.5)   # per-pair multiplier (clean vs drifty run)
_V4_REF_DIR_DEG   = 1.5
_V4_REF_PERP_M    = 0.02
_V4_FRAG_P        = (0.65, 0.25, 0.10)  # P(1|2|3 query fragments per real edge)
_V4_SIGN_FLIP     = 0.5      # SLAM endpoint order is arbitrary → ~50% flipped

_SCENARIOS  = ['room', 'submap', 'relocalize', 'loop', 'dense_sparse', 'manhattan', 'corridor', 'hard_noise', 'adversarial', 'outdoor', 'street', 'street_submap', 'collision']
# v6: add 'street' at 0.14, previous weights scaled by 0.86 (indoor
# distribution otherwise unchanged for retention when fine-tuning from v5).
# v7: add 'street_submap' at 0.12 (asymmetric spatial coverage — the KITTI
# submap-localization test showed v6 never saw a query covering only a
# fraction of the reference's AREA); previous weights scaled by 0.88.
# v8: add 'flip_room' at 0.10 (repetitive-room flip alias — the measured
# chess/seq05-class matcher blindness), previous weights scaled by 0.90.
# v9: flip_room 0.14. v10: flip_room REPLACED by 'collision' at 0.16
# (measured failure = learned-descriptor collision on dense similar
# lines, NOT rotational symmetry; flip_room modeled the wrong thing).
_SCENARIO_P = np.array([0.0756, 0.0567, 0.0567, 0.0693, 0.0378, 0.0882,
                        0.0567, 0.0567, 0.0945, 0.0567, 0.1029, 0.1008,
                        0.16])   # adversarial also bumped 0.078->0.0945
_SCENARIO_P = _SCENARIO_P / _SCENARIO_P.sum()   # exact normalization


# ── Plücker line primitives ───────────────────────────────────────────────────

def _random_rotation() -> np.ndarray:
    A = np.random.randn(3, 3).astype(np.float64)
    Q, R = np.linalg.qr(A)
    Q *= np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float32)


def _apply_sim3(L6: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    m, d  = L6[:, :3], L6[:, 3:]
    d_new = (R @ d.T).T
    m_new = s * (R @ m.T).T + np.cross(t[None], d_new)
    return np.concatenate([m_new, d_new], axis=1).astype(np.float32)


def _physical_noise(L: np.ndarray, dir_sigma_deg: float, perp_sigma_m: float,
                    dir_cap_deg: float = 90.0) -> np.ndarray:
    """Perturb lines physically: Rayleigh rotation of the direction plus a
    Rayleigh perpendicular offset of the foot point, then recompute m.
    Preserves m·d = 0 (unlike additive Gaussian noise on m)."""
    if len(L) == 0:
        return L
    m, d = L[:, :3], L[:, 3:]
    n = len(L)
    p0 = np.cross(d, m)                                   # foot of perpendicular
    ax = np.cross(d, np.random.randn(n, 3))
    ax /= np.linalg.norm(ax, axis=1, keepdims=True) + 1e-9
    ang = np.radians(np.minimum(np.random.rayleigh(dir_sigma_deg, n), dir_cap_deg))
    ca, sa = np.cos(ang)[:, None], np.sin(ang)[:, None]
    d_new = ca * d + sa * np.cross(ax, d) + (1 - ca) * (ax * d).sum(1, keepdims=True) * ax
    d_new /= np.linalg.norm(d_new, axis=1, keepdims=True) + 1e-9
    off = np.random.randn(n, 3)
    off -= (off * d_new).sum(1, keepdims=True) * d_new
    off *= (np.random.rayleigh(perp_sigma_m, n)
            / (np.linalg.norm(off, axis=1) + 1e-9))[:, None]
    m_new = np.cross(p0 + off, d_new)
    return np.concatenate([m_new, d_new], axis=1).astype(np.float32)


def _make_outliers(n: int, n_clusters: int = 5, spread: float = 0.2,
                   pos_range: float = 3.0) -> np.ndarray:
    if n == 0:
        return np.zeros((0, 6), np.float32)
    n_per   = max(1, n // n_clusters)
    extras  = n - n_per * n_clusters
    anchors = np.random.randn(n_clusters, 3).astype(np.float32)
    anchors /= np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-9
    parts = []
    for i, a in enumerate(anchors):
        cnt = n_per + (1 if i < extras else 0)
        d = a[None] + np.random.randn(cnt, 3).astype(np.float32) * spread
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        p = np.random.uniform(-pos_range, pos_range, (cnt, 3)).astype(np.float32)
        parts.append(np.concatenate([np.cross(p, d), d], axis=1).astype(np.float32))
    out = np.concatenate(parts, 0)
    return out[np.random.permutation(len(out))][:n]


def _make_plane_patch(n: int, pos_range: float = 3.0) -> np.ndarray:
    if n == 0:
        return np.zeros((0, 6), np.float32)
    normal = np.random.randn(3).astype(np.float64)
    normal /= np.linalg.norm(normal) + 1e-9
    perp = np.random.randn(3).astype(np.float64)
    perp -= perp.dot(normal) * normal
    u = perp / (np.linalg.norm(perp) + 1e-9)
    v = np.cross(normal, u)
    angles = np.random.uniform(0.0, 2.0 * np.pi, n)
    D = (np.outer(np.cos(angles), u) + np.outer(np.sin(angles), v)).astype(np.float32)
    D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-9
    a = np.random.uniform(-pos_range, pos_range, n)
    b = np.random.uniform(-pos_range, pos_range, n)
    P = (np.outer(a, u) + np.outer(b, v)).astype(np.float32)
    return np.concatenate([np.cross(P, D), D], axis=1).astype(np.float32)


def _make_wireframe(n: int, scale: float = 2.0) -> np.ndarray:
    if n == 0:
        return np.zeros((0, 6), np.float32)
    dims = np.random.uniform(0.3, 1.5, 3) * scale
    signs = np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],
                      [ 1,-1,-1],[ 1,-1,1],[ 1, 1,-1],[ 1, 1,1]], np.float64)
    corners = ((_random_rotation().astype(np.float64) @ (signs * dims[None]).T).T
               + np.random.uniform(-scale, scale, 3))
    edge_pairs = [(0,1),(2,3),(4,5),(6,7),(0,2),(1,3),(4,6),(5,7),(0,4),(1,5),(2,6),(3,7)]
    lines = []
    for a, b in edge_pairs:
        diff = corners[b] - corners[a]
        ln = np.linalg.norm(diff)
        if ln < 1e-9:
            continue
        d = (diff / ln).astype(np.float32)
        p = ((corners[a] + corners[b]) * 0.5).astype(np.float32)
        lines.append(np.concatenate([np.cross(p, d), d]))
    if not lines:
        return _make_outliers(n)
    arr = np.array(lines, np.float32)
    return arr[np.random.choice(len(arr), n, replace=(n > len(arr)))]


def _make_line_bundle(n: int, spread: float = 0.5, pos_range: float = 3.0) -> np.ndarray:
    if n == 0:
        return np.zeros((0, 6), np.float32)
    focus = np.random.uniform(-pos_range, pos_range, 3).astype(np.float32)
    D = np.random.randn(n, 3).astype(np.float32)
    D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-9
    offsets = np.random.randn(n, 3).astype(np.float32) * spread
    P = focus[None] + offsets - (offsets * D).sum(axis=1, keepdims=True) * D
    return np.concatenate([np.cross(P, D), D], axis=1).astype(np.float32)


def _make_parallel_group(n: int, spread: float = 0.35, pos_range: float = 3.0) -> np.ndarray:
    if n == 0:
        return np.zeros((0, 6), np.float32)
    anchor = np.random.randn(3).astype(np.float32)
    anchor /= np.linalg.norm(anchor) + 1e-9
    D = anchor[None] + np.random.randn(n, 3).astype(np.float32) * spread
    D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-9
    P = np.random.uniform(-pos_range, pos_range, (n, 3)).astype(np.float32)
    return np.concatenate([np.cross(P, D), D], axis=1).astype(np.float32)


def _make_grid_patch(n: int, pos_range: float = 3.0) -> np.ndarray:
    if n == 0:
        return np.zeros((0, 6), np.float32)
    normal = np.random.randn(3).astype(np.float64)
    normal /= np.linalg.norm(normal) + 1e-9
    perp = np.random.randn(3).astype(np.float64)
    perp -= perp.dot(normal) * normal
    u = (perp / (np.linalg.norm(perp) + 1e-9)).astype(np.float32)
    v = np.cross(normal, u).astype(np.float32)
    nh, nv = n // 2, n - n // 2
    Dh = np.tile(u, (nh, 1))
    Ph = (np.random.uniform(-pos_range, pos_range, (nh, 1)) * v[None]
          + np.random.uniform(-pos_range, pos_range, (nh, 1)) * u[None]).astype(np.float32)
    Dv = np.tile(v, (nv, 1))
    Pv = (np.random.uniform(-pos_range, pos_range, (nv, 1)) * u[None]
          + np.random.uniform(-pos_range, pos_range, (nv, 1)) * v[None]).astype(np.float32)
    return np.concatenate([
        np.concatenate([np.cross(Ph, Dh), Dh], axis=1),
        np.concatenate([np.cross(Pv, Dv), Dv], axis=1),
    ], axis=0).astype(np.float32)


def _make_staircase(n: int) -> np.ndarray:
    if n == 0:
        return np.zeros((0, 6), np.float32)
    R = _random_rotation().astype(np.float64)
    step_dir, up_dir, depth_dir = R[:, 0], R[:, 1], R[:, 2]
    step_h = np.random.uniform(0.15, 0.30)
    step_d = np.random.uniform(0.25, 0.40)
    origin = np.random.uniform(-2.0, 2.0, 3)
    # Cap at 15 steps so the staircase stays within ~3m displacement (bounded moments).
    # n lines are sampled with replacement from these fixed steps.
    n_steps = 15
    lines = []
    for i in range(n_steps):
        o = (origin + i * (step_h * up_dir + step_d * depth_dir)).astype(np.float32)
        d = step_dir.astype(np.float32)
        lines.append(np.concatenate([np.cross(o, d), d]))
        d2 = up_dir.astype(np.float32)
        lines.append(np.concatenate([np.cross(o, d2), d2]))
    arr = np.array(lines, np.float32)
    return arr[np.random.choice(len(arr), n, replace=(n > len(arr)))]


def _make_manhattan_world(n: int, pos_range: float = 3.0) -> np.ndarray:
    """Three orthogonal line families — mimics walls/floor/ceiling of real indoor rooms."""
    if n == 0:
        return np.zeros((0, 6), np.float32)
    R = _random_rotation()
    parts = []
    for i in range(3):
        ni = n // 3 + (1 if i < n % 3 else 0)
        if ni == 0:
            continue
        base = R[:, i].astype(np.float32)
        noise_std = float(np.random.uniform(0.0, 0.07))
        d = base[None] + np.random.randn(ni, 3).astype(np.float32) * noise_std
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        p = np.random.uniform(-pos_range, pos_range, (ni, 3)).astype(np.float32)
        parts.append(np.concatenate([np.cross(p, d), d], axis=1).astype(np.float32))
    return np.concatenate(parts).astype(np.float32)


def _make_corridor_pool(n: int, pos_range: float = 3.0) -> np.ndarray:
    """One or two dominant directions — mimics corridors, staircases, long hallways."""
    if n == 0:
        return np.zeros((0, 6), np.float32)
    R = _random_rotation()
    n_dirs = np.random.randint(1, 3)
    parts = []
    splits = np.random.dirichlet(np.ones(n_dirs)) * n
    for i, ni in enumerate(splits.astype(int)):
        if ni == 0:
            continue
        base = R[:, i % 3].astype(np.float32)
        noise_std = float(np.random.uniform(0.0, 0.10))
        d = base[None] + np.random.randn(ni, 3).astype(np.float32) * noise_std
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        p = np.random.uniform(-pos_range, pos_range, (ni, 3)).astype(np.float32)
        parts.append(np.concatenate([np.cross(p, d), d], axis=1).astype(np.float32))
    return np.concatenate(parts).astype(np.float32)


def _make_adversarial_pool(n: int, pos_range: float = 3.0) -> np.ndarray:
    """
    Three tight direction-cluster families aligned to orthogonal axes of a random frame.
    All clusters span the full spatial volume → the model must use moment context, not
    direction alone, to distinguish correspondences (mirrors wall/floor/ceiling ambiguity
    in structured indoor scenes).
    """
    if n == 0:
        return np.zeros((0, 6), np.float32)
    R = _random_rotation()
    parts = []
    for i in range(3):
        ni = n // 3 + (1 if i < n % 3 else 0)
        if ni == 0:
            continue
        base = R[:, i].astype(np.float32)
        spread = float(np.random.uniform(0.02, 0.09))   # tight cluster, but not perfectly parallel
        d = base[None] + np.random.randn(ni, 3).astype(np.float32) * spread
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        p = np.random.uniform(-pos_range, pos_range, (ni, 3)).astype(np.float32)
        parts.append(np.concatenate([np.cross(p, d), d], axis=1).astype(np.float32))
    pool = np.concatenate(parts, axis=0)
    return pool[np.random.permutation(len(pool))].astype(np.float32)


def _make_large_scale_pool(n: int, pos_range: float = 12.0) -> np.ndarray:
    """Building-scale outdoor pool: large planar facades + manhattan structures."""
    if n == 0:
        return np.zeros((0, 6), np.float32)
    n_mw = int(n * np.random.uniform(0.30, 0.65))
    n_plane = n - n_mw
    parts = []
    if n_mw > 0:
        parts.append(_make_manhattan_world(n_mw, pos_range=pos_range))
    if n_plane > 0:
        parts.append(_make_plane_patch(n_plane, pos_range=pos_range))
    pool = np.concatenate(parts, axis=0)
    return pool[np.random.permutation(len(pool))].astype(np.float32)


def _make_street_pool(n: int) -> np.ndarray:
    """v6 street-canyon pool, calibrated to the measured KITTI camera/LiDAR
    line maps (2026-07-16: extent ~(80-520) x (8-12) x (5-10) m slab,
    60-75% street-axis horizontal directions, 13-28% verticals, periodic
    facade repetition = the 180-deg flip alias, s_gt ~ 1.0).

    The returned order is a weighted shuffle with verticals and ground
    edges first: the inlier slice big_pool[:k] is therefore rich in the
    structures both sensors genuinely share as infinite lines (wall
    corners, curbs), while facade horizontals — whose height bands differ
    between LiDAR (bottom metres) and camera (full height) — land mostly
    in the outlier slices. This models the height-band disjointness that
    caps real cross-modal overlap at 18-24%."""
    if n == 0:
        return np.zeros((0, 6), np.float32)
    # 1-3 chained street segments (straight run, L-corner, U/Z route —
    # KITTI 06 is a ~520 m straight, 07 a 240x240 m loop). Corners spread
    # the foot points of street-axis lines across the plane, matching the
    # measured 20-60:1 in-plane slab anisotropy.
    n_seg = np.random.randint(1, 4)
    W = float(np.random.uniform(3.0, 8.0))        # half-width
    H = float(np.random.uniform(4.0, 10.0))       # facade height
    period = float(np.random.uniform(5.0, 15.0))  # facade repetition
    ez = np.array([0.0, 0.0, 1.0])
    lines, wts = [], []

    def add(p, d, w, Ryaw, org):
        d = Ryaw @ (np.asarray(d, np.float64) + np.random.randn(3) * 0.05)
        d /= np.linalg.norm(d) + 1e-9
        p = Ryaw @ np.asarray(p, np.float64) + org
        lines.append(np.concatenate([np.cross(p, d), d]))
        wts.append(w)

    org = np.zeros(3)
    yaw = 0.0
    per_seg = (2 * n) // n_seg + 1
    for _seg in range(n_seg):
        L = float(np.random.uniform(40.0, 250.0))
        c, si = np.cos(yaw), np.sin(yaw)
        Ryaw = np.array([[c, -si, 0.0], [si, c, 0.0], [0.0, 0.0, 1.0]])
        ex = np.array([1.0, 0.0, 0.0])
        made = 0
        while made < per_seg:
            r = np.random.random()
            side = -1.0 if np.random.random() < 0.5 else 1.0
            xg = (np.round(np.random.uniform(0.0, L) / period) * period
                  + np.random.randn() * 0.4)
            if r < 0.20:    # vertical wall corner / pole (shared cross-modally)
                add([xg, side * W * np.random.uniform(0.9, 1.1), 0.0],
                    ez, 1.0, Ryaw, org)
            elif r < 0.60:  # facade horizontal along the street axis
                add([np.random.uniform(0.0, L), side * W,
                     np.random.uniform(0.3, H)], ex, 0.35, Ryaw, org)
            elif r < 0.74:  # ground / curb / lane edges (shared)
                add([np.random.uniform(0.0, L),
                     np.random.uniform(-W, W), 0.0], ex, 0.8, Ryaw, org)
            elif r < 0.82:  # cross-street horizontal (facade returns, gates)
                add([xg, side * W, np.random.uniform(0.3, H)],
                    [0.0, 1.0, 0.0], 0.5, Ryaw, org)
            else:           # oblique clutter (vegetation, vehicles, LiDAR
                            # curvature edges) — real maps measure only
                            # 0.15-0.21 dominant-direction concentration
                add([np.random.uniform(0.0, L), np.random.uniform(-W, W),
                     np.random.uniform(0.0, 0.6 * H)],
                    np.random.randn(3), 0.4, Ryaw, org)
            made += 1
        # chain: advance to segment end, turn ~90 deg either way
        org = org + Ryaw @ np.array([L, 0.0, 0.0])
        yaw += np.deg2rad(np.random.uniform(70.0, 110.0)
                          * (1 if np.random.random() < 0.5 else -1))
    A = np.stack(lines).astype(np.float32)
    # recentre so moments stay O(extent), like a SLAM map's local frame
    p0 = np.cross(A[:, :3], A[:, 3:])
    shift = -np.median(p0, axis=0)
    A[:, :3] += np.cross(shift[None].astype(np.float32), A[:, 3:])
    w = np.asarray(wts, np.float64)
    order = np.random.choice(len(A), size=n, replace=False, p=w / w.sum())
    return A[order]


def _apply_drift(lines: np.ndarray, n_groups: int = 4, sigma: float = 0.10) -> np.ndarray:
    """
    Spatially correlated drift — simulates accumulated SLAM drift.
    Lines are partitioned into n_groups random spatial buckets; each bucket
    is rigidly translated by δ ~ N(0, σ²I), i.e. m += δ × d, which is the
    exact Plücker transform of a translation and preserves m·d = 0.
    """
    n = len(lines)
    if n == 0:
        return lines
    out = lines.copy()
    group_ids = np.random.randint(0, n_groups, n)
    for g in range(n_groups):
        mask = group_ids == g
        if mask.sum() == 0:
            continue
        delta = np.random.randn(3).astype(np.float32) * sigma
        out[mask, :3] += np.cross(delta[None], out[mask, 3:])
    return out


def _make_confuser_lines(n: int, ref_lines: np.ndarray, spread: float = 0.08,
                         pos_range: float = 3.0,
                         moment_dup: bool = False) -> np.ndarray:
    """Outlier lines with directions near those in ref_lines — structurally
    confusing. With moment_dup=True (v10), the confuser is a near-DUPLICATE:
    same direction AND a moment close to the source line (small perpendicular
    foot-point offset), so its full 6D descriptor collides with a real line
    while it has NO true correspondence. This is the measured real-map
    failure — the learned descriptor produces spurious matches on
    locally-similar lines — so it forces global-context discrimination."""
    if n == 0 or len(ref_lines) == 0:
        return _make_outliers(n, pos_range=pos_range)
    idx = np.random.randint(0, len(ref_lines), n)
    d = ref_lines[idx, 3:].copy()
    d += np.random.randn(n, 3).astype(np.float32) * spread
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    if moment_dup:
        # near-duplicate: shift the source foot point by a modest perpendicular
        # offset (line stays "nearby and parallel" — a strong descriptor twin)
        m0, d0 = ref_lines[idx, :3], ref_lines[idx, 3:]
        p0 = np.cross(d0, m0)                          # foot of perpendicular
        off = np.random.randn(n, 3).astype(np.float32)
        off -= (off * d).sum(1, keepdims=True) * d
        off /= np.linalg.norm(off, axis=1, keepdims=True) + 1e-9
        off *= np.random.uniform(0.15, 0.7, (n, 1)).astype(np.float32)
        return np.concatenate([np.cross(p0 + off, d), d],
                              axis=1).astype(np.float32)
    p = np.random.uniform(-pos_range, pos_range, (n, 3)).astype(np.float32)
    return np.concatenate([np.cross(p, d), d], axis=1).astype(np.float32)


def _make_collision_pool(n: int, pos_range: float = 3.0) -> np.ndarray:
    """v10 dense-room pool: 2-4 wall/floor planes, each carrying many lines
    whose directions vary SMOOTHLY across a ~25deg arc (graded similarity,
    not discrete clusters) at dense positions — the locally-ambiguous,
    highly-similar line structure of a real cluttered room (chess/stairs),
    where the measured failure is learned-descriptor collision, NOT rotational
    symmetry (ref self-similarity is low, 0.01-0.03). Forces the network to
    separate near-identical lines by global moment context."""
    if n == 0:
        return np.zeros((0, 6), np.float32)
    n_planes = np.random.randint(2, 5)
    parts = []
    per = n // n_planes + 1
    for _ in range(n_planes):
        normal = np.random.randn(3); normal /= np.linalg.norm(normal) + 1e-9
        u = np.random.randn(3); u -= u.dot(normal) * normal
        u /= np.linalg.norm(u) + 1e-9
        v = np.cross(normal, u)
        # a dominant in-plane axis with graded angular spread (~25 deg arc)
        base_ang = np.random.uniform(0, 2 * np.pi)
        angs = base_ang + np.random.uniform(-0.22, 0.22, per)  # ~+-12.5 deg
        d = (np.cos(angs)[:, None] * u + np.sin(angs)[:, None] * v)
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        a = np.random.uniform(-pos_range, pos_range, per)
        b = np.random.uniform(-pos_range, pos_range, per)
        off = np.random.uniform(-0.3, 0.3, per)        # slight off-plane
        p = (a[:, None] * u + b[:, None] * v
             + off[:, None] * normal).astype(np.float32)
        parts.append(np.concatenate([np.cross(p, d), d.astype(np.float32)],
                                    axis=1).astype(np.float32))
    pool = np.concatenate(parts, axis=0)
    return pool[np.random.permutation(len(pool))].astype(np.float32)[:n]


def _make_flip_room_pool(n: int, pos_range: float = 3.0) -> np.ndarray:
    """v8 repetitive room: a Manhattan-world room containing a COPY of a
    large fraction of its own structure rotated by 90/180 deg about the
    vertical axis (with jitter) — the internal symmetry that makes real
    rooms like 7-Scenes chess flip-alias. A matcher that relies on local
    direction/moment patterns scores ZERO here (every line has a
    convincing rotated twin); disambiguation requires the asymmetric
    minority + global moment context. Targets the measured chess/seq05 /
    stairs failure mode (matcher P@200 = 0 despite 1,000 GT pairs)."""
    if n == 0:
        return np.zeros((0, 6), np.float32)
    # v9 recalibration (v8's exact 2deg/5cm twin made the matcher learn a
    # brittle "reject all symmetric matches" rule -> real-pair REGRESSION,
    # zero transfer). Fixes: (a) the twin is a NEAR-flip, angle jittered off
    # 90/180 by up to +-12deg (real rooms are not exactly orthogonal);
    # (b) heavy, variable geometric noise (4-11deg / 8-22cm) so twins are
    # convincing-but-imperfect, not artificial duplicates; (c) more partial
    # (45-75% of base) and a LARGER asymmetric minority — the disambiguating
    # signal the model must learn to weight.
    n_base = max(8, int(n * np.random.uniform(0.35, 0.50)))
    base = _make_manhattan_world(n_base, pos_range=pos_range)
    ang = np.deg2rad(np.random.choice([90.0, 180.0, 270.0])
                     + np.random.uniform(-12.0, 12.0))
    # near-vertical flip axis, slightly tilted (real rooms aren't gravity-exact)
    axis = np.array([np.random.uniform(-0.12, 0.12),
                     np.random.uniform(-0.12, 0.12), 1.0])
    axis /= np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    Rz = (np.eye(3) + np.sin(ang) * K
          + (1 - np.cos(ang)) * (K @ K)).astype(np.float32)
    m, d = base[:, :3], base[:, 3:]
    twin = np.concatenate([(Rz @ m.T).T, (Rz @ d.T).T],
                          axis=1).astype(np.float32)
    twin = _physical_noise(twin, np.random.uniform(4.0, 11.0),
                           np.random.uniform(0.08, 0.22))
    n_copy = int(len(twin) * np.random.uniform(0.45, 0.75))
    twin = twin[np.random.choice(len(twin), n_copy, replace=False)]
    # asymmetric minority: unique clutter that breaks the symmetry
    n_rest = max(0, n - len(base) - len(twin))
    parts = [base, twin]
    if n_rest > 0:
        parts.append(_make_outliers(n_rest, pos_range=pos_range))
    pool = np.concatenate(parts, axis=0)
    return pool[np.random.permutation(len(pool))].astype(np.float32)[:n]


def _make_pool(n: int, pool_type: str = 'mixed') -> np.ndarray:
    if n == 0:
        return np.zeros((0, 6), np.float32)

    if pool_type == 'collision':
        return _make_collision_pool(n)

    if pool_type in ('street', 'street_submap'):
        # v6: no shuffle here — _make_street_pool's weighted order IS the
        # inlier-selection bias (verticals/ground first)
        return _make_street_pool(n)

    if pool_type == 'manhattan':
        n_mw = int(n * np.random.uniform(0.55, 0.85))
        n_misc = n - n_mw
        parts = [_make_manhattan_world(n_mw)]
        if n_misc > 0:
            for maker, k in zip(
                [_make_plane_patch, _make_wireframe, _make_outliers],
                np.maximum(0, np.round(
                    np.random.dirichlet(np.ones(3)) * n_misc
                ).astype(int))
            ):
                if k > 0:
                    parts.append(maker(k))
        pool = np.concatenate(parts, axis=0)
        return pool[np.random.permutation(len(pool))].astype(np.float32)

    if pool_type == 'corridor':
        n_corr = int(n * np.random.uniform(0.55, 0.85))
        n_misc = n - n_corr
        parts = [_make_corridor_pool(n_corr)]
        if n_misc > 0:
            for maker, k in zip(
                [_make_staircase, _make_parallel_group, _make_outliers],
                np.maximum(0, np.round(
                    np.random.dirichlet(np.ones(3)) * n_misc
                ).astype(int))
            ):
                if k > 0:
                    parts.append(maker(k))
        pool = np.concatenate(parts, axis=0)
        return pool[np.random.permutation(len(pool))].astype(np.float32)

    if pool_type == 'adversarial':
        # Dominant adversarial clusters + small outlier fringe
        n_adv  = int(n * np.random.uniform(0.65, 0.90))
        n_misc = n - n_adv
        parts  = [_make_adversarial_pool(n_adv)]
        if n_misc > 0:
            parts.append(_make_outliers(n_misc))
        pool = np.concatenate(parts, axis=0)
        return pool[np.random.permutation(len(pool))].astype(np.float32)

    if pool_type == 'outdoor':
        pos_range = float(np.random.uniform(8.0, 20.0))
        return _make_large_scale_pool(n, pos_range=pos_range)

    # Default 'mixed' pool
    n_planes     = np.random.randint(0, 5)
    n_boxes      = np.random.randint(0, 4)
    n_bundles    = np.random.randint(0, 3)
    n_parallels  = np.random.randint(0, 3)
    n_grids      = np.random.randint(0, 3)
    n_staircases = np.random.randint(0, 3)
    tasks = ([_make_plane_patch]    * n_planes     +
             [_make_wireframe]      * n_boxes      +
             [_make_line_bundle]    * n_bundles    +
             [_make_parallel_group] * n_parallels  +
             [_make_grid_patch]     * n_grids      +
             [_make_staircase]      * n_staircases)
    if not tasks:
        return _make_outliers(n)
    fracs = np.random.dirichlet(np.ones(len(tasks) + 1))
    alloc = np.floor(fracs * n).astype(int)
    alloc[-1] += n - alloc.sum()
    parts = [maker(k) for maker, k in zip(tasks, alloc[:-1]) if k > 0]
    if alloc[-1] > 0:
        parts.append(_make_outliers(alloc[-1]))
    pool = np.concatenate(parts, axis=0)
    return pool[np.random.permutation(len(pool))].astype(np.float32)


# ── Pair generator ────────────────────────────────────────────────────────────

def _generate_pair() -> dict:
    scenario = np.random.choice(_SCENARIOS, p=_SCENARIO_P)

    pool_type   = 'mixed'
    use_confusers = False  # whether to add near-parallel confuser outliers

    if scenario == 'room':
        n2           = np.random.randint(80, 1100)
        n1           = max(30, min(int(n2 * np.random.uniform(0.3, 3.0)), 1100))
        overlap_frac = float(np.random.beta(3.0, 2.0))
        noise1       = float(np.exp(np.random.uniform(np.log(0.005), np.log(0.12))))
        noise2       = float(np.exp(np.random.uniform(np.log(0.001), np.log(0.04))))
    elif scenario == 'submap':
        n1           = np.random.randint(30, 150)
        n2           = np.random.randint(max(n1, 80), 700)
        overlap_frac = float(np.random.beta(2.5, 3.5))
        noise1       = float(np.exp(np.random.uniform(np.log(0.010), np.log(0.15))))
        noise2       = float(np.exp(np.random.uniform(np.log(0.001), np.log(0.04))))
    elif scenario == 'relocalize':
        base         = int(np.exp(np.random.uniform(np.log(30), np.log(400))))
        r            = np.random.uniform(0.5, 2.0)
        n1           = max(30, int(base * r))
        n2           = max(30, int(base / r))
        overlap_frac = float(np.random.beta(2.5, 2.5))
        noise1 = noise2 = float(np.exp(np.random.uniform(np.log(0.005), np.log(0.10))))
    elif scenario == 'loop':
        n1 = n2      = np.random.randint(40, 500)
        overlap_frac = float(np.random.beta(5.0, 2.0))
        noise1 = noise2 = float(np.exp(np.random.uniform(np.log(0.003), np.log(0.06))))
    elif scenario == 'dense_sparse':
        n2           = np.random.randint(150, 700)
        n1           = max(30, int(n2 * np.random.uniform(0.10, 0.50)))
        overlap_frac = float(np.random.beta(2.5, 2.0))
        noise1       = float(np.exp(np.random.uniform(np.log(0.005), np.log(0.12))))
        noise2       = float(np.exp(np.random.uniform(np.log(0.001), np.log(0.03))))
    elif scenario == 'manhattan':
        # Indoor manhattan world — 3 orthogonal line families, moderate noise
        n2           = np.random.randint(80, 900)
        n1           = max(30, min(int(n2 * np.random.uniform(0.2, 2.5)), 900))
        overlap_frac = float(np.random.beta(3.0, 2.5))
        noise1       = float(np.exp(np.random.uniform(np.log(0.005), np.log(0.12))))
        noise2       = float(np.exp(np.random.uniform(np.log(0.001), np.log(0.04))))
        pool_type    = 'manhattan'
    elif scenario == 'hard_noise':
        # Very few inliers (5-20%), high noise, confuser outliers near inlier directions.
        # Models the worst-case SLAM drift / low-overlap scenario.
        n2           = np.random.randint(80, 600)
        n1           = max(30, int(n2 * np.random.uniform(0.3, 2.0)))
        overlap_frac = float(np.random.beta(1.5, 6.0))   # mode ≈ 0.08, mean ≈ 0.20
        noise1       = float(np.exp(np.random.uniform(np.log(0.05), np.log(0.50))))  # v3: raised ceiling 0.30→0.50
        noise2       = float(np.exp(np.random.uniform(np.log(0.01), np.log(0.10))))
        use_confusers = True
        pool_type    = 'mixed'
    elif scenario == 'adversarial':
        # 3 tight orthogonal direction families, mixed positions — forces moment-based disambiguation.
        # Directly targets the structured/indoor dense-overlap failure mode.
        n2           = np.random.randint(80, 800)
        n1           = max(30, min(int(n2 * np.random.uniform(0.3, 2.5)), 800))
        overlap_frac = float(np.random.beta(3.0, 2.5))
        noise1       = float(np.exp(np.random.uniform(np.log(0.005), np.log(0.12))))
        noise2       = float(np.exp(np.random.uniform(np.log(0.001), np.log(0.04))))
        pool_type    = 'adversarial'
        use_confusers = True
    elif scenario == 'outdoor':
        # Large-scale outdoor: building facades and structures at 8–20 m spatial range.
        # Reference |m| ~ 10–20 m (metric LiDAR); query |m| ~ 4–8 m (mono drone / camera).
        n2           = np.random.randint(60, 500)
        n1           = max(30, min(int(n2 * np.random.uniform(0.2, 2.0)), 500))
        overlap_frac = float(np.random.beta(2.5, 2.0))
        noise1       = float(np.exp(np.random.uniform(np.log(0.02), np.log(0.60))))
        noise2       = float(np.exp(np.random.uniform(np.log(0.005), np.log(0.20))))
        pool_type    = 'outdoor'
    elif scenario == 'street':
        # v6 KITTI-calibrated street canyon: metric-metric scale, large
        # elongated clouds (real: query 1.3-2.6k, ref 1.8-2.3k lines),
        # low overlap (measured 18-24% matched query fraction), periodic
        # facades whose repetitions ARE the confuser structure — no
        # synthetic confusers needed; the pool aliases itself under
        # 180-deg flips and street-axis translations.
        n2           = np.random.randint(600, 2200)
        n1           = max(200, min(int(n2 * np.random.uniform(0.4, 1.6)),
                                    2200))
        overlap_frac = float(np.random.beta(2.0, 7.0))   # mean ~0.22
        noise1       = 0.3
        noise2       = 0.1
        pool_type    = 'street'
    elif scenario == 'street_submap':
        # v7: submap localization — the query covers only a contiguous
        # 15-35% WINDOW of the reference street's area (a short mono drive
        # inside a full prior LiDAR map), at mono-ambiguous scale. The
        # reference keeps the whole street; query inliers AND outliers are
        # confined to the window (asymmetric spatial coverage — the
        # distribution v6 lacked). Overlap is the shared fraction WITHIN
        # the window (same height-band logic as 'street').
        n2           = np.random.randint(800, 2200)
        n1           = np.random.randint(150, 600)
        overlap_frac = float(np.random.beta(2.0, 6.0))   # mean ~0.25 of window
        noise1       = 0.3
        noise2       = 0.1
        pool_type    = 'street_submap'
    elif scenario == 'collision':
        # v10 dense-room descriptor-collision (see _make_collision_pool):
        # many locally-similar lines + heavy near-DUPLICATE query confusers
        # (moment_dup). Targets the measured chess/seq05 failure: the matcher
        # produces spurious flip-consistent matches because local descriptors
        # collide, not because the room is symmetric. High density, moderate
        # overlap, confusers ON.
        n2           = np.random.randint(300, 1100)
        n1           = max(80, min(int(n2 * np.random.uniform(0.4, 2.0)),
                                   1100))
        overlap_frac = float(np.random.beta(2.5, 3.0))
        noise1       = float(np.exp(np.random.uniform(np.log(0.005),
                                                      np.log(0.12))))
        noise2       = float(np.exp(np.random.uniform(np.log(0.001),
                                                      np.log(0.04))))
        use_confusers = True
        pool_type    = 'collision'
    else:  # corridor
        # Corridor / stairwell — 1-2 dominant directions, higher noise
        n2           = np.random.randint(50, 450)
        n1           = max(30, min(int(n2 * np.random.uniform(0.2, 1.8)), 450))
        overlap_frac = float(np.random.beta(2.0, 3.0))
        noise1       = float(np.exp(np.random.uniform(np.log(0.010), np.log(0.15))))
        noise2       = float(np.exp(np.random.uniform(np.log(0.002), np.log(0.05))))
        pool_type    = 'corridor'

    k  = max(_MIN_INLIERS, int(overlap_frac * min(n1, n2)))
    n1 = max(n1, k)
    n2 = max(n2, k)

    # v4: scale distribution matched to measured real pairs (median 2.1).
    log_s = np.random.normal(_SCALE_LOG_CENTER, _SCALE_LOG_STD)
    log_s = np.clip(log_s, np.log(_SCALE_CLIP[0]), np.log(_SCALE_CLIP[1]))
    s   = float(np.exp(log_s))
    R   = _random_rotation()
    # v2 fix: t_range_eff = 0.4 * s so that the inverse translation
    # |t_i| = |t|/s ≤ 0.4m for ALL s (both s < 1 and s > 1).
    # Prevents the t × d cross term from inflating query moments.
    t_range_eff = 0.4 * s
    if scenario == 'street':
        # v6: both maps are metric (LiDAR GT frame vs depth-lifted camera
        # map) — s_gt measured 0.999/1.003 on KITTI. Translation offsets
        # are tens of metres at street scale.
        s = float(np.exp(np.clip(np.random.normal(0.0, 0.12),
                                 np.log(0.7), np.log(1.4))))
        t_range_eff = 20.0 * s
    elif scenario == 'street_submap':
        # v7: a standalone mono submap has ARBITRARY scale (unlike the
        # depth-lifted full camera maps) and lives in its own local frame
        s = float(np.exp(np.clip(np.random.normal(0.0, 0.45),
                                 np.log(1 / 3.0), np.log(3.0))))
        t_range_eff = 20.0 * s
    t   = np.random.uniform(-t_range_eff, t_range_eff, 3).astype(np.float32)
    s_i, R_i, t_i = 1.0 / s, R.T, -(R.T @ t) / s

    n_out1    = max(0, n1 - k)
    n_out2    = max(0, n2 - k)
    pool_size = k + n_out1 + n_out2 + max(k // 4, 8)
    # Allow near-parallel configurations (lowered from 0.30 so the model
    # sees the corridor/manhattan distributions it will face at test time).
    _NONDEGEN_THR = 0.05
    for _attempt in range(10):
        big_pool = _make_pool(pool_size, pool_type=pool_type)
        dirs = big_pool[:, 3:]
        sv = np.linalg.svd(dirs - dirs.mean(0, keepdims=True), compute_uv=False)
        if sv[-1] / (sv[0] + 1e-9) >= _NONDEGEN_THR:
            break

    if pool_type == 'street_submap':
        # v7: confine ALL query material (inliers + outliers) to one
        # contiguous spatial window of the street; the reference keeps the
        # whole route. A ball around a route point is a segment of the 1-D
        # street corridor. Partition order: [window | non-window | leftover
        # window] so the query slice is pure window and the ref-outlier
        # slice is dominated by the rest of the street.
        p0 = np.cross(big_pool[:, :3], big_pool[:, 3:])
        ctr = np.median(p0, axis=0)
        ext = float(np.linalg.norm(p0 - ctr, axis=1).max()) + 1e-6
        anchor = p0[np.random.randint(len(p0))]
        rho = np.random.uniform(0.12, 0.30) * ext
        win = np.linalg.norm(p0 - anchor, axis=1) < rho
        while win.sum() < 3 * _MIN_INLIERS and rho < 2.5 * ext:
            rho *= 1.4
            win = np.linalg.norm(p0 - anchor, axis=1) < rho
        n_win = int(win.sum())
        k = max(_MIN_INLIERS, min(k, int(0.7 * n_win)))
        n_out1 = min(n_out1, max(0, n_win - k))
        wi = np.where(win)[0]
        ni = np.where(~win)[0]
        big_pool = big_pool[np.concatenate([wi[:k + n_out1], ni,
                                            wi[k + n_out1:]])]
        n_out2 = min(n_out2, len(big_pool) - k - n_out1)

    inliers_ref = big_pool[:k].copy()
    out_q_ref   = big_pool[k:k + n_out1]
    out_r       = big_pool[k + n_out1:k + n_out1 + n_out2]

    # v4: per-pair severity replaces the per-scenario Gaussian noise levels;
    # hard scenarios are drifty runs (higher severity), not a different model.
    severity = np.random.uniform(*_V4_SEVERITY) * (2.0 if scenario == 'hard_noise' else 1.0)
    # v6: street positional noise is measured in street metres (KITTI GT
    # correspondences: perp median 0.32-0.35 m -> Rayleigh sigma ~0.28;
    # LiDAR reference edges are range-noisy too). Direction noise keeps the
    # v4 calibration (measured 5.4-6.0 deg median on KITTI — same as indoor).
    _street = scenario in ('street', 'street_submap')
    perp_q_sigma = 0.28 if _street else _V4_PERP_SIGMA_M
    perp_r_sigma = 0.08 if _street else _V4_REF_PERP_M
    drift_scale  = 6.0 if _street else 1.0

    if k > 0:
        # v4 fragmentation: each reference edge appears as 1–3 query fragments
        # with independent noise draws → many-to-one matches + near-duplicate
        # query lines, matching real LSD mono maps.
        n_frag = np.random.choice([1, 2, 3], size=k, p=_V4_FRAG_P)
        frag_src = np.repeat(np.arange(k), n_frag)      # fragment -> ref inlier
        inliers_q = _apply_sim3(inliers_ref[frag_src], s_i, R_i, t_i)
        # noise in the query frame; perpendicular noise is measured in the
        # METRIC frame, so divide by s
        inliers_q = _physical_noise(inliers_q,
                                    _V4_DIR_SIGMA_DEG * severity,
                                    perp_q_sigma * severity / s,
                                    _V4_DIR_CAP_DEG)
        inliers_ref = _physical_noise(inliers_ref, _V4_REF_DIR_DEG,
                                      perp_r_sigma)

        # Drift augmentation (20% of pairs): spatially correlated moment noise on query.
        if np.random.random() < 0.20:
            inliers_q = _apply_drift(
                inliers_q,
                n_groups=np.random.randint(2, 6),
                sigma=float(np.random.uniform(0.05, 0.25)) * drift_scale,
            )
    else:
        inliers_q = inliers_ref = np.zeros((0, 6), np.float32)
        frag_src = np.zeros(0, np.int64)
    k_q = len(frag_src)

    if use_confusers and k > 0:
        # Replace a fraction of regular outliers with confuser lines (near-parallel to inliers).
        n_conf = max(0, int(n_out1 * np.random.uniform(0.3, 0.7)))
        n_reg  = n_out1 - n_conf
        out_q_reg  = (_apply_sim3(out_q_ref[:n_reg], s_i, R_i, t_i) if n_reg > 0
                      else np.zeros((0, 6), np.float32))
        out_q_conf = _make_confuser_lines(n_conf, inliers_q, moment_dup=(scenario == 'collision')) if n_conf > 0 else np.zeros((0, 6), np.float32)
        out_q_parts = [p for p in [out_q_reg, out_q_conf] if len(p) > 0]
        out_q = np.concatenate(out_q_parts) if out_q_parts else np.zeros((0, 6), np.float32)
    else:
        out_q = (_apply_sim3(out_q_ref, s_i, R_i, t_i) if n_out1 > 0
                 else np.zeros((0, 6), np.float32))
    # v4: outliers are real scene lines too — same physical noise as inliers
    out_q = _physical_noise(out_q, _V4_DIR_SIGMA_DEG * severity,
                            perp_q_sigma * severity / s, _V4_DIR_CAP_DEG)

    q_parts = [p for p in [inliers_q, out_q]   if len(p) > 0]
    r_parts = [p for p in [inliers_ref, out_r] if len(p) > 0]
    query_all = (np.concatenate(q_parts, axis=0) if q_parts else _make_pool(max(n1, 10), pool_type))
    ref_all   = (np.concatenate(r_parts, axis=0) if r_parts else _make_pool(max(n2, 10), pool_type))

    # v4: ~50% Plücker sign flips on BOTH clouds (endpoint order is arbitrary
    # in SLAM; a flipped [m,d] is the same line, so labels are unaffected)
    for arr in (query_all, ref_all):
        flip = np.random.random(len(arr)) < _V4_SIGN_FLIP
        arr[flip] *= -1.0

    i_q = np.random.permutation(len(query_all))
    i_r = np.random.permutation(len(ref_all))
    query_all, ref_all = query_all[i_q], ref_all[i_r]
    q_pos = np.argsort(i_q)[:k_q]          # fragment j -> row in query_all
    r_pos = np.argsort(i_r)[:k]            # ref inlier i -> row in ref_all

    # v4 convention swap: plucker1 = REFERENCE (metric), plucker2 = QUERY —
    # same as the 7scenes_mesh / real generators, so datasets mix cleanly and
    # the dataloader's "normalize by plucker2 (query) std" stays correct.
    # (s, R, t) maps query -> reference, i.e. plucker2 -> plucker1.
    m_ref = r_pos[frag_src].astype(np.int32)
    m_qry = q_pos.astype(np.int32)

    return dict(
        plucker1 = ref_all.astype(np.float32),
        plucker2 = query_all.astype(np.float32),
        matches  = np.stack([m_ref, m_qry], 0) if k_q > 0 else np.zeros((2, 0), np.int32),
        R_gt     = R.astype(np.float32),
        t_gt     = t.reshape(3, 1).astype(np.float32),
        s_gt     = np.float32(s),
    )


# ── Worker (runs in child process) ────────────────────────────────────────────

def _worker(args):
    n, seed = args
    np.random.seed(seed)
    keys  = ['matches', 'plucker1', 'plucker2', 'R_gt', 't_gt', 's_gt']
    chunk = {k: [] for k in keys}
    for _ in range(n):
        pair = _generate_pair()
        for k in keys:
            chunk[k].append(pair[k])
    return chunk


# ── Dataset saving ────────────────────────────────────────────────────────────

def _save(data: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for k, v in data.items():
        with open(os.path.join(out_dir, f'{k}.pkl'), 'wb') as f:
            pickle.dump(v, f, protocol=4)
    s  = np.array(data['s_gt'])
    n1 = np.array([p.shape[0] for p in data['plucker1']])
    n2 = np.array([p.shape[0] for p in data['plucker2']])
    ki = np.array([m.shape[1] for m in data['matches']])
    print(f'  Saved {len(data["t_gt"])} pairs → {out_dir}')
    print(f'    scale   : [{s.min():.3f}, {s.max():.3f}]  median {np.median(s):.3f}')
    print(f'    n1      : [{n1.min()}, {n1.max()}]  median {int(np.median(n1))}')
    print(f'    n2      : [{n2.min()}, {n2.max()}]  median {int(np.median(n2))}')
    print(f'    inliers : [{ki.min()}, {ki.max()}]  median {int(np.median(ki))}')


# ── Main generation loop ──────────────────────────────────────────────────────

def generate_split(n_pairs: int, workers: int, chunk_size: int,
                   seed: int, label: str) -> dict:
    n_chunks  = max(1, (n_pairs + chunk_size - 1) // chunk_size)
    remaining = n_pairs
    tasks     = []
    for i in range(n_chunks):
        n = min(chunk_size, remaining)
        tasks.append((n, seed + i * 7919))
        remaining -= n
        if remaining <= 0:
            break

    keys     = ['matches', 'plucker1', 'plucker2', 'R_gt', 't_gt', 's_gt']
    combined = {k: [] for k in keys}
    t0, done = time.time(), 0

    if workers == 1:
        for task in tasks:
            chunk = _worker(task)
            for k in keys:
                combined[k].extend(chunk[k])
            done += task[0]
            elapsed = time.time() - t0
            print(f'  {label}: {done}/{n_pairs}  ({done/elapsed:.0f} pairs/s)', end='\r')
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for chunk in ex.map(_worker, tasks):
                for k in keys:
                    combined[k].extend(chunk[k])
                done += chunk_size
                elapsed = time.time() - t0
                print(f'  {label}: {min(done, n_pairs)}/{n_pairs}  '
                      f'({done/elapsed:.0f} pairs/s)', end='\r')

    print()
    return combined


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--n_train',    type=int, default=200_000)
    ap.add_argument('--n_valid',    type=int, default=2_000)
    ap.add_argument('--out_dir',    default=str(_ROOT / 'dataset'))
    ap.add_argument('--name',       default='synthetic_v3',
                    help='Dataset name prefix (default: synthetic_v3). '
                         'Outputs to <out_dir>/<name>_train and <out_dir>/<name>_valid.')
    ap.add_argument('--chunk_size', type=int, default=2_000)
    ap.add_argument('--workers',    type=int,
                    default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument('--seed',       type=int, default=42)
    args = ap.parse_args()

    print(f'Generating synthetic dataset  '
          f'train={args.n_train:,}  valid={args.n_valid:,}  '
          f'workers={args.workers}  seed={args.seed}')
    print(f'Output: {args.out_dir}  name: {args.name}')
    print(f'Scale distribution: LogNormal(log({np.exp(_SCALE_LOG_CENTER):.1f}), {_SCALE_LOG_STD}) '
          f'clipped to {_SCALE_RANGE}')

    print('\n=== Training split ===')
    _save(generate_split(args.n_train, args.workers, args.chunk_size,
                         seed=args.seed, label='train'),
          os.path.join(args.out_dir, f'{args.name}_train'))

    print('\n=== Validation split ===')
    _save(generate_split(args.n_valid, args.workers, args.chunk_size,
                         seed=args.seed + 999_983, label='valid'),
          os.path.join(args.out_dir, f'{args.name}_valid'))

    print('\nDone.')


if __name__ == '__main__':
    main()
