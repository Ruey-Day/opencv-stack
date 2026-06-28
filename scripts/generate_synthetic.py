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
python scripts/generate_synthetic.py
python scripts/generate_synthetic.py --n_train 500000 --n_valid 5000
python scripts/generate_synthetic.py --n_train 200000 --workers 8 --seed 123
"""
import os
import sys
import pickle
import argparse
import time
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


# ── Constants ─────────────────────────────────────────────────────────────────

_SCALE_RANGE      = (0.3, 8.0)   # tightened: avoids s_i > 3.3 to prevent moment explosion
_ROOM_SCALE_RANGE = (0.3, 5.0)
_MIN_INLIERS      = 20

# v2 calibration: mono→metric SLAM pairs have s > 1 (metric larger than mono).
# LogNormal(log(2.5), 1.0) → median s≈2.5, P25≈0.9, P75≈6.8; 75% of samples s > 1.
_SCALE_LOG_CENTER = np.log(2.5)
_SCALE_LOG_STD    = 1.0

_SCENARIOS  = ['room', 'submap', 'relocalize', 'loop', 'dense_sparse', 'manhattan', 'corridor', 'hard_noise', 'adversarial', 'outdoor']
_SCENARIO_P = np.array([0.12, 0.09,  0.09,       0.11,  0.06,          0.14,        0.09,        0.09,         0.12,          0.09])


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


def _apply_drift(lines: np.ndarray, n_groups: int = 4, sigma: float = 0.10) -> np.ndarray:
    """
    Spatially correlated moment noise — simulates accumulated SLAM drift.
    Lines are partitioned into n_groups random spatial buckets; each bucket
    receives the same random moment offset δ ~ N(0, σ²I).
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
        out[mask, :3] += np.random.randn(3).astype(np.float32) * sigma
    return out


def _make_confuser_lines(n: int, ref_lines: np.ndarray, spread: float = 0.08,
                         pos_range: float = 3.0) -> np.ndarray:
    """Outlier lines with directions near those in ref_lines — structurally confusing."""
    if n == 0 or len(ref_lines) == 0:
        return _make_outliers(n, pos_range=pos_range)
    # Sample directions close to random reference directions + small noise
    idx = np.random.randint(0, len(ref_lines), n)
    d = ref_lines[idx, 3:].copy()
    d += np.random.randn(n, 3).astype(np.float32) * spread
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    p = np.random.uniform(-pos_range, pos_range, (n, 3)).astype(np.float32)
    return np.concatenate([np.cross(p, d), d], axis=1).astype(np.float32)


def _make_pool(n: int, pool_type: str = 'mixed') -> np.ndarray:
    if n == 0:
        return np.zeros((0, 6), np.float32)

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

    # v2 fix: sample scale from LogNormal biased toward s > 1 (mono→metric scenario).
    # Clipped to _SCALE_RANGE; ~80% of samples have s > 1.
    log_s = np.random.normal(_SCALE_LOG_CENTER, _SCALE_LOG_STD)
    log_s = np.clip(log_s, np.log(_SCALE_RANGE[0]), np.log(_SCALE_RANGE[1]))
    s   = float(np.exp(log_s))
    R   = _random_rotation()
    # v2 fix: t_range_eff = 0.4 * s so that the inverse translation
    # |t_i| = |t|/s ≤ 0.4m for ALL s (both s < 1 and s > 1).
    # Prevents the t × d cross term from inflating query moments.
    t_range_eff = 0.4 * s
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

    inliers_ref = big_pool[:k].copy()
    out_q_ref   = big_pool[k:k + n_out1]
    out_r       = big_pool[k + n_out1:k + n_out1 + n_out2]

    # Direction noise: hard_noise uses up to 0.12 rad (~7°); others up to 0.04 rad.
    dir_noise_q   = np.random.uniform(0.0, 0.12 if use_confusers else 0.04)
    dir_noise_ref = np.random.uniform(0.0, 0.08 if use_confusers else 0.03)

    if k > 0:
        inliers_q = _apply_sim3(inliers_ref, s_i, R_i, t_i)
        inliers_q  [:, :3] += np.random.randn(k, 3).astype(np.float32) * noise1
        inliers_ref[:, :3] += np.random.randn(k, 3).astype(np.float32) * noise2
        inliers_q  [:, 3:] += np.random.randn(k, 3).astype(np.float32) * dir_noise_q
        inliers_q  [:, 3:] /= np.linalg.norm(inliers_q  [:, 3:], axis=1, keepdims=True) + 1e-9
        inliers_ref[:, 3:] += np.random.randn(k, 3).astype(np.float32) * dir_noise_ref
        inliers_ref[:, 3:] /= np.linalg.norm(inliers_ref[:, 3:], axis=1, keepdims=True) + 1e-9

        # Drift augmentation (20% of pairs): spatially correlated moment noise on query.
        if np.random.random() < 0.20:
            inliers_q = _apply_drift(
                inliers_q,
                n_groups=np.random.randint(2, 6),
                sigma=float(np.random.uniform(0.05, 0.25)),
            )

        # Rare direction flips (15% of pairs): negate m and d on 2–8% of inliers,
        # simulating ±180° orientation ambiguity in SLAM line extraction.
        if k > 2 and np.random.random() < 0.15:
            n_flip   = max(1, int(k * np.random.uniform(0.02, 0.08)))
            flip_idx = np.random.choice(k, n_flip, replace=False)
            inliers_q[flip_idx, :3] *= -1
            inliers_q[flip_idx, 3:] *= -1
    else:
        inliers_q = inliers_ref = np.zeros((0, 6), np.float32)

    if use_confusers and k > 0:
        # Replace a fraction of regular outliers with confuser lines (near-parallel to inliers).
        n_conf = max(0, int(n_out1 * np.random.uniform(0.3, 0.7)))
        n_reg  = n_out1 - n_conf
        out_q_reg  = (_apply_sim3(out_q_ref[:n_reg], s_i, R_i, t_i) if n_reg > 0
                      else np.zeros((0, 6), np.float32))
        out_q_conf = _make_confuser_lines(n_conf, inliers_q) if n_conf > 0 else np.zeros((0, 6), np.float32)
        out_q_parts = [p for p in [out_q_reg, out_q_conf] if len(p) > 0]
        out_q = np.concatenate(out_q_parts) if out_q_parts else np.zeros((0, 6), np.float32)
    else:
        out_q = (_apply_sim3(out_q_ref, s_i, R_i, t_i) if n_out1 > 0
                 else np.zeros((0, 6), np.float32))

    q_parts = [p for p in [inliers_q, out_q]   if len(p) > 0]
    r_parts = [p for p in [inliers_ref, out_r] if len(p) > 0]
    query_all = (np.concatenate(q_parts, axis=0) if q_parts else _make_pool(max(n1, 10), pool_type))
    ref_all   = (np.concatenate(r_parts, axis=0) if r_parts else _make_pool(max(n2, 10), pool_type))

    i1 = np.random.permutation(len(query_all))
    i2 = np.random.permutation(len(ref_all))
    query_all, ref_all = query_all[i1], ref_all[i2]
    m1 = np.argsort(i1)[:k].astype(np.int32)
    m2 = np.argsort(i2)[:k].astype(np.int32)

    return dict(
        plucker1 = query_all.astype(np.float32),
        plucker2 = ref_all.astype(np.float32),
        matches  = np.stack([m1, m2], 0) if k > 0 else np.zeros((2, 0), np.int32),
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
