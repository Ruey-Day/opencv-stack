"""
Pair generators for Sim(3) line-matching training.

Two scenarios are supported:

symmetric (original)
    Both sides draw from the same pool.  plucker1 = monocular SLAM subset
    (arbitrary scale, noisy moments, SLAM_RATIO fraction of metric lines).
    plucker2 = metric map subset.  Sizes are variable — determined by the
    overlap fraction and a random outlier ratio, no hard cap.

submap  (new)
    plucker1 = small monocular submap (30–120 lines, arbitrary scale).
    plucker2 = big metric SLAM map (100–500 lines, metric scale).
    Only COVERAGE_FRAC of plucker2 overlaps with the submap; the rest are
    realistic context lines from the same or a different map, making this
    a low-overlap, highly asymmetric registration problem.

Variable sizes
    Neither scenario uses a fixed line count per side.  The DataLoader must
    use variable_collate (see dataloader.py) or batch_size=1.
"""

import numpy as np
import msgpack

# ── Overlap curriculum ────────────────────────────────────────────────────────

OVERLAP_LEVELS = np.array([0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 1.00])
OVERLAP_PROBS  = np.array([0.10, 0.10, 0.12, 0.12, 0.12, 0.12, 0.12, 0.10, 0.10])

_OVERLAP_PROBS_EASY = np.array([0.02, 0.02, 0.04, 0.07, 0.10, 0.20, 0.25, 0.20, 0.10])
_OVERLAP_PROBS_MID  = np.array([0.05, 0.06, 0.08, 0.10, 0.13, 0.16, 0.16, 0.14, 0.12])


def get_curriculum_probs(phase_frac: float) -> np.ndarray:
    """Interpolate overlap distribution from easy→hard as training progresses."""
    if phase_frac < 0.3:
        t = phase_frac / 0.3
        probs = (1.0 - t) * _OVERLAP_PROBS_EASY + t * _OVERLAP_PROBS_MID
    else:
        t = (phase_frac - 0.3) / 0.7
        probs = (1.0 - t) * _OVERLAP_PROBS_MID + t * OVERLAP_PROBS
    probs = np.clip(probs, 0.0, None)
    return (probs / probs.sum()).astype(np.float64)


# ── Physical constants ────────────────────────────────────────────────────────

SCALE_RANGE      = (0.1, 10.0)
SLAM_NOISE_MIN   = 0.02
SLAM_NOISE_MAX   = 0.30
SLAM_RATIO_MIN   = 0.10   # fraction of metric inliers observed by SLAM
SLAM_RATIO_MAX   = 0.60
OUTLIER_FRAC_MIN = 0.30   # fraction of each side that are outliers
OUTLIER_FRAC_MAX = 0.70
MAX_SIDE         = 600    # soft upper-bound per side (prevents OOM on huge pools)

# Submap scenario
SUBMAP_N_MIN     = 30    # smallest possible submap
SUBMAP_N_MAX     = 120   # largest possible submap (plucker1)
COVERAGE_MIN     = 0.05  # fraction of big-map pool covered by submap
COVERAGE_MAX     = 0.35
BIGMAP_REST_MAX  = 450   # max non-overlapping lines added to plucker2


# ── DB loader ─────────────────────────────────────────────────────────────────

def load_pool_from_db(db_path: str) -> np.ndarray:
    """Load line landmarks from a Structure-PLP-SLAM map as Plücker (K, 6)."""
    with open(db_path, "rb") as f:
        data = msgpack.unpack(f, raw=False)
    pool = []
    for lm in data.get("landmarks_line", {}).values():
        pw = lm.get("pos_w") or lm.get("pos")
        if pw is None or len(pw) < 6:
            continue
        p1 = np.array(pw[:3], np.float32)
        p2 = np.array(pw[3:6], np.float32)
        diff = p2 - p1
        ln = float(np.linalg.norm(diff))
        if ln < 0.01:
            continue
        d = diff / ln
        m = np.cross((p1 + p2) * 0.5, d)
        pool.append(np.concatenate([m, d]).astype(np.float32))
    return np.array(pool, np.float32) if pool else np.zeros((0, 6), np.float32)


# ── SIM(3) helpers ────────────────────────────────────────────────────────────

def random_rotation() -> np.ndarray:
    A = np.random.randn(3, 3).astype(np.float64)
    Q, R = np.linalg.qr(A)
    Q *= np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float32)


def apply_sim3_plucker(L6: np.ndarray, s: float, R: np.ndarray,
                       t: np.ndarray) -> np.ndarray:
    m, d  = L6[:, :3], L6[:, 3:]
    d_new = (R @ d.T).T
    m_new = s * (R @ m.T).T + np.cross(t[None], d_new)
    return np.concatenate([m_new, d_new], axis=1).astype(np.float32)


# ── Outlier generator ─────────────────────────────────────────────────────────

def make_outliers(n: int, n_clusters: int = 5, spread: float = 0.2,
                  pos_range: float = 3.0) -> np.ndarray:
    if n == 0:
        return np.zeros((0, 6), np.float32)
    n_per  = max(1, n // n_clusters)
    extras = n - n_per * n_clusters
    anchors = np.random.randn(n_clusters, 3).astype(np.float32)
    anchors /= np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-9
    parts = []
    for i, a in enumerate(anchors):
        cnt = n_per + (1 if i < extras else 0)
        d = a[None] + np.random.randn(cnt, 3).astype(np.float32) * spread
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        p = np.random.uniform(-pos_range, pos_range, (cnt, 3)).astype(np.float32)
        m = np.cross(p, d)
        parts.append(np.concatenate([m, d], axis=1).astype(np.float32))
    out = np.concatenate(parts, 0)
    return out[np.random.permutation(len(out))][:n]


def _random_outliers_from_pool(pool: np.ndarray, n: int) -> np.ndarray:
    """Draw n random lines from pool; fall back to make_outliers if pool too small."""
    if pool is not None and len(pool) >= n > 0:
        idx = np.random.choice(len(pool), n, replace=False)
        return pool[idx]
    return make_outliers(n)


# ── Symmetric pair builder (cap-free) ─────────────────────────────────────────

def _build_pair(rgbd_in: np.ndarray,
                pool_filler_slam=None,
                pool_filler_rgbd=None) -> dict:
    """
    Cap-free symmetric pair.

    Sizes are determined by the inlier count and a random outlier fraction,
    not a fixed global constant.  Both sides stay below MAX_SIDE.

    plucker1 — SLAM-like (arbitrary scale, noisy, SLAM_RATIO fraction of inliers)
    plucker2 — metric reference (all rgbd_in inliers + realistic outliers)
    """
    n_rgbd = len(rgbd_in)

    # SLAM side: pick a random subset of the metric inliers
    slam_ratio = np.random.uniform(SLAM_RATIO_MIN, SLAM_RATIO_MAX)
    n_slam_in  = max(3, min(int(round(n_rgbd * slam_ratio)), n_rgbd))
    idx_slam   = np.random.choice(n_rgbd, n_slam_in, replace=False)

    slam_in_metric = rgbd_in[idx_slam].copy()
    noise_sigma = np.random.uniform(SLAM_NOISE_MIN, SLAM_NOISE_MAX)
    slam_in_metric[:, :3] += np.random.randn(n_slam_in, 3).astype(np.float32) * noise_sigma

    # Random Sim(3) — maps metric frame → SLAM frame
    log_s = np.random.uniform(np.log(SCALE_RANGE[0]), np.log(SCALE_RANGE[1]))
    s = float(np.exp(log_s))
    R = random_rotation()
    t = np.random.uniform(-2.0, 2.0, 3).astype(np.float32)

    s_inv, R_inv = 1.0 / s, R.T
    t_inv = -R_inv @ t / s
    slam_in_slam = apply_sim3_plucker(slam_in_metric, s_inv, R_inv, t_inv)

    # Outlier ratio — random, same for both sides
    out_frac   = np.random.uniform(OUTLIER_FRAC_MIN, OUTLIER_FRAC_MAX)
    n_out_slam = min(int(round(n_slam_in * out_frac / (1.0 - out_frac))),
                     MAX_SIDE - n_slam_in)
    n_out_rgbd = min(int(round(n_rgbd * out_frac / (1.0 - out_frac))),
                     MAX_SIDE - n_rgbd)
    n_out_slam = max(0, n_out_slam)
    n_out_rgbd = max(0, n_out_rgbd)

    # Realistic outlier lines from the pool when available
    slam_fill = _random_outliers_from_pool(
        apply_sim3_plucker(pool_filler_slam, s_inv, R_inv, t_inv)
        if pool_filler_slam is not None and len(pool_filler_slam) > 0 else None,
        n_out_slam,
    )
    rgbd_fill = _random_outliers_from_pool(pool_filler_rgbd, n_out_rgbd)

    slam_all = (np.concatenate([slam_in_slam, slam_fill], 0)
                if len(slam_fill) else slam_in_slam)
    rgbd_all = (np.concatenate([rgbd_in, rgbd_fill], 0)
                if len(rgbd_fill) else rgbd_in)

    # Shuffle both sides; track where each inlier ended up
    i1 = np.random.permutation(len(slam_all))
    i2 = np.random.permutation(len(rgbd_all))
    slam_all = slam_all[i1];  rgbd_all = rgbd_all[i2]
    inv1 = np.argsort(i1);    inv2 = np.argsort(i2)

    m_slam = np.array([inv1[j]           for j in range(n_slam_in)], dtype=np.int32)
    m_rgbd = np.array([inv2[idx_slam[j]] for j in range(n_slam_in)], dtype=np.int32)

    return dict(
        plucker1 = slam_all.astype(np.float32),
        plucker2 = rgbd_all.astype(np.float32),
        matches  = np.stack([m_slam, m_rgbd], axis=0),
        R_gt     = R.astype(np.float32),
        t_gt     = t.reshape(3, 1).astype(np.float32),
        s_gt     = np.float32(s),
    )


def _zero_overlap_pair() -> dict:
    """Pair with no true correspondences — variable size random outlier lines."""
    log_s = np.random.uniform(np.log(SCALE_RANGE[0]), np.log(SCALE_RANGE[1]))
    n1 = np.random.randint(30, 101)
    n2 = np.random.randint(30, 101)
    return dict(
        plucker1 = make_outliers(n1),
        plucker2 = make_outliers(n2),
        matches  = np.zeros((2, 0), dtype=np.int32),
        R_gt     = np.eye(3, dtype=np.float32),
        t_gt     = np.zeros((3, 1), dtype=np.float32),
        s_gt     = np.float32(np.exp(log_s)),
    )


# ── Submap pair builder ───────────────────────────────────────────────────────

def _build_submap_pair(big_pool: np.ndarray,
                       coverage_frac: float,
                       context_pool=None) -> dict:
    """
    Asymmetric submap registration pair.

    plucker1 — small monocular submap (arbitrary scale, few lines, SUBMAP_N_MIN–MAX)
    plucker2 — big metric SLAM map  (metric scale, many lines)

    Only coverage_frac of the big map spatially overlaps with the submap.
    The rest of plucker2 contains realistic non-overlapping map lines (context),
    making it a hard low-overlap problem.

    Sim(3) T maps big-map frame → submap frame (i.e. plucker1 = T · plucker2).
    """
    n_pool    = len(big_pool)
    n_overlap = max(6, min(int(round(coverage_frac * n_pool)), n_pool))

    # Lines from the pool that fall inside the submap's coverage area
    idx_overlap  = np.random.choice(n_pool, n_overlap, replace=False)
    overlap_lines = big_pool[idx_overlap]   # (n_overlap, 6) metric scale

    # Submap observes a random subset of the overlap region
    hi = min(SUBMAP_N_MAX, n_overlap)
    lo = min(SUBMAP_N_MIN, hi)
    n_submap = max(3, int(np.random.randint(lo, hi + 1)))
    idx_submap    = np.random.choice(n_overlap, n_submap, replace=False)
    submap_metric = overlap_lines[idx_submap].copy()

    # SLAM-like noise on moments
    noise_sigma = np.random.uniform(SLAM_NOISE_MIN, SLAM_NOISE_MAX)
    submap_metric[:, :3] += np.random.randn(n_submap, 3).astype(np.float32) * noise_sigma

    # Sim(3): big-map frame → submap frame
    log_s = np.random.uniform(np.log(SCALE_RANGE[0]), np.log(SCALE_RANGE[1]))
    s = float(np.exp(log_s))
    R = random_rotation()
    t = np.random.uniform(-2.0, 2.0, 3).astype(np.float32)

    s_inv, R_inv = 1.0 / s, R.T
    t_inv = -R_inv @ t / s
    submap_lines = apply_sim3_plucker(submap_metric, s_inv, R_inv, t_inv)

    # Big-map side: overlap lines + non-overlapping context from the same pool
    non_overlap_mask = np.ones(n_pool, dtype=bool)
    non_overlap_mask[idx_overlap] = False
    non_overlap_idx = np.where(non_overlap_mask)[0]

    n_rest = min(len(non_overlap_idx), BIGMAP_REST_MAX)
    if n_rest > 0:
        rest_idx   = np.random.choice(non_overlap_idx, n_rest, replace=False)
        rest_lines = big_pool[rest_idx]
    else:
        rest_lines = np.zeros((0, 6), np.float32)

    # Optionally mix in lines from a second (cross-map) pool
    if context_pool is not None and len(context_pool) > 0:
        n_cross = min(len(context_pool), max(0, BIGMAP_REST_MAX - n_rest))
        if n_cross > 0:
            cross_idx  = np.random.choice(len(context_pool), n_cross, replace=False)
            rest_lines = (np.concatenate([rest_lines, context_pool[cross_idx]], 0)
                          if len(rest_lines) else context_pool[cross_idx])

    bigmap_lines = (np.concatenate([overlap_lines, rest_lines], 0)
                    if len(rest_lines) else overlap_lines)

    # Shuffle both sides
    i1 = np.random.permutation(len(submap_lines))
    i2 = np.random.permutation(len(bigmap_lines))
    submap_lines = submap_lines[i1]
    bigmap_lines = bigmap_lines[i2]
    inv1 = np.argsort(i1);  inv2 = np.argsort(i2)

    # Matches: submap line k ↔ overlap_lines[idx_submap[k]]
    # Before shuffle, overlap_lines occupied rows 0..n_overlap-1 in bigmap_lines.
    m_sub = np.array([inv1[k]             for k in range(n_submap)], dtype=np.int32)
    m_big = np.array([inv2[idx_submap[k]] for k in range(n_submap)], dtype=np.int32)

    return dict(
        plucker1 = submap_lines.astype(np.float32),   # small submap (arb. scale)
        plucker2 = bigmap_lines.astype(np.float32),   # big metric map
        matches  = np.stack([m_sub, m_big], axis=0),
        R_gt     = R.astype(np.float32),
        t_gt     = t.reshape(3, 1).astype(np.float32),
        s_gt     = np.float32(s),
    )


# ── Public generators ─────────────────────────────────────────────────────────

def generate_pair(pool6: np.ndarray, overlap_probs=None) -> dict | None:
    """Intra-map symmetric pair (cap-free)."""
    if len(pool6) < 6:
        return None
    probs   = OVERLAP_PROBS if overlap_probs is None else overlap_probs
    overlap = float(np.random.choice(OVERLAP_LEVELS, p=probs))
    if overlap == 0.0:
        return _zero_overlap_pair()
    n_rgbd = max(4, min(int(round(overlap * len(pool6))), MAX_SIDE))
    if len(pool6) < n_rgbd:
        return None
    idx_rgbd  = np.random.choice(len(pool6), n_rgbd, replace=False)
    remaining = np.delete(pool6, idx_rgbd, axis=0)
    return _build_pair(pool6[idx_rgbd].copy(),
                       pool_filler_rgbd=remaining if len(remaining) else None)


def generate_inter_map_pair(pool_a: np.ndarray, pool_b: np.ndarray,
                             overlap_probs=None) -> dict | None:
    """Cross-map symmetric pair: inliers from pool_a, SLAM context from pool_b."""
    if len(pool_a) < 6 or len(pool_b) < 4:
        return None
    probs   = OVERLAP_PROBS if overlap_probs is None else overlap_probs
    overlap = float(np.random.choice(OVERLAP_LEVELS, p=probs))
    if overlap == 0.0:
        return _zero_overlap_pair()
    n_rgbd = max(4, min(int(round(overlap * len(pool_a))), MAX_SIDE))
    if len(pool_a) < n_rgbd:
        return None
    idx_rgbd  = np.random.choice(len(pool_a), n_rgbd, replace=False)
    remaining = np.delete(pool_a, idx_rgbd, axis=0)
    return _build_pair(pool_a[idx_rgbd].copy(),
                       pool_filler_slam=pool_b,
                       pool_filler_rgbd=remaining if len(remaining) else None)


def generate_submap_pair(big_pool: np.ndarray,
                          context_pool=None) -> dict | None:
    """
    Submap registration pair (asymmetric, cap-free).

    plucker1 (submap)  — 30–120 lines, arbitrary scale
    plucker2 (big map) — 100–500 lines, metric scale
    """
    if len(big_pool) < max(6, SUBMAP_N_MIN):
        return None

    coverage_frac = float(np.random.uniform(COVERAGE_MIN, COVERAGE_MAX))
    return _build_submap_pair(big_pool, coverage_frac, context_pool=context_pool)
