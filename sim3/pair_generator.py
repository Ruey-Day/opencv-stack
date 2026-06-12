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


# ── Outlier generator (kept as a building block) ─────────────────────────────

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


# ── Structured pool generators ────────────────────────────────────────────────
# All orientations are drawn uniformly from SO(3) — no axis-aligned or
# Manhattan-world bias — so the network cannot exploit any scene-type prior.

def make_plane_patch(n: int, pos_range: float = 3.0) -> np.ndarray:
    """N lines lying on a randomly oriented plane (any surface type)."""
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
    M = np.cross(P, D)
    return np.concatenate([M, D], axis=1).astype(np.float32)


def make_wireframe(n: int, scale: float = 2.0) -> np.ndarray:
    """Edges of a randomly oriented and scaled cuboid, subsampled/repeated to n."""
    if n == 0:
        return np.zeros((0, 6), np.float32)
    dims = np.random.uniform(0.3, 1.5, 3) * scale
    # 8 corners of axis-aligned box, then randomly rotated
    signs = np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],
                      [ 1,-1,-1],[ 1,-1,1],[ 1, 1,-1],[ 1, 1,1]], np.float64)
    corners = signs * dims[None]
    R = random_rotation().astype(np.float64)
    origin = np.random.uniform(-scale, scale, 3)
    corners = (R @ corners.T).T + origin
    # 12 edges: 4 parallel to each of the 3 local axes
    edge_pairs = [
        (0,1),(2,3),(4,5),(6,7),
        (0,2),(1,3),(4,6),(5,7),
        (0,4),(1,5),(2,6),(3,7),
    ]
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
        return make_outliers(n)
    arr = np.array(lines, np.float32)
    idx = np.random.choice(len(arr), n, replace=(n > len(arr)))
    return arr[idx]


def make_line_bundle(n: int, spread: float = 0.5, pos_range: float = 3.0) -> np.ndarray:
    """N lines passing near a common focus (corners, poles, radiating structures)."""
    if n == 0:
        return np.zeros((0, 6), np.float32)
    focus = np.random.uniform(-pos_range, pos_range, 3).astype(np.float32)
    D = np.random.randn(n, 3).astype(np.float32)
    D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-9
    offsets = np.random.randn(n, 3).astype(np.float32) * spread
    # Project offset perpendicular to direction so each line passes near focus
    proj = (offsets * D).sum(axis=1, keepdims=True) * D
    P = focus[None] + offsets - proj
    M = np.cross(P, D)
    return np.concatenate([M, D], axis=1).astype(np.float32)


def make_structured_pool(n: int) -> np.ndarray:
    """
    Diverse synthetic line pool from random geometric primitives.

    Each call independently draws random counts of plane patches, wireframe
    edges, and line bundles, then fills any remainder with random lines.
    All primitive orientations are uniform over SO(3) — no Manhattan-world
    or scene-type prior.  Suitable as a drop-in replacement for any context
    where real map pools are unavailable.
    """
    if n == 0:
        return np.zeros((0, 6), np.float32)

    n_planes  = np.random.randint(0, 5)   # 0–4 random planes
    n_boxes   = np.random.randint(0, 4)   # 0–3 wireframes
    n_bundles = np.random.randint(0, 3)   # 0–2 line bundles

    # Build a flat list of (maker_fn, instance_count) tasks
    tasks = ([(make_plane_patch,  1)] * n_planes +
             [(make_wireframe,    1)] * n_boxes  +
             [(make_line_bundle,  1)] * n_bundles)

    if not tasks:
        return make_outliers(n)

    # Distribute n lines across all primitive instances via Dirichlet split
    fracs = np.random.dirichlet(np.ones(len(tasks) + 1))   # +1 for random fill
    alloc = np.floor(fracs * n).astype(int)
    alloc[-1] += n - alloc.sum()   # give rounding remainder to random fill

    parts = []
    for (maker, _), k in zip(tasks, alloc[:-1]):
        if k > 0:
            parts.append(maker(k))
    if alloc[-1] > 0:
        parts.append(make_outliers(alloc[-1]))

    pool = np.concatenate(parts, axis=0)
    return pool[np.random.permutation(len(pool))].astype(np.float32)


def _random_outliers_from_pool(pool: np.ndarray, n: int) -> np.ndarray:
    """Draw n lines from pool; fall back to make_structured_pool if pool too small."""
    if pool is not None and len(pool) >= n > 0:
        idx = np.random.choice(len(pool), n, replace=False)
        return pool[idx]
    return make_structured_pool(n)


# ── Symmetric pair builder (cap-free) ─────────────────────────────────────────

def _build_pair(rgbd_in: np.ndarray,
                pool_filler_slam=None,
                pool_filler_rgbd=None,
                force_scale: float = None) -> dict:
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
    if force_scale is not None:
        s = float(force_scale)
    else:
        s = float(np.exp(np.random.uniform(np.log(SCALE_RANGE[0]), np.log(SCALE_RANGE[1]))))
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
    """Pair with no true correspondences — structured pools with no inliers."""
    log_s = np.random.uniform(np.log(SCALE_RANGE[0]), np.log(SCALE_RANGE[1]))
    n1 = np.random.randint(30, 101)
    n2 = np.random.randint(30, 101)
    return dict(
        plucker1 = make_structured_pool(n1),
        plucker2 = make_structured_pool(n2),
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

def generate_pair(pool6: np.ndarray, overlap_probs=None,
                  force_scale: float = None) -> dict | None:
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
                       pool_filler_rgbd=remaining if len(remaining) else None,
                       force_scale=force_scale)


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
    n = len(big_pool)
    if n < max(6, SUBMAP_N_MIN):
        return None

    # Ensure coverage_frac is large enough that n_overlap >= SUBMAP_N_MIN.
    # For small pools this may push coverage above COVERAGE_MAX, which is fine
    # since the submap still samples a random fraction of the overlap region.
    min_cov_needed = SUBMAP_N_MIN / n
    cov_lo = max(COVERAGE_MIN, min_cov_needed)
    cov_hi = max(COVERAGE_MAX, min_cov_needed * 1.2)
    coverage_frac = float(np.random.uniform(cov_lo, cov_hi))
    return _build_submap_pair(big_pool, coverage_frac, context_pool=context_pool)
