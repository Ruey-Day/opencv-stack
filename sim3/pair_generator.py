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
ROOM_SCALE_RANGE = (0.1,  5.0)   # tighter range for indoor room scenario
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


def make_parallel_group(n: int, spread: float = 0.05,
                        pos_range: float = 3.0) -> np.ndarray:
    """N nearly-parallel lines — simulates corridors, pipes, rails, power lines."""
    if n == 0:
        return np.zeros((0, 6), np.float32)
    anchor = np.random.randn(3).astype(np.float32)
    anchor /= np.linalg.norm(anchor) + 1e-9
    D = anchor[None] + np.random.randn(n, 3).astype(np.float32) * spread
    D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-9
    P = np.random.uniform(-pos_range, pos_range, (n, 3)).astype(np.float32)
    M = np.cross(P, D)
    return np.concatenate([M, D], axis=1).astype(np.float32)


def make_grid_patch(n: int, pos_range: float = 3.0) -> np.ndarray:
    """Lines in two perpendicular in-plane directions — fences, grids, lattices."""
    if n == 0:
        return np.zeros((0, 6), np.float32)
    normal = np.random.randn(3).astype(np.float64)
    normal /= np.linalg.norm(normal) + 1e-9
    perp = np.random.randn(3).astype(np.float64)
    perp -= perp.dot(normal) * normal
    u = (perp / (np.linalg.norm(perp) + 1e-9)).astype(np.float32)
    v = np.cross(normal, u).astype(np.float32)

    nh = n // 2
    nv = n - nh

    # Horizontal lines: direction u, positions along v axis
    Dh = np.tile(u, (nh, 1))
    Ph = (np.random.uniform(-pos_range, pos_range, (nh, 1)) * v[None]
          + np.random.uniform(-pos_range, pos_range, (nh, 1)) * u[None]).astype(np.float32)
    Mh = np.cross(Ph, Dh)

    # Vertical lines: direction v, positions along u axis
    Dv = np.tile(v, (nv, 1))
    Pv = (np.random.uniform(-pos_range, pos_range, (nv, 1)) * u[None]
          + np.random.uniform(-pos_range, pos_range, (nv, 1)) * v[None]).astype(np.float32)
    Mv = np.cross(Pv, Dv)

    return np.concatenate([
        np.concatenate([Mh, Dh], axis=1),
        np.concatenate([Mv, Dv], axis=1),
    ], axis=0).astype(np.float32)


def make_staircase(n: int) -> np.ndarray:
    """Stair nosings + risers: coplanar parallel lines at regular offsets.

    Covers staircases, ladders, shelves, bleachers — any scene with lines
    that are simultaneously parallel AND coplanar (a mode absent from both
    make_parallel_group and make_plane_patch individually).
    """
    if n == 0:
        return np.zeros((0, 6), np.float32)
    R = random_rotation().astype(np.float64)
    step_dir  = R[:, 0]   # direction along each nosing edge
    up_dir    = R[:, 1]   # vertical (step rise)
    depth_dir = R[:, 2]   # stair climbing direction

    step_h  = np.random.uniform(0.15, 0.30)   # rise ~20 cm
    step_d  = np.random.uniform(0.25, 0.40)   # run  ~30 cm
    origin  = np.random.uniform(-2.0, 2.0, 3)

    n_steps = max(3, n // 2)
    lines = []
    for i in range(n_steps):
        o = (origin + i * (step_h * up_dir + step_d * depth_dir)).astype(np.float32)
        # nosing edge — horizontal, parallel to step_dir
        d = step_dir.astype(np.float32)
        lines.append(np.concatenate([np.cross(o, d), d]))
        # riser edge — vertical face
        d2 = up_dir.astype(np.float32)
        lines.append(np.concatenate([np.cross(o, d2), d2]))

    arr = np.array(lines, np.float32)
    idx = np.random.choice(len(arr), n, replace=(n > len(arr)))
    return arr[idx]


def make_room_pool(n: int) -> np.ndarray:
    """
    Manhattan-world indoor room scene calibrated to 7-Scenes / real indoor SLAM.

    Scene half-extents match typical office/kitchen rooms (2–5 m).  Moments
    are bounded by the room radius → |m| ≤ ~4, matching real SLAM map statistics.

    Direction distribution:
      75 % near-axis-aligned (3 main axes, ±1.7° Gaussian spread)
            → walls, floor, ceiling, furniture edges
      25 % fully random  → oblique architectural features

    A random full-SO(3) rotation is applied to the whole scene so the network
    cannot learn a fixed-axis prior — it must recognise Manhattan structure as
    a geometric pattern, not a fixed orientation.
    """
    if n == 0:
        return np.zeros((0, 6), np.float32)

    # Room half-extents: width/depth 1–2.5 m, height 1–1.5 m (full room 2–5 m)
    hw = np.random.uniform(1.0, 2.5)
    hd = np.random.uniform(1.0, 2.5)
    hh = np.random.uniform(1.0, 1.5)

    # --- Midpoints ----------------------------------------------------------
    # 60 % biased toward the 6 room surfaces (walls, floor, ceiling)
    n_surf = int(0.60 * n)
    n_int  = n - n_surf

    # Pick a random face for each surface point, then sample on that face
    face_half = np.array([[hw,0,0],[-hw,0,0],[0,hh,0],[0,-hh,0],[0,0,hd],[0,0,-hd]],
                          np.float32)
    fi = np.random.randint(6, size=n_surf)
    fn = face_half[fi]                                     # normal direction
    axis = np.argmax(np.abs(fn), axis=1)                   # which axis is fixed

    P_surf = np.random.uniform(-1.0, 1.0, (n_surf, 3)).astype(np.float32)
    P_surf *= np.array([hw, hh, hd], np.float32)
    # Pin the normal axis to the face and add a tiny inward offset
    P_surf[np.arange(n_surf), axis] = fn[np.arange(n_surf), axis]
    P_surf += fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-9) \
              * np.random.uniform(-0.05, 0.0, (n_surf, 1)).astype(np.float32)

    P_int = np.column_stack([
        np.random.uniform(-hw, hw, n_int),
        np.random.uniform(-hh, hh, n_int),
        np.random.uniform(-hd, hd, n_int),
    ]).astype(np.float32)

    P = np.concatenate([P_surf, P_int], axis=0) if n_int > 0 else P_surf

    # --- Directions ---------------------------------------------------------
    n_ax  = int(0.75 * n)
    n_rnd = n - n_ax

    ax_idx = np.random.randint(3, size=n_ax)
    D_ax   = np.zeros((n_ax, 3), np.float32)
    D_ax[np.arange(n_ax), ax_idx] = 1.0
    D_ax  += np.random.randn(n_ax, 3).astype(np.float32) * 0.03   # ~1.7° noise
    D_ax  /= np.linalg.norm(D_ax, axis=1, keepdims=True) + 1e-9

    D_rnd  = (np.random.randn(n_rnd, 3).astype(np.float32)
              if n_rnd > 0 else np.zeros((0, 3), np.float32))
    if n_rnd > 0:
        D_rnd /= np.linalg.norm(D_rnd, axis=1, keepdims=True) + 1e-9

    D = np.concatenate([D_ax, D_rnd], axis=0)

    # Shuffle so axis-aligned / random lines are mixed throughout the cloud
    perm = np.random.permutation(n)
    P, D = P[perm], D[perm]

    M = np.cross(P, D)
    pool = np.concatenate([M, D], axis=1).astype(np.float32)

    # Random full-SO(3) rotation: removes the fixed-axis prior
    R = random_rotation()
    pool = apply_sim3_plucker(pool, 1.0, R, np.zeros(3, np.float32))
    return pool


def make_structured_pool(n: int) -> np.ndarray:
    """
    Diverse synthetic line pool from random geometric primitives.

    Randomly combines plane patches, wireframe edges, line bundles, parallel
    groups, grid patches, staircase structures, and unstructured lines.
    All orientations are drawn uniformly from SO(3) — no Manhattan-world or
    axis-aligned bias.
    """
    if n == 0:
        return np.zeros((0, 6), np.float32)

    n_planes     = np.random.randint(0, 5)   # 0–4 planes
    n_boxes      = np.random.randint(0, 4)   # 0–3 wireframe boxes
    n_bundles    = np.random.randint(0, 3)   # 0–2 line bundles (corners/poles)
    n_parallels  = np.random.randint(0, 3)   # 0–2 parallel groups (corridors/rails)
    n_grids      = np.random.randint(0, 3)   # 0–2 grid patches (fences/lattices)
    n_staircases = np.random.randint(0, 3)   # 0–2 staircase structures

    tasks = ([(make_plane_patch,    1)] * n_planes     +
             [(make_wireframe,      1)] * n_boxes      +
             [(make_line_bundle,    1)] * n_bundles    +
             [(make_parallel_group, 1)] * n_parallels  +
             [(make_grid_patch,     1)] * n_grids      +
             [(make_staircase,      1)] * n_staircases)

    if not tasks:
        return make_outliers(n)

    # Distribute n lines across all primitive instances + a random-fill slot
    fracs = np.random.dirichlet(np.ones(len(tasks) + 1))
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
    # Direction noise: short segments have high angular uncertainty in monocular SLAM
    dir_noise = np.random.uniform(0.0, 0.05)
    slam_in_metric[:, 3:] += np.random.randn(n_slam_in, 3).astype(np.float32) * dir_noise
    slam_in_metric[:, 3:] /= np.linalg.norm(slam_in_metric[:, 3:], axis=1, keepdims=True) + 1e-9

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

    # SLAM-like noise on moments and directions
    noise_sigma = np.random.uniform(SLAM_NOISE_MIN, SLAM_NOISE_MAX)
    submap_metric[:, :3] += np.random.randn(n_submap, 3).astype(np.float32) * noise_sigma
    dir_noise = np.random.uniform(0.0, 0.05)
    submap_metric[:, 3:] += np.random.randn(n_submap, 3).astype(np.float32) * dir_noise
    submap_metric[:, 3:] /= np.linalg.norm(submap_metric[:, 3:], axis=1, keepdims=True) + 1e-9

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


# ── Fully-synthetic diverse pair generator ────────────────────────────────────

# Scenario names and sampling probabilities
_SCENARIOS = ['room', 'submap', 'relocalize', 'loop', 'dense_sparse', 'zero_overlap']
_SCENARIO_P = np.array([0.30, 0.22, 0.18, 0.14, 0.10, 0.06])


def generate_diverse_pair() -> dict:
    """
    Generate one fully-synthetic training pair from a randomly chosen scenario.

    No map files are required — all line pools are produced by make_structured_pool,
    which combines random geometric primitives (plane patches, wireframes, line
    bundles, parallel groups, grid patches) with no axis-aligned or scene-type
    prior.

    Scenarios and their real-world analogues
    ----------------------------------------
    submap      (30%) — small/noisy/sparse monocular query submap vs large/clean
                        metric reference map.  Low overlap (beta(1.5,5)), high
                        moment noise on the query side, 10–150 vs 80–700 lines.
    relocalize  (25%) — cross-session re-localization: two maps of similar size
                        from different SLAM runs.  Moderate overlap, similar noise.
    loop        (20%) — loop-closure: both maps from the same session, high
                        overlap (beta(5,2)), low noise.
    dense_sparse(15%) — one map is very dense (RGB-D / LiDAR) the other sparse
                        (monocular).  Any overlap, large density ratio.
    zero_overlap(10%) — completely disjoint views — no inliers (hard negative).

    Scale: log-uniform in [0.1, 10] for all scenarios (Sim(3) dataset).
    Line counts: variable per pair — NOT padded to any fixed size.

    Returns
    -------
    dict with keys: plucker1 (n1,6), plucker2 (n2,6), matches (2,k),
                    R_gt (3,3), t_gt (3,1), s_gt scalar.
    plucker1 = query / SLAM side   (arbitrary scale, noisy)
    plucker2 = reference / metric side  (metric scale, cleaner)
    """
    scenario = np.random.choice(_SCENARIOS, p=_SCENARIO_P)

    # Pool function and SIM(3) ranges: room scenario uses physically-bounded
    # pools and a tighter scale range to match real indoor SLAM statistics.
    if scenario == 'room':
        pool_fn     = make_room_pool
        s_lo, s_hi  = np.log(ROOM_SCALE_RANGE[0]), np.log(ROOM_SCALE_RANGE[1])
        t_range     = 1.5   # smaller translation (room-sized offsets)
    else:
        pool_fn     = make_structured_pool
        s_lo, s_hi  = np.log(SCALE_RANGE[0]), np.log(SCALE_RANGE[1])
        t_range     = 2.0

    # ── Zero-overlap: two unrelated structured pools ──────────────────────────
    if scenario == 'zero_overlap':
        n1 = np.random.randint(10, 400)
        n2 = np.random.randint(10, 600)
        s  = float(np.exp(np.random.uniform(s_lo, s_hi)))
        return dict(
            plucker1 = pool_fn(n1),
            plucker2 = pool_fn(n2),
            matches  = np.zeros((2, 0), np.int32),
            R_gt     = np.eye(3, dtype=np.float32),
            t_gt     = np.zeros((3, 1), dtype=np.float32),
            s_gt     = np.float32(s),
        )

    # ── Scenario-specific size / overlap / noise parameters ──────────────────
    if scenario == 'room':
        # Line counts match real 7-Scenes SLAM maps (100–1000 lines per side)
        n2           = np.random.randint(80, 1100)           # metric reference
        ratio        = np.random.uniform(0.3, 3.0)
        n1           = max(30, min(int(n2 * ratio), 1100))   # mono query
        overlap_frac = float(np.random.beta(3.0, 2.0))       # moderate-to-high overlap
        noise1       = float(np.exp(np.random.uniform(np.log(0.005), np.log(0.15))))
        noise2       = float(np.exp(np.random.uniform(np.log(0.001), np.log(0.05))))

    elif scenario == 'submap':
        n1           = np.random.randint(10, 150)
        n2           = np.random.randint(max(n1, 80), 700)
        overlap_frac = float(np.random.beta(1.5, 5.0))       # bias: low overlap
        noise1       = float(np.exp(np.random.uniform(np.log(0.020), np.log(0.50))))
        noise2       = float(np.exp(np.random.uniform(np.log(0.001), np.log(0.05))))

    elif scenario == 'relocalize':
        base         = int(np.exp(np.random.uniform(np.log(30), np.log(400))))
        r            = np.random.uniform(0.5, 2.0)
        n1           = max(10, int(base * r))
        n2           = max(10, int(base / r))
        overlap_frac = float(np.random.beta(2.5, 2.5))       # moderate overlap
        noise_shared = float(np.exp(np.random.uniform(np.log(0.005), np.log(0.20))))
        noise1       = noise2 = noise_shared

    elif scenario == 'loop':
        n_both       = np.random.randint(30, 500)
        n1           = n2 = n_both
        overlap_frac = float(np.random.beta(5.0, 2.0))       # bias: high overlap
        noise_shared = float(np.exp(np.random.uniform(np.log(0.005), np.log(0.15))))
        noise1       = noise2 = noise_shared

    else:  # dense_sparse
        n2           = np.random.randint(150, 700)            # dense reference
        density_r    = np.random.uniform(0.05, 0.40)
        n1           = max(5, int(n2 * density_r))            # sparse query
        overlap_frac = float(np.random.beta(2.0, 2.0))
        noise1       = float(np.exp(np.random.uniform(np.log(0.010), np.log(0.30))))
        noise2       = float(np.exp(np.random.uniform(np.log(0.001), np.log(0.03))))

    # ── Shared inlier count ───────────────────────────────────────────────────
    k  = max(0, int(overlap_frac * min(n1, n2)))
    n1 = max(n1, k)
    n2 = max(n2, k)

    # ── Random Sim(3): query frame → reference frame ──────────────────────────
    # p_ref = s * R * p_query + t
    s   = float(np.exp(np.random.uniform(s_lo, s_hi)))
    R   = random_rotation()
    t   = np.random.uniform(-t_range, t_range, 3).astype(np.float32)
    # Inverse Sim(3): reference → query
    s_i = 1.0 / s
    R_i = R.T
    t_i = -(R_i @ t) * s_i

    # ── Inlier lines ──────────────────────────────────────────────────────────
    if k > 0:
        # Generate inlier pool in the reference frame, then transform to query
        inliers_ref = pool_fn(k + max(k // 4, 8))[:k].copy()
        inliers_q   = apply_sim3_plucker(inliers_ref, s_i, R_i, t_i)
        # Independent moment noise on each side
        inliers_q  [:, :3] += np.random.randn(k, 3).astype(np.float32) * noise1
        inliers_ref[:, :3] += np.random.randn(k, 3).astype(np.float32) * noise2
        # Direction noise for both sides
        dir_noise1 = np.random.uniform(0.0, 0.05)
        dir_noise2 = np.random.uniform(0.0, 0.02)
        inliers_q  [:, 3:] += np.random.randn(k, 3).astype(np.float32) * dir_noise1
        inliers_q  [:, 3:] /= np.linalg.norm(inliers_q  [:, 3:], axis=1, keepdims=True) + 1e-9
        inliers_ref[:, 3:] += np.random.randn(k, 3).astype(np.float32) * dir_noise2
        inliers_ref[:, 3:] /= np.linalg.norm(inliers_ref[:, 3:], axis=1, keepdims=True) + 1e-9
    else:
        inliers_q   = np.zeros((0, 6), np.float32)
        inliers_ref = np.zeros((0, 6), np.float32)

    # ── Outlier lines (no correspondences) ───────────────────────────────────
    n_out1 = max(0, n1 - k)
    n_out2 = max(0, n2 - k)

    # Query outliers: generated in the reference frame, then transformed to
    # query frame so moment magnitudes are consistent with the inliers.
    out_q = (apply_sim3_plucker(pool_fn(n_out1), s_i, R_i, t_i)
             if n_out1 > 0 else np.zeros((0, 6), np.float32))
    out_r = (pool_fn(n_out2)
             if n_out2 > 0 else np.zeros((0, 6), np.float32))

    # ── Assemble and shuffle ──────────────────────────────────────────────────
    q_parts = [p for p in [inliers_q, out_q]   if len(p) > 0]
    r_parts = [p for p in [inliers_ref, out_r] if len(p) > 0]

    query_all = (np.concatenate(q_parts, axis=0)
                 if q_parts else make_structured_pool(max(n1, 10)))
    ref_all   = (np.concatenate(r_parts, axis=0)
                 if r_parts else make_structured_pool(max(n2, 10)))

    i1 = np.random.permutation(len(query_all))
    i2 = np.random.permutation(len(ref_all))
    query_all = query_all[i1]
    ref_all   = ref_all[i2]

    # Inliers occupied rows 0..k-1 before the shuffle
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
