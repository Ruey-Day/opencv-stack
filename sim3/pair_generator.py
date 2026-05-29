"""
pair_generator.py
=================
Core SLAM-map pair generation logic shared between:
  - scripts/generate_slam_map_dataset.py  (offline pkl generation)
  - sim3/dataloader.py                    (online LiveSim3PluckerData)
"""
import numpy as np
import msgpack

# ── Constants ─────────────────────────────────────────────────────────────────

N_P1_TOTAL = 200
N_P2_TOTAL = 200

SCALE_RANGE    = (0.1, 10.0)
SLAM_NOISE_MIN = 0.02
SLAM_NOISE_MAX = 0.30
SLAM_RATIO_MIN = 0.10
SLAM_RATIO_MAX = 0.60

OVERLAP_LEVELS = np.array([0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 1.00])
OVERLAP_PROBS  = np.array([0.10, 0.10, 0.12, 0.12, 0.12, 0.12, 0.12, 0.10, 0.10])

# Curriculum schedules: easy → mid → hard (= OVERLAP_PROBS)
# Easy: dense pairs dominate so the network gets strong gradient signal early
# Mid: balanced, introduces sparser pairs
# Hard: full distribution including lots of zero/near-zero overlap
_OVERLAP_PROBS_EASY = np.array([0.02, 0.02, 0.04, 0.07, 0.10, 0.20, 0.25, 0.20, 0.10])
_OVERLAP_PROBS_MID  = np.array([0.05, 0.06, 0.08, 0.10, 0.13, 0.16, 0.16, 0.14, 0.12])


def get_curriculum_probs(phase_frac: float) -> np.ndarray:
    """Return overlap probabilities for a given training phase fraction [0, 1].

    0.0–0.3  : easy → mid  (dense pairs first)
    0.3–1.0  : mid  → hard (gradually introduce sparse/zero-overlap)
    """
    if phase_frac < 0.3:
        t = phase_frac / 0.3
        probs = (1.0 - t) * _OVERLAP_PROBS_EASY + t * _OVERLAP_PROBS_MID
    else:
        t = (phase_frac - 0.3) / 0.7
        probs = (1.0 - t) * _OVERLAP_PROBS_MID + t * OVERLAP_PROBS
    probs = np.clip(probs, 0.0, None)
    return (probs / probs.sum()).astype(np.float64)


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


# ── Shared pair-building core ─────────────────────────────────────────────────

def _zero_overlap_pair() -> dict:
    log_s = np.random.uniform(np.log(SCALE_RANGE[0]), np.log(SCALE_RANGE[1]))
    return dict(
        plucker1=make_outliers(N_P1_TOTAL),
        plucker2=make_outliers(N_P2_TOTAL),
        matches=np.zeros((2, 0), dtype=np.int32),
        R_gt=np.eye(3, dtype=np.float32),
        t_gt=np.zeros((3, 1), dtype=np.float32),
        s_gt=np.float32(np.exp(log_s)),
    )


def _build_pair(rgbd_in: np.ndarray, n_rgbd: int,
                pool_filler_slam=None) -> dict:
    """
    Shared core: given an (n_rgbd, 6) RGBD subset, derive the SLAM side,
    apply random SIM(3), pad to fixed sizes, and return a pair dict.

    pool_filler_slam: optional (K, 6) pool to draw realistic outlier lines for
    the mono/SLAM side. If None, uses random make_outliers.
    """
    slam_ratio = np.random.uniform(SLAM_RATIO_MIN, SLAM_RATIO_MAX)
    n_slam_in  = max(3, int(round(n_rgbd * slam_ratio)))
    idx_slam   = np.random.choice(n_rgbd, n_slam_in, replace=False)

    slam_in_metric = rgbd_in[idx_slam].copy()
    noise_sigma = np.random.uniform(SLAM_NOISE_MIN, SLAM_NOISE_MAX)
    slam_in_metric[:, :3] += (np.random.randn(n_slam_in, 3).astype(np.float32)
                               * noise_sigma)

    log_s = np.random.uniform(np.log(SCALE_RANGE[0]), np.log(SCALE_RANGE[1]))
    s     = float(np.exp(log_s))
    R     = random_rotation()
    t     = np.random.uniform(-2.0, 2.0, 3).astype(np.float32)

    s_inv, R_inv = 1.0 / s, R.T
    t_inv = -R_inv @ t / s
    slam_in_slam = apply_sim3_plucker(slam_in_metric, s_inv, R_inv, t_inv)

    n_slam_in = min(n_slam_in, N_P1_TOTAL)
    n_rgbd_in = min(n_rgbd,    N_P2_TOTAL)
    n_out_slam = N_P1_TOTAL - n_slam_in
    n_out_rgbd = N_P2_TOTAL - n_rgbd_in

    if pool_filler_slam is not None and len(pool_filler_slam) >= n_out_slam > 0:
        idx_f = np.random.choice(len(pool_filler_slam), n_out_slam, replace=False)
        slam_fill = apply_sim3_plucker(pool_filler_slam[idx_f], s_inv, R_inv, t_inv)
    else:
        slam_fill = make_outliers(n_out_slam)

    slam_all = np.concatenate([slam_in_slam[:n_slam_in], slam_fill], 0)
    rgbd_all = np.concatenate([rgbd_in[:n_rgbd_in], make_outliers(n_out_rgbd)], 0)

    i1 = np.random.permutation(N_P1_TOTAL)
    i2 = np.random.permutation(N_P2_TOTAL)
    slam_all, rgbd_all = slam_all[i1], rgbd_all[i2]
    inv1, inv2 = np.argsort(i1), np.argsort(i2)

    m_slam = np.array([inv1[j]           for j in range(n_slam_in)], dtype=np.int32)
    m_rgbd = np.array([inv2[idx_slam[j]] for j in range(n_slam_in)], dtype=np.int32)

    return dict(
        plucker1=slam_all.astype(np.float32),
        plucker2=rgbd_all.astype(np.float32),
        matches=np.stack([m_slam, m_rgbd], axis=0),
        R_gt=R.astype(np.float32),
        t_gt=t.reshape(3, 1).astype(np.float32),
        s_gt=np.float32(s),
    )


# ── Public generators ─────────────────────────────────────────────────────────

def generate_pair(pool6: np.ndarray, overlap_probs=None) -> dict | None:
    """Intra-map pair: both sides derived from the same pool."""
    if len(pool6) < 6:
        return None
    probs = OVERLAP_PROBS if overlap_probs is None else overlap_probs
    overlap = float(np.random.choice(OVERLAP_LEVELS, p=probs))
    if overlap == 0.0:
        return _zero_overlap_pair()
    n_rgbd = max(4, min(int(round(overlap * len(pool6))), N_P2_TOTAL))
    if len(pool6) < n_rgbd:
        return None
    idx_rgbd = np.random.choice(len(pool6), n_rgbd, replace=False)
    return _build_pair(pool6[idx_rgbd].copy(), n_rgbd, pool_filler_slam=None)


def generate_inter_map_pair(pool_a: np.ndarray, pool_b: np.ndarray,
                             overlap_probs=None) -> dict | None:
    """Cross-map pair: inliers from pool_a, SLAM-side outlier filler from pool_b."""
    if len(pool_a) < 6 or len(pool_b) < 4:
        return None
    probs = OVERLAP_PROBS if overlap_probs is None else overlap_probs
    overlap = float(np.random.choice(OVERLAP_LEVELS, p=probs))
    if overlap == 0.0:
        return _zero_overlap_pair()
    n_rgbd = max(4, min(int(round(overlap * len(pool_a))), N_P2_TOTAL))
    if len(pool_a) < n_rgbd:
        return None
    idx_rgbd = np.random.choice(len(pool_a), n_rgbd, replace=False)
    return _build_pair(pool_a[idx_rgbd].copy(), n_rgbd, pool_filler_slam=pool_b)
