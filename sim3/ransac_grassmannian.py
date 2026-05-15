"""
Grassmannian SIM(3) RANSAC for metric scale recovery from 3D line maps.

Theoretical contribution
------------------------
A 3D line is characterised by its Plücker coordinates L = [m; d] ∈ R^6,
where d ∈ S^2 is the unit direction and m = p × d is the moment vector.
Normalised to unit norm, these embed lines as points on the real projective
Grassmannian  G(1,5) ≅ G(2,4).  The geodesic (principal-angle) distance is:

    θ(L1, L2) = arccos( |L1_norm · L2_norm| )

We use this as the RANSAC inlier metric for SIM(3) estimation, giving a
theoretically-grounded, rotation/scale/translation-equivariant outlier test
that is more principled than ad-hoc endpoint or direction-only thresholds.

SIM(3) line transformation
--------------------------
Under SIM(3) T = (R, t, s):
    point:  p  →  s·R·p + t
    direction: d  →  R·d          (scale-free)
    moment:    m  →  s·R·m + t × (R·d)

Minimal solver (3 line pairs → 7 DOF = 3 rotation + 3 translation + 1 scale):
    1. Rotation R  — direction Procrustes / Wahba problem via SVD.
    2. Scale s and translation t — linear least squares given R:
           m2_i = s·R·m1_i + skew(t)·(R·d1_i)
       stacked as  [skew(R·d1_i) | R·m1_i] · [t; s] = m2_i.
"""

import numpy as np


# ── Plücker utilities ─────────────────────────────────────────────────────────

def plucker_from_endpoints(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Convert 3D line endpoints to Plücker coordinates [m; d].

    Args:
        p1: (N, 3) or (3,) start points
        p2: (N, 3) or (3,) end points

    Returns:
        L: (6, N)  rows 0-2 = moment m = p1×d,  rows 3-5 = unit direction d
    """
    p1 = np.atleast_2d(p1).astype(float)   # (N, 3)
    p2 = np.atleast_2d(p2).astype(float)
    diff = p2 - p1
    norms = np.linalg.norm(diff, axis=1, keepdims=True)
    d = diff / (norms + 1e-12)              # (N, 3) unit directions
    m = np.cross(p1, d)                     # (N, 3) moment vectors
    return np.vstack([m.T, d.T])            # (6, N)


def normalize_plucker(L: np.ndarray) -> np.ndarray:
    """Normalize each Plücker vector to unit norm.  L: (6, N) → (6, N)."""
    norms = np.linalg.norm(L, axis=0, keepdims=True)
    return L / (norms + 1e-12)


def grassmannian_distance(L1: np.ndarray, L2: np.ndarray) -> np.ndarray:
    """
    Principal angle between Plücker lines on the Grassmannian G(1,5).

    Both L1 and L2 must already be unit-normalised.

    Returns:
        (N,) principal angles in radians ∈ [0, π/2]
    """
    dots = np.sum(L1 * L2, axis=0)                   # (N,)
    cos_theta = np.clip(np.abs(dots), 0.0, 1.0)
    return np.arccos(cos_theta)                        # (N,)


# ── SIM(3) solvers ────────────────────────────────────────────────────────────

def _skew(v: np.ndarray) -> np.ndarray:
    """3×3 skew-symmetric matrix for vector v (3,)."""
    return np.array([
        [ 0.0,  -v[2],  v[1]],
        [ v[2],   0.0, -v[0]],
        [-v[1],  v[0],  0.0],
    ])


def solve_rotation(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """
    Kabsch–Procrustes: find R ∈ SO(3) minimising Σ‖R·d1_i − d2_i‖².

    Handles line direction sign ambiguity: each direction pair (d1_i, d2_i)
    can represent the same undirected line, so we flip d1_i when d1_i·d2_i < 0
    before computing the cross-covariance matrix.

    Args:
        d1: (3, N)  unit direction vectors  (source)
        d2: (3, N)  unit direction vectors  (target)

    Returns:
        R: (3, 3) rotation matrix
    """
    # Align direction signs: flip source direction if it points away from target
    dots = np.sum(d1 * d2, axis=0)          # (N,)
    signs = np.where(dots >= 0, 1.0, -1.0)
    d1_aligned = d1 * signs[np.newaxis, :]  # sign-aligned source directions

    M = d2 @ d1_aligned.T                   # (3, 3) cross-covariance
    U, _, Vt = np.linalg.svd(M)
    det = np.linalg.det(U @ Vt)
    D = np.diag([1.0, 1.0, float(det)])     # ensure det(R) = +1
    return U @ D @ Vt


def solve_translation_scale(
    L1: np.ndarray, L2: np.ndarray, R: np.ndarray,
    s_prior: float = -1.0, lambda_s: float = 5.0,
) -> tuple:
    """
    Linear solve for translation t and scale s given rotation R.

    SIM(3) moment constraint:
        m2 = s·R·m1 + skew(t)·(R·d1)

    Rearranged as a linear system in x = [t (3); s (1)]:
        [ -skew(R·d1_i) | R·m1_i ] · x = m2_i    for each line i.

    When the scene has many near-parallel lines (e.g. chessboard), the system
    is rank-deficient for translation.  An optional scale prior s_prior with
    Tikhonov weight lambda_s regularises the scale component, preventing the
    minimum-norm lstsq from collapsing s to near zero.

    Args:
        L1: (6, N)  Plücker [m1; d1]  (source — SLAM, arb. scale)
        L2: (6, N)  Plücker [m2; d2]  (target — DA3, metric)
        R:  (3, 3)
        s_prior:    if > 0, add a soft constraint s ≈ s_prior
        lambda_s:   weight of the scale-prior term

    Returns:
        t: (3,) translation,   s: float scale  (> 0 means SLAM is smaller)
    """
    m1, d1 = L1[:3], L1[3:]   # (3, N)
    m2     = L2[:3]            # (3, N)
    N = L1.shape[1]

    Rd1 = R @ d1               # (3, N)
    Rm1 = R @ m1               # (3, N)

    A = np.zeros((3 * N, 4))
    b = np.zeros(3 * N)
    for i in range(N):
        row = 3 * i
        # t × (R·d1) = -(R·d1) × t = -skew(R·d1) · t
        A[row:row + 3, :3] = -_skew(Rd1[:, i])  # coefficient of t
        A[row:row + 3,  3] =  Rm1[:, i]          # coefficient of s  (R·m1_i)
        b[row:row + 3]     =  m2[:, i]

    if s_prior > 0:
        # Tikhonov: add virtual row [0, 0, 0, lambda_s] · x = lambda_s · s_prior
        A = np.vstack([A, [0.0, 0.0, 0.0, lambda_s]])
        b = np.append(b, lambda_s * s_prior)

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    t, s = x[:3], float(x[3])

    # Resolve sign ambiguity: scale must be positive
    if s < 0:
        s = -s
        t = -t

    return t, s


# ── SIM(3) line transform ─────────────────────────────────────────────────────

def transform_lines(
    L: np.ndarray, R: np.ndarray, t: np.ndarray, s: float
) -> np.ndarray:
    """
    Apply SIM(3) = (R, t, s) to Plücker line coordinates.

    d' = R·d
    m' = s·R·m + skew(t)·(R·d)

    Args:
        L: (6, N)  [m; d]
        R: (3, 3),  t: (3,),  s: float

    Returns:
        L_out: (6, N)
    """
    m, d   = L[:3], L[3:]
    d_out  = R @ d                           # (3, N)
    m_out  = s * (R @ m) + _skew(t) @ d_out  # (3, N)
    return np.vstack([m_out, d_out])


# ── RANSAC outer loop ─────────────────────────────────────────────────────────

def solve_translation_fixed_scale(
    L1: np.ndarray, L2: np.ndarray, R: np.ndarray, s: float,
) -> np.ndarray:
    """
    Solve for translation t with R and s fixed.

    Given SIM(3) constraint  m2 = s·R·m1 + t × (R·d1),
    rearrange as  -skew(R·d1_i) · t = m2_i − s·(R·m1_i)
    and solve via least squares.  This 3N×3 system is well-conditioned
    whenever at least two correspondences have different directions.

    Args:
        L1, L2: (6, N) Plücker [m; d]
        R: (3, 3)
        s: fixed scale factor

    Returns:
        t: (3,) translation vector
    """
    m1, d1 = L1[:3], L1[3:]
    m2     = L2[:3]
    N      = L1.shape[1]

    Rd1 = R @ d1       # (3, N)
    Rm1 = R @ m1       # (3, N)

    A = np.zeros((3 * N, 3))
    b = np.zeros(3 * N)
    for i in range(N):
        row = 3 * i
        A[row:row + 3] = -_skew(Rd1[:, i])            # -skew(R·d1) coeff of t
        b[row:row + 3] = m2[:, i] - s * Rm1[:, i]     # residual

    t, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return t


def ransac_sim3(
    L1: np.ndarray,
    L2: np.ndarray,
    n_iter: int = 1000,
    inlier_angle_rad: float = 0.15,    # ≈ 8.6°
    min_inliers: int = 6,
    min_sample: int = 3,
    seed: int = 42,
    s_prior: float = -1.0,
    lambda_s: float = 5.0,
) -> tuple:
    """
    RANSAC SIM(3) solver using Grassmannian distance as the inlier metric.

    The algorithm:
      1. Sample ``min_sample`` random line correspondences.
      2. Estimate R via Procrustes, then (t, s) via linear least squares.
      3. Transform all source lines and compute Grassmannian distance to
         their targets; count inliers below ``inlier_angle_rad``.
      4. Keep the hypothesis with the most inliers.
      5. Re-estimate the SIM(3) on all inliers (final refinement).

    Args:
        L1:  (6, N)  Plücker coords of source lines  (SLAM, arb. scale)
        L2:  (6, N)  Plücker coords of target lines  (DA3, metric)
        n_iter: RANSAC iterations
        inlier_angle_rad: Grassmannian distance threshold (radians)
        min_inliers: minimum inliers to declare a valid hypothesis
        min_sample:  minimal sample size (≥ 3 for numerical stability)
        seed: RNG seed for reproducibility
        s_prior: if > 0, regularise scale toward this value (use rough estimate)
        lambda_s: strength of scale regularisation

    Returns:
        R_best: (3, 3),  t_best: (3,),  s_best: float,
        inlier_mask: (N,) bool,  n_inliers: int
    """
    assert L1.shape == L2.shape, "L1 and L2 must have the same shape."
    N = L1.shape[1]
    rng = np.random.default_rng(seed)

    # Stratified direction bins for minimal-sample selection.
    # Bin each line by its dominant axis to avoid degenerate all-parallel samples.
    dominant_axis = np.argmax(np.abs(L1[3:]), axis=0)  # (N,) in {0,1,2}
    bins = [np.where(dominant_axis == ax)[0] for ax in range(3)]
    bins = [b for b in bins if len(b) > 0]             # drop empty bins

    # Pre-normalise only for the Grassmannian distance check (inlier test).
    # The SIM(3) *solver* always uses the physical (unnormalised) coordinates.
    L2_n = normalize_plucker(L2)

    def _evaluate(R_c, t_c, s_c):
        """Transform L1 and count Grassmannian inliers."""
        L1_tf = normalize_plucker(transform_lines(L1, R_c, t_c, s_c))
        dists = grassmannian_distance(L1_tf, L2_n)
        mask  = dists < inlier_angle_rad
        return mask, int(mask.sum()), dists

    best_ic   = 0
    best_R    = np.eye(3)
    best_t    = np.zeros(3)
    best_s    = s_prior if s_prior > 0 else 1.0
    best_mask = np.zeros(N, dtype=bool)

    # ── Seed with the rough hypothesis (s_prior, I, 0) ────────────────────
    # For scenes with many near-parallel lines (e.g. chessboard), the joint
    # (t, s) linear solve is ill-conditioned.  Seeding from the rough scale
    # (estimated from moment-magnitude ratios) avoids getting stuck in a
    # degenerate minimum.
    if s_prior > 0:
        R_seed = solve_rotation(L1[3:], L2[3:])   # full-set direction Procrustes
        t_seed = solve_translation_fixed_scale(L1, L2, R_seed, s_prior)
        mask_seed, ic_seed, _ = _evaluate(R_seed, t_seed, s_prior)
        if ic_seed > best_ic:
            best_ic, best_mask = ic_seed, mask_seed
            best_R, best_t, best_s = R_seed, t_seed, s_prior

    # ── RANSAC iterations ─────────────────────────────────────────────────
    for _ in range(n_iter):
        # Stratified sampling: pick at least one line from each direction bin
        # to prevent degenerate all-parallel minimal samples.
        if len(bins) >= min_sample:
            chosen_bins = rng.choice(len(bins), min_sample, replace=False)
            idx = np.array([rng.choice(bins[b]) for b in chosen_bins])
        else:
            idx = rng.choice(N, min_sample, replace=False)

        try:
            R_cand = solve_rotation(L1[3:, idx], L2[3:, idx])

            if s_prior > 0:
                # Two-stage: fix scale to prior, solve translation only.
                # More robust for scenes dominated by parallel lines.
                s_cand = s_prior
                t_cand = solve_translation_fixed_scale(
                    L1[:, idx], L2[:, idx], R_cand, s_cand
                )
            else:
                # Joint (t, s) solve when no prior is available.
                t_cand, s_cand = solve_translation_scale(
                    L1[:, idx], L2[:, idx], R_cand,
                    s_prior=s_prior, lambda_s=lambda_s,
                )
                if s_cand <= 0 or not np.isfinite(s_cand):
                    continue
        except (np.linalg.LinAlgError, ValueError):
            continue

        mask, ic, _ = _evaluate(R_cand, t_cand, s_cand)
        if ic > best_ic:
            best_ic   = ic
            best_mask = mask
            best_R    = R_cand
            best_t    = t_cand
            best_s    = s_cand

    # ── Final refinement on all inliers ───────────────────────────────────
    if best_ic >= min_inliers:
        try:
            R_ref  = solve_rotation(L1[3:, best_mask], L2[3:, best_mask])
            # Refine scale from moment-magnitude median (robust to translation)
            s_ref  = float(np.median(
                np.linalg.norm(L2[:3, best_mask], axis=0) /
                (np.linalg.norm((R_ref @ L1[3:, best_mask]) * 0 +
                                R_ref @ L1[:3, best_mask], axis=0) + 1e-12)
            ))
            if s_ref <= 0 or not np.isfinite(s_ref):
                s_ref = best_s
            t_ref  = solve_translation_fixed_scale(
                L1[:, best_mask], L2[:, best_mask], R_ref, s_ref
            )
            mask_ref, ic_ref, _ = _evaluate(R_ref, t_ref, s_ref)
            if ic_ref >= best_ic:
                best_R, best_t, best_s = R_ref, t_ref, s_ref
                best_mask, best_ic = mask_ref, ic_ref
        except (np.linalg.LinAlgError, ValueError):
            pass

    return best_R, best_t, best_s, best_mask, best_ic


# ── Nearest-neighbour correspondence finder ───────────────────────────────────

def find_correspondences(
    L1: np.ndarray,
    L2: np.ndarray,
    max_angle_rad: float = 0.30,   # ≈ 17°
    max_per_query: int = 1,
) -> tuple:
    """
    Find tentative line correspondences by nearest-neighbour in Plücker space.

    For each line in L1 (source), find the closest line in L2 (target) by
    Grassmannian distance.  Only pairs below ``max_angle_rad`` are kept.

    Args:
        L1: (6, M)  source Plücker coordinates (after rough alignment)
        L2: (6, K)  target Plücker coordinates
        max_angle_rad: maximum Grassmannian distance to accept a match
        max_per_query: how many nearest neighbours to keep per source line (1)

    Returns:
        idx1: (P,) indices into L1
        idx2: (P,) indices into L2
        dists: (P,) Grassmannian distances of accepted pairs
    """
    L1_n = normalize_plucker(L1)   # (6, M)
    L2_n = normalize_plucker(L2)   # (6, K)

    # cos similarity matrix via dot product
    # dots[i,j] = L1_n[:,i] · L2_n[:,j]
    dots = L1_n.T @ L2_n           # (M, K)
    cos_mat = np.clip(np.abs(dots), 0.0, 1.0)
    angle_mat = np.arccos(cos_mat)  # (M, K) Grassmannian distances

    idx1_list, idx2_list, dist_list = [], [], []

    for i in range(L1_n.shape[1]):
        nn_idx = int(np.argmin(angle_mat[i]))
        nn_dist = float(angle_mat[i, nn_idx])
        if nn_dist < max_angle_rad:
            idx1_list.append(i)
            idx2_list.append(nn_idx)
            dist_list.append(nn_dist)

    if not idx1_list:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([])

    return (
        np.array(idx1_list, dtype=int),
        np.array(idx2_list, dtype=int),
        np.array(dist_list),
    )


# ── Trajectory scale correction ───────────────────────────────────────────────

def apply_sim3_to_poses(
    poses_twc: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    s: float,
) -> np.ndarray:
    """
    Apply a SIM(3) transform to an array of camera-to-world poses.

    The SIM(3) maps points in the SLAM world frame to the metric DA3 frame:
        p_metric = s · R · p_slam + t

    Camera position:   t_wc_new = s · R · t_wc_old + t
    Camera rotation:   R_wc_new = R · R_wc_old       (scale-free)

    Args:
        poses_twc: (N, 4, 4) camera-to-world transforms (SLAM scale)
        R: (3, 3),  t: (3,),  s: float

    Returns:
        poses_corrected: (N, 4, 4) in metric scale
    """
    poses_corrected = poses_twc.copy()
    for i in range(len(poses_twc)):
        T = poses_twc[i]
        t_old = T[:3, 3]
        R_old = T[:3, :3]
        poses_corrected[i, :3, 3]  = s * (R @ t_old) + t
        poses_corrected[i, :3, :3] = R @ R_old
    return poses_corrected
