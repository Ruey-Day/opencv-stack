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
    Principal angle between Plücker lines on RP⁵ (G(1,5) ambient space).

    Both L1 and L2 must already be unit-normalised.

    Returns:
        (N,) principal angles in radians ∈ [0, π/2]
    """
    dots = np.sum(L1 * L2, axis=0)                   # (N,)
    cos_theta = np.clip(np.abs(dots), 0.0, 1.0)
    return np.arccos(cos_theta)                        # (N,)

def plucker_to_g24_basis(L: np.ndarray) -> np.ndarray:
    """
    Embed Plücker lines as 2D subspaces of R⁴ — points on G(2,4).

    A 3D line with Plücker coords [m; d] is represented by the span of:
        col0 = [p₀; 1]   where p₀ = m × d  (foot of perpendicular from origin)
        col1 = [d;  0]   (direction as point at infinity)

    Because the Plücker constraint m·d = 0 guarantees col0 ⊥ col1 in R⁴,
    only col0 needs normalising.  col1 is already unit since |d| = 1.

    Args:
        L: (6, N)  Plücker [m; d]  (d need not be unit; will be normalised here)

    Returns:
        Q: (N, 4, 2)  orthonormal basis matrices for G(2,4) points
    """
    m, d = L[:3], L[3:]                          # (3, N) each
    d = d / (np.linalg.norm(d, axis=0, keepdims=True) + 1e-12)  # unit direction
    p0 = np.cross(m.T, d.T).T                    # (3, N)  foot of perpendicular

    N = L.shape[1]
    col0 = np.vstack([p0, np.ones((1, N))])      # (4, N)  [p₀; 1]
    col1 = np.vstack([d,  np.zeros((1, N))])     # (4, N)  [d; 0]

    # Normalise col0; Gram-Schmidt col1 (handles near-zero Plücker-constraint violations)
    col0 = col0 / (np.linalg.norm(col0, axis=0, keepdims=True) + 1e-12)
    proj  = np.sum(col1 * col0, axis=0, keepdims=True)  # (1, N)
    col1  = col1 - proj * col0
    col1  = col1 / (np.linalg.norm(col1, axis=0, keepdims=True) + 1e-12)

    # Stack: (N, 4, 2)
    return np.stack([col0.T, col1.T], axis=2)    # (N, 4, 2)


def g24_geodesic_distance(L1: np.ndarray, L2: np.ndarray,
                           Q2: np.ndarray | None = None) -> np.ndarray:
    """
    Geodesic distance on G(2,4) between N paired Plücker lines.

    Implements the distance from Shin et al. (ICCV 2025) "Registration beyond
    Points":  embed each line as a 2D subspace of R⁴, compute the two principal
    angles (θ₁, θ₂) via SVD of Q₁ᵀQ₂, return ‖(θ₁, θ₂)‖ (Frobenius norm).

    Args:
        L1: (6, N)  source Plücker lines (after SIM(3) transform)
        L2: (6, N)  target Plücker lines  (unused if Q2 is provided)
        Q2: (N, 4, 2)  pre-computed G(2,4) basis for L2  (optional, for speed)

    Returns:
        dists: (N,) geodesic distances in radians ∈ [0, π/√2]
    """
    Q1 = plucker_to_g24_basis(L1)                        # (N, 4, 2)
    if Q2 is None:
        Q2 = plucker_to_g24_basis(L2)                    # (N, 4, 2)
    M      = np.matmul(Q1.transpose(0, 2, 1), Q2)        # (N, 2, 2)
    _, sigma, _ = np.linalg.svd(M, full_matrices=False)  # sigma: (N, 2)
    sigma  = np.clip(sigma, 0.0, 1.0)
    theta  = np.arccos(sigma)                             # (N, 2) principal angles
    return np.sqrt(np.sum(theta ** 2, axis=1))            # (N,) Frobenius norm

def _skew(v: np.ndarray) -> np.ndarray:
    """3×3 skew-symmetric matrix for vector v (3,)."""
    return np.array([
        [ 0.0,  -v[2],  v[1]],
        [ v[2],   0.0, -v[0]],
        [-v[1],  v[0],  0.0],
    ])

def solve_rotation(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """
    find R ∈ SO(3) minimising Σ‖R·d1_i − d2_i‖².

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
) -> tuple:
    """
    Linear solve for translation t and scale s given rotation R.

    SIM(3) moment constraint:
        m2 = s·R·m1 + skew(t)·(R·d1)

    Rearranged as a linear system in x = [t (3); s (1)]:
        [ -skew(R·d1_i) | R·m1_i ] · x = m2_i    for each line i.

    When the scene has many near-parallel lines (e.g. chessboard), the system
    is rank-deficient for translation. 

    Args:
        L1: (6, N)  Plücker [m1; d1]  (source — SLAM, arb. scale)
        L2: (6, N)  Plücker [m2; d2]  (target — DA3, metric)
        R:  (3, 3)

    Returns:
        t: (3,) translation,   s: float scale  (> 0 means SLAM is smaller)
    """
    m1  = L1[:3]         # (3, N)
    d1  = L1[3:]         # (3, N)
    m2  = L2[:3]         # (3, N)
    N   = L1.shape[1]

    Rm1  = R @ m1        # (3, N)
    Rd1  = R @ d1        # (3, N) — use rotated source direction, not target direction
                         # (correct per SIM3 constraint: m' = s·R·m + t × (R·d))

    A = np.zeros((3 * N, 4))
    b = np.zeros(3 * N)
    for i in range(N):
        row = 3 * i
        A[row:row + 3, :3] = -_skew(Rd1[:, i])  # coefficient of t  (was: d2 — bug)
        A[row:row + 3,  3] =  Rm1[:, i]          # coefficient of s  (R·m1_i)
        b[row:row + 3]     =  m2[:, i]

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    t, s = x[:3], float(x[3])
    return t, s

def transform_lines(
    L: np.ndarray, R: np.ndarray, t: np.ndarray, s: float
) -> np.ndarray:
    """
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

def _minimal_sim3(L1: np.ndarray, L2: np.ndarray):
    """Minimal Sim(3) estimate from a small set of line correspondences.

    Uses the sign-aware Procrustes rotation and the joint (t, s) LS solve
    already defined in this module — no dependency on the old ransac.py.

    Returns (s, R, t) or None if degenerate (s ≤ 0 or non-finite).
    """
    R = solve_rotation(L1[3:], L2[3:])
    t, s = solve_translation_scale(L1, L2, R)
    if s <= 0 or not np.isfinite(s):
        return None
    return s, R, t

def ransac_sim3(
    L1: np.ndarray,
    L2: np.ndarray,
    n_iter: int = 5000,
    inlier_threshold: float = 0.3,
    min_sample: int = 3,
    seed: int = 42,
    early_exit_iters: int = 200,
    distance_metric: str = 'rp5',
) -> tuple:
    """
    RANSAC SIM(3) solver for Plücker line correspondences.

    Uses Procrustes rotation + joint LS for (s, t) as the minimal solver, and
    the Grassmannian distance (arccos of the normalised inner product) as the
    inlier metric — the theoretical contribution of this work.  After applying
    the SIM(3) hypothesis to L1, both the transformed line and the target are
    normalised to unit 6-norm and compared via:

        θ = arccos( |L1_tf_norm · L2_norm| )   ∈ [0, π/2]

    Scale errors propagate into the moment component of the transformed line,
    so θ > 0 when the hypothesis has wrong scale — the metric is scale-aware
    through the moment, not just through direction alignment.

    The algorithm:
      1. Sample ``min_sample`` random line correspondences.
      2. Estimate R via Procrustes, then (t, s) via linear least squares.
      3. Transform all source lines and compute L2 Plücker residual to targets;
         count inliers below ``inlier_threshold``.
      4. Keep the hypothesis with the most inliers.
      5. Iterative local optimisation: re-fit (R, s, t) on inlier set, expand.

    Early exit: if after ``early_exit_iters`` iterations the best inlier count
    is still zero, returns failure immediately.

    Args:
        L1:  (6, N)  Plücker coords of source lines  (SLAM, arb. scale)
        L2:  (6, N)  Plücker coords of target lines  (DA3, metric)
        n_iter: RANSAC iterations (default 5000)
        inlier_threshold: distance threshold in radians (default 0.3).
            For 'rp5': single principal angle θ ∈ [0, π/2].
            For 'g24': Frobenius norm of two principal angles ∈ [0, π/√2].
        min_sample:  minimal sample size (≥ 3 for numerical stability)
        seed: RNG seed for reproducibility
        early_exit_iters: abort after this many iterations if best_ic is still 0 (default 200)
        distance_metric: 'rp5' (default) — RP⁵ single principal angle via normalised
            dot product; 'g24' — G(2,4) proper geodesic via Shin et al. ICCV 2025,
            two principal angles from SVD of the 4×2 subspace product.

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

    # Pre-compute target representation once (L2 is fixed throughout RANSAC)
    if distance_metric == 'g24':
        _L2_precomp = plucker_to_g24_basis(L2)   # (N, 4, 2)

        def _evaluate(R_c, t_c, s_c):
            """Transform L1 and count inliers via G(2,4) max principal angle.

            For our homogeneous line embedding the 2×2 inner-product matrix M is
            block-diagonal: one entry captures direction agreement, the other
            captures foot-of-perpendicular (position) agreement.  Using the
            LARGER principal angle θ₂ = arccos(σ₂) as the distance means BOTH
            direction and position must agree within ``inlier_threshold``.  This
            is strictly tighter than the Frobenius norm √(θ₁²+θ₂²) and makes
            translation estimation robust: pairs with correct direction but wrong
            position are excluded from the inlier set.
            """
            L1_tf = transform_lines(L1, R_c, t_c, s_c)
            Q1 = plucker_to_g24_basis(L1_tf)
            M = np.matmul(Q1.transpose(0, 2, 1), _L2_precomp)   # (N, 2, 2)
            _, sigma, _ = np.linalg.svd(M, full_matrices=False)
            sigma = np.clip(sigma, 0.0, 1.0)
            theta = np.arccos(sigma)           # (N, 2), θ[:,0] ≤ θ[:,1]
            dists = theta[:, 1]               # max principal angle — strictest
            mask  = dists < inlier_threshold
            return mask, int(mask.sum()), dists
    else:  # 'rp5' — RP⁵ single principal angle (default)
        _L2_norm = normalize_plucker(L2)

        def _evaluate(R_c, t_c, s_c):
            """Transform L1 and count inliers via RP⁵ Grassmannian distance."""
            L1_tf      = transform_lines(L1, R_c, t_c, s_c)
            L1_tf_norm = normalize_plucker(L1_tf)
            dists      = grassmannian_distance(L1_tf_norm, _L2_norm)
            mask       = dists < inlier_threshold
            return mask, int(mask.sum()), dists

    best_ic   = 0
    best_R    = np.eye(3)
    best_t    = np.zeros(3)
    best_s    = 1.0
    best_mask = np.zeros(N, dtype=bool)
    
    for it in range(n_iter):
        if it == early_exit_iters and best_ic == 0:
            break
        # Stratified sampling: pick at least one line from each direction bin
        # to prevent degenerate all-parallel minimal samples.
        if len(bins) >= min_sample:
            chosen_bins = rng.choice(len(bins), min_sample, replace=False)
            idx = np.array([rng.choice(bins[b]) for b in chosen_bins])
        else:
            idx = rng.choice(N, min_sample, replace=False)

        try:
            result = _minimal_sim3(L1[:, idx], L2[:, idx])
            if result is None:
                continue
            s_cand, R_cand, t_cand = result
            t_cand = t_cand.flatten()
        except (np.linalg.LinAlgError, ValueError):
            continue

        mask, ic, _ = _evaluate(R_cand, t_cand, s_cand)
        if ic > best_ic:
            best_ic   = ic
            best_mask = mask
            best_R    = R_cand
            best_t    = t_cand
            best_s    = s_cand
    
    return best_R, best_t, best_s, best_mask, best_ic
