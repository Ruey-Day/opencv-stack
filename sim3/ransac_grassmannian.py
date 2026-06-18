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
from sim3.ransac import model_estimate_sim3 as _minimal_sim3


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

    if s_prior > 0:
        # Tikhonov: add virtual row [0, 0, 0, lambda_s] · x = lambda_s · s_prior
        A = np.vstack([A, [0.0, 0.0, 0.0, lambda_s]])
        b = np.append(b, lambda_s * s_prior)

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    t, s = x[:3], float(x[3])
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
    t_prior: np.ndarray | None = None, lambda_t: float = 2.0,
) -> np.ndarray:
    """
    Solve for translation t with R and s fixed.

    Given SIM(3) constraint  m2 = s·R·m1 + t × (R·d1),
    rearrange as  -skew(R·d1_i) · t = m2_i − s·(R·m1_i)
    and solve via least squares.  This 3N×3 system is well-conditioned
    whenever at least two correspondences have different directions.

    When ``t_prior`` is provided, a Tikhonov term  λ·I · t = λ·t_prior  is
    appended to the LS system.  This anchors the minimum-norm solution toward
    t_prior in directions where the line correspondences leave t unconstrained
    (e.g. horizontal-only lines leave the vertical translation unconstrained).
    The prior weight lambda_t is tiny relative to the line equations, so it
    does not affect well-constrained directions.

    Args:
        L1, L2: (6, N) Plücker [m; d]
        R: (3, 3)
        s: fixed scale factor
        t_prior: (3,) soft anchor for t in unconstrained directions
        lambda_t: Tikhonov weight for t_prior (default 2.0)

    Returns:
        t: (3,) translation vector
    """
    m1 = L1[:3]
    d1 = L1[3:]
    m2 = L2[:3]
    N  = L1.shape[1]

    Rm1 = R @ m1       # (3, N)
    Rd1 = R @ d1       # (3, N)  use rotated source direction (correct SIM3 constraint)

    A = np.zeros((3 * N, 3))
    b = np.zeros(3 * N)
    for i in range(N):
        row = 3 * i
        A[row:row + 3] = -_skew(Rd1[:, i])            # -skew(R·d1) coeff of t
        b[row:row + 3] = m2[:, i] - s * Rm1[:, i]     # residual

    if t_prior is not None:
        tp = np.asarray(t_prior, dtype=np.float64)
        A = np.vstack([A, lambda_t * np.eye(3)])
        b = np.concatenate([b, lambda_t * tp])

    t, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return t


def ransac_sim3(
    L1: np.ndarray,
    L2: np.ndarray,
    n_iter: int = 5000,
    inlier_threshold: float = 0.3,
    min_inliers: int = 6,
    min_sample: int = 2,
    seed: int = 42,
    s_prior: float = -1.0,
    lambda_s: float = 5.0,
    early_exit_iters: int = 200,
    lo_iters: int = 10,
    distance_metric: str = 'rp5',
    lo_dir_threshold: float | None = None,
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
        min_inliers: minimum inliers to declare a valid hypothesis
        min_sample:  minimal sample size (≥ 3 for numerical stability)
        seed: RNG seed for reproducibility
        s_prior: if > 0, regularise scale toward this value (use rough estimate)
        lambda_s: strength of scale regularisation
        early_exit_iters: abort after this many iterations if best_ic is still 0 (default 200)
        lo_iters: local-optimization passes — each pass re-fits (R, s, t) via joint
            LS on the current inlier set and expands inliers; stops early if count
            stops growing (default 10)
        distance_metric: 'rp5' (default) — RP⁵ single principal angle via normalised
            dot product; 'g24' — G(2,4) proper geodesic via Shin et al. ICCV 2025,
            two principal angles from SVD of the 4×2 subspace product.
        lo_dir_threshold: if set, the LO Procrustes step uses only inlier pairs whose
            direction angle (arccos|R·d1 · d2|) is below this value (radians).
            Translation/scale LS still uses all inliers.  Useful when
            ``inlier_threshold`` is loose enough to accept position-matched pairs
            with large direction disagreement — tighter direction subset gives
            cleaner R while the full inlier pool gives accurate t.
            Typical value: 0.15 rad (≈ 8.6°).  Default: None (use all inliers).

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

    # ── Local optimization: maximize inlier count (good for t, s) ───────────
    if best_ic >= min_inliers:
        for _ in range(lo_iters):
            try:
                R_ref = solve_rotation(L1[3:, best_mask], L2[3:, best_mask])
                # Joint (t, s) solve — full LS correctly accounts for t×(R·d).
                t_ref, s_ref = solve_translation_scale(
                    L1[:, best_mask], L2[:, best_mask], R_ref,
                    s_prior=best_s, lambda_s=lambda_s,
                )
                if s_ref <= 0 or not np.isfinite(s_ref):
                    break
                mask_ref, ic_ref, _ = _evaluate(R_ref, t_ref, s_ref)
                if ic_ref >= best_ic:
                    best_R, best_t, best_s = R_ref, t_ref, s_ref
                    best_mask, best_ic = mask_ref, ic_ref
                else:
                    break  # inlier set stopped growing — converged
            except (np.linalg.LinAlgError, ValueError):
                break

    # ── Rotation polish: direction-tight Procrustes on the final inlier set ──
    # The standard LO above maximises inlier count but may converge with a
    # slightly biased R because position-matched pairs with loose direction
    # agreement bias the Procrustes.  This phase iterates direction-tight
    # Procrustes on the fixed inlier set with scale held constant (s from
    # the standard LO is accurate; only R and t need refinement).
    if lo_dir_threshold is not None and best_ic >= min_inliers:
        L1_in = L1[:, best_mask]
        L2_in = L2[:, best_mask]
        R_pol = best_R.copy()
        s_fixed = float(best_s)   # lock scale — only refine R and t
        t_anchor = best_t.copy()  # LO translation — anchor unconstrained directions
        for _ in range(lo_iters):
            try:
                Rd1 = R_pol @ L1_in[3:]
                dir_cos = np.abs(np.sum(Rd1 * L2_in[3:], axis=0))
                dir_tight = dir_cos > np.cos(lo_dir_threshold)
                if dir_tight.sum() < 3:
                    break
                R_new = solve_rotation(
                    L1_in[3:][:, dir_tight], L2_in[3:][:, dir_tight]
                )
                # Anchor t toward the LO estimate in unconstrained directions.
                t_new = solve_translation_fixed_scale(
                    L1_in, L2_in, R_new, s_fixed,
                    t_prior=t_anchor, lambda_t=2.0,
                )
                if not np.all(np.isfinite(t_new)):
                    break
                if np.allclose(R_new, R_pol, atol=1e-6):
                    best_R, best_t = R_new, t_new
                    break  # converged
                R_pol = R_new
                best_R, best_t = R_new, t_new
            except (np.linalg.LinAlgError, ValueError):
                break

    return best_R, best_t, best_s, best_mask, best_ic


# ── Post-RANSAC translation polish ───────────────────────────────────────────

def polish_translation(
    L1: np.ndarray,
    L2: np.ndarray,
    R: np.ndarray,
    s: float,
    t_init: np.ndarray | None = None,
    max_dir_angle: float = 0.10,
    n_iter: int = 5,
) -> np.ndarray:
    """
    Refine translation given fixed R and s via iterative direction-NN search.

    The Grassmannian inlier metric used in RANSAC is direction-dominated, so
    the recovered translation can be biased when inliers are mostly parallel
    lines.  This function sidesteps the network correspondences entirely:

      1. Apply (R, s, t_current) to ALL source lines.
      2. For each transformed source line, find the nearest target line by
         direction (Grassmannian NN, ±sign handled).
      3. Solve t from the moment equations of direction-matched pairs.
      4. Repeat until convergence (typically 3–5 iterations).

    Using the full line sets (not just the top-K network pairs) gives many
    more direction-diverse correspondences and a much better-conditioned
    translation system.

    Args:
        L1: (6, N1)  full source Plücker lines  (mono, all lines)
        L2: (6, N2)  full target Plücker lines  (metric, all lines)
        R, s:        fixed rotation and scale from RANSAC
        t_init:      starting translation (use RANSAC estimate; default zeros)
        max_dir_angle: Grassmannian direction threshold in radians (default 0.10 ≈ 6°)
        n_iter:      ICP-style iterations (default 5)

    Returns:
        t: (3,) refined translation
    """
    t = t_init.copy() if t_init is not None else np.zeros(3, dtype=np.float64)

    # Pre-normalise target directions once
    d2 = L2[3:]
    d2_n = d2 / (np.linalg.norm(d2, axis=0, keepdims=True) + 1e-12)   # (3, N2)
    cos_thresh = np.cos(max_dir_angle)

    for _ in range(n_iter):
        L1_tf = transform_lines(L1, R, t, s)
        d1_n  = L1_tf[3:]
        d1_n  = d1_n / (np.linalg.norm(d1_n, axis=0, keepdims=True) + 1e-12)  # (3, N1)

        # Direction NN (±sign): cos_mat[i,j] = |d1_i · d2_j|
        cos_mat = np.abs(d1_n.T @ d2_n)          # (N1, N2)
        nn_idx  = np.argmax(cos_mat, axis=1)      # (N1,) best match in L2
        nn_cos  = cos_mat[np.arange(L1.shape[1]), nn_idx]
        good    = nn_cos > cos_thresh             # (N1,) direction-quality filter

        if good.sum() < 3:
            break

        try:
            t = solve_translation_fixed_scale(
                L1[:, good], L2[:, nn_idx[good]], R, s
            )
        except np.linalg.LinAlgError:
            break

    return t


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
