"""
Grassmannian SIM(3) RANSAC for metric scale recovery from 3D line maps.

A 3D line is characterised by its Plücker coordinates L = [m; d] ∈ R^6,
where d ∈ S^2 is the unit direction and m = p × d is the moment vector.
Embedded as a 2D subspace of R^4 via the affine Grassmannian Y_z map
(Shin et al., ICCV 2025), lines are points on G(2,4).

SIM(3) line transformation
--------------------------
    direction: d  →  R·d
    moment:    m  →  s·R·m + t × (R·d)

Minimal solver (3 line pairs → 7 DOF):
    1. Rotation R  — SVD Procrustes (L2) or IRLS weighted Procrustes (L1)
    2. Scale s and translation t — linear least squares given R:
           m2_i = s·R·m1_i + skew(t)·(R·d1_i)
       stacked as  [-skew(R·d1_i) | R·m1_i] · [t; s] = m2_i.

Inlier refinement (refine_sim3):
    Joint linear solve for P=s·R and t in one LS:
        P·m1_i + skew(t)·(R·d1_i) = m2_i
    Then extract s = det(P)^(1/3), R = P/s, projected to SO(3).
"""
import numpy as np

def plucker_from_endpoints(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Convert 3D line endpoints to Plücker coordinates [m; d], shape (6, N)."""
    p1 = np.atleast_2d(p1).astype(float)
    p2 = np.atleast_2d(p2).astype(float)
    diff = p2 - p1
    d = diff / (np.linalg.norm(diff, axis=1, keepdims=True) + 1e-12)
    m = np.cross(p1, d)
    return np.vstack([m.T, d.T])  # (6, N)


def plucker_to_g24_basis(L: np.ndarray) -> np.ndarray:
    """
    Embed Plücker lines as 2D subspaces of R⁴ — points on G(2,4).

    Implements the Y_z affine Grassmannian embedding from Shin et al. (ICCV 2025):
        col0 = [p₀; 1] / ‖·‖   where p₀ = m × d  (foot of perpendicular)
        col1 = [d;  0]           (direction at infinity, Gram-Schmidt orthogonalised)

    Args:
        L: (6, N)  Plücker [m; d]

    Returns:
        Q: (N, 4, 2)  orthonormal basis matrices
    """
    m, d = L[:3], L[3:]
    d = d / (np.linalg.norm(d, axis=0, keepdims=True) + 1e-12)
    p0 = np.cross(m.T, d.T).T   # (3, N) foot of perpendicular

    N = L.shape[1]
    col0 = np.vstack([p0, np.ones((1, N))])
    col1 = np.vstack([d,  np.zeros((1, N))])

    col0 = col0 / (np.linalg.norm(col0, axis=0, keepdims=True) + 1e-12)
    col1 = col1 - col0 * (col1 * col0).sum(axis=0, keepdims=True)
    col1 = col1 / (np.linalg.norm(col1, axis=0, keepdims=True) + 1e-12)

    return np.stack([col0.T, col1.T], axis=2)  # (N, 4, 2)


def g24_geodesic_distance(L1: np.ndarray, L2: np.ndarray,
                           Q2: np.ndarray | None = None) -> np.ndarray:
    """
    Geodesic distance on G(2,4) between N paired Plücker lines (Shin et al. ICCV 2025).
    Returns (N,) Frobenius norm of principal angles ∈ [0, π/√2].
    """
    Q1 = plucker_to_g24_basis(L1)
    if Q2 is None:
        Q2 = plucker_to_g24_basis(L2)
    _, sigma, _ = np.linalg.svd(np.matmul(Q1.transpose(0, 2, 1), Q2), full_matrices=False)
    theta = np.arccos(np.clip(sigma, 0.0, 1.0))
    return np.sqrt(np.sum(theta ** 2, axis=1))


def _skew(v: np.ndarray) -> np.ndarray:
    """3×3 skew-symmetric matrix for vector v (3,)."""
    return np.array([
        [ 0.0,  -v[2],  v[1]],
        [ v[2],   0.0, -v[0]],
        [-v[1],  v[0],  0.0],
    ])


def solve_rotation_l2(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """
    SVD Procrustes: global minimiser of Σ‖R·d1_i − d2_i‖² over SO(3).

    Handles undirected lines by flipping d1_i when d1_i·d2_i < 0 before
    building the cross-covariance.  O(N) — single SVD.
    """
    dots = (d1 * d2).sum(axis=0)
    d1_aligned = d1 * np.where(dots >= 0, 1.0, -1.0)[np.newaxis, :]
    U, _, Vt = np.linalg.svd(d2 @ d1_aligned.T)
    det = np.linalg.det(U @ Vt)
    return U @ np.diag([1.0, 1.0, float(det)]) @ Vt


def solve_rotation_l1(
    d1: np.ndarray,
    d2: np.ndarray,
    n_iter: int = 20,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    IRLS: minimises Σ‖R·d1_i − d2_i‖₁ via iteratively reweighted Procrustes.

    Each iteration:
      1. Compute residuals r_i = ‖R·d1_i_signed − d2_i‖ with sign-alignment.
      2. Set weights w_i = 1 / max(r_i, eps).
      3. Solve weighted Procrustes via SVD of Σ w_i · d2_i · d1_i_signed^T.

    Warm-started from the L2 Procrustes solution.  Converges in ~10 iterations.
    """
    R = solve_rotation_l2(d1, d2)
    for _ in range(n_iter):
        u = R @ d1
        signs = np.where((u * d2).sum(axis=0) >= 0, 1.0, -1.0)
        u_signed = u * signs
        residuals = np.linalg.norm(u_signed - d2, axis=0)
        w = 1.0 / np.maximum(residuals, eps)
        d1_signed = d1 * signs
        H = (d2 * w) @ d1_signed.T
        U, _, Vt = np.linalg.svd(H)
        det = np.linalg.det(U @ Vt)
        R_new = U @ np.diag([1.0, 1.0, float(det)]) @ Vt
        if np.linalg.norm(R_new - R) < 1e-8:
            break
        R = R_new
    return R


def solve_translation_scale(L1: np.ndarray, L2: np.ndarray, R: np.ndarray) -> tuple:
    """
    Linear solve for (t, s) given R via the SIM(3) moment constraint:
        m2_i = s·R·m1_i + skew(t)·(R·d1_i)
    Stacked as [-skew(R·d1_i) | R·m1_i] · [t; s] = m2_i  (3N × 4 least squares).

    Returns:
        t: (3,),  s: float
    """
    m1, d1, m2 = L1[:3], L1[3:], L2[:3]
    N   = L1.shape[1]
    Rm1 = R @ m1
    Rd1 = R @ d1

    A = np.zeros((3 * N, 4))
    b = np.zeros(3 * N)
    for i in range(N):
        row = 3 * i
        A[row:row + 3, :3] = -_skew(Rd1[:, i])
        A[row:row + 3,  3] =  Rm1[:, i]
        b[row:row + 3]     =  m2[:, i]

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return x[:3], float(x[3])


def refine_sim3(
    L1: np.ndarray,
    L2: np.ndarray,
    R_init: np.ndarray,
    t_init: np.ndarray,
    s_init: float,
) -> tuple:
    """
    Joint linear refinement of (R, t, s) on a set of inlier line pairs.

    Instead of solving R and (t, s) sequentially, this writes a single
    linear system in P = s·R (3×3 free matrix) and t:

        P·m1_i + skew(t)·(R_cur·d1_i) = m2_i          (3N × 12 LS)

    where R_cur·d1_i is approximated using the current rotation estimate.
    After solving, s and R are recovered from P:

        s = det(P)^(1/3),   R = P / s  → projected to SO(3) via SVD.

    This is tighter than sequential solving because the moment residuals
    jointly constrain the scale, rotation, and translation rather than
    accumulating errors from the direction-only stage.

    Args:
        L1, L2:  (6, N) Plücker [m; d] inlier pairs
        R_init, t_init, s_init: initial estimate (from RANSAC best hypothesis)

    Returns:
        R: (3,3), t: (3,), s: float
    """
    m1, d1, m2 = L1[:3], L1[3:], L2[:3]
    N = L1.shape[1]
    Rd1 = R_init @ d1  # (3, N) — current direction estimate

    # Build 3N × 12 system:  [m1_i^T ⊗ I₃ | -skew(Rd1_i)] · [vec(P); t] = m2_i
    A = np.zeros((3 * N, 12))
    b = np.zeros(3 * N)
    for i in range(N):
        row = 3 * i
        # columns 0-8: vec(P) contribution — P·m1_i = [m1_i^T ⊗ I₃] · vec(P)
        A[row:row + 3, :9] = np.kron(m1[:, i], np.eye(3))
        # columns 9-11: t contribution — skew(t)·Rd1_i = -skew(Rd1_i)·t
        A[row:row + 3, 9:] = -_skew(Rd1[:, i])
        b[row:row + 3]     =  m2[:, i]

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    P = x[:9].reshape(3, 3)
    t = x[9:]

    # Extract s and R from P = s·R
    s = float(np.cbrt(np.linalg.det(P)))
    if s <= 0 or not np.isfinite(s):
        return R_init, t_init, s_init

    U, _, Vt = np.linalg.svd(P / s)
    det = np.linalg.det(U @ Vt)
    R = U @ np.diag([1.0, 1.0, float(det)]) @ Vt
    return R, t, s


def transform_lines(L: np.ndarray, R: np.ndarray, t: np.ndarray, s: float) -> np.ndarray:
    """Apply SIM(3) to Plücker lines: d' = R·d,  m' = s·R·m + skew(t)·(R·d)."""
    d_out = R @ L[3:]
    m_out = s * (R @ L[:3]) + _skew(t) @ d_out
    return np.vstack([m_out, d_out])


def _minimal_sim3(
    L1: np.ndarray,
    L2: np.ndarray,
    rotation_mode: str = 'l2',
):
    """
    Minimal SIM(3) estimate from a small set of line correspondences.

    Args:
        rotation_mode: 'l2' — SVD Procrustes (fastest, global L2 optimum);
                       'l1' — IRLS weighted Procrustes (robust to direction outliers).

    Returns (s, R, t) or None if degenerate.
    """
    if rotation_mode == 'l1':
        R = solve_rotation_l1(L1[3:], L2[3:])
    else:
        R = solve_rotation_l2(L1[3:], L2[3:])

    t, s = solve_translation_scale(L1, L2, R)
    if s <= 0 or not np.isfinite(s):
        return None
    return s, R, t


def _direction_bins(L: np.ndarray) -> dict:
    """
    Partition N lines into 4 direction quadrant bins on the upper hemisphere.

    Plücker directions are undirected (d ≡ -d), so each direction is
    canonicalized to the upper hemisphere before binning:
        - flip d so z > 0; if z ≈ 0 use y > 0; if both ≈ 0 use x > 0.
    The 4 bins are quadrants of the upper hemisphere: (sign(dx), sign(dy)).

    Returns a dict mapping bin_id → list of line indices.
    """
    d = L[3:].copy()  # (3, N)
    d /= np.linalg.norm(d, axis=0, keepdims=True) + 1e-12

    flip = np.where(np.abs(d[2]) > 1e-6, np.sign(d[2]),
           np.where(np.abs(d[1]) > 1e-6, np.sign(d[1]),
                    np.sign(d[0])))
    flip = np.where(flip == 0, 1.0, flip)
    d *= flip[np.newaxis, :]

    bin_ids = (d[0] >= 0).astype(int) * 2 + (d[1] >= 0).astype(int)
    groups: dict = {}
    for i, b in enumerate(bin_ids):
        groups.setdefault(int(b), []).append(i)
    return groups


def ransac_sim3(
    L1: np.ndarray,
    L2: np.ndarray,
    n_iter: int = 5000,
    inlier_threshold: float = 0.3,
    min_sample: int = 3,
    seed: int = 42,
    early_exit_iters: int = 200,
    rotation_mode: str = 'l2',
    refine: bool = True,
) -> tuple:
    """
    Inlier metric: G(2,4) max principal angle (Shin et al. ICCV 2025).
    Each line is embedded as a 2D subspace of R⁴ via the Y_z affine Grassmannian
    map. The larger of the two principal angles between source (transformed) and
    target subspaces is used as the residual — it requires both direction AND
    position to agree within inlier_threshold.

    Each iteration:
      1. Sample ``min_sample`` lines, preferring one per direction quadrant bin
         (stratified) to avoid degenerate near-parallel samples. Falls back to
         uniform random when fewer distinct bins exist than min_sample.
      2. Solve R (L2 SVD or L1 IRLS), then (t, s) via LS.
      3. Count G(2,4) inliers below inlier_threshold.
      4. Keep the hypothesis with the most inliers.
    After the loop, optionally refine on all inliers via joint linear solve.

    Args:
        L1, L2:           (6, N) Plücker [m; d] source and target lines
        n_iter:           RANSAC iterations (default 5000)
        inlier_threshold: max principal angle threshold in radians ∈ [0, π/2]
        min_sample:       minimal sample size (default 3)
        seed:             RNG seed
        early_exit_iters: quit early if best inlier count is still 0
        rotation_mode:    'l2' (SVD Procrustes) or 'l1' (IRLS, slower per iter)
        refine:           if True, polish best hypothesis via joint linear solve
                          on all inliers before returning

    Returns:
        R: (3,3), t: (3,), s: float, inlier_mask: (N,) bool, n_inliers: int
    """
    assert L1.shape == L2.shape
    N   = L1.shape[1]
    rng = np.random.default_rng(seed)

    bin_groups   = _direction_bins(L1)
    diverse_bins = [b for b, idxs in bin_groups.items() if len(idxs) > 0]
    can_stratify = len(diverse_bins) >= min_sample

    _L2_basis = plucker_to_g24_basis(L2)   # (N, 4, 2)

    def _sample():
        if can_stratify:
            chosen = rng.choice(diverse_bins, min_sample, replace=False)
            return np.array([rng.choice(bin_groups[b]) for b in chosen])
        return rng.choice(N, min_sample, replace=False)

    def _evaluate(R_c, t_c, s_c):
        L1_tf = transform_lines(L1, R_c, t_c, s_c)
        Q1    = plucker_to_g24_basis(L1_tf)
        _, sigma, _ = np.linalg.svd(
            np.matmul(Q1.transpose(0, 2, 1), _L2_basis), full_matrices=False)
        dists = np.arccos(np.clip(sigma, 0.0, 1.0))[:, 1]  # max principal angle
        mask  = dists < inlier_threshold
        return mask, int(mask.sum())

    best_ic, best_R, best_t, best_s = 0, np.eye(3), np.zeros(3), 1.0
    best_mask = np.zeros(N, dtype=bool)

    for it in range(n_iter):
        if it == early_exit_iters and best_ic == 0:
            break

        idx = _sample()

        try:
            result = _minimal_sim3(L1[:, idx], L2[:, idx], rotation_mode=rotation_mode)
            if result is None:
                continue
            s_cand, R_cand, t_cand = result
            t_cand = t_cand.flatten()
        except (np.linalg.LinAlgError, ValueError):
            continue

        mask, ic = _evaluate(R_cand, t_cand, s_cand)
        if ic > best_ic:
            best_ic, best_mask, best_R, best_t, best_s = ic, mask, R_cand, t_cand, s_cand

    if refine and best_ic >= 4:
        inlier_idx = np.where(best_mask)[0]
        R_ref, t_ref, s_ref = refine_sim3(
            L1[:, inlier_idx], L2[:, inlier_idx], best_R, best_t, best_s)
        # re-evaluate on all lines with refined estimate
        mask_ref, ic_ref = _evaluate(R_ref, t_ref, s_ref)
        if ic_ref >= best_ic:
            best_R, best_t, best_s, best_mask, best_ic = R_ref, t_ref, s_ref, mask_ref, ic_ref

    return best_R, best_t, best_s, best_mask, best_ic
