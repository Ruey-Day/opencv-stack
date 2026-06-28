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
    1. Rotation R  — Riemannian gradient descent minimising the geodesic cost
                     Σ arccos²(|R·d1_i · d2_i|), warm-started from Procrustes.
    2. Scale s and translation t — linear least squares given R:
           m2_i = s·R·m1_i + skew(t)·(R·d1_i)
       stacked as  [-skew(R·d1_i) | R·m1_i] · [t; s] = m2_i.
"""
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.transform import Rotation as ScipyRot

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


def solve_rotation(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """
    Global optimum of Σ‖R·d1_i − d2_i‖² via SVD (Wahba / Procrustes).
    Sign ambiguity handled by flipping d1_i when d1_i·d2_i < 0.
    """
    dots = (d1 * d2).sum(axis=0)
    d1_aligned = d1 * np.where(dots >= 0, 1.0, -1.0)[np.newaxis, :]
    U, _, Vt = np.linalg.svd(d2 @ d1_aligned.T)
    det = np.linalg.det(U @ Vt)
    return U @ np.diag([1.0, 1.0, float(det)]) @ Vt


def solve_rotation_geodesic(
    d1: np.ndarray, d2: np.ndarray,
    n_steps: int = 20,
    lr: float = 0.3,
) -> np.ndarray:
    """
    Global optimum of Σ arccos²(|R·d1_i · d2_i|) via Riemannian gradient descent.

    Warm-started from Procrustes.  For 3 generic non-parallel lines the cost has
    a unique global minimum, so the warm start is guaranteed to be in its basin
    of attraction and gradient descent converges to the global solution.

    Lie-algebra gradient:
        ∂f/∂ω = −2 Σ_i  (θ_i / sin θ_i) · sign(c_i) · (u_i × d2_i)
    where u_i = R·d1_i,  c_i = u_i · d2_i,  θ_i = arccos(|c_i|).
    """
    R = solve_rotation(d1, d2)

    d1n = d1 / (np.linalg.norm(d1, axis=0, keepdims=True) + 1e-12)
    d2n = d2 / (np.linalg.norm(d2, axis=0, keepdims=True) + 1e-12)

    for _ in range(n_steps):
        u     = R @ d1n
        c     = np.clip((u * d2n).sum(axis=0), -1 + 1e-9, 1 - 1e-9)
        theta = np.arccos(np.abs(c))
        sin_t = np.sin(theta)
        w     = np.where(sin_t < 1e-7, 1.0, theta / sin_t)   # θ/sinθ → 1 near 0

        grad  = -2.0 * (np.cross(u.T, d2n.T).T * (w * np.sign(c))).sum(axis=1)

        omega = -lr * grad
        angle = np.linalg.norm(omega)
        if angle < 1e-10:
            break
        K  = _skew(omega / angle)
        R  = (np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)) @ R

    U, _, Vt = np.linalg.svd(R)
    det = np.linalg.det(U @ Vt)
    return U @ np.diag([1.0, 1.0, float(det)]) @ Vt


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


def transform_lines(L: np.ndarray, R: np.ndarray, t: np.ndarray, s: float) -> np.ndarray:
    """Apply SIM(3) to Plücker lines: d' = R·d,  m' = s·R·m + skew(t)·(R·d)."""
    d_out = R @ L[3:]
    m_out = s * (R @ L[:3]) + _skew(t) @ d_out
    return np.vstack([m_out, d_out])


def solve_rotation_global(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """
    Global minimisation of Σ arccos²(|R·d1_i · d2_i|) via differential evolution
    over the axis-angle ball of radius π (covers all of SO(3)).
    No warm start — pure global search followed by L-BFGS-B polish.
    """
    d1n = d1 / (np.linalg.norm(d1, axis=0, keepdims=True) + 1e-12)
    d2n = d2 / (np.linalg.norm(d2, axis=0, keepdims=True) + 1e-12)

    def cost(rotvec):
        R = ScipyRot.from_rotvec(rotvec).as_matrix()
        u = R @ d1n
        c = np.clip((u * d2n).sum(axis=0), -1 + 1e-9, 1 - 1e-9)
        return float(np.sum(np.arccos(np.abs(c)) ** 2))

    bounds = [(-np.pi, np.pi)] * 3
    result = differential_evolution(cost, bounds, seed=0, tol=1e-10, polish=True)
    return ScipyRot.from_rotvec(result.x).as_matrix()


def _minimal_sim3(L1: np.ndarray, L2: np.ndarray):
    """
    Minimal SIM(3) estimate from a small set of line correspondences.

    Rotation is solved by globally minimising the geodesic cost
    Σ arccos²(|R·d1_i · d2_i|) via differential evolution (no warm start).
    Then (t, s) are recovered via linear least squares given R.

    Returns (s, R, t) or None if degenerate.
    """
    R = solve_rotation_geodesic(L1[3:], L2[3:])
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
      2. Solve R by globally minimising the geodesic cost, then (t, s) via LS.
      3. Count G(2,4) inliers below inlier_threshold.
      4. Keep the hypothesis with the most inliers.

    Args:
        L1, L2:           (6, N) Plücker [m; d] source and target lines
        n_iter:           RANSAC iterations (default 5000)
        inlier_threshold: max principal angle threshold in radians ∈ [0, π/2]
        min_sample:       minimal sample size (default 3)
        seed:             RNG seed
        early_exit_iters: quit early if best inlier count is still 0

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
            result = _minimal_sim3(L1[:, idx], L2[:, idx])
            if result is None:
                continue
            s_cand, R_cand, t_cand = result
            t_cand = t_cand.flatten()
        except (np.linalg.LinAlgError, ValueError):
            continue

        mask, ic = _evaluate(R_cand, t_cand, s_cand)
        if ic > best_ic:
            best_ic, best_mask, best_R, best_t, best_s = ic, mask, R_cand, t_cand, s_cand

    return best_R, best_t, best_s, best_mask, best_ic
