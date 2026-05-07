"""
Sim(3) RANSAC for Plücker line correspondences.

Replaces the SE(3) solver in the original PlueckerNet (lib/ransac_l2l.py).

Key change:  after estimating R from direction pairs (same SVD as SE(3)),
we jointly solve for scale s and translation t from the moment equations:

    m2_i = s · R m1_i  +  t × d2_i

Rewritten as a 3n × 4 linear system (per inlier pair i):

    [ R m1_i | −[d2_i×] ]  [s  ]  =  m2_i
                            [t  ]

→ least-squares solution for (s, t).
"""
import numpy as np


def _p6(plucker):
    """Return the first 6 rows of a Plücker array (strips RGB if 9D)."""
    return plucker[:6] if plucker.shape[0] > 6 else plucker


def skew(x):
    x = x.flatten()
    return np.array([[ 0,    -x[2],  x[1]],
                     [ x[2],  0,    -x[0]],
                     [-x[1],  x[0],  0   ]], dtype=np.float32)


def estimate_rotation(d1, d2):
    """Estimate R from direction correspondences via SVD.

    Args:
        d1, d2: (3, n) unit direction arrays
    Returns:
        R: (3, 3) rotation matrix (det = +1)
    """
    M = d2 @ d1.T          # (3, 3)
    U, _, Vh = np.linalg.svd(M)
    R = U @ Vh
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vh
    return R.astype(np.float32)


def solve_scale_translation(m1p, m2, d2):
    """Solve for (s, t) given R-rotated source moments and target moments.

    Args:
        m1p: (n, 3)  R · m1  for each line
        m2:  (n, 3)  target moments
        d2:  (n, 3)  target directions
    Returns:
        s_est: float
        t_est: (3, 1) float32
    """
    n = m1p.shape[0]
    A = np.zeros((3 * n, 4), dtype=np.float32)
    b = np.zeros(3 * n,      dtype=np.float32)

    for i in range(n):
        row = 3 * i
        A[row:row+3, 0]  =  m1p[i]        # coefficient of s
        A[row:row+3, 1:] = -skew(d2[i])   # coefficient of t  (t×d = −[d×]t)
        b[row:row+3]     =  m2[i]

    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return float(result[0]), result[1:].reshape(3, 1).astype(np.float32)


def model_estimate_sim3(plucker1, plucker2):
    """Minimal Sim(3) solver from 2 line correspondences.

    Args:
        plucker1, plucker2: (6, 2) arrays  [m0 m1 m2 d0 d1 d2] per column
    Returns:
        (s, R, t) or None if degenerate (s ≤ 0)
    """
    plucker1, plucker2 = _p6(plucker1), _p6(plucker2)

    d1 = plucker1[3:, :]   # (3, 2)
    d2 = plucker2[3:, :]   # (3, 2)
    R = estimate_rotation(d1, d2)

    m1p = (R @ plucker1[:3, :]).T   # (2, 3)
    m2  = plucker2[:3, :].T         # (2, 3)
    d2T = d2.T                      # (2, 3)

    s, t = solve_scale_translation(m1p, m2, d2T)
    if s <= 0:
        return None
    return s, R, t


def sim3_motion_matrix(s, R, t):
    """Build 6×6 Sim(3) line motion matrix.

    L2 = M_sim3 @ L1,  where L = [m; d].

    M_sim3 = [ s·R   [t×]·R ]
             [  0       R   ]
    """
    M = np.zeros((6, 6), dtype=np.float32)
    M[:3, :3] = s * R
    M[:3, 3:] = skew(t) @ R
    M[3:, 3:] = R
    return M


def score_sim3(plucker1, plucker2, s, R, t, threshold):
    """Return boolean inlier mask for a Sim(3) hypothesis.

    Args:
        plucker1, plucker2: (6+, n) arrays (9D if colors present)
        threshold: L2 residual cutoff on the 6D Plücker vector (ignores color)
    """
    M = sim3_motion_matrix(s, R, t)
    p1_6d, p2_6d = _p6(plucker1), _p6(plucker2)
    residual = np.linalg.norm(p2_6d - M @ p1_6d, axis=0)
    return residual < threshold


def best_fit_sim3(plucker1, plucker2):
    """Refined Sim(3) from all inlier correspondences (overdetermined LS).

    Args:
        plucker1, plucker2: (6+, n) inlier arrays (9D if colors present)
    Returns:
        (s, R, t)
    """
    plucker1, plucker2 = _p6(plucker1), _p6(plucker2)

    R = estimate_rotation(plucker1[3:, :], plucker2[3:, :])
    m1p = (R @ plucker1[:3, :]).T
    m2  = plucker2[:3, :].T
    d2  = plucker2[3:, :].T
    s, t = solve_scale_translation(m1p, m2, d2)
    return s, R, t


def run_ransac_sim3(plucker1, plucker2,
                    max_iterations=200,
                    inlier_threshold=0.05):
    """Sim(3) RANSAC on Plücker line correspondences.

    Args:
        plucker1, plucker2:  (6, n) arrays
        max_iterations:      number of RANSAC hypotheses
        inlier_threshold:    L2 residual threshold (in the 6-vector Plücker space)

    Returns:
        s_est, R_est, t_est, best_ic, best_ic_mask
        All None / 0 / None on failure.
    """
    _, N = plucker1.shape
    if N < 2:
        return None, None, None, 0, None

    best_ic       = 0
    best_ic_mask  = None
    best_s, best_R, best_t = None, None, None

    for _ in range(max_iterations):
        inds = np.random.choice(N, 2, replace=False)
        result = model_estimate_sim3(plucker1[:, inds], plucker2[:, inds])
        if result is None:
            continue
        s_h, R_h, t_h = result

        mask = score_sim3(plucker1, plucker2, s_h, R_h, t_h, inlier_threshold)
        ic   = int(np.sum(mask))

        if ic > best_ic:
            best_ic      = ic
            best_ic_mask = mask
            best_s, best_R, best_t = s_h, R_h, t_h

    # Refine on full inlier set
    if best_ic_mask is not None and best_ic > 1:
        best_s, best_R, best_t = best_fit_sim3(
            plucker1[:, best_ic_mask],
            plucker2[:, best_ic_mask],
        )
        if best_s <= 0:
            return None, None, None, 0, None

    return best_s, best_R, best_t, best_ic, best_ic_mask
