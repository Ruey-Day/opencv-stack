"""
sim3_solver.py — THE shipped Sim(3) line-cloud registration solver, one file.

Self-contained (no imports from other solver modules). SHIPPED entry point is
`solve_sim3_unified` — a single-pass 3-line hemisphere RANSAC:

    from lib.sim3_solver import Sim3Solver, solve_sim3_unified
    solver = Sim3Solver((q1, q2), (r1, r2))                 # (N,3) endpoints
    R, t, s = solve_sim3_unified(solver, prob_matrix)       # top-200 matches

Endpoints are consumed once by segments_to_plucker(); every decision after that
is pure Plücker [m; d] (sign ambiguity [m;d] ~ [-m;-d]). Pipeline:

  0. scene pre-normalization — both clouds scaled so the reference median
     p0-radius (p0 = m x d) is 1 (balances the G(2,4) position/homogeneous split).
  1. ONE 3-line RANSAC — sample line triples from the top-k matches; per-line
     max-|comp| HEMISPHERE sign canon; batched SVD Procrustes rotation; batched
     joint (t,s) moment solve. 3 non-concurrent lines pin scale; hemisphere canon
     is direction-only, coplanarity-independent (beats position-winding).
  2. G(2,4) filter — score every full-Sim(3) hypothesis by whole-map affine-
     Grassmann agreement (angle ladder 1,2 deg); keep the argmax. The position-
     aware G(2,4) vetoes the direction-consistent facade flips L2/G(1,3) accept.
  3. strict G(2,4) inlier gate (6 deg, fixed) -> G(2,4)-MANIFOLD joint (R,t,s)
     refine (damped numerical GN, Cauchy kernel 2 deg; converges on relative gain).
     L2/moment refine is AVOIDED — it destabilises rotation under scale coupling.

No metric-scale prior, no PCM, no separate rotation stage. Real v19 matcher
(cached prob): 7-Scenes 18/37 @5deg, KITTI 2/2; ~130 ms/pair CPU.
"""
import numpy as np
import torch


def segments_to_plucker(p1, p2):
    d = p2 - p1
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    m = np.cross((p1 + p2) * 0.5, d)
    return np.concatenate([m, d], axis=1).astype(np.float32)


# ═══ rotation stage (formerly lib/rotation_stage.py, online variant) ═══


# ═══ Grassmannian single-rotation RANSAC (Shin et al.-inspired baseline) ══════
# Direction-only G(2,4) geodesic rotation fit inside a robust RANSAC — the
# deployable Grassmannian rotation baseline reported in the paper's Rotation
# Estimator Study. Faithful to Shin et al.'s optimize_R (sign-invariant Cauchy
# geodesic direction fit); their exact branch-and-bound does NOT transfer to
# noisy cross-modal correspondences (its 0.4deg inlier gate is degenerate on
# real matcher noise, and loosening it makes the BnB intractable), whereas the
# same criterion in a RANSAC scored at the empirical ~5deg noise scale matches
# our L2 candidate stage. Consolidated here from the former lib/grassmann_bnb.py.


# ── affine G(2,4) embedding + analytic 2x2 principal angles (position-aware) ──
# The affine Grassmannian embeds a line as a 2-plane of R^4 (normalized foot
# point [p0;1] and direction [d;0]); the max principal angle between two such
# planes requires BOTH direction and position to agree -- the criterion that
# breaks the direction-only false consensus.
def _yz_np(p0, d):                                # (...,3),(...,3) -> (...,4,2)
    one = np.ones_like(p0[..., :1]); zero = np.zeros_like(p0[..., :1])
    c0 = np.concatenate([p0, one], -1)
    c0 = c0 / (np.linalg.norm(c0, axis=-1, keepdims=True) + 1e-12)
    c1 = np.concatenate([d, zero], -1)
    c1 = c1 - c0 * (c1 * c0).sum(-1, keepdims=True)
    c1 = c1 / (np.linalg.norm(c1, axis=-1, keepdims=True) + 1e-12)
    return np.stack([c0, c1], -1)


def _s2_np(M):                                    # cos(MAX principal angle) of (...,2,2)
    F = (M ** 2).sum((-1, -2))
    det = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
    disc = np.sqrt(np.clip(F * F - 4 * det * det, 0.0, None))
    return np.sqrt(np.clip((F - disc) * 0.5, 0.0, 1.0))


class Sim3Solver:
    def __init__(self, q_ends, r_ends, prenorm_radius=1.0, device=None, **_ignored):
        self.dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        q1, q2 = (np.asarray(a, float) for a in q_ends)
        r1, r2 = (np.asarray(a, float) for a in r_ends)

        # ── 0. scene pre-normalization (PURE Plücker: positions anchored
        # at the foot-of-perpendicular p0 = m x d, never at endpoints) ────
        pr_raw = segments_to_plucker(r1, r2)
        p0_r_raw = np.cross(pr_raw[:, :3], pr_raw[:, 3:])
        rad = float(np.median(np.linalg.norm(
            p0_r_raw - np.median(p0_r_raw, axis=0), axis=1)))
        self.alpha = prenorm_radius / (rad + 1e-9)
        self.q1, self.q2 = q1 * self.alpha, q2 * self.alpha
        self.r1, self.r2 = r1 * self.alpha, r2 * self.alpha

        self.p_q = segments_to_plucker(self.q1, self.q2)
        self.p_r = segments_to_plucker(self.r1, self.r2)


def _skew(v: np.ndarray) -> np.ndarray:
    """3×3 skew-symmetric matrix for vector v (3,)."""
    return np.array([
        [ 0.0,  -v[2],  v[1]],
        [ v[2],   0.0, -v[0]],
        [-v[1],  v[0],  0.0],
    ])


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


# ════════════════════════════════════════════════════════════════════════════
# SHIPPED unified Sim(3) solver (2026-08-14). ONE 3-line hemisphere RANSAC ->
# G(2,4) filter -> strict G(2,4) inlier gate -> G(2,4)-manifold refine. Single
# pass, no separate rotation stage, no PCM, no prior. `solve_sim3_unified` is the
# deployable entry; it consumes a Sim3Solver's pre-normalised p_q/p_r/alpha.
# ════════════════════════════════════════════════════════════════════════════
def _maxabs_sign(D):                                       # hemisphere canon: max-|comp| positive
    j = np.abs(D).argmax(-1); return np.sign(np.take_along_axis(D, j[..., None], -1))[..., 0]

def _procrustes_batch(Dq, Dr):                             # (NS,L,3) matched dirs -> R (NS,3,3)
    U, _, Vt = np.linalg.svd(np.einsum('nli,nlj->nij', Dr, Dq))
    dt = np.linalg.det(U @ Vt); Dg = np.zeros((len(U), 3, 3)); Dg[:, 0, 0] = Dg[:, 1, 1] = 1; Dg[:, 2, 2] = dt
    return U @ Dg @ Vt

def _solve_ts_batch(Mq, Dq, Mr, Rs):                       # L-line joint (t,s) per hypothesis
    NS, L, _ = Mq.shape; Rd = np.einsum('nij,nlj->nli', Rs, Dq); Rm = np.einsum('nij,nlj->nli', Rs, Mq)
    A = np.zeros((NS, 3 * L, 4)); b = np.zeros((NS, 3 * L))
    for l in range(L):
        r = 3 * l; rd = Rd[:, l]
        A[:, r + 0, 1] = rd[:, 2]; A[:, r + 0, 2] = -rd[:, 1]; A[:, r + 1, 0] = -rd[:, 2]
        A[:, r + 1, 2] = rd[:, 0]; A[:, r + 2, 0] = rd[:, 1]; A[:, r + 2, 1] = -rd[:, 0]
        A[:, r:r + 3, 3] = Rm[:, l]; b[:, r:r + 3] = Mr[:, l]
    AtA = np.einsum('nij,nik->njk', A, A) + 1e-9 * np.eye(4)
    x = np.linalg.solve(AtA, np.einsum('nij,ni->nj', A, b)); return x[:, :3], x[:, 3]

def _g24_count_batch(dq, mq, dr, mr, Rs, ts, ss, cost):    # (H,) sum over ladder of G(2,4) inliers
    Yr = _yz_np(np.cross(mr.T, dr.T), dr.T); out = np.empty(len(Rs))
    for lo in range(0, len(Rs), 300):
        Rc, tc, sc = Rs[lo:lo + 300], ts[lo:lo + 300], ss[lo:lo + 300]
        d_t = np.einsum('hij,jn->hin', Rc, dq)
        m_t = sc[:, None, None] * np.einsum('hij,jn->hin', Rc, mq) + np.cross(tc[:, :, None], d_t, axis=1)
        Yq = _yz_np(np.transpose(np.cross(m_t, d_t, axis=1), (0, 2, 1)), np.transpose(d_t, (0, 2, 1)))
        cs = _s2_np(np.einsum('hkai,kaj->hkij', Yq, Yr))
        out[lo:lo + 300] = (cs[:, :, None] > cost[None, None, :]).sum(1).sum(1)
    return out

def _g24_cos_all(plq, plr, R, t, s):                       # cos(principal angle) per correspondence
    mq, dq, mr, dr = plq[:3], plq[3:], plr[:3], plr[3:]
    d_t = R @ dq; m_t = s * (R @ mq) + np.cross(t[None, :], d_t.T).T
    Yq = _yz_np(np.cross(m_t.T, d_t.T), d_t.T); Yr = _yz_np(np.cross(mr.T, dr.T), dr.T)
    return _s2_np(np.einsum('nak,nal->nkl', Yq, Yr))

def _refine_g24(plq, plr, idx, R, t, s, rb, n_iter=200, tol=1e-7):   # G(2,4)-manifold joint refine
    mq, dq, mr, dr = plq[:3, idx], plq[3:, idx], plr[:3, idx], plr[3:, idx]
    Yr = _yz_np(np.cross(mr.T, dr.T), dr.T)
    def g(R, t, s):
        d_t = R @ dq; m_t = s * (R @ mq) + np.cross(t[None, :], d_t.T).T
        Yq = _yz_np(np.cross(m_t.T, d_t.T), d_t.T); M = np.einsum('nak,nal->nkl', Yq, Yr)
        th = np.arccos(np.clip(_s2_np(M), -1, 1)); w = 1 / (1 + (th / rb) ** 2)
        return ((M ** 2).sum((1, 2)) * w).sum()
    eps = 1e-4; step = .05; cur = g(R, t, s)
    for _ in range(n_iter):
        gr = np.zeros(7)
        for k in range(3):
            w = np.zeros(3); w[k] = eps; gr[k] = (g(_expm_so3(w) @ R, t, s) - g(_expm_so3(-w) @ R, t, s)) / (2 * eps)
        for k in range(3):
            e = np.zeros(3); e[k] = eps; gr[3 + k] = (g(R, t + e, s) - g(R, t - e, s)) / (2 * eps)
        gr[6] = (g(R, t, s * np.exp(eps)) - g(R, t, s * np.exp(-eps))) / (2 * eps)
        if np.linalg.norm(gr) < 1e-9:
            break
        for _bt in range(20):
            Rn = _expm_so3(step * gr[:3]) @ R; tn = t + step * gr[3:6]; sn = s * np.exp(step * gr[6]); v = g(Rn, tn, sn)
            if v > cur:
                gain = v - cur; R, t, s, cur = Rn, tn, sn, v; step *= 1.3
                if gain < tol * max(cur, 1e-9):
                    return R, np.asarray(t), s
                break
            step *= .5
        else:
            break
    return R, np.asarray(t), s

def _expm_so3(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K

_SIGN_COMBOS = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1]], float)


def solve_sim3_unified(solver, prob, topk=200, nsamp=4000, taus=(1.0, 2.0),
                       refit_tau_deg=4.0, refine_rb_deg=1.0, refine_tol=1e-4, seed=0,
                       smin=0.05, smax=20.0, sign_search=True):
    """SHIPPED Sim(3) estimate from a matcher probability matrix. `solver` is a
    Sim3Solver holding the pre-normalised p_q/p_r/alpha. Returns (R, t, s) with t
    in the ORIGINAL (un-prenormalised) metric frame. One 3-line RANSAC
    → G(2,4) ladder filter → strict G(2,4) inlier gate → G(2,4)-manifold refine.

    sign_search=True (default, 2026-08-15): per-triple search over the 4 query
    sign combos, keeping the lowest Procrustes direction residual. The old
    per-map hemisphere canon (sign_search=False) is coordinate-frame dependent
    — its cross-map agreement decays ~90%→36% as the relative rotation grows
    to 180 deg, which manufactured ~176 deg flip-locks on scrambled inputs
    (scrambled 7-Scenes: 4→8 strict, rot<5 6→14 with the search; standard
    benchmarks neutral). Rotation is ~3% of runtime, so the 4x there is free.

    Defaults retuned 2026-08-14 on the regenerated benchmark
    (nsamp 3000→4000, refit_tau 6→4, refine_rb 2→1). smin/smax: candidate
    scale sanity clamp — pre-scale the query by the extent ratio (or widen
    smax) when true scales can exceed 20 (KITTI mono_best reaches ~39)."""
    cost = np.array([np.cos(np.deg2rad(a)) for a in taus])
    nq = prob.shape[1]; k = min(topk, prob.size)
    flat = np.argpartition(prob.ravel(), -k)[-k:]; ir = flat // nq; iq = flat % nq
    plq = solver.p_q[iq].T.astype(float); plr = solver.p_r[ir].T.astype(float)
    K = plq.shape[1]; rng = np.random.default_rng(seed)
    T = rng.integers(0, K, (nsamp, 3)); T = T[(T[:, 0] != T[:, 1]) & (T[:, 0] != T[:, 2]) & (T[:, 1] != T[:, 2])]
    if len(T) == 0:
        return np.eye(3), np.zeros(3), 1.0
    Dq = np.stack([plq[3:, T[:, l]].T for l in range(3)], 1); Dr = np.stack([plr[3:, T[:, l]].T for l in range(3)], 1)
    Mq = np.stack([plq[:3, T[:, l]].T for l in range(3)], 1); Mr = np.stack([plr[:3, T[:, l]].T for l in range(3)], 1)
    sr = _maxabs_sign(Dr)
    Drc, Mrc = Dr * sr[..., None], Mr * sr[..., None]
    if sign_search:
        best_res = np.full(len(T), np.inf)
        Rs = np.zeros((len(T), 3, 3)); sq = np.ones((len(T), 3))
        for combo in _SIGN_COMBOS:
            Dqs = Dq * combo[None, :, None]
            Rc = _procrustes_batch(Dqs, Drc)
            res = ((np.einsum('nij,nlj->nli', Rc, Dqs) - Drc) ** 2).sum((1, 2))
            m = res < best_res
            best_res[m] = res[m]; Rs[m] = Rc[m]; sq[m] = combo
    else:
        sq = _maxabs_sign(Dq)
        Rs = _procrustes_batch(Dq * sq[..., None], Drc)
    ts, ss = _solve_ts_batch(Mq * sq[..., None], Dq * sq[..., None], Mrc, Rs)
    ok = (ss > smin) & (ss < smax)
    if not ok.any():
        return np.eye(3), np.zeros(3), 1.0
    Rs, ts, ss = Rs[ok], ts[ok], ss[ok]
    sc = _g24_count_batch(plq[3:], plq[:3], plr[3:], plr[:3], Rs, ts, ss, cost)
    bi = int(np.argmax(sc)); R, t, s = Rs[bi], ts[bi], float(ss[bi])
    if refit_tau_deg > 0:                                  # strict G(2,4) inlier gate → refine
        cs = _g24_cos_all(plq, plr, R, np.asarray(t), s)
        idx = np.where(cs > np.cos(np.deg2rad(refit_tau_deg)))[0]
        if len(idx) < 6:
            idx = np.argpartition(cs, -min(20, len(cs)))[-min(20, len(cs)):]
        R, t, s = _refine_g24(plq, plr, idx, R.copy(), np.asarray(t).copy(), s,
                              np.deg2rad(refine_rb_deg), tol=refine_tol)
    return R, np.asarray(t) / solver.alpha, s

