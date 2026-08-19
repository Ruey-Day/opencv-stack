"""sim3_solver.py — SCALAR's Sim(3) line-map registration solver, one file.

SHIPPED (2026-08-19): the Grass 2-LINE pipeline, the only solver.

    from lib.sim3_solver import Sim3Solver, solve_sim3
    solver = Sim3Solver((q1, q2), (r1, r2))          # (N,3) endpoints each
    R, t, s = solve_sim3(solver, prob_matrix)        # matcher prob (n_ref,n_q)

Lines are Plücker [m; d] (sign ambiguity [m;d] ~ [-m;-d]); positions are always
foot points p0 = d x m (sign-invariant). Pipeline, every stage ablation-tested:

  0. prenorm: both clouds scaled so the reference median foot radius is 1 —
     all thresholds below are calibrated in these units (ablation: removing it
     costs 1 pair indoors, 2/5 outdoors).
  1. top-K matcher pairs (K=200) -> nsamp random 2-line samples.
  2. SKEW OBSERVABILITY GATE |(d1xd2).(f2-f1)| > 0.2 on both clouds — two
     lines fix Sim(3) only if skew; near-coplanar pairs are scale-degenerate
     yet self-consistent, so no residual test can catch them (+1 pair AND
     ~2x faster: rejected samples cost nothing).
  3. minimal solve: hemisphere-canon SVD Procrustes (directions) + joint 4x4
     linear (t,s) from the moment constraint. Optional: `iters` runs the
     closed-form alternating descent on the Grassmann projection cost
     (`_alternate`, sign-invariant, also the E2E training head) — benchmark
     accuracy is identical at 0 iterations, so the default skips it.
  4. CONSISTENCY FILTER: the projection cost of the fitted pair (8 constraints
     vs 7 DoF -> 1 redundant: contaminated pairs cannot fit, cost stays high;
     pure pairs fit to ~0). cost < 0.12, plus the (smin, smax) scale clamp.
  5. PROP G(2,4) SELECTION: every hypothesis scored over all K pairs by
     sum(max(0, 1 - theta/tau)) where theta is the max principal angle between
     the affine-Grassmann embeddings — position-aware (vetoes direction-
     consistent flips), graded (precision outranks equal loose counts; a plain
     hard count is brittle in tau), one parameter. tau=3 deg indoors; ~9 deg
     for high-noise/outdoor regimes.
  6. strict G(2,4) gate (4 deg) -> Cauchy-robust manifold refine of (R,t,s)
     (finite-difference ascent; essential outdoors: 3/5 -> 1/5 without).

v27 caches, CPU: 7-Scenes 21/37 @5 (med 3.84 deg / 4.0% / 0.121 m) at ~70 ms;
KITTI mono_best 3/5 (tau=9, extent prenorm) at ~100 ms; submaps matcher-bound.

`solve_sim3_unified` is kept as a compatibility alias: legacy kwargs whose
semantics changed (taus ladder, sign_search, refine_tol) are accepted and
ignored; it runs THIS solver.
"""
import numpy as np

# ── Plücker + scene prenorm ─────────────────────────────────────────────────


def segments_to_plucker(p1, p2):
    d = p2 - p1
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    m = np.cross((p1 + p2) * 0.5, d)
    return np.concatenate([m, d], axis=1).astype(np.float32)


class Sim3Solver:
    """Scene pre-normalization + Plücker conversion. alpha scales both clouds
    so the reference median foot-point radius is 1; solve_sim3 returns t in
    the original metric frame (divides by alpha)."""

    def __init__(self, q_ends, r_ends, prenorm_radius=1.0, **_ignored):
        q1, q2 = (np.asarray(a, float) for a in q_ends)
        r1, r2 = (np.asarray(a, float) for a in r_ends)
        pr_raw = segments_to_plucker(r1, r2)
        p0 = np.cross(pr_raw[:, :3], pr_raw[:, 3:])
        rad = float(np.median(np.linalg.norm(p0 - np.median(p0, 0), axis=1)))
        self.alpha = prenorm_radius / (rad + 1e-9)
        self.q1, self.q2 = q1 * self.alpha, q2 * self.alpha
        self.r1, self.r2 = r1 * self.alpha, r2 * self.alpha
        self.p_q = segments_to_plucker(self.q1, self.q2)
        self.p_r = segments_to_plucker(self.r1, self.r2)


def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def solve_translation_scale(L1, L2, R):
    """Linear 3Nx4 LS for (t, s) from m2 = s R m1 + t x (R d1). L*: (6, N)."""
    m1, d1, m2 = L1[:3], L1[3:], L2[:3]
    N = L1.shape[1]
    Rm1, Rd1 = R @ m1, R @ d1
    A = np.zeros((3 * N, 4)); b = np.zeros(3 * N)
    for i in range(N):
        r = 3 * i
        A[r:r + 3, :3] = -_skew(Rd1[:, i])
        A[r:r + 3, 3] = Rm1[:, i]
        b[r:r + 3] = m2[:, i]
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return x[:3], float(x[3])


def transform_lines(L, R, t, s):
    """Apply Sim(3): d' = R d,  m' = s R m + t x d'. L: (6, N)."""
    d_out = R @ L[3:]
    m_out = s * (R @ L[:3]) + _skew(t) @ d_out
    return np.vstack([m_out, d_out])


def _feet(d, m):
    """Foot points d x m — invariant to the Plücker sign."""
    return np.cross(d, m)


# ── G(2,4) primitives: affine-Grassmann embedding + principal angle ─────────


def _yz_np(p0, d):                                # (...,3),(...,3) -> (...,4,2)
    one = np.ones_like(p0[..., :1]); zero = np.zeros_like(p0[..., :1])
    c0 = np.concatenate([p0, one], -1)
    c0 = c0 / (np.linalg.norm(c0, axis=-1, keepdims=True) + 1e-12)
    c1 = np.concatenate([d, zero], -1)
    c1 = c1 - c0 * (c1 * c0).sum(-1, keepdims=True)
    c1 = c1 / (np.linalg.norm(c1, axis=-1, keepdims=True) + 1e-12)
    return np.stack([c0, c1], -1)


def _s2_np(M):                                    # cos(MAX principal angle)
    F = (M ** 2).sum((-1, -2))
    det = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
    disc = np.sqrt(np.clip(F * F - 4 * det * det, 0.0, None))
    return np.sqrt(np.clip((F - disc) * 0.5, 0.0, 1.0))


# ── batched minimal-solve building blocks ───────────────────────────────────


def _maxabs_sign(D):                              # hemisphere canon per line
    j = np.abs(D).argmax(-1)
    return np.sign(np.take_along_axis(D, j[..., None], -1))[..., 0]


def _procrustes_batch(Dq, Dr):                    # (NS,L,3) dirs -> R (NS,3,3)
    U, _, Vt = np.linalg.svd(np.einsum('nli,nlj->nij', Dr, Dq))
    dt = np.linalg.det(U @ Vt)
    Dg = np.zeros((len(U), 3, 3)); Dg[:, 0, 0] = Dg[:, 1, 1] = 1; Dg[:, 2, 2] = dt
    return U @ Dg @ Vt


def _solve_ts_batch(Mq, Dq, Mr, Rs):              # joint (t,s) per hypothesis
    NS, L, _ = Mq.shape
    Rd = np.einsum('nij,nlj->nli', Rs, Dq); Rm = np.einsum('nij,nlj->nli', Rs, Mq)
    A = np.zeros((NS, 3 * L, 4)); b = np.zeros((NS, 3 * L))
    for l in range(L):
        r = 3 * l; rd = Rd[:, l]
        A[:, r + 0, 1] = rd[:, 2]; A[:, r + 0, 2] = -rd[:, 1]
        A[:, r + 1, 0] = -rd[:, 2]; A[:, r + 1, 2] = rd[:, 0]
        A[:, r + 2, 0] = rd[:, 1]; A[:, r + 2, 1] = -rd[:, 0]
        A[:, r:r + 3, 3] = Rm[:, l]; b[:, r:r + 3] = Mr[:, l]
    AtA = np.einsum('nij,nik->njk', A, A) + 1e-9 * np.eye(4)
    x = np.linalg.solve(AtA, np.einsum('nij,ni->nj', A, b))
    return x[:, :3], x[:, 3]


# ── closed-form alternating descent on the Grassmann projection cost ────────
# (derivation note; sign-invariant; per-line weights w keep both blocks
# closed-form — this is also the differentiable E2E training head)


def _alternate(dq, fq, dr, fr, R, t, s, n_iter, w=None):
    """dq,fq,dr,fr: (NS,L,3); R:(NS,3,3); t:(NS,3); s:(NS,). Returns updated
    R,t,s and the per-sample projection cost (used as the consistency filter;
    at n_iter=0 it is the cost of the init)."""
    if w is None:
        w = np.ones(dq.shape[:2])
    eta = 1.0 / np.sqrt(1.0 + (fr ** 2).sum(-1))
    y = eta[..., None] * fr
    for _ in range(n_iter):
        u = np.einsum('nij,nlj->nli', R, dq)
        q = s[:, None, None] * np.einsum('nij,nlj->nli', R, fq) + t[:, None, :]
        lam = (u * dr).sum(-1) / s[:, None]
        qp = q - u * (u * q).sum(-1, keepdims=True)
        yp = y - u * (u * y).sum(-1, keepdims=True)
        gam = ((qp * yp).sum(-1) + eta) / ((qp ** 2).sum(-1) + 1.0)
        alp = (u * (y - gam[..., None] * q)).sum(-1) / s[:, None]
        x_dir = dq * lam[..., None]
        x_aff = dq * alp[..., None] + gam[..., None] * fq
        wg = w * gam
        H = (wg * gam).sum(1)
        Hs = np.where(H < 1e-12, 1.0, H)
        mux = (wg[..., None] * x_aff).sum(1) / Hs[:, None]
        muy = (wg[..., None] * y).sum(1) / Hs[:, None]
        xa = x_aff - gam[..., None] * mux[:, None, :]
        ya = y - gam[..., None] * muy[:, None, :]
        M = np.einsum('nl,nli,nlj->nij', w, dr, x_dir) \
            + np.einsum('nl,nli,nlj->nij', w, ya, xa)
        U, S_, Vt = np.linalg.svd(M)
        det = np.linalg.det(U @ Vt)
        D = np.zeros_like(M); D[:, 0, 0] = D[:, 1, 1] = 1; D[:, 2, 2] = det
        Rn = U @ D @ Vt
        Sx = (w * (x_dir ** 2).sum(-1)).sum(1) + (w * (xa ** 2).sum(-1)).sum(1)
        sn = np.clip((S_[:, 0] + S_[:, 1] + det * S_[:, 2]) /
                     np.where(Sx < 1e-12, 1.0, Sx), 1e-3, 1e3)
        tn = muy - sn[:, None] * np.einsum('nij,nj->ni', Rn, mux)
        keep = ~((H < 1e-12) | (Sx < 1e-12) | ~np.isfinite(sn))
        R = np.where(keep[:, None, None], Rn, R)
        s = np.where(keep, sn, s)
        t = np.where(keep[:, None], tn, t)
    u = np.einsum('nij,nlj->nli', R, dq)
    q = s[:, None, None] * np.einsum('nij,nlj->nli', R, fq) + t[:, None, :]
    cdir = (1.0 - (u * dr).sum(-1) ** 2).sum(1)
    qp = q - u * (u * q).sum(-1, keepdims=True)
    yp = y - u * (u * y).sum(-1, keepdims=True)
    gam = ((qp * yp).sum(-1) + eta) / ((qp ** 2).sum(-1) + 1.0)
    caff = ((yp - gam[..., None] * qp) ** 2).sum((1, 2)) + ((eta - gam) ** 2).sum(1)
    return R, t, s, cdir + caff


# ── hypothesis scoring + refine ─────────────────────────────────────────────


def _score_prop(plq, plr, Rs, ts, ss, tau_rad, chunk=1024):
    """Per-hypothesis score sum(max(0, 1 - theta/tau)) over all candidate
    pairs; theta = max principal angle of the G(2,4) embeddings."""
    mq, dq, mr, dr = plq[:3], plq[3:], plr[:3], plr[3:]
    Yr = _yz_np(np.cross(mr.T, dr.T), dr.T)
    out = np.empty(len(Rs))
    for lo in range(0, len(Rs), chunk):
        Rc, tc, sc = Rs[lo:lo + chunk], ts[lo:lo + chunk], ss[lo:lo + chunk]
        d_t = np.einsum('hij,jn->hin', Rc, dq)
        m_t = sc[:, None, None] * np.einsum('hij,jn->hin', Rc, mq) \
            + np.cross(tc[:, :, None], d_t, axis=1)
        Yq = _yz_np(np.transpose(np.cross(m_t, d_t, axis=1), (0, 2, 1)),
                    np.transpose(d_t, (0, 2, 1)))
        th = np.arccos(np.clip(_s2_np(np.einsum('hkai,kaj->hkij', Yq, Yr)),
                               -1.0, 1.0))
        out[lo:lo + chunk] = np.clip(1.0 - th / tau_rad, 0.0, None).sum(1)
    return out


def _g24_count_batch(dq, mq, dr, mr, Rs, ts, ss, cost):
    """LEGACY primitive (pre-2026-08-19 ladder counting) kept for the
    experiment harness; the shipped scorer is _score_prop."""
    Yr = _yz_np(np.cross(mr.T, dr.T), dr.T)
    out = np.empty(len(Rs))
    for lo in range(0, len(Rs), 1024):
        Rc, tc, sc = Rs[lo:lo + 1024], ts[lo:lo + 1024], ss[lo:lo + 1024]
        d_t = np.einsum('hij,jn->hin', Rc, dq)
        m_t = sc[:, None, None] * np.einsum('hij,jn->hin', Rc, mq) \
            + np.cross(tc[:, :, None], d_t, axis=1)
        Yq = _yz_np(np.transpose(np.cross(m_t, d_t, axis=1), (0, 2, 1)),
                    np.transpose(d_t, (0, 2, 1)))
        cs = _s2_np(np.einsum('hkai,kaj->hkij', Yq, Yr))
        out[lo:lo + 1024] = (cs[:, :, None] > cost[None, None, :]).sum(1).sum(1)
    return out


def _g24_cos_all(plq, plr, R, t, s):
    mq, dq, mr, dr = plq[:3], plq[3:], plr[:3], plr[3:]
    d_t = R @ dq; m_t = s * (R @ mq) + np.cross(t[None, :], d_t.T).T
    Yq = _yz_np(np.cross(m_t.T, d_t.T), d_t.T)
    Yr = _yz_np(np.cross(mr.T, dr.T), dr.T)
    return _s2_np(np.einsum('nak,nal->nkl', Yq, Yr))


def _expm_so3(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _refine_g24(plq, plr, idx, R, t, s, rb, n_iter=200, tol=1e-4):
    """Cauchy-robust joint (R,t,s) ascent of the G(2,4) agreement over the
    strict inlier pool (finite differences + backtracking). Essential in
    high-noise regimes; the plain/IRLS alternation refine is within ~1 pair
    but this keeps the edge (measured 2026-08-18)."""
    mq, dq, mr, dr = plq[:3, idx], plq[3:, idx], plr[:3, idx], plr[3:, idx]
    Yr = _yz_np(np.cross(mr.T, dr.T), dr.T)

    def g(R, t, s):
        d_t = R @ dq; m_t = s * (R @ mq) + np.cross(t[None, :], d_t.T).T
        Yq = _yz_np(np.cross(m_t.T, d_t.T), d_t.T)
        M = np.einsum('nak,nal->nkl', Yq, Yr)
        th = np.arccos(np.clip(_s2_np(M), -1, 1))
        return ((M ** 2).sum((1, 2)) / (1 + (th / rb) ** 2)).sum()

    eps = 1e-4; step = .05; cur = g(R, t, s)
    for _ in range(n_iter):
        gr = np.zeros(7)
        for k in range(3):
            w = np.zeros(3); w[k] = eps
            gr[k] = (g(_expm_so3(w) @ R, t, s) - g(_expm_so3(-w) @ R, t, s)) / (2 * eps)
        for k in range(3):
            e = np.zeros(3); e[k] = eps
            gr[3 + k] = (g(R, t + e, s) - g(R, t - e, s)) / (2 * eps)
        gr[6] = (g(R, t, s * np.exp(eps)) - g(R, t, s * np.exp(-eps))) / (2 * eps)
        if np.linalg.norm(gr) < 1e-9:
            break
        for _bt in range(20):
            Rn = _expm_so3(step * gr[:3]) @ R
            tn = t + step * gr[3:6]; sn = s * np.exp(step * gr[6])
            v = g(Rn, tn, sn)
            if v > cur:
                gain = v - cur; R, t, s, cur = Rn, tn, sn, v; step *= 1.3
                if gain < tol * max(cur, 1e-9):
                    return R, np.asarray(t), s
                break
            step *= .5
        else:
            break
    return R, np.asarray(t), s


# ── THE solver ──────────────────────────────────────────────────────────────


def solve_sim3(solver, prob, topk=200, nsamp=4000, tau_deg=3.0,
               refit_tau_deg=4.0, refine_rb_deg=1.0, skew_min=0.2,
               cost_max=0.12, iters=(0, 0), cost_phase1=1.5,
               smin=0.05, smax=20.0, seed=0):
    """SHIPPED Sim(3) estimate from a matcher probability matrix (see module
    docstring for the pipeline). Returns (R, t, s), t in the original metric
    frame. tau_deg: prop-score cutoff — 3 (indoor default) to ~9 (high-noise/
    outdoor). smin/smax: scale sanity clamp — widen (or extent-prenormalize
    the query) when true scales can exceed 20."""
    nq = prob.shape[1]; k = min(topk, prob.size)
    flat = np.argpartition(prob.ravel(), -k)[-k:]
    ir, iq = flat // nq, flat % nq
    plq = solver.p_q[iq].T.astype(float); plr = solver.p_r[ir].T.astype(float)
    K = plq.shape[1]
    rng = np.random.default_rng(seed)
    T = rng.integers(0, K, (nsamp, 2))
    T = T[T[:, 0] != T[:, 1]]
    if skew_min > 0:                                 # observability gate FIRST
        ok = np.ones(len(T), bool)
        for pl in (plq, plr):
            a, b = pl[3:, T[:, 0]].T, pl[3:, T[:, 1]].T
            fa = _feet(a, pl[:3, T[:, 0]].T); fb = _feet(b, pl[:3, T[:, 1]].T)
            ok &= np.abs((np.cross(a, b) * (fb - fa)).sum(1)) > skew_min
        T = T[ok]
    if len(T) == 0:
        return np.eye(3), np.zeros(3), 1.0
    dq = np.stack([plq[3:, T[:, l]].T for l in range(2)], 1)
    dr = np.stack([plr[3:, T[:, l]].T for l in range(2)], 1)
    mq = np.stack([plq[:3, T[:, l]].T for l in range(2)], 1)
    mr = np.stack([plr[:3, T[:, l]].T for l in range(2)], 1)
    fq, fr = _feet(dq, mq), _feet(dr, mr)
    sgq, sgr = _maxabs_sign(dq), _maxabs_sign(dr)    # minimal solve (init)
    dqc, mqc = dq * sgq[..., None], mq * sgq[..., None]
    drc, mrc = dr * sgr[..., None], mr * sgr[..., None]
    Rs = _procrustes_batch(dqc, drc)
    ts, ss = _solve_ts_batch(mqc, dqc, mrc, Rs)
    ss = np.clip(np.abs(ss), 1e-3, 1e3)
    Rs, ts, ss, cost = _alternate(dq, fq, dr, fr, Rs, ts, ss, iters[0])
    if iters[1] > 0:                                 # optional polish phase
        keep = np.isfinite(cost) & (cost < cost_phase1)
        Rs, ts, ss = Rs[keep], ts[keep], ss[keep]
        dq, fq, dr, fr = dq[keep], fq[keep], dr[keep], fr[keep]
        if len(Rs) == 0:
            return np.eye(3), np.zeros(3), 1.0
        Rs, ts, ss, cost = _alternate(dq, fq, dr, fr, Rs, ts, ss, iters[1])
    keep = np.isfinite(cost) & (ss > smin) & (ss < smax)   # consistency filter
    if cost_max > 0:
        keep &= cost < cost_max
    Rs, ts, ss = Rs[keep], ts[keep], ss[keep]
    if len(Rs) == 0:
        return np.eye(3), np.zeros(3), 1.0
    sc = _score_prop(plq, plr, Rs, ts, ss, np.deg2rad(tau_deg))
    bi = int(np.argmax(sc))
    R, t, s = Rs[bi], ts[bi], float(ss[bi])
    if refit_tau_deg > 0:                            # strict gate -> refine
        cs = _g24_cos_all(plq, plr, R, np.asarray(t), s)
        idx = np.where(cs > np.cos(np.deg2rad(refit_tau_deg)))[0]
        if len(idx) < 6:
            k2 = min(20, len(cs)); idx = np.argpartition(cs, -k2)[-k2:]
        R, t, s = _refine_g24(plq, plr, idx, R.copy(), np.asarray(t).copy(), s,
                              np.deg2rad(refine_rb_deg))
    return R, np.asarray(t) / solver.alpha, s


def solve_sim3_unified(solver, prob, topk=200, nsamp=4000, refit_tau_deg=4.0,
                       refine_rb_deg=1.0, smin=0.05, smax=20.0, seed=0,
                       **_legacy_ignored):
    """Compatibility alias for pre-2026-08-19 callers: runs THE shipped
    solver. Legacy kwargs with changed semantics (taus ladder, sign_search,
    refine_tol, ...) are accepted and ignored."""
    return solve_sim3(solver, prob, topk=topk, nsamp=nsamp,
                      refit_tau_deg=refit_tau_deg, refine_rb_deg=refine_rb_deg,
                      smin=smin, smax=smax, seed=seed)
