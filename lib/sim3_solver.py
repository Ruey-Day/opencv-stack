"""
sim3_solver.py — THE Sim(3) line-cloud registration solver, one file.

Self-contained: no imports from other solver modules (the legacy
ModeSolver / scipy rotation variants live in attic/solver_modes.py and
attic/rotation_stage.py; lib/ransac_grassmannian.py remains only as the
max-inlier RANSAC *baseline* for ablations and trainer validation).

    from lib.sim3_solver import Sim3Solver
    solver = Sim3Solver((q1, q2), (r1, r2))            # (N,3) endpoints
    s, R, t, info = solver.register(prob=prob_matrix)  # prob optional

Endpoints are consumed once by segments_to_plucker(); every solver
decision after that is pure Plücker [m; d]. Shipped pipeline (bare
register() defaults = matcher-driven, no arbitration; on the 37-pair
seq01-ref 7-Scenes benchmark 2026-07-16: 14/37 @5°, 19/37 @10°, med
5.4%/5.3°/0.35 m, ~0.3 s; with arbitrate=True: 14/37 @5°, 22/37 @10°,
med 3.6%/5.1°/0.22 m, ~0.4 s, and rescues the KITTI 07 180° flip.
NOTE: outcomes are sensitive to sub-ULP GPU numerics in the matcher
forward pass — certify A/B claims in the SAME session):

  0. scene pre-normalization — both clouds scaled so the reference median
     p0-radius (p0 = m x d) is 2.5 m; extent-ratio scale prior s0 with
     band s0*[1/1.8, 1.8] (without it: s->0 collapse attractor).
  1. rotation candidates — 6k batched 2-pair Procrustes hypotheses + 8k
     uniform SO(3) grid, scored under a matcher-weighted pair objective
     AND a correspondence-free direction-distribution objective (hinge
     kernels); top-8 peaks each, ~14 after dedup. Never top-1: Manhattan
     flips dominate every direction-only objective.
  2. PCM candidate filter — soft pairwise-consistency degree (angle +
     distance-ratio invariants, Gaussian kernels around the graph-median
     pairwise scale); filters the (t, s) proposal pool ONLY (cannot
     arbitrate flips — angle relations are rotation-invariant).
  3. (t, s) proposals — ALL 2-pair line-moment solves among direction-
     consistent surviving candidates, one batched normal-equations pass;
     wide dedup (no inlier-count shortlist); + scale-sweep fallback.
  4. selection — every pose scored in one fused GPU pass: length-weighted
     moment-residual verification against the FULL maps (tau_m
     0.1/0.2/0.4, 20° gate) x log-normal scale prior.
  5. guarded final refit — ONE fixed-association LS solve (direction
     Procrustes + joint (t,s) moment LS) over the winning pose's
     verified inliers, kept only if the verification score does not
     drop. NOT iterative: re-associated ICP drops success 18->11/35
     at ~100x cost (refine=True ablation).
"""
import numpy as np
import torch


def segments_to_plucker(p1, p2):
    d = p2 - p1
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    m = np.cross((p1 + p2) * 0.5, d)
    return np.concatenate([m, d], axis=1).astype(np.float32)


# ═══ rotation stage (formerly lib/rotation_stage.py, online variant) ═══


def _rot_angle_deg(Ra, Rb):
    c = (np.trace(Ra @ Rb.T) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def super_fibonacci(n):
    """Uniform-ish SO(3) quaternion sampling (Alexa, CVPR 2022)."""
    phi = np.sqrt(2.0)
    psi = 1.533751168755204288118041
    s = np.arange(n, dtype=np.float64) + 0.5
    r = np.sqrt(s / n)
    R_ = np.sqrt(1.0 - s / n)
    alpha = 2 * np.pi * s / phi
    beta = 2 * np.pi * s / psi
    return np.stack([r * np.sin(alpha), r * np.cos(alpha),
                     R_ * np.sin(beta), R_ * np.cos(beta)], axis=1)


def quat_to_mat(q):
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def axis_angle_mats(deltas_deg):
    """Identity + small rotations about 13 axes for local grid refinement."""
    axes = []
    for a in ([1, 0, 0], [0, 1, 0], [0, 0, 1],
              [1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1],
              [0, 1, 1], [0, 1, -1],
              [1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]):
        v = np.asarray(a, float)
        axes.append(v / np.linalg.norm(v))
    mats = [np.eye(3)]
    for d in deltas_deg:
        ang = np.deg2rad(d)
        for ax in axes:
            for sgn in (1, -1):
                c, s = np.cos(sgn * ang), np.sin(sgn * ang)
                x, y, z = ax
                Kx = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                mats.append(np.eye(3) + s * Kx + (1 - c) * (Kx @ Kx))
    return np.stack(mats)


def _fib_hemisphere(n):
    i = np.arange(2 * n) + 0.5
    phi = np.arccos(1 - 2 * i / (2 * n))
    theta = np.pi * (1 + 5 ** 0.5) * i
    g = np.stack([np.cos(theta) * np.sin(phi),
                  np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)
    return g[g[:, 2] >= 0]                            # ~n upper-hemisphere pts


def _cluster_dirs(d, w, n_bins=64):
    """Quantize antipodal directions into weighted bins. d: (3,N), w: (N,).
    Returns centers (3,C), weights (C,)."""
    g = _fib_hemisphere(n_bins)                       # (C~n_bins, 3)
    a = g @ d                                         # (C, N) signed dots
    bins = np.argmax(np.abs(a), axis=0)               # nearest bin per dir
    sgn = np.sign(a[bins, np.arange(d.shape[1])])
    sgn[sgn == 0] = 1.0
    C = len(g)
    sums = np.zeros((C, 3))
    ws = np.zeros(C)
    np.add.at(sums, bins, (d * (w * sgn)[None, :]).T)
    np.add.at(ws, bins, w)
    keep = ws > 0
    ctr = sums[keep]
    ctr /= (np.linalg.norm(ctr, axis=1, keepdims=True) + 1e-12)
    return ctr.T, ws[keep] / ws[keep].sum()


def _t_hinge_pop_score(Rs, dq, wq, dr, wr, cos_c):
    """Population correlation with hinge kernel (torch tensors, batched)."""
    Rd = torch.einsum('hij,jq->hiq', Rs, dq)                 # (H, 3, Cq)
    dots = torch.einsum('hiq,ir->hqr', Rd, dr).abs()         # (H, Cq, Cr)
    k = (dots - cos_c).clamp_min(0.0) / (1.0 - cos_c)
    return torch.einsum('hqr,q,r->h', k, wq, wr)


def _t_hinge_pair_score(Rs, dq, dr, w, cos_c):
    """Weighted matcher-pair correlation with hinge kernel (torch)."""
    dots = torch.einsum('hij,jm,im->hm', Rs, dq, dr).abs()   # (H, M)
    return ((dots - cos_c).clamp_min(0.0) / (1.0 - cos_c)) @ w


def _t_peaks_and_refine(Rs, sc, score_fn, n_peaks, dev, peak_sep_deg=12.0):
    """Vectorized peak extraction + local grid refinement (torch)."""
    cos_sep = float(np.cos(np.deg2rad(peak_sep_deg)))
    alive = torch.ones(len(Rs), dtype=torch.bool, device=dev)
    peaks = []
    for _ in range(n_peaks):
        if not alive.any():
            break
        idx = torch.where(alive)[0]
        h = idx[sc[idx].argmax()]
        Rp = Rs[h]
        peaks.append(Rp)
        # suppress everything within peak_sep of the new peak:
        # angle(Ra, Rb) < sep  <=>  (trace(Ra^T Rb) - 1) / 2 > cos(sep)
        tr = torch.einsum('hij,ij->h', Rs, Rp)
        alive &= (tr - 1.0) / 2.0 <= cos_sep
    global _P1_CACHE, _P2_CACHE
    try:
        P1, P2 = _P1_CACHE.to(dev), _P2_CACHE.to(dev)
    except NameError:
        P1 = torch.from_numpy(axis_angle_mats([3.0, 1.5])).float()
        P2 = torch.from_numpy(axis_angle_mats([0.75])).float()
        _P1_CACHE, _P2_CACHE = P1, P2
        P1, P2 = P1.to(dev), P2.to(dev)
    if not peaks:
        return []
    # refine ALL peaks together: one scoring pass per round instead of
    # per-peak kernel-launch loops (identical argmax per peak)
    R = torch.stack(peaks)                                   # (M, 3, 3)
    M = len(R)
    for P in (P1, P2):
        p = len(P)
        cand = torch.einsum('pij,mjk->mpik', P, R)           # (M, p, 3, 3)
        sc_r = score_fn(cand.reshape(-1, 3, 3)).view(M, p)
        R = cand[torch.arange(M, device=dev), sc_r.argmax(1)]
    fin = score_fn(R).cpu().numpy()
    refined = [(R[m], float(fin[m])) for m in range(M)]
    refined.sort(key=lambda x: -x[1])
    return refined


def rotation_candidates_fast(d_q, w_q, d_r, w_r,
                             pair_dq=None, pair_dr=None, pair_w=None,
                             cand_dq=None, cand_dr=None,
                             n_hyp=24000, n_grid=16000, n_peaks=12,
                             kernel_deg=8.0, dedup_deg=5.0,
                             q_bins=192, r_bins=384,
                             seed=0, device=None):
    """Online rotation candidate set: same objectives as rotation_candidates
    but RANSAC-sampled 2-pair hypotheses + fine-binned hinge-kernel scoring
    on the GPU (torch; numpy in/out, no scipy). ~40-120 ms on GPU.

    cand_dq, cand_dr: (3, K) matched candidate directions for hypothesis
        sampling (e.g. the network top-200). Defaults to pair_dq/pair_dr.
    """
    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(seed)
    hq = cand_dq if cand_dq is not None else pair_dq
    hr = cand_dr if cand_dr is not None else pair_dr
    cos_c = float(np.cos(np.deg2rad(kernel_deg)))
    f32 = lambda a: torch.as_tensor(np.ascontiguousarray(a),
                                    dtype=torch.float32, device=dev)

    # hypotheses: sampled 2-pair Procrustes, 4 sign combos, batched on GPU
    K = hq.shape[1]
    n_pairs = max(1, n_hyp // 4)
    i = rng.integers(0, K, n_pairs)
    j = rng.integers(0, K, n_pairs)
    i, j = i[i != j], j[i != j]
    tq, tr = f32(hq), f32(hr)
    a1 = torch.cat([tq[:, i], tq[:, i], -tq[:, i], -tq[:, i]], dim=1)
    b1 = torch.cat([tq[:, j], -tq[:, j], tq[:, j], -tq[:, j]], dim=1)
    a2 = tr[:, i].repeat(1, 4)
    b2 = tr[:, j].repeat(1, 4)
    c1 = torch.cross(a1, b1, dim=0)
    c2 = torch.cross(a2, b2, dim=0)
    n1 = c1.norm(dim=0)
    good = n1 > 0.1
    H = (torch.einsum('ip,jp->pij', a2, a1) +
         torch.einsum('ip,jp->pij', b2, b1) +
         torch.einsum('ip,jp->pij', c2 / (c2.norm(dim=0) + 1e-12),
                                    c1 / (n1 + 1e-12)))[good]
    U, _, Vt = torch.linalg.svd(H)
    R = U @ Vt
    bad = torch.linalg.det(R) < 0
    if bad.any():
        U = U.clone()
        U[bad, :, -1] *= -1
        R[bad] = U[bad] @ Vt[bad]
    Rs = [R]
    if n_grid > 0:                                    # small uniform insurance
        Rs.append(f32(quat_to_mat(super_fibonacci(n_grid))))
    Rs = torch.cat(Rs, dim=0)

    # fine-binned population scoring (binning stays cheap in numpy)
    cq, cwq = _cluster_dirs(d_q, w_q, n_bins=q_bins)
    cr, cwr = _cluster_dirs(d_r, w_r, n_bins=r_bins)
    cq, cwq, cr, cwr = f32(cq), f32(cwq), f32(cr), f32(cwr)

    def pop_fn(R_):
        out = torch.empty(len(R_), device=dev)
        for lo in range(0, len(R_), 8192):
            out[lo:lo + 8192] = _t_hinge_pop_score(
                R_[lo:lo + 8192], cq, cwq, cr, cwr, cos_c)
        return out

    out = []
    for R_, s in _t_peaks_and_refine(Rs, pop_fn(Rs), pop_fn, n_peaks, dev):
        out.append((R_.cpu().numpy().astype(float), 'corr', s))

    if pair_dq is not None:
        pdq, pdr = f32(pair_dq), f32(pair_dr)
        pw = f32(pair_w.clip(min=0))
        pw = pw / (pw.sum() + 1e-12)
        pair_fn = lambda R_: _t_hinge_pair_score(R_, pdq, pdr, pw, cos_c)
        for R_, s in _t_peaks_and_refine(Rs, pair_fn(Rs), pair_fn,
                                         n_peaks, dev):
            out.append((R_.cpu().numpy().astype(float), 'probcorr', s))

    dedup = []
    for R_, src, s in out:
        if all(_rot_angle_deg(R_, R0) > dedup_deg for R0, _, _ in dedup):
            dedup.append((R_, src, s))
    return dedup


class Sim3Solver:
    def __init__(self, q_ends, r_ends,
                 prenorm_radius=2.5, scale_band=1.8,
                 seg_tau_m=(0.5, 1.0), seg_ang_deg=20.0,
                 n_scale_starts=6, sigma_log_prior=0.6, device=None,
                 q_traj=None, r_traj=None):
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

        # extent-ratio scale prior from foot-of-perpendicular clouds
        # (pure Plücker; endpoint-midpoint variant measured 0.76-1.05x GT)
        self.p0_q = np.cross(self.p_q[:, :3], self.p_q[:, 3:])
        self.p0_r = np.cross(self.p_r[:, :3], self.p_r[:, 3:])
        def _spread(p0):
            return np.median(np.linalg.norm(p0 - np.median(p0, 0), axis=1))
        self.s0 = _spread(self.p0_r) / (_spread(self.p0_q) + 1e-12)
        self.s_lo, self.s_hi = self.s0 / scale_band, self.s0 * scale_band
        self._scale_band = scale_band
        self._s_lo0, self._s_hi0 = self.s_lo, self.s_hi
        # additional prior centers (see pcm_scale_band in register): the
        # extent-ratio prior assumes both maps cover the same AREA and is
        # biased for submap queries; the PCM graph-median pairwise scale of
        # matcher candidates is coverage-robust when the matcher is right.
        self._s_centers = [self.s0]
        self.sigma_log = sigma_log_prior
        self.n_scale_starts = n_scale_starts

        # verification structures (taus are REAL metres — scale by alpha)
        self.taus = tuple(t * self.alpha for t in seg_tau_m)
        self.cos_seg_ang = np.cos(np.deg2rad(seg_ang_deg))
        seg_dir = self.r2 - self.r1
        self.ref_len = np.linalg.norm(seg_dir, axis=1)
        self.ref_u = seg_dir / (self.ref_len[:, None] + 1e-12)
        self.ref_mid = (self.r1 + self.r2) / 2
        self._t_ref_mid = torch.as_tensor(self.ref_mid, dtype=torch.float32,
                                          device=self.dev)
        self.q_len = np.linalg.norm(self.q2 - self.q1, axis=1)
        self.q_mid = (self.q1 + self.q2) / 2
        qd = self.q2 - self.q1
        self.q_u = qd / (self.q_len[:, None] + 1e-12)

        # moment-space verification structures: canonical-sign 6D Plücker
        # tree over the reference ([m; d] with the sign fixed by the largest
        # |d| component, so d and -d index identically)
        self._ref_canon = self._canon6(self.p_r)
        self._t_ref6 = torch.as_tensor(self._ref_canon, dtype=torch.float32,
                                       device=self.dev)
        # torch copies of the query for batched scoring
        self._t_qm = torch.as_tensor(self.p_q[:, :3], dtype=torch.float32,
                                     device=self.dev)
        self._t_qd = torch.as_tensor(self.p_q[:, 3:], dtype=torch.float32,
                                     device=self.dev)
        # PURE Plücker: uniform verification weights (segment length is an
        # endpoint quantity and cross-modal fragments differ in extent)
        self._t_qlen = torch.ones(len(self.p_q), dtype=torch.float32,
                                  device=self.dev)
        # arm-free perp verification structures (foot points + directions)
        self._t_ref_p0 = torch.as_tensor(self.p0_r, dtype=torch.float32,
                                         device=self.dev)
        self._t_ref_d = torch.as_tensor(self.p_r[:, 3:], dtype=torch.float32,
                                        device=self.dev)
        # optional keyframe trajectories (K,3): an independent witness that
        # breaks the flip symmetry of repetitive line structure — under a
        # room/facade flip the transformed query trajectory lands in space
        # the reference camera never visited (measured on the benchmark:
        # truth-pose agreement 0.35-0.70 vs 0.00-0.12 at flip poses)
        self._traj_w = 0.0
        self._t_q_traj = self._t_r_traj = None
        if q_traj is not None and r_traj is not None \
                and len(q_traj) >= 5 and len(r_traj) >= 5:
            self._t_q_traj = torch.as_tensor(
                np.asarray(q_traj, float) * self.alpha,
                dtype=torch.float32, device=self.dev)
            self._t_r_traj = torch.as_tensor(
                np.asarray(r_traj, float) * self.alpha,
                dtype=torch.float32, device=self.dev)

    @staticmethod
    def _canon6(L):
        """Sign-canonicalize (N, 6) Plücker vectors: flip [m; d] jointly so
        the largest-|.| direction component is positive."""
        d = L[:, 3:]
        idx = np.argmax(np.abs(d), axis=1)
        sgn = np.sign(d[np.arange(len(d)), idx])
        sgn[sgn == 0] = 1.0
        return L * sgn[:, None]

    # ── verification ───────────────────────────────────────────────────────
    @staticmethod
    def _pt_seg(p, a, u, L):
        ap = p - a
        tp = np.clip((ap * u).sum(-1), 0.0, L)
        return np.linalg.norm(p - (a + tp[..., None] * u), axis=-1)

    def seg_score(self, R, t, s, tau=None, k=12):
        """Length-weighted fraction of query segments whose midpoint lies
        within tau (normalized) of a direction-consistent ref segment."""
        tau = self.taus[-1] if tau is None else tau
        am = (s * (R @ self.q_mid.T)).T + t
        au = (R @ self.q_u.T).T
        t_am = torch.as_tensor(am, dtype=torch.float32, device=self.dev)
        nn = torch.cdist(t_am, self._t_ref_mid).topk(
            k, largest=False).indices.cpu().numpy()
        d = self._pt_seg(am[:, None, :], self.r1[nn], self.ref_u[nn],
                         self.ref_len[nn])
        ok = (d < tau) & (np.abs((au[:, None, :] * self.ref_u[nn]).sum(-1))
                          > self.cos_seg_ang)
        inl = ok.any(axis=1)
        return float((self.q_len * inl).sum() / (self.q_len.sum() + 1e-9))

    def prior(self, s):
        return float(self._sprior(np.asarray([s]))[0])

    def _sprior(self, ss):
        """Log-normal scale prior, max over the active centers (vectorized).
        With the default single center this is the classic extent-ratio
        prior; pcm_scale_band adds the candidate-pairwise-scale center."""
        ss = np.asarray(ss, float)
        out = np.zeros_like(ss)
        for c in self._s_centers:
            np.maximum(out, np.exp(-np.log(ss / c) ** 2
                                   / (2 * self.sigma_log ** 2)), out=out)
        return out

    def moment_score(self, R, t, s, tau_m, ang_deg=20.0, k=8):
        """Single-pose wrapper over moment_scores_batch (one tau)."""
        return float(self.moment_scores_batch(
            np.asarray(R)[None], np.asarray(t).reshape(1, 3),
            np.asarray([s]), tau_ms=(tau_m,), ang_deg=ang_deg, k=k)[0])

    def moment_scores_batch(self, Rs, ts, ss, tau_ms=(0.1, 0.2, 0.4),
                            ang_deg=20.0, k=8, chunk=64):
        """Sum of length-weighted moment-inlier fractions at each tau_m,
        for a BATCH of poses. Rs (P,3,3), ts (P,3), ss (P,). One fused GPU
        pass — no trees, no python per-pose loops."""
        P = len(Rs)
        Rt = torch.as_tensor(np.asarray(Rs), dtype=torch.float32,
                             device=self.dev)
        tt = torch.as_tensor(np.asarray(ts), dtype=torch.float32,
                             device=self.dev)
        st = torch.as_tensor(np.asarray(ss), dtype=torch.float32,
                             device=self.dev)
        cos_gate = float(np.cos(np.deg2rad(ang_deg)))
        out = torch.zeros(P, device=self.dev)
        wsum = self._t_qlen.sum() + 1e-9
        for lo in range(0, P, chunk):
            R_ = Rt[lo:lo + chunk]
            t_ = tt[lo:lo + chunk]
            s_ = st[lo:lo + chunk]
            d_out = torch.einsum('pij,nj->pni', R_, self._t_qd)
            m_out = s_[:, None, None] * torch.einsum('pij,nj->pni', R_,
                                                     self._t_qm)                 + torch.cross(t_[:, None, :].expand_as(d_out), d_out, dim=-1)
            L = torch.cat([m_out, d_out], dim=-1)          # (p, N, 6)
            # canonical sign by largest |d| component
            didx = d_out.abs().argmax(-1, keepdim=True)
            sgn = torch.gather(d_out, -1, didx).sign()
            sgn[sgn == 0] = 1.0
            L = L * sgn
            p, N, _ = L.shape
            nn = torch.cdist(L.reshape(-1, 6), self._t_ref6).topk(
                k, largest=False).indices                  # (p*N, k)
            ref = self._t_ref6[nn].reshape(p, N, k, 6)
            cosang = (L[:, :, None, 3:] * ref[..., 3:]).sum(-1)
            psgn = cosang.sign()
            psgn[psgn == 0] = 1.0
            dm = (L[:, :, None, :3] * psgn[..., None]
                  - ref[..., :3]).norm(dim=-1)
            ok_ang = cosang.abs() > cos_gate
            sc = torch.zeros(p, device=self.dev)
            for tau in tau_ms:
                inl = ((dm < tau) & ok_ang).any(-1)        # (p, N)
                sc += (inl * self._t_qlen).sum(-1) / wsum
            out[lo:lo + chunk] = sc
        return out.cpu().numpy()

    def perp_scores_batch(self, Rs, ts, ss, tau_ms=(0.1, 0.2, 0.4),
                          ang_deg=20.0, k=8, chunk=64):
        """Arm-free verification: fraction of query lines whose transformed
        INFINITE line passes within tau (normalized units) of a
        direction-consistent reference line, via perpendicular offset of
        the reference foot point from the transformed line (p0-NN
        association). Unlike the moment residual (||dm|| ~ r sin dtheta,
        amplified by the line's arm length r), this residual is evaluated
        AT the line's own location, so the same normalized taus transfer
        across scene scales (indoor rooms and street maps alike)."""
        P = len(Rs)
        Rt = torch.as_tensor(np.asarray(Rs), dtype=torch.float32,
                             device=self.dev)
        tt = torch.as_tensor(np.asarray(ts, np.float32).reshape(P, 3),
                             device=self.dev)
        st = torch.as_tensor(np.asarray(ss, np.float32), device=self.dev)
        cos_gate = float(np.cos(np.deg2rad(ang_deg)))
        out = torch.zeros(P, device=self.dev)
        wsum = self._t_qlen.sum() + 1e-9
        for lo in range(0, P, chunk):
            R_ = Rt[lo:lo + chunk]
            t_ = tt[lo:lo + chunk]
            s_ = st[lo:lo + chunk]
            d_out = torch.einsum('pij,nj->pni', R_, self._t_qd)
            m_out = s_[:, None, None] * torch.einsum('pij,nj->pni', R_,
                                                     self._t_qm) \
                + torch.cross(t_[:, None, :].expand_as(d_out), d_out,
                              dim=-1)
            p0 = torch.cross(m_out, d_out, dim=-1)         # (p, N, 3)
            p, N, _ = p0.shape
            nn = torch.cdist(p0.reshape(-1, 3), self._t_ref_p0).topk(
                k, largest=False).indices                  # (p*N, k)
            rd = self._t_ref_d[nn].reshape(p, N, k, 3)
            rp = self._t_ref_p0[nn].reshape(p, N, k, 3)
            cosang = (d_out[:, :, None, :] * rd).sum(-1)
            dp = rp - p0[:, :, None, :]
            perp = (dp - (dp * d_out[:, :, None, :]).sum(-1, keepdim=True)
                    * d_out[:, :, None, :]).norm(dim=-1)
            ok_ang = cosang.abs() > cos_gate
            sc = torch.zeros(p, device=self.dev)
            for tau in tau_ms:
                inl = ((perp < tau) & ok_ang).any(-1)
                sc += (inl * self._t_qlen).sum(-1) / wsum
            out[lo:lo + chunk] = sc
        return out.cpu().numpy()

    def _traj_rot_candidates(self, n_grid=32768, n_peaks=12,
                             taus=(0.2, 0.4, 0.8)):
        """Trajectory-scored rotation candidates: score a uniform SO(3)
        grid by directed trajectory chamfer (query keyframes transformed
        with candidate R, scale s0-band sweep, centroid-aligned
        translation) and refine the top peaks. Repetitive line structure
        defeats the direction-population objectives (flips dominate), but
        session trajectories are asymmetric — this recovers rotations the
        line objectives never propose. No planarity/gravity assumption."""
        if self._t_q_traj is None:
            return []
        dev = self.dev
        qc = self._t_q_traj - self._t_q_traj.mean(0, keepdim=True)
        rc = self._t_r_traj - self._t_r_traj.mean(0, keepdim=True)
        scales = torch.as_tensor([self.s0 / 1.3, self.s0, self.s0 * 1.3],
                                 dtype=torch.float32, device=dev)

        def score_fn(Rs_t):
            out = torch.zeros(len(Rs_t), device=dev)
            for lo in range(0, len(Rs_t), 2048):
                R_ = Rs_t[lo:lo + 2048]
                tk = torch.einsum('pij,kj->pki', R_, qc)     # (p, K, 3)
                best = None
                for s_ in scales:
                    d = torch.cdist(s_ * tk, rc[None].expand(len(R_), -1, -1)
                                    ).min(-1).values
                    sc = torch.zeros(len(R_), device=dev)
                    for tau in taus:
                        sc += (d < tau).float().mean(-1)
                    best = sc if best is None else torch.maximum(best, sc)
                out[lo:lo + 2048] = best
            return out

        Rs = torch.as_tensor(quat_to_mat(super_fibonacci(n_grid)),
                             dtype=torch.float32, device=dev)
        peaks = _t_peaks_and_refine(Rs, score_fn(Rs), score_fn, n_peaks,
                                    dev)
        return [(R_.cpu().numpy().astype(float), 'trajgrid', float(s))
                for R_, s in peaks]

    def traj_scores_batch(self, Rs, ts, ss, taus=(0.2, 0.4, 0.8),
                          chunk=256):
        """Trajectory-agreement term: sum over taus of the fraction of
        transformed query keyframe centres within tau (normalized) of the
        reference keyframe trajectory (directed chamfer). Requires both
        trajectories (see __init__); returns zeros otherwise."""
        P = len(Rs)
        if self._t_q_traj is None:
            return np.zeros(P)
        Rt = torch.as_tensor(np.asarray(Rs), dtype=torch.float32,
                             device=self.dev)
        tt = torch.as_tensor(np.asarray(ts, np.float32).reshape(P, 3),
                             device=self.dev)
        st = torch.as_tensor(np.asarray(ss, np.float32), device=self.dev)
        out = torch.zeros(P, device=self.dev)
        for lo in range(0, P, chunk):
            R_, t_, s_ = Rt[lo:lo + chunk], tt[lo:lo + chunk], st[lo:lo + chunk]
            tk = (s_[:, None, None]
                  * torch.einsum('pij,kj->pki', R_, self._t_q_traj)
                  + t_[:, None, :])                       # (p, K, 3)
            d = torch.cdist(tk, self._t_r_traj[None].expand(len(R_), -1, -1)
                            ).min(-1).values              # (p, K)
            sc = torch.zeros(len(R_), device=self.dev)
            for tau in taus:
                sc += (d < tau).float().mean(-1)
            out[lo:lo + chunk] = sc
        return out.cpu().numpy()

    def _verify_batch(self, Rs, ts, ss, verify, tau_ms, ang_deg):
        if verify == 'both':
            # one scale-free rule for indoor AND street maps: the moment
            # residual carries along-line evidence where arms are short
            # (indoors) but is arm-amplified into saturation outdoors;
            # the perp residual is arm-free. Their SUM lets whichever is
            # informative at the scene's scale dominate.
            base = (self.moment_scores_batch(Rs, ts, ss, tau_ms=tau_ms,
                                             ang_deg=ang_deg)
                    + self.perp_scores_batch(Rs, ts, ss, tau_ms=tau_ms,
                                             ang_deg=ang_deg))
        else:
            fn = (self.perp_scores_batch if verify == 'perp'
                  else self.moment_scores_batch)
            base = fn(Rs, ts, ss, tau_ms=tau_ms, ang_deg=ang_deg)
        if self._t_q_traj is not None and self._traj_w > 0:
            base = base + self._traj_w * self.traj_scores_batch(Rs, ts, ss)
        return base

    def sel_score(self, R, t, s, verify='segment', use_prior=True):
        if verify == 'moment':
            base = sum(self.moment_score(R, t, s, tau_m)
                       for tau_m in (0.1, 0.2, 0.4))
        else:
            base = sum(self.seg_score(R, t, s, tau) for tau in self.taus)
        return base * (self.prior(s) if use_prior else 1.0)

    # ── line-based ICP (no point solves: moment equations only) ────────────
    def icp(self, R, t, s, n_iter=15, gates_m=(6.0, 3.0, 1.5, 1.0)):
        for it in range(n_iter):
            gate = gates_m[min(it // 3, len(gates_m) - 1)] * self.alpha
            am = (s * (R @ self.q_mid.T)).T + t
            t_am = torch.as_tensor(am, dtype=torch.float32, device=self.dev)
            dmid_t, nn_t = torch.cdist(t_am, self._t_ref_mid).min(dim=1)
            dmid, nn = dmid_t.cpu().numpy(), nn_t.cpu().numpy()
            au = (R @ self.q_u.T).T
            cosang = np.abs((au * self.ref_u[nn]).sum(1))
            m = (dmid < gate) & (cosang > self.cos_seg_ang)
            if m.sum() < 10:
                break
            sgn = np.sign((au[m] * self.ref_u[nn[m]]).sum(1))
            sgn[sgn == 0] = 1.0
            # rotation: sign-aware Procrustes on directions
            H = (self.ref_u[nn[m]] * sgn[:, None]).T @ self.q_u[m]
            U, _, Vt = np.linalg.svd(H)
            R2 = U @ np.diag([1., 1., float(np.linalg.det(U @ Vt))]) @ Vt
            # (s, t): Plücker moment equations of the associated LINE pairs
            L1 = self.p_q[m].T.copy() * sgn[None, :]
            L2 = self.p_r[nn[m]].T
            m1, m2 = L1[:3], L2[:3]
            Rd1 = R2 @ L1[3:]
            Rm1 = R2 @ m1
            t2, s2 = t, s
            for _ in range(2):
                txd = np.cross(np.broadcast_to(t2, (Rd1.shape[1], 3)),
                               Rd1.T).T
                num = ((m2 - txd) * Rm1).sum(axis=0)
                den = (m1 ** 2).sum(axis=0)
                ok = den > 1e-8
                if ok.sum() < 4:
                    break
                s2 = float(np.clip(np.median(num[ok] / den[ok]),
                                   self.s_lo, self.s_hi))
                rhs = (m2 - s2 * Rm1).T
                n = Rd1.shape[1]
                A = np.zeros((n, 3, 3))
                x, y, z = Rd1[0], Rd1[1], Rd1[2]
                A[:, 0, 1], A[:, 0, 2] = z, -y      # -skew(Rd1)
                A[:, 1, 0], A[:, 1, 2] = -z, x
                A[:, 2, 0], A[:, 2, 1] = y, -x
                t2, *_ = np.linalg.lstsq(A.reshape(-1, 3), rhs.reshape(-1),
                                         rcond=None)
            done = (abs(s2 - s) / s < 1e-4
                    and np.linalg.norm(t2 - t) < 1e-4)
            R, t, s = R2, t2, s2
            if done:
                break
        return R, t, s

    # ── pairwise consistency maximization (PCM) candidate filter ──────────
    def pcm_filter(self, pl_q, pl_r, keep_frac=0.5, min_keep=30,
                   sigma_ang=6.0, sigma_s=0.25):
        """Pairwise-consistency candidate filter (PCM, Mangelson et al.
        ICRA 2018; dense-subgraph relaxation as in CLIPPER). Pure Plücker
        pairwise invariants between two candidate correspondences:
          * inter-direction angle (preserved by any Sim(3));
          * inter-line distance ratio dist_r(i,j)/dist_q(i,j), which must
            equal the global scale s.
        Real map noise makes strict max-clique brittle (GT-GT edge density
        ~0.6), so instead each candidate is scored by its SOFT consistency
        degree: sum over partners of a Gaussian angle-agreement kernel
        times a Gaussian scale-agreement kernel around the graph-level
        weighted-median pairwise scale. Top keep_frac by degree survive.
        NOTE: angle relations are invariant under ANY rotation, so
        symmetric-structure aliases (Manhattan flips) form their own
        consistent subgraph — PCM purifies random mismatches; the
        moment-residual verification still arbitrates flips."""
        K = pl_q.shape[1]
        dq, dr = pl_q[3:].T, pl_r[3:].T                     # (K, 3)
        p0q = np.cross(pl_q[:3].T, dq)
        p0r = np.cross(pl_r[:3].T, dr)

        def pair_geom(d, p0):
            ang = np.arccos(np.abs(np.clip(d @ d.T, -1, 1)))
            n = np.cross(d[:, None, :], d[None, :, :])      # (K, K, 3)
            nn = np.linalg.norm(n, axis=-1)
            dp = p0[None, :, :] - p0[:, None, :]
            dist_skew = np.abs((dp * n).sum(-1)) / (nn + 1e-12)
            perp = np.linalg.norm(
                dp - (dp * d[:, None, :]).sum(-1, keepdims=True)
                * d[:, None, :], axis=-1)
            return ang, np.where(nn > 1e-3, dist_skew, perp)

        ang_q, dist_q = pair_geom(dq, p0q)
        ang_r, dist_r = pair_geom(dr, p0r)
        w_ang = np.exp(-0.5 * ((ang_q - ang_r)
                               / np.deg2rad(sigma_ang)) ** 2)
        s_ij = dist_r / (dist_q + 1e-9)
        informative = dist_q > 0.2          # near-collinear pairs carry no
        ls = np.log(np.clip(s_ij, 1e-6, 1e6))   # scale information
        m = (informative & (w_ang > 0.3)
             & (s_ij > self.s_lo) & (s_ij < self.s_hi))
        s_med = float(np.median(s_ij[m])) if m.sum() > 10 else self.s0
        self._pcm_s_med = s_med
        w_s = np.where(informative,
                       np.exp(-0.5 * ((ls - np.log(s_med)) / sigma_s) ** 2),
                       0.5)
        A = w_ang * w_s
        np.fill_diagonal(A, 0)
        deg = A.sum(1)
        k = max(min_keep, int(K * keep_frac))
        keep = np.zeros(K, bool)
        keep[np.argsort(-deg)[:k]] = True
        return keep

    # ── (t, s) proposals from matcher candidates ───────────────────────────
    def _ts_proposals(self, R, pl_q, pl_r, eps_cand_deg=25.0, max_combos=6000,
                      bound_scale=True):
        """All 2-pair line-moment (t, s) solves among direction-consistent
        candidates, BATCHED (one normal-equations solve for every combo)."""
        dots = np.abs((R @ pl_q[3:] * pl_r[3:]).sum(axis=0))
        cand = np.where(dots > np.cos(np.deg2rad(eps_cand_deg)))[0]
        if len(cand) < 2:
            return []
        ti, tj = np.triu_indices(len(cand), k=1)
        combos = np.stack([cand[ti], cand[tj]], axis=1).astype(np.int64)
        if len(combos) > max_combos:
            sel = np.random.default_rng(0).choice(len(combos), max_combos,
                                                  replace=False)
            combos = combos[sel]
        C = len(combos)
        m1 = pl_q[:3, combos].transpose(1, 0, 2)          # (C, 3, 2)
        d1 = pl_q[3:, combos].transpose(1, 0, 2)
        m2 = pl_r[:3, combos].transpose(1, 0, 2)
        Rm1 = np.einsum('ij,cjk->cik', R, m1)             # (C, 3, 2)
        Rd1 = np.einsum('ij,cjk->cik', R, d1)
        # rows for pair member p: [-skew(Rd1_p) | Rm1_p] x [t; s] = m2_p
        A = np.zeros((C, 6, 4))
        for pmem in range(2):
            x, y, z = Rd1[:, 0, pmem], Rd1[:, 1, pmem], Rd1[:, 2, pmem]
            r0 = 3 * pmem
            A[:, r0 + 0, 1], A[:, r0 + 0, 2] = z, -y      # -skew(Rd1)
            A[:, r0 + 1, 0], A[:, r0 + 1, 2] = -z, x
            A[:, r0 + 2, 0], A[:, r0 + 2, 1] = y, -x
            A[:, r0:r0 + 3, 3] = Rm1[:, :, pmem]
        b = m2.transpose(0, 2, 1).reshape(C, 6)
        At = A.transpose(0, 2, 1)
        AtA = At @ A + 1e-9 * np.eye(4)[None]
        Atb = np.einsum('cij,cj->ci', At, b)
        try:
            # batched solve needs the trailing singleton on the RHS
            x = np.linalg.solve(AtA, Atb[..., None])[..., 0]   # (C, 4)
        except np.linalg.LinAlgError:
            return []
        t_all, s_all = x[:, :3], x[:, 3]
        ok = np.isfinite(s_all) & (s_all > 0)
        if bound_scale:
            ok &= (s_all > self.s_lo) & (s_all < self.s_hi)
        t_all, s_all = t_all[ok], s_all[ok]
        if len(s_all) == 0:
            return []
        # O(n) dedupe on a quantized (log s, t) grid
        key = np.stack([np.round(np.log(s_all) / 0.05),
                        np.round(t_all[:, 0] / 0.15),
                        np.round(t_all[:, 1] / 0.15),
                        np.round(t_all[:, 2] / 0.15)], axis=1)
        _, uniq = np.unique(key, axis=0, return_index=True)
        if len(uniq) > 300:
            # np.unique sorts by key — a [:300] slice would keep only the
            # smallest-scale clusters; subsample instead
            uniq = np.random.default_rng(0).choice(uniq, 300, replace=False)
        return [(t_all[i], float(s_all[i])) for i in uniq]

    # ── guarded final inlier refit (Shin et al.-style polish, no ICP) ─────
    def _polish(self, R, t, s, taus, ang_deg, tau_assoc=0.2):
        """ONE fixed-association least-squares refit at the winning pose:
        associate each query line to its NN in canonical 6D Plücker space,
        keep pairs passing the direction gate + moment residual, then
        direction-Procrustes for R and a joint linear (t, s) moment solve
        over ALL inliers. NO re-association / iteration (that is the ICP
        ablation, which diverges — same failure Shin et al. report for
        their Refine stage on Room1). Caller must guard by score."""
        Rt = torch.as_tensor(R, dtype=torch.float32, device=self.dev)
        tt = torch.as_tensor(np.asarray(t, np.float32).reshape(3),
                             device=self.dev)
        d_out = self._t_qd @ Rt.T
        m_out = float(s) * (self._t_qm @ Rt.T) \
            + torch.cross(tt.expand_as(d_out), d_out, dim=-1)
        L = torch.cat([m_out, d_out], dim=-1)
        didx = d_out.abs().argmax(-1, keepdim=True)
        sgn1 = torch.gather(d_out, -1, didx).sign()
        sgn1[sgn1 == 0] = 1.0
        L = L * sgn1
        nn = torch.cdist(L, self._t_ref6).argmin(1)
        ref = self._t_ref6[nn]
        cosang = (L[:, 3:] * ref[:, 3:]).sum(-1)
        psgn = cosang.sign()
        psgn[psgn == 0] = 1.0
        dm = (L[:, :3] * psgn[:, None] - ref[:, :3]).norm(dim=-1)
        ok = (cosang.abs() > np.cos(np.deg2rad(ang_deg))) & (dm < tau_assoc)
        m = ok.cpu().numpy()
        if m.sum() < 10:
            return R, t, s
        i = np.where(m)[0]
        j = nn.cpu().numpy()[i]
        # total sign so that transformed(sig * p_q[i]) aligns with RAW
        # p_r[j]: canon sign of the transformed query (sgn1), pair sign
        # vs the CANONICALIZED ref (psgn), and the ref's own canon sign
        # (c_j) — _t_ref6 rows are sign-flipped vs p_r where c_j = -1
        dr = self.p_r[j, 3:]
        c_j = np.sign(dr[np.arange(len(j)),
                         np.argmax(np.abs(dr), axis=1)])
        c_j[c_j == 0] = 1.0
        sig = (sgn1[:, 0] * psgn).cpu().numpy()[i] * c_j
        L1 = self.p_q[i].astype(float) * sig[:, None]        # (K, 6)
        L2 = self.p_r[j].astype(float)
        # rotation: Procrustes on directions
        H = L2[:, 3:].T @ L1[:, 3:]
        U, _, Vt = np.linalg.svd(H)
        Rn = U @ np.diag([1., 1., float(np.linalg.det(U @ Vt))]) @ Vt
        # joint (t, s): stacked moment equations, one LS solve
        Rd1 = (Rn @ L1[:, 3:].T)                             # (3, K)
        Rm1 = (Rn @ L1[:, :3].T)
        K = L1.shape[0]
        A = np.zeros((K, 3, 4))
        x, y, z = Rd1[0], Rd1[1], Rd1[2]
        A[:, 0, 1], A[:, 0, 2] = z, -y                      # -skew(Rd1)
        A[:, 1, 0], A[:, 1, 2] = -z, x
        A[:, 2, 0], A[:, 2, 1] = y, -x
        A[:, :, 3] = Rm1.T
        Af = A.reshape(-1, 4)
        bf = L2[:, :3].reshape(-1)
        sol, *_ = np.linalg.lstsq(Af, bf, rcond=None)
        tn, sn = sol[:3], float(sol[3])
        if not np.isfinite(sn) or sn <= 0:
            return R, t, s
        sn = float(np.clip(sn, self.s_lo, self.s_hi))
        return Rn, tn, sn

    def _round2(self, R, t, s, sc, verify, verify_taus, verify_ang,
                bound_scale, k=4, gate=0.35, ang_deg=12.0):
        """Second proposal round at the selected pose (gate in normalized
        units — association radius scales with the scene)."""
        Rt = torch.as_tensor(R, dtype=torch.float32, device=self.dev)
        tt = torch.as_tensor(np.asarray(t, np.float32).reshape(3),
                             device=self.dev)
        d_out = self._t_qd @ Rt.T
        m_out = float(s) * (self._t_qm @ Rt.T) \
            + torch.cross(tt.expand_as(d_out), d_out, dim=-1)
        p0 = torch.cross(m_out, d_out, dim=-1)
        nn = torch.cdist(p0, self._t_ref_p0).topk(k, largest=False)
        iq = np.repeat(np.arange(len(self.p_q)), k)
        ir = nn.indices.cpu().numpy().reshape(-1)
        ok = (nn.values.cpu().numpy().reshape(-1) < gate) & \
             (np.abs(((R @ self.p_q[iq, 3:].T).T
                      * self.p_r[ir, 3:]).sum(1))
              > np.cos(np.deg2rad(ang_deg)))
        iq, ir = iq[ok], ir[ok]
        if len(iq) < 10:
            return R, t, s, sc
        pl_q = self.p_q[iq].T.astype(float)
        pl_r = self.p_r[ir].T.astype(float)
        keep = self.pcm_filter(pl_q, pl_r, keep_frac=0.3, min_keep=40)
        Rs2 = [R]
        ts2 = [np.asarray(t, float).reshape(3)]
        ss2 = [float(s)]
        for tp, sp in self._ts_proposals(R, pl_q[:, keep], pl_r[:, keep],
                                         bound_scale=bound_scale):
            Rs2.append(R); ts2.append(tp); ss2.append(sp)
        if len(Rs2) == 1:
            return R, t, s, sc
        ss2 = np.asarray(ss2)
        sc2 = self._verify_batch(Rs2, ts2, ss2, verify, verify_taus,
                                 verify_ang)
        if bound_scale:
            sc2 = sc2 * self._sprior(ss2)
        j = int(np.argmax(sc2))
        return Rs2[j], ts2[j], float(ss2[j]), float(sc2[j])

    # ── entry point ────────────────────────────────────────────────────────
    def register(self, prob=None, topk=200, refine=False,
                 verify='both', bound_scale=True, rot_grid=8000,
                 verify_taus=(0.1, 0.2, 0.4), verify_ang=20.0, pcm=True,
                 rot_hyp=6000, rot_peaks=8, polish=True, round2=True,
                 beam_width=3, arbitrate=False, pcm_scale_band=False,
                 pcm_keep_frac=0.5, pcm_sigma_ang=6.0, pcm_sigma_s=0.25,
                 polish_tau=0.2, traj_weight=0.0, traj_propose=False):
        """prob: (N_ref, N_qry) ScalePluckerNet probability matrix (torch
        or numpy) over self.p_r x self.p_q rows — the intended way to run
        the solver (prob=None falls back to correspondence-free seeding
        and exists for the arbitration branch / ablations). Returns
        (s, R, t, info) in ORIGINAL (un-normalized) coordinates.

        DEFAULTS: the default configuration IS the shipping method
        (pure Plücker, no refinement, moment-residual verification,
        scale band on, PCM-filtered proposal pool + halved rotation
        budget — 18/35 @5deg, 25/35 @10deg, 0.1 s on the 35-pair
        benchmark). Flags exist for ablations:
        refine=True       append the guarded line-ICP polish (ablation:
                          REDUCES success 18->11/35 at ~100x the cost)
        verify='segment'  endpoint-based segment score instead of the
                          moment residual (requires endpoints)
        bound_scale=False accept any s > 0 — reopens the s->0 collapse
                          attractor; measured 92-96% scale errors
        pcm=False         skip the pairwise-consistency proposal filter;
                          only safe with the full rotation budget
                          (rot_hyp=24000, rot_grid=16000, rot_peaks=12
                          -> 18/24/35 at 0.2 s; reduced budget without
                          PCM drops to 17/24)
        arbitrate=True    ALSO run a full matcher-free solve and keep the
                          pose the whole-map verification scores higher
                          (2x cost). Off by default: the method is
                          matcher-driven; this is the safety net for
                          out-of-distribution matcher inputs (it rescued
                          KITTI 06 where OOD candidates assemble a
                          mutually-consistent false consensus)."""
        # trajectory-agreement verification term (active only when both
        # keyframe trajectories were passed to __init__)
        self._traj_w = float(traj_weight)
        if prob is not None and arbitrate:
            # SELF-ARBITRATION: the matcher is one evidence source among
            # several; when its candidates are out-of-distribution garbage
            # (e.g. cross-modal street maps) they can assemble a false
            # consensus that poisons the hypothesis pool. Run the solve
            # WITH and WITHOUT the matcher and keep the pose the whole-map
            # verification scores higher — the same arbitration principle
            # used everywhere else in the estimator, at 2x cost.
            out_m = self.register(prob=prob, topk=topk, refine=refine,
                                  verify=verify, bound_scale=bound_scale,
                                  rot_grid=rot_grid,
                                  verify_taus=verify_taus,
                                  verify_ang=verify_ang, pcm=pcm,
                                  rot_hyp=rot_hyp, rot_peaks=rot_peaks,
                                  polish=polish, round2=round2,
                                  beam_width=beam_width, arbitrate=False,
                                  pcm_scale_band=pcm_scale_band,
                                  pcm_keep_frac=pcm_keep_frac,
                                  pcm_sigma_ang=pcm_sigma_ang,
                                  pcm_sigma_s=pcm_sigma_s,
                                  polish_tau=polish_tau,
                                  traj_weight=traj_weight,
                                  traj_propose=traj_propose)
            out_f = self.register(prob=None, topk=topk, refine=refine,
                                  verify=verify, bound_scale=bound_scale,
                                  rot_grid=rot_grid,
                                  verify_taus=verify_taus,
                                  verify_ang=verify_ang, pcm=pcm,
                                  rot_hyp=rot_hyp, rot_peaks=rot_peaks,
                                  polish=polish, round2=round2,
                                  beam_width=beam_width, arbitrate=False,
                                  pcm_keep_frac=pcm_keep_frac,
                                  pcm_sigma_ang=pcm_sigma_ang,
                                  pcm_sigma_s=pcm_sigma_s,
                                  polish_tau=polish_tau,
                                  traj_weight=traj_weight,
                                  traj_propose=traj_propose)
            if out_m[0] is None:
                return out_f
            if out_f[0] is None:
                return out_m
            return out_m if out_m[3]['score'] >= out_f[3]['score'] else out_f

        pair_dq = pair_dr = pair_w = cand_dq = cand_dr = None
        pl_q_cand = pl_r_cand = pl_q_prop = pl_r_prop = None
        if prob is not None:
            pt = torch.as_tensor(prob)
            k = min(topk, pt.numel())
            _, flat = torch.topk(pt.flatten(), k=k)
            ir = (flat // pt.size(-1)).cpu().numpy()
            iq = (flat % pt.size(-1)).cpu().numpy()
            pl_q_cand = self.p_q[iq].T.astype(float)
            pl_r_cand = self.p_r[ir].T.astype(float)
            # PCM filters the (t, s) proposal pool only: rotation
            # hypotheses keep the full candidate set (filtering before the
            # rotation stage loses rotation-critical pairs on low-purity
            # scenes).
            pl_q_prop, pl_r_prop = pl_q_cand, pl_r_cand
            if pcm:
                # reset scale-band state (register may be called repeatedly)
                self.s_lo, self.s_hi = self._s_lo0, self._s_hi0
                self._s_centers = [self.s0]
                keep = self.pcm_filter(pl_q_cand, pl_r_cand,
                                       keep_frac=pcm_keep_frac,
                                       sigma_ang=pcm_sigma_ang,
                                       sigma_s=pcm_sigma_s)
                pl_q_prop = pl_q_cand[:, keep]
                pl_r_prop = pl_r_cand[:, keep]
                if pcm_scale_band and np.isfinite(self._pcm_s_med) \
                        and self._pcm_s_med > 0:
                    # coverage-robust second scale hypothesis: the median
                    # pairwise distance-ratio of matcher candidates (from
                    # the PCM graph) does not assume equal map coverage.
                    sp = self._pcm_s_med
                    self._s_centers = [self.s0, sp]
                    self.s_lo = min(self.s_lo, sp / self._scale_band)
                    self.s_hi = max(self.s_hi, sp * self._scale_band)
            cand_dq, cand_dr = pl_q_cand[3:], pl_r_cand[3:]
            m = min(5, pt.shape[0])
            pv, pi = torch.topk(pt, k=m, dim=0)
            pair_dq = np.repeat(self.p_q[:, 3:], m, axis=0).T.astype(float)
            pair_dr = self.p_r[pi.cpu().numpy().T.reshape(-1), 3:] \
                .T.astype(float)
            pair_w = pv.cpu().numpy().T.reshape(-1).clip(min=0).astype(float)
        else:
            # correspondence-free: seed hypotheses from mutual direction
            # structure only (uniform grid inside rotation_candidates_fast)
            sub = np.random.default_rng(0).choice(
                len(self.q1), min(200, len(self.q1)), replace=False)
            cand_dq = self.p_q[sub, 3:].T.astype(float)
            t_qm = torch.as_tensor(self.q_mid[sub], dtype=torch.float32,
                                   device=self.dev)
            nnr = torch.cdist(t_qm, self._t_ref_mid).argmin(1).cpu().numpy()
            cand_dr = self.p_r[nnr, 3:].T.astype(float)

        cands = rotation_candidates_fast(
            self.p_q[:, 3:].T.astype(float), self.q_len,
            self.p_r[:, 3:].T.astype(float), self.ref_len,
            pair_dq=pair_dq, pair_dr=pair_dr, pair_w=pair_w,
            cand_dq=cand_dq, cand_dr=cand_dr, n_hyp=rot_hyp,
            n_grid=rot_grid, n_peaks=rot_peaks, device=self.dev)
        if self._t_q_traj is not None and self._traj_w > 0 and traj_propose:
            # EXPERIMENTAL (off by default — measured net-harmful on the
            # 7-Scenes probe set: the trajectory chamfer is too broad a
            # rotation objective for compact handheld trajectories, its
            # wrong peaks displace good beam entries): trajectory-scored
            # rotation candidates.
            cands = cands + self._traj_rot_candidates()

        cq = np.median(self.p0_q, axis=0)
        cr = np.median(self.p0_r, axis=0)
        sweep_lo, sweep_hi = ((self.s_lo * 1.05, self.s_hi * 0.95)
                              if bound_scale
                              else (self.s0 / 5.0, self.s0 * 5.0))
        best = (None, None, None, -1.0)
        beam = []
        if not refine and verify in ('moment', 'perp', 'both'):
            # fully batched path: gather every candidate pose, score all in
            # one fused GPU pass (no per-pose python loops, no trees)
            Rs_all, ts_all, ss_all, src_all = [], [], [], []
            for R, src, _ in cands:
                if pl_q_cand is not None:
                    for t, sc_ in self._ts_proposals(
                            R, pl_q_prop, pl_r_prop, bound_scale=bound_scale):
                        Rs_all.append(R); ts_all.append(t); ss_all.append(sc_)
                        src_all.append(src)
                for s0 in np.geomspace(sweep_lo, sweep_hi,
                                       self.n_scale_starts):
                    Rs_all.append(R)
                    ts_all.append(cr - s0 * (R @ cq))
                    ss_all.append(float(s0))
                    src_all.append(src)
                    if self._t_q_traj is not None and self._traj_w > 0 \
                            and traj_propose:
                        # trajectory-centroid-aligned start (tied to the
                        # experimental trajectory proposals above)
                        Rs_all.append(R)
                        ts_all.append(
                            self._t_r_traj.mean(0).cpu().numpy()
                            - s0 * (R @ self._t_q_traj.mean(0).cpu()
                                    .numpy()))
                        ss_all.append(float(s0))
                        src_all.append(src)
            if not Rs_all:
                return None, None, None, dict(score=-1.0)
            ss_np = np.asarray(ss_all)
            scores = self._verify_batch(Rs_all, ts_all, ss_np, verify,
                                        verify_taus, verify_ang)
            if bound_scale:
                scores = scores * self._sprior(ss_np)
            order = np.argsort(-scores)
            beam = []          # top rotation-distinct poses for round 2
            for i in order:
                if all(_rot_angle_deg(Rs_all[i], Rb) > 15.0
                       for Rb, _, _, _ in beam):
                    beam.append((Rs_all[i], ts_all[i], ss_all[i],
                                 float(scores[i])))
                    if len(beam) >= beam_width:
                        break
            # trajectory-derived rotations are proposed with CRUDE (t, s)
            # (centroid starts), so they rarely out-score polished flip
            # poses in round 1 — force the best of them into the beam so
            # round-2 re-association gives them a refined (t, s) before
            # the final argmax (a fair polished-vs-polished comparison)
            n_traj_beam = 0
            for i in order:
                if n_traj_beam >= 2:
                    break
                if src_all[i] != 'trajgrid':
                    continue
                if all(_rot_angle_deg(Rs_all[i], Rb) > 15.0
                       for Rb, _, _, _ in beam):
                    beam.append((Rs_all[i], ts_all[i], ss_all[i],
                                 float(scores[i])))
                    n_traj_beam += 1
            best = beam[0]
        else:
            for R, src, _ in cands:
                starts = []
                if pl_q_cand is not None:
                    props = self._ts_proposals(R, pl_q_prop, pl_r_prop,
                                               bound_scale=bound_scale)
                    props.sort(key=lambda x: -self.sel_score(
                        R, x[0], x[1], verify=verify, use_prior=bound_scale))
                    starts += [(R, t, s)
                               for t, s in props[:2 if refine else 10]]
                for s0 in np.geomspace(sweep_lo, sweep_hi,
                                       self.n_scale_starts):
                    starts.append((R, cr - s0 * (R @ cq), float(s0)))
                for R0, t0, s0 in starts:
                    Ri, ti, si = (self.icp(R0, t0, s0) if refine
                                  else (R0, t0, s0))
                    sc = self.sel_score(Ri, ti, si, verify=verify,
                                        use_prior=bound_scale)
                    if sc > best[3]:
                        best = (Ri, ti, si, sc)
        R, t, s, sc = best
        if R is None:
            return None, None, None, dict(score=-1.0)
        if round2 and verify in ('moment', 'perp', 'both') and not refine:
            # position-aware re-proposal round, run as a BEAM over the top
            # rotation-distinct poses: associate query -> ref by foot-point
            # proximity AT each pose, PCM-filter, re-solve (t, s) at the
            # fixed rotation, re-verify; final pose = argmax across beams.
            # Refining only the argmax lets a false consensus keep its
            # advantage — the runner-up (often the true rotation) never
            # gets its re-proposal round.
            hyps = beam if beam else [(R, t, s, sc)]
            best2 = (R, t, s, sc)
            for Rb, tb, sb, scb in hyps:
                r2 = self._round2(Rb, tb, sb, scb, verify, verify_taus,
                                  verify_ang, bound_scale)
                if r2[3] > best2[3]:
                    best2 = r2
            R, t, s, sc = best2
        if polish and verify in ('moment', 'perp', 'both') and not refine:
            Rn, tn, sn = self._polish(R, t, s, verify_taus, verify_ang,
                                      tau_assoc=polish_tau)
            sc_new = float(self._verify_batch(
                [Rn], [tn], np.asarray([sn]), verify, verify_taus,
                verify_ang)[0])
            if bound_scale:
                sc_new *= self.prior(sn)
            # the verification score is a step function of the pose, so a
            # continuously better refit usually keeps the SAME inlier set;
            # accept on >= (equal support, LS-optimal within it)
            if sc_new >= sc:
                R, t, s, sc = Rn, tn, sn, sc_new
        return (float(s), R, np.asarray(t).reshape(3) / self.alpha,
                dict(score=sc, alpha=self.alpha, s0=self.s0,
                     n_rot_candidates=len(cands)))


# ═══ Max-inlier RANSAC baseline (ablations/tools only — NOT the method) ═══
# Merged from lib/ransac_grassmannian.py 2026-07-17 (module moved to
# ../attic/). G(2,4) max-principal-angle inlier counting per Shin et al.
# (ICCV 2025); kept as the classical consensus-maximization baseline the
# paper argues against, and as the trainer-era validation solver.
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

    No sign pre-alignment.  When both line clouds share a consistent direction
    convention (the same physical line yields the same sign in both maps), the
    raw cross-covariance H = Σ d2_i d1_i^T = R * Σ d1_i d1_i^T converges to
    c·R for a full-rank direction set, and SVD recovers R exactly.

    Raw-dot sign alignment ("flip d1 if d1·d2 < 0") is *harmful* here: for
    large rotation angles, ~60% of GT-matched direction pairs have negative
    raw dots even though d2 = R @ d1 exactly.  Flipping those directions
    inverts the cross-covariance and produces ~180° rotation errors.  See the
    "Rotation Estimation Metric Analysis" section in README for the evaluation.
    """
    U, _, Vt = np.linalg.svd(d2 @ d1.T)
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
      1. Predict d2 as R @ d1; set sign ε_i = sign((R @ d1_i) · d2_i).
      2. Set weights w_i = 1 / max(‖ε_i R d1_i − d2_i‖, eps).
      3. Solve weighted Procrustes via SVD of Σ w_i · d2_i · (ε_i d1_i)^T.

    Warm-started from L2 Procrustes (raw, no sign pre-alignment).
    Sign alignment inside the loop is EM-style (based on current R, not raw
    dots) so it is correct regardless of the rotation magnitude.
    Converges in ~10 iterations.
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
