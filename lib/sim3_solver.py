"""sim3_solver.py — SCALAR's Sim(3) line-map registration solver, one file.

    from lib.sim3_solver import Sim3Solver, solve_sim3
    solver = Sim3Solver((q1, q2), (r1, r2))          # (N,3) endpoints each
    R, t, s = solve_sim3(solver, prob)               # matcher prob (n_ref, n_q)

Lines are Plücker [m; d] (sign ambiguity [m;d] ~ [-m;-d]); positions are always
foot points f = d x m (sign-invariant).  ONE pipeline, no variants — every
stage and every default carries a controlled A/B recorded in the root
CLAUDE.md (with the scratchpad_*.txt logs); the alternatives that lost are
listed at the bottom of this docstring so they are not re-litigated, but they
are no longer code.

  0. NORMALIZATION (Sim3Solver): both clouds scaled by alpha so the reference
     median foot radius is 1.  This is the SINGLE normalization in the system —
     the matcher consumes the same lines via `Sim3Solver.matcher_input()`.
  1. top-K matcher pairs (K=200, LOAD-BEARING: 150/100 -> -4/-9 pairs) ->
     nsamp random 2-line samples (LESS IS MORE: 2000 indoor / 4000 outdoor are
     interior optima — more samples = more impostor exposure).
  2. SCALE-FREE SKEW OBSERVABILITY GATE, pre-solve: |(d1xd2).(f2-f1)| >
     skew_min x (per-cloud median foot radius), both clouds — two lines fix
     Sim(3) only if skew; near-coplanar pairs are scale-degenerate yet
     self-consistent (no residual test can catch them).  Rejected samples are
     never solved.
  3. MINIMAL SOLVE: query line 1 is hemisphere-canonicalized and BOTH signs of
     line 2 are proposed (2 hypotheses/sample), letting G(2,4) arbitrate.
     Each: SVD Procrustes on directions + joint 4x4 linear (t,s) from the
     moment constraint, then alt_iters=3 closed-form alternating steps on the
     Grassmann projection cost (`_alternate`, sign-invariant).  3 steps
     suffice — the L2 init lands in-basin and 3 absorb the residual below the
     ~8 deg correspondence noise floor (3 = 25 = 100 empirically).
  4. CONSISTENCY FILTER: converged projection cost < 0.12 (8 constraints vs
     7 DoF -> contaminated pairs cannot fit; quality-inert at every tested
     threshold — retained purely to cut scoring cost).
  5. G(2,4) SELECTION (`_score_g24`): every hypothesis scored over all K pairs
     by the inlier COUNT #(theta < tau) on the max principal angle theta
     between affine-Grassmann embeddings — position-aware, so it vetoes the
     direction-consistent flips that ordinary consensus accepts.  ONE
     parameter: tau_deg = 2.5 indoor / 3 outdoor.  Argmax wins.  END.

TESTED AND REJECTED (do not re-add): refinement of any kind (Cauchy G(2,4)
finite-difference, alternation-L2, R-locked LS, iterative re-gated LS — all
lose to none; the residual translation error equals its line-observable
projection, t_gr/t_e ~ 0.96, so it is correspondence-bound, not a manifold
artifact); a scale prior or clamp (the position-aware criterion cannot reward
collapsed scales); the graded 'prop' scoring kernel and the 3-rung tau ladder
(indistinguishable over 8 seeds, and the count needs a far smaller
indoor/outdoor tau gap with better outdoor scale); dropping the alpha
normalization, or reusing the matcher's whitened frame (catastrophic on
matcher-free outdoor sets — see Sim3Solver); 3-line and mixed 2L/3L minimal
sets; residual-based sign selection ('res'/'res2'/'resa' — validated, but a
wash, and its compute saving is capped at ~25% because random line-pair
angles concentrate exactly where the residual test is blind); the per-line
hemisphere canon for BOTH lines (~2x faster and tied on every shipped
benchmark, but degrades badly under large relative rotation: 7-Scenes
scrambled @5 13.6 vs 17.6).

NOTE single-run differences of 1-3 pairs on 37 cases are WITHIN RANSAC seed
noise (std ~2): compare configs over several `seed` values, never one run.

v27 caches, CPU: 7-Scenes 20.2 +/- 2.3 @5 over seeds (23/37 at seed 0; rot med
3.62 deg, <5 deg 28/37, s 4.6%, t 0.177 m) at ~80 ms/pair; KITTI mono_best 2/5
with SOLVE_SIM3_OUTDOOR; submaps matcher-bound.
"""
import numpy as np

# ── Plücker + the one scene normalization ───────────────────────────────────


def segments_to_plucker(p1, p2):
    d = p2 - p1
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    m = np.cross((p1 + p2) * 0.5, d)
    return np.concatenate([m, d], axis=1).astype(np.float32)


class Sim3Solver:
    """Plücker conversion + THE scene normalization, shared with the matcher.

    `alpha` = 1 / median reference foot-point radius; both clouds are scaled by
    it, so `p_q`/`p_r` are the normalized Plücker lines the solver's thresholds
    are calibrated in.  `matcher_input()` returns those SAME lines with the
    moment whitening the network expects — the normalization is computed once
    and the geometry is never scaled twice.  `solve_sim3` divides t by alpha,
    so poses come back in the caller's metric frame.

    Dropping alpha (raw metric units) was tried 2026-08-20 and REVERTED: on the
    5-pair matcher-fed KITTI cache raw looked better (@5 0.6 -> 1.6), but on the
    SAME maps with matcher-free correspondence sets it collapses — mono_best
    15%: 5/30 -> 1/30, rot med 6.3 -> 63 deg; submap 15%: 10/70 -> 1/70.
    Reusing the matcher's whitened frame instead is INTERMEDIATE and still
    loses (mono 3/30, submap 4/70): that factor is alpha/std ~ alpha*s, which
    normalizes the QUERY while the comparison happens in the REFERENCE frame,
    leaving the target of the comparison s times too large.  The rule (verified
    by frame swap): whichever cloud is the untransformed TARGET must sit at
    unit scale.  Matcher-free @5 3-way (7S/mono/submap), tau retuned per cell:
    alpha 58/5/10, raw 64/2/3, matcher-frame 60/3/4."""

    def __init__(self, q_ends, r_ends):
        q1, q2 = (np.asarray(a, float) for a in q_ends)
        r1, r2 = (np.asarray(a, float) for a in r_ends)
        pr_raw = segments_to_plucker(r1, r2)
        f = np.cross(pr_raw[:, :3], pr_raw[:, 3:])
        rad = float(np.median(np.linalg.norm(f - np.median(f, 0), axis=1)))
        self.alpha = 1.0 / (rad + 1e-9)
        self.q1, self.q2 = q1 * self.alpha, q2 * self.alpha
        self.r1, self.r2 = r1 * self.alpha, r2 * self.alpha
        self.p_q = segments_to_plucker(self.q1, self.q2)
        self.p_r = segments_to_plucker(self.r1, self.r2)

    def matcher_input(self):
        """(p_q, p_r) as the network expects them: the already-alpha-normalized
        Plücker lines with moments whitened by the query moment std.  Reuses
        p_q/p_r — no second Plücker conversion, no second scaling."""
        std = float(self.p_q[:, :3].std()) + 1e-6

        def w(p):
            return np.concatenate([p[:, :3] / std, p[:, 3:]], 1).astype(np.float32)

        return w(self.p_q), w(self.p_r)


def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def solve_translation_scale(L1, L2, R):
    """Linear 3Nx4 LS for (t, s) from m2 = s R m1 + t x (R d1). L*: (6, N).
    Not used by solve_sim3 (which batches it) — kept for baselines/solvers.py."""
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
    return np.vstack([s * (R @ L[:3]) + _skew(t) @ d_out, d_out])


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
# (SCALAR derivation note; sign-invariant; per-line weights w keep both blocks
# closed-form, which is what would make this usable as a differentiable head)


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


# ── hypothesis scoring ──────────────────────────────────────────────────────


def _score_g24(plq, plr, Rs, ts, ss, tau_rad, chunk=1024):
    """Per-hypothesis G(2,4) inlier count over ALL candidate pairs: #(theta <
    tau), theta = max principal angle between the affine-Grassmann embeddings.
    The single hottest stage (~60-65% of solver runtime, measured), so it
    thresholds cos(theta) against cos(tau) directly instead of taking arccos of
    every hypothesis x pair, and uses matmul (batched BLAS) for the Gram."""
    mq, dq, mr, dr = plq[:3], plq[3:], plr[:3], plr[3:]
    Yr = _yz_np(np.cross(mr.T, dr.T), dr.T)
    cos_tau = np.cos(tau_rad)
    out = np.empty(len(Rs))
    for lo in range(0, len(Rs), chunk):
        Rc, tc, sc = Rs[lo:lo + chunk], ts[lo:lo + chunk], ss[lo:lo + chunk]
        d_t = Rc @ dq                                   # (H,3,K)
        m_t = sc[:, None, None] * (Rc @ mq) + np.cross(tc[:, :, None], d_t, axis=1)
        Yq = _yz_np(np.transpose(np.cross(m_t, d_t, axis=1), (0, 2, 1)),
                    np.transpose(d_t, (0, 2, 1)))
        # Gram of the two 4x2 bases; matmul dispatches to batched BLAS, which
        # is ~10x faster than the equivalent einsum and bit-identical.
        out[lo:lo + chunk] = (_s2_np(np.matmul(np.swapaxes(Yq, -2, -1), Yr))
                              > cos_tau).sum(1)
    return out


# ── optional GPU backend ────────────────────────────────────────────────────
# `solve_sim3(..., device='cuda')` offloads the two heavy stages to torch.  The
# cheap stages (top-k, sampling, skew gate, minimal solve) stay in numpy — ~9%
# of runtime, and moving them would only add transfers.  torch is imported
# lazily, so the CPU path keeps numpy as its only dependency.
#
# The two stages do NOT want the same device.  Measured per stage on an RTX
# 5090 (fp32), hypotheses = 2 x nsamp:
#
#   stage            4k hyp (shipped)   16k     64k     256k
#   G(2,4) scoring   48.4 -> 6.1 ms      -- biggest win at every size --
#   alternation      22.4 -> 23.2 ms    2.11x   3.27x   3.87x
#
# Scoring is a dense batched matmul and wins everywhere (7.9x even at the
# shipped size); the alternation is dominated by batched 3x3 SVD plus many
# small kernels, so it only pays above ~8k hypotheses (= GPU_ALT_MIN).  Below
# that it stays on the CPU even when device='cuda', so the flag is never a
# pessimization.  End to end, same card:
#
#   nsamp     hyps     cpu ms   cuda fp32   cuda fp64
#     2000     4000       71.1    52.5 1.35x  63.8 1.11x
#     8000    16000      244.2   210.3 1.16x 263.5 0.93x
#    32000    64000     1098.1   429.7 2.56x 923.1 1.19x
#   128000   256000     4680.2  1279.9 3.66x 3764.4 1.24x
#
# So: the GPU is worth ~1.3-1.9x at the shipped nsamp and 2.5-3.7x if you raise
# it.  (Caveat: measured while two trainings held the card at 100% — an idle
# GPU will do better, and these ratios are noisy at the smaller sizes.)
#
# gpu_dtype: 'float32' (default) is the reason to use a GPU at all.  'float64'
# reproduces the CPU path EXACTLY (rot diff 0, s diff 0 on every benchmark
# pair) but only while the alternation is on the CPU — once it moves to the GPU
# fp64 collapses to ~1.2x, because consumer cards run fp64 at 1/64 rate.
#
# fp32 is NOT bit-identical to the CPU path (different reduction orders): the
# G(2,4) score is an integer count, so a hypothesis within fp noise of the tau
# boundary can tip either way and change the argmax.  Judge a device the way
# you judge any config change — multi-seed benchmark means, not per-pair
# equality.  Measured fp32 vs CPU: 7-Scenes @5 20.2+/-1.7 vs 20.2+/-2.3,
# matcher-free and mono_best identical to the decimal.

GPU_ALT_MIN = 8000        # hypotheses below which the alternation stays on CPU


def _torch():
    import torch
    return torch


def _yz_t(p0, d):                                 # (...,3),(...,3) -> (...,4,2)
    t = _torch()
    one = t.ones_like(p0[..., :1])
    c0 = t.cat([p0, one], -1)
    c0 = c0 / (c0.norm(dim=-1, keepdim=True) + 1e-12)
    c1 = t.cat([d, t.zeros_like(d[..., :1])], -1)
    c1 = c1 - c0 * (c1 * c0).sum(-1, keepdim=True)
    c1 = c1 / (c1.norm(dim=-1, keepdim=True) + 1e-12)
    return t.stack([c0, c1], -1)


def _s2_t(M):                                     # cos(MAX principal angle)
    t = _torch()
    F = (M ** 2).sum((-1, -2))
    det = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
    disc = t.sqrt(t.clamp(F * F - 4 * det * det, min=0.0))
    return t.sqrt(t.clamp((F - disc) * 0.5, 0.0, 1.0))


def _alternate_t(dq, fq, dr, fr, R, tr, s, n_iter):
    """torch mirror of `_alternate` (unweighted; the CPU version's `w` hook is
    for a differentiable head, which this backend does not serve)."""
    t = _torch()
    eta = 1.0 / t.sqrt(1.0 + (fr ** 2).sum(-1))
    y = eta[..., None] * fr
    for _ in range(n_iter):
        u = t.einsum('nij,nlj->nli', R, dq)
        q = s[:, None, None] * t.einsum('nij,nlj->nli', R, fq) + tr[:, None, :]
        lam = (u * dr).sum(-1) / s[:, None]
        qp = q - u * (u * q).sum(-1, keepdim=True)
        yp = y - u * (u * y).sum(-1, keepdim=True)
        gam = ((qp * yp).sum(-1) + eta) / ((qp ** 2).sum(-1) + 1.0)
        alp = (u * (y - gam[..., None] * q)).sum(-1) / s[:, None]
        x_dir = dq * lam[..., None]
        x_aff = dq * alp[..., None] + gam[..., None] * fq
        H = (gam * gam).sum(1)
        Hs = t.where(H < 1e-12, t.ones_like(H), H)
        mux = (gam[..., None] * x_aff).sum(1) / Hs[:, None]
        muy = (gam[..., None] * y).sum(1) / Hs[:, None]
        xa = x_aff - gam[..., None] * mux[:, None, :]
        ya = y - gam[..., None] * muy[:, None, :]
        M = t.einsum('nli,nlj->nij', dr, x_dir) + t.einsum('nli,nlj->nij', ya, xa)
        U, S_, Vt = t.linalg.svd(M)
        det = t.linalg.det(U @ Vt)
        D = t.zeros_like(M); D[:, 0, 0] = D[:, 1, 1] = 1; D[:, 2, 2] = det
        Rn = U @ D @ Vt
        Sx = (x_dir ** 2).sum(-1).sum(1) + (xa ** 2).sum(-1).sum(1)
        sn = t.clamp((S_[:, 0] + S_[:, 1] + det * S_[:, 2]) /
                     t.where(Sx < 1e-12, t.ones_like(Sx), Sx), 1e-3, 1e3)
        tn = muy - sn[:, None] * t.einsum('nij,nj->ni', Rn, mux)
        keep = ~((H < 1e-12) | (Sx < 1e-12) | ~t.isfinite(sn))
        R = t.where(keep[:, None, None], Rn, R)
        s = t.where(keep, sn, s)
        tr = t.where(keep[:, None], tn, tr)
    u = t.einsum('nij,nlj->nli', R, dq)
    q = s[:, None, None] * t.einsum('nij,nlj->nli', R, fq) + tr[:, None, :]
    cdir = (1.0 - (u * dr).sum(-1) ** 2).sum(1)
    qp = q - u * (u * q).sum(-1, keepdim=True)
    yp = y - u * (u * y).sum(-1, keepdim=True)
    gam = ((qp * yp).sum(-1) + eta) / ((qp ** 2).sum(-1) + 1.0)
    caff = ((yp - gam[..., None] * qp) ** 2).sum((1, 2)) + ((eta - gam) ** 2).sum(1)
    return R, tr, s, cdir + caff


def _score_g24_t(plq, plr, Rs, ts, ss, tau_rad, chunk=4096):
    """torch mirror of `_score_g24`; inputs/outputs are torch tensors."""
    t = _torch()
    mq, dq, mr, dr = plq[:3], plq[3:], plr[:3], plr[3:]
    Yr = _yz_t(t.linalg.cross(mr.T, dr.T), dr.T)
    cos_tau = float(np.cos(tau_rad))
    out = t.empty(len(Rs), device=Rs.device, dtype=Rs.dtype)
    for lo in range(0, len(Rs), chunk):
        Rc, tc, sc = Rs[lo:lo + chunk], ts[lo:lo + chunk], ss[lo:lo + chunk]
        d_t = Rc @ dq                                   # (H,3,K)
        m_t = sc[:, None, None] * (Rc @ mq) \
            + t.linalg.cross(tc[:, :, None].expand_as(d_t), d_t, dim=1)
        Yq = _yz_t(t.linalg.cross(m_t, d_t, dim=1).transpose(1, 2),
                   d_t.transpose(1, 2))
        out[lo:lo + chunk] = (_s2_t(Yq.transpose(-2, -1) @ Yr)
                              > cos_tau).sum(1).to(out.dtype)
    return out


# ── THE solver ──────────────────────────────────────────────────────────────


def solve_sim3(solver, prob, topk=200, nsamp=2000, tau_deg=2.5,
               skew_min=0.05, cost_max=0.12, alt_iters=3, seed=0,
               device='cpu', gpu_dtype='float32'):
    """THE Sim(3) estimate from a matcher probability matrix — the module
    docstring carries the pipeline, the frozen defaults and the rejected
    variants.  Defaults = INDOOR profile; OUTDOOR = `SOLVE_SIM3_OUTDOOR`.
    Returns (R, t, s) with t in the caller's original metric frame.

    `device='cuda'` offloads the G(2,4) scoring (and, above GPU_ALT_MIN
    hypotheses, the alternation) to torch: ~1.3-1.9x at the shipped nsamp,
    ~3.7x at nsamp 128000.  Results are statistically equivalent, NOT
    bit-identical (`gpu_dtype='float64'` is exact) — see the GPU backend
    section above."""
    nq = prob.shape[1]; k = min(topk, prob.size)
    flat = np.argpartition(prob.ravel(), -k)[-k:]
    ir, iq = flat // nq, flat % nq
    plq = solver.p_q[iq].T.astype(float); plr = solver.p_r[ir].T.astype(float)
    K = plq.shape[1]
    rng = np.random.default_rng(seed)
    T = rng.integers(0, K, (nsamp, 2))
    T = T[T[:, 0] != T[:, 1]]
    if skew_min > 0:                # scale-free observability gate, PRE-solve
        ok = np.ones(len(T), bool)
        for pl in (plq, plr):
            f_all = _feet(pl[3:].T, pl[:3].T)
            thr = skew_min * (float(np.median(
                np.linalg.norm(f_all, axis=1))) + 1e-9)
            a, b = pl[3:, T[:, 0]].T, pl[3:, T[:, 1]].T
            ok &= np.abs((np.cross(a, b)
                          * (f_all[T[:, 1]] - f_all[T[:, 0]])).sum(1)) > thr
        T = T[ok]
    if len(T) == 0:
        return np.eye(3), np.zeros(3), 1.0
    dq = np.stack([plq[3:, T[:, l]].T for l in range(2)], 1)
    dr = np.stack([plr[3:, T[:, l]].T for l in range(2)], 1)
    mq = np.stack([plq[:3, T[:, l]].T for l in range(2)], 1)
    mr = np.stack([plr[:3, T[:, l]].T for l in range(2)], 1)
    fq, fr = _feet(dq, mq), _feet(dr, mr)

    # SIGN: canonicalize query line 1, propose BOTH signs of line 2 and let
    # G(2,4) arbitrate.  Both proposals are solved as ONE batch of 2*NS
    # samples.  The canon feeds ONLY the init — `_alternate` gets the raw
    # quantities (it is sign-invariant, and the feet f = d x m already are).
    s1 = _maxabs_sign(dq[:, :1])[:, 0]
    sq = np.concatenate([np.stack([s1, s1], 1), np.stack([s1, -s1], 1)])
    dq2, mq2, fq2 = (np.concatenate([a, a]) for a in (dq, mq, fq))
    dr2, mr2, fr2 = (np.concatenate([a, a]) for a in (dr, mr, fr))
    sgr = _maxabs_sign(dr2)[..., None]
    dqc, mqc = dq2 * sq[..., None], mq2 * sq[..., None]
    Rs = _procrustes_batch(dqc, dr2 * sgr)
    ts, ss = _solve_ts_batch(mqc, dqc, mr2 * sgr, Rs)
    ss = np.clip(np.abs(ss), 1e-3, 1e3)

    if device == 'cpu':
        Rs, ts, ss, cost = _alternate(dq2, fq2, dr2, fr2, Rs, ts, ss, alt_iters)
        keep = np.isfinite(cost)                     # consistency filter
        if cost_max > 0:
            keep &= cost < cost_max
        Rs, ts, ss = Rs[keep], ts[keep], ss[keep]
        if len(Rs) == 0:
            return np.eye(3), np.zeros(3), 1.0
        sc = _score_g24(plq, plr, Rs, ts, ss, np.deg2rad(tau_deg))
        bi = int(np.argmax(sc))
        return Rs[bi], np.asarray(ts[bi]) / solver.alpha, float(ss[bi])

    torch = _torch()                                 # GPU backend
    dt = getattr(torch, gpu_dtype)

    def to(a):
        return torch.as_tensor(np.ascontiguousarray(a), device=device, dtype=dt)

    if len(dq2) >= GPU_ALT_MIN:                      # big batch: alternate on GPU
        Rs, ts, ss, cost = _alternate_t(to(dq2), to(fq2), to(dr2), to(fr2),
                                        to(Rs), to(ts), to(ss), alt_iters)
        keep = torch.isfinite(cost)
        if cost_max > 0:
            keep &= cost < cost_max
        Rs, ts, ss = Rs[keep], ts[keep], ss[keep]
    else:                                            # small batch: CPU is faster
        Rs, ts, ss, cost = _alternate(dq2, fq2, dr2, fr2, Rs, ts, ss, alt_iters)
        keep = np.isfinite(cost)
        if cost_max > 0:
            keep &= cost < cost_max
        Rs, ts, ss = to(Rs[keep]), to(ts[keep]), to(ss[keep])
    if len(Rs) == 0:
        return np.eye(3), np.zeros(3), 1.0
    sc = _score_g24_t(to(plq), to(plr), Rs, ts, ss, np.deg2rad(tau_deg))
    bi = int(torch.argmax(sc))
    R = Rs[bi].double().cpu().numpy()
    U, _, Vt = np.linalg.svd(R)          # fp32 drifts off SO(3) by ~1e-6;
    D = np.diag([1.0, 1.0, np.linalg.det(U @ Vt)])   # re-project so callers
    return (U @ D @ Vt,                  # can rely on R being orthonormal
            ts[bi].double().cpu().numpy() / solver.alpha,
            float(ss[bi]))


# Outdoor/high-noise profile (KITTI mono_best-validated):
SOLVE_SIM3_OUTDOOR = dict(tau_deg=3.0, skew_min=0.05, nsamp=4000)
