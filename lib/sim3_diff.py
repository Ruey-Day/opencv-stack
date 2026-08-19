"""sim3_diff.py — differentiable Sim(3) head for END-TO-END training (v1).

Torch port of the weighted closed-form machinery in lib/sim3_solver.py:
weighted hemisphere-canon Procrustes + joint (t,s) init, then the weighted
alternating descent on the Grassmann projection cost (`alternate_w`, the
weighted `_alternate`). Per training pair the head consumes the matcher's
top-K candidate correspondences with weights w = P[pairs] and returns ONE
differentiable pose; gradients reach the matcher through every weighted sum
(dpose/dP), teaching it which correspondences the solver should trust.

Protocol (train_e2e.py): warm-start iterations under no_grad, the last
`unroll` iterations carry gradients (flat memory, no deep SVD-backward
chains). Inference is untouched — lib/sim3_solver.solve_sim3 runs exactly as
benchmarked; E2E changes only the network weights.
"""
import torch


def _feet(d, m):
    return torch.linalg.cross(d, m)


def _hemi_sign(D):
    """Per-line max-|component| hemisphere canon (detached, like a ReLU mask)."""
    j = D.abs().argmax(-1, keepdim=True)
    s = torch.gather(D, -1, j).sign().squeeze(-1)
    return torch.where(s == 0, torch.ones_like(s), s).detach()


def weighted_init(dq, mq, dr, mr, w):
    """Weighted closed-form init: hemisphere-canon Procrustes on directions +
    weighted joint linear (t, s) from the moment constraint. All (L,3), w (L,).
    Differentiable in w and the line coordinates."""
    sq, sr = _hemi_sign(dq), _hemi_sign(dr)
    dqc, mqc = dq * sq[:, None], mq * sq[:, None]
    drc, mrc = dr * sr[:, None], mr * sr[:, None]
    M = torch.einsum('l,li,lj->ij', w, drc, dqc)
    U, _, Vt = torch.linalg.svd(M)
    det = torch.det(U @ Vt)
    one = torch.ones_like(det)
    R = U @ torch.diag_embed(torch.stack([one, one, det])) @ Vt
    Rd, Rm = dqc @ R.T, mqc @ R.T
    L = dq.shape[0]
    A = dq.new_zeros(L, 3, 4)
    A[:, 0, 1] = Rd[:, 2]; A[:, 0, 2] = -Rd[:, 1]
    A[:, 1, 0] = -Rd[:, 2]; A[:, 1, 2] = Rd[:, 0]
    A[:, 2, 0] = Rd[:, 1]; A[:, 2, 1] = -Rd[:, 0]
    A[:, :, 3] = Rm
    sw = w.clamp_min(0).sqrt()[:, None, None]
    Af = (A * sw).reshape(-1, 4)
    bf = (mrc * sw[:, :, 0]).reshape(-1)
    AtA = Af.T @ Af + 1e-9 * torch.eye(4, device=A.device, dtype=A.dtype)
    x = torch.linalg.solve(AtA, Af.T @ bf)
    return R, x[:3], x[3].abs().clamp(1e-3, 1e3)


def alternate_w(dq, fq, dr, fr, R, t, s, n_iter, w):
    """Weighted alternating descent (torch port of _alternate, single pair).
    dq,fq,dr,fr: (L,3); w: (L,). Returns updated (R, t, s)."""
    eta = 1.0 / torch.sqrt(1.0 + (fr ** 2).sum(-1))
    y = eta[:, None] * fr
    for _ in range(n_iter):
        u = dq @ R.T
        q = s * (fq @ R.T) + t
        qp = q - u * (u * q).sum(-1, keepdim=True)
        yp = y - u * (u * y).sum(-1, keepdim=True)
        gam = ((qp * yp).sum(-1) + eta) / ((qp ** 2).sum(-1) + 1.0)
        lam = (u * dr).sum(-1) / s
        alp = (u * (y - gam[:, None] * q)).sum(-1) / s
        x_dir = dq * lam[:, None]
        x_aff = dq * alp[:, None] + gam[:, None] * fq
        wg = w * gam
        H = (wg * gam).sum().clamp_min(1e-12)
        mux = (wg[:, None] * x_aff).sum(0) / H
        muy = (wg[:, None] * y).sum(0) / H
        xa = x_aff - gam[:, None] * mux
        ya = y - gam[:, None] * muy
        M = torch.einsum('l,li,lj->ij', w, dr, x_dir) \
            + torch.einsum('l,li,lj->ij', w, ya, xa)
        U, S_, Vt = torch.linalg.svd(M)
        det = torch.det(U @ Vt)
        one = torch.ones_like(det)
        Rn = U @ torch.diag_embed(torch.stack([one, one, det])) @ Vt
        Sx = ((w * (x_dir ** 2).sum(-1)).sum()
              + (w * (xa ** 2).sum(-1)).sum()).clamp_min(1e-12)
        sn = ((S_[0] + S_[1] + det * S_[2]) / Sx).clamp(1e-3, 1e3)
        tn = muy - sn * (Rn @ mux)
        R, t, s = Rn, tn, sn
    return R, t, s


def pose_head(l_ref, l_qry, w, warm=30, unroll=8):
    """Differentiable pose from weighted candidate correspondences.
    l_ref/l_qry: (L,6) Plücker [m;d] (prenormed frame); w: (L,) weights.
    Returns (R, t, s) mapping query -> reference."""
    mq, dq = l_qry[:, :3], l_qry[:, 3:]
    mr, dr = l_ref[:, :3], l_ref[:, 3:]
    fq, fr = _feet(dq, mq), _feet(dr, mr)
    with torch.no_grad():
        R0, t0, s0 = weighted_init(dq, mq, dr, mr, w)
        R0, t0, s0 = alternate_w(dq, fq, dr, fr, R0, t0, s0, warm, w)
    return alternate_w(dq, fq, dr, fr, R0.detach(), t0.detach(),
                       s0.detach(), unroll, w)


def pose_loss(R, t, s, R_gt, t_gt, s_gt, lam_t=0.5, lam_s=2.0):
    """Angle-based pose loss (prenormed frame). Returns (loss, rot_err_rad)."""
    cosang = (((R * R_gt).sum() - 1.0) / 2.0).clamp(-1 + 1e-7, 1 - 1e-7)
    rot = torch.acos(cosang)
    return rot + lam_t * (t - t_gt).norm() + lam_s * (s / s_gt).log().abs(), rot
