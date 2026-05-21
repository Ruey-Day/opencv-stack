"""
differentiable_solver.py
========================
Differentiable weighted SIM(3) solver for Plücker line correspondences.

Given a soft correspondence matrix P (B, N, M) and Plücker line sets
p1 (B, N, 6) and p2 (B, M, 6), estimates the SIM(3) transformation
(R, s, t) via:

  1. Weighted direction Procrustes (SVD) for R.
  2. Weighted least-squares from the moment constraint for (s, t):
         m2 = s · R m1  +  t × d2

The solver is fully differentiable w.r.t. prob_matrix, so gradients flow
from pose error back into the correspondence network.
"""
import torch


def _batch_skew(v):
    """Build batched 3×3 skew-symmetric matrices.
    Args:
        v: (B, N, 3)
    Returns:
        (B, N, 3, 3)
    """
    B, N, _ = v.shape
    z = torch.zeros(B, N, device=v.device, dtype=v.dtype)
    row0 = torch.stack([z,        -v[..., 2],  v[..., 1]], dim=-1)
    row1 = torch.stack([v[..., 2],  z,         -v[..., 0]], dim=-1)
    row2 = torch.stack([-v[..., 1], v[..., 0],  z       ], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)  # [B, N, 3, 3]


def soft_sim3(prob_matrix, plucker1, plucker2, reg=1e-4):
    """
    Differentiable weighted SIM(3) estimate from soft correspondences.

    Args:
        prob_matrix: (B, N, M) correspondence probabilities
        plucker1:   (B, N, 6) [m; d] source Plücker lines
        plucker2:   (B, M, 6) [m; d] target Plücker lines
        reg:        regularisation for SVD and linear solve stability

    Returns:
        R_est: (B, 3, 3)
        s_est: (B,)
        t_est: (B, 3)
        w:     (B, N) per-source confidence weights
    """
    B = prob_matrix.shape[0]

    # Normalise each source row to get soft assignment weights
    P_norm = prob_matrix / (prob_matrix.sum(dim=-1, keepdim=True) + 1e-8)

    # Per-source confidence: total probability mass assigned
    w = prob_matrix.sum(dim=-1)  # [B, N]

    # Soft matched target Plücker line per source line
    q = torch.bmm(P_norm, plucker2)  # [B, N, 6]

    d1      = plucker1[:, :, 3:]   # [B, N, 3] source unit directions
    d2_soft = q[:, :, 3:]          # [B, N, 3] soft target directions
    m1      = plucker1[:, :, :3]   # [B, N, 3] source moments
    m2_soft = q[:, :, :3]          # [B, N, 3] soft target moments

    # ── Weighted direction Procrustes → R ────────────────────────────────────
    # H = Σ_n w_n · d2_n ⊗ d1_n   [B, 3, 3]
    H = torch.einsum('bn,bni,bnj->bij', w, d2_soft, d1)
    H = H + reg * torch.eye(3, device=H.device, dtype=H.dtype).unsqueeze(0)

    U, _, Vh = torch.linalg.svd(H)
    det = torch.linalg.det(U @ Vh)   # [B]
    diag = torch.diag_embed(torch.stack([
        torch.ones(B, device=H.device, dtype=H.dtype),
        torch.ones(B, device=H.device, dtype=H.dtype),
        det.sign(),
    ], dim=-1))                       # [B, 3, 3]
    R_est = U @ diag @ Vh             # [B, 3, 3]

    # ── Weighted LS for (s, t) from moment constraint ─────────────────────
    # m2 = s · R m1 + t × d2  →  [R m1 | −skew(d2)] [s; t] = m2
    m1_rot  = torch.einsum('bij,bnj->bni', R_est, m1)  # [B, N, 3]
    skew_d2 = _batch_skew(d2_soft)                      # [B, N, 3, 3]

    # A: coefficient matrix [B, N, 3, 4]
    A = torch.cat([m1_rot.unsqueeze(-1), -skew_d2], dim=-1)

    # Weighted normal equations
    ATA = torch.einsum('bn,bnrj,bnrk->bjk', w, A, A)   # [B, 4, 4]
    ATb = torch.einsum('bn,bnrj,bnr->bj',  w, A, m2_soft)  # [B, 4]

    reg_mat = reg * torch.eye(4, device=ATA.device, dtype=ATA.dtype).unsqueeze(0)
    x = torch.linalg.solve(ATA + reg_mat, ATb)          # [B, 4]

    s_est = x[:, 0]    # [B]
    t_est = x[:, 1:]   # [B, 3]

    return R_est, s_est, t_est, w


def sim3_pose_loss(R_est, s_est, R_gt, s_gt):
    """
    Rotation + scale loss between estimated and GT SIM(3).

    Rotation: normalised Frobenius ||R_est − R_gt||_F² / 4  ∈ [0, 1]
    Scale:    (log s_est − log s_gt)²

    Translation is not penalised — it is underconstrained when the
    correspondence weights are diffuse over a flat scene.

    Args:
        R_est: (B, 3, 3)
        s_est: (B,)
        R_gt:  (B, 3, 3)  will be moved to R_est's device
        s_gt:  (B,)       will be moved to R_est's device

    Returns:
        scalar loss, mean over valid (s_est > 0) samples
    """
    device = R_est.device
    R_gt = R_gt.to(device).float()
    s_gt = s_gt.to(device).float()
    if s_gt.dim() > 1:
        s_gt = s_gt.squeeze(-1)

    L_rot   = ((R_est - R_gt) ** 2).sum(dim=(-2, -1)) / 4.0          # [B]
    s_clamp = s_est.clamp(min=0.05, max=50.0)
    L_scale = (torch.log(s_clamp) - torch.log(s_gt.clamp(min=0.05))) ** 2  # [B]

    # Skip pairs with invalid estimated scale OR invalid GT scale (zero-overlap pairs)
    valid   = ((s_est > 0.05) & (s_gt > 0.05)).float()
    n_valid = valid.sum().clamp(min=1.0)
    return (valid * (L_rot + L_scale)).sum() / n_valid
