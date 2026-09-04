
import torch
import torch.nn.functional as F

def correspondenceProbabilityBCE(P, C):
    num_pos = F.relu(C.sum(dim=(-2,-1))-1.0) + 1.0
    num_neg = F.relu((1.0- C).sum(dim=(-2,-1)) -1.0) + 1.0

    loss = ((P + 1e-20).log() * C).sum(dim=(-2,-1)) * 0.5 / num_pos
    loss += ((1.0 - P + 1e-20 ).log() * (1.0 - C)).sum(dim=(-2,-1)) * 0.5 / num_neg

    return -loss

def correspondenceLoss(P, C_gt):
    # Using precomputed C_gt
    return correspondenceProbabilityBCE(P, C_gt).mean() # [-1, 1)

def matchabilityLoss(r, c, C_gt, eps=1e-12):
    """Supervise the Sinkhorn marginal heads (r over plucker1/ref rows, c over
    plucker2/query cols) with GT matchability. r and c are softmax
    distributions over lines; the target is the normalized indicator of
    lines that have at least one GT correspondence, so mass is pushed off
    matchless lines (the balanced-OT pollution: v21 matchability change).
    Cross-entropy per sample, mean over the batch; samples with no GT
    matches contribute 0."""
    m1 = (C_gt.sum(dim=-1) > 0).float()               # (B, N1)
    m2 = (C_gt.sum(dim=-2) > 0).float()               # (B, N2)
    s1 = m1.sum(dim=-1, keepdim=True)
    s2 = m2.sum(dim=-1, keepdim=True)
    p1 = m1 / s1.clamp_min(1.0)
    p2 = m2 / s2.clamp_min(1.0)
    ce1 = -(p1 * (r + eps).log()).sum(dim=-1) * (s1.squeeze(-1) > 0)
    ce2 = -(p2 * (c + eps).log()).sum(dim=-1) * (s2.squeeze(-1) > 0)
    return (ce1 + ce2).mean()


def dustbinLoss(P, C_gt, eps=1e-12):
    """Supervise the OT DUSTBIN (SuperGlue-style), which nothing else does.

    THE BUG THIS FIXES (measured 2026-08-29, v50): prob_mat_sinkhorn returns
    only the (B,Nr,Nq) slice, so the bin row/column never enters the loss.
    That leaves a DEGENERATE gradient -- pushing bin_score down makes the bin
    unattractive, so more mass stays in the real block, so P on true matches
    rises, so the BCE falls. Nothing opposes it, and v50 duly drove bin_score
    1.0 -> -13.5, at which point the bin holds 0.01% of each row's mass (vs
    82.6% at init): the dustbin was switched fully OFF and the run sat 7.7
    points below the no-dustbin control at epoch 45.

    Why the BCE alone cannot hold it: early in training the model cannot yet
    identify matches, so mass in the bin sends log P(true match) toward -inf,
    and the positive term (weighted 0.5/num_pos, large when matches are few)
    dominates. The model evacuates the bin to protect it.

    The bin entries need no plumbing: each real row of the FULL augmented plan
    sums to 1, so bin_i = 1 - sum_j P[i,j] exactly (verified numerically).
    Unmatched lines are supervised TO the bin, which anchors bin_score.
    """
    row_bin = (1.0 - P.sum(dim=-1)).clamp_min(eps)          # (B, Nr)
    col_bin = (1.0 - P.sum(dim=-2)).clamp_min(eps)          # (B, Nq)
    unm_r = (C_gt.sum(dim=-1) == 0).float()
    unm_c = (C_gt.sum(dim=-2) == 0).float()
    lr = -(row_bin.log() * unm_r).sum(-1) / unm_r.sum(-1).clamp_min(1.0)
    lc = -(col_bin.log() * unm_c).sum(-1) / unm_c.sum(-1).clamp_min(1.0)
    return 0.5 * (lr + lc).mean()


class TotalLoss(torch.nn.Module):
    def __init__(self, match_w: float = 0.0, dustbin_w: float = 0.0):
        super(TotalLoss, self).__init__()
        self.match_w = match_w
        self.dustbin_w = dustbin_w
    def forward(self, P, C_gt, r=None, c=None):
        loss = correspondenceLoss(P, C_gt).view(1)
        if self.match_w > 0.0 and r is not None:
            loss = loss + self.match_w * matchabilityLoss(r, c, C_gt).view(1)
        if self.dustbin_w > 0.0:
            loss = loss + self.dustbin_w * dustbinLoss(P, C_gt).view(1)
        return loss
