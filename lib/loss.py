
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


class TotalLoss(torch.nn.Module):
    def __init__(self, match_w: float = 0.0):
        super(TotalLoss, self).__init__()
        self.match_w = match_w
    def forward(self, P, C_gt, r=None, c=None):
        loss = correspondenceLoss(P, C_gt).view(1)
        if self.match_w > 0.0 and r is not None:
            loss = loss + self.match_w * matchabilityLoss(r, c, C_gt).view(1)
        return loss
