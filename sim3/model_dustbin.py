"""
PluckerNetKnnDustbin — SuperGlue-style dustbin extension of PluckerNetKnn.

Adds a learnable dustbin row and column to the (N×M) Sinkhorn probability
matrix, producing (N+1)×(M+1).  Unmatched or non-overlapping lines are
assigned to the dustbin rather than forced onto wrong correspondences.

Architecture is identical to PluckerNetKnn except:
  - bin_dist : learnable "distance" used to fill the dustbin row/col of M
  - bin_score: learnable "probability budget" added to r and c for the dustbin

Weights from a pre-trained PluckerNetKnn checkpoint can be loaded with
strict=False — only bin_dist and bin_score are newly initialized.
"""
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# PluckerNet lives one directory up
_PLUECKERNET = os.path.join(os.path.dirname(__file__), '..', '..', 'PlueckerNet')
sys.path.insert(0, os.path.abspath(_PLUECKERNET))

from model.model_plucker import FeatureExtractorGraph, pairwiseL2Dist, prob_mat_sinkhorn


class PluckerNetKnnDustbin(nn.Module):
    """PluckerNetKnn with a learnable dustbin row and column."""

    def __init__(self, config):
        super().__init__()
        self.config      = config
        self.in_channel  = getattr(config, 'in_channel', 6)

        self.FeatureExtractor = FeatureExtractorGraph(config, self.in_channel)
        self.pairwiseL2Dist   = pairwiseL2Dist

        mu          = config.net_lambda
        tolerance   = 1e-9
        iterations  = config.net_maxiter
        self.sinkhorn = prob_mat_sinkhorn(config, mu, tolerance, iterations)

        # Dustbin parameters
        # bin_dist       : distance placed in the extra row/col of the cost matrix
        #                  (initialised to ~median L2 distance between unit vectors ≈ 1.0)
        # bin_score_logit: logit for the dustbin probability budget via sigmoid.
        #                  sigmoid(0) = 0.5; keeping it in (0,1) prevents P_aug > 1
        #                  which would cause log(1-P) = NaN in BCE loss.
        self.bin_dist        = nn.Parameter(torch.tensor(1.0))
        self.bin_score_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid → 0.5

    def forward(self, plucker1, plucker2):
        """
        Args:
            plucker1: (B, N, 6)
            plucker2: (B, M, 6)

        Returns:
            P_aug : (B, N+1, M+1) — dustbin is last row/col
            r     : (B, N)        — matchability logits for set 1
            c     : (B, M)        — matchability logits for set 2
        """
        # Feature extraction (same as PluckerNetKnn)
        p1f, p2f, p1_prob, p2_prob = self.FeatureExtractor(
            plucker1.transpose(-2, -1), plucker2.transpose(-2, -1)
        )
        p1f = p1f.transpose(-2, -1)          # (B, N, 128)
        p2f = p2f.transpose(-2, -1)          # (B, M, 128)
        p1f = F.normalize(p1f, p=2, dim=-1)
        p2f = F.normalize(p2f, p=2, dim=-1)

        M   = self.pairwiseL2Dist(p1f, p2f)  # (B, N, M_sz)
        r   = p1_prob.squeeze(1)              # (B, N)
        c   = p2_prob.squeeze(1)              # (B, M)

        B, N, M_sz = M.shape

        # Dustbin distance: clamp away from zero so K never saturates to 1
        bd = self.bin_dist.abs().clamp(min=0.05)

        # Augment M → (B, N+1, M+1)
        #   last col  = bd  (cost of sending line_i to dustbin)
        #   last row  = bd  (cost of assigning line_j to dustbin)
        #   corner    = 0   (dustbin-to-dustbin, absorbed by normalisation)
        bin_col    = bd.expand(B, N, 1)
        bin_row    = bd.expand(B, 1, M_sz)
        corner     = torch.zeros(B, 1, 1, device=M.device, dtype=M.dtype)
        M_aug      = torch.cat(
            [torch.cat([M,       bin_col], dim=-1),   # (B, N,   M+1)
             torch.cat([bin_row, corner ], dim=-1)],  # (B, 1,   M+1)
            dim=-2                                     # (B, N+1, M+1)
        )

        # Augment marginals → (B, N+1) and (B, M+1)
        # sigmoid keeps bs in (0, 1) so every P_aug element stays ≤ 1
        # (each row i sums to r_aug[i] ≤ 1 → no element can exceed 1),
        # preventing log(1-P) from going NaN in the BCE loss.
        bs    = torch.sigmoid(self.bin_score_logit).clamp(min=0.01).expand(B, 1)
        r_aug = torch.cat([r, bs], dim=-1)
        c_aug = torch.cat([c, bs], dim=-1)

        P_aug = self.sinkhorn(M_aug, r_aug, c_aug)    # (B, N+1, M+1)
        return P_aug, r, c
