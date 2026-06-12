"""
Asymmetric Alternating Attention variant of PluckerNetKnn.

Motivation (FUSER, Jiang et al. CVPR 2026)
-------------------------------------------
The original SpatialAttentionalGNN shares a single AttentionalPropagation module
between both plucker1 (SLAM) and plucker2 (metric) updates at every layer.  This
is reasonable for symmetric matching problems but sub-optimal here because the two
sides are structurally different:

  plucker1 — SLAM lines: arbitrary scale, noisy moments, sparse subset of the scene
  plucker2 — metric lines: metric scale, accurate, dense reference map

This module replaces the shared GNN with AlternatingAttentionBlocks, each containing
four *independent* AttentionalPropagation modules:

  self0  — SLAM lines    attending to SLAM lines    (within-source context)
  self1  — metric lines  attending to metric lines  (within-target context)
  cross0 — SLAM lines    attending to metric lines  (source reads target)
  cross1 — metric lines  attending to SLAM lines    (target reads source)

Separating self and cross attention per side lets the network learn that
"geometric consistency among SLAM lines" and "geometric consistency among metric
map lines" are different statistical patterns, and that the information flow
from metric→SLAM (scale correction signal) differs from SLAM→metric.

Parameter parity
----------------
Default n_blocks=3 gives 3 × 4 = 12 AttentionalPropagation modules, matching
the 12 modules in the default SpatialAttentionalGNN (GNN_layers=['self','cross']*6).
Increase n_blocks for more expressive models at the cost of compute.
"""

import os
import sys

import torch
import torch.nn as nn

# Ensure PlueckerNet is importable when this module is loaded directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PLUCKERNET = os.path.join(_HERE, '..', 'PlueckerNet')
if _PLUCKERNET not in sys.path:
    sys.path.insert(0, _PLUCKERNET)

from model.model_plucker import (
    AttentionalPropagation,
    MLP,
    conv_in_seq_direction_moment_knn,
    pairwiseL2Dist,
    prob_mat_sinkhorn,
)


class AlternatingAttentionBlock(nn.Module):
    """
    One alternating block with four independent attention modules.

    Pass order within each block:
      1. self0  : SLAM   → SLAM   (within-source geometric context)
      2. self1  : metric → metric (within-target geometric context)
      3. cross0 : SLAM   → metric (source attends target for scale/pose signal)
      4. cross1 : metric → SLAM   (target attends source to localise matches)
    """

    def __init__(self, feature_dim: int, num_heads: int = 4):
        super().__init__()
        self.self0  = AttentionalPropagation(feature_dim, num_heads)
        self.self1  = AttentionalPropagation(feature_dim, num_heads)
        self.cross0 = AttentionalPropagation(feature_dim, num_heads)
        self.cross1 = AttentionalPropagation(feature_dim, num_heads)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        # (B, D, N) — residual updates applied sequentially
        desc0 = desc0 + self.self0(desc0, desc0)
        desc1 = desc1 + self.self1(desc1, desc1)
        desc0 = desc0 + self.cross0(desc0, desc1)
        desc1 = desc1 + self.cross1(desc1, desc0)
        return desc0, desc1


class AsymmetricAlternatingGNN(nn.Module):
    """
    Stack of AlternatingAttentionBlocks followed by the same global-context
    regression head used in SpatialAttentionalGNN.

    Args:
        feature_dim: channel width (must match net_nchannel)
        n_blocks:    number of alternating blocks (default 3 → 12 attn modules,
                     matching the default 12-layer SpatialAttentionalGNN)
        num_heads:   attention heads per module (default 4)
    """

    def __init__(self, feature_dim: int, n_blocks: int = 3, num_heads: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([
            AlternatingAttentionBlock(feature_dim, num_heads)
            for _ in range(n_blocks)
        ])
        # Same regression head as SpatialAttentionalGNN — concat global stats
        # from the opposite side before projecting to matchability logits.
        self.mlp = MLP([feature_dim * 3, feature_dim * 2, feature_dim * 2, feature_dim])

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        for block in self.blocks:
            desc0, desc1 = block(desc0, desc1)

        # Global-context regression (identical to SpatialAttentionalGNN)
        desc0_global = torch.cat(
            (desc0.mean(dim=-1, keepdim=True), desc0.max(dim=-1, keepdim=True)[0]),
            dim=-2,
        ).repeat(1, 1, desc1.size(-1))
        desc1_global = torch.cat(
            (desc1.mean(dim=-1, keepdim=True), desc1.max(dim=-1, keepdim=True)[0]),
            dim=-2,
        ).repeat(1, 1, desc0.size(-1))

        desc0_regress = self.mlp(torch.cat((desc0, desc1_global), dim=-2))
        desc1_regress = self.mlp(torch.cat((desc1, desc0_global), dim=-2))

        return desc0, desc1, desc0_regress, desc1_regress


class FeatureExtractorGraphAlt(nn.Module):
    """Drop-in replacement for FeatureExtractorGraph using AsymmetricAlternatingGNN."""

    def __init__(self, config, in_channel: int):
        super().__init__()
        d = config['net_nchannel']
        n_blocks = int(getattr(config, 'alt_n_blocks', 3))

        self.conv_in    = conv_in_seq_direction_moment_knn(d, in_channel=in_channel)
        self.gnn        = AsymmetricAlternatingGNN(d, n_blocks=n_blocks, num_heads=4)
        self.final_proj = nn.Conv1d(d, d, kernel_size=1, bias=True)
        self.regress    = nn.Conv1d(d, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        desc0, desc1, x_regress, y_regress = self.gnn(self.conv_in(x), self.conv_in(y))
        mdesc0 = self.final_proj(desc0)
        mdesc1 = self.final_proj(desc1)
        x_prob = self.regress(x_regress).softmax(dim=-1)  # (B, 1, N)
        y_prob = self.regress(y_regress).softmax(dim=-1)  # (B, 1, N)
        return mdesc0, mdesc1, x_prob, y_prob


class PluckerNetKnnAlt(nn.Module):
    """
    PluckerNet with asymmetric alternating attention.

    Identical forward interface to PluckerNetKnn:
        forward(plucker1, plucker2) -> (P, r, c)

    Select via --model alt in train.py.
    """

    def __init__(self, config):
        super().__init__()
        self.config     = config
        self.in_channel = getattr(config, 'in_channel', 6)

        self.FeatureExtractor = FeatureExtractorGraphAlt(config, self.in_channel)
        self.pairwiseL2Dist   = pairwiseL2Dist
        self.sinkhorn         = prob_mat_sinkhorn(
            config, config.net_lambda, 1e-9, config.net_maxiter
        )

    def forward(self, plucker1: torch.Tensor, plucker2: torch.Tensor):
        p1f, p2f, p1_prob, p2_prob = self.FeatureExtractor(
            plucker1.transpose(-2, -1),
            plucker2.transpose(-2, -1),
        )
        p1f = p1f.transpose(-2, -1)
        p2f = p2f.transpose(-2, -1)
        p1f = torch.nn.functional.normalize(p1f, p=2, dim=-1)
        p2f = torch.nn.functional.normalize(p2f, p=2, dim=-1)

        M = self.pairwiseL2Dist(p1f, p2f)
        r = p1_prob.squeeze(1)   # (B, N1)
        c = p2_prob.squeeze(1)   # (B, N2)
        P = self.sinkhorn(M, r, c)
        return P, r, c
