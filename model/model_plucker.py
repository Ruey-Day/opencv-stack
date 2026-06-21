import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


def knn(x, k):
    """k-nearest neighbours by L2. x: (B, D, N) → idx: (B, N, k)."""
    dist = torch.cdist(x.transpose(1, 2), x.transpose(1, 2))  # (B, N, N)
    return dist.topk(k=k, dim=-1, largest=False)[1]


def get_graph_feature(x, k=10, idx=None):
    """Edge features for each point from its k-nearest neighbours.
    x: (B, D, N) → (B, 2D, N, k)"""
    B, D, N = x.shape
    if idx is None:
        idx = knn(x, k=min(k, N))
    nb_knns = idx.size(-1)

    idx_flat = (idx + torch.arange(B, device=x.device).view(-1, 1, 1) * N).view(-1)
    x_t      = x.permute(0, 2, 1).contiguous()                      # (B, N, D)
    neighbors = x_t.view(B * N, D)[idx_flat].view(B, N, nb_knns, D)
    center    = x_t.view(B, N, 1, D).expand(-1, -1, nb_knns, -1)   # view, no alloc

    return torch.cat([neighbors - center, center], dim=3).permute(0, 3, 1, 2).contiguous()


def MLP(channels: list, do_gn=True):
    n, layers = len(channels), []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < n - 1:
            if do_gn:
                layers.append(nn.GroupNorm(4, channels[i]))
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


def pairwiseL2Dist(x1, x2):
    """Batched pairwise L2. (B,N,D),(B,M,D) → (B,N,M)."""
    x1_norm2 = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm2 = x2.pow(2).sum(dim=-1, keepdim=True)
    return torch.baddbmm(
        x2_norm2.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2
    ).add_(x1_norm2).clamp_min_(1e-30).sqrt_()


class prob_mat_sinkhorn(nn.Module):
    def __init__(self, config, mu=0.1, tolerance=1e-9, iterations=30):
        super().__init__()
        self.mu         = mu
        self.iterations = iterations
        self.eps        = 1e-12

    def forward(self, M, r=None, c=None):
        K = (-M / self.mu).exp()
        K = K / K.sum(dim=(-2, -1), keepdim=True).clamp_min_(self.eps)

        u = r.unsqueeze(-1)
        c = c.unsqueeze(-1)
        r = u.clone()
        v = torch.ones_like(c)
        # Fixed-count loop: eliminates per-iteration CUDA sync from while-loop norm check
        for _ in range(self.iterations):
            v = c / K.transpose(-2, -1).matmul(u).clamp_min_(self.eps)
            u = r / K.matmul(v).clamp_min_(self.eps)

        return (u * K) * v.transpose(-2, -1)


class conv_in_seq_direction_moment_knn(nn.Module):
    def __init__(self, out_channel: int, in_channel: int = 6):
        super().__init__()
        half = out_channel // 2

        self.conv_direction = nn.Conv2d(6, half // 8, 1)
        self.conv_moment    = nn.Conv2d((in_channel - 3) * 2, half // 8, 1)
        self.mlp_direction  = MLP([half // 8, half // 4, half // 2, half])
        self.mlp_moment     = MLP([half // 8, half // 4, half // 2, half])
        self.mlp_merged     = MLP([out_channel, out_channel, out_channel])

    def forward(self, x):
        # Compute knn index once on direction subspace, reuse for moment subspace
        dir_feat = x[:, :3, :]
        idx = knn(dir_feat, k=min(10, dir_feat.size(-1)))

        x_dir = self.mlp_direction(
            self.conv_direction(get_graph_feature(dir_feat, idx=idx)).mean(dim=-1))
        x_mom = self.mlp_moment(
            self.conv_moment(get_graph_feature(x[:, 3:, :], idx=idx)).mean(dim=-1))

        return self.mlp_merged(torch.cat([x_dir, x_mom], dim=1))


def attention(query, key, value):
    """Scaled dot-product attention — uses flash attention when available.
    (B, d, H, N) → (B, d, H, N)"""
    q = query.permute(0, 2, 3, 1)   # (B, H, N, d)
    k = key.permute(0, 2, 3, 1)
    v = value.permute(0, 2, 3, 1)
    return F.scaled_dot_product_attention(q, k, v).permute(0, 3, 1, 2), None


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim       = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj  = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        B = query.size(0)
        query, key, value = [l(x).view(B, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(B, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp  = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        return self.mlp(torch.cat([x, self.attn(x, source, source)], dim=1))


class SpatialAttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([AttentionalPropagation(feature_dim, 4)
                                     for _ in range(len(layer_names))])
        self.names = layer_names
        self.mlp   = MLP([feature_dim * 3, feature_dim * 2, feature_dim * 2, feature_dim])

    def forward(self, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            src0, src1 = (desc1, desc0) if name == 'cross' else (desc0, desc1)
            desc0 = desc0 + layer(desc0, src0)
            desc1 = desc1 + layer(desc1, src1)

        # Per-point matchability prior: each side gets global (mean+max) context from the other
        N0, N1 = desc0.size(-1), desc1.size(-1)
        g0 = torch.cat([desc0.mean(-1, keepdim=True), desc0.max(-1, keepdim=True)[0]], dim=1)
        g1 = torch.cat([desc1.mean(-1, keepdim=True), desc1.max(-1, keepdim=True)[0]], dim=1)

        # expand is zero-copy; cat materialises once
        desc0_reg = self.mlp(torch.cat([desc0, g1.expand(-1, -1, N0)], dim=1))
        desc1_reg = self.mlp(torch.cat([desc1, g0.expand(-1, -1, N1)], dim=1))

        return desc0, desc1, desc0_reg, desc1_reg


class FeatureExtractorGraph(nn.Module):
    def __init__(self, config, in_channel):
        super().__init__()
        nc = config['net_nchannel']
        self.conv_in    = conv_in_seq_direction_moment_knn(nc, in_channel=in_channel)
        self.gnn        = SpatialAttentionalGNN(nc, config['GNN_layers'])
        self.final_proj = nn.Conv1d(nc, nc, kernel_size=1, bias=True)
        self.regress    = nn.Conv1d(nc, 1,  kernel_size=1, bias=True)

    def forward(self, x, y):
        desc0, desc1, x_prob, y_prob = self.gnn(self.conv_in(x), self.conv_in(y))
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        x_prob = self.regress(x_prob).softmax(dim=-1)
        y_prob = self.regress(y_prob).softmax(dim=-1)
        return mdesc0, mdesc1, x_prob, y_prob


class PluckerNetKnn(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channel       = getattr(config, 'in_channel', 6)
        self.FeatureExtractor = FeatureExtractorGraph(config, self.in_channel)
        self.sinkhorn         = prob_mat_sinkhorn(config, config.net_lambda, 1e-9,
                                                  config.net_maxiter)

    def forward(self, plucker1, plucker2):
        f1, f2, p1, p2 = self.FeatureExtractor(plucker1.transpose(-2, -1),
                                                plucker2.transpose(-2, -1))
        f1 = F.normalize(f1.transpose(-2, -1), p=2, dim=-1)
        f2 = F.normalize(f2.transpose(-2, -1), p=2, dim=-1)
        M  = pairwiseL2Dist(f1, f2)
        r, c = p1.squeeze(1), p2.squeeze(1)
        return self.sinkhorn(M, r, c), r, c
