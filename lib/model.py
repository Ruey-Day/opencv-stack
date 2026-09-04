import math
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

def graff_knn_idx(raw, k):
    """AFFINE-GRASSMANNIAN KNN graph (v34, graff_knn flag) — user's proposal.

    Each line is a point on the affine Grassmannian Graff(1,3) ~ a 2-plane in
    R^4: span{ c0 = [p0; 1]/||.||,  c1 = orth([d; 0]) }.  This is exactly the
    representation the SOLVER already scores with (_yz_np / _s2_np), so the
    matcher's neighbourhood and the solver's inlier criterion finally use ONE
    geometric primitive.

    The CHORDAL metric on this manifold is L2-EMBEDDABLE: for the rank-2
    orthogonal projector P = Y Y^T,
        ||P_i - P_j||_F^2 = 2 (2 - sum_k cos^2 theta_k) = 2 (sin^2 t1 + sin^2 t2)
    so stacking the 10 unique entries of the symmetric 4x4 projector (sqrt(2)
    on the off-diagonals) turns the Grassmannian distance into a plain
    Euclidean distance -> ordinary KNN, same cost as before.

    THE LENGTH SCALE (formerly mis-described here as "no hyper-parameter").
    Homogenizing [p0; 1] adds a LENGTH (p0) to a dimensionless 1, so a length
    scale is unavoidable -- it is dimensional analysis, not a design choice, and
    GraffMatch (Lusk et al., RA-L 2022) carries the same constant as `rho`.
    Here it is lambda = c * sigma with sigma = the per-cloud median foot radius
    and c HARDCODED TO 1.  MEASURED 2026-08-22 (neighbour consistency under GT
    correspondence, sweeping c over 0.03..30):
        found8 synthetic  peak c=1.0   |  7-Scenes real  peak c=2.0
        KITTI mono_best   peak c=1.0   |  c=1 is within 2% of peak on ALL three
        plateau c=0.5..3.0 within ~4%; only |log c| > 1 actually hurts.
    So the constant is real but EMPIRICALLY INERT here, and tying it to sigma
    makes it scale-free -> no per-domain recalibration.  WHY ours is flatter
    than GraffMatch's: rho matters for them because they compare DISTANT
    landmark pairs, where the principal angle saturates toward pi/2 past ~2 m.
    We only ever take NEAREST neighbours -- measured median theta2 is 1-3 deg
    with 0% above 85 deg -- i.e. deep in the linear regime, so their failure
    mode does not arise.

    Invariances (all unit-tested): SIGN — flipping [m;d] leaves p0 = d x m
    untouched and sends c1 -> -c1, so the SUBSPACE and hence P are unchanged;
    ROTATION — R acts as the orthogonal diag(R,1) on R^4, and Frobenius norms
    are orthogonally invariant; TRANSLATION — p0 is mean-centred first;
    SCALE — p0 is divided by its median radius.  raw: (B,6,N) -> idx (B,N,k).
    """
    m, d = raw[:, :3], raw[:, 3:]
    d = F.normalize(d, dim=1)
    p0 = torch.cross(d, m, dim=1)                           # (B,3,N)
    # TRANSLATION: foot points are NOT translation-equivariant --
    #   p0' = sRp0 + t - d'(d'.t)  (the shift leaks in direction-dependently),
    # so mean-centring does NOT remove a world translation.  Use the canonical
    # Sim(3)-EQUIVARIANT origin instead: c = argmin_x sum_i dist(x, line_i)^2,
    # i.e. A c = b with A = sum (I - d d^T), b = sum (I - d d^T) p0.  Under
    # (s,R,t) this maps c -> sRc + t exactly, and the perpendicular offset
    #   q = (I - d d^T)(p0 - c)   transforms as   q -> s R q,
    # a pure rotation+scale that the median-radius division then normalises.
    B, _, N = p0.shape
    dT = d.transpose(1, 2)                                  # (B,N,3)
    Proj = torch.eye(3, device=d.device, dtype=d.dtype).expand(B, N, 3, 3) \
        - dT.unsqueeze(-1) * dT.unsqueeze(-2)               # (B,N,3,3)
    A = Proj.sum(1) + 1e-6 * torch.eye(3, device=d.device, dtype=d.dtype)
    b = (Proj @ p0.transpose(1, 2).unsqueeze(-1)).sum(1)    # (B,3,1)
    c = torch.linalg.solve(A, b)                            # (B,3,1)
    dp = (p0 - c)                                           # (B,3,N)
    p0 = dp - d * (d * dp).sum(1, keepdim=True)             # perpendicular part
    sig = p0.norm(dim=1).median(dim=1).values.clamp_min(1e-9)
    p0 = p0 / sig.view(-1, 1, 1)                            # scale norm.
    one = torch.ones_like(p0[:, :1])
    c0 = F.normalize(torch.cat([p0, one], dim=1), dim=1)    # (B,4,N)
    c1 = torch.cat([d, torch.zeros_like(one)], dim=1)
    c1 = F.normalize(c1 - c0 * (c1 * c0).sum(1, keepdim=True), dim=1)
    s2, feats = 2.0 ** 0.5, []
    for a in range(4):                                      # vec(P), 10 dims
        for b in range(a, 4):
            v = c0[:, a] * c0[:, b] + c1[:, a] * c1[:, b]
            feats.append(v if a == b else s2 * v)
    return knn(torch.stack(feats, dim=1), min(k, raw.size(-1)))

def graff_stats(raw):
    """Sim(3)-equivariant origin c and scale sigma, computed PER CLOUD.

    SHARED (query-derived) stats were tried first and are WRONG — measured
    2026-08-24. sigma_ref/sigma_query IS essentially the scale factor s, so
    sharing one sigma throws the other cloud off by exactly the quantity we want
    to be invariant to: the reference landed at an effective c of median 0.31
    (p5 0.024), with 62% of pairs outside the calibrated plateau and 27% deep in
    the degenerate regime. The homogeneous coordinate of c0 collapsed from the
    balanced 0.707 to 0.252, i.e. position/direction balance destroyed, and the
    run was -2.8/-4.7/-5.8 pts behind the 3-branch baseline over epochs 0/1/2.

    PER-CLOUD puts BOTH clouds at the calibrated c=1 (see graff_knn_idx: c=1 is
    optimal on found8 / 7-Scenes / KITTI with a 0.5-3.0 plateau). It makes the
    node features fully Sim(3)-INVARIANT, which also drops the scale ratio --
    an accepted, deliberate trade (the solver recovers scale from the
    correspondences; the matcher's job is association).
    """
    d = F.normalize(raw[:, 3:], dim=1)
    p0 = torch.cross(d, raw[:, :3], dim=1)
    B, _, N = p0.shape
    dT = d.transpose(1, 2)
    Proj = torch.eye(3, device=d.device, dtype=d.dtype).expand(B, N, 3, 3) \
        - dT.unsqueeze(-1) * dT.unsqueeze(-2)
    A = Proj.sum(1) + 1e-6 * torch.eye(3, device=d.device, dtype=d.dtype)
    b = (Proj @ p0.transpose(1, 2).unsqueeze(-1)).sum(1)
    c = torch.linalg.solve(A, b)                                  # (B,3,1)
    dp = p0 - c
    perp = dp - d * (d * dp).sum(1, keepdim=True)
    sig = perp.norm(dim=1).median(dim=1).values.clamp_min(1e-9)   # (B,)
    return c, sig

def graff_YP(raw, c, sig):
    """Affine-Grassmannian coordinates of every line, under a GIVEN (c, sigma).

    Returns Y (B,N,4,2) orthonormal basis of the 2-plane, and vecP (B,10,N) the
    10 unique entries of P = Y Y^T with sqrt(2) on the off-diagonals, so that
    Euclidean distance on vecP IS the chordal Grassmann distance. Same
    construction as GraffMatch Eq. 3-4 (Lusk et al., RA-L 2022).
    """
    d = F.normalize(raw[:, 3:], dim=1)
    p0 = torch.cross(d, raw[:, :3], dim=1)
    dp = p0 - c
    perp = (dp - d * (d * dp).sum(1, keepdim=True)) / sig.view(-1, 1, 1)
    one = torch.ones_like(perp[:, :1])
    c0 = F.normalize(torch.cat([perp, one], dim=1), dim=1)        # (B,4,N)
    c1 = torch.cat([d, torch.zeros_like(one)], dim=1)
    c1 = F.normalize(c1 - c0 * (c1 * c0).sum(1, keepdim=True), dim=1)
    s2, feats = 2.0 ** 0.5, []
    for a in range(4):
        for b in range(a, 4):
            v = c0[:, a] * c0[:, b] + c1[:, a] * c1[:, b]
            feats.append(v if a == b else s2 * v)
    Y = torch.stack([c0, c1], dim=-1).permute(0, 2, 1, 3)         # (B,N,4,2)
    return Y, torch.stack(feats, dim=1)                            # (B,10,N)

def attention(query, key, value, bias=None):
    """Scaled dot-product attention — uses flash attention when available.
    (B, d, H, N) → (B, d, H, N)"""
    q = query.permute(0, 2, 3, 1)   # (B, H, N, d)
    k = key.permute(0, 2, 3, 1)
    v = value.permute(0, 2, 3, 1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias).permute(0, 3, 1, 2), None

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim       = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj  = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value, bias=None):
        B = query.size(0)
        query, key, value = [l(x).view(B, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value, bias=bias)
        return self.merge(x.contiguous().view(B, self.dim * self.num_heads, -1))

class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp  = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source, bias=None):
        return self.mlp(torch.cat([x, self.attn(x, source, source, bias=bias)], dim=1))

class SpatialAttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list, num_heads: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([AttentionalPropagation(feature_dim, num_heads)
                                     for _ in range(len(layer_names))])
        self.names = layer_names
        self.mlp   = MLP([feature_dim * 3, feature_dim * 2, feature_dim * 2, feature_dim])

    def forward(self, desc0, desc1, bias0=None, bias1=None):
        for layer, name in zip(self.layers, self.names):
            src0, src1 = (desc1, desc0) if name == 'cross' else (desc0, desc1)
            # geometric bias ONLY on self-attention: within one cloud the
            # geometry is known, but the CROSS relation is the unknown pose.
            b0, b1 = (None, None) if name == 'cross' else (bias0, bias1)
            desc0 = desc0 + layer(desc0, src0, bias=b0)
            desc1 = desc1 + layer(desc1, src1, bias=b1)

        # Per-point matchability prior: each side gets global (mean+max) context from the other
        N0, N1 = desc0.size(-1), desc1.size(-1)
        g0 = torch.cat([desc0.mean(-1, keepdim=True), desc0.max(-1, keepdim=True)[0]], dim=1)
        g1 = torch.cat([desc1.mean(-1, keepdim=True), desc1.max(-1, keepdim=True)[0]], dim=1)

        # expand is zero-copy; cat materialises once
        desc0_reg = self.mlp(torch.cat([desc0, g1.expand(-1, -1, N0)], dim=1))
        desc1_reg = self.mlp(torch.cat([desc1, g0.expand(-1, -1, N1)], dim=1))

        return desc0, desc1, desc0_reg, desc1_reg


class prob_mat_sinkhorn(nn.Module):
    """Entropic-OT matching layer (balanced transport, learned marginals r/c).

    The SuperGlue dustbin variant was tested and REJECTED (2026-08-30): it
    produced genuinely calibrated scores (P.max 0.92 vs balanced OT's 0.01
    ceiling) but calibration never converted into registration accuracy, and
    all three exploitation routes closed -- threshold selection +0.33 +/- 3.27
    over 12 paired seeds, dustbin-defined pool 4.78% purity vs top-300's 5.10%,
    prop-to-P sampling gain traced to pool size not score. Sinkhorn optimises a
    PER-PAIR objective; the residual errors are INTER-PAIR (aliasing).
    Iterations: 30 is already past convergence (10 -> 2181 true inliers,
    30 -> 2159, 100 -> 2155); mu=0.1 sits on a 0.1-0.2 plateau.
    """

    def __init__(self, config, mu=0.1, tolerance=1e-9, iterations=30):
        super().__init__()
        self.mu = mu
        self.iterations = iterations
        self.eps = 1e-12

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


class conv_in_graff(nn.Module):
    """THE input encoder: ONE affine-Grassmannian branch, ONE pooling, ONE MLP.

    Per line i, over its k=10 nearest neighbours j in Graff(1,3):
        edge_ij = cat(vec(P)_j - vec(P)_i, vec(P)_i)        (20-D)
        node_i  = MLP( mean_j Conv(edge_ij) )               (32 -> 128)

    vec(P) is the 10 unique entries of the projector P = Y Y^T onto the line's
    affine 2-plane, sqrt(2) on the off-diagonals so Euclidean distance IS the
    chordal Grassmann distance (GraffMatch Eq. 3-4, Lusk et al. RA-L 2022).

    WHY THERE IS NO SIGN HANDLING HERE.  Flipping [m;d] -> [-m;-d] is the SAME
    LINE: it leaves p0 = d x m untouched and sends c1 -> -c1, so the SUBSPACE
    and hence P are unchanged.  Measured exactly 0.000e+00 under random
    per-line flips.  The k-NN graph and graff_stats also work from foot points,
    which are themselves sign-invariant.  So the hemisphere canon is a literal
    no-op on this path and the sign-even embedding (sign_inv) is unnecessary --
    both removed 2026-09-04.  Historically the ambiguity is real (47.3% of true
    cross-modal matches carry opposite Plucker sign) and the max|comp| canon cut
    it to a 7% residual seam; the Grassmannian removes it by construction.

    WHY THERE IS NO SIGMA NORMALISATION.  sigma (median perpendicular radius)
    is a per-cloud statistic that ASSUMES both clouds cover comparable content;
    measured 21.5% error at full overlap and 35.8% at 1/8 crop.  Dropping it
    (rawgeo) trades that bias for a dependence on training-range coverage.
    Both were measured; rawgeo ships.

    TESTED AND REJECTED (do not re-add): explicit principal-angle edge channels
    (mathematically redundant, <f_j,f_i> = cos^2 t1 + cos^2 t2 exactly; 20-D
    matched 22-D at 5 seeds), max/min pooling (mean is 2.8x more k-robust on
    real maps), fuse_width 16 and 64 (32 is an interior optimum), separate
    node/edge branches (pooled separately, destroying the pairing), geometric
    attention bias, graph wavelets, 6 layers, size-adaptive k.
    """

    KNN_K = 10          # insensitive across [6,24]; see CLAUDE.md
    EDGE_CH = 20        # cat(vec(P)_j - vec(P)_i, vec(P)_i)
    WIDTH = 32          # interior optimum: 16 and 64 both measured worse

    def __init__(self, out_channel: int):
        super().__init__()
        # names kept as conv_fuse/mlp_fuse so existing checkpoints load
        self.conv_fuse = nn.Conv2d(self.EDGE_CH, self.WIDTH, 1)
        self.mlp_fuse = MLP([self.WIDTH, out_channel, out_channel, out_channel])

    def forward(self, x, raw=None):
        assert raw is not None, 'graff encoder needs the raw [m;d] input'
        idx = graff_knn_idx(raw, self.KNN_K)
        c_, _ = graff_stats(raw)                       # sigma unused (rawgeo)
        _, vecP = graff_YP(raw, c_, torch.ones_like(_))
        edge = get_graph_feature(vecP, idx=idx)          # (B,20,N,k)
        return self.mlp_fuse(self.conv_fuse(edge).mean(dim=-1))


class FeatureExtractorGraph(nn.Module):
    """Grassmannian encoder -> 12-layer self/cross attention GNN -> descriptors."""

    def __init__(self, config, in_channel: int = 6):
        super().__init__()
        nc = config['descriptor_dim'] if 'descriptor_dim' in config \
            else config['net_nchannel']
        self.conv_in = conv_in_graff(nc)
        heads = int(config['attn_heads']) if 'attn_heads' in config else 4
        self.gnn = SpatialAttentionalGNN(nc, config['GNN_layers'], num_heads=heads)
        self.final_proj = nn.Conv1d(nc, nc, kernel_size=1, bias=True)
        self.regress = nn.Conv1d(nc, 1, kernel_size=1, bias=True)

    def forward(self, x, y):
        desc0, desc1, x_prob, y_prob = self.gnn(
            self.conv_in(x, raw=x), self.conv_in(y, raw=y))
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        return (mdesc0, mdesc1,
                self.regress(x_prob).softmax(dim=-1),
                self.regress(y_prob).softmax(dim=-1))


class PluckerNetKnn(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channel = getattr(config, 'in_channel', 6)
        self.FeatureExtractor = FeatureExtractorGraph(config, self.in_channel)
        self.sinkhorn = prob_mat_sinkhorn(config, config.net_lambda, 1e-9,
                                          config.net_maxiter)

    def forward(self, plucker1, plucker2):
        f1, f2, p1, p2 = self.FeatureExtractor(plucker1.transpose(-2, -1),
                                               plucker2.transpose(-2, -1))
        f1 = F.normalize(f1.transpose(-2, -1), p=2, dim=-1)
        f2 = F.normalize(f2.transpose(-2, -1), p=2, dim=-1)
        M = pairwiseL2Dist(f1, f2)
        r, c = p1.squeeze(1), p2.squeeze(1)
        return self.sinkhorn(M, r, c), r, c
