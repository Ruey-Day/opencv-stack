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
    """Entropic-OT matching layer.

    dustbin=False (pre-v24): BALANCED transport with the learned soft
    marginals r/c — every line's mass MUST be transported, so the 80-97% of
    real cross-modal lines with no true counterpart dump probability onto
    wrong pairs (measured: flat ranking, P@10 ≈ P@200).

    dustbin=True (v24): SuperGlue-style log-space OT with a learnable bin
    row/column (bin_score). Real rows/cols carry unit mass, the bins carry
    the opposite side's count, so unmatched lines can park their mass in the
    bin instead of polluting real pairs. Returns the (B, Nr, Nq) slice
    without the bins; entries of confidently matched pairs are ~1, so the
    BCE loss is unchanged. r/c are ignored by the transport in this mode
    (they remain supervised matchability predictors used by the loss and
    for inference-time gating)."""

    def __init__(self, config, mu=0.1, tolerance=1e-9, iterations=30):
        super().__init__()
        self.mu         = mu
        self.iterations = iterations
        self.eps        = 1e-12
        self.dustbin    = bool(config['dustbin']) if 'dustbin' in config else False
        if self.dustbin:
            self.bin_score = nn.Parameter(torch.tensor(1.0))

    def forward(self, M, r=None, c=None):
        if self.dustbin:
            return self._forward_dustbin(M)
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

    def _forward_dustbin(self, M):
        B, Nr, Nq = M.shape
        S = -M / self.mu                                   # similarity logits
        alpha = self.bin_score.expand(B, 1, 1)
        S = torch.cat([S, alpha.expand(B, Nr, 1)], dim=2)
        S = torch.cat([S, alpha.expand(B, 1, Nq + 1)], dim=1)  # (B,Nr+1,Nq+1)
        one = S.new_tensor(1.0)
        ms, ns = one * Nr, one * Nq
        norm = -(ms + ns).log()
        log_mu = torch.cat([norm.expand(Nr), ns.log()[None] + norm]).expand(B, -1)
        log_nu = torch.cat([norm.expand(Nq), ms.log()[None] + norm]).expand(B, -1)
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for _ in range(self.iterations):
            u = log_mu - torch.logsumexp(S + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(S + u.unsqueeze(2), dim=1)
        Z = S + u.unsqueeze(2) + v.unsqueeze(1) - norm     # total mass Nr+Ns
        return Z.exp()[:, :Nr, :Nq]


GEO_KNN_W = 0.5   # direction weight in sigma-units: sin(90deg) contributes ~0.7


def geo_knn_idx(raw, k):
    """Sign-invariant GEOMETRIC KNN graph (v29, geo_knn flag).

    The base graph was KNN on the first 3 feature channels — raw moments are
    SIGN-dependent (m and -m are L2-far: spatially adjacent lines with
    opposite stored signs never become neighbors) and origin-lever-arm
    quantities; under sign_inv the learned even embedding hides the sign
    issue but the space is arbitrary early in training. This graph instead
    uses explicit geometry, all sign-invariant:
      position:  foot points p0 = d x m  (p0(-L) = p0(L)), normalized by the
                 per-cloud median foot radius
      direction: vec(d d^T) with sqrt(2) off-diagonals, whose pairwise L2 is
                 exactly sqrt(2) sin(theta_ij) — the |cos| metric in a form
                 plain KNN consumes; weighted by GEO_KNN_W
    Restores upstream's direction-aware neighborhoods without its sign
    fragility. raw: (B, 6, N) RAW [m; d] -> idx (B, N, k)."""
    m, d = raw[:, :3], raw[:, 3:]
    d = F.normalize(d, dim=1)
    p0 = torch.cross(d, m, dim=1)                     # (B, 3, N)
    # mean centre (rotation-EQUIVARIANT — a coordinate-wise median is not,
    # which perturbed the normalizer under rotation and broke graph
    # invariance via distance ties)
    ctr = p0.mean(dim=2, keepdim=True)
    sig = (p0 - ctr).norm(dim=1).median(dim=1).values.clamp_min(1e-9)
    pn = p0 / sig.view(-1, 1, 1)
    dx, dy, dz = d[:, 0:1], d[:, 1:2], d[:, 2:3]
    s2 = 2.0 ** 0.5
    outer = torch.cat([dx * dx, dy * dy, dz * dz,
                       s2 * dx * dy, s2 * dx * dz, s2 * dy * dz], dim=1)
    feat = torch.cat([pn, GEO_KNN_W * outer], dim=1)
    return knn(feat, min(k, raw.size(-1)))


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
    Euclidean distance -> ordinary KNN, same cost as before, and with NO
    direction/position weighting hyper-parameter: the manifold fixes the
    relative weighting intrinsically (unlike geo_knn's hand-set GEO_KNN_W).

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


def geo_edge_features(raw, idx):
    """Sim(3)-INVARIANT relative-geometry edge features (v24, geo_edge flag).

    raw: (B, 6, N) RAW [m; d] Plucker lines (pre sign-embedding); idx: (B, N, k)
    neighbour indices. For each edge (i, j) returns 3 channels:
      0: |d_i . d_j|                     rotation/scale/translation/sign inv.
      1: sin of the inter-line angle     (conditioning companion of 0)
      2: log line-to-line perpendicular distance, normalized by the per-line
         median over its k neighbours    scale-invariant by construction
    A 2-line configuration of infinite lines has exactly these two Sim(3)
    invariants (angle + normalized distance) — feeding them explicitly removes
    the coordinate-statistics shortcut that made fine-tuned models fit
    sequences instead of the modality (v23 finding). Near-parallel edges fall
    back to point-to-line distance. Output: (B, 3, N, k)."""
    B, _, N = raw.shape
    k = idx.size(-1)
    m, d = raw[:, :3].transpose(1, 2), raw[:, 3:].transpose(1, 2)   # (B,N,3)
    d = F.normalize(d, dim=-1)
    p0 = torch.cross(d, m, dim=-1)                                  # foot points
    bi = torch.arange(B, device=raw.device).view(-1, 1, 1)
    dj = d[bi, idx]                                                 # (B,N,k,3)
    pj = p0[bi, idx]
    di = d.unsqueeze(2)
    dp = pj - p0.unsqueeze(2)
    csign = (di * dj).sum(-1)                                       # signed
    cos = csign.abs().clamp(max=1.0)
    sin = (1.0 - cos ** 2).clamp_min(0.0).sqrt()
    # exact min distance between the two infinite lines via closest points
    # (translation-exact for ANY angle; the naive point-to-line fallback is
    # only slide-invariant for EXACTLY parallel lines, so the parallel branch
    # is reserved for sin < 1e-3)
    A_ = (dp * di).sum(-1)
    B_ = (dp * dj).sum(-1)
    denom = (1.0 - csign ** 2).clamp_min(1e-9)
    tb = (csign * A_ - B_) / denom
    ta = A_ + csign * tb
    v = dp + tb.unsqueeze(-1) * dj - ta.unsqueeze(-1) * di
    d_skew = v.norm(dim=-1)
    d_par = torch.cross(dp, di.expand_as(dp), dim=-1).norm(dim=-1)
    dist = torch.where(sin > 1e-3, d_skew, d_par)
    med = dist.median(dim=-1, keepdim=True)[0].clamp_min(1e-9)
    # ratio floor 0.01: below that the pair is "essentially intersecting" —
    # no information, and the log only amplifies float cancellation noise
    logd = (dist / med).clamp_min(0.01).log()
    return torch.stack([cos, sin, logd], dim=1)                     # (B,3,N,k)


class conv_in_seq_direction_moment_knn(nn.Module):
    def __init__(self, out_channel: int, in_channel: int = 6,
                 dual_knn: bool = False, geo_edge: bool = False,
                 geo_knn: bool = False, graff_knn: bool = False):
        super().__init__()
        half = out_channel // 2

        self.geo_edge = geo_edge
        self.geo_knn = geo_knn
        self.graff_knn = graff_knn
        if geo_edge:
            # third branch mirroring the other two; merged MLP widens
            self.conv_geo = nn.Conv2d(3, half // 8, 1)
            self.mlp_geo  = MLP([half // 8, half // 4, half // 2, half])

        self.conv_direction = nn.Conv2d(6, half // 8, 1)
        self.conv_moment    = nn.Conv2d((in_channel - 3) * 2, half // 8, 1)
        self.mlp_direction  = MLP([half // 8, half // 4, half // 2, half])
        self.mlp_moment     = MLP([half // 8, half // 4, half // 2, half])
        merged_in = out_channel + (half if geo_edge else 0)
        self.mlp_merged     = MLP([merged_in, out_channel, out_channel])
        # NAMING WARNING (verified 2026-07-24): these variable names are
        # inherited from upstream PlueckerNet, which uses [d, m] channel order.
        # OUR data is [m, d] (see lib/dataloader.py + segments_to_plucker), so
        # channels 0:3 are MOMENTS and 3:6 are DIRECTIONS. Therefore the tensor
        # called `dir_feat` below actually holds MOMENTS and `mom_feat` holds
        # DIRECTIONS. (Checked numerically: channels 3:6 are unit-norm.)
        #
        # What this means for dual_knn: the BASE graph `idx` is built on
        # channels 0:3 = moments, i.e. POSITION space — both branches already
        # aggregate over spatially-near lines. Setting dual_knn=True gives the
        # second branch its own graph in DIRECTION space, so direction features
        # aggregate over similarly-ORIENTED lines instead of merely nearby ones
        # (previously they were smeared across spatially-near but
        # differently-oriented lines). Adds NO parameters (same convs, different
        # neighbour indices), so an existing checkpoint loads unchanged.
        self.dual_knn = dual_knn

    def forward(self, x, raw=None):
        dir_feat = x[:, :3, :]
        if self.graff_knn:
            assert raw is not None, 'graff_knn needs the raw [m;d] input'
            idx = graff_knn_idx(raw, 10)
        elif self.geo_knn:
            assert raw is not None, 'geo_knn needs the raw [m;d] input'
            idx = geo_knn_idx(raw, 10)
        else:
            idx = knn(dir_feat, k=min(10, dir_feat.size(-1)))
        x_dir = self.mlp_direction(
            self.conv_direction(get_graph_feature(dir_feat, idx=idx)).mean(dim=-1))
        if self.dual_knn:
            mom_feat = x[:, 3:, :]
            idx_m = knn(mom_feat, k=min(10, mom_feat.size(-1)))
            x_mom = self.mlp_moment(
                self.conv_moment(get_graph_feature(mom_feat, idx=idx_m)).mean(dim=-1))
        else:
            x_mom = self.mlp_moment(
                self.conv_moment(get_graph_feature(x[:, 3:, :], idx=idx)).mean(dim=-1))
        parts = [x_dir, x_mom]
        if self.geo_edge:
            assert raw is not None, 'geo_edge needs the raw [m;d] input'
            geo = geo_edge_features(raw, idx)
            parts.append(self.mlp_geo(self.conv_geo(geo).mean(dim=-1)))
        return self.mlp_merged(torch.cat(parts, dim=1))


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
    def __init__(self, feature_dim: int, layer_names: list, num_heads: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([AttentionalPropagation(feature_dim, num_heads)
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


class SymmetricInputEmbed(nn.Module):
    """Per-line sign-EVEN input embedding: e_i = phi(x_i) + phi(-x_i), with phi a
    per-line (1x1) nonlinear MLP. Because it is computed per line and even in that
    line's Plucker sign, e_i is unchanged when line i is flipped [m;d]->[-m;-d].
    The downstream KNN graph + edge features are then built on these even node
    features, so the WHOLE matcher is per-line sign-invariant BY CONSTRUCTION --
    no random-flip augmentation and no hemisphere canonicalisation needed. This is
    the matcher analogue of the solver's p0 = m x d invariance."""
    def __init__(self, ch=6):
        super().__init__()
        self.phi = nn.Sequential(nn.Conv1d(ch, 32, 1), nn.GELU(), nn.Conv1d(32, ch, 1))
    def forward(self, x):                       # x: (B, ch, N)
        return self.phi(x) + self.phi(-x)


class FeatureExtractorGraph(nn.Module):
    def __init__(self, config, in_channel):
        super().__init__()
        nc = config['net_nchannel']
        dual_knn = bool(config['dual_knn']) if 'dual_knn' in config else False
        self.geo_edge = bool(config['geo_edge']) if 'geo_edge' in config else False
        self.geo_knn = bool(config['geo_knn']) if 'geo_knn' in config else False
        self.graff_knn = bool(config['graff_knn']) if 'graff_knn' in config else False
        self.sign_inv = bool(config['sign_inv']) if 'sign_inv' in config else False
        self.sym = SymmetricInputEmbed(in_channel) if self.sign_inv else None
        self.conv_in    = conv_in_seq_direction_moment_knn(
            nc, in_channel=in_channel, dual_knn=dual_knn,
            geo_edge=self.geo_edge, geo_knn=self.geo_knn,
            graff_knn=self.graff_knn)
        self.gnn        = SpatialAttentionalGNN(
            nc, config['GNN_layers'],
            num_heads=int(config['attn_heads']) if 'attn_heads' in config else 4)
        self.final_proj = nn.Conv1d(nc, nc, kernel_size=1, bias=True)
        self.regress    = nn.Conv1d(nc, 1,  kernel_size=1, bias=True)

    def forward(self, x, y):
        need_raw = self.geo_edge or self.geo_knn or self.graff_knn
        raw_x, raw_y = (x, y) if need_raw else (None, None)
        if self.sym is not None:                # per-line sign-even embedding first
            x, y = self.sym(x), self.sym(y)
        desc0, desc1, x_prob, y_prob = self.gnn(self.conv_in(x, raw=raw_x),
                                                self.conv_in(y, raw=raw_y))
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
