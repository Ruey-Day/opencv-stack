"""
register.py
===========
Register two Structure-PLP-SLAM line maps using ScalePluckerNet + Grassmannian
RANSAC to recover a SIM(3) transformation (scale, rotation, translation).

Typical use: mono map (arbitrary scale) → RGB-D metric map.

Usage:
    python register.py \\
        --db_src  path/to/mono_map.db \\
        --db_tgt  path/to/metric_map.db \\
        --checkpoint output/slam_map/<run>/best_val_checkpoint.pth

    # More RANSAC runs for a harder scene:
    python register.py --db_src mono.db --db_tgt metric.db \\
        --n_runs 30 --n_iter 2000 --topk 200

Output (stdout):
    Estimated scale, rotation, translation + inlier count.
    If --gt_scale is given, reports % error vs ground truth.
"""

import os, sys, argparse
import numpy as np
import torch
import msgpack
from easydict import EasyDict as edict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from lib.ransac_grassmannian import ransac_sim3
from lib.utils import load_model


def load_pool_from_db(db_path: str) -> np.ndarray:
    """Load line landmarks from a Structure-PLP-SLAM map as Plücker [m, d] (K, 6)."""
    with open(db_path, "rb") as f:
        data = msgpack.unpack(f, raw=False)
    pool = []
    for lm in data.get("landmarks_line", {}).values():
        pw = lm.get("pos_w") or lm.get("pos")
        if pw is None or len(pw) < 6:
            continue
        p1 = np.array(pw[:3], np.float32)
        p2 = np.array(pw[3:6], np.float32)
        diff = p2 - p1
        ln = float(np.linalg.norm(diff))
        if ln < 0.01:
            continue
        d = diff / ln
        m = np.cross((p1 + p2) * 0.5, d)
        pool.append(np.concatenate([m, d]).astype(np.float32))
    return np.array(pool, np.float32) if pool else np.zeros((0, 6), np.float32)


def load_network(ckpt_path: str, device: torch.device):
    configs = edict(dict(
        model='KNNContextNormNet', net_depth=12, net_nchannel=128,
        GNN_layers=['self', 'cross'] * 6, net_gcnorm=True, net_batchnorm=True,
        net_topK=2000, net_lambda=0.1, net_maxiter=30, net_knn=10,
        in_channel=6, normalize_n_lines=200, normalize_n_inliers=60,
        gpu_inds=0, use_gpu=True,
    ))
    Model = load_model('PluckerNetKnn')
    model = Model(configs).to(device)
    state = torch.load(ckpt_path, weights_only=False, map_location=device)
    sd = state.get('state_dict', state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:    print(f"  Missing keys:    {missing[:3]}")
    if unexpected: print(f"  Unexpected keys: {unexpected[:3]}")
    model.eval()
    return model


def register(db_src: str, db_tgt: str, ckpt_path: str,
             n_runs: int = 10, n_iter: int = 1000,
             threshold: float = 0.3, topk: int = 100,
             max_lines: int = 200, gt_scale: float | None = None,
             device: torch.device | None = None):
    """
    Register src map onto tgt map.  Returns (scale, R, t, inlier_count) or
    (None, None, None, 0) if RANSAC fails.

    src  — the map to be scaled (e.g. monocular)
    tgt  — the metric reference map (e.g. RGB-D)
    """
    if device is None:
        device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')

    p_src = load_pool_from_db(db_src)
    p_tgt = load_pool_from_db(db_tgt)
    print(f"src map : {len(p_src):4d} lines  ({db_src})")
    print(f"tgt map : {len(p_tgt):4d} lines  ({db_tgt})")

    model = load_network(ckpt_path, device)

    # Subsample to training distribution
    if max_lines and p_src.shape[0] > max_lines:
        idx = np.random.choice(p_src.shape[0], max_lines, replace=False)
        p_src = p_src[idx]
    if max_lines and p_tgt.shape[0] > max_lines:
        idx = np.random.choice(p_tgt.shape[0], max_lines, replace=False)
        p_tgt = p_tgt[idx]
    print(f"Network input: {p_src.shape[0]} src × {p_tgt.shape[0]} tgt lines")

    t1 = torch.from_numpy(p_src[None]).to(device)  # (1, N, 6)
    t2 = torch.from_numpy(p_tgt[None]).to(device)  # (1, M, 6)

    with torch.no_grad():
        prob_matrix, _, _ = model(t1, t2)  # (1, N, M)

    k = min(topk, p_src.shape[0] * p_tgt.shape[0])
    _, topk_flat = torch.topk(prob_matrix.flatten(start_dim=-2), k=k, dim=-1)
    idx1 = (topk_flat // prob_matrix.size(-1))[0].cpu().numpy()
    idx2 = (topk_flat  %  prob_matrix.size(-1))[0].cpu().numpy()
    pl1, pl2 = p_src[idx1].T, p_tgt[idx2].T   # (6, k) each

    print(f"Top-{k} correspondences → RANSAC ({n_runs} runs × {n_iter} iters) …")

    best_ic = -1
    best_s, best_R, best_t = None, None, None
    for run_i in range(n_runs):
        R, t, s, _, ic = ransac_sim3(pl1, pl2,
                                      n_iter=n_iter,
                                      inlier_threshold=threshold,
                                      seed=run_i)
        if ic > best_ic and s is not None and s > 0:
            best_ic, best_s, best_R, best_t = ic, s, R, t

    print(f"\n{'='*55}")
    print(f"{'':12s}   {'Estimated':>10s}   {'GT':>8s}   {'Error':>8s}")
    print(f"{'-'*55}")
    if best_s is not None:
        if gt_scale is not None:
            err = abs(best_s - gt_scale) / gt_scale * 100
            print(f"{'scale':12s}   {best_s:>10.4f}   {gt_scale:>8.4f}   {err:>6.1f}%")
        else:
            print(f"{'scale':12s}   {best_s:>10.4f}")
        print(f"{'inliers':12s}   {best_ic:>10d}")
        print(f"{'='*55}")
        print(f"\nR =\n{np.round(best_R, 4)}\nt = {np.round(best_t.flatten(), 4)}")
    else:
        print("RANSAC failed — no valid hypothesis found")
    print(f"{'='*55}")

    return best_s, best_R, best_t, best_ic


def main():
    DEFAULT_CKPT = os.path.join(
        SCRIPT_DIR, "output", "synthetic",
        "synthetic-v3", "best_val_checkpoint.pth",
    )

    ap = argparse.ArgumentParser(description="Register two SLAM line maps via ScalePluckerNet.")
    ap.add_argument('--db_src',      required=True,        help='Source (mono) map .db')
    ap.add_argument('--db_tgt',      required=True,        help='Target (metric) map .db')
    ap.add_argument('--checkpoint',  default=DEFAULT_CKPT, help='Network checkpoint .pth')
    ap.add_argument('--n_iter',      type=int,   default=2000,  help='RANSAC iterations per run')
    ap.add_argument('--n_runs',      type=int,   default=10,    help='Independent RANSAC runs')
    ap.add_argument('--threshold',   type=float, default=0.3,   help='L2 inlier threshold (m)')
    ap.add_argument('--topk',        type=int,   default=200,   help='Top-K correspondences')
    ap.add_argument('--max_lines',   type=int,   default=200,   help='Max lines per side')
    ap.add_argument('--gt_scale',    type=float, default=None,  help='GT scale for error report')
    args = ap.parse_args()

    register(
        db_src=args.db_src, db_tgt=args.db_tgt, ckpt_path=args.checkpoint,
        n_runs=args.n_runs, n_iter=args.n_iter, threshold=args.threshold,
        topk=args.topk, max_lines=args.max_lines, gt_scale=args.gt_scale,
    )


if __name__ == "__main__":
    main()
