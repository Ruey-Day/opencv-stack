"""
python eval.py --checkpoint output/joint/.../best_val_checkpoint.pth \\
    --dataset replica_gs,7scenes_gs,se3real_sim3

# Single dataset, single threshold:
python eval.py --checkpoint output/slam_map/.../best_val_checkpoint.pth \\
    --dataset slam_map --threshold 0.5

# Sweep RANSAC thresholds (faster: model runs once, RANSAC re-run per threshold):
python eval.py --checkpoint output/slam_map/.../best_val_checkpoint.pth \\
    --dataset slam_map --sweep

# Grassmannian RANSAC backend:
python eval.py --checkpoint ... --dataset slam_map --ransac grassmannian
"""

import os
import sys
import json
import warnings
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'PlueckerNet'))
sys.path.insert(0, ROOT_DIR)

from sim3.dataloader import Sim3PluckerData

class _DustbinWrapper(nn.Module):
    """Strips dustbin row/col so the caller sees a standard (B, N, M) matrix."""
    def __init__(self, model):
        super().__init__()
        self.inner = model

    def forward(self, p1, p2):
        P_aug, r, c = self.inner(p1, p2)
        return P_aug[:, :-1, :-1], r, c

_NEUTRAL_LAB = torch.tensor([50.0, 0.0, 0.0])

class _Pad6Dto9DWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner = model

    def forward(self, p1, p2):
        if p1.shape[-1] == 6:
            pad = _NEUTRAL_LAB.to(p1.device).view(1, 1, 3)
            p1 = torch.cat([p1, pad.expand(p1.shape[0], p1.shape[1], 3)], dim=-1)
            p2 = torch.cat([p2, pad.expand(p2.shape[0], p2.shape[1], 3)], dim=-1)
        return self.inner(p1, p2)

def _build_config(checkpoint_path, data_dir, dataset):
    """Load config embedded in checkpoint; fall back to safe defaults."""
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
    cfg  = ckpt.get('config')
    if cfg is None:
        cfg = edict(dict(
            model='KNNContextNormNet', net_depth=12, net_nchannel=128,
            GNN_layers=['self', 'cross'] * 6, net_gcnorm=True, net_batchnorm=True,
            net_topK=2000, net_lambda=0.1, net_maxiter=30, net_knn=10,
            in_channel=6, normalize_n_lines=200, normalize_n_inliers=60,
            gpu_inds=0, use_gpu=True,
        ))
        print('  Warning: no config in checkpoint — using defaults (200 lines, 6D)')
    if not isinstance(cfg, edict):
        cfg = edict(cfg)
    cfg.dataset  = dataset
    cfg.data_dir = data_dir
    cfg.resume   = None
    cfg.weights  = None
    cfg.model_nb = 'eval'
    return cfg, ckpt


def _load_model(cfg, ckpt, device):
    from lib.utils import load_model as _lm
    sd = ckpt.get('state_dict', ckpt.get('model', ckpt))
    is_dustbin = any('bin_dist' in k or 'bin_score' in k for k in sd)

    if is_dustbin:
        from sim3.model_dustbin import PluckerNetKnnDustbin
        model = PluckerNetKnnDustbin(cfg)
        model.load_state_dict(sd, strict=True)
        print('  (dustbin checkpoint — wrapping for eval)')
        return _DustbinWrapper(model)
    else:
        Model = _lm('PluckerNetKnn')
        model = Model(cfg)
        model.load_state_dict(sd, strict=False)
        return model

def _rot_err_deg(R_est, R_gt):
    tr = np.clip((np.trace(R_est @ R_gt.T) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(tr)))

def _scale_err_log(s_est, s_gt):
    return float(abs(np.log(max(float(s_est), 1e-6)) - np.log(max(float(s_gt), 1e-6))))

def _trans_err(t_est, t_gt):
    return float(np.linalg.norm(np.asarray(t_est).flatten() - np.asarray(t_gt).flatten()))

def _overlap_bucket(n_inliers):
    if n_inliers == 0:
        return 'no_overlap (0%)'
    if n_inliers <= 200:
        return 'sparse (~30%)'
    return 'dense (~70%)'

def _run_inference(model, val_loader, device, max_pairs):
    """Run model forward on all val pairs; cache top-K correspondences."""
    model.eval()
    cache = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_pairs and i >= max_pairs:
                break
            matches, p1, p2, R_gt, t_gt, s_gt = batch

            prob, _, _ = model(p1.to(device), p2.to(device))

            k = min(100, prob.shape[1] * prob.shape[2])
            _, flat = torch.topk(prob.flatten(start_dim=-2), k=k, dim=-1)
            i1 = (flat // prob.shape[-1]).squeeze(0).cpu().numpy()
            i2 = (flat  % prob.shape[-1]).squeeze(0).cpu().numpy()

            n_inliers = int(matches.sum().item())
            match_mat = matches[0].numpy()
            ir = float(match_mat[i1, i2].sum() / k * 100)

            cache.append(dict(
                p1k=p1[0, i1, :6].numpy().T,   # (6, k)
                p2k=p2[0, i2, :6].numpy().T,
                n_inliers=n_inliers,
                ir=ir,
                R_gt=R_gt[0].numpy(),
                t_gt=t_gt[0].numpy(),
                s_gt=float(s_gt[0]),
            ))
            torch.cuda.empty_cache()
    return cache

def _eval_cached(cache, run_ransac_fn):
    """Run RANSAC over pre-cached correspondences; return per-scene metrics."""
    all_rot, all_trans, all_scale, all_ir = [], [], [], []
    buckets = {b: {'n': 0, 'rot': [], 'trans': [], 'scale': [], 'inlier_ratio': []}
               for b in ['no_overlap (0%)', 'sparse (~30%)', 'dense (~70%)']}

    for c in cache:
        bucket = _overlap_bucket(c['n_inliers'])
        err_r, err_t, err_s = 180.0, float('inf'), float('inf')

        if c['n_inliers'] > 0 and c['p1k'].shape[1] > 3:
            best_s, best_R, best_t, _, _ = run_ransac_fn(c['p1k'], c['p2k'])
            if best_R is not None:
                err_r = _rot_err_deg(best_R, c['R_gt'])
                err_t = _trans_err(best_t, c['t_gt'])
                if best_s is not None and best_s > 0 and c['s_gt'] > 0:
                    err_s = _scale_err_log(best_s, c['s_gt'])

        all_rot.append(err_r)
        all_trans.append(err_t)
        all_scale.append(err_s)
        all_ir.append(c['ir'])

        bd = buckets[bucket]
        bd['n'] += 1
        bd['rot'].append(err_r)
        bd['trans'].append(err_t)
        bd['scale'].append(err_s)
        bd['inlier_ratio'].append(c['ir'])

    return all_rot, all_trans, all_scale, all_ir, buckets

def _summarise(rot, trans, scale, ir):
    rots  = np.array(rot)
    trans_fin = np.array([x for x in trans if np.isfinite(x)])
    scale_fin = np.array([x for x in scale if np.isfinite(x)])

    ths = np.arange(7) * 5
    hist, _ = np.histogram(rots, ths)
    q_acc = np.cumsum(hist.astype(float) / len(rots))

    return dict(
        recall_rot       = float(np.mean(q_acc[:4])),
        pct_lt5          = float(q_acc[0]) if len(q_acc) > 0 else 0.0,
        pct_lt10         = float(q_acc[1]) if len(q_acc) > 1 else 0.0,
        pct_lt20         = float(q_acc[3]) if len(q_acc) > 3 else 0.0,
        med_rot          = float(np.median(rots)),
        med_trans        = float(np.median(trans_fin)) if len(trans_fin) else float('inf'),
        med_scale_err    = float(np.median(scale_fin)) if len(scale_fin) else float('inf'),
        avg_inlier_ratio = float(np.mean(ir)),
        n_pairs          = len(rot),
    )

def _print_bucket_table(buckets):
    header = ['no_overlap (0%)', 'sparse (~30%)', 'dense (~70%)']
    w = 20
    print(f"\n  {'Overlap':<{w}} {'N':>5} {'recall_rot':>10} {'med_rot°':>9} "
          f"{'med_trans':>10} {'med_s_err':>10} {'inlier%':>8}")
    print(f"  {'-'*(w + 5 + 10 + 9 + 10 + 10 + 8 + 6)}")
    for b in header:
        d = buckets.get(b)
        if not d or d['n'] == 0:
            continue
        rots   = np.array(d['rot'])
        trans  = np.array(d['trans'])
        serrs  = np.array([x for x in d['scale'] if np.isfinite(x)])
        irs    = np.array(d['inlier_ratio'])
        recall = float((rots < 20).mean())
        med_r  = float(np.median(rots))
        med_t  = float(np.median(trans[np.isfinite(trans)])) if len(trans) else float('nan')
        med_s  = float(np.median(serrs)) if len(serrs) else float('nan')
        avg_ir = float(np.mean(irs))
        print(f"  {b:<{w}} {d['n']:>5} {recall:>10.3f} {med_r:>9.2f} "
              f"{med_t:>10.3f} {med_s:>10.3f} {avg_ir:>7.1f}%")

def parse_args():
    p = argparse.ArgumentParser(description='ScalePlueckerNet unified evaluation')
    p.add_argument('--checkpoint', required=True, help='Path to .pth checkpoint')
    p.add_argument('--dataset',    default='slam_map',
                   help='Comma-separated val split names (default: slam_map)')
    p.add_argument('--data_dir',   default=os.path.join(SCRIPT_DIR, 'dataset'))
    p.add_argument('--ransac',     default='sim3', choices=['sim3', 'grassmannian'],
                   help='RANSAC backend (default: sim3)')
    p.add_argument('--threshold',  type=float, default=None,
                   help='Single inlier threshold (default: 0.3 unless --sweep is set)')
    p.add_argument('--sweep',      action='store_true',
                   help='Sweep over [0.1, 0.3, 0.5, 1.0] thresholds. '
                        'Model runs once; only RANSAC is re-run per threshold.')
    p.add_argument('--n_iter',     type=int, default=500,
                   help='RANSAC iterations per hypothesis (default: 500)')
    p.add_argument('--max_pairs',  type=int, default=800,
                   help='Max val pairs to evaluate (default: 800)')
    p.add_argument('--out_dir',    default=os.path.join(SCRIPT_DIR, 'results', 'eval'),
                   help='Directory for JSON result files')
    p.add_argument('--label',      default=None, help='Human-readable label for results')
    return p.parse_args()

def main():
    args   = parse_args()
    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')

    datasets   = [d.strip() for d in args.dataset.split(',')]
    thresholds = [args.threshold] if args.threshold else \
                 ([0.1, 0.3, 0.5, 1.0] if args.sweep else [0.3])

    # Build a threshold-aware RANSAC wrapper
    if args.ransac == 'grassmannian':
        from sim3.ransac_grassmannian import ransac_sim3 as _rg
        def _make_ransac(thr):
            def _fn(p1k, p2k):
                R, t, s, mask, ic = _rg(p1k, p2k, n_iter=args.n_iter,
                                         inlier_angle_rad=thr)
                return s, R, t, ic, mask
            return _fn
        print(f'RANSAC: Grassmannian  n_iter={args.n_iter}')
    else:
        from sim3.ransac import run_ransac_sim3 as _rs
        def _make_ransac(thr):
            def _fn(p1k, p2k):
                return _rs(p1k, p2k, max_iterations=args.n_iter,
                           inlier_threshold=thr)
            return _fn
        print(f'RANSAC: Sim3 L2  n_iter={args.n_iter}')

    all_results = {}

    for dataset in datasets:
        print(f"\n{'='*65}")
        print(f"Dataset: {dataset}_valid")
        print(f"{'='*65}")

        cfg, ckpt = _build_config(args.checkpoint, args.data_dir, dataset)
        val_ds    = Sim3PluckerData(phase='valid', config=cfg)
        if len(val_ds) == 0:
            print(f'  ERROR: {dataset}_valid is empty — skipping')
            continue
        print(f'  Val scenes: {len(val_ds)}')

        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                                drop_last=False, num_workers=2)

        model = _load_model(cfg, ckpt, device)

        # Handle 9D checkpoint evaluated on a 6D dataset
        ckpt_ch  = getattr(cfg, 'in_channel', 6)
        sample   = next(iter(val_loader))[1]
        data_ch  = sample.shape[-1]
        if ckpt_ch == 9 and data_ch == 6:
            print('  (9D checkpoint on 6D dataset — padding with neutral LAB)')
            model = _Pad6Dto9DWrapper(model)

        model = model.to(device)

        # Run model inference once for this dataset
        print(f'  Running model inference (max {args.max_pairs} pairs)…')
        cache = _run_inference(model, val_loader, device, args.max_pairs)
        print(f'  Cached {len(cache)} pairs')

        if len(thresholds) > 1:
            # Threshold sweep — model already ran, only RANSAC varies
            print(f'\n{"threshold":>10s}  {"recall_rot":>10s}  {"<5°":>6s}  '
                  f'{"<10°":>6s}  {"<20°":>6s}  {"med_rot":>8s}  '
                  f'{"med_s_err":>9s}  {"inlier%":>7s}')
            print('-' * 80)
            for thr in thresholds:
                rot, trans, scale, ir, _ = _eval_cached(cache, _make_ransac(thr))
                m = _summarise(rot, trans, scale, ir)
                print(f'{thr:>10.3f}  {m["recall_rot"]:>10.3f}  '
                      f'{m["pct_lt5"]:>6.3f}  {m["pct_lt10"]:>6.3f}  '
                      f'{m["pct_lt20"]:>6.3f}  {m["med_rot"]:>8.1f}°  '
                      f'{m["med_scale_err"]:>9.3f}  '
                      f'{m["avg_inlier_ratio"]:>7.1f}%  (n={m["n_pairs"]})')
        else:
            # Single threshold — full breakdown + JSON output
            thr = thresholds[0]
            rot, trans, scale, ir, buckets = _eval_cached(cache, _make_ransac(thr))
            m = _summarise(rot, trans, scale, ir)

            print(f'\n  Overall ({m["n_pairs"]} scenes):')
            print(f'    recall_rot={m["recall_rot"]:.3f}  '
                  f'med_rot={m["med_rot"]:.2f}°  '
                  f'med_trans={m["med_trans"]:.3f}  '
                  f'med_s_err={m["med_scale_err"]:.3f}  '
                  f'inlier%={m["avg_inlier_ratio"]:.1f}%')
            _print_bucket_table(buckets)

            os.makedirs(args.out_dir, exist_ok=True)
            label = args.label or (
                f'{os.path.basename(os.path.dirname(args.checkpoint))} → {dataset}'
            )
            safe    = label.replace(' ', '_').replace('/', '-').replace('→', 'on')
            out_f   = os.path.join(args.out_dir, f'{safe}.json')
            with open(out_f, 'w') as f:
                json.dump({'label': label, 'checkpoint': args.checkpoint,
                           'dataset': dataset, 'threshold': thr,
                           'ransac': args.ransac, 'metrics': m,
                           'by_overlap': {b: {k: v for k, v in d.items() if k != 'n'}
                                          for b, d in buckets.items()}},
                          f, indent=2)
            print(f'\n  Results saved: {out_f}')
            all_results[dataset] = m
    
    if len(all_results) > 1:
        print(f'\n{"="*75}')
        print(f'{"Dataset":<20} {"recall_rot":>10} {"med_rot (°)":>12} '
              f'{"med_trans (m)":>14} {"inlier_ratio":>13}')
        print(f'{"-"*75}')
        for ds, m in all_results.items():
            print(f'{ds:<20} {m["recall_rot"]:>10.3f} {m["med_rot"]:>12.2f} '
                  f'{m["med_trans"]:>14.3f} {m["avg_inlier_ratio"]:>12.1f}%')
        print(f'{"="*75}')

if __name__ == '__main__':
    main()
