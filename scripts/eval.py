"""
  Cross-dataset: evaluate a checkpoint on one or more val splits

  python scripts/eval.py --weights output/joint/2026-05-09/best_val_checkpoint.pth \
      --dataset replica_gs

  python scripts/eval.py --weights output/joint/2026-05-09/best_val_checkpoint.pth \
      --dataset replica_gs,7scenes_gs,se3real_sim3
"""

import os
import sys
import argparse
import json
import time
import warnings
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

SCRIPTS_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT         = os.path.dirname(SCRIPTS_DIR)
PLUECKERNET  = os.path.abspath(os.path.join(ROOT, 'PlueckerNet'))
sys.path.insert(0, PLUECKERNET)
sys.path.insert(0, ROOT)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class _DustbinWrapper(nn.Module):
    """Strips dustbin row/col from P_aug so Sim3Trainer._valid_epoch() works unchanged."""
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

def _load_model(configs, weights_path):
    ckpt = torch.load(weights_path, weights_only=False)
    sd   = ckpt.get('state_dict', ckpt.get('model', ckpt))
    is_dustbin = any('bin_dist' in k or 'bin_score' in k for k in sd)

    if is_dustbin:
        from sim3.model_dustbin import PluckerNetKnnDustbin
        model = PluckerNetKnnDustbin(configs)
        model.load_state_dict(sd, strict=True)
        print('  (dustbin checkpoint — wrapping for eval)')
        return _DustbinWrapper(model)
    else:
        from lib.utils import load_model
        Model = load_model('PluckerNetKnn')
        model = Model(configs)
        model.load_state_dict(sd)
        return model

def rot_err_deg(R_est, R_gt):
    tr = np.clip((np.trace(R_est @ R_gt.T) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(tr)))

def scale_err_log(s_est, s_gt):
    return float(abs(np.log(max(float(s_est), 1e-6)) - np.log(max(float(s_gt), 1e-6))))

def trans_err(t_est, t_gt):
    return float(np.linalg.norm(np.asarray(t_est).flatten() - np.asarray(t_gt).flatten()))

def _overlap_bucket(n_inliers):
    """Map inlier count to a human-readable overlap bucket."""
    if n_inliers == 0:
        return 'no_overlap (0%)'
    if n_inliers <= 200:       # covers 5–30% levels
        return 'sparse (~30%)'
    return 'dense (~70%)'      # covers 50–100% levels

def _print_bucket_table(bucket_data):
    """Print per-overlap-bucket metric table."""
    buckets = ['no_overlap (0%)', 'sparse (~30%)', 'dense (~70%)']
    w = 20
    print(f"\n  {'Overlap':<{w}} {'N':>5} {'recall_rot':>10} {'med_rot°':>9} "
          f"{'med_trans':>10} {'med_s_err':>10} {'inlier%':>8}")
    print(f"  {'-'*(w+5+10+9+10+10+8+6)}")
    for b in buckets:
        d = bucket_data.get(b)
        if d is None or d['n'] == 0:
            continue
        rots   = np.array(d['rot'])
        trans  = np.array(d['trans'])
        serrs  = np.array([x for x in d['scale'] if np.isfinite(x)])
        irs    = np.array(d['inlier_ratio'])
        recall = float((rots < 20).mean())
        med_r  = float(np.median(rots))
        med_t  = float(np.median(trans))
        med_s  = float(np.median(serrs)) if len(serrs) else float('nan')
        avg_ir = float(np.mean(irs))
        print(f"  {b:<{w}} {d['n']:>5} {recall:>10.3f} {med_r:>9.2f} "
              f"{med_t:>10.3f} {med_s:>10.3f} {avg_ir:>7.1f}%")

def cross_dataset_eval(weights_path, datasets, data_dir, out_dir, label=None, ransac='sim3'):
    from easydict import EasyDict as edict
    from torch.utils.data import DataLoader
    from sim3.dataloader import Sim3PluckerData

    if ransac == 'grassmannian':
        from sim3.ransac_grassmannian import ransac_sim3 as _ransac_g
        def _run_ransac(p1k, p2k):
            R, t, s, mask, ic = _ransac_g(p1k, p2k, n_iter=1000, inlier_angle_rad=0.10)
            return s, R, t, ic, mask
        print(f"  RANSAC: Grassmannian")
    else:
        from sim3.ransac import run_ransac_sim3 as _run_ransac_sim3
        def _run_ransac(p1k, p2k):
            return _run_ransac_sim3(p1k, p2k, inlier_threshold=0.1)
        print(f"  RANSAC: Sim3 L2")

    ckpt = torch.load(weights_path, weights_only=False)
    configs = ckpt.get('config')
    if configs is None:
        raise ValueError('Checkpoint has no "config" key — cannot infer model settings.')
    if not isinstance(configs, edict):
        configs = edict(configs)

    results = {}
    for dataset in datasets:
        run_label = label or f"{os.path.basename(os.path.dirname(weights_path))} → {dataset}"
        print(f"\n{'='*60}")
        print(f"Cross-dataset eval: {run_label}")
        print(f"  weights : {weights_path}")
        print(f"  dataset : {dataset}_valid  ({data_dir})")
        print(f"{'='*60}\n")

        cfg = edict(dict(configs))
        cfg.dataset  = dataset
        cfg.data_dir = data_dir
        cfg.resume   = None
        cfg.weights  = None
        cfg.model_nb = 'eval'

        val_loader = DataLoader(
            Sim3PluckerData(phase='valid', config=cfg),
            batch_size=1, shuffle=False, drop_last=False, num_workers=2,
        )
        if len(val_loader.dataset) == 0:
            print(f"ERROR: {dataset}_valid is empty — skipping.")
            continue

        print(f"Validation set: {len(val_loader.dataset)} scenes")

        model = _load_model(cfg, weights_path)
        checkpoint_channels = getattr(configs, 'in_channel', 6)
        sample = next(iter(val_loader))[1]
        data_channels = min(sample.shape[1], sample.shape[2])
        if checkpoint_channels == 9 and data_channels == 6:
            print('  (9D checkpoint on 6D dataset — padding with neutral LAB)')
            model = _Pad6Dto9DWrapper(model)
        model = model.to(DEVICE).eval()

        # Per-scene accumulators
        all_rot, all_trans, all_scale, all_ir = [], [], [], []
        bucket_data = {b: {'n': 0, 'rot': [], 'trans': [], 'scale': [], 'inlier_ratio': []}
                       for b in ['no_overlap (0%)', 'sparse (~30%)', 'dense (~70%)']}

        with torch.no_grad():
            for batch in val_loader:
                matches, p1, p2, R_gt, t_gt, s_gt = batch
                n_inliers = int(matches.sum().item())
                bucket    = _overlap_bucket(n_inliers)

                p1_d = p1.to(DEVICE)
                p2_d = p2.to(DEVICE)
                prob, _, _ = model(p1_d, p2_d)

                k = min(100, prob.shape[1] * prob.shape[2])
                _, flat = torch.topk(prob.flatten(start_dim=-2), k=k, dim=-1)
                i1 = (flat // prob.shape[-1]).squeeze(0).cpu().numpy()
                i2 = (flat  % prob.shape[-1]).squeeze(0).cpu().numpy()
                
                match_mat = matches[0].numpy()
                ir = float(match_mat[i1, i2].sum() / k * 100)
                
                err_r, err_t, err_s = 180.0, float('inf'), float('inf')
                if n_inliers > 0 and k > 3:
                    p1k = p1[0, i1, :6].numpy().T
                    p2k = p2[0, i2, :6].numpy().T
                    best_s, best_R, best_t, _, _ = _run_ransac(p1k, p2k)
                    if best_R is not None:
                        err_r = rot_err_deg(best_R, R_gt[0].numpy())
                        err_t = trans_err(best_t, t_gt[0].numpy())
                        sv    = float(s_gt[0])
                        if best_s > 0 and sv > 0:
                            err_s = scale_err_log(best_s, sv)

                all_rot.append(err_r); all_trans.append(err_t)
                all_scale.append(err_s); all_ir.append(ir)
                bd = bucket_data[bucket]
                bd['n'] += 1
                bd['rot'].append(err_r); bd['trans'].append(err_t)
                bd['scale'].append(err_s); bd['inlier_ratio'].append(ir)
        
        rots  = np.array(all_rot)
        trans = np.array(all_trans)
        fins  = np.array([x for x in all_scale if np.isfinite(x)])
        recall_rot       = float((rots < 20).mean())
        med_rot          = float(np.median(rots))
        med_trans        = float(np.median(trans[np.isfinite(trans)]))
        med_scale_err    = float(np.median(fins)) if len(fins) else float('nan')
        avg_inlier_ratio = float(np.mean(all_ir))

        metrics = dict(recall_rot=recall_rot, med_rot=med_rot,
                       med_trans=med_trans, med_scale_err=med_scale_err,
                       avg_inlier_ratio=avg_inlier_ratio)

        print(f"\n  Overall ({len(all_rot)} scenes):")
        print(f"    recall_rot={recall_rot:.3f}  med_rot={med_rot:.2f}°  "
              f"med_trans={med_trans:.3f}  med_s_err={med_scale_err:.3f}  "
              f"inlier%={avg_inlier_ratio:.1f}%")

        _print_bucket_table(bucket_data)

        os.makedirs(out_dir, exist_ok=True)
        safe = run_label.replace(' ', '_').replace('/', '-').replace('→', 'on')
        out_path = os.path.join(out_dir, f'{safe}.json')
        with open(out_path, 'w') as f:
            json.dump({'label': run_label, 'weights': weights_path,
                       'dataset': dataset, 'metrics': metrics,
                       'by_overlap': {b: {k: v for k, v in d.items() if k != 'n'}
                                      for b, d in bucket_data.items()}}, f, indent=2)
        results[dataset] = metrics
    
    if results:
        print(f"\n{'='*75}")
        print(f"{'Dataset':<20} {'recall_rot':>10} {'med_rot (°)':>12} {'med_trans (m)':>14} {'inlier_ratio':>13}")
        print(f"{'-'*75}")
        for ds, m in results.items():
            print(f"{ds:<20} {m['recall_rot']:>10.3f} {m['med_rot']:>12.2f} "
                  f"{m['med_trans']:>14.3f} {m['avg_inlier_ratio']:>12.1f}%")
        print(f"{'='*75}")

    return results

def main():
    p = argparse.ArgumentParser(description='ScalePlueckerNet evaluation')
    p.add_argument('--weights',      default=None,  required=True,
                   help='Checkpoint path (required for cross-dataset and chess modes)')
    p.add_argument('--dataset',      default='semantic3D,structured3D,replica_gs,7scenes_gs',
                   help='Comma-separated val split names (default: all four datasets)')
    p.add_argument('--data_dir',     default=os.path.join(ROOT, 'dataset'))
    p.add_argument('--label',        default=None,  help='Human-readable label')
    p.add_argument('--out_dir',      default=os.path.join(ROOT, 'results', 'eval_cross_dataset'))
    p.add_argument('--ransac',       default='sim3', choices=['sim3', 'grassmannian'],
                   help='RANSAC backend: sim3 (L2, default) | grassmannian (angle-based)')
    
    args = p.parse_args()

    datasets = [d.strip() for d in args.dataset.split(',')]
    cross_dataset_eval(args.weights, datasets, args.data_dir, args.out_dir, args.label, args.ransac)

if __name__ == '__main__':
    main()
