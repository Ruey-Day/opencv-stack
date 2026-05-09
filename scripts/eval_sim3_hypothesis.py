#!/usr/bin/env python3
"""
eval_sim3_hypothesis.py — Verify: Sim3-Net is better on Sim3 and similar on SE3.

Hypothesis
----------
Training on the original PlueckerNet-style synthetic dataset WITH random scale
augmentation should produce a model that:
  (H1) Recovers scale correctly when scale ≠ 1  (Sim3 scenario)
  (H2) Matches SE3-Net rotation/translation accuracy when scale = 1  (SE3 scenario)
  (H3) SE3-Net structurally fails (scale error = |log s|) under Sim3

Evaluation
----------
  C1  Synthetic SE3 test  (scale = 1.0, 200 scenes) — SE3 scenario
  C2  Synthetic Sim3 test (scale uniform in [0.3, 3.0], 200 scenes) — Sim3 scenario

Methods compared
  M1  SE3-PlueckerNet  — original weights + SE3 RANSAC  (input: [d, m])
  M2  Sim3-Net (synth) — trained on sim3_synthetic      (input: [m, d])

Usage
-----
    # After training completes:
    python scripts/eval_sim3_hypothesis.py
    # Or with a specific checkpoint:
    python scripts/eval_sim3_hypothesis.py --weights output/sim3_synthetic/YYYY-MM-DD/best_val_checkpoint.pth

Outputs saved to results/eval_hypothesis/
"""

import argparse
import os
import sys
import time
import warnings
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

warnings.filterwarnings("ignore")

ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLUECKERNET  = os.path.abspath(os.path.join(ROOT, "..", "PlueckerNet"))
sys.path.insert(0, PLUECKERNET)
sys.path.insert(0, ROOT)

SE3_WEIGHTS_DEFAULT  = os.path.join(PLUECKERNET, "output", "semantic3D",
                                    "preTrained", "best_val_checkpoint_real.pth")
SIM3_WEIGHTS_DEFAULT = os.path.join(ROOT, "output", "sim3_synthetic",
                                    "2026-05-08", "best_val_checkpoint.pth")
OUT_DIR = os.path.join(ROOT, "results", "eval_hypothesis")
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Plücker geometry ───────────────────────────────────────────────────────────

def random_rotation(rng):
    A = rng.standard_normal((3, 3))
    Q, R_mat = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R_mat))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float32)


def make_direction_clustered_lines(rng, n, n_dir_clusters=10, dir_spread=0.15, pos_range=2.0):
    n_per = n // n_dir_clusters
    extras = n - n_per * n_dir_clusters
    anchor_dirs = rng.standard_normal((n_dir_clusters, 3)).astype(np.float32)
    anchor_dirs /= np.linalg.norm(anchor_dirs, axis=1, keepdims=True)
    parts = []
    for i, anchor_d in enumerate(anchor_dirs):
        cnt = n_per + (1 if i < extras else 0)
        noise = (rng.standard_normal((cnt, 3)) * dir_spread).astype(np.float32)
        d = anchor_d[None] + noise
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        p = rng.uniform(-pos_range, pos_range, (cnt, 3)).astype(np.float32)
        m = np.cross(p, d)
        parts.append(np.concatenate([m, d], axis=1))
    lines = np.concatenate(parts, axis=0)
    idx = rng.permutation(len(lines))
    return lines[idx]


def apply_sim3(lines, s, R, t):
    m, d = lines[:, :3], lines[:, 3:6]
    d_new = (R @ d.T).T
    m_new = s * (R @ m.T).T + np.cross(t, d_new)
    return np.concatenate([m_new, d_new], axis=1).astype(np.float32)


def generate_test_scenes(n_scenes, scale_fn, seed=0, n_inliers=100, n_outliers=30):
    """Generate test scenes. scale_fn(rng) returns the scale for each scene."""
    rng = np.random.default_rng(seed)
    scenes = []
    for _ in range(n_scenes):
        lines1_in = make_direction_clustered_lines(rng, n_inliers)
        s = float(scale_fn(rng))
        R = random_rotation(rng)
        t = rng.uniform(-1.5, 1.5, 3).astype(np.float32)
        lines2_in = apply_sim3(lines1_in, s, R, t)

        n_out = max(1, 3)
        lines1_out = make_direction_clustered_lines(rng, n_outliers, n_dir_clusters=n_out)
        lines2_out = make_direction_clustered_lines(rng, n_outliers, n_dir_clusters=n_out)

        lines1 = np.concatenate([lines1_in, lines1_out])
        lines2 = np.concatenate([lines2_in, lines2_out])

        idx1 = rng.permutation(len(lines1))
        idx2 = rng.permutation(len(lines2))
        lines1 = lines1[idx1]
        lines2 = lines2[idx2]

        inv1 = np.argsort(idx1)
        inv2 = np.argsort(idx2)
        src = inv1[:n_inliers]
        tgt = inv2[:n_inliers]

        scenes.append({
            'plucker1': lines1.astype(np.float32),
            'plucker2': lines2.astype(np.float32),
            'matches':  np.stack([src, tgt], axis=0).astype(np.int32),
            'R_gt':     R,
            't_gt':     t,
            's_gt':     np.float32(s),
        })
    return scenes


# ── Model loading ──────────────────────────────────────────────────────────────

def load_se3_model(weights_path):
    from easydict import EasyDict as edict
    from model.model_plucker import PluckerNetKnn
    import lib.ransac_l2l as _rm

    def _skew_fixed(x):
        x = np.asarray(x).flatten()
        return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    _rm.skew = _skew_fixed
    from lib.ransac_l2l import run_ransac

    cfg   = edict(net_nchannel=128, GNN_layers=["self", "cross"] * 6,
                  net_lambda=0.1, net_maxiter=30, net_topK=200)
    model = PluckerNetKnn(cfg).to(DEVICE)
    ckpt  = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[SE3  model] {weights_path}")
    return model, run_ransac


def load_sim3_model(weights_path):
    from easydict import EasyDict as edict
    from model.model_plucker import PluckerNetKnn
    cfg   = edict(net_nchannel=128, GNN_layers=["self", "cross"] * 6,
                  net_lambda=0.1, net_maxiter=30, net_topK=200)
    model = PluckerNetKnn(cfg).to(DEVICE)
    ckpt  = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[Sim3 model] {weights_path}")
    return model


# ── Inference helpers ──────────────────────────────────────────────────────────

def md_to_dm(L): return np.hstack([L[:, 3:], L[:, :3]])


@torch.no_grad()
def _topk(model, L1, L2, topk):
    t1 = torch.from_numpy(L1).unsqueeze(0).to(DEVICE)
    t2 = torch.from_numpy(L2).unsqueeze(0).to(DEVICE)
    P, _, _ = model(t1, t2)
    k = min(topk, P.shape[1] * P.shape[2])
    _, flat = torch.topk(P.flatten(start_dim=-2), k=k, dim=-1)
    i1 = (flat // P.shape[-1]).squeeze(0).cpu().numpy()
    i2 = (flat  % P.shape[-1]).squeeze(0).cpu().numpy()
    return i1, i2


_FAIL = dict(R=np.eye(3), t=np.zeros(3), s=1.0, ic=0, ms=0.0)


def run_se3_net(model, ransac_fn, scene, topk=200, threshold=0.5):
    L1, L2 = scene['plucker1'], scene['plucker2']
    t0 = time.perf_counter()
    i1, i2 = _topk(model, md_to_dm(L1), md_to_dm(L2), topk)
    p1, p2 = md_to_dm(L1)[i1].T, md_to_dm(L2)[i2].T
    R, t, ic, _ = ransac_fn(p1, p2, inlier_threshold=threshold)
    ms = (time.perf_counter() - t0) * 1000
    if R is None:
        return {**_FAIL, 'ms': ms}
    return dict(R=R, t=np.asarray(t).flatten(), s=1.0, ic=int(ic), ms=ms)


def run_sim3_net(model, scene, topk=100, threshold=0.1):
    from sim3.ransac import run_ransac_sim3
    L1, L2 = scene['plucker1'], scene['plucker2']
    t0 = time.perf_counter()
    i1, i2 = _topk(model, L1, L2, topk)
    p1, p2 = L1[i1].T, L2[i2].T
    s, R, t, ic, _ = run_ransac_sim3(p1, p2, inlier_threshold=threshold)
    ms = (time.perf_counter() - t0) * 1000
    if R is None:
        return {**_FAIL, 'ms': ms}
    return dict(R=R, t=np.asarray(t).flatten(), s=float(s) if s else 1.0,
                ic=int(ic), ms=ms)


# ── Metrics ────────────────────────────────────────────────────────────────────

def rot_err_deg(R_est, R_gt):
    tr = np.clip((np.trace(R_est @ R_gt.T) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(tr)))


def scale_err_log(s_est, s_gt):
    return float(abs(np.log(max(s_est, 1e-6)) - np.log(max(s_gt, 1e-6))))


def compute_metrics(res, scene):
    return {
        'rot':   rot_err_deg(res['R'], scene['R_gt']),
        'trans': float(np.linalg.norm(res['t'] - scene['t_gt'])),
        'scale': scale_err_log(res['s'], float(scene['s_gt'])),
        'ic':    res['ic'],
        'ms':    res['ms'],
    }


def eval_on_scenes(run_fn, scenes):
    metrics = []
    for sc in scenes:
        res = run_fn(sc)
        metrics.append(compute_metrics(res, sc))
    return metrics


def summarise(metrics, label):
    rots   = [m['rot']   for m in metrics]
    trans  = [m['trans'] for m in metrics]
    scales = [m['scale'] for m in metrics]
    inliers = [m['ic']   for m in metrics]
    print(f"  {label:35s}  "
          f"med_rot={np.median(rots):6.2f}°  "
          f"mean_rot={np.mean(rots):6.2f}°  "
          f"med_s_err={np.median(scales):.3f}  "
          f"med_trans={np.median(trans):.4f}m  "
          f"med_ic={np.median(inliers):.0f}")
    return {
        'med_rot':  float(np.median(rots)),
        'mean_rot': float(np.mean(rots)),
        'med_trans': float(np.median(trans)),
        'med_scale': float(np.median(scales)),
        'mean_scale': float(np.mean(scales)),
        'med_ic':   float(np.median(inliers)),
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

def savefig(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [saved] {os.path.relpath(path, ROOT)}")


def cdf_plot(ax, metrics_list, labels, colors, key, xlabel, title, xlim=None):
    for m_list, label, color in zip(metrics_list, labels, colors):
        vals = sorted(m[key] for m in m_list)
        cdf  = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, cdf, label=label, color=color, lw=2)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Fraction of scenes", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    if xlim is not None:
        ax.set_xlim(0, xlim)


def bar_pair(ax, se3_val, sim3_val, ylabel, title, colors):
    bars = ax.bar(['SE3-Net', 'Sim3-Net\n(synth)'], [se3_val, sim3_val],
                  color=colors, edgecolor='white', alpha=0.88, width=0.5)
    top = max(se3_val, sim3_val)
    for bar, v in zip(bars, [se3_val, sim3_val]):
        label = f'{v:.2f}' if v < 100 else f'{v:.1f}'
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + top * 0.04,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, top * 1.4 + 1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=SIM3_WEIGHTS_DEFAULT,
                        help='Path to Sim3-Net checkpoint (trained on sim3_synthetic)')
    parser.add_argument('--se3_weights', default=SE3_WEIGHTS_DEFAULT,
                        help='Path to SE3-PlueckerNet checkpoint')
    parser.add_argument('--n_scenes', type=int, default=200,
                        help='Number of test scenes per experiment')
    parser.add_argument('--skip_se3', action='store_true',
                        help='Skip SE3-Net evaluation (faster, avoids domain-mismatch noise)')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\nDevice: {DEVICE}")
    print(f"Sim3 weights: {args.weights}")
    print(f"SE3  weights: {args.se3_weights}")
    print(f"Output: {OUT_DIR}")

    # ── Generate test scenes ─────────────────────────────────────────────────
    print(f"\nGenerating {args.n_scenes} SE3 test scenes (scale = 1.0) ...")
    se3_scenes = generate_test_scenes(args.n_scenes,
                                      scale_fn=lambda rng: 1.0,
                                      seed=1000)

    print(f"Generating {args.n_scenes} Sim3 test scenes (scale ∈ [0.3, 3.0]) ...")
    sim3_scenes = generate_test_scenes(args.n_scenes,
                                       scale_fn=lambda rng: float(
                                           np.exp(rng.uniform(np.log(0.3), np.log(3.0)))),
                                       seed=2000)

    sim3_scales = [float(sc['s_gt']) for sc in sim3_scenes]
    print(f"  GT scale range: [{min(sim3_scales):.2f}, {max(sim3_scales):.2f}]")

    # ── Load models ──────────────────────────────────────────────────────────
    sim3_model = load_sim3_model(args.weights)

    se3_model, se3_ransac = None, None
    if not args.skip_se3 and os.path.exists(args.se3_weights):
        se3_model, se3_ransac = load_se3_model(args.se3_weights)
    elif not args.skip_se3:
        print(f"[WARNING] SE3 weights not found at {args.se3_weights} — skipping SE3-Net")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print("\n─── C1: SE3 test (scale = 1.0) ───")
    sim3_on_se3 = eval_on_scenes(
        lambda sc: run_sim3_net(sim3_model, sc, topk=100, threshold=0.1),
        se3_scenes)
    sum_sim3_se3 = summarise(sim3_on_se3, "Sim3-Net on SE3 test")

    se3_on_se3 = None
    if se3_model is not None:
        se3_on_se3 = eval_on_scenes(
            lambda sc: run_se3_net(se3_model, se3_ransac, sc, topk=200, threshold=0.5),
            se3_scenes)
        sum_se3_se3 = summarise(se3_on_se3, "SE3-Net  on SE3 test (note: domain shift)")

    print("\n─── C2: Sim3 test (scale ∈ [0.3, 3.0]) ───")
    sim3_on_sim3 = eval_on_scenes(
        lambda sc: run_sim3_net(sim3_model, sc, topk=100, threshold=0.1),
        sim3_scenes)
    sum_sim3_sim3 = summarise(sim3_on_sim3, "Sim3-Net on Sim3 test")

    se3_on_sim3 = None
    if se3_model is not None:
        se3_on_sim3 = eval_on_scenes(
            lambda sc: run_se3_net(se3_model, se3_ransac, sc, topk=200, threshold=0.5),
            sim3_scenes)
        sum_se3_sim3 = summarise(se3_on_sim3, "SE3-Net  on Sim3 test (note: domain shift)")

    # ── Hypothesis check ─────────────────────────────────────────────────────
    print("\n══ Hypothesis verification ══")
    h1_ok = sum_sim3_sim3['med_scale'] < 0.10
    h2_ok = sum_sim3_se3['med_rot'] < 5.0
    print(f"  H1 (scale recovery when s≠1):  "
          f"med_scale_err = {sum_sim3_sim3['med_scale']:.3f}  "
          f"→ {'PASS ✓' if h1_ok else 'FAIL ✗'}")
    print(f"  H2 (rotation accuracy when s=1): "
          f"med_rot = {sum_sim3_se3['med_rot']:.2f}°  "
          f"→ {'PASS ✓' if h2_ok else 'FAIL ✗'}")

    # ── Figures ──────────────────────────────────────────────────────────────
    COLORS = ['#1f77b4', '#d62728']    # blue=SE3-Net, red=Sim3-Net

    # Figure h01: Rotation error CDFs — SE3 and Sim3 test sets
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if se3_on_se3:
        cdf_plot(axes[0], [se3_on_se3, sim3_on_se3],
                 ['SE3-Net (Semantic3D trained)', 'Sim3-Net (synthetic trained)'],
                 COLORS, 'rot', 'Rotation error (°)',
                 'C1 — SE3 test (scale=1)\nRotation error CDF', xlim=30)
    else:
        cdf_plot(axes[0], [sim3_on_se3], ['Sim3-Net (synthetic trained)'],
                 [COLORS[1]], 'rot', 'Rotation error (°)',
                 'C1 — SE3 test (scale=1)\nRotation error CDF', xlim=30)
    if se3_on_sim3:
        cdf_plot(axes[1], [se3_on_sim3, sim3_on_sim3],
                 ['SE3-Net (Semantic3D trained)', 'Sim3-Net (synthetic trained)'],
                 COLORS, 'rot', 'Rotation error (°)',
                 'C2 — Sim3 test (scale ∈ [0.3, 3.0])\nRotation error CDF', xlim=60)
    else:
        cdf_plot(axes[1], [sim3_on_sim3], ['Sim3-Net (synthetic trained)'],
                 [COLORS[1]], 'rot', 'Rotation error (°)',
                 'C2 — Sim3 test (scale ∈ [0.3, 3.0])\nRotation error CDF', xlim=60)
    fig.suptitle("Experiment C — Rotation Error: SE3 scenario (left) vs Sim3 scenario (right)\n"
                 "Both metrics should be low for Sim3-Net → scale head does not hurt SE3",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    savefig(fig, "fig_h01_rotation_cdf.png")

    # Figure h02: Scale error CDFs on Sim3 test
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if se3_on_sim3:
        cdf_plot(axes[0], [se3_on_sim3, sim3_on_sim3],
                 ['SE3-Net (always s=1)', 'Sim3-Net (recovers scale)'],
                 COLORS, 'scale', 'Scale error |log(ŝ/s)|',
                 'C2 — Sim3 test: Scale error CDF', xlim=2.5)
    else:
        cdf_plot(axes[0], [sim3_on_sim3], ['Sim3-Net (recovers scale)'],
                 [COLORS[1]], 'scale', 'Scale error |log(ŝ/s)|',
                 'C2 — Sim3 test: Scale error CDF', xlim=2.5)
    # Scale error on SE3 test (should be ~0 for both)
    cdf_plot(axes[1], [sim3_on_se3],
             ['Sim3-Net on SE3 test (s=1, scale error ≈ 0)'],
             [COLORS[1]], 'scale', 'Scale error |log(ŝ/s)|',
             'C1 — SE3 test: Scale error CDF\n(GT s=1 → Sim3 should recover s≈1)', xlim=1.0)
    fig.suptitle("Experiment C — Scale Error\n"
                 "Left: SE3-Net structurally fails (s always 1), Sim3-Net recovers scale\n"
                 "Right: on SE3 test (s=1), Sim3-Net also returns s≈1",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    savefig(fig, "fig_h02_scale_error_cdf.png")

    # Figure h03: Summary bar chart comparing median metrics
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics_sim3_on_se3  = [sum_sim3_se3['med_rot'],  sum_sim3_se3['med_scale'],  sum_sim3_se3['med_ic']]
    metrics_sim3_on_sim3 = [sum_sim3_sim3['med_rot'], sum_sim3_sim3['med_scale'], sum_sim3_sim3['med_ic']]

    scenarios = ['C1: SE3 test\n(scale=1)', 'C2: Sim3 test\n(scale∈[0.3,3])']
    x = np.arange(2)
    for ax, vals, ylabel, title in zip(
        axes,
        [
            [sum_sim3_se3['med_rot'],   sum_sim3_sim3['med_rot']],
            [sum_sim3_se3['med_scale'], sum_sim3_sim3['med_scale']],
            [sum_sim3_se3['med_ic'],    sum_sim3_sim3['med_ic']],
        ],
        ['Median rotation error (°)', 'Median scale error |log(ŝ/s)|', 'Median RANSAC inliers'],
        ['Sim3-Net: Rotation error\n(should be low in both)',
         'Sim3-Net: Scale error\n(should be near 0 in C1 and C2)',
         'Sim3-Net: Inlier count\n(should be high in both)'],
    ):
        bars = ax.bar(x, vals, color=['#2ca02c', '#ff7f0e'], edgecolor='white', alpha=0.88, width=0.5)
        top = max(vals) if max(vals) > 0 else 1.0
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + top * 0.04,
                    f'{v:.2f}' if v < 10 else f'{v:.1f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, top * 1.45 + 1e-9)

    fig.suptitle(
        "Experiment C — Sim3-Net (trained on synthetic PlueckerNet-style data + scale)\n"
        "Key result: similar rotation accuracy on SE3 (C1) and Sim3 (C2); scale recovery on C2",
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    savefig(fig, "fig_h03_sim3net_summary.png")

    # ── Print summary table ──────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("EXPERIMENT C — RESULTS TABLE")
    print("═" * 70)
    print(f"{'Scenario':<30} {'Method':<28} {'med_rot(°)':>10} {'med_s_err':>10} {'med_ic':>8}")
    print("─" * 70)

    rows = [
        ("C1: SE3 test (s=1.0)",   "Sim3-Net (synth)",   sum_sim3_se3),
        ("C2: Sim3 test (s∈[0.3,3])", "Sim3-Net (synth)", sum_sim3_sim3),
    ]
    if se3_on_se3:
        rows.insert(0, ("C1: SE3 test (s=1.0)",   "SE3-Net (semantic3D)*", sum_se3_se3))
    if se3_on_sim3:
        rows.insert(len(rows) - 1 if se3_on_se3 else 0,
                    ("C2: Sim3 test (s∈[0.3,3])", "SE3-Net (semantic3D)*", sum_se3_sim3))

    prev_scenario = None
    for scenario, method, s in rows:
        if scenario != prev_scenario and prev_scenario is not None:
            print("─" * 70)
        prev_scenario = scenario
        print(f"  {scenario:<28} {method:<28} "
              f"{s['med_rot']:>10.2f} {s['med_scale']:>10.3f} {s['med_ic']:>8.0f}")

    print("─" * 70)
    if se3_on_se3 or se3_on_sim3:
        print("  * SE3-Net trained on Semantic3D (real LiDAR); evaluated on synthetic data.")
        print("    Domain mismatch is expected — SE3-Net numbers on synthetic are not directly comparable.")
    print("\n  Key finding:")
    print(f"    H1 scale recovery  (Sim3-Net, C2): med_scale_err = {sum_sim3_sim3['med_scale']:.3f}"
          f"  → {'PASS' if h1_ok else 'FAIL'}")
    print(f"    H2 SE3 accuracy    (Sim3-Net, C1): med_rot       = {sum_sim3_se3['med_rot']:.2f}°"
          f"  → {'PASS' if h2_ok else 'FAIL'}")
    print("═" * 70)
    print(f"\nFigures saved to {os.path.relpath(OUT_DIR, ROOT)}/")


if __name__ == "__main__":
    main()
