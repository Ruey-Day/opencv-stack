#!/usr/bin/env python3
"""
eval_benchmark_replica.py — 4-method benchmark adding Replica-trained model.

Methods compared
----------------
  M1 SE3-PlueckerNet       : original pretrained weights + SE3 RANSAC
  M2 Sim3-Net (synthetic)  : trained on synthetic direction-clustered data
  M3 Sim3-Net (Replica)    : trained on real Replica RGBD geometry  ← new
  M4 Pure-Sim3-RANSAC      : direction cosine-NN + Sim3 RANSAC (no network)

Outputs saved to results/eval_replica/  (original results/eval/ untouched)

Experiments are identical to eval_benchmark.py:
  A1 Cube wireframe — SE3 transform
  A2 Cube wireframe — Sim3 transform (s=1.8)
  B1 Chess cross-sequence RGBD (metric scale)
  B2 Chess cross-sequence RGB-only (simulated scale ambiguity)
"""

import os, sys, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import torch

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT          = os.path.dirname(os.path.abspath(__file__))
PLUECKERNET   = os.path.abspath(os.path.join(ROOT, "..", "PlueckerNet"))
SE3_WEIGHTS   = os.path.join(PLUECKERNET, "output", "semantic3D",
                              "preTrained", "best_val_checkpoint_real.pth")
SIM3_SYNTH_W  = os.path.join(ROOT, "output", "sim3_synthetic",
                              "2026-04-12", "best_val_checkpoint.pth")
SIM3_REPL_W   = os.path.join(ROOT, "output", "replica",
                              "2026-04-22", "best_val_checkpoint.pth")
OUT_DIR       = os.path.join(ROOT, "results", "eval_replica")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, PLUECKERNET)
sys.path.insert(0, ROOT)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

METHOD_NAMES  = ["SE3-PlueckerNet", "Sim3-Net (synth)", "Sim3-Net (Replica)", "Pure-Sim3-RANSAC"]
METHOD_COLORS = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c"]   # blue, red, orange, green

# ── reuse all helpers from the original benchmark ──────────────────────────────
# Import after setting sys.path so relative imports inside work.
from eval_benchmark import (
    load_se3_model, load_sim3_model,
    make_cube_lines_md, apply_sim3_md,
    load_chess_frames, build_point_cloud, extract_lines_md,
    md_to_dm,
    _topk_from_prob, net_topk_se3, net_topk_sim3, direction_nn_topk,
    run_se3_net, run_sim3_net, run_pure_ransac,
    compute_metrics, rot_err_deg, trans_err, scale_err_log,
    plot_lines_3d,
)

CHESS_SEQ1 = "/home/rueyday/Downloads/chess/seq-01"
CHESS_SEQ3 = "/home/rueyday/Downloads/chess/seq-03"


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_replica_model():
    from easydict import EasyDict as edict
    from model.model_plucker import PluckerNetKnn
    cfg   = edict(net_nchannel=128, GNN_layers=["self", "cross"] * 6,
                  net_lambda=0.1, net_maxiter=30, net_topK=200)
    model = PluckerNetKnn(cfg).to(DEVICE)
    ckpt  = torch.load(SIM3_REPL_W, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[Replica model] loaded  {SIM3_REPL_W}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 4-method plot helpers
# ═══════════════════════════════════════════════════════════════════════════════

def bar4(ax, vals, ylabel, title):
    x    = np.arange(4)
    bars = ax.bar(x, vals, color=METHOD_COLORS, edgecolor='white', alpha=0.88, width=0.55)
    finite = [v for v in vals if np.isfinite(v)]
    top    = max(finite) if finite else 1.0
    for bar, v in zip(bars, vals):
        if not np.isfinite(v): v_str = '∞'
        elif v < 0.001:        v_str = f'{v:.2e}'
        elif v < 10:           v_str = f'{v:.3f}'
        else:                  v_str = f'{v:.1f}'
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + top * 0.03,
                v_str, ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(["SE3-Net", "Sim3\n(synth)", "Sim3\n(Replica)", "Pure\nRANSAC"],
                        fontsize=7.5)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    if finite:
        ax.set_ylim(0, top * 1.45 + 1e-9)


def timing_bar4(ax, results, title):
    net_t = [r['t_net_ms']    for r in results]
    ran_t = [r['t_ransac_ms'] for r in results]
    x = np.arange(4)
    ax.bar(x, net_t, color=METHOD_COLORS, edgecolor='white', alpha=0.6, width=0.5, label='Network')
    ax.bar(x, ran_t, bottom=net_t, color=METHOD_COLORS, edgecolor='white',
           alpha=0.95, width=0.5, label='RANSAC', hatch='//')
    totals = [n + r for n, r in zip(net_t, ran_t)]
    for xi, v in enumerate(totals):
        ax.text(xi, v + max(totals) * 0.03,
                f'{v:.0f}ms', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(["SE3-Net", "Sim3\n(synth)", "Sim3\n(Replica)", "Pure\nRANSAC"],
                        fontsize=7.5)
    ax.set_ylabel("Time (ms)", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [saved] {os.path.relpath(path, ROOT)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment runners — 4 methods
# ═══════════════════════════════════════════════════════════════════════════════

def run4(label, fns, R_gt, t_gt, s_gt):
    results = []
    for name, fn in fns:
        r = fn()
        m = compute_metrics(r, R_gt, t_gt, s_gt)
        results.append({**r, **m})
        print(f"    {name:22s}  rot={m['rot']:6.2f}°  t={m['trans']:.4f}m  "
              f"s_err={m['scale']:.3f}  ic={m['ic']}  total={m['t_tot']:.0f}ms")
    return results


def run_cube_experiment(model_se3, ransac_se3, model_synth, model_replica):
    print("\n" + "═" * 65)
    print("Experiment A — Synthetic Cube Wireframe  (4 methods)")
    print("═" * 65)

    L_src = make_cube_lines_md(n_per_edge=10, cube_scale=1.0)
    print(f"  {len(L_src)} cube lines (3 direction clusters × 40)")

    R_gt = Rotation.from_rotvec(
        np.radians(45) * np.array([0.5, 0.7, 0.4]) / np.linalg.norm([0.5, 0.7, 0.4])
    ).as_matrix()
    t_gt  = np.array([0.50, 0.30, 0.20])
    s_gt  = 1.8
    L_se3  = apply_sim3_md(L_src, 1.0, R_gt, t_gt)
    L_sim3 = apply_sim3_md(L_src, s_gt, R_gt, t_gt)

    # Figure r01: cube line visualization (unchanged from original)
    fig = plt.figure(figsize=(15, 5))
    for col_i, (L, title, clr) in enumerate(zip(
        [L_src, L_se3, L_sim3],
        ["Source lines", f"SE3 transform\n(R=45°, t=[0.5,0.3,0.2])",
         f"Sim3 transform\n(R,t + scale={s_gt})"],
        ["steelblue", "tomato", "darkorange"]
    )):
        ax = fig.add_subplot(1, 3, col_i + 1, projection="3d")
        plot_lines_3d(ax, L, color=clr, n=80, half=0.25)
        ax.set_title(title, fontsize=10)
        ax.view_init(elev=25, azim=40)
    fig.suptitle("Synthetic Cube Wireframe — Plücker Line Sets", fontsize=13, fontweight='bold')
    plt.tight_layout()
    savefig(fig, "fig_eval_r01_cube_lines.png")

    # A1 — SE3 transform
    print("\n  A1  SE3 transform  (s=1.0)")
    res_a1 = run4("A1", [
        ("SE3-Net",          lambda: run_se3_net(model_se3, ransac_se3, L_src, L_se3,  topk=200, threshold=0.3)),
        ("Sim3-Net (synth)", lambda: run_sim3_net(model_synth,          L_src, L_se3,  topk=100, threshold=0.1)),
        ("Sim3-Net (Repl.)", lambda: run_sim3_net(model_replica,        L_src, L_se3,  topk=100, threshold=0.1)),
        ("Pure-RANSAC",      lambda: run_pure_ransac(                   L_src, L_se3,  topk=100, threshold=0.1)),
    ], R_gt, t_gt, 1.0)

    # A2 — Sim3 transform
    print(f"\n  A2  Sim3 transform  (s={s_gt})")
    res_a2 = run4("A2", [
        ("SE3-Net",          lambda: run_se3_net(model_se3, ransac_se3, L_src, L_sim3, topk=200, threshold=0.3)),
        ("Sim3-Net (synth)", lambda: run_sim3_net(model_synth,          L_src, L_sim3, topk=100, threshold=0.1)),
        ("Sim3-Net (Repl.)", lambda: run_sim3_net(model_replica,        L_src, L_sim3, topk=100, threshold=0.1)),
        ("Pure-RANSAC",      lambda: run_pure_ransac(                   L_src, L_sim3, topk=100, threshold=0.1)),
    ], R_gt, t_gt, s_gt)

    # Figure r02: A1 bars
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    bar4(axes[0], [r['rot']   for r in res_a1], "Rotation error (°)", "A1 (SE3): Rotation error")
    bar4(axes[1], [r['trans'] for r in res_a1], "Translation error (m)", "A1 (SE3): Translation error")
    bar4(axes[2], [r['scale'] for r in res_a1], "Scale error |log(ŝ/s)|", "A1 (SE3): Scale error\n(GT s=1.0)")
    bar4(axes[3], [r['ic']    for r in res_a1], "RANSAC inliers", "A1 (SE3): Inlier count")
    timing_bar4(axes[4], res_a1, "A1 (SE3): Compute time")
    axes[2].set_ylim(0, 0.5)
    fig.suptitle("Experiment A1 — Cube Wireframe, SE3 Transform\n"
                 "4 methods: SE3-Net / Sim3-Net (synth) / Sim3-Net (Replica) / Pure-RANSAC",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    savefig(fig, "fig_eval_r02_cube_se3.png")

    # Figure r03: A2 bars
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    bar4(axes[0], [r['rot']   for r in res_a2], "Rotation error (°)", "A2 (Sim3): Rotation error")
    bar4(axes[1], [r['trans'] for r in res_a2], "Translation error (m)", "A2 (Sim3): Translation error")
    bar4(axes[2], [r['scale'] for r in res_a2], "Scale error |log(ŝ/s)|",
         f"A2 (Sim3): Scale error\n(GT s={s_gt}; SE3-Net always s=1)")
    bar4(axes[3], [r['ic']    for r in res_a2], "RANSAC inliers", "A2 (Sim3): Inlier count")
    timing_bar4(axes[4], res_a2, "A2 (Sim3): Compute time")
    fig.suptitle(f"Experiment A2 — Cube Wireframe, Sim3 Transform (scale={s_gt})\n"
                 "SE3-Net fails on translation+scale; Sim3 nets and Pure-RANSAC recover all 3",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    savefig(fig, "fig_eval_r03_cube_sim3.png")

    return res_a1, res_a2


def run_chess_experiment(model_se3, ransac_se3, model_synth, model_replica):
    print("\n" + "═" * 65)
    print("Experiment B — Chess 7-Scenes cross-sequence  (4 methods)")
    print("═" * 65)

    print("  Loading seq-01 ...")
    frames1 = load_chess_frames(CHESS_SEQ1, n_frames=25, frame_step=40)
    cloud1  = build_point_cloud(frames1)
    L1      = extract_lines_md(cloud1, n_lines=300)
    print(f"  {cloud1.shape[0]:,} pts → {len(L1)} lines")

    print("  Loading seq-03 ...")
    frames3 = load_chess_frames(CHESS_SEQ3, n_frames=25, frame_step=40)
    cloud3  = build_point_cloud(frames3)
    L3      = extract_lines_md(cloud3, n_lines=300)
    print(f"  {cloud3.shape[0]:,} pts → {len(L3)} lines")

    R_gt = np.eye(3, dtype=np.float32)
    t_gt = np.zeros(3, dtype=np.float32)
    s_rgb_sim = 1.8

    L3_scaled = L3.copy()
    L3_scaled[:, :3] *= s_rgb_sim   # simulate monocular scale ambiguity

    # B1 — RGBD
    print("\n  B1  RGBD (GT s=1.0)")
    res_b1 = run4("B1", [
        ("SE3-Net",          lambda: run_se3_net(model_se3, ransac_se3, L1, L3,        topk=200, threshold=0.5)),
        ("Sim3-Net (synth)", lambda: run_sim3_net(model_synth,          L1, L3,        topk=100, threshold=0.15)),
        ("Sim3-Net (Repl.)", lambda: run_sim3_net(model_replica,        L1, L3,        topk=100, threshold=0.15)),
        ("Pure-RANSAC",      lambda: run_pure_ransac(                   L1, L3,        topk=150, threshold=0.15)),
    ], R_gt, t_gt, 1.0)

    # B2 — RGB-only
    print(f"\n  B2  RGB-only, moments×{s_rgb_sim} (GT s={s_rgb_sim})")
    res_b2 = run4("B2", [
        ("SE3-Net",          lambda: run_se3_net(model_se3, ransac_se3, L1, L3_scaled, topk=200, threshold=0.5)),
        ("Sim3-Net (synth)", lambda: run_sim3_net(model_synth,          L1, L3_scaled, topk=100, threshold=0.15)),
        ("Sim3-Net (Repl.)", lambda: run_sim3_net(model_replica,        L1, L3_scaled, topk=100, threshold=0.15)),
        ("Pure-RANSAC",      lambda: run_pure_ransac(                   L1, L3_scaled, topk=150, threshold=0.15)),
    ], R_gt, t_gt, s_rgb_sim)

    # Figure r04: B1 bars
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    bar4(axes[0], [r['rot']   for r in res_b1], "Rotation error (°)", "B1 (RGBD): Rotation error")
    bar4(axes[1], [r['trans'] for r in res_b1], "Translation error (m)", "B1 (RGBD): Translation error")
    bar4(axes[2], [r['scale'] for r in res_b1], "Scale error |log(ŝ/s)|", "B1 (RGBD): Scale error\n(GT s=1.0)")
    bar4(axes[3], [r['ic']    for r in res_b1], "RANSAC inliers", "B1 (RGBD): Inlier count")
    timing_bar4(axes[4], res_b1, "B1 (RGBD): Compute time")
    fig.suptitle("Experiment B1 — Chess Cross-Sequence RGBD (seq-01 ↔ seq-03)\n"
                 "GT: R=I, t=0, s=1 — does Replica training improve real-data performance?",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    savefig(fig, "fig_eval_r04_chess_rgbd.png")

    # Figure r05: B2 bars
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    bar4(axes[0], [r['rot']   for r in res_b2], "Rotation error (°)", "B2 (RGB-only): Rotation error")
    bar4(axes[1], [r['trans'] for r in res_b2], "Translation error (m)", "B2 (RGB-only): Translation error")
    bar4(axes[2], [r['scale'] for r in res_b2], "Scale error |log(ŝ/s)|",
         f"B2 (RGB-only): Scale error\n(GT s={s_rgb_sim}; SE3-Net always s=1)")
    bar4(axes[3], [r['ic']    for r in res_b2], "RANSAC inliers", "B2 (RGB-only): Inlier count")
    timing_bar4(axes[4], res_b2, "B2 (RGB-only): Compute time")
    fig.suptitle(f"Experiment B2 — Chess Cross-Sequence RGB-only (scale ambiguous)\n"
                 f"seq-03 moments ×{s_rgb_sim} simulates monocular → GT s={s_rgb_sim}",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    savefig(fig, "fig_eval_r05_chess_rgb_scale.png")

    return res_b1, res_b2


def make_summary_figure(res_a1, res_a2, res_b1, res_b2):
    experiments = ["A1 Cube SE3", "A2 Cube Sim3", "B1 Chess RGBD", "B2 Chess RGB+scale"]
    metrics = {
        'rot':   [[r['rot']   for r in res] for res in [res_a1, res_a2, res_b1, res_b2]],
        'trans': [[r['trans'] for r in res] for res in [res_a1, res_a2, res_b1, res_b2]],
        'scale': [[r['scale'] for r in res] for res in [res_a1, res_a2, res_b1, res_b2]],
        'time':  [[r['t_tot'] for r in res] for res in [res_a1, res_a2, res_b1, res_b2]],
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    def grouped_bar(ax, data_per_exp, ylabel, title, ylim=None):
        n_exp, n_meth = len(experiments), 4
        w, x = 0.18, np.arange(n_exp)
        for mi in range(n_meth):
            vals      = [data_per_exp[ei][mi] for ei in range(n_exp)]
            vals_plot = [min(v, 180) if np.isfinite(v) else 180 for v in vals]
            ax.bar(x + (mi - 1.5) * w, vals_plot, width=w,
                   color=METHOD_COLORS[mi], alpha=0.85, edgecolor='white',
                   label=METHOD_NAMES[mi])
        ax.set_xticks(x)
        ax.set_xticklabels(experiments, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        if ylim:
            ax.set_ylim(0, ylim)
        ax.legend(fontsize=7.5)
        ax.grid(axis='y', alpha=0.3)

    grouped_bar(axes[0], metrics['rot'],   "Rotation error (°)",      "Rotation Error",    ylim=90)
    grouped_bar(axes[1], metrics['trans'], "Translation error (m)",   "Translation Error",  ylim=3)
    grouped_bar(axes[2], metrics['scale'], "Scale error |log(ŝ/s)|",  "Scale Error (log-ratio)")
    grouped_bar(axes[3], metrics['time'],  "Total compute time (ms)", "Compute Time")

    axes[2].axhline(0.1, color='gray', ls='--', lw=1, label='acceptable threshold')
    axes[2].legend(fontsize=7.5)

    fig.suptitle(
        "Summary: SE3-Net / Sim3-Net (synthetic) / Sim3-Net (Replica) / Pure-RANSAC\n"
        "across 4 experiments — key question: does Replica training close the real-data gap?",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    savefig(fig, "fig_eval_r06_summary.png")


def print_table(res_a1, res_a2, res_b1, res_b2):
    COLS = ["rot(°)", "t(m)", "s_err", "inliers", "net(ms)", "ran(ms)", "tot(ms)"]
    EXPS = [
        ("A1 Cube SE3   (s=1.0)", res_a1),
        ("A2 Cube Sim3  (s=1.8)", res_a2),
        ("B1 Chess RGBD (s=1.0)", res_b1),
        ("B2 Chess RGB  (s=1.8)", res_b2),
    ]

    hdr = f"{'Experiment':<26} {'Method':<24}" + "".join(f"{c:>10}" for c in COLS)
    print("\n" + "═" * len(hdr))
    print("RESULTS TABLE — 4 METHODS")
    print("═" * len(hdr))
    print(hdr)
    print("─" * len(hdr))

    for exp_name, results in EXPS:
        for mi, (meth, r) in enumerate(zip(METHOD_NAMES, results)):
            label = exp_name if mi == 0 else ""
            s_str = f"{r['scale']:.3f}" if np.isfinite(r['scale']) else "   ∞   "
            row   = f"{label:<26} {meth:<24}{r['rot']:>10.2f}{r['trans']:>10.4f}{s_str:>10}"
            row  += f"{r['ic']:>10d}{r['t_net_ms']:>10.0f}{r['t_ransac_ms']:>10.0f}{r['t_total_ms']:>10.0f}"
            print(row)
        print("─" * len(hdr))


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"Device: {DEVICE}")
    print(f"Output: {OUT_DIR}")

    model_se3, ransac_se3 = load_se3_model()
    model_synth           = load_sim3_model()   # synthetic-trained
    model_replica         = load_replica_model()

    res_a1, res_a2 = run_cube_experiment(model_se3, ransac_se3, model_synth, model_replica)
    res_b1, res_b2 = run_chess_experiment(model_se3, ransac_se3, model_synth, model_replica)

    make_summary_figure(res_a1, res_a2, res_b1, res_b2)
    print_table(res_a1, res_a2, res_b1, res_b2)

    print("\n" + "═" * 65)
    print(f"All figures saved to {os.path.relpath(OUT_DIR, ROOT)}/")
    print("═" * 65)


if __name__ == "__main__":
    main()
