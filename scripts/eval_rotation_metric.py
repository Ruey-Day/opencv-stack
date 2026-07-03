"""
Evaluate rotation estimation metrics on GT line matches from 7scenes_valid.

Loads ground-truth matched Plücker pairs and tests several rotation solvers
in isolation, then reports rotation-error statistics.

Methods compared
----------------
  l2_raw       : SVD Procrustes WITHOUT sign alignment (sanity-check baseline)
  l2_aligned   : SVD Procrustes with raw-dot sign alignment (current default)
  l2_robust    : SVD Procrustes with iterative sign alignment (EM-style)
  l1_aligned   : IRLS L1 Procrustes with sign alignment (current default)
  refine_joint : joint linear refinement of (s, R, t) seeded from l2_robust

Convention
----------
  plucker1 = metric reference,  plucker2 = mono query.
  R_gt maps mono → metric:  d_metric = R_gt @ d_mono.
  Solvers called as solve_rotation(d_mono, d_metric) → R ≈ R_gt.

Direction sign info
-------------------
  Plücker lines are undirected: d ≡ -d.
  The raw dot d_mono · d_metric is ≈ d_mono · (R_gt @ d_mono) — positive when
  d_mono is near the R_gt rotation axis, negative for many other directions.
  For random large-angle rotations, ~60% of GT pairs have negative raw dots,
  so raw-dot sign alignment often flips the wrong pairs.  The EM approach
  iteratively re-aligns signs under the current R estimate instead.

Usage
-----
  cd ScalePluckerNet
  python scripts/eval_rotation_metric.py [--dataset 7scenes_valid] [--n 500]
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lib.ransac_grassmannian import (
    solve_rotation_l2,
    solve_rotation_l1,
    solve_translation_scale,
    refine_sim3,
    transform_lines,
    g24_geodesic_distance,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """Geodesic rotation error in degrees."""
    trace = np.trace(R_est.T @ R_gt)
    cos_angle = (trace - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def solve_rotation_l2_raw(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """SVD Procrustes WITHOUT sign alignment — baseline to show the problem."""
    U, _, Vt = np.linalg.svd(d2 @ d1.T)
    det = np.linalg.det(U @ Vt)
    return U @ np.diag([1.0, 1.0, float(det)]) @ Vt


def solve_rotation_l1_raw(
    d1: np.ndarray,
    d2: np.ndarray,
    n_iter: int = 20,
    eps: float = 1e-6,
) -> np.ndarray:
    """IRLS seeded from l2_raw (no pre-alignment), then EM sign updates inside."""
    R = solve_rotation_l2_raw(d1, d2)
    for _ in range(n_iter):
        u = R @ d1
        signs = np.where((u * d2).sum(axis=0) >= 0, 1.0, -1.0)
        u_signed = u * signs
        residuals = np.linalg.norm(u_signed - d2, axis=0)
        w = 1.0 / np.maximum(residuals, eps)
        d1_signed = d1 * signs
        H = (d2 * w) @ d1_signed.T
        U, _, Vt = np.linalg.svd(H)
        det = np.linalg.det(U @ Vt)
        R_new = U @ np.diag([1.0, 1.0, float(det)]) @ Vt
        if np.linalg.norm(R_new - R) < 1e-8:
            break
        R = R_new
    return R


def solve_rotation_l2_robust(
    d1: np.ndarray,
    d2: np.ndarray,
    n_iter: int = 10,
) -> np.ndarray:
    """
    SVD Procrustes with iterative (EM-style) sign alignment.

    Instead of aligning signs using raw d1·d2 (which fails when R is large),
    we use the CURRENT R estimate to predict d2 and align signs accordingly:
        ε_i = sign((R @ d1_i) · d2_i)

    Seed: l2_raw (no pre-alignment) so we don't start from a broken estimate.
    Typically converges in 3–5 iterations.
    """
    R = solve_rotation_l2_raw(d1, d2)   # seed from raw, not from l2_aligned
    for _ in range(n_iter):
        pred = R @ d1          # (3, N) predicted d2 directions
        signs = np.where((pred * d2).sum(axis=0) >= 0, 1.0, -1.0)
        d1_signed = d1 * signs
        H = d2 @ d1_signed.T
        U, _, Vt = np.linalg.svd(H)
        det = np.linalg.det(U @ Vt)
        R_new = U @ np.diag([1.0, 1.0, float(det)]) @ Vt
        if np.linalg.norm(R_new - R, 'fro') < 1e-9:
            break
        R = R_new
    return R


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="7scenes_valid",
                    help="Dataset folder name under dataset/ (default: 7scenes_valid)")
    ap.add_argument("--n", type=int, default=500,
                    help="Max scenes to evaluate (default: 500)")
    ap.add_argument("--min_inliers", type=int, default=5,
                    help="Skip scenes with fewer GT matches (default: 5)")
    args = ap.parse_args()

    data_dir = _ROOT / "dataset" / args.dataset
    if not data_dir.exists():
        print(f"Dataset not found: {data_dir}")
        sys.exit(1)

    def load(name):
        with open(data_dir / f"{name}.pkl", "rb") as f:
            return pickle.load(f, encoding="latin1")

    matches_all = load("matches")
    p1_all      = load("plucker1")
    p2_all      = load("plucker2")
    R_gt_all    = load("R_gt")
    t_gt_all    = load("t_gt")
    s_gt_all    = load("s_gt")

    n_scenes = min(args.n, len(R_gt_all))
    print(f"Dataset : {args.dataset}  ({n_scenes} scenes)")

    methods = ["l2_raw", "l2_aligned", "l2_robust", "l1_aligned", "l1_raw", "refine_joint"]
    errors  = {m: [] for m in methods}
    frac_negdot_raw = []   # raw dots before any R-guided alignment
    frac_negdot_post = []  # dots after l2_robust alignment
    n_skipped = 0

    for i in range(n_scenes):
        m_idx, q_idx = matches_all[i][0], matches_all[i][1]
        n_gt = len(m_idx)
        if n_gt < args.min_inliers:
            n_skipped += 1
            continue

        R_gt = R_gt_all[i].astype(np.float64)
        t_gt = t_gt_all[i].ravel().astype(np.float64)
        s_gt = float(s_gt_all[i])

        # Convention: R_gt maps mono (plucker2) → metric (plucker1).
        # Call solvers as solve_rotation(d_mono, d_metric) → R ≈ R_gt.
        L_mono   = p2_all[i][q_idx].astype(np.float64).T   # (6, n_gt) mono/query
        L_metric = p1_all[i][m_idx].astype(np.float64).T   # (6, n_gt) metric/reference

        d_mono   = L_mono[3:]   / (np.linalg.norm(L_mono[3:],   axis=0, keepdims=True) + 1e-12)
        d_metric = L_metric[3:] / (np.linalg.norm(L_metric[3:], axis=0, keepdims=True) + 1e-12)

        # Raw dots: d_mono · d_metric ≈ d_mono · (R_gt @ d_mono)
        # Negative when d_mono points "away" from R_gt @ d_mono (large-angle rotation).
        dots_raw = (d_mono * d_metric).sum(axis=0)
        frac_negdot_raw.append(float(np.mean(dots_raw < 0)))

        # ── l2_raw: no sign alignment ─────────────────────────────────────────
        R_raw = solve_rotation_l2_raw(d_mono, d_metric)
        errors["l2_raw"].append(rotation_error_deg(R_raw, R_gt))

        # ── l2_aligned: raw-dot sign alignment (current RANSAC default) ───────
        R_l2  = solve_rotation_l2(d_mono, d_metric)
        errors["l2_aligned"].append(rotation_error_deg(R_l2, R_gt))

        # ── l2_robust: iterative EM sign alignment ────────────────────────────
        R_rob = solve_rotation_l2_robust(d_mono, d_metric)
        errors["l2_robust"].append(rotation_error_deg(R_rob, R_gt))

        # Track post-alignment dot fractions (how many pairs the robust solver aligned)
        pred = R_rob @ d_mono
        dots_post = (pred * d_metric).sum(axis=0)
        frac_negdot_post.append(float(np.mean(dots_post < 0)))

        # ── l1_aligned: IRLS seeded from l2_aligned (current RANSAC l1 mode) ──
        R_l1  = solve_rotation_l1(d_mono, d_metric)
        errors["l1_aligned"].append(rotation_error_deg(R_l1, R_gt))

        # ── l1_raw: IRLS seeded from l2_raw (no pre-alignment) ───────────────
        R_l1r = solve_rotation_l1_raw(d_mono, d_metric)
        errors["l1_raw"].append(rotation_error_deg(R_l1r, R_gt))

        # ── refine_joint: joint (R, s, t) solve seeded from l2_robust ─────────
        # Use full Plücker lines (metric = L2 target, mono = L1 source)
        t0, s0 = solve_translation_scale(L_mono, L_metric, R_rob)
        if s0 > 0 and np.isfinite(s0):
            R_ref, _, _ = refine_sim3(L_mono, L_metric, R_rob, t0, s0)
            errors["refine_joint"].append(rotation_error_deg(R_ref, R_gt))
        else:
            errors["refine_joint"].append(errors["l2_robust"][-1])

    # ── Report ────────────────────────────────────────────────────────────────
    N = len(errors["l2_aligned"])
    print(f"Evaluated: {N} scenes  (skipped {n_skipped} with < {args.min_inliers} GT matches)\n")

    print("Direction sign stats (raw = before alignment, post = after l2_robust):")
    print(f"  mean frac antiparallel raw:  {np.mean(frac_negdot_raw):.2f}")
    print(f"  mean frac antiparallel post: {np.mean(frac_negdot_post):.2f}  (residual disagreements)")
    print()

    hdr = f"{'Method':<16}  {'Med [°]':>7}  {'Mean [°]':>8}  {'<1°':>5}  {'<5°':>5}  {'<10°':>6}  {'<45°':>6}"
    print(hdr)
    print("-" * len(hdr))

    for m in methods:
        errs = np.array(errors[m])
        med  = np.median(errs)
        mn   = np.mean(errs)
        r1   = np.mean(errs < 1)
        r5   = np.mean(errs < 5)
        r10  = np.mean(errs < 10)
        r45  = np.mean(errs < 45)
        print(f"{m:<16}  {med:>7.2f}  {mn:>8.2f}  {r1:>5.2f}  {r5:>5.2f}  {r10:>6.2f}  {r45:>6.2f}")

    print()
    print("Per-scene breakdown (first 10 scenes):")
    print(f"  {'i':>4}  {'n_gt':>5}  {'raw_neg':>8}  {'l2raw[°]':>9}  {'l2rob[°]':>9}  {'l1raw[°]':>9}  {'refine[°]':>10}")
    for i in range(min(10, N)):
        print(f"  {i:>4}  {len(matches_all[i][0]):>5}  {frac_negdot_raw[i]:>8.2f}"
              f"  {errors['l2_raw'][i]:>9.2f}"
              f"  {errors['l2_robust'][i]:>9.2f}"
              f"  {errors['l1_raw'][i]:>9.2f}"
              f"  {errors['refine_joint'][i]:>10.2f}")


if __name__ == "__main__":
    main()
