#!/usr/bin/env python3
"""
Generate 7scenes_train and 7scenes_valid datasets from mesh.db line clouds.

For each 7-Scenes scene, loads the 3-D mesh line cloud (metric world frame)
and generates synthetic SIM(3) pairs that match real mono→metric SLAM statistics:

  plucker1  — subsampled metric reference lines (from mesh, world frame)
  plucker2  — mono-style query lines (inverse SIM3 + SLAM noise + outliers)
  R_gt / t_gt / s_gt  — SIM3 that maps plucker2 (mono) → plucker1 (metric)

Split:
  train: chess, fire, office, pumpkin, redkitchen, stairs  (N_PER_SCENE pairs each)
  valid: heads + 20 % of each train scene (N_VALID_SCENE pairs each)

Scale distribution mirrors real mono→metric pairs:
  s ~ LogNormal(log(2.5), 1.0) clipped to [0.3, 8.0]
  (metric map is ~2.5× larger than mono map on average)

Usage
-----
  cd ScalePluckerNet
  python scripts/generate_7scenes_dataset.py
  python scripts/generate_7scenes_dataset.py --n_per_scene 3000 --n_valid 600 --workers 8
"""

import argparse
import os
import sys
import pickle
import time
import numpy as np
import msgpack
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

_ROOT     = Path(__file__).resolve().parent.parent          # ScalePluckerNet/
_TESTDATA = _ROOT.parent / "test_data" / "7scenes"

SCENES_TRAIN = ["chess", "fire", "office", "pumpkin", "redkitchen", "stairs"]
SCENES_VALID  = ["chess", "fire", "office", "pumpkin", "redkitchen", "stairs"]

# SIM(3) scale distribution (mono→metric)
_SCALE_LOG_MU  = np.log(2.5)
_SCALE_LOG_STD = 1.0
_SCALE_CLIP    = (0.3, 8.0)

_MIN_INLIERS = 20
_MIN_MESH    = 80      # skip scenes with fewer mesh lines


# ── Plücker helpers ───────────────────────────────────────────────────────────

def _random_rotation() -> np.ndarray:
    A = np.random.randn(3, 3).astype(np.float64)
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float32)


def _apply_sim3(L: np.ndarray, s: float,
                R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Transform Plücker lines [m, d] by SIM(3)(s, R, t).

    L_out[m] = s * R @ m + t × (R @ d)
    L_out[d] = R @ d
    """
    m, d  = L[:, :3].copy(), L[:, 3:].copy()
    d_out = (R @ d.T).T
    m_out = s * (R @ m.T).T + np.cross(t[None], d_out)
    return np.concatenate([m_out, d_out], axis=1).astype(np.float32)


def _normalize_directions(L: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(L[:, 3:], axis=1, keepdims=True)
    out = L.copy()
    out[:, 3:] /= np.where(norms > 0, norms, 1.0)
    return out


def _make_random_outliers(n: int, pos_range: float = 1.5) -> np.ndarray:
    """Random Plücker lines in [m, d] form (for the mono/query frame)."""
    if n == 0:
        return np.zeros((0, 6), np.float32)
    d = np.random.randn(n, 3).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    p = np.random.uniform(-pos_range, pos_range, (n, 3)).astype(np.float32)
    m = np.cross(p, d)
    return np.concatenate([m, d], axis=1)


# ── Mesh DB loader ────────────────────────────────────────────────────────────

def load_mesh_lines(db_path: Path) -> np.ndarray:
    """Load mesh.db → (N, 6) float32 Plücker lines [m, d] in world frame."""
    with open(db_path, "rb") as f:
        data = msgpack.unpack(f, raw=False)
    lines = []
    for lm in data.get("landmarks_line", {}).values():
        pw = lm.get("pos_w") or lm.get("pos")
        if pw is None or len(pw) < 6:
            continue
        p1 = np.array(pw[:3], np.float64)
        p2 = np.array(pw[3:6], np.float64)
        diff = p2 - p1
        ln = float(np.linalg.norm(diff))
        if ln < 0.01:
            continue
        d = diff / ln
        m = np.cross((p1 + p2) * 0.5, d)
        lines.append(np.concatenate([m, d]).astype(np.float32))
    return np.array(lines, np.float32) if lines else np.zeros((0, 6), np.float32)


# ── Pair generator ────────────────────────────────────────────────────────────

def _generate_one(mesh_lines: np.ndarray, rng_seed: int) -> dict | None:
    """Generate one (reference, query, GT-SIM3) pair from a mesh line pool."""
    np.random.seed(rng_seed)
    N = len(mesh_lines)
    if N < _MIN_MESH:
        return None

    # ── SIM(3) parameters ────────────────────────────────────────────────────
    s = float(np.clip(np.exp(np.random.normal(_SCALE_LOG_MU, _SCALE_LOG_STD)),
                      *_SCALE_CLIP))
    R = _random_rotation()
    t_range = 0.4 * s
    t = np.random.uniform(-t_range, t_range, 3).astype(np.float32)

    # Inverse SIM(3): mono → metric means L_metric = SIM3(s,R,t)(L_mono)
    # So L_mono = SIM3(1/s, R^T, -(1/s)*R^T*t)(L_metric)
    s_inv = 1.0 / s
    R_inv = R.T
    t_inv = (-s_inv * (R_inv @ t)).astype(np.float32)

    # ── Reference: subsample metric lines from mesh ───────────────────────────
    # Size range mirrors real RGBD SLAM map line counts (50–300 lines)
    n_ref = int(np.random.randint(50, min(300, N + 1)))
    idx_ref = np.random.choice(N, n_ref, replace=False)
    plucker1 = mesh_lines[idx_ref].copy()     # (n_ref, 6) metric

    # ── RGBD SLAM noise on reference (RGBD map also has line fitting noise) ───
    ref_m_noise = np.random.uniform(0.01, 0.08)
    ref_d_noise = np.random.uniform(0.005, 0.04)
    plucker1[:, :3] += np.random.normal(0, ref_m_noise, (n_ref, 3)).astype(np.float32)
    d_ref_noisy = plucker1[:, 3:] + np.random.normal(0, ref_d_noise, (n_ref, 3)).astype(np.float32)
    norms_ref = np.linalg.norm(d_ref_noisy, axis=1, keepdims=True)
    plucker1[:, 3:] = d_ref_noisy / np.where(norms_ref > 0, norms_ref, 1.0)

    # ── Query: apply inverse SIM(3) → mono frame ────────────────────────────
    # Apply SIM(3) to the clean mesh lines (before ref noise) for correct GT
    mono_full = _apply_sim3(mesh_lines[idx_ref], s_inv, R_inv, t_inv)

    # ── Inlier subset (not all mono lines are "visible" in a short SLAM run) ─
    # Real mono SLAM maps see 5–25 % of the scene mesh; 50 % was too easy.
    inlier_frac = np.random.uniform(0.05, 0.25)
    n_inlier    = max(_MIN_INLIERS, int(n_ref * inlier_frac))
    n_inlier    = min(n_inlier, n_ref)

    inlier_src_idx = np.random.choice(n_ref, n_inlier, replace=False)
    inlier_mono    = mono_full[inlier_src_idx].copy()

    # ── Mono SLAM noise (larger than RGBD — monocular has more drift/fitting error)
    m_noise_std = np.random.uniform(0.05, 0.30)
    d_noise_std = np.random.uniform(0.02, 0.12)
    inlier_mono[:, :3] += np.random.normal(0, m_noise_std, (n_inlier, 3)).astype(np.float32)
    d_noisy = inlier_mono[:, 3:] + np.random.normal(0, d_noise_std, (n_inlier, 3)).astype(np.float32)
    norms = np.linalg.norm(d_noisy, axis=1, keepdims=True)
    inlier_mono[:, 3:] = d_noisy / np.where(norms > 0, norms, 1.0)

    # ── Outlier query lines (scene content not shared with reference) ─────────
    n_qry_total  = int(np.random.randint(max(n_inlier, 50), max(n_inlier + 1, 250)))
    n_outlier    = n_qry_total - n_inlier
    outlier_mono = _make_random_outliers(n_outlier, pos_range=max(0.5, 1.5 * s_inv))

    # ── Combine and shuffle query ─────────────────────────────────────────────
    plucker2_raw = np.concatenate([inlier_mono, outlier_mono], axis=0)  # inliers first
    perm_qry     = np.random.permutation(n_qry_total)
    plucker2     = plucker2_raw[perm_qry]

    # ── Shuffle reference ─────────────────────────────────────────────────────
    perm_ref     = np.random.permutation(n_ref)
    plucker1     = plucker1[perm_ref]

    # ── Build matches (2, n_inlier) ───────────────────────────────────────────
    # After perm_qry, position k in plucker2 came from plucker2_raw[perm_qry[k]].
    ref_pos_map  = np.argsort(perm_ref)         # ref_pos_map[j] = new pos of original j
    # Inliers are at raw positions [0, n_inlier); raw position j maps to inlier_src_idx[j].
    inlier_mask  = perm_qry < n_inlier          # which new positions are inliers?
    qry_new_pos  = np.where(inlier_mask)[0]     # (n_inlier,) new positions in plucker2
    orig_inlier  = perm_qry[qry_new_pos]        # (n_inlier,) raw inlier indices [0, n_inlier)
    ref_orig_idx = inlier_src_idx[orig_inlier]  # (n_inlier,) original plucker1 indices
    ref_new_pos  = ref_pos_map[ref_orig_idx]    # (n_inlier,) new positions in plucker1
    matches = np.stack([ref_new_pos, qry_new_pos]).astype(np.int32)  # (2, n_inlier)

    return {
        "plucker1": plucker1,
        "plucker2": plucker2,
        "matches":  matches,
        "R_gt":     R.astype(np.float32),
        "t_gt":     t.reshape(3, 1).astype(np.float32),
        "s_gt":     np.float32(s),
    }


# ── Worker (top-level so it can be pickled by ProcessPoolExecutor) ────────────

_WORKER_MESH = None   # module-level cache filled by initializer


def _worker_init(mesh_by_scene):
    global _WORKER_MESH
    _WORKER_MESH = mesh_by_scene


def _worker_generate(args):
    scene_name, seed = args
    return scene_name, _generate_one(_WORKER_MESH[scene_name], seed)


# ── Dataset saver ─────────────────────────────────────────────────────────────

def save_dataset(pairs: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = ["matches", "plucker1", "plucker2", "R_gt", "t_gt", "s_gt"]
    data = {k: [] for k in keys}
    for p in pairs:
        for k in keys:
            data[k].append(p[k])
    for k in keys:
        with open(out_dir / f"{k}.pkl", "wb") as f:
            pickle.dump(data[k], f, protocol=4)
    print(f"  Saved {len(pairs)} pairs → {out_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_per_scene", type=int, default=3000,
                    help="Training pairs per training scene (default: 3000)")
    ap.add_argument("--n_valid",     type=int, default=500,
                    help="Validation pairs per validation scene (default: 500)")
    ap.add_argument("--workers",     type=int, default=4)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--out_dir",     default=str(_ROOT / "dataset"))
    args = ap.parse_args()

    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)

    # ── Load all mesh.db files ────────────────────────────────────────────────
    all_scenes  = SCENES_TRAIN + SCENES_VALID
    mesh_by_scene = {}
    for scene in all_scenes:
        db = _TESTDATA / "mesh" / f"{scene}_mesh.db"
        if not db.exists():
            print(f"[skip] {scene}: mesh.db not found")
            continue
        lines = load_mesh_lines(db)
        if len(lines) < _MIN_MESH:
            print(f"[skip] {scene}: only {len(lines)} mesh lines")
            continue
        mesh_by_scene[scene] = lines
        print(f"  {scene}: {len(lines)} mesh lines loaded")

    # ── Build task list ───────────────────────────────────────────────────────
    # Train and val use the same scene meshes but disjoint random seeds —
    # every pair is a unique (subsample, SIM3) draw so no pair appears in both splits.
    train_tasks, valid_tasks = [], []

    base_seed = args.seed
    for scene in SCENES_TRAIN:
        if scene not in mesh_by_scene:
            continue
        for i in range(args.n_per_scene):
            train_tasks.append((scene, base_seed + len(train_tasks) * 7))

    for scene in SCENES_VALID:
        if scene not in mesh_by_scene:
            continue
        for i in range(args.n_valid):
            valid_tasks.append((scene, base_seed + 10_000_000 + len(valid_tasks) * 7))

    print(f"\nGenerating {len(train_tasks)} train + {len(valid_tasks)} valid pairs "
          f"({args.workers} workers)...")

    def run_tasks(tasks):
        results = []
        if args.workers <= 1:
            _worker_init(mesh_by_scene)
            for t in tasks:
                _, pair = _worker_generate(t)
                if pair is not None:
                    results.append(pair)
        else:
            with ProcessPoolExecutor(max_workers=args.workers,
                                     initializer=_worker_init,
                                     initargs=(mesh_by_scene,)) as ex:
                futs = {ex.submit(_worker_generate, t): t for t in tasks}
                for fut in as_completed(futs):
                    _, pair = fut.result()
                    if pair is not None:
                        results.append(pair)
        return results

    t0 = time.time()
    train_pairs = run_tasks(train_tasks)
    print(f"  train: {len(train_pairs)} pairs in {time.time()-t0:.1f}s")

    t1 = time.time()
    valid_pairs = run_tasks(valid_tasks)
    print(f"  valid: {len(valid_pairs)} pairs in {time.time()-t1:.1f}s")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_dataset(train_pairs, out_dir / "7scenes_train")
    save_dataset(valid_pairs, out_dir / "7scenes_valid")

    # Print scale stats
    all_s = [float(p["s_gt"]) for p in train_pairs]
    print(f"\nScale stats (train): median={np.median(all_s):.2f}  "
          f"P25={np.percentile(all_s,25):.2f}  P75={np.percentile(all_s,75):.2f}")
    n_inliers = [p["matches"].shape[1] for p in train_pairs]
    n_ref     = [p["plucker1"].shape[0] for p in train_pairs]
    print(f"Inlier ratio (train): median={np.median(np.array(n_inliers)/np.array(n_ref)):.2f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
