#!/usr/bin/env python3
"""
Generate kitti_train / kitti_valid datasets for ScalePluckerNet.

Scenario: register a mono-SLAM submap (small spatial chunk) against the
full metric LiDAR line cloud of a KITTI sequence.

  plucker1  — metric reference: background lines (full route) + chunk lines,
               centred on chunk so moments are manageable (±RADIUS m)
  plucker2  — mono query: SIM(3)-inverse of chunk lines + SLAM noise + outliers
  R_gt / t_gt / s_gt  — SIM(3) mapping plucker2 (mono) → plucker1 (metric)

Key difference from 7-Scenes:
  - Reference is MUCH larger than the inlier set (full route vs. chunk)
  - Chunk is a spatial sphere with radius 40–150 m from a random centre
  - Coordinate system is centred on the chunk to keep moments ≤ ±RADIUS

Train sequences : 00 01 02 03 04 05 06 07 08
Val   sequences : 09 10

Usage
-----
  cd ScalePluckerNet
  python scripts/generate_kitti_dataset.py
  python scripts/generate_kitti_dataset.py --n_per_seq 2000 --n_valid 400 --workers 8
"""

import argparse
import pickle
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import msgpack
import numpy as np

_ROOT      = Path(__file__).resolve().parent.parent      # ScalePluckerNet/
_MESH_DIR  = _ROOT.parent / "test_data" / "kitti" / "mesh"

SEQS_TRAIN = ["00", "01", "02", "03", "04", "05", "06", "07", "08"]
SEQS_VALID = ["09", "10"]

# Spatial chunk
CHUNK_RADIUS_MIN =  40.0   # metres
CHUNK_RADIUS_MAX = 150.0   # metres

# Reference size (background lines outside chunk + inlier lines inside chunk)
N_REF_BG_RANGE  = (150, 500)   # background non-inlier lines in plucker1
N_INLIER_RANGE  = (15,  150)   # inlier lines (appear in both ref and query)
N_QRY_MAX       = 250          # max total query lines (inliers + outliers)
MIN_CHUNK_LINES = 30           # skip chunk if too few lines in radius

# SIM(3) scale (mono → metric)
_SCALE_LOG_MU   = np.log(2.5)
_SCALE_LOG_STD  = 1.0
_SCALE_CLIP     = (0.3, 8.0)

_MIN_MESH = 500   # skip sequence if fewer than this many mesh lines


# ── Plücker helpers ───────────────────────────────────────────────────────────

def _shift_origin(L: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Shift Plücker lines [m, d] by translating world origin by vector c.
    New moment: m' = m - c × d
    """
    out = L.copy()
    out[:, :3] -= np.cross(c[None], L[:, 3:])
    return out


def _random_rotation() -> np.ndarray:
    A = np.random.randn(3, 3).astype(np.float64)
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float32)


def _apply_sim3(L: np.ndarray, s: float,
                R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """SIM(3)(s, R, t) on Plücker lines [m, d]."""
    m, d   = L[:, :3].copy(), L[:, 3:].copy()
    d_out  = (R @ d.T).T
    m_out  = s * (R @ m.T).T + np.cross(t[None], d_out)
    return np.concatenate([m_out, d_out], axis=1).astype(np.float32)


def _make_random_outliers(n: int, pos_range: float = 3.0) -> np.ndarray:
    if n == 0:
        return np.zeros((0, 6), np.float32)
    d = np.random.randn(n, 3).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    p = np.random.uniform(-pos_range, pos_range, (n, 3)).astype(np.float32)
    m = np.cross(p, d)
    return np.concatenate([m, d], axis=1)


# ── Mesh DB loader ────────────────────────────────────────────────────────────

def load_mesh_lines(db_path: Path):
    """Load mesh.db → (N,6) Plücker [m,d] and (N,3) midpoints."""
    with open(db_path, "rb") as f:
        data = msgpack.unpack(f, raw=False)
    lines, midpoints = [], []
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
        d   = diff / ln
        mid = (p1 + p2) * 0.5
        m   = np.cross(mid, d)
        lines.append(np.concatenate([m, d]).astype(np.float32))
        midpoints.append(mid.astype(np.float32))
    if not lines:
        return np.zeros((0, 6), np.float32), np.zeros((0, 3), np.float32)
    return np.array(lines, np.float32), np.array(midpoints, np.float32)


# ── Pair generator ────────────────────────────────────────────────────────────

def _generate_one(lines: np.ndarray, midpoints: np.ndarray,
                  rng_seed: int) -> dict | None:
    """Generate one submap-to-fullmap pair from a KITTI sequence."""
    np.random.seed(rng_seed)
    N = len(lines)
    if N < _MIN_MESH:
        return None

    # ── Pick a random chunk centre and radius ─────────────────────────────────
    radius = np.random.uniform(CHUNK_RADIUS_MIN, CHUNK_RADIUS_MAX)
    centre_idx = np.random.randint(N)
    centre = midpoints[centre_idx].copy()   # (3,) in world frame

    # Find lines inside the chunk
    dists = np.linalg.norm(midpoints - centre[None], axis=1)
    chunk_mask = dists <= radius
    n_chunk = int(chunk_mask.sum())
    if n_chunk < MIN_CHUNK_LINES:
        return None

    chunk_idx = np.where(chunk_mask)[0]
    bg_idx    = np.where(~chunk_mask)[0]

    # ── Shift all lines to chunk-local coordinates ────────────────────────────
    # This keeps moments ≤ ±RADIUS metres (manageable for the network)
    lines_local = _shift_origin(lines, centre)

    # ── SIM(3) parameters ─────────────────────────────────────────────────────
    s = float(np.clip(np.exp(np.random.normal(_SCALE_LOG_MU, _SCALE_LOG_STD)),
                      *_SCALE_CLIP))
    R = _random_rotation()
    t_range = 0.4 * s
    t = np.random.uniform(-t_range, t_range, 3).astype(np.float32)

    s_inv = 1.0 / s
    R_inv = R.T
    t_inv = (-s_inv * (R_inv @ t)).astype(np.float32)

    # ── Inlier selection from chunk ───────────────────────────────────────────
    n_inlier = int(np.random.randint(N_INLIER_RANGE[0],
                                     min(N_INLIER_RANGE[1], n_chunk) + 1))
    n_inlier = max(n_inlier, N_INLIER_RANGE[0])
    if n_chunk < n_inlier:
        return None

    inlier_chunk_pos = np.random.choice(n_chunk, n_inlier, replace=False)
    inlier_idx = chunk_idx[inlier_chunk_pos]       # indices into full array
    inlier_lines_local = lines_local[inlier_idx]   # (n_inlier, 6), metric local

    # ── Reference: inliers + background ──────────────────────────────────────
    n_bg = int(np.random.randint(*N_REF_BG_RANGE))
    n_bg = min(n_bg, len(bg_idx))
    if n_bg > 0:
        bg_sel = np.random.choice(len(bg_idx), n_bg, replace=False)
        bg_lines_local = lines_local[bg_idx[bg_sel]]  # metric local
    else:
        bg_lines_local = np.zeros((0, 6), np.float32)

    # Combine: inliers first (position in reference 0..n_inlier-1)
    ref_raw = np.concatenate([inlier_lines_local, bg_lines_local], axis=0)

    # Add small metric reference noise (RGBD/LiDAR fitting error)
    ref_m_noise = np.random.uniform(0.05, 0.30)   # larger for outdoor (LiDAR)
    ref_d_noise = np.random.uniform(0.005, 0.04)
    n_ref = len(ref_raw)
    ref_raw[:, :3] += np.random.normal(0, ref_m_noise, (n_ref, 3)).astype(np.float32)
    d_noisy = ref_raw[:, 3:] + np.random.normal(0, ref_d_noise, (n_ref, 3)).astype(np.float32)
    norms = np.linalg.norm(d_noisy, axis=1, keepdims=True)
    ref_raw[:, 3:] = d_noisy / np.where(norms > 0, norms, 1.0)

    # ── Query: apply inverse SIM(3) to inliers (clean, before ref noise) ─────
    mono_inlier = _apply_sim3(inlier_lines_local, s_inv, R_inv, t_inv)

    # SLAM noise (larger than RGBD — monocular drift in outdoor environments)
    m_noise_std = np.random.uniform(0.05, 0.40)
    d_noise_std = np.random.uniform(0.02, 0.12)
    mono_inlier[:, :3] += np.random.normal(0, m_noise_std,
                                            (n_inlier, 3)).astype(np.float32)
    d_q = mono_inlier[:, 3:] + np.random.normal(0, d_noise_std,
                                                  (n_inlier, 3)).astype(np.float32)
    norms_q = np.linalg.norm(d_q, axis=1, keepdims=True)
    mono_inlier[:, 3:] = d_q / np.where(norms_q > 0, norms_q, 1.0)

    # Outlier query lines (SLAM picks up lines outside the shared chunk)
    n_qry_total = int(np.random.randint(max(n_inlier, 50),
                                        max(n_inlier + 1, N_QRY_MAX)))
    n_outlier = n_qry_total - n_inlier
    outlier_range = max(2.0, radius * s_inv * 0.8)
    outlier_mono = _make_random_outliers(n_outlier, pos_range=outlier_range)

    # ── Combine and shuffle ───────────────────────────────────────────────────
    qry_raw  = np.concatenate([mono_inlier, outlier_mono], axis=0)
    perm_qry = np.random.permutation(n_qry_total)
    plucker2 = qry_raw[perm_qry]

    perm_ref = np.random.permutation(n_ref)
    plucker1 = ref_raw[perm_ref]

    # ── Build matches (2, n_inlier) ───────────────────────────────────────────
    # Inliers are at raw positions [0, n_inlier) in ref_raw and qry_raw
    ref_pos_map = np.argsort(perm_ref)           # ref_pos_map[j] = new pos of raw j
    inlier_mask = perm_qry < n_inlier
    qry_new_pos = np.where(inlier_mask)[0]
    orig_inlier = perm_qry[qry_new_pos]          # raw inlier indices in qry_raw
    ref_new_pos = ref_pos_map[orig_inlier]        # matches in plucker1 (raw pos 0..n_inlier-1)
    matches = np.stack([ref_new_pos, qry_new_pos]).astype(np.int32)

    return {
        "plucker1": plucker1,
        "plucker2": plucker2,
        "matches":  matches,
        "R_gt":     R.astype(np.float32),
        "t_gt":     t.reshape(3, 1).astype(np.float32),
        "s_gt":     np.float32(s),
    }


# ── Worker ────────────────────────────────────────────────────────────────────

_WORKER_DATA = None   # (lines, midpoints) by seq, filled by initializer


def _worker_init(data_by_seq):
    global _WORKER_DATA
    _WORKER_DATA = data_by_seq


def _worker_generate(args):
    seq, seed = args
    lines, midpoints = _WORKER_DATA[seq]
    return seq, _generate_one(lines, midpoints, seed)


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
    ap.add_argument("--n_per_seq", type=int, default=2000,
                    help="Training pairs per sequence (default: 2000)")
    ap.add_argument("--n_valid",   type=int, default=400,
                    help="Validation pairs per val sequence (default: 400)")
    ap.add_argument("--workers",   type=int, default=4)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--out_dir",   default=str(_ROOT / "dataset"))
    args = ap.parse_args()

    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)

    # ── Load all mesh.db files ────────────────────────────────────────────────
    data_by_seq = {}
    for seq in SEQS_TRAIN + SEQS_VALID:
        db = _MESH_DIR / f"{seq}_mesh.db"
        if not db.exists():
            print(f"[skip] seq {seq}: mesh.db not found at {db}")
            continue
        lines, midpoints = load_mesh_lines(db)
        if len(lines) < _MIN_MESH:
            print(f"[skip] seq {seq}: only {len(lines)} lines")
            continue
        data_by_seq[seq] = (lines, midpoints)
        print(f"  seq {seq}: {len(lines):,} lines loaded")

    # ── Build task lists ──────────────────────────────────────────────────────
    base = args.seed
    train_tasks, valid_tasks = [], []

    for seq in SEQS_TRAIN:
        if seq not in data_by_seq:
            continue
        for i in range(args.n_per_seq):
            train_tasks.append((seq, base + len(train_tasks) * 7))

    for seq in SEQS_VALID:
        if seq not in data_by_seq:
            continue
        for i in range(args.n_valid):
            valid_tasks.append((seq, base + 10_000_000 + len(valid_tasks) * 7))

    print(f"\nGenerating {len(train_tasks)} train + {len(valid_tasks)} val pairs "
          f"({args.workers} workers)…")

    def run_tasks(tasks):
        results = []
        if args.workers <= 1:
            _worker_init(data_by_seq)
            for t in tasks:
                _, pair = _worker_generate(t)
                if pair is not None:
                    results.append(pair)
        else:
            with ProcessPoolExecutor(max_workers=args.workers,
                                     initializer=_worker_init,
                                     initargs=(data_by_seq,)) as ex:
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

    save_dataset(train_pairs, out_dir / "kitti_train")
    save_dataset(valid_pairs, out_dir / "kitti_valid")

    # Print stats
    all_s   = [float(p["s_gt"]) for p in train_pairs]
    all_ir  = [p["matches"].shape[1] / p["plucker1"].shape[0] for p in train_pairs]
    all_ref = [p["plucker1"].shape[0] for p in train_pairs]
    all_qry = [p["plucker2"].shape[0] for p in train_pairs]
    print(f"\nTrain stats:")
    print(f"  scale   : median={np.median(all_s):.2f}  P25={np.percentile(all_s,25):.2f}  P75={np.percentile(all_s,75):.2f}")
    print(f"  inlier% : median={np.median(all_ir)*100:.1f}%  P25={np.percentile(all_ir,25)*100:.1f}%  P75={np.percentile(all_ir,75)*100:.1f}%")
    print(f"  n_ref   : median={np.median(all_ref):.0f}  n_qry: median={np.median(all_qry):.0f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
