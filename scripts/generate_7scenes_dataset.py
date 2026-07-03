"""
Generate 7scenes_mesh_train and 7scenes_mesh_valid datasets from mesh.db line clouds.

All noise/outlier parameters are calibrated against REAL mono-SLAM maps
(tools/analyze_real_noise.py, 33 aligned 7-Scenes sequences, 2026-07-02):

  real correspondence errors (mono line vs mesh line, after GT SIM3):
    direction error : median 6.1 deg, P90 11.9 deg   (heavy-tailed)
    perp offset     : median 8.3 cm,  P90 13.4 cm
    moment residual : median 0.21 m,  P90 0.49 m
  inlier fraction vs mesh reference : P25-P95 = 51-64 %
  GT scale (mono->metric)           : median 2.1, P25-P95 = 1.7-3.4
  outliers are SCENE-STRUCTURED     : real scene lines (<1 deg to nearest
                                      mesh direction), NOT isotropic junk

Realism features (vs the old i.i.d.-Gaussian generator):
  * noise injected in the METRIC frame, so residuals match measurements
    independent of the sampled scale s
  * per-pair severity multiplier (easy sequences vs drifty ones)
  * query = spatially coherent region (sphere), like a real SLAM run
  * outliers = real mesh lines absent from the reference subset
    (+ a small isotropic junk fraction)
  * random Plücker sign flips ([m,d] -> [-m,-d]): SLAM endpoint order is
    arbitrary, so ~50 % of real correspondences are sign-flipped

  plucker1  — metric reference lines (mesh subsample + background)
  plucker2  — mono query lines (inverse SIM3 + calibrated noise)
  R_gt / t_gt / s_gt  — SIM3 mapping plucker2 (mono) → plucker1 (metric)

Split (scene-held-out, like the KITTI generator's seq 09/10 split):
  train: chess, fire, heads, pumpkin, redkitchen, stairs  (N_PER_SCENE pairs each)
  valid: office  (held-out scene, N_VALID pairs)

Usage
-----
  cd ScalePluckerNet
  python scripts/generate_7scenes_dataset.py
  python scripts/generate_7scenes_dataset.py --n_per_scene 3000 --n_valid 1200 --workers 8
"""

import argparse
import pickle
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import msgpack
import numpy as np

_ROOT     = Path(__file__).resolve().parent.parent          # ScalePluckerNet/
_TESTDATA = _ROOT.parent / "test_data" / "7scenes"

SCENES_TRAIN = ["chess", "fire", "heads", "pumpkin", "redkitchen", "stairs"]
SCENES_VALID = ["office"]

# SIM(3) scale distribution — measured: median 2.1, P25-P95 = 1.7-3.4
_SCALE_LOG_MU  = np.log(2.2)
_SCALE_LOG_STD = 0.45
_SCALE_CLIP    = (0.7, 7.0)

# Query noise, injected in the METRIC frame (matches measured residuals).
# Rayleigh(sigma): median = 1.177*sigma, P90 = 2.146*sigma
_DIR_SIGMA_DEG = 5.2    # -> median ~6.1 deg, P90 ~11.2 deg at severity 1
_DIR_CAP_DEG   = 25.0
_PERP_SIGMA_M  = 0.07   # -> median ~8.2 cm, P90 ~15 cm at severity 1
_PERP_CAP_M    = 0.35
_SEVERITY      = (0.6, 1.5)   # per-pair multiplier (clean vs drifty runs)

# Reference noise (mesh is near-perfect; RGBD/DA3 maps have a little)
_REF_DIR_SIGMA_DEG = 1.5
_REF_PERP_SIGMA_M  = 0.02

# Pair composition
_REGION_RADIUS = (1.0, 3.0)   # metres — spatial extent of a mono run
_N_QRY_RANGE   = (60, 250)
_N_REF_RANGE   = (100, 350)
_OVERLAP_FRAC  = (0.25, 0.65)  # fraction of query lines present in reference
_JUNK_FRAC_MAX = 0.10          # isotropic junk outliers in query
_MIN_REGION    = 40
_MIN_INLIERS   = 8
_MIN_MESH      = 80


# ── Plücker helpers ───────────────────────────────────────────────────────────

def _random_rotation() -> np.ndarray:
    A = np.random.randn(3, 3).astype(np.float64)
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float32)


def _apply_sim3(L: np.ndarray, s: float,
                R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Transform Plücker lines [m, d] by SIM(3)(s, R, t)."""
    m, d  = L[:, :3].copy(), L[:, 3:].copy()
    d_out = (R @ d.T).T
    m_out = s * (R @ m.T).T + np.cross(t[None], d_out)
    return np.concatenate([m_out, d_out], axis=1).astype(np.float32)


def _rand_perp(d: np.ndarray) -> np.ndarray:
    """Random unit vectors perpendicular to each row of d (N,3)."""
    r = np.random.randn(*d.shape)
    v = np.cross(d, r)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    # degenerate cross products: retry with a fixed axis
    bad = (n < 1e-8).ravel()
    if bad.any():
        v[bad] = np.cross(d[bad], np.array([1.0, 0.5, 0.25]))
        n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / n


def _noisy_lines(mid: np.ndarray, d: np.ndarray,
                 dir_sigma_deg: float, perp_sigma_m: float,
                 dir_cap_deg: float = 90.0, perp_cap_m: float = 10.0):
    """Perturb (midpoint, direction) with Rayleigh-distributed rotation of the
    direction and Rayleigh perpendicular midpoint offset. Returns [m,d] Plücker."""
    n = len(d)
    # rotate each direction by angle ~ Rayleigh(sigma) about a random ⟂ axis
    theta = np.minimum(np.random.rayleigh(np.deg2rad(dir_sigma_deg), n),
                       np.deg2rad(dir_cap_deg))
    u = _rand_perp(d)
    d_out = np.cos(theta)[:, None] * d + np.sin(theta)[:, None] * u
    d_out /= np.linalg.norm(d_out, axis=1, keepdims=True)
    # perpendicular midpoint offset ~ Rayleigh(sigma)
    r = np.minimum(np.random.rayleigh(perp_sigma_m, n), perp_cap_m)
    mid_out = mid + r[:, None] * _rand_perp(d_out)
    m_out = np.cross(mid_out, d_out)
    return np.concatenate([m_out, d_out], axis=1).astype(np.float32)


def _make_junk_outliers(n: int, centre: np.ndarray, pos_range: float) -> np.ndarray:
    """Isotropic junk lines (spurious SLAM landmarks) in the metric frame."""
    if n == 0:
        return np.zeros((0, 6), np.float32)
    d = np.random.randn(n, 3)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    p = centre[None] + np.random.uniform(-pos_range, pos_range, (n, 3))
    m = np.cross(p, d)
    return np.concatenate([m, d], axis=1).astype(np.float32)


def _random_sign_flip(L: np.ndarray, frac: float = 0.5) -> np.ndarray:
    """Flip [m,d] -> [-m,-d] on a random subset (same physical line;
    SLAM endpoint order is arbitrary)."""
    flip = np.random.rand(len(L)) < frac
    out = L.copy()
    out[flip] *= -1.0
    return out


# ── Mesh DB loader ────────────────────────────────────────────────────────────

def load_mesh_lines(db_path: Path):
    """Load mesh.db → (N,6) Plücker [m,d], (N,3) midpoints, (N,3) directions."""
    with open(db_path, "rb") as f:
        data = msgpack.unpack(f, raw=False)
    lines, mids, dirs = [], [], []
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
        mid = (p1 + p2) * 0.5
        lines.append(np.concatenate([np.cross(mid, d), d]).astype(np.float32))
        mids.append(mid.astype(np.float32))
        dirs.append(d.astype(np.float32))
    if not lines:
        z = np.zeros((0, 3), np.float32)
        return np.zeros((0, 6), np.float32), z, z
    return (np.array(lines, np.float32),
            np.array(mids,  np.float32),
            np.array(dirs,  np.float32))


# ── Pair generator ────────────────────────────────────────────────────────────

def _generate_one(mesh: tuple, rng_seed: int) -> dict | None:
    """Generate one (reference, query, GT-SIM3) pair from a scene mesh."""
    np.random.seed(rng_seed)
    lines, mids, dirs = mesh
    N = len(lines)
    if N < _MIN_MESH:
        return None

    # ── SIM(3) parameters (measured scale distribution) ──────────────────────
    s = float(np.clip(np.exp(np.random.normal(_SCALE_LOG_MU, _SCALE_LOG_STD)),
                      *_SCALE_CLIP))
    R = _random_rotation()
    t_range = 0.4 * s
    t = np.random.uniform(-t_range, t_range, 3).astype(np.float32)
    s_inv = 1.0 / s
    R_inv = R.T
    t_inv = (-s_inv * (R_inv @ t)).astype(np.float32)

    severity = np.random.uniform(*_SEVERITY)

    # ── Query region: spatially coherent sphere (like a real mono run) ───────
    radius = np.random.uniform(*_REGION_RADIUS)
    centre = mids[np.random.randint(N)]
    region = np.where(np.linalg.norm(mids - centre[None], axis=1) <= radius)[0]
    if len(region) < _MIN_REGION:
        return None

    n_qry = min(len(region), np.random.randint(*_N_QRY_RANGE))
    qry_src = np.random.choice(region, n_qry, replace=False)

    # ── Overlap: which query lines exist in the reference ────────────────────
    f_overlap = np.random.uniform(*_OVERLAP_FRAC)
    in_ref = np.random.rand(n_qry) < f_overlap
    n_inlier = int(in_ref.sum())
    if n_inlier < _MIN_INLIERS:
        return None
    inlier_src = qry_src[in_ref]          # mesh indices shared by both maps

    # ── Reference: shared lines + background from the rest of the scene ──────
    # Background can be inside OR outside the region (an RGBD/mesh map sees
    # more of the scene than the mono run) — those inside act as confusers.
    n_ref = np.random.randint(*_N_REF_RANGE)
    bg_pool = np.setdiff1d(np.arange(N), qry_src)   # never sources of query lines
    n_bg = min(max(n_ref - n_inlier, 0), len(bg_pool))
    bg_sel = np.random.choice(bg_pool, n_bg, replace=False)
    ref_src = np.concatenate([inlier_src, bg_sel])  # inliers first
    n_ref = len(ref_src)

    plucker1 = _noisy_lines(mids[ref_src], dirs[ref_src],
                            _REF_DIR_SIGMA_DEG, _REF_PERP_SIGMA_M)

    # ── Query: calibrated metric-frame noise, then exact inverse SIM3 ────────
    qry_metric = _noisy_lines(mids[qry_src], dirs[qry_src],
                              _DIR_SIGMA_DEG * severity,
                              _PERP_SIGMA_M * severity,
                              _DIR_CAP_DEG, _PERP_CAP_M)
    # small isotropic junk fraction (spurious landmarks)
    n_junk = int(np.random.uniform(0.0, _JUNK_FRAC_MAX) * n_qry)
    junk = _make_junk_outliers(n_junk, centre, radius)
    qry_metric = np.concatenate([qry_metric, junk], axis=0)
    n_qry_total = len(qry_metric)

    mono = _apply_sim3(qry_metric, s_inv, R_inv, t_inv)

    # ── Random Plücker sign flips (arbitrary SLAM endpoint order) ─────────────
    mono     = _random_sign_flip(mono, 0.5)
    plucker1 = _random_sign_flip(plucker1, 0.5)

    # ── Shuffle both clouds ───────────────────────────────────────────────────
    perm_qry = np.random.permutation(n_qry_total)
    perm_ref = np.random.permutation(n_ref)
    plucker2 = mono[perm_qry]
    plucker1 = plucker1[perm_ref]

    # ── Build matches (2, n_inlier) ───────────────────────────────────────────
    # Raw query positions of inliers = indices where in_ref is True; their
    # reference raw positions are 0..n_inlier-1 (inliers first in ref_src).
    qry_raw_inlier = np.where(in_ref)[0]            # raw pos in qry_metric
    ref_pos_map = np.argsort(perm_ref)              # raw ref pos -> new pos
    qry_pos_map = np.argsort(perm_qry)              # raw qry pos -> new pos
    ref_new = ref_pos_map[np.arange(n_inlier)]
    qry_new = qry_pos_map[qry_raw_inlier]
    matches = np.stack([ref_new, qry_new]).astype(np.int32)

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
    ap.add_argument("--n_valid",     type=int, default=1200,
                    help="Validation pairs (held-out scene) (default: 1200)")
    ap.add_argument("--workers",     type=int, default=4)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--out_dir",     default=str(_ROOT / "dataset"))
    ap.add_argument("--name",        default="7scenes_mesh",
                    help="Dataset name → <out_dir>/<name>_train and <name>_valid "
                         "(default: 7scenes_mesh)")
    args = ap.parse_args()

    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)

    # ── Load all mesh.db files ────────────────────────────────────────────────
    mesh_by_scene = {}
    for scene in SCENES_TRAIN + SCENES_VALID:
        db = _TESTDATA / "mesh" / f"{scene}_mesh.db"
        if not db.exists():
            print(f"[skip] {scene}: mesh.db not found")
            continue
        mesh = load_mesh_lines(db)
        if len(mesh[0]) < _MIN_MESH:
            print(f"[skip] {scene}: only {len(mesh[0])} mesh lines")
            continue
        mesh_by_scene[scene] = mesh
        print(f"  {scene}: {len(mesh[0])} mesh lines loaded")

    # ── Build task list ───────────────────────────────────────────────────────
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

    save_dataset(train_pairs, out_dir / f"{args.name}_train")
    save_dataset(valid_pairs, out_dir / f"{args.name}_valid")

    # ── Stats ─────────────────────────────────────────────────────────────────
    all_s  = [float(p["s_gt"]) for p in train_pairs]
    all_ir = [p["matches"].shape[1] / p["plucker2"].shape[0] for p in train_pairs]
    all_rf = [p["plucker1"].shape[0] for p in train_pairs]
    all_qr = [p["plucker2"].shape[0] for p in train_pairs]
    print(f"\nTrain stats (targets from tools/analyze_real_noise.py):")
    print(f"  scale        : median={np.median(all_s):.2f}  "
          f"P25={np.percentile(all_s,25):.2f}  P75={np.percentile(all_s,75):.2f}"
          f"   (real: 2.1 / 1.7 / 2.8)")
    print(f"  qry inlier % : median={np.median(all_ir)*100:.0f}%  "
          f"P25={np.percentile(all_ir,25)*100:.0f}%  P75={np.percentile(all_ir,75)*100:.0f}%"
          f"   (real vs mesh: ~51-64%)")
    print(f"  n_ref median : {np.median(all_rf):.0f}   n_qry median: {np.median(all_qr):.0f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
