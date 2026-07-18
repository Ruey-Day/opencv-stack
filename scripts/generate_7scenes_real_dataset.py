"""
Generate 7scenes_real_train / 7scenes_real_valid datasets whose QUERY side is
REAL mono-SLAM line maps (not synthetic noise on mesh lines).

Motivation (2026-07-03 diagnosis): 7scenes_mesh_v1 gets 121/200 correspondence
precision on synthetic queries but 0/200 on real mono maps of the SAME scene.
Calibrated i.i.d. noise does not reproduce the structural character of real
mono lines (fragmentation, trajectory-clustered density, photometric edges).
The only faithful noise model is the real data itself.

Per pair:
  query      = spatially coherent subset of a real *_mono.db line map,
               re-framed by a random SIM(3) (random R, t; scale chosen so the
               pair's GT scale follows the same LogNormal as 7scenes_mesh)
  reference  = scene mesh lines (70 %) or the GT-aligned RGBD map (30 %):
               the query's gate-matched lines (thinned to a target overlap
               fraction) + region-biased background lines
  matches    = mono line -> reference line, valid when the GT-aligned mono
               line lies within 15 deg / 15 cm of the reference line
               (same gate as tools/analyze_real_noise.py)

  plucker1 = reference (metric),  plucker2 = query (mono)
  R_gt / t_gt / s_gt map plucker2 -> plucker1   (same convention as
  generate_7scenes_dataset.py)

Split (scene-held-out, same as 7scenes_mesh):
  train: chess, fire, heads, pumpkin, redkitchen, stairs
  valid: office

Usage
-----
  cd ScalePluckerNet
  python scripts/generate_7scenes_real_dataset.py --n_per_seq 460 --n_valid_per_seq 60
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import msgpack
import numpy as np

_ROOT     = Path(__file__).resolve().parent.parent            # ScalePluckerNet/
_TESTDATA = _ROOT.parent / "test_data" / "7scenes"
sys.path.insert(0, str(_ROOT.parent / "tools"))

from analyze_real_noise import (load_gt_tum, load_keyframes, load_segments,
                                read_db, umeyama_sim3)

# Scene spec: (dataset_root, scene, has_mesh, rgbd_db_glob, rgbd_gt_relpath)
_7S  = _ROOT.parent / "test_data" / "7scenes"
_REP = _ROOT.parent / "test_data" / "replica"

def _spec_7s(scene):
    return dict(root=_7S, scene=scene, mesh=_7S / "mesh" / f"{scene}_mesh.db",
                rgbd_glob="*_rgbd.db", rgbd_gt=None)   # gt at <seq>/gt_tum.txt

def _spec_rep(scene):
    return dict(root=_REP, scene=scene, mesh=None,
                rgbd_glob="full_rgbd.db", rgbd_gt="full/gt_tum.txt")

SCENES_TRAIN = ([_spec_7s(s) for s in
                 ["chess", "fire", "heads", "pumpkin", "redkitchen", "stairs"]] +
                [_spec_rep(s) for s in
                 ["office0", "office1", "office2", "office3", "office4",
                  "room0", "room1"]])
SCENES_VALID = [_spec_7s("office"), _spec_rep("room2")]

# Same pair-scale distribution as generate_7scenes_dataset.py
_SCALE_LOG_MU  = np.log(2.2)
_SCALE_LOG_STD = 0.45
_SCALE_CLIP    = (0.7, 7.0)

# GT correspondence gate (same as tools/analyze_real_noise.py)
_GATE_ANG_DEG  = 15.0
_GATE_PERP_M   = 0.15

# Pair composition (mirrors generate_7scenes_dataset.py; query range extended
# because real maps are registered whole at test time)
_N_QRY_RANGE   = (60, 400)
_N_REF_RANGE   = (100, 350)
_OVERLAP_FRAC  = (0.25, 0.65)   # fraction of query lines present in reference
_REGION_BIAS   = 0.7            # background refs drawn near the query region
_MIN_INLIERS   = 8
_RGBD_REF_PROB = 0.3
_LEN_PCT       = 40             # same line-length filter as eval
_MAX_ALIGN_RMSE = 0.30

# Small reference noise for the (near-perfect) mesh, as in the mesh generator
_REF_DIR_SIGMA_DEG = 1.5
_REF_PERP_SIGMA_M  = 0.02


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _plucker(p1, p2):
    d = p2 - p1
    ln = np.linalg.norm(d, axis=1, keepdims=True)
    d = d / ln
    m = np.cross((p1 + p2) * 0.5, d)
    return np.concatenate([m, d], axis=1).astype(np.float32)


def _random_rotation(rng):
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _apply_sim3_pts(p, s, R, t):
    return (s * (R @ p.T)).T + t


def _apply_sim3_plucker(L, s, R, t):
    m, d = L[:, :3], L[:, 3:]
    d_out = (R @ d.T).T
    m_out = s * (R @ m.T).T + np.cross(t[None], d_out)
    return np.concatenate([m_out, d_out], axis=1).astype(np.float32)


def _noisy_segments(p1, p2, dir_sigma_deg, perp_sigma_m, rng):
    """Small Rayleigh direction + perpendicular midpoint noise on segments."""
    d = p2 - p1
    ln = np.linalg.norm(d, axis=1, keepdims=True)
    d = d / ln
    mid = (p1 + p2) / 2
    n = len(p1)
    ax = np.cross(d, rng.normal(size=(n, 3)))
    ax /= np.linalg.norm(ax, axis=1, keepdims=True) + 1e-9
    ang = np.radians(rng.rayleigh(dir_sigma_deg, n))
    ca, sa = np.cos(ang)[:, None], np.sin(ang)[:, None]
    d_new = ca * d + sa * np.cross(ax, d) + (1 - ca) * (ax * d).sum(1, keepdims=True) * ax
    d_new /= np.linalg.norm(d_new, axis=1, keepdims=True)
    off = rng.normal(size=(n, 3))
    off -= (off * d_new).sum(1, keepdims=True) * d_new
    off *= (rng.rayleigh(perp_sigma_m, n) / (np.linalg.norm(off, axis=1) + 1e-9))[:, None]
    mid_new = mid + off
    return mid_new - d_new * ln / 2, mid_new + d_new * ln / 2


# ── Per-sequence preprocessing ────────────────────────────────────────────────

def gt_align(db, gt_path):
    kf_ts, kf_c = load_keyframes(db)
    if len(kf_ts) < 8:
        return None, "too few keyframes"
    gt_ts, gt_p = load_gt_tum(gt_path)
    idx = np.searchsorted(gt_ts, kf_ts).clip(0, len(gt_ts) - 1)
    idx_lo = (idx - 1).clip(0)
    pick = np.where(np.abs(gt_ts[idx_lo] - kf_ts) < np.abs(gt_ts[idx] - kf_ts), idx_lo, idx)
    ok = np.abs(gt_ts[pick] - kf_ts) < 0.02
    if ok.sum() < 8:
        return None, "too few ts matches"
    s, R, t = umeyama_sim3(kf_c[ok], gt_p[pick][ok])
    aligned = _apply_sim3_pts(kf_c[ok], s, R, t)
    rmse = float(np.sqrt(((aligned - gt_p[pick][ok]) ** 2).sum(1).mean()))
    if rmse > _MAX_ALIGN_RMSE:
        return None, f"align rmse {rmse:.2f}"
    return (s, R, t), None


def best_gate_match(q1g, q2g, r1, r2):
    """Best reference line per GT-aligned query segment within the gate.

    Returns (N_q,) int index into reference, -1 where no match."""
    qd = q2g - q1g
    qd /= np.linalg.norm(qd, axis=1, keepdims=True)
    qmid = (q1g + q2g) / 2
    rd = r2 - r1
    rd /= np.linalg.norm(rd, axis=1, keepdims=True)
    ang = np.arccos(np.abs(qd @ rd.T).clip(0, 1))                  # (Q, R)
    diff = qmid[:, None, :] - r1[None, :, :]
    perp = np.linalg.norm(np.cross(diff, rd[None, :, :]), axis=2)  # (Q, R)
    score = perp + 1.0 * ang
    ok = (np.degrees(ang) < _GATE_ANG_DEG) & (perp < _GATE_PERP_M)
    score[~ok] = np.inf
    best = score.argmin(axis=1)
    best[~np.isfinite(score.min(axis=1))] = -1
    return best


class SeqData:
    """Everything needed to sample pairs from one mono sequence."""
    def __init__(self, scene, seq, mono_p1, mono_p2, align, refs):
        self.scene, self.seq = scene, seq
        self.p1, self.p2 = mono_p1, mono_p2          # mono frame segments
        self.mid = (mono_p1 + mono_p2) / 2
        self.align = align                            # mono -> GT SIM3
        # refs: dict ref_name -> (r1, r2, best_match_idx (N_q,))
        self.refs = refs


def load_scene_sequences(spec, rng):
    """Load mesh + rgbd reference pools and every GT-alignable mono seq."""
    scene = spec["scene"]
    scene_dir = spec["root"] / scene
    ref_pools = {}
    if spec["mesh"] is not None:
        m1, m2 = load_segments(read_db(spec["mesh"]), min_nfnd=0,
                               len_percentile=_LEN_PCT)
        ref_pools["mesh"] = (m1, m2)

    rgbd_paths = sorted(scene_dir.glob(spec["rgbd_glob"]))
    if rgbd_paths:
        rgbd_db = read_db(rgbd_paths[0])
        if spec["rgbd_gt"] is not None:
            rgbd_gt = scene_dir / spec["rgbd_gt"]
        else:
            rseq = rgbd_paths[0].name.replace("_rgbd.db", "")
            rgbd_gt = scene_dir / rseq / "gt_tum.txt"
        al, why = gt_align(rgbd_db, rgbd_gt)
        if al is not None:
            g1, g2 = load_segments(rgbd_db, len_percentile=_LEN_PCT)
            sa, Ra, ta = al
            ref_pools["rgbd"] = (_apply_sim3_pts(g1, sa, Ra, ta),
                                 _apply_sim3_pts(g2, sa, Ra, ta))
        else:
            print(f"  [{scene}] rgbd ref skipped: {why}")
    if not ref_pools:
        print(f"  [{scene}] no usable reference, skipped")
        return []

    seqs = []
    for mono_path in sorted(scene_dir.glob("*_mono.db")):
        seq = mono_path.name.replace("_mono.db", "")
        if seq == "full":
            continue
        gt_path = scene_dir / seq / "gt_tum.txt"
        if not gt_path.exists():
            continue
        db = read_db(mono_path)
        al, why = gt_align(db, gt_path)
        if al is None:
            print(f"  [{scene}/{seq}] skipped: {why}")
            continue
        q1, q2 = load_segments(db, len_percentile=_LEN_PCT)
        if len(q1) < 60:
            print(f"  [{scene}/{seq}] skipped: only {len(q1)} lines")
            continue
        sa, Ra, ta = al
        q1g, q2g = _apply_sim3_pts(q1, sa, Ra, ta), _apply_sim3_pts(q2, sa, Ra, ta)
        refs = {}
        for name, (r1, r2) in ref_pools.items():
            best = best_gate_match(q1g, q2g, r1, r2)
            refs[name] = (r1, r2, best)
        main_ref = "mesh" if "mesh" in refs else "rgbd"
        n_m = (refs[main_ref][2] >= 0).sum()
        print(f"  [{scene}/{seq}] {len(q1)} lines, {main_ref}-matched {n_m} "
              f"({n_m / len(q1) * 100:.0f}%)")
        seqs.append(SeqData(scene, seq, q1, q2, al, refs))
    return seqs


# ── Pair sampling ─────────────────────────────────────────────────────────────

def make_pair(sd: SeqData, rng):
    if "mesh" not in sd.refs:
        ref_name = "rgbd"
    else:
        ref_name = "rgbd" if ("rgbd" in sd.refs and rng.random() < _RGBD_REF_PROB) else "mesh"
    r1, r2, best = sd.refs[ref_name]

    # 1) spatially coherent query subset: anchor + nearest neighbours
    n_total = len(sd.p1)
    n_qry = min(n_total, rng.integers(*_N_QRY_RANGE))
    anchor = rng.integers(n_total)
    order = np.argsort(np.linalg.norm(sd.mid - sd.mid[anchor], axis=1))
    q_idx = order[:n_qry]

    matched = q_idx[best[q_idx] >= 0]
    if len(matched) < _MIN_INLIERS:
        return None

    # 2) overlap control: thin the matched reference lines
    f_overlap = rng.uniform(*_OVERLAP_FRAC)
    ref_of_matched = np.unique(best[matched])
    n_keep = max(_MIN_INLIERS // 2 + 1, int(round(f_overlap * len(ref_of_matched))))
    if n_keep < len(ref_of_matched):
        ref_keep = rng.choice(ref_of_matched, n_keep, replace=False)
    else:
        ref_keep = ref_of_matched

    # 3) background reference lines, biased to the query region (GT frame)
    sa, Ra, ta = sd.align
    centroid = _apply_sim3_pts(sd.mid[q_idx].mean(0, keepdims=True), sa, Ra, ta)[0]
    rmid = (r1 + r2) / 2
    n_ref = int(rng.integers(*_N_REF_RANGE))
    n_bg = max(n_ref - len(ref_keep), 20)
    bg_pool = np.setdiff1d(np.arange(len(r1)), ref_keep)
    if len(bg_pool) > 0:
        dist = np.linalg.norm(rmid[bg_pool] - centroid, axis=1)
        radius = np.percentile(np.linalg.norm(rmid[ref_keep] - centroid, axis=1), 90) * 1.5 + 0.5
        near = bg_pool[dist <= radius]
        far = bg_pool[dist > radius]
        n_near = min(int(round(n_bg * _REGION_BIAS)), len(near))
        n_far = min(n_bg - n_near, len(far))
        bg = np.concatenate([
            rng.choice(near, n_near, replace=False) if n_near else np.empty(0, int),
            rng.choice(far, n_far, replace=False) if n_far else np.empty(0, int),
        ])
    else:
        bg = np.empty(0, int)
    ref_idx = np.concatenate([ref_keep, bg]).astype(int)
    rng.shuffle(ref_idx)

    # 4) reference Plücker (mesh gets a little noise; rgbd is already real)
    rp1, rp2 = r1[ref_idx], r2[ref_idx]
    if ref_name == "mesh":
        rp1, rp2 = _noisy_segments(rp1, rp2, _REF_DIR_SIGMA_DEG, _REF_PERP_SIGMA_M, rng)
    plucker1 = _plucker(rp1, rp2)

    # 5) query re-framing: random SIM3 so the pair's GT scale ~ LogNormal
    s_pair = float(np.clip(np.exp(rng.normal(_SCALE_LOG_MU, _SCALE_LOG_STD)), *_SCALE_CLIP))
    sq = sa / s_pair
    Rq = _random_rotation(rng)
    tq = rng.normal(size=3) * 0.5
    qp1 = _apply_sim3_pts(sd.p1[q_idx], sq, Rq, tq)
    qp2 = _apply_sim3_pts(sd.p2[q_idx], sq, Rq, tq)
    plucker2 = _plucker(qp1, qp2)

    # GT: query -> reference(GT) frame
    R_gt = Ra @ Rq.T
    t_gt = ta - s_pair * (R_gt @ tq)

    # 6) random Plücker sign flips on both sides
    for pl in (plucker1, plucker2):
        flip = rng.random(len(pl)) < 0.5
        pl[flip] *= -1.0

    # 7) match indices in the pair's local indexing (row0=ref, row1=query)
    ref_pos = {g: i for i, g in enumerate(ref_idx)}
    rows, cols = [], []
    for local_q, global_q in enumerate(q_idx):
        b = best[global_q]
        if b >= 0 and b in ref_pos:
            rows.append(ref_pos[b])
            cols.append(local_q)
    if len(cols) < _MIN_INLIERS:
        return None
    matches = np.stack([np.array(rows, np.int32), np.array(cols, np.int32)])

    return dict(matches=matches, plucker1=plucker1, plucker2=plucker2,
                R_gt=R_gt.astype(np.float32),
                t_gt=t_gt.reshape(3, 1).astype(np.float32),
                s_gt=np.float32(s_pair))


def generate_split(scenes, n_per_seq, out_dir, seed):
    rng = np.random.default_rng(seed)
    data = dict(matches=[], plucker1=[], plucker2=[], R_gt=[], t_gt=[], s_gt=[])
    for spec in scenes:
        print(f"{spec['scene']}:")
        seqs = load_scene_sequences(spec, rng)
        for sd in seqs:
            made, tries = 0, 0
            while made < n_per_seq and tries < n_per_seq * 5:
                tries += 1
                pair = make_pair(sd, rng)
                if pair is None:
                    continue
                for k in data:
                    data[k].append(pair[k])
                made += 1
            print(f"    {sd.seq}: {made} pairs")
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, v in data.items():
        with open(out_dir / f"{k}.pkl", "wb") as f:
            pickle.dump(v, f)
    n = len(data["s_gt"])
    inl = np.array([m.shape[1] for m in data["matches"]])
    nq = np.array([p.shape[0] for p in data["plucker2"]])
    print(f"-> {out_dir}: {n} pairs | inliers median {np.median(inl):.0f} | "
          f"query lines median {np.median(nq):.0f} | "
          f"inlier frac median {np.median(inl / nq) * 100:.0f}%")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_seq", type=int, default=460)
    ap.add_argument("--n_valid_per_seq", type=int, default=60)
    ap.add_argument("--out_dir", default=str(_ROOT / "dataset"))
    ap.add_argument("--name", default="7scenes_real",
                    help="dataset dir prefix (<name>_train / <name>_valid)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.out_dir)
    generate_split(SCENES_TRAIN, args.n_per_seq, out / f"{args.name}_train", args.seed)
    generate_split(SCENES_VALID, args.n_valid_per_seq, out / f"{args.name}_valid", args.seed + 1)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
