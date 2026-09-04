#!/usr/bin/env python3
"""
Generate a large, fully-synthetic dataset for Sim(3) line correspondence training.

No map files required — all pools come from random geometric primitives
(plane patches, wireframes, line bundles, parallel groups, grid patches,
staircases) with no axis-aligned or Manhattan-world bias.

Scenarios
---------
room        (30%) — indoor room scale, moderate-to-high overlap
submap      (10%) — sparse/noisy monocular query vs large clean metric reference
relocalize  (22%) — cross-session re-localization, moderate overlap
loop        (28%) — loop-closure, high overlap, both sides similar
dense_sparse(10%) — very different line densities (RGB-D vs monocular)

Output
------
dataset/synthetic_train/   dataset/synthetic_valid/

Each split uses the standard 6-pickle format:
    matches.pkl  plucker1.pkl  plucker2.pkl  R_gt.pkl  t_gt.pkl  s_gt.pkl

Usage
-----
python generate_synthetic.py
python generate_synthetic.py --n_train 500000 --n_valid 5000
python generate_synthetic.py --n_train 200000 --workers 8 --seed 123
"""
import os
import sys
import pickle
import argparse
import time
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))


# ── Constants ─────────────────────────────────────────────────────────────────

_SCALE_RANGE      = (0.3, 8.0)   # tightened: avoids s_i > 3.3 to prevent moment explosion
_ROOM_SCALE_RANGE = (0.3, 5.0)
_MIN_INLIERS      = 20

# v4 calibration: match measured GT scale of real mono→metric pairs
# (tools/analyze_real_noise.py 2026-07-03: median 2.1, P25–P95 = 1.7–3.4).
# v5: wider spread/clip — the 33-seq GT benchmark (2026-07-07) observed GT
# scales 0.74–10.65 (heads at the old lower clip 0.7, stairs OUTSIDE the old
# upper clip 7.0); cover them with margin while keeping the same median.
_SCALE_LOG_CENTER = np.log(2.2)
_SCALE_LOG_STD    = 0.60
_SCALE_CLIP       = (0.4, 13.0)

# v4 real-structure noise (calibrated against tools/analyze_real_noise.py):
# physical perturbation of (foot point, direction) with m recomputed, so the
# Plücker constraint m·d = 0 holds — Gaussian noise directly on m violates it
# and real/test lines never do.
_V4_DIR_SIGMA_DEG = 5.2      # Rayleigh → median ~6.1°, P90 ~11.2°
_V4_DIR_CAP_DEG   = 25.0
_V4_PERP_SIGMA_M  = 0.07     # Rayleigh, in the METRIC frame
_V4_SEVERITY      = (0.6, 1.5)   # per-pair multiplier (clean vs drifty run)
_V4_REF_DIR_DEG   = 1.5
_V4_REF_PERP_M    = 0.02
_V4_FRAG_P        = (0.65, 0.25, 0.10)  # P(1|2|3 query fragments per real edge)
_V4_SIGN_FLIP     = 0.5      # SLAM endpoint order is arbitrary → ~50% flipped

_INDOOR = bool(int(os.environ.get('INDOOR', '0')))   # v37 foundation flag

_SCENARIOS  = ['room', 'submap', 'relocalize', 'loop', 'dense_sparse', 'manhattan', 'corridor', 'hard_noise', 'adversarial', 'outdoor', 'street', 'street_submap', 'collision', 'street_mono']
if _INDOOR:                      # v37: three building-scale indoor families
    _SCENARIOS = _SCENARIOS + ['building', 'atrium', 'cluttered']
# Weights below are current; their revision history is in git. One durable
# finding: a 'flip_room' scenario (repetitive-room ROTATIONAL alias) was tried
# and REPLACED by 'collision' -- the measured matcher failure is descriptor
# collision on dense similar lines, not rotational symmetry.
_SCENARIO_P = np.array([0.0756, 0.0567, 0.0567, 0.0693, 0.0378, 0.0882,
                        0.0567, 0.0567, 0.0945, 0.0567, 0.1029, 0.1008,
                        0.16, 0.0])   # last = street_mono (off unless MONO=1)
_SCENARIO_P = _SCENARIO_P / _SCENARIO_P.sum()   # exact normalization
# v37 INDOOR: the three building-scale indoor families take 0.18 total
# (0.06 each), the rest scaled by 0.82 — enough mass to learn the regime
# without displacing the street/submap families the KITTI rows depend on.
if _INDOOR:
    _SCENARIO_P = np.concatenate([_SCENARIO_P * 0.82, np.full(3, 0.06)])
    _SCENARIO_P = _SCENARIO_P / _SCENARIO_P.sum()

# v24 SUBCAL: street_submap weight 0.10 -> 0.16, others scaled down
if bool(int(os.environ.get('SUBCAL', '0'))):
    _i_ss = _SCENARIOS.index('street_submap')
    _SCENARIO_P = _SCENARIO_P * (1 - 0.16) / (1 - _SCENARIO_P[_i_ss])
    _SCENARIO_P[_i_ss] = 0.16
    _SCENARIO_P = _SCENARIO_P / _SCENARIO_P.sum()


# ── Plücker line primitives ───────────────────────────────────────────────────

def _random_rotation() -> np.ndarray:
    A = np.random.randn(3, 3).astype(np.float64)
    Q, R = np.linalg.qr(A)
    Q *= np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float32)


# ROT_CAL=1 : sample the PAIR (inter-map) rotation from the measured real
# distribution instead of Haar-uniform SO(3). Real 7-Scenes mono->metric map
# rotations are median 29 deg / max 76 deg (uniform SO(3) is median ~131 deg,
# 4x too broad — the main sim-to-real gap, and what defeats sign canon on the
# training data). Rayleigh(24.6 deg) angle -> median ~29, mean ~31; uniform
# axis (real axes are general, tilt median 63 deg from vertical). Capped 85 deg.
_ROT_CAL = bool(int(os.environ.get('ROT_CAL', '0')))

# SOUND=1 : coverage-corrected regime sampling (2026-09-04).  Two measured
# defects in found8, both fixed here WITHOUT narrowing support:
#
#  (a) ROTATION.  "Uniform in angle over [0,180]" sounds assumption-free, but
#      uniformity is PARAMETERIZATION-dependent: it puts median 80-90 deg and
#      only ~32% of pairs below 58 deg, while both benchmarks sit at 1-29 deg.
#      Measured consequence: just 8.0% of found8 lands in the 7-Scenes
#      operating regime.  The fix is NOT a cap (v19 capped at 90 deg and paid
#      for it: 4.9% inlier retention at 180 deg vs v41's 91.2%) but a MIXTURE
#      -- half uniform-in-angle, half log-uniform in angle.  Support is
#      unchanged (still full SO(3), still 31% of pairs above 90 deg); the
#      log-uniform half just buys equal mass per OCTAVE of rotation, so the
#      1-30 deg band stops being starved.
#
#  (b) INDOOR SCALE.  building/atrium/cluttered drew scale log-uniform over
#      [0.25,100] -- 6 octaves, which is assumption-free in the parameter but
#      leaves only 18.1% of their mass in the indoor band, WORSE than the
#      generic room/manhattan scenarios' 36%.  Since those three differ from
#      room ONLY in the line pool, they should share its difficulty regime;
#      SOUND gives them the same base scale law (plus the metric<->metric bump,
#      which is a real modality cell, not a fit).
#
# This is importance sampling, not benchmark fitting: nothing is removed from
# the support, density is added where the method is deployed.
_SOUND = bool(int(os.environ.get('SOUND', '0')))
# BROAD=1: train on SET broad ranges (rotation, noise severity) that cover the
# realistic cross-modal regime with margin, INSTEAD of statistics calibrated to
# the benchmark. Stronger paper argument (we bound the regime, not fit it) and a
# bet on better generalization to the failure cases. Takes precedence over ROT_CAL.
_BROAD = bool(int(os.environ.get('BROAD', '0')))

# KITTI_MONO=1: recalibrate ONLY the two street scenarios (street, street_submap)
# to the CURRENT KITTI mono->LiDAR test regime, measured 2026-07-31 on
# test_data/kitti/gt_corr/*_mono_best.npz (seqs 03/05/07/10):
#   * scale s_gt = 8-39   (was ~1.0: the old metric camlines->LiDAR calibration)
#   * true-corr overlap  = 2-8% of the query lines  (was Beta(2,7) ~= 22%)
#   * ref (LiDAR) 1.3-5.6k lines, query (mono) 0.9-3.3k lines, ref denser
#   * corr noise dir ~8-10 deg, perp ~0.36 m (metric frame) — covered by the
#     BROAD severity range already; only scale + overlap + counts move here.
# Indoor scenarios are UNTOUCHED, so 7-Scenes is preserved (it never used street).
_KITTI_MONO = bool(int(os.environ.get('KITTI_MONO', '0')))

# FULL=1: the ONE definitive full-range dataset — spans the whole Sim(3) regime
# with MARGIN rather than fitting any single test set. Where KITTI_MONO fits the
# street scale/overlap to the measured KITTI 03/05/07/10 seqs, FULL widens street
# scale to LogNormal(log12,0.8) clip[4,48] and overlap to Beta(1.7,20) (~8%, tail
# to ~20% — bounds KITTI's 2-8% with margin, not fitted), and lifts the base
# (indoor/general) scale ceiling to 16 so scale is a CONTINUUM 0.4-48 with no
# indoor/street gap. Implies BROAD (bounded rotation 0-90 + severity). All 13
# scenarios. This is the dataset the shipped model trains on; the per-regime
# BROAD/KITTI_MONO recipes are superseded by it. FULL=0 -> byte-identical to before.
_FULL = bool(int(os.environ.get('FULL', '0')))
if _FULL:
    _BROAD = True

# FOUND=1: the FOUNDATION dataset (v21) — trains a matcher that works on ANY
# input frame, not the benchmark's. Implies FULL. Motivated by the measured
# 2026-08-14 failures (KITTI submap scramble 0/14 while GT-corr oracle passes):
#   * rotation: FULL capped the pair rotation at 90 deg; the submap scramble is
#     Haar SO(3) (median ~131 deg) — the model had NEVER seen the test regime.
#     FOUND: angle ~ U[0, 180] deg, uniform axis.
#   * scale: FULL capped at 48; submap composed true scales reach ~80.
#     FOUND: base clip [0.25, 32], street clip [3, 100].
#   * drift: real mono maps warp smoothly along the trajectory (measured warp
#     median 2-13 m with +/-10% local scale on KITTI mono_best). FOUND replaces
#     the bucket-translation _apply_drift with _apply_traj_drift (smooth Sim(3)
#     field along a random axis) applied to the WHOLE query map.
#   * sizes are snapped to a x1.5 grid (64..4096) so same-shape pairs can be
#     BATCHED at train time (the model is latency-bound below ~1.5k lines).
_FOUND = bool(int(os.environ.get('FOUND', '0')))
if _FOUND:
    _FULL = True
    _BROAD = True

_SIZE_GRID = np.array([64, 96, 128, 192, 256, 384, 512, 768,
                       1024, 1536, 2048, 3072, 4096])

# GHOST=1 (v22, implies FOUND): near-miss ghost outliers. Real cross-modal
# maps are dominated by a CONTINUUM of almost-right lines (measured on KITTI
# mono_best vs LiDAR, drift-corrected: only ~20% of query lines lie within
# 1 m of a compatible reference line; median nearest-compatible distance
# ~3.5 m ≈ 2.3% of map extent). The generator's outliers were either true
# structure + small noise or random clutter — nothing taught "close but
# wrong", and the matcher scores P@200 ≈ 0 outdoors (v21 ep237). Under
# GHOST, a U[0.25,0.55] fraction of QUERY outliers become ghosts: copies of
# reference structure displaced perpendicular by LogUniform(0.5%, 8%) of the
# map extent + Rayleigh(6 deg) direction jitter, UNLABELED.
_GHOST = bool(int(os.environ.get('GHOST', '0')))
if _GHOST:
    _FOUND = True
    _FULL = True
    _BROAD = True

# SUBCAL=1 (v24, implies GHOST): recalibrate street_submap to the MEASURED
# submap-quarter benchmark (test_data/kitti/submap_gt_corr, 14 kept quarters):
# query 115-821 lines after corridor/length filters (generator had 250-1400),
# matched-query fraction median ~0.35 (generator had Beta(1.7,18) ~ 9% —
# calibrated to FULL-map overlap, ~4x too sparse for quarters whose corridor
# lies entirely inside the reference). Also bumps the street_submap scenario
# weight 0.10 -> 0.16 (others renormalized) — submap localization is the one
# benchmark still at 0/14.
_SUBCAL = bool(int(os.environ.get('SUBCAL', '0')))
if _SUBCAL:
    _GHOST = True
    _FOUND = True
    _FULL = True
    _BROAD = True

# DEALIAS=1 (v27, implies SUBCAL): de-aliased street pool. Measured 2026-08-17:
# the v6 street pool has median ALIAS MULTIPLICITY 20 (geometrically
# indistinguishable twins per line at the 8-deg/3%-extent noise floor) vs
# median 3 on the REAL KITTI lidarv3 maps — the generator made streets 6-7x
# more ambiguous than reality, capping ANY per-line matcher at ~1/20 = 5%
# P@200 in-domain (exactly where v19/v25/v26 all saturate). Sources: period-
# grid vertical positions (node collisions stack identical poles), one
# constant facade half-width W, uniform facade heights, uniform ground-edge
# offsets. The DEALIAS pool draws collision-free Poisson vertical sites,
# per-building facade depth/floor heights, few distinct curb/lane lines and
# more oblique clutter — targeting real-map multiplicity (~3).
# INDOOR=1 (v37 foundation): adds the BUILDING-SCALE indoor regimes that the
# 13 existing families do not cover, aimed at NCLT (Segway + HDL-32E through
# campus buildings: corridor networks, atriums, planar ground-robot motion,
# sensor at ~1-2 m) and ScanNet++ (dense laser scan vs iPhone RGB-D: room to
# multi-room, heavy furniture clutter, full-height handheld coverage).
#   GEOMETRIC GAP BEING FILLED: our indoor pools live at pos_range ~3 m and the
#   street pools at 40-250 m — nothing in between, yet NCLT corridors/atriums
#   and ScanNet++ multi-room scans sit at 15-100 m.  Three families:
#     building  corridor spine + rooms opening off it, per-room yaw jitter
#     atrium    tall open volume, railings/balconies repeated at floor heights
#     cluttered dense short furniture-like edges at varied orientations
#   plus an indoor HEIGHT-BAND asymmetry (ground robot 0-2.5 m vs handheld/
#   laser 0-4 m), the indoor analogue of the street height-band disjointness.
if _INDOOR:                      # defined near _SCENARIOS (needed earlier)
    os.environ.setdefault('DEALIAS', '1')

_DEALIAS = bool(int(os.environ.get('DEALIAS', '0')))
if _DEALIAS:
    _SUBCAL = True
    _GHOST = True
    _FOUND = True
    _FULL = True
    _BROAD = True

# MONO=1 (implies FOUND): add the ALIGNED mono full-map -> LiDAR regime as the
# 'street_mono' scenario (weight 0.10, others renormalized). FOUND draws the
# inter-map rotation ~U[0,180] independent of scale, so the small-rotation +
# large-scale corner that IS real KITTI mono_best (measured rot 0.6-6 deg, scale
# 9-39) is only ~1.2% of found3 — street_mono covers it directly (small rotation,
# scale 8-42, mono_best-calibrated counts/overlap). Compose with GHOST/SUBCAL freely.
_MONO = bool(int(os.environ.get('MONO', '0')))
if _MONO:
    _FOUND = True
    _FULL = True
    _BROAD = True
    _i_sm = _SCENARIOS.index('street_mono')
    _SCENARIO_P = _SCENARIO_P * (1 - 0.10) / (1 - _SCENARIO_P[_i_sm])
    _SCENARIO_P[_i_sm] = 0.10
    _SCENARIO_P = _SCENARIO_P / _SCENARIO_P.sum()


def _make_ghost_lines(src: np.ndarray, n: int) -> np.ndarray:
    """n near-miss copies of random lines from src (same frame): foot point
    displaced perpendicular to the line by LogUniform(0.005, 0.08) x extent,
    direction jittered Rayleigh(6 deg) capped 20. Preserves m·d = 0."""
    if n <= 0 or len(src) == 0:
        return np.zeros((0, 6), np.float32)
    pick = np.random.randint(0, len(src), n)
    m, d = src[pick, :3].astype(np.float64), src[pick, 3:].astype(np.float64)
    p0 = np.cross(d, m)
    ctr = p0.mean(0)
    ext = float(np.abs(np.cross(src[:, 3:], src[:, :3]) - ctr).max()) + 1e-9
    off = np.random.randn(n, 3)
    off -= (off * d).sum(1, keepdims=True) * d
    off /= np.linalg.norm(off, axis=1, keepdims=True) + 1e-9
    mag = np.exp(np.random.uniform(np.log(0.005), np.log(0.08), n)) * ext
    ang = np.radians(np.minimum(np.random.rayleigh(6.0, n), 20.0))
    ax = np.cross(d, np.random.randn(n, 3))
    ax /= np.linalg.norm(ax, axis=1, keepdims=True) + 1e-9
    ca, sa = np.cos(ang)[:, None], np.sin(ang)[:, None]
    d_new = ca * d + sa * np.cross(ax, d) \
        + (1 - ca) * (ax * d).sum(1, keepdims=True) * ax
    d_new /= np.linalg.norm(d_new, axis=1, keepdims=True) + 1e-9
    m_new = np.cross(p0 + off * mag[:, None], d_new)
    return np.concatenate([m_new, d_new], 1).astype(np.float32)

def _pair_rotation() -> np.ndarray:
    # BROAD=1: rotation angle drawn from a SET range [0, 90 deg] that broadly
    # covers realistic cross-modal inter-map rotations (7-Scenes measured max is
    # 76 deg) WITHOUT fitting the measured 29-deg median — a parameter we set to
    # bound the regime, not a statistic we calibrate. ROT_CAL=1 is the calibrated
    # (Rayleigh, median 29) variant. Neither -> Haar-uniform SO(3).
    if _BROAD:
        ax = np.random.randn(3); ax /= np.linalg.norm(ax) + 1e-12
        # FOUND: full SO(3) coverage (uniform angle 0-180). FULL/BROAD: [0, 90].
        _hi = 180.0 if _FOUND else 90.0
        if _SOUND and np.random.random() < 0.5:
            # log-uniform half: equal mass per octave of rotation (see _SOUND).
            ang = np.radians(float(np.exp(np.random.uniform(np.log(0.5), np.log(_hi)))))
        else:
            ang = np.radians(np.random.uniform(0.0, _hi))
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        return (np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)).astype(np.float32)
    if not _ROT_CAL:
        return _random_rotation()
    ax = np.random.randn(3); ax /= np.linalg.norm(ax) + 1e-12
    ang = np.radians(min(float(np.random.rayleigh(24.6)), 85.0))
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
    return R.astype(np.float32)


def _apply_sim3(L6: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    m, d  = L6[:, :3], L6[:, 3:]
    d_new = (R @ d.T).T
    m_new = s * (R @ m.T).T + np.cross(t[None], d_new)
    return np.concatenate([m_new, d_new], axis=1).astype(np.float32)


def _physical_noise(L: np.ndarray, dir_sigma_deg: float, perp_sigma_m: float,
                    dir_cap_deg: float = 90.0) -> np.ndarray:
    """Perturb lines physically: Rayleigh rotation of the direction plus a
    Rayleigh perpendicular offset of the foot point, then recompute m.
    Preserves m·d = 0 (unlike additive Gaussian noise on m)."""
    if len(L) == 0:
        return L
    m, d = L[:, :3], L[:, 3:]
    n = len(L)
    p0 = np.cross(d, m)                                   # foot of perpendicular
    ax = np.cross(d, np.random.randn(n, 3))
    ax /= np.linalg.norm(ax, axis=1, keepdims=True) + 1e-9
    ang = np.radians(np.minimum(np.random.rayleigh(dir_sigma_deg, n), dir_cap_deg))
    ca, sa = np.cos(ang)[:, None], np.sin(ang)[:, None]
    d_new = ca * d + sa * np.cross(ax, d) + (1 - ca) * (ax * d).sum(1, keepdims=True) * ax
    d_new /= np.linalg.norm(d_new, axis=1, keepdims=True) + 1e-9
    off = np.random.randn(n, 3)
    off -= (off * d_new).sum(1, keepdims=True) * d_new
    off *= (np.random.rayleigh(perp_sigma_m, n)
            / (np.linalg.norm(off, axis=1) + 1e-9))[:, None]
    m_new = np.cross(p0 + off, d_new)
    return np.concatenate([m_new, d_new], axis=1).astype(np.float32)


def _make_outliers(n: int, n_clusters: int = 5, spread: float = 0.2,
                   pos_range: float = 3.0) -> np.ndarray:
    if n == 0:
        return np.zeros((0, 6), np.float32)
    n_per   = max(1, n // n_clusters)
    extras  = n - n_per * n_clusters
    anchors = np.random.randn(n_clusters, 3).astype(np.float32)
    anchors /= np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-9
    parts = []
    for i, a in enumerate(anchors):
        cnt = n_per + (1 if i < extras else 0)
        d = a[None] + np.random.randn(cnt, 3).astype(np.float32) * spread
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        p = np.random.uniform(-pos_range, pos_range, (cnt, 3)).astype(np.float32)
        parts.append(np.concatenate([np.cross(p, d), d], axis=1).astype(np.float32))
    out = np.concatenate(parts, 0)
    return out[np.random.permutation(len(out))][:n]


def _make_plane_patch(n: int, pos_range: float = 3.0) -> np.ndarray:
    if n == 0:
        return np.zeros((0, 6), np.float32)
    normal = np.random.randn(3).astype(np.float64)
    normal /= np.linalg.norm(normal) + 1e-9
    perp = np.random.randn(3).astype(np.float64)
    perp -= perp.dot(normal) * normal
    u = perp / (np.linalg.norm(perp) + 1e-9)
    v = np.cross(normal, u)
    angles = np.random.uniform(0.0, 2.0 * np.pi, n)
    D = (np.outer(np.cos(angles), u) + np.outer(np.sin(angles), v)).astype(np.float32)
    D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-9
    a = np.random.uniform(-pos_range, pos_range, n)
    b = np.random.uniform(-pos_range, pos_range, n)
    P = (np.outer(a, u) + np.outer(b, v)).astype(np.float32)
    return np.concatenate([np.cross(P, D), D], axis=1).astype(np.float32)


def _make_wireframe(n: int, scale: float = 2.0) -> np.ndarray:
    if n == 0:
        return np.zeros((0, 6), np.float32)
    dims = np.random.uniform(0.3, 1.5, 3) * scale
    signs = np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],
                      [ 1,-1,-1],[ 1,-1,1],[ 1, 1,-1],[ 1, 1,1]], np.float64)
    corners = ((_random_rotation().astype(np.float64) @ (signs * dims[None]).T).T
               + np.random.uniform(-scale, scale, 3))
    edge_pairs = [(0,1),(2,3),(4,5),(6,7),(0,2),(1,3),(4,6),(5,7),(0,4),(1,5),(2,6),(3,7)]
    lines = []
    for a, b in edge_pairs:
        diff = corners[b] - corners[a]
        ln = np.linalg.norm(diff)
        if ln < 1e-9:
            continue
        d = (diff / ln).astype(np.float32)
        p = ((corners[a] + corners[b]) * 0.5).astype(np.float32)
        lines.append(np.concatenate([np.cross(p, d), d]))
    if not lines:
        return _make_outliers(n)
    arr = np.array(lines, np.float32)
    return arr[np.random.choice(len(arr), n, replace=(n > len(arr)))]


def _make_line_bundle(n: int, spread: float = 0.5, pos_range: float = 3.0) -> np.ndarray:
    if n == 0:
        return np.zeros((0, 6), np.float32)
    focus = np.random.uniform(-pos_range, pos_range, 3).astype(np.float32)
    D = np.random.randn(n, 3).astype(np.float32)
    D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-9
    offsets = np.random.randn(n, 3).astype(np.float32) * spread
    P = focus[None] + offsets - (offsets * D).sum(axis=1, keepdims=True) * D
    return np.concatenate([np.cross(P, D), D], axis=1).astype(np.float32)


def _make_parallel_group(n: int, spread: float = 0.35, pos_range: float = 3.0) -> np.ndarray:
    if n == 0:
        return np.zeros((0, 6), np.float32)
    anchor = np.random.randn(3).astype(np.float32)
    anchor /= np.linalg.norm(anchor) + 1e-9
    D = anchor[None] + np.random.randn(n, 3).astype(np.float32) * spread
    D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-9
    P = np.random.uniform(-pos_range, pos_range, (n, 3)).astype(np.float32)
    return np.concatenate([np.cross(P, D), D], axis=1).astype(np.float32)


def _make_grid_patch(n: int, pos_range: float = 3.0) -> np.ndarray:
    if n == 0:
        return np.zeros((0, 6), np.float32)
    normal = np.random.randn(3).astype(np.float64)
    normal /= np.linalg.norm(normal) + 1e-9
    perp = np.random.randn(3).astype(np.float64)
    perp -= perp.dot(normal) * normal
    u = (perp / (np.linalg.norm(perp) + 1e-9)).astype(np.float32)
    v = np.cross(normal, u).astype(np.float32)
    nh, nv = n // 2, n - n // 2
    Dh = np.tile(u, (nh, 1))
    Ph = (np.random.uniform(-pos_range, pos_range, (nh, 1)) * v[None]
          + np.random.uniform(-pos_range, pos_range, (nh, 1)) * u[None]).astype(np.float32)
    Dv = np.tile(v, (nv, 1))
    Pv = (np.random.uniform(-pos_range, pos_range, (nv, 1)) * u[None]
          + np.random.uniform(-pos_range, pos_range, (nv, 1)) * v[None]).astype(np.float32)
    return np.concatenate([
        np.concatenate([np.cross(Ph, Dh), Dh], axis=1),
        np.concatenate([np.cross(Pv, Dv), Dv], axis=1),
    ], axis=0).astype(np.float32)


def _make_staircase(n: int) -> np.ndarray:
    if n == 0:
        return np.zeros((0, 6), np.float32)
    R = _random_rotation().astype(np.float64)
    step_dir, up_dir, depth_dir = R[:, 0], R[:, 1], R[:, 2]
    step_h = np.random.uniform(0.15, 0.30)
    step_d = np.random.uniform(0.25, 0.40)
    origin = np.random.uniform(-2.0, 2.0, 3)
    # Cap at 15 steps so the staircase stays within ~3m displacement (bounded moments).
    # n lines are sampled with replacement from these fixed steps.
    n_steps = 15
    lines = []
    for i in range(n_steps):
        o = (origin + i * (step_h * up_dir + step_d * depth_dir)).astype(np.float32)
        d = step_dir.astype(np.float32)
        lines.append(np.concatenate([np.cross(o, d), d]))
        d2 = up_dir.astype(np.float32)
        lines.append(np.concatenate([np.cross(o, d2), d2]))
    arr = np.array(lines, np.float32)
    return arr[np.random.choice(len(arr), n, replace=(n > len(arr)))]


def _make_manhattan_world(n: int, pos_range: float = 3.0) -> np.ndarray:
    """Three orthogonal line families — mimics walls/floor/ceiling of real indoor rooms."""
    if n == 0:
        return np.zeros((0, 6), np.float32)
    R = _random_rotation()
    parts = []
    for i in range(3):
        ni = n // 3 + (1 if i < n % 3 else 0)
        if ni == 0:
            continue
        base = R[:, i].astype(np.float32)
        noise_std = float(np.random.uniform(0.0, 0.07))
        d = base[None] + np.random.randn(ni, 3).astype(np.float32) * noise_std
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        p = np.random.uniform(-pos_range, pos_range, (ni, 3)).astype(np.float32)
        parts.append(np.concatenate([np.cross(p, d), d], axis=1).astype(np.float32))
    return np.concatenate(parts).astype(np.float32)


def _make_corridor_pool(n: int, pos_range: float = 3.0) -> np.ndarray:
    """One or two dominant directions — mimics corridors, staircases, long hallways."""
    if n == 0:
        return np.zeros((0, 6), np.float32)
    R = _random_rotation()
    n_dirs = np.random.randint(1, 3)
    parts = []
    splits = np.random.dirichlet(np.ones(n_dirs)) * n
    for i, ni in enumerate(splits.astype(int)):
        if ni == 0:
            continue
        base = R[:, i % 3].astype(np.float32)
        noise_std = float(np.random.uniform(0.0, 0.10))
        d = base[None] + np.random.randn(ni, 3).astype(np.float32) * noise_std
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        p = np.random.uniform(-pos_range, pos_range, (ni, 3)).astype(np.float32)
        parts.append(np.concatenate([np.cross(p, d), d], axis=1).astype(np.float32))
    return np.concatenate(parts).astype(np.float32)


def _make_adversarial_pool(n: int, pos_range: float = 3.0) -> np.ndarray:
    """
    Three tight direction-cluster families aligned to orthogonal axes of a random frame.
    All clusters span the full spatial volume → the model must use moment context, not
    direction alone, to distinguish correspondences (mirrors wall/floor/ceiling ambiguity
    in structured indoor scenes).
    """
    if n == 0:
        return np.zeros((0, 6), np.float32)
    R = _random_rotation()
    parts = []
    for i in range(3):
        ni = n // 3 + (1 if i < n % 3 else 0)
        if ni == 0:
            continue
        base = R[:, i].astype(np.float32)
        spread = float(np.random.uniform(0.02, 0.09))   # tight cluster, but not perfectly parallel
        d = base[None] + np.random.randn(ni, 3).astype(np.float32) * spread
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        p = np.random.uniform(-pos_range, pos_range, (ni, 3)).astype(np.float32)
        parts.append(np.concatenate([np.cross(p, d), d], axis=1).astype(np.float32))
    pool = np.concatenate(parts, axis=0)
    return pool[np.random.permutation(len(pool))].astype(np.float32)


def _make_large_scale_pool(n: int, pos_range: float = 12.0) -> np.ndarray:
    """Building-scale outdoor pool: large planar facades + manhattan structures."""
    if n == 0:
        return np.zeros((0, 6), np.float32)
    n_mw = int(n * np.random.uniform(0.30, 0.65))
    n_plane = n - n_mw
    parts = []
    if n_mw > 0:
        parts.append(_make_manhattan_world(n_mw, pos_range=pos_range))
    if n_plane > 0:
        parts.append(_make_plane_patch(n_plane, pos_range=pos_range))
    pool = np.concatenate(parts, axis=0)
    return pool[np.random.permutation(len(pool))].astype(np.float32)


def _make_street_pool_dealias(n: int) -> np.ndarray:
    """v27 DE-ALIASED street pool: same slab geometry / direction mix /
    cross-modal weighting as the v6 pool, but with the alias sources removed
    (see the _DEALIAS comment at the top). Every structural element gets a
    distinct geometric identity: vertical sites are a collision-free Poisson
    process, facades belong to BUILDINGS with per-building depth and floor
    heights, ground edges are a handful of distinct curb/lane offsets."""
    if n == 0:
        return np.zeros((0, 6), np.float32)
    n_seg = np.random.randint(2, 6)   # found6: more yaw diversity -> real dir-conc 0.12-0.21
    ez = np.array([0.0, 0.0, 1.0])
    ex = np.array([1.0, 0.0, 0.0])
    lines, wts = [], []

    def add(p, d, w, Ryaw, org):
        d = Ryaw @ (np.asarray(d, np.float64) + np.random.randn(3) * 0.05)
        d /= np.linalg.norm(d) + 1e-9
        p = Ryaw @ np.asarray(p, np.float64) + org
        lines.append(np.concatenate([np.cross(p, d), d]))
        wts.append(w)

    org = np.zeros(3)
    yaw = 0.0
    per_seg = (2 * n) // n_seg + 1
    for _seg in range(n_seg):
        L = float(np.random.uniform(40.0, 250.0))
        W = float(np.random.uniform(3.0, 8.0))
        H = float(np.random.uniform(4.0, 10.0))
        c, si = np.cos(yaw), np.sin(yaw)
        Ryaw = np.array([[c, -si, 0.0], [si, c, 0.0], [0.0, 0.0, 1.0]])
        # collision-free vertical sites: Poisson-ish walk, min separation 2.5 m
        sites = []
        x = float(np.random.uniform(0.0, 6.0))
        while x < L:
            sites.append(x)
            x += max(5.5, np.random.lognormal(np.log(9.0), 0.5))
        sites = np.array(sites)
        np.random.shuffle(sites)
        site_i = 0
        # buildings: per-building facade depth offset and distinct floor heights
        blds = []
        for side in (-1.0, 1.0):
            x0 = 0.0
            while x0 < L:
                blen = float(np.random.uniform(8.0, 30.0))
                depth = W + float(np.random.uniform(-1.5, 1.5))
                floors = np.random.uniform(0.3, H, np.random.randint(2, 5))
                blds.append((side, x0, min(x0 + blen, L), depth, floors))
                x0 += blen
        # ground: few distinct curb/lane offsets for the whole segment
        n_g = np.random.randint(2, 5)
        gys = np.random.uniform(-W, W, n_g)
        made = 0
        # direction mix matched to REAL maps (dominant-direction concentration
        # 0.15-0.21): street-axis parallels cut 54% -> 26%, oblique clutter up
        # to 37% — at street scale the 3%-extent alias gate spans the whole
        # cross-section, so exact-parallel families are the alias mass and
        # only direction diversity (as in real curvature-edge maps) breaks it
        while made < per_seg:
            r = np.random.random()
            if r < 0.18 and site_i < len(sites):     # unique vertical site
                side = -1.0 if np.random.random() < 0.5 else 1.0
                # extra tilt jitter: real corner/pole lines lean (found6 —
                # verticals were the 0.30-0.42 direction-concentration mode
                # vs real maps' 0.12-0.21)
                add([sites[site_i], side * W * np.random.uniform(0.8, 1.15),
                     0.0], ez + np.random.randn(3) * 0.06, 1.0, Ryaw, org)
                site_i += 1
            elif r < 0.34:                            # facade horizontal
                side, xa, xb, depth, floors = blds[np.random.randint(len(blds))]
                add([np.random.uniform(xa, xb), side * depth,
                     np.random.uniform(0.3, H)], ex, 0.35, Ryaw, org)
            elif r < 0.38:                            # curb/lane edge, fresh offset
                add([np.random.uniform(0.0, L),
                     float(np.random.uniform(-W, W)),
                     0.0], ex, 0.8, Ryaw, org)
            elif r < 0.50:                            # cross-street horizontal
                side, xa, xb, depth, floors = blds[np.random.randint(len(blds))]
                add([float(np.random.uniform(xa, xb)), side * depth,
                     np.random.uniform(0.3, H)], [0.0, 1.0, 0.0], 0.5,
                    Ryaw, org)
            else:                                     # oblique clutter (37%)
                add([np.random.uniform(0.0, L), np.random.uniform(-W, W),
                     np.random.uniform(0.0, 0.6 * H)],
                    np.random.randn(3), 0.4, Ryaw, org)
            made += 1
        org = org + Ryaw @ np.array([L, 0.0, 0.0])
        yaw += np.deg2rad(np.random.uniform(70.0, 110.0)
                          * (1 if np.random.random() < 0.5 else -1))
    A = np.stack(lines).astype(np.float32)
    p0 = np.cross(A[:, :3], A[:, 3:])
    shift = -np.median(p0, axis=0)
    A[:, :3] += np.cross(shift[None].astype(np.float32), A[:, 3:])
    w = np.asarray(wts, np.float64)
    order = np.random.choice(len(A), size=n, replace=(n > len(A)),
                             p=w / w.sum())
    return A[order]


def _make_street_pool(n: int) -> np.ndarray:
    """v6 street-canyon pool, calibrated to the measured KITTI camera/LiDAR
    line maps (2026-07-16: extent ~(80-520) x (8-12) x (5-10) m slab,
    60-75% street-axis horizontal directions, 13-28% verticals, periodic
    facade repetition = the 180-deg flip alias, s_gt ~ 1.0).

    The returned order is a weighted shuffle with verticals and ground
    edges first: the inlier slice big_pool[:k] is therefore rich in the
    structures both sensors genuinely share as infinite lines (wall
    corners, curbs), while facade horizontals — whose height bands differ
    between LiDAR (bottom metres) and camera (full height) — land mostly
    in the outlier slices. This models the height-band disjointness that
    caps real cross-modal overlap at 18-24%."""
    if n == 0:
        return np.zeros((0, 6), np.float32)
    # 1-3 chained street segments (straight run, L-corner, U/Z route —
    # KITTI 06 is a ~520 m straight, 07 a 240x240 m loop). Corners spread
    # the foot points of street-axis lines across the plane, matching the
    # measured 20-60:1 in-plane slab anisotropy.
    n_seg = np.random.randint(1, 4)
    W = float(np.random.uniform(3.0, 8.0))        # half-width
    H = float(np.random.uniform(4.0, 10.0))       # facade height
    period = float(np.random.uniform(5.0, 15.0))  # facade repetition
    ez = np.array([0.0, 0.0, 1.0])
    lines, wts = [], []

    def add(p, d, w, Ryaw, org):
        d = Ryaw @ (np.asarray(d, np.float64) + np.random.randn(3) * 0.05)
        d /= np.linalg.norm(d) + 1e-9
        p = Ryaw @ np.asarray(p, np.float64) + org
        lines.append(np.concatenate([np.cross(p, d), d]))
        wts.append(w)

    org = np.zeros(3)
    yaw = 0.0
    per_seg = (2 * n) // n_seg + 1
    for _seg in range(n_seg):
        L = float(np.random.uniform(40.0, 250.0))
        c, si = np.cos(yaw), np.sin(yaw)
        Ryaw = np.array([[c, -si, 0.0], [si, c, 0.0], [0.0, 0.0, 1.0]])
        ex = np.array([1.0, 0.0, 0.0])
        made = 0
        while made < per_seg:
            r = np.random.random()
            side = -1.0 if np.random.random() < 0.5 else 1.0
            xg = (np.round(np.random.uniform(0.0, L) / period) * period
                  + np.random.randn() * 0.4)
            if r < 0.20:    # vertical wall corner / pole (shared cross-modally)
                add([xg, side * W * np.random.uniform(0.9, 1.1), 0.0],
                    ez, 1.0, Ryaw, org)
            elif r < 0.60:  # facade horizontal along the street axis
                add([np.random.uniform(0.0, L), side * W,
                     np.random.uniform(0.3, H)], ex, 0.35, Ryaw, org)
            elif r < 0.74:  # ground / curb / lane edges (shared)
                add([np.random.uniform(0.0, L),
                     np.random.uniform(-W, W), 0.0], ex, 0.8, Ryaw, org)
            elif r < 0.82:  # cross-street horizontal (facade returns, gates)
                add([xg, side * W, np.random.uniform(0.3, H)],
                    [0.0, 1.0, 0.0], 0.5, Ryaw, org)
            else:           # oblique clutter (vegetation, vehicles, LiDAR
                            # curvature edges) — real maps measure only
                            # 0.15-0.21 dominant-direction concentration
                add([np.random.uniform(0.0, L), np.random.uniform(-W, W),
                     np.random.uniform(0.0, 0.6 * H)],
                    np.random.randn(3), 0.4, Ryaw, org)
            made += 1
        # chain: advance to segment end, turn ~90 deg either way
        org = org + Ryaw @ np.array([L, 0.0, 0.0])
        yaw += np.deg2rad(np.random.uniform(70.0, 110.0)
                          * (1 if np.random.random() < 0.5 else -1))
    A = np.stack(lines).astype(np.float32)
    # recentre so moments stay O(extent), like a SLAM map's local frame
    p0 = np.cross(A[:, :3], A[:, 3:])
    shift = -np.median(p0, axis=0)
    A[:, :3] += np.cross(shift[None].astype(np.float32), A[:, 3:])
    w = np.asarray(wts, np.float64)
    order = np.random.choice(len(A), size=n, replace=False, p=w / w.sum())
    return A[order]


def _apply_drift(lines: np.ndarray, n_groups: int = 4, sigma: float = 0.10) -> np.ndarray:
    """
    Spatially correlated drift — simulates accumulated SLAM drift.
    Lines are partitioned into n_groups random spatial buckets; each bucket
    is rigidly translated by δ ~ N(0, σ²I), i.e. m += δ × d, which is the
    exact Plücker transform of a translation and preserves m·d = 0.
    """
    n = len(lines)
    if n == 0:
        return lines
    out = lines.copy()
    group_ids = np.random.randint(0, n_groups, n)
    for g in range(n_groups):
        mask = group_ids == g
        if mask.sum() == 0:
            continue
        delta = np.random.randn(3).astype(np.float32) * sigma
        out[mask, :3] += np.cross(delta[None], out[mask, 3:])
    return out


def _apply_traj_drift(lines: np.ndarray) -> np.ndarray:
    """
    FOUND drift model: smooth trajectory-correlated Sim(3) warp, matching the
    drift measured on real KITTI mono maps (build_kitti_gt_corr_v2: warp
    median 2-13 m over 100-400 m extents = 1-5% of extent, local scale
    p10/p90 ~ 0.95/1.09, slowly varying along the drive).

    Lines are parameterized by the projection tau of their perpendicular foot
    point onto a random axis (a proxy for "position along the trajectory").
    Piecewise-linear control knots define smooth rotation / scale /
    translation fields over tau; each line is warped by its local Sim(3)
    about the moving centre. Preserves m . d = 0 exactly.
    """
    n = len(lines)
    if n < 4:
        return lines
    m, d = lines[:, :3].astype(np.float64), lines[:, 3:].astype(np.float64)
    p0 = np.cross(d, m)                       # perpendicular foot points
    ctr = p0.mean(0)
    ext = float(np.abs(p0 - ctr).max()) + 1e-9
    u = np.random.randn(3); u /= np.linalg.norm(u) + 1e-12
    tau = (p0 - ctr) @ u
    tau = (tau - tau.min()) / (tau.max() - tau.min() + 1e-9)

    K = np.random.randint(4, 9)               # control knots
    kt = np.linspace(0.0, 1.0, K)
    # random-walk knot values (drift accumulates), zeroed at the start
    def walk(scale, dim):
        w = np.cumsum(np.random.randn(K, dim), axis=0) * scale
        return w - w[0]
    rot_amp   = np.radians(np.random.uniform(0.5, 4.0))
    scale_amp = np.random.uniform(0.02, 0.12)
    trans_amp = np.random.uniform(0.01, 0.05) * ext
    k_rot   = walk(rot_amp / np.sqrt(K), 3)         # axis-angle per knot
    k_scale = walk(scale_amp / np.sqrt(K), 1)[:, 0]  # log-scale per knot
    k_tr    = walk(trans_amp / np.sqrt(K), 3)

    w_rot = np.stack([np.interp(tau, kt, k_rot[:, i]) for i in range(3)], 1)
    sig   = np.exp(np.interp(tau, kt, k_scale))
    delta = np.stack([np.interp(tau, kt, k_tr[:, i]) for i in range(3)], 1)

    th = np.linalg.norm(w_rot, axis=1, keepdims=True) + 1e-12
    ax = w_rot / th
    ca, sa = np.cos(th), np.sin(th)
    def rot(v):                                # Rodrigues, rowwise
        return (ca * v + sa * np.cross(ax, v)
                + (1 - ca) * (ax * v).sum(1, keepdims=True) * ax)
    p_new = sig[:, None] * rot(p0 - ctr) + ctr + delta
    d_new = rot(d)
    d_new /= np.linalg.norm(d_new, axis=1, keepdims=True) + 1e-12
    m_new = np.cross(p_new, d_new)
    return np.concatenate([m_new, d_new], 1).astype(np.float32)


def _quantize_cloud(pl: np.ndarray, matches: np.ndarray, row: int):
    """FOUND: uniformly subsample a cloud to the largest _SIZE_GRID value <= n
    (cap 4096) so same-shape pairs batch at train time. Uniform (not
    inlier-preserving) so the inlier fraction stays honest; dropped matches
    are removed."""
    n = len(pl)
    g = int(_SIZE_GRID[_SIZE_GRID <= max(n, _SIZE_GRID[0])].max())
    g = min(g, n)
    if n == g:
        return pl, matches
    sel = np.sort(np.random.choice(n, g, replace=False))
    remap = -np.ones(n, np.int64); remap[sel] = np.arange(g)
    mi = matches.copy()
    if mi.shape[1] > 0:
        mi[row] = remap[mi[row]]
        mi = mi[:, mi[row] >= 0]
    return pl[sel], mi


def _make_confuser_lines(n: int, ref_lines: np.ndarray, spread: float = 0.08,
                         pos_range: float = 3.0,
                         moment_dup: bool = False) -> np.ndarray:
    """Outlier lines with directions near those in ref_lines — structurally
    confusing. With moment_dup=True (v10), the confuser is a near-DUPLICATE:
    same direction AND a moment close to the source line (small perpendicular
    foot-point offset), so its full 6D descriptor collides with a real line
    while it has NO true correspondence. This is the measured real-map
    failure — the learned descriptor produces spurious matches on
    locally-similar lines — so it forces global-context discrimination."""
    if n == 0 or len(ref_lines) == 0:
        return _make_outliers(n, pos_range=pos_range)
    idx = np.random.randint(0, len(ref_lines), n)
    d = ref_lines[idx, 3:].copy()
    d += np.random.randn(n, 3).astype(np.float32) * spread
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    if moment_dup:
        # near-duplicate: shift the source foot point by a modest perpendicular
        # offset (line stays "nearby and parallel" — a strong descriptor twin)
        m0, d0 = ref_lines[idx, :3], ref_lines[idx, 3:]
        p0 = np.cross(d0, m0)                          # foot of perpendicular
        off = np.random.randn(n, 3).astype(np.float32)
        off -= (off * d).sum(1, keepdims=True) * d
        off /= np.linalg.norm(off, axis=1, keepdims=True) + 1e-9
        off *= np.random.uniform(0.15, 0.7, (n, 1)).astype(np.float32)
        return np.concatenate([np.cross(p0 + off, d), d],
                              axis=1).astype(np.float32)
    p = np.random.uniform(-pos_range, pos_range, (n, 3)).astype(np.float32)
    return np.concatenate([np.cross(p, d), d], axis=1).astype(np.float32)


def _make_collision_pool(n: int, pos_range: float = 3.0) -> np.ndarray:
    """v10 dense-room pool: 2-4 wall/floor planes, each carrying many lines
    whose directions vary SMOOTHLY across a ~25deg arc (graded similarity,
    not discrete clusters) at dense positions — the locally-ambiguous,
    highly-similar line structure of a real cluttered room (chess/stairs),
    where the measured failure is learned-descriptor collision, NOT rotational
    symmetry (ref self-similarity is low, 0.01-0.03). Forces the network to
    separate near-identical lines by global moment context."""
    if n == 0:
        return np.zeros((0, 6), np.float32)
    n_planes = np.random.randint(2, 5)
    parts = []
    per = n // n_planes + 1
    for _ in range(n_planes):
        normal = np.random.randn(3); normal /= np.linalg.norm(normal) + 1e-9
        u = np.random.randn(3); u -= u.dot(normal) * normal
        u /= np.linalg.norm(u) + 1e-9
        v = np.cross(normal, u)
        # a dominant in-plane axis with graded angular spread (~25 deg arc)
        base_ang = np.random.uniform(0, 2 * np.pi)
        angs = base_ang + np.random.uniform(-0.22, 0.22, per)  # ~+-12.5 deg
        d = (np.cos(angs)[:, None] * u + np.sin(angs)[:, None] * v)
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        a = np.random.uniform(-pos_range, pos_range, per)
        b = np.random.uniform(-pos_range, pos_range, per)
        off = np.random.uniform(-0.3, 0.3, per)        # slight off-plane
        p = (a[:, None] * u + b[:, None] * v
             + off[:, None] * normal).astype(np.float32)
        parts.append(np.concatenate([np.cross(p, d), d.astype(np.float32)],
                                    axis=1).astype(np.float32))
    pool = np.concatenate(parts, axis=0)
    return pool[np.random.permutation(len(pool))].astype(np.float32)[:n]


def _make_building_pool(n: int) -> np.ndarray:
    """v37 INDOOR: a floor plan — one corridor spine with rooms opening off it.

    NCLT/ScanNet++ interiors are neither a single 3 m room nor a 250 m street:
    corridors run 20-100 m, rooms hang off them, and each room's walls are
    axis-aligned to ITS OWN frame (buildings are not perfectly Manhattan), so
    per-room yaw jitter is what breaks the global alias family that a single
    Manhattan frame would create."""
    if n == 0:
        return np.zeros((0, 6), np.float32)
    # FULL RANGE (not fitted): a "building" spans a small suite to a campus
    # wing, and its structural REGULARITY varies per pair (jit) so the family
    # covers tight-Manhattan through cluttered-irregular rather than centring
    # on any measured median.
    L = float(np.random.uniform(8.0, 150.0))          # corridor length
    W = float(np.random.uniform(1.0, 6.0))            # corridor half-width
    H = float(np.random.uniform(2.2, 6.0))            # ceiling height
    jit = float(np.random.uniform(0.01, 0.12))        # per-pair regularity
    ez = np.array([0.0, 0.0, 1.0]); ex = np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, 1.0, 0.0])
    lines = []
    def add(p, d):
        # jitter 0.04: real indoor maps measure dir-concentration 0.11-0.20 and
        # alias multiplicity ~1 (7-Scenes RGB-D refs), i.e. interiors are NOT
        # tight Manhattan once LSD/depth noise is included
        d = np.asarray(d, np.float64) + np.random.randn(3) * jit
        d /= np.linalg.norm(d) + 1e-9
        lines.append(np.concatenate([np.cross(np.asarray(p, np.float64), d), d]))
    # rooms along both sides, each with its own small yaw
    rooms = []
    x = 0.0
    while x < L:
        rl = float(np.random.uniform(3.0, 9.0))
        for side in (-1.0, 1.0):
            yaw = np.random.uniform(-0.06, 0.06)
            rooms.append((x, min(x + rl, L), side, yaw,
                          float(np.random.uniform(2.0, 15.0))))
        x += rl
    for _ in range(2 * n):
        r = np.random.random()
        if r < 0.13:                                   # corridor: wall corners
            side = -1.0 if np.random.random() < 0.5 else 1.0
            add([np.random.uniform(0, L), side * W, 0.0],
                ez + np.random.randn(3) * 0.05)
        elif r < 0.28:                                 # corridor floor/ceiling run
            side = -1.0 if np.random.random() < 0.5 else 1.0
            add([np.random.uniform(0, L), side * W,
                 np.random.choice([0.0, H])], ex)
        elif r < 0.55 and rooms:                       # room walls (own yaw)
            xa, xb, side, yaw, depth = rooms[np.random.randint(len(rooms))]
            c, sn = np.cos(yaw), np.sin(yaw)
            Ry = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
            u = np.random.random()
            base = [np.random.uniform(xa, xb), side * (W + np.random.uniform(0, depth)),
                    np.random.uniform(0, H)]
            d = (ez + np.random.randn(3) * 0.05) if u < 0.22 \
                else (Ry @ (ex if u < 0.60 else ey))
            add(base, d)
        else:                                          # doors, signage, clutter
            add([np.random.uniform(0, L), np.random.uniform(-W - 6, W + 6),
                 np.random.uniform(0, H)], np.random.randn(3))
    A = np.stack(lines).astype(np.float32)
    p0 = np.cross(A[:, :3], A[:, 3:])
    A[:, :3] += np.cross((-np.median(p0, 0))[None].astype(np.float32), A[:, 3:])
    return A[np.random.choice(len(A), n, replace=(n > len(A)))]


def _make_atrium_pool(n: int) -> np.ndarray:
    """v37 INDOOR: a tall open volume with railings/balconies repeated at floor
    heights — the NCLT campus atrium / ScanNet++ stairwell regime.  The floor
    repetition is deliberate structure (it is what makes these spaces alias
    vertically), but each floor gets its own inset so the copies are not
    identical."""
    if n == 0:
        return np.zeros((0, 6), np.float32)
    R = float(np.random.uniform(3.0, 45.0))            # atrium half-extent
    nf = np.random.randint(1, 9)                       # floors
    fh = float(np.random.uniform(2.5, 6.0))
    jit = float(np.random.uniform(0.01, 0.12))         # per-pair regularity
    ez = np.array([0.0, 0.0, 1.0])
    lines = []
    def add(p, d):
        d = np.asarray(d, np.float64) + np.random.randn(3) * jit
        d /= np.linalg.norm(d) + 1e-9
        lines.append(np.concatenate([np.cross(np.asarray(p, np.float64), d), d]))
    insets = np.random.uniform(0.0, 0.25, nf) * R
    for _ in range(2 * n):
        r = np.random.random()
        f = np.random.randint(nf)
        z = f * fh
        rr = R - insets[f]
        if r < 0.40:                                   # balcony railing runs
            th = np.random.uniform(0, 2 * np.pi)
            add([rr * np.cos(th), rr * np.sin(th), z + np.random.uniform(0.9, 1.1)],
                [-np.sin(th), np.cos(th), 0.0])
        elif r < 0.54:                                 # full-height columns
            th = np.random.uniform(0, 2 * np.pi)
            add([rr * np.cos(th), rr * np.sin(th), 0.0], ez)
        elif r < 0.70:                                 # floor slab edges
            th = np.random.uniform(0, 2 * np.pi)
            add([rr * np.cos(th), rr * np.sin(th), z],
                [-np.sin(th), np.cos(th), 0.0])
        else:                                          # stairs / clutter
            add([np.random.uniform(-R, R), np.random.uniform(-R, R),
                 np.random.uniform(0, nf * fh)], np.random.randn(3))
    A = np.stack(lines).astype(np.float32)
    p0 = np.cross(A[:, :3], A[:, 3:])
    A[:, :3] += np.cross((-np.median(p0, 0))[None].astype(np.float32), A[:, 3:])
    return A[np.random.choice(len(A), n, replace=(n > len(A)))]


def _make_cluttered_pool(n: int) -> np.ndarray:
    """v37 INDOOR: a room whose line budget is dominated by FURNITURE — many
    short edges in small locally-axis-aligned clusters at varied orientations.
    ScanNet++ laser scans resolve this clutter; an iPhone/mono pass sees only
    part of it, which is the cross-modal asymmetry this family supplies."""
    if n == 0:
        return np.zeros((0, 6), np.float32)
    ext = float(np.random.uniform(1.5, 25.0))
    H = float(np.random.uniform(2.0, 6.0))
    jit = float(np.random.uniform(0.01, 0.12))         # per-pair regularity
    lines = []
    def add(p, d):
        d = np.asarray(d, np.float64) + np.random.randn(3) * jit
        d /= np.linalg.norm(d) + 1e-9
        lines.append(np.concatenate([np.cross(np.asarray(p, np.float64), d), d]))
    n_obj = np.random.randint(3, 45)
    objs = []
    for _ in range(n_obj):
        ctr = np.array([np.random.uniform(-ext, ext), np.random.uniform(-ext, ext),
                        np.random.uniform(0.0, 1.2)])
        yaw = np.random.uniform(0, np.pi)
        c, sn = np.cos(yaw), np.sin(yaw)
        Ry = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
        w = np.random.randn(3) * np.random.uniform(0.02, 0.35)   # TILT per object
        th = np.linalg.norm(w) + 1e-12
        K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]) / th
        Rt = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
        objs.append((ctr, Rt @ Ry, np.random.uniform(0.3, 1.2, 3)))
    for _ in range(2 * n):
        if np.random.random() < 0.75 and objs:          # furniture edge
            ctr, Robj, size = objs[np.random.randint(len(objs))]
            axis = Robj @ np.eye(3)[np.random.randint(3)]
            off = Robj @ (np.random.uniform(-1, 1, 3) * size)
            add(ctr + off, axis)
        else:                                           # room shell
            u = np.random.random()
            if u < 0.5:
                side = np.random.choice([-1.0, 1.0]); ax = np.random.randint(2)
                p = [0.0, 0.0, np.random.uniform(0, H)]; p[ax] = side * ext
                p[1 - ax] = np.random.uniform(-ext, ext)
                add(p, np.eye(3)[2] if np.random.random() < 0.4 else np.eye(3)[1 - ax])
            else:
                add([np.random.uniform(-ext, ext), np.random.uniform(-ext, ext),
                     np.random.choice([0.0, H])], np.eye(3)[np.random.randint(2)])
    A = np.stack(lines).astype(np.float32)
    p0 = np.cross(A[:, :3], A[:, 3:])
    A[:, :3] += np.cross((-np.median(p0, 0))[None].astype(np.float32), A[:, 3:])
    return A[np.random.choice(len(A), n, replace=(n > len(A)))]


def _make_pool(n: int, pool_type: str = 'mixed') -> np.ndarray:
    if n == 0:
        return np.zeros((0, 6), np.float32)

    if pool_type == 'collision':
        return _make_collision_pool(n)

    if pool_type == 'building':
        return _make_building_pool(n)
    if pool_type == 'atrium':
        return _make_atrium_pool(n)
    if pool_type == 'cluttered':
        return _make_cluttered_pool(n)

    if pool_type in ('street', 'street_submap'):
        # v6: no shuffle here — _make_street_pool's weighted order IS the
        # inlier-selection bias (verticals/ground first)
        return (_make_street_pool_dealias(n) if _DEALIAS
                else _make_street_pool(n))

    if pool_type == 'manhattan':
        n_mw = int(n * np.random.uniform(0.55, 0.85))
        n_misc = n - n_mw
        parts = [_make_manhattan_world(n_mw)]
        if n_misc > 0:
            for maker, k in zip(
                [_make_plane_patch, _make_wireframe, _make_outliers],
                np.maximum(0, np.round(
                    np.random.dirichlet(np.ones(3)) * n_misc
                ).astype(int))
            ):
                if k > 0:
                    parts.append(maker(k))
        pool = np.concatenate(parts, axis=0)
        return pool[np.random.permutation(len(pool))].astype(np.float32)

    if pool_type == 'corridor':
        n_corr = int(n * np.random.uniform(0.55, 0.85))
        n_misc = n - n_corr
        parts = [_make_corridor_pool(n_corr)]
        if n_misc > 0:
            for maker, k in zip(
                [_make_staircase, _make_parallel_group, _make_outliers],
                np.maximum(0, np.round(
                    np.random.dirichlet(np.ones(3)) * n_misc
                ).astype(int))
            ):
                if k > 0:
                    parts.append(maker(k))
        pool = np.concatenate(parts, axis=0)
        return pool[np.random.permutation(len(pool))].astype(np.float32)

    if pool_type == 'adversarial':
        # Dominant adversarial clusters + small outlier fringe
        n_adv  = int(n * np.random.uniform(0.65, 0.90))
        n_misc = n - n_adv
        parts  = [_make_adversarial_pool(n_adv)]
        if n_misc > 0:
            parts.append(_make_outliers(n_misc))
        pool = np.concatenate(parts, axis=0)
        return pool[np.random.permutation(len(pool))].astype(np.float32)

    if pool_type == 'outdoor':
        pos_range = float(np.random.uniform(8.0, 20.0))
        return _make_large_scale_pool(n, pos_range=pos_range)

    # Default 'mixed' pool
    n_planes     = np.random.randint(0, 5)
    n_boxes      = np.random.randint(0, 4)
    n_bundles    = np.random.randint(0, 3)
    n_parallels  = np.random.randint(0, 3)
    n_grids      = np.random.randint(0, 3)
    n_staircases = np.random.randint(0, 3)
    tasks = ([_make_plane_patch]    * n_planes     +
             [_make_wireframe]      * n_boxes      +
             [_make_line_bundle]    * n_bundles    +
             [_make_parallel_group] * n_parallels  +
             [_make_grid_patch]     * n_grids      +
             [_make_staircase]      * n_staircases)
    if not tasks:
        return _make_outliers(n)
    fracs = np.random.dirichlet(np.ones(len(tasks) + 1))
    alloc = np.floor(fracs * n).astype(int)
    alloc[-1] += n - alloc.sum()
    parts = [maker(k) for maker, k in zip(tasks, alloc[:-1]) if k > 0]
    if alloc[-1] > 0:
        parts.append(_make_outliers(alloc[-1]))
    pool = np.concatenate(parts, axis=0)
    return pool[np.random.permutation(len(pool))].astype(np.float32)



# ── Scenario table ────────────────────────────────────────────────────────────
# The generator samples TWO independent things: a STRUCTURE (which pool of line
# primitives builds the scene) and a DIFFICULTY REGIME (overlap fraction, noise,
# cloud sizes).  They used to be entangled in a 193-line if/elif chain of 16
# hand-written branches, which invited "why these sixteen?" -- and three of them
# (building / atrium / cluttered) were byte-identical apart from the pool.
#
# The table below is that chain, transcribed.  Same distributions, declarative.
# The numbers are FITTED, not chosen -- overlap Betas and size ranges for the
# street family come from measured KITTI statistics (annotations kept inline).
# Do not "tidy" those; they are the calibration the zero-shot claim rests on.
#
#   size:  ('ratio', n2_lo, n2_hi, r_lo, r_hi, n1_min, n1_cap)
#              n2 ~ U[n2_lo,n2_hi];  n1 = clip(n2 * U[r_lo,r_hi], n1_min, n1_cap)
#          ('indep', n1_lo, n1_hi, n2_lo, n2_hi)      both drawn independently
#          ('equal', lo, hi)                          n1 == n2
#          ('logbase', b_lo, b_hi, r_lo, r_hi)        log-uniform base, ratio split
#          ('nested', n1_lo, n1_hi, n2_lo, n2_hi)     n2 >= n1 (submap in a map)
#   ovl:   Beta(a, b) on the matched fraction of min(n1, n2)
#   nz1/nz2: log-uniform noise (metres) on query / reference; a scalar means fixed

def _street_sizes():
    """The street family's sizes/overlap are flag-dependent and CALIBRATED."""
    if _DEALIAS:   return ('indep', 500, 3500, 800, 6000), (2.5, 7.5)    # mf med ~0.17
    if _FULL:      return ('indep', 500, 3500, 800, 6000), (1.7, 20.0)   # ~8%, bounds KITTI 2-8
    if _KITTI_MONO:return ('indep', 900, 3300, 1300, 5600), (2.0, 30.0)  # real 4.5-8%
    return ('ratio', 600, 2200, 0.4, 1.6, 200, 2200), (2.0, 7.0)         # mean ~0.22


def _street_submap_sizes():
    if _SUBCAL:    return ('indep', 100, 850, 800, 6000), (2.4, 6.8)     # real 27-62%
    if _FULL:      return ('indep', 250, 1400, 800, 6000), (1.7, 18.0)   # ~9%
    if _KITTI_MONO:return ('indep', 300, 1200, 1300, 5600), (2.0, 26.0)  # mean ~0.07
    return ('indep', 150, 600, 800, 2200), (2.0, 6.0)                    # mean ~0.25


_INDOOR_CLEAN = dict(nz1=(0.005, 0.12), nz2=(0.001, 0.04))
_BUILDING     = dict(size=('ratio', 120, 4096, 0.08, 2.0, 64, 4096),
                     ovl=(1.3, 3.0), nz1=(0.005, 0.35), nz2=(0.001, 0.08))

_SPEC = {
    'room':        dict(pool='mixed',   size=('ratio', 80, 1100, 0.3, 3.0, 30, 1100),
                        ovl=(3.0, 2.0), **_INDOOR_CLEAN),
    'submap':      dict(pool='mixed',   size=('nested', 30, 150, 80, 700),
                        ovl=(2.5, 3.5), nz1=(0.010, 0.15), nz2=(0.001, 0.04)),
    'relocalize':  dict(pool='mixed',   size=('logbase', 30, 400, 0.5, 2.0),
                        ovl=(2.5, 2.5), nz1=(0.005, 0.10), nz2='same'),
    'loop':        dict(pool='mixed',   size=('equal', 40, 500),
                        ovl=(5.0, 2.0), nz1=(0.003, 0.06), nz2='same'),
    'dense_sparse':dict(pool='mixed',   size=('ratio', 150, 700, 0.10, 0.50, 30, 10**9),
                        ovl=(2.5, 2.0), nz1=(0.005, 0.12), nz2=(0.001, 0.03)),
    'manhattan':   dict(pool='manhattan', size=('ratio', 80, 900, 0.2, 2.5, 30, 900),
                        ovl=(3.0, 2.5), **_INDOOR_CLEAN),
    'corridor':    dict(pool='corridor', size=('ratio', 50, 450, 0.2, 1.8, 30, 450),
                        ovl=(2.0, 3.0), nz1=(0.010, 0.15), nz2=(0.002, 0.05)),
    'hard_noise':  dict(pool='mixed',   size=('ratio', 80, 600, 0.3, 2.0, 30, 10**9),
                        ovl=(1.5, 6.0), nz1=(0.05, 0.50), nz2=(0.01, 0.10), conf=True),
    'adversarial': dict(pool='adversarial', size=('ratio', 80, 800, 0.3, 2.5, 30, 800),
                        ovl=(3.0, 2.5), conf=True, **_INDOOR_CLEAN),
    'collision':   dict(pool='collision', size=('ratio', 300, 1100, 0.4, 2.0, 80, 1100),
                        ovl=(2.5, 3.0), conf=True, **_INDOOR_CLEAN),
    'outdoor':     dict(pool='outdoor', size=('ratio', 60, 500, 0.2, 2.0, 30, 500),
                        ovl=(2.5, 2.0), nz1=(0.02, 0.60), nz2=(0.005, 0.20)),
    'street':      dict(pool='street',  size=None, ovl=None, nz1=0.3, nz2=0.1),
    'street_submap':dict(pool='street_submap', size=None, ovl=None, nz1=0.3, nz2=0.1),
    'street_mono': dict(pool='street',  size=('indep', 300, 3300, 300, 5600),
                        ovl=(2.0, 9.0), nz1=0.3, nz2=0.1),
    'building':    dict(pool='building',  **_BUILDING),
    'atrium':      dict(pool='atrium',    **_BUILDING),
    'cluttered':   dict(pool='cluttered', **_BUILDING),
}


def _draw_sizes(spec):
    kind = spec[0]
    if kind == 'ratio':
        _, n2lo, n2hi, rlo, rhi, n1min, n1cap = spec
        n2 = np.random.randint(n2lo, n2hi)
        return max(n1min, min(int(n2 * np.random.uniform(rlo, rhi)), n1cap)), n2
    if kind == 'indep':
        _, a, b, c, d = spec
        return np.random.randint(a, b), np.random.randint(c, d)
    if kind == 'equal':
        n = np.random.randint(spec[1], spec[2]);  return n, n
    if kind == 'logbase':
        _, blo, bhi, rlo, rhi = spec
        base = int(np.exp(np.random.uniform(np.log(blo), np.log(bhi))))
        r = np.random.uniform(rlo, rhi)
        return max(30, int(base * r)), max(30, int(base / r))
    if kind == 'nested':
        _, a, b, c, d = spec
        n1 = np.random.randint(a, b)
        return n1, np.random.randint(max(n1, c), d)
    raise ValueError(spec)


def _logu(rng):
    return float(np.exp(np.random.uniform(np.log(rng[0]), np.log(rng[1]))))


def _sample_scenario():
    """-> (scenario, n1, n2, overlap_frac, noise1, noise2, pool_type, confusers)"""
    name = np.random.choice(_SCENARIOS, p=_SCENARIO_P)
    sp = _SPEC[name]
    size, ovl = sp['size'], sp['ovl']
    if name == 'street':          size, ovl = _street_sizes()
    elif name == 'street_submap': size, ovl = _street_submap_sizes()
    n1, n2 = _draw_sizes(size)
    nz1 = sp['nz1'] if np.isscalar(sp['nz1']) else _logu(sp['nz1'])
    nz2 = nz1 if sp['nz2'] == 'same' else (
        sp['nz2'] if np.isscalar(sp['nz2']) else _logu(sp['nz2']))
    return (name, n1, n2, float(np.random.beta(*ovl)), nz1, nz2,
            sp['pool'], sp.get('conf', False))


# ── Pair generator ────────────────────────────────────────────────────────────

def _generate_pair() -> dict:
    (scenario, n1, n2, overlap_frac, noise1, noise2,
     pool_type, use_confusers) = _sample_scenario()

    k  = max(_MIN_INLIERS, int(overlap_frac * min(n1, n2)))
    n1 = max(n1, k)
    n2 = max(n2, k)

    # v4: scale distribution matched to measured real pairs (median 2.1).
    # FULL: lift the base (indoor/general) scale ceiling so scale is a smooth
    # continuum into the street regime (no indoor/street bimodal gap).
    _sc_std = (1.0 if _FOUND else 0.75) if _FULL else _SCALE_LOG_STD
    _sc_hi  = (32.0 if _FOUND else 16.0) if _FULL else _SCALE_CLIP[1]
    _sc_lo  = 0.25 if _FOUND else _SCALE_CLIP[0]
    log_s = np.random.normal(_SCALE_LOG_CENTER, _sc_std)
    log_s = np.clip(log_s, np.log(_sc_lo), np.log(_sc_hi))
    s   = float(np.exp(log_s))
    R   = _pair_rotation()          # ROT_CAL env -> real-calibrated (median 29deg) vs Haar
    if scenario == 'street_mono':
        # aligned world frames (mono SLAM vs LiDAR, common gravity-up trajectory):
        # SMALL inter-map rotation (measured 0.6-6 deg on KITTI mono_best), NOT the
        # FOUND uniform-SO(3) scramble. This is the joint (small-rot, large-scale)
        # regime the base rotation draw under-covers.
        ax = np.random.randn(3); ax /= np.linalg.norm(ax) + 1e-12
        ang = np.radians(min(float(np.random.rayleigh(3.0)), 12.0))
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        R = (np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)).astype(np.float32)
    # v2 fix: t_range_eff = 0.4 * s so that the inverse translation
    # |t_i| = |t|/s ≤ 0.4m for ALL s (both s < 1 and s > 1).
    # Prevents the t × d cross term from inflating query moments.
    t_range_eff = 0.4 * s
    if scenario == 'street':
        if _FULL:
            # full-range mono->LiDAR: scale BOUNDED [4,48] with margin (not fitted
            # to the KITTI 8-39 measurement); translation street-scale in ref metres.
            _st_std, _st_lo, _st_hi = ((0.9, 3.0, 100.0) if _FOUND
                                       else (0.80, 4.0, 48.0))
            s = float(np.exp(np.clip(np.random.normal(np.log(12.0), _st_std),
                                     np.log(_st_lo), np.log(_st_hi))))
            t_range_eff = 10.0 * s
        elif _KITTI_MONO:
            # mono->LiDAR: the query is a monocular SLAM map at ARBITRARY large
            # scale (s_gt measured 8-39 on KITTI 03/05/07/10); the reference is
            # the metric LiDAR map. Translation is street-scale in ref metres.
            s = float(np.exp(np.clip(np.random.normal(np.log(15.0), 0.5),
                                     np.log(7.0), np.log(42.0))))
            t_range_eff = 10.0 * s
        else:
            # v6: both maps metric (LiDAR GT frame vs depth-lifted camera map)
            # — s_gt measured 0.999/1.003 on the old camlines->LiDAR setup.
            s = float(np.exp(np.clip(np.random.normal(0.0, 0.12),
                                     np.log(0.7), np.log(1.4))))
            t_range_eff = 20.0 * s
    elif scenario == 'street_submap':
        if _FULL:
            _st_std, _st_lo, _st_hi = ((0.9, 3.0, 100.0) if _FOUND
                                       else (0.80, 4.0, 48.0))
            s = float(np.exp(np.clip(np.random.normal(np.log(12.0), _st_std),
                                     np.log(_st_lo), np.log(_st_hi))))
            t_range_eff = 10.0 * s
        elif _KITTI_MONO:
            s = float(np.exp(np.clip(np.random.normal(np.log(15.0), 0.5),
                                     np.log(7.0), np.log(42.0))))
            t_range_eff = 10.0 * s
        else:
            # v7: a standalone mono submap has ARBITRARY scale (unlike the
            # depth-lifted full camera maps) and lives in its own local frame
            s = float(np.exp(np.clip(np.random.normal(0.0, 0.45),
                                     np.log(1 / 3.0), np.log(3.0))))
            t_range_eff = 20.0 * s
    elif scenario == 'street_mono':
        # arbitrary large mono scale (real KITTI mono_best s_gt 9-39); translation
        # street-scale in reference metres.
        s = float(np.exp(np.clip(np.random.normal(np.log(16.0), 0.45),
                                 np.log(8.0), np.log(42.0))))
        t_range_eff = 10.0 * s
    if scenario in ('building', 'atrium', 'cluttered'):
        # HALF the indoor pairs are metric<->metric (LiDAR vs laser scan:
        # NCLT-to-NCLT, ScanNet++ laser-to-iPhone-with-depth, s ~ 1) and half
        # are mono at unknown scale (s 2-40) — covering both cells of the
        # indoor modality study with margin rather than fitting either.
        # FULL RANGE: log-uniform over the whole 0.25-100 continuum, with a
        # 30% bump at metric<->metric (s~1, LiDAR-to-laser) so both the
        # same-units and unknown-scale cells are covered without fitting either
        if np.random.random() < 0.30:
            s = float(np.exp(np.random.uniform(np.log(0.7), np.log(1.4))))
        elif _SOUND:
            # same base law as room/manhattan/corridor -- these scenarios differ
            # from them only in the line POOL, so they share the regime.
            s = float(np.exp(np.clip(np.random.normal(_SCALE_LOG_CENTER, 1.0),
                                     np.log(0.25), np.log(100.0))))
        else:
            s = float(np.exp(np.random.uniform(np.log(0.25), np.log(100.0))))
        t_range_eff = float(np.random.uniform(0.1, 1.0)) * s
    t   = np.random.uniform(-t_range_eff, t_range_eff, 3).astype(np.float32)
    s_i, R_i, t_i = 1.0 / s, R.T, -(R.T @ t) / s

    n_out1    = max(0, n1 - k)
    n_out2    = max(0, n2 - k)
    pool_size = k + n_out1 + n_out2 + max(k // 4, 8)
    # Allow near-parallel configurations (lowered from 0.30 so the model
    # sees the corridor/manhattan distributions it will face at test time).
    _NONDEGEN_THR = 0.05
    for _attempt in range(10):
        big_pool = _make_pool(pool_size, pool_type=pool_type)
        dirs = big_pool[:, 3:]
        sv = np.linalg.svd(dirs - dirs.mean(0, keepdims=True), compute_uv=False)
        if sv[-1] / (sv[0] + 1e-9) >= _NONDEGEN_THR:
            break

    if pool_type == 'street_submap':
        # v7: confine ALL query material (inliers + outliers) to one
        # contiguous spatial window of the street; the reference keeps the
        # whole route. A ball around a route point is a segment of the 1-D
        # street corridor. Partition order: [window | non-window | leftover
        # window] so the query slice is pure window and the ref-outlier
        # slice is dominated by the rest of the street.
        p0 = np.cross(big_pool[:, :3], big_pool[:, 3:])
        ctr = np.median(p0, axis=0)
        ext = float(np.linalg.norm(p0 - ctr, axis=1).max()) + 1e-6
        anchor = p0[np.random.randint(len(p0))]
        rho = np.random.uniform(0.12, 0.30) * ext
        win = np.linalg.norm(p0 - anchor, axis=1) < rho
        while win.sum() < 3 * _MIN_INLIERS and rho < 2.5 * ext:
            rho *= 1.4
            win = np.linalg.norm(p0 - anchor, axis=1) < rho
        n_win = int(win.sum())
        k = max(_MIN_INLIERS, min(k, int(0.7 * n_win)))
        n_out1 = min(n_out1, max(0, n_win - k))
        wi = np.where(win)[0]
        ni = np.where(~win)[0]
        big_pool = big_pool[np.concatenate([wi[:k + n_out1], ni,
                                            wi[k + n_out1:]])]
        n_out2 = min(n_out2, len(big_pool) - k - n_out1)

    inliers_ref = big_pool[:k].copy()
    out_q_ref   = big_pool[k:k + n_out1]
    out_r       = big_pool[k + n_out1:k + n_out1 + n_out2]

    # v4: per-pair severity replaces the per-scenario Gaussian noise levels;
    # hard scenarios are drifty runs (higher severity), not a different model.
    # BROAD: widen the per-pair noise severity to a SET range covering clean->very
    # noisy (dir sigma ~1.6-15.6 deg, perp ~2-21 cm) so the matcher generalizes
    # across the whole realistic noise regime rather than a calibrated band.
    _sev_range = (0.3, 3.0) if _BROAD else _V4_SEVERITY
    severity = np.random.uniform(*_sev_range) * (2.0 if scenario == 'hard_noise' else 1.0)
    # v6: street positional noise is measured in street metres (KITTI GT
    # correspondences: perp median 0.32-0.35 m -> Rayleigh sigma ~0.28;
    # LiDAR reference edges are range-noisy too). Direction noise keeps the
    # v4 calibration (measured 5.4-6.0 deg median on KITTI — same as indoor).
    _street = scenario in ('street', 'street_submap', 'street_mono')
    perp_q_sigma = 0.28 if _street else _V4_PERP_SIGMA_M
    perp_r_sigma = 0.08 if _street else _V4_REF_PERP_M
    drift_scale  = 6.0 if _street else 1.0

    if k > 0:
        # v4 fragmentation: each reference edge appears as 1–3 query fragments
        # with independent noise draws → many-to-one matches + near-duplicate
        # query lines, matching real LSD mono maps.
        n_frag = np.random.choice([1, 2, 3], size=k, p=_V4_FRAG_P)
        frag_src = np.repeat(np.arange(k), n_frag)      # fragment -> ref inlier
        inliers_q = _apply_sim3(inliers_ref[frag_src], s_i, R_i, t_i)
        # noise in the query frame; perpendicular noise is measured in the
        # METRIC frame, so divide by s
        inliers_q = _physical_noise(inliers_q,
                                    _V4_DIR_SIGMA_DEG * severity,
                                    perp_q_sigma * severity / s,
                                    _V4_DIR_CAP_DEG)
        inliers_ref = _physical_noise(inliers_ref, _V4_REF_DIR_DEG,
                                      perp_r_sigma)

        # Drift augmentation (20% of pairs): spatially correlated moment noise on query.
        # FOUND: replaced by the whole-map trajectory drift applied below.
        if not _FOUND and np.random.random() < 0.20:
            inliers_q = _apply_drift(
                inliers_q,
                n_groups=np.random.randint(2, 6),
                sigma=float(np.random.uniform(0.05, 0.25)) * drift_scale,
            )
    else:
        inliers_q = inliers_ref = np.zeros((0, 6), np.float32)
        frag_src = np.zeros(0, np.int64)
    k_q = len(frag_src)

    if use_confusers and k > 0:
        # Replace a fraction of regular outliers with confuser lines (near-parallel to inliers).
        n_conf = max(0, int(n_out1 * np.random.uniform(0.3, 0.7)))
        n_reg  = n_out1 - n_conf
        out_q_reg  = (_apply_sim3(out_q_ref[:n_reg], s_i, R_i, t_i) if n_reg > 0
                      else np.zeros((0, 6), np.float32))
        out_q_conf = _make_confuser_lines(n_conf, inliers_q, moment_dup=(scenario == 'collision')) if n_conf > 0 else np.zeros((0, 6), np.float32)
        out_q_parts = [p for p in [out_q_reg, out_q_conf] if len(p) > 0]
        out_q = np.concatenate(out_q_parts) if out_q_parts else np.zeros((0, 6), np.float32)
    else:
        out_q = (_apply_sim3(out_q_ref, s_i, R_i, t_i) if n_out1 > 0
                 else np.zeros((0, 6), np.float32))
    # v4: outliers are real scene lines too — same physical noise as inliers
    out_q = _physical_noise(out_q, _V4_DIR_SIGMA_DEG * severity,
                            perp_q_sigma * severity / s, _V4_DIR_CAP_DEG)

    # v22 GHOST: convert a fraction of query outliers into near-miss ghosts
    # (displaced copies of true query-frame structure, unlabeled)
    if _GHOST and len(out_q) > 4:
        src = np.concatenate([inliers_q, out_q]) if len(inliers_q) else out_q
        n_g = int(np.random.uniform(0.25, 0.55) * len(out_q))
        if n_g > 0:
            out_q = np.concatenate([out_q[:len(out_q) - n_g],
                                    _make_ghost_lines(src, n_g)])

    q_parts = [p for p in [inliers_q, out_q]   if len(p) > 0]
    r_parts = [p for p in [inliers_ref, out_r] if len(p) > 0]
    query_all = (np.concatenate(q_parts, axis=0) if q_parts else _make_pool(max(n1, 10), pool_type))
    ref_all   = (np.concatenate(r_parts, axis=0) if r_parts else _make_pool(max(n2, 10), pool_type))

    # FOUND: smooth trajectory-correlated Sim(3) drift over the WHOLE query
    # map (35% of pairs) — a coherent map-level warp, matching real mono SLAM
    # drift; correspondences stay labeled (the point is noise tolerance).
    if _FOUND and np.random.random() < 0.35:
        query_all = _apply_traj_drift(query_all)

    # No sign-flip augmentation: the Grassmannian encoder is sign-invariant
    # BY CONSTRUCTION (measured 0.000e+00), so flipping only burned capacity.
    # Do not re-add.

    i_q = np.random.permutation(len(query_all))
    i_r = np.random.permutation(len(ref_all))
    query_all, ref_all = query_all[i_q], ref_all[i_r]
    q_pos = np.argsort(i_q)[:k_q]          # fragment j -> row in query_all
    r_pos = np.argsort(i_r)[:k]            # ref inlier i -> row in ref_all

    # v4 convention swap: plucker1 = REFERENCE (metric), plucker2 = QUERY —
    # same as the 7scenes_mesh / real generators, so datasets mix cleanly and
    # the dataloader's "normalize by plucker2 (query) std" stays correct.
    # (s, R, t) maps query -> reference, i.e. plucker2 -> plucker1.
    m_ref = r_pos[frag_src].astype(np.int32)
    m_qry = q_pos.astype(np.int32)
    matches = (np.stack([m_ref, m_qry], 0) if k_q > 0
               else np.zeros((2, 0), np.int32))

    if _FOUND:   # snap both clouds to the batching size grid
        ref_all, matches = _quantize_cloud(ref_all, matches, 0)
        query_all, matches = _quantize_cloud(query_all, matches, 1)

    return dict(
        plucker1 = ref_all.astype(np.float32),
        plucker2 = query_all.astype(np.float32),
        matches  = matches.astype(np.int32),
        R_gt     = R.astype(np.float32),
        t_gt     = t.reshape(3, 1).astype(np.float32),
        s_gt     = np.float32(s),
    )


# ── Worker (runs in child process) ────────────────────────────────────────────

def _worker(args):
    n, seed = args
    np.random.seed(seed)
    keys  = ['matches', 'plucker1', 'plucker2', 'R_gt', 't_gt', 's_gt']
    chunk = {k: [] for k in keys}
    for _ in range(n):
        pair = _generate_pair()
        for k in keys:
            chunk[k].append(pair[k])
    return chunk


# ── Dataset saving ────────────────────────────────────────────────────────────

def _save(data: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for k, v in data.items():
        with open(os.path.join(out_dir, f'{k}.pkl'), 'wb') as f:
            pickle.dump(v, f, protocol=4)
    s  = np.array(data['s_gt'])
    n1 = np.array([p.shape[0] for p in data['plucker1']])
    n2 = np.array([p.shape[0] for p in data['plucker2']])
    ki = np.array([m.shape[1] for m in data['matches']])
    print(f'  Saved {len(data["t_gt"])} pairs → {out_dir}')
    print(f'    scale   : [{s.min():.3f}, {s.max():.3f}]  median {np.median(s):.3f}')
    print(f'    n1      : [{n1.min()}, {n1.max()}]  median {int(np.median(n1))}')
    print(f'    n2      : [{n2.min()}, {n2.max()}]  median {int(np.median(n2))}')
    print(f'    inliers : [{ki.min()}, {ki.max()}]  median {int(np.median(ki))}')


# ── Main generation loop ──────────────────────────────────────────────────────

def generate_split(n_pairs: int, workers: int, chunk_size: int,
                   seed: int, label: str) -> dict:
    n_chunks  = max(1, (n_pairs + chunk_size - 1) // chunk_size)
    remaining = n_pairs
    tasks     = []
    for i in range(n_chunks):
        n = min(chunk_size, remaining)
        tasks.append((n, seed + i * 7919))
        remaining -= n
        if remaining <= 0:
            break

    keys     = ['matches', 'plucker1', 'plucker2', 'R_gt', 't_gt', 's_gt']
    combined = {k: [] for k in keys}
    t0, done = time.time(), 0

    if workers == 1:
        for task in tasks:
            chunk = _worker(task)
            for k in keys:
                combined[k].extend(chunk[k])
            done += task[0]
            elapsed = time.time() - t0
            print(f'  {label}: {done}/{n_pairs}  ({done/elapsed:.0f} pairs/s)', end='\r')
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for chunk in ex.map(_worker, tasks):
                for k in keys:
                    combined[k].extend(chunk[k])
                done += chunk_size
                elapsed = time.time() - t0
                print(f'  {label}: {min(done, n_pairs)}/{n_pairs}  '
                      f'({done/elapsed:.0f} pairs/s)', end='\r')

    print()
    return combined


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--n_train',    type=int, default=200_000)
    ap.add_argument('--n_valid',    type=int, default=2_000)
    ap.add_argument('--out_dir',    default=str(_ROOT / 'dataset'))
    ap.add_argument('--name',       default='synthetic_v3',
                    help='Dataset name prefix (default: synthetic_v3). '
                         'Outputs to <out_dir>/<name>_train and <out_dir>/<name>_valid.')
    ap.add_argument('--chunk_size', type=int, default=2_000)
    ap.add_argument('--workers',    type=int,
                    default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument('--seed',       type=int, default=42)
    args = ap.parse_args()

    print(f'Generating synthetic dataset  '
          f'train={args.n_train:,}  valid={args.n_valid:,}  '
          f'workers={args.workers}  seed={args.seed}')
    print(f'Output: {args.out_dir}  name: {args.name}')
    if _FULL:
        print(f'Mode: FULL (full-range) — scale continuum 0.4-48 (base clip 16 + street '
              f'[4,48]), street overlap Beta(1.7,20) bounded-not-fit, BROAD rotation/severity, '
              f'all {len(_SCENARIOS)} scenarios')
        print(f'Flags: FOUND={_FOUND} INDOOR={_INDOOR} MONO={_MONO} GHOST={_GHOST} '
              f'SUBCAL={_SUBCAL} DEALIAS={_DEALIAS} KITTI_MONO={_KITTI_MONO} '
              f'SOUND={_SOUND}')
    else:
        print(f'Scale distribution: LogNormal(log({np.exp(_SCALE_LOG_CENTER):.1f}), {_SCALE_LOG_STD}) '
              f'clipped to {_SCALE_RANGE}   [BROAD={_BROAD} KITTI_MONO={_KITTI_MONO}]')

    print('\n=== Training split ===')
    _save(generate_split(args.n_train, args.workers, args.chunk_size,
                         seed=args.seed, label='train'),
          os.path.join(args.out_dir, f'{args.name}_train'))

    print('\n=== Validation split ===')
    _save(generate_split(args.n_valid, args.workers, args.chunk_size,
                         seed=args.seed + 999_983, label='valid'),
          os.path.join(args.out_dir, f'{args.name}_valid'))

    print('\nDone.')


if __name__ == '__main__':
    main()
