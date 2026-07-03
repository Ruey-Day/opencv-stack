# ScalePluckerNet

Extends [PlueckerNet](https://github.com/Liumouliu/PlueckerNet) (Liu et al., CVPR 2021) from **SE(3)** to **Sim(3)** — jointly recovering rotation R, translation t, *and scale s* from Plücker line correspondences.

**Problem:** when matching monocular SLAM line landmarks (scale-ambiguous) to metric RGB-D map lines, the SE(3) RANSAC structurally fails — it absorbs the unknown scale into a spurious translation, giving wrong pose estimates even when correspondences are correct.

**Solution:** a minimal two-correspondence Sim(3) solver (SVD for rotation, joint LS for scale+translation) replacing the SE(3) backend. The PluckerNetKnn correspondence network is used without modification — scale enters only through moment magnitudes, which the Sinkhorn matcher learns to ignore.

---

## Dependencies

```bash
conda activate torch5090   # Python 3.11, PyTorch 2.6, CUDA
```

`../PlueckerNet/` must exist at the same directory level as this repo — all entry points add it to `sys.path` automatically.

---

## Plücker Line Format

All Sim3 code uses **`[m, d]` order** — moment first, direction last:

```
line = [m0, m1, m2, d0, d1, d2]   shape (6,) or (N, 6)
```

Transformation law under Sim(3) with scale s, rotation R, translation t:
```
d' = R d
m' = s·R·m + t × d'
```

The original SE(3) PlueckerNet uses `[d, m]` order (direction first). Use `md_to_dm()` when passing lines into the original PlueckerNet code.

---

## Workflow

There are two ways to build training data, both producing the same pkl format:

| Route | Script | Best for |
|-------|--------|---------|
| Add random scale to an existing SE3 pkl dataset | `scripts/make_sim3_from_se3.py` | Quickly extend any SE3 dataset you already have |
| Generate a fully-synthetic dataset from scratch | `scripts/generate_synthetic_large.py` | Large, diverse, map-free training |

---

## Route 1 — Add Random Scale to an SE3 Dataset

`scripts/make_sim3_from_se3.py` takes any SE3 pkl split and applies a random log-uniform scale to each sample, producing a valid Sim(3) pair.

**What it does mathematically:**

For each sample the script draws `s ~ log-uniform[scale_min, scale_max]` and rescales plucker1 moments:
```
m1_sim3 = s·(m1 - t×d1) + t×d1
```
This is exact: if `m1_se3 = R·m2 + t×d1` then `m1_sim3 = s·(R·m2) + t×d1`. Directions are unchanged (scale does not affect directions). A `s_gt.pkl` file is added to the output.

**Usage:**
```bash
python scripts/make_sim3_from_se3.py \
    --src dataset/structured3D_valid \
    --dst dataset/structured3D_sim3_valid

python scripts/make_sim3_from_se3.py \
    --src dataset/semantic3D_train \
    --dst dataset/semantic3D_sim3_train

# Custom scale range (default: 0.1–10, log-uniform):
python scripts/make_sim3_from_se3.py \
    --src dataset/structured3D_train \
    --dst dataset/structured3D_sim3_train \
    --scale_min 0.5 --scale_max 2.0
```

**When to use this vs synthetic data:**
Use this route when you have a domain-specific SE3 dataset (e.g. semantic3D, structured3D) and want to add scale variation without changing the geometry distribution. Use Route 2 when you want maximum diversity and no dependency on any real scene data.

---

## Route 2 — Fully-Synthetic Dataset

`scripts/generate_synthetic_large.py` generates training pairs from random geometric primitives — no SLAM maps, no real sensor data required.

### Why synthetic?

Real-world SLAM maps impose a distribution over scene geometry (mostly indoor, Manhattan-world, specific noise levels). The synthetic generator makes no such assumption — all primitive orientations are drawn uniformly from SO(3), so the network learns the geometry of line correspondence rather than scene-type priors.

### Pool geometry — five primitive types

Each line pool is an independent random mix of:

| Primitive | Function | What it simulates |
|-----------|----------|-------------------|
| Plane patch | `make_plane_patch` | Any flat surface — façade, floor, road, rooftop |
| Wireframe box | `make_wireframe` | Object edges — vehicles, machines, furniture |
| Line bundle | `make_line_bundle` | Corners, poles, any radiating structure |
| Parallel group | `make_parallel_group` | Corridors, rails, pipes, power lines |
| Grid patch | `make_grid_patch` | Fences, lattices, structural grids |

`make_structured_pool(n)` (in `sim3/pair_generator.py`) samples random counts of each primitive type, distributes `n` lines across them via a Dirichlet split, and fills remaining lines with directional noise. Every call is independent.

### Pair construction

Each pair calls `make_structured_pool` **three times independently**:

```
inlier_pool  = make_structured_pool(k + buffer)   # shared GT lines (reference frame)
out_ref_pool = make_structured_pool(n_out2)        # reference-only outliers
out_q_pool   = make_structured_pool(n_out1)        # query-only outliers
```

```
plucker2 = inlier_pool[:k]  +  out_ref_pool                     (reference / big map)
plucker1 = Sim3⁻¹(inlier_pool[:k])  +  Sim3⁻¹(out_q_pool)     (query / submap)
```

Query outliers are generated in the reference frame then transformed to the query frame so that moment magnitudes remain consistent with the inlier lines. The two sides share only the inlier lines — their outlier regions come from completely different primitive mixes.

### Scenarios

`generate_diverse_pair()` draws one scenario per pair:

| Scenario | Prob | Query lines | Reference lines | Overlap | Notes |
|----------|------|------------|-----------------|---------|-------|
| `submap` | 30% | 10–150 | 80–700 | beta(1.5, 5) — low | High moment noise; primary use case |
| `relocalize` | 25% | similar | similar | beta(2.5, 2.5) — moderate | Cross-session, symmetric noise |
| `loop` | 20% | 30–500 | same | beta(5, 2) — high | Loop closure |
| `dense_sparse` | 15% | 5–40% of ref | 150–700 | beta(2, 2) — any | RGB-D vs monocular density mismatch |
| `zero_overlap` | 10% | 10–400 | 10–600 | 0 | Hard negatives |

Scale: log-uniform in [0.1, 10]. Line counts are **fully variable per pair** — not padded to any fixed size.

### Generate

```bash
# Default: 200k train, 2k valid (cpu_count−1 workers)
python scripts/generate_synthetic_large.py

# Large run used for current training:
python scripts/generate_synthetic_large.py --n_train 500000 --n_valid 5000 --workers 8

# Options:
#   --n_train     Training pairs (default: 200 000)
#   --n_valid     Validation pairs (default: 2 000)
#   --workers     Parallel CPU workers (default: cpu_count−1)
#   --chunk_size  Pairs per worker task (default: 2 000)
#   --seed        RNG seed (default: 42)
#   --out_dir     Root dataset dir (default: ./dataset)
```

Output: `dataset/synthetic_train/` and `dataset/synthetic_valid/`.

**Dataset stats** (500k train, 5k valid, seed 42):

| Split | Pairs | n1 (query) | n2 (reference) | Inliers/pair | Zero-overlap |
|-------|-------|-----------|----------------|--------------|-------------|
| `synthetic_train` | 500 000 | median 113, max 796 | median 278, max 795 | median 27, max 495 | 54 223 (10.8%) |
| `synthetic_valid` | 5 000 | median 114, max 776 | median 271, max 699 | median 27, max 486 | 554 (11.1%) |

Scale log-uniform [0.10, 10.00], median ≈ 1.0. Generation: ~4 600 pairs/s on 7 CPU workers.

---

## Dataset pkl Format

Every split (from either route) is a directory of six pickle files:

| File | Shape per sample | dtype |
|------|-----------------|-------|
| `matches.pkl` | `(2, n_inliers)` — row 0 = src indices, row 1 = tgt indices | int32 |
| `plucker1.pkl` | `(n1, 6)` variable | float32 |
| `plucker2.pkl` | `(n2, 6)` variable | float32 |
| `R_gt.pkl` | `(3, 3)` | float32 |
| `t_gt.pkl` | `(3, 1)` | float32 |
| `s_gt.pkl` | scalar | float32 |

**Why sparse indices, not a dense N×N matrix?**

The original PlueckerNet dataset fixes every scene to exactly 700 lines per side. The reason: the loss compares a predicted N×N probability matrix against a GT N×N binary match matrix — uniform N makes batching trivial without a custom collate function. The fixed size also bakes in the implicit assumption that every scene has equally dense, equally complete line observations, which fails in practice (a sparse monocular query and a dense RGB-D reference have very different line counts).

ScalePluckerNet stores matches as sparse `(2, k)` index arrays instead, letting n1 and n2 vary freely per pair. Batching is handled by `variable_collate` in `sim3/dataloader.py`, which zero-pads the match matrix only within a mini-batch to the local maximum N. This lets the network see the true density asymmetry (e.g. n1=30 submap lines vs n2=400 reference lines) rather than having it normalized away.

---

## Training

### From scratch

```bash
python train.py --dataset synthetic --epochs 1000 --lr 5e-4
```

### Warm-start from existing weights

```bash
python train.py --dataset synthetic \
    --pretrain output/se3sim3/se3sim3-finetune/best_val_checkpoint.pth \
    --lr 1e-4 --gamma 0.999 --epochs 500
```

### Resume an interrupted run

```bash
python train.py --dataset synthetic \
    --resume output/synthetic/2026-06-12/checkpoint.pth
```

### Background launch (recommended for long runs)

```bash
nohup /home/rueyday/miniconda3/envs/torch5090/bin/python train.py \
    --dataset synthetic \
    --pretrain output/se3sim3/se3sim3-finetune/best_val_checkpoint.pth \
    --lr 5e-4 --epochs 1000 \
    --ransac grassmannian \
    --train_epoch_size 16000 \
    >> output/synthetic_train.log 2>&1 &

tail -f output/synthetic_train.log
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `joint` | pkl split name under `./dataset/` (e.g. `synthetic`, `structured3D_sim3`) |
| `--pretrain` | — | Warm-start from checkpoint (`strict=False`, weights only) |
| `--resume` | — | Full resume: restores optimizer + scheduler state |
| `--epochs` | `1000` | Total training epochs |
| `--batch` | `1` | Mini-batch size; keep 1 for variable-length pairs |
| `--iter_size` | `32` | Gradient accumulation steps; effective batch = `batch × iter_size` |
| `--lr` | `5e-4` | Initial learning rate |
| `--gamma` | `0.99` | ExponentialLR decay per epoch; `1.0` = constant LR |
| `--cosine_lr` | off | Use CosineAnnealingWarmRestarts instead of ExponentialLR |
| `--ransac` | `grassmannian` | Validation RANSAC backend: `sim3` or `grassmannian` |
| `--train_epoch_size` | `0` | If > 0, sample this many pairs per epoch with replacement (keeps epochs short on large datasets; 0 = full dataset) |
| `--model` | `knn` | Architecture: `knn` (original PluckerNetKnn) or `alt` (alternating attention) |
| `--name` | today's date | Checkpoint subdirectory: `output/<dataset>/<name>/` |
| `--workers` | `8` | DataLoader worker processes |
| `--gpu` | `0` | CUDA device index |

### Validation metric

The sole checkpoint-selection metric is **`avg_inlier_ratio`** — the fraction of the top-100 predicted correspondences that are GT inlier matches, averaged across the validation set. Higher is better; random baseline ≈ 1%. The training log also prints `med_rot`, `med_trans`, and `med_scale_err(log)` as diagnostics but these do not affect checkpoint saving.

**InlierProb during training:** the logged `InlierProb` starts positive (~+N) at initialization and should go negative as the network learns. Negative means the model assigns more probability mass to true inliers. Target: approaching −1.

---

## Rotation Estimation Metric Analysis

We evaluated three rotation estimation approaches on ground-truth matched
Plücker pairs from `7scenes_valid` (500 scenes, 20–73 GT matches per scene).
Script: `scripts/eval_rotation_metric.py`.

### Results on GT matches (500 scenes, `7scenes_valid`)

Solver applied directly to GT matched pairs (no RANSAC, upper bound):

| Method | Med err | Mean err | <1° | <5° | <10° |
|--------|---------|---------|-----|-----|------|
| **L2 Procrustes (no sign align)** | **1.43°** | **1.62°** | 0.35 | **0.99** | **1.00** |
| L1 IRLS (no sign align) | 1.63° | 1.85° | 0.30 | 0.98 | 1.00 |
| L2 Procrustes (raw-dot sign align) | 179.0° | 174.5° | 0.00 | 0.01 | 0.01 |
| L1 IRLS (raw-dot sign align) | 179.1° | 174.6° | 0.01 | 0.02 | 0.02 |

RANSAC with GT-only correspondences (tests hypothesis generation quality):

| Rotation solver in minimal sampler | Med err | Catastrophic failures (>90°) |
|-------------------------------------|---------|-------------------------------|
| L2 aligned (old default) | 6.27° | 5/20 scenes |
| **L2 raw (no sign align)** | **2.81°** | **0/20 scenes** |

**L2 Procrustes without sign alignment is the clear winner in both settings.**
Applied in `lib/ransac_grassmannian.py` via `solve_rotation_l2_raw`.

### Why L2 > L1

L2 Procrustes is the maximum-likelihood estimator when direction noise is
Gaussian — which it approximately is for line fitting in both SLAM map
sources. L1 IRLS is designed to be robust against outliers, but RANSAC
already filters outliers before any refinement step: the residual inlier set
has small, near-Gaussian direction errors. L1 therefore adds unnecessary
variance (median 1.63° vs 1.43°) without any benefit. L2 also has a
closed-form global optimum (single SVD), while L1 requires iterative
convergence with sensitivity to the starting point.

### Why we do not use the geodesic distance on SO(3)

The geodesic distance on SO(3) between two rotations R₁ and R₂ applied to a
direction d is `∠(R₁d, R₂d) = arccos(d^T R₁^T R₂ d)`. For small angular
residuals θ ≪ 1:

```
arccos(cos θ) ≈ θ   and   ||R₁d - R₂d||² ≈ 2(1 - cos θ) ≈ θ²
```

Minimising the L2 direction residuals is therefore equivalent to minimising
the geodesic distance for small residuals. RANSAC enforces this regime: only
line pairs whose G(2,4) principal angle falls below the inlier threshold
(default 0.3 rad) are used for refinement. Pairs with large angular errors
are already excluded as outliers. Using the geodesic distance explicitly
would add computational overhead with no accuracy gain over L2 in this
low-residual regime.

### Sign alignment pitfall

63% of GT matched direction pairs have a negative raw dot product
(`d_mono · d_metric < 0`). This does **not** indicate antiparallel directions:
it simply means the rotation maps `d_mono` to the opposite hemisphere of
`d_metric`, which is normal for large rotation angles. The raw-dot sign
alignment heuristic ("flip d₁ if d₁·d₂ < 0") incorrectly treats this as an
antiparallel correspondence and flips 63% of source directions, inverting the
cross-covariance sum and producing ~180° rotation errors.

Our implementation avoids this by not applying sign alignment when both line
clouds share a consistent direction convention (both derived from the same
mesh source). For maps with independent direction signs, use the iterative
EM alignment (`solve_rotation_l2_robust`), which re-aligns signs under the
current R estimate rather than the raw dot product.

---

## Sim(3) RANSAC Backends

Two interchangeable backends; select with `--ransac sim3` or `--ransac grassmannian`.

**Minimal solver (shared by both):**
1. Estimate R from direction pairs via SVD (Wahba problem — L2 Procrustes, no sign alignment)
2. Solve for (s, t) jointly from moment equations via 3n×4 least-squares

**`sim3` backend** (`sim3/ransac.py`): plain RANSAC, 2-correspondence minimal sets.

**`grassmannian` backend** (`sim3/ransac_grassmannian.py`, default): adds
- Stratified direction-bin sampling to avoid degenerate all-parallel minimal sets
- LO-RANSAC: up to 10 local-optimisation re-fits on growing inlier sets

Both backends expose the same call signature:

```python
from sim3.ransac_grassmannian import ransac_sim3   # or from sim3.ransac

R, t, s, inlier_mask, n_inliers = ransac_sim3(
    plucker1.T,          # (6, N) — columns are lines
    plucker2.T,          # (6, N)
    inlier_threshold=0.3,
)
# Returns (None, None, None, None, 0) on failure — always check R is not None
```

---

## Evaluation

```bash
python scripts/eval.py \
    --checkpoint output/synthetic/2026-06-12/best_val_checkpoint.pth \
    --dataset synthetic \
    --ransac grassmannian

# Multiple datasets (comma-separated):
python scripts/eval.py \
    --checkpoint output/se3sim3/se3sim3-finetune/best_val_checkpoint.pth \
    --dataset semantic3D_sim3,structured3D_sim3 \
    --ransac grassmannian --threshold 0.3
```

Reports: `avg_inlier_ratio`, `med_rot`, `med_trans`, `med_scale_err` — broken down by overlap bucket (low / medium / high / zero). To reproduce the SE3 vs Sim3 comparison table below, run `eval.py` on `semantic3D` and `semantic3D_sim3` with the SE3-pretrained checkpoint, then again with the se3sim3 checkpoint.

---

## Inference / Registration

`register.py` registers two Plücker line maps (from Structure-PLP-SLAM `.db` files) using the network + RANSAC to recover a Sim(3) transformation:

```bash
python register.py \
    --db_src path/to/mono_map.db \
    --db_tgt path/to/metric_map.db \
    --checkpoint output/se3sim3/se3sim3-finetune/best_val_checkpoint.pth

# More RANSAC runs for a harder scene:
python register.py --db_src mono.db --db_tgt metric.db \
    --n_runs 30 --n_iter 2000 --topk 200
```

---

## Known Pitfalls

**`torch.load` weights_only error on resume.** Checkpoints store an `EasyDict` config object. PyTorch 2.6 defaults to `weights_only=True`, which rejects EasyDict. Fixed in `sim3/trainer.py` via `torch.load(..., weights_only=False)`.

**InlierProb stays positive for many epochs.** This is normal at the start of training, especially when warm-starting from SE3 weights onto a new data distribution. It will cross zero once the network starts discriminating inliers and typically reaches −0.5 to −1 at convergence.

**SE(3) solver silently fails when scale ≠ 1.** The original PlueckerNet RANSAC absorbs `(s−1)·R·m` into a spurious translation estimate, producing rotation errors up to ~90° on Sim(3) data. This is the core motivation for the Sim(3) extension — always use `sim3/ransac.py` or `sim3/ransac_grassmannian.py`.

**Large synthetic datasets need `--train_epoch_size`.** 500k pairs × iter_size=32 → ~15 600 gradient steps/epoch → 4–6 h/epoch. Use `--train_epoch_size 16000` to keep epochs to ~30 min and save checkpoints more often. `RandomSampler(replacement=True)` is used so the full dataset is still seen over multiple epochs.

**`variable_collate` required for batch > 1.** The DataLoader uses `variable_collate` from `sim3/dataloader.py` automatically when `--batch > 1`. With `--batch 1` (default) no collate is needed. Do not set `--batch > 1` without this or you will get shape errors on the match matrix.

---

## Results

See the [project page](https://rueyday.github.io/ScalePluckerNet/) for evaluation tables. Run `scripts/eval.py` to generate numbers with your checkpoint:

```bash
python scripts/eval.py \
    --checkpoint output/<run>/best_val_checkpoint.pth \
    --dataset synthetic_valid \
    --ransac grassmannian
```

Key metrics: `recall_rot` (fraction of pairs with rot error < 5°), `avg_inlier_ratio`, `med_rot`, `med_scale_err`.
