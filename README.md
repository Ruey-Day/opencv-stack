# ScalePlueckerNet

**ScalePlueckerNet** extends [PlueckerNet](https://github.com/Liumouliu/PlueckerNet) (Liu et al., CVPR 2021) from **SE(3)** to **Sim(3)** — jointly recovering rotation R, translation t, *and scale s* from Plücker line correspondences.

| Part | What it does |
|------|-------------|
| **1 — Failure analysis** | Proves analytically and verifies experimentally that the SE(3) Plücker solver structurally fails when scale is unknown. |
| **2 — Sim(3) training** | Extends PlueckerNet with a new Sim(3) RANSAC solver, multi-source dataset pipeline, and a modified trainer that jointly recovers scale, rotation, and translation. |

---

## Research Context

PlueckerNet learns to match 3D line correspondences between two scenes using Plücker coordinates. Its RANSAC back-end then recovers the relative SE(3) pose. A natural extension is **Sim(3)** — the similarity group that adds uniform scale — which arises in monocular SLAM, scale-ambiguous reconstruction, and multi-session mapping where two maps share geometry but were built at different metric scales.

**Key design insight:** the correspondence network does not need to change. The Sinkhorn matching learns scale-agnostic features: directions `d` are unit vectors under Sim(3), and the relative moment structure within each point set is preserved up to a global scale. Only the RANSAC back-end needs to be extended to Sim(3). Scale is then recovered analytically from the moment magnitudes of the matched pairs.

**Critical implementation note:** moment vectors `m` must **not** be normalized before feeding to the network. Their magnitude encodes scene scale; normalizing them destroys the only signal that makes scale estimation possible.

---

## Repository Layout

```
scale-aware-PlueckerNet/
├── sim3/
│   ├── dataloader.py          # Sim3PluckerData — loads [m,d] format + s_gt
│   ├── trainer.py             # Sim3Trainer — validation uses Sim(3) RANSAC
│   ├── trainer_dustbin.py     # DustbinTrainer — dustbin extension
│   ├── model_dustbin.py       # PluckerNetKnnDustbin — dustbin token
│   ├── ransac.py              # Sim(3) RANSAC — L2 residual in Plücker space
│   ├── ransac_grassmannian.py # Sim(3) RANSAC — Grassmannian angle metric
│   └── __init__.py
│
├── scripts/
│   ├── convert_se3_datasets.py         # Step 1: convert semantic3D/structured3D
│   ├── generate_se3real_sim3_dataset.py # Step 2a: scale-augment the SE3 datasets
│   ├── generate_replica_gs_dataset.py  # Step 2b: Replica RGBD world-space lines
│   ├── generate_7scenes_gs_dataset.py  # Step 2c: 7-Scenes RGBD world-space lines
│   ├── _pair_gen.py                    # Shared pair generation utilities
│   ├── combine_joint_dataset.py        # Step 3: merge all sources into joint split
│   ├── eval.py                         # Evaluation entry point
│   └── visualize_lines.py              # 3D line visualisation helper
│
├── train.py                   # Training entry point
├── chess_plueckernet_demo.py  # Part 1 failure analysis demo
│
├── dataset/
│   ├── replica_gs_train/      # 14,000 scenes
│   ├── replica_gs_valid/      # 400 scenes
│   ├── 7scenes_gs_train/      # 9,000 scenes
│   ├── 7scenes_gs_valid/      # 300 scenes
│   ├── se3real_sim3_train/    # 4,658 scenes (semantic3D + structured3D + scale aug)
│   ├── se3real_sim3_valid/    # 823 scenes
│   ├── joint_train/           # 27,658 scenes (all three combined)
│   └── joint_valid/           # 1,523 scenes
│
├── output/                    # Checkpoints and TensorBoard logs
└── results/                   # Evaluation figures and JSON outputs
```

Parent repo `../PlueckerNet/` must exist on `sys.path` — all entry points add it automatically.

---

## Part 1 — Why SE(3) PlueckerNet Fails on Sim(3)

### Plücker coordinates

A 3D line through point `p` with unit direction `d` is encoded as:

```
L = (m, d)    m = p × d   (moment vector)
```

### Transformation laws

| Group | Direction | Moment |
|-------|-----------|--------|
| **SE(3)** `(R, t)` | `d′ = R d` | `m′ = R m + t × R d` |
| **Sim(3)** `(s, R, t)` | `d′ = R d` | `m′ = s · R m + t × R d` |

### Why the SE(3) solver fails

The SE(3) solver assumes `m₂ = R m₁ + t × R d₁` and minimises the residual:

```
e  =  m₂  −  R m₁  −  t × R d₁
   =  (s − 1) · R m₁        for a Sim(3)-transformed pair
```

This residual is **non-zero for any s ≠ 1** and **cannot be made zero by any choice of t**. It grows linearly with `|s − 1|` and with the magnitude of the moment vectors.

Consequences:
- **Rotation is unaffected** — direction vectors are unit vectors, so `d′ = Rd` is scale-invariant. The SVD rotation step recovers `R` exactly.
- **Translation is badly biased** — the solver absorbs `(s−1)·Rm₁` into a spurious translation, giving wrong `t` and a meaningless pose estimate.

### Experimental verification (7-Scenes Chess)

Pipeline: 30 RGBD frames → voxel point cloud → PCA-based 3D line extraction → Plücker registration.

| | Rotation error | Translation error | Plücker residual |
|---|---|---|---|
| **SE(3)** solver on SE(3) data | **0.000°** | **0.000 m** | **0.000** ✓ |
| **SE(3)** solver on Sim(3) data (s=1.45) | 0.000° | **0.858 m** | **0.630** ✗ |

Ground-truth transform: R = 27°, t = [0.35, −0.20, 0.15] m, s = 1.45.

### Figures

| Figure | Description |
|--------|-------------|
| ![fig1](results/fig1_voxel_overview.png) | Point cloud stitched from 30 RGBD frames |
| ![fig2](results/fig2_SE3_registration.png) | SE(3) registration — perfect alignment |
| ![fig3](results/fig3_SIM3_registration_failure.png) | Sim(3) registration with SE(3) solver — rotation correct, translation wrong |
| ![fig4](results/fig4_why_sim3_fails.png) | Analytical residual as a function of scale `s` |
| ![fig5](results/fig5_pluecker_lines_3d.png) | Extracted 3D Plücker lines visualised in scene |

### Running the failure analysis demo

```bash
conda activate depth_anything
export CHESS_DATA_DIR=/path/to/7-Scenes/chess/seq-01
python chess_plueckernet_demo.py
# figures written to results/
```

---

## Part 2 — Sim(3)-Aware PlueckerNet

### Model Architecture

```
  Sequence A (RGBD / monocular)        Sequence B (RGBD / monocular)
         │                                      │
  ┌──────▼──────┐                       ┌───────▼──────┐
  │ Line detect │                       │ Line detect  │
  │ (GlueStick) │                       │ (GlueStick)  │
  └──────┬──────┘                       └───────┬──────┘
         │ 2D segments                          │ 2D segments
  ┌──────▼──────┐                       ┌───────▼──────┐
  │ Lift to 3D  │                       │ Lift to 3D   │
  │  Plücker    │                       │  Plücker     │
  │  [m, d]     │                       │  [m, d]      │
  └──────┬──────┘                       └───────┬──────┘
         │ L₁ ∈ ℝᴺ¹ˣ⁶                          │ L₂ ∈ ℝᴺ²ˣ⁶
         └──────────────┬───────────────────────┘
                        │
               ┌────────▼────────┐
               │ ScalePlueckerNet │
               │  (PluckerNetKnn) │
               │                 │
               │  ┌───────────┐  │
               │  │KNN encoder│  │    moments  [m] → KNN graph → Conv2d → (B,64,N)
               │  │  per cloud│  │    directions[d] → KNN graph → Conv2d → (B,64,N)
               │  └─────┬─────┘  │    concat → MLP → (B,128,N)
               │        │        │
               │  ┌─────▼─────┐  │
               │  │Spatial GNN│  │    12 layers alternating:
               │  │self+cross │  │      self-attention  (within one cloud)
               │  │ ×6 each   │  │      cross-attention (between clouds)
               │  └─────┬─────┘  │    each: MultiHeadedAttention(4 heads) + MLP residual
               │        │        │
               │  ┌─────▼─────┐  │
               │  │ Sinkhorn  │  │    pairwise L2 distance matrix (N₁×N₂)
               │  │  OT 30it  │  │    → Sinkhorn normalisation (30 iterations)
               │  └─────┬─────┘  │    → doubly-stochastic prob_matrix ∈ ℝᴺ¹ˣᴺ²
               └────────┼────────┘
                        │ prob_matrix
               ┌────────▼────────┐
               │  Top-K select   │    argmax top-K entries → K candidate pairs (i₁, i₂)
               └────────┬────────┘
                        │ K matched Plücker pairs
               ┌────────▼────────┐
               │  Sim(3) RANSAC  │    Stage 1: R  from direction SVD   (scale-invariant)
               │                 │    Stage 2: s,t from moment LS       (3n×4 system)
               │  minimal solver │    Inlier: ‖L₂ − M_sim3·L₁‖₂ < τ
               │  n=2 pairs      │    Refine on all inliers
               └────────┬────────┘
                        │
                    s,  R,  t
```

**Why the network does not need to change for Sim(3):** directions transform as `d′ = Rd` (scale-invariant), so the KNN graph structure is identical in both views. The GNN's cross-attention learns to match lines with the same rotated direction and consistent moment ratio. Scale is recovered analytically by RANSAC from the moment magnitudes of the matched pairs — the network never sees it explicitly.

### Network layers (PluckerNetKnn)

| Layer | Input → Output | Notes |
|-------|---------------|-------|
| KNN graph conv — moments | `(B,3,N) → (B,64,N)` | `get_graph_feature` + Conv2d + MLP |
| KNN graph conv — directions | `(B,3,N) → (B,64,N)` | same structure |
| MLP merge | `(B,128,N) → (B,128,N)` | concat → Linear → BN → ReLU |
| Self-attention ×6 | `(B,128,N) → (B,128,N)` | MultiHead(4 heads, dim=32) + MLP residual |
| Cross-attention ×6 | `(B,128,N₁)+(B,128,N₂) → same` | queries from one cloud, keys/values from other |
| Pairwise L2 distance | `(B,128,N₁),(B,128,N₂) → (B,N₁,N₂)` | |
| Sinkhorn OT | `(B,N₁,N₂) → (B,N₁,N₂)` | 30 iterations, temperature λ=0.1 |

**Output:** `prob_matrix` — soft doubly-stochastic matrix where `prob_matrix[b,i,j]` is the probability that line `i` in cloud 1 corresponds to line `j` in cloud 2.

### Plücker line format

All data uses **[m, d] order** — moment first, direction last:

```
L = [m₀  m₁  m₂  d₀  d₁  d₂]     m = p × d  (moment),  d = unit direction
```

The original SE(3) PlueckerNet uses `[d, m]` order. Conversion is handled by `scripts/convert_se3_datasets.py`.

### Sim(3) RANSAC — minimal solver (`sim3/ransac.py`)

Given `R` from direction SVD (same as SE(3)), scale and translation are solved jointly from **2 line pairs** (6 equations, 4 unknowns):

```
m₂  =  s · R m₁  +  t × d₂

⎡ Rm₁₁  −skew(d₂₁) ⎤ ⎡ s ⎤   ⎡ m₂₁ ⎤
⎢                   ⎥ ⎢   ⎥ = ⎢     ⎥     A ∈ ℝ⁶ˣ⁴
⎣ Rm₁₂  −skew(d₂₂) ⎦ ⎣ t ⎦   ⎣ m₂₂ ⎦

→  x = lstsq(A, b)     (closed form, no iteration)
```

Hypotheses with `s ≤ 0` are rejected. Inlier criterion: `‖L₂ − M_sim3·L₁‖₂ < τ`, where the Sim(3) motion matrix is:

```
M_sim3 = [ s·R    [t×]·R ]     L₂  =  M_sim3 · L₁
          [  0       R   ]
```

The best hypothesis is refined on all inliers via overdetermined least squares.

### Training objective

The network is trained with a binary cross-entropy loss on the predicted `prob_matrix` vs the ground-truth match matrix `C_gt`:

```
L = −( C_gt · log(P + ε) + (1 − C_gt) · log(1 − P + ε) ).mean()
```

**RANSAC is not used during training** — it is only called during validation to report pose metrics. The training checkpoint is selected by `avg_inlier_ratio` (the fraction of the network's top-K predictions that are true inlier correspondences), which is purely a function of the predicted `prob_matrix`.

### Validation metrics

| Metric | Description |
|--------|-------------|
| `recall_rot` | Fraction of scenes with rotation error < 20° |
| `med_rot` | Median rotation error (degrees) |
| `med_trans` | Median translation error |
| `med_scale_err` | Median log-ratio scale error `|log(ŝ/s)|` |
| `avg_inlier_ratio` | Average % of top-K candidates that are true inliers ← **primary metric** |

---

## Dataset Pipeline

The full pipeline runs in three steps. Total dataset: **27,658 train / 1,523 valid** scenes.

```
Step 1  convert_se3_datasets.py         semantic3D + structured3D → [m,d] + s_gt=1.0
Step 2a generate_se3real_sim3_dataset.py  + random Sim(3) scale augmentation
Step 2b generate_replica_gs_dataset.py  Replica RGBD → world-space GlueStick lines
Step 2c generate_7scenes_gs_dataset.py  7-Scenes RGBD → world-space GlueStick lines
Step 3  combine_joint_dataset.py        merge and shuffle all sources
```

### Step 1 — Convert SE3 datasets (`scripts/convert_se3_datasets.py`)

Converts the original PlueckerNet datasets from `[d, m]` Plücker order to `[m, d]` and adds `s_gt = 1.0`:

```bash
python scripts/convert_se3_datasets.py
# reads  ../PlueckerNet/dataset/{semantic3D,structured3D}_{train,valid}/
# writes ./dataset/{semantic3D,structured3D}_{train,valid}/
```

### Step 2a — se3real Sim(3) augmentation (`scripts/generate_se3real_sim3_dataset.py`)

Merges semantic3D and structured3D and applies random Sim(3) scale per scene:

- **15%** of scenes: keep `s = 1.0` (pure SE(3) — preserves base capability)
- **85%** of scenes: draw `s ~ log-uniform(0.1, 10.0)`, then apply:
  - `plucker2[:, :3] *= s` (scale moments; directions unchanged)
  - `t_gt *= s`

```bash
python scripts/generate_se3real_sim3_dataset.py
# output: dataset/se3real_sim3_{train,valid}/
# sizes:  4,658 train / 823 valid
```

### Step 2b/c — World-space GlueStick datasets

`generate_replica_gs_dataset.py` and `generate_7scenes_gs_dataset.py` build 3D Plücker line pools from RGBD sequences using GlueStick line detection:

**World-space line pool construction** (per scene):
1. Run GlueStick 2D line detection on every N-th frame (CPU only — `SPWireframeDescriptor.to('cpu')`)
2. Unproject 2D segments to 3D using depth + camera pose, compute Plücker `[m, d]`
3. Accumulate into a per-scene pool; deduplicate via voxel-hash NMS (`pos_voxel=0.10 m`, `dir_voxel=0.04 ≈ 2.3°`), keeping the line with the largest `‖m‖` per cell

**Pair generation** (`scripts/_pair_gen.py`):
- `N_TOTAL = 700` lines per pair (padded with random outliers if needed)
- `N_MAX_INLIERS = 490` at 100% overlap
- Scale: `s ~ log-uniform(0.1, 10.0)` applied to the target cloud
- Overlap is sampled at discrete levels with the following probability distribution:

| Overlap | Probability | n_inliers |
|---------|-------------|-----------|
| 0% (zero-overlap) | 12% | 0 — `s_gt = 0.0` signals no GT pose |
| 5% | 8% | ~25 |
| 10% | 8% | ~49 |
| 20% | 9% | ~98 |
| 30% | 9% | ~147 |
| 50% | 10% | ~245 |
| 70% | 10% | ~343 |
| 100% | 34% | 490 |

```bash
# Run both generators in parallel (each takes ~2–4 hours on RGBD data):
python scripts/generate_replica_gs_dataset.py &
python scripts/generate_7scenes_gs_dataset.py &
wait

# sizes: replica_gs: 14,000 train / 400 valid
#        7scenes_gs:  9,000 train / 300 valid
```

**Note:** GlueStick must run on CPU. GPU inference crashes with `SPWireframeDescriptor`. Only the `['lines']` output is used — GlueStick's own matching is not used.

### Step 3 — Combine into joint split (`scripts/combine_joint_dataset.py`)

Merges all three source splits, shuffles with a fixed seed, and saves:

```bash
python scripts/combine_joint_dataset.py
# output: dataset/joint_{train,valid}/
# sizes:  27,658 train (14k + 9k + 4.6k) / 1,523 valid
```

### Dataset sizes summary

| Split | Train | Valid | Source |
|-------|-------|-------|--------|
| `replica_gs` | 14,000 | 400 | Replica RGBD, GlueStick world-space lines |
| `7scenes_gs` | 9,000 | 300 | 7-Scenes RGBD, GlueStick world-space lines |
| `se3real_sim3` | 4,658 | 823 | semantic3D + structured3D + scale aug |
| `joint` | 27,658 | 1,523 | All three combined |

### Dataset format

Each split is a directory of 6 pickle files (lists of numpy arrays):

| File | Shape per sample | dtype |
|------|-----------------|-------|
| `matches.pkl` | `(2, n_inliers)` — row 0 = src indices, row 1 = tgt indices | int32 |
| `plucker1.pkl` | `(N_TOTAL, 6)` | float32 |
| `plucker2.pkl` | `(N_TOTAL, 6)` | float32 |
| `R_gt.pkl` | `(3, 3)` | float32 |
| `t_gt.pkl` | `(3, 1)` | float32 |
| `s_gt.pkl` | scalar (`0.0` = zero-overlap, no valid pose) | float32 |

---

## Training

Entry point: `train.py`

```bash
conda activate torch5090
cd /home/rueyday/scale-aware-PlueckerNet

# Train on all datasets (default):
python train.py

# Train on a single source:
python train.py --dataset se3real_sim3

# Resume a run:
python train.py --resume output/joint/<date>/checkpoint.pth

# Fine-tune with dustbin from a joint checkpoint:
python train.py --dustbin \
    --pretrain output/joint/<date>/best_val_checkpoint.pth --lr 2e-4

# Multiple simultaneous runs (use --name to avoid checkpoint clashes):
python train.py --dataset joint --name run_a &
python train.py --dataset joint --dustbin --name run_b &
```

### All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `joint` | `semantic3D \| structured3D \| replica_gs \| 7scenes_gs \| se3real_sim3 \| joint` |
| `--data_dir` | `./dataset` | Dataset root |
| `--epochs` | 400 | Training epochs |
| `--batch` | 32 | Batch size |
| `--lr` | 5e-4 | Learning rate |
| `--gpu` | 0 | GPU index |
| `--workers` | 8 | DataLoader workers (reduce to 4 when running multiple jobs) |
| `--in_channel` | 6 | Input channels: `6` = geometry only, `9` = Plücker + LAB color |
| `--dustbin` | off | Enable learnable dustbin token (SuperGlue-style) |
| `--cosine_lr` | off | CosineAnnealingWarmRestarts instead of ExponentialLR |
| `--pretrain` | — | Warm-start from checkpoint (`strict=False`) |
| `--resume` | — | Resume training from checkpoint |
| `--name` | today's date | Override run name (prevents clashes when running multiple jobs) |

Checkpoints and TensorBoard logs: `output/<dataset>/<name>/`

### GPU memory

Each job at `--batch 32` uses approximately 11–13 GB VRAM. On a 32 GB GPU:
- **2 jobs**: fits comfortably
- **3 jobs**: will OOM intermittently; reduce to `--batch 16` for secondary jobs

Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation when running multiple jobs.

---

## Evaluation

Entry point: `scripts/eval.py`

Three evaluation modes:

### Mode 1 — Cross-dataset eval (default)

Evaluates a checkpoint on one or more validation splits. Reports per-scene metrics and breaks results down by overlap level.

```bash
python scripts/eval.py \
    --weights output/joint/<date>/best_val_checkpoint.pth \
    --dataset replica_gs,7scenes_gs,se3real_sim3
```

**Overlap stratification:** each scene is bucketed by inlier count:

| Bucket | Inlier count | Covers |
|--------|-------------|--------|
| `no_overlap (0%)` | 0 | Zero-overlap pairs (no GT pose) |
| `sparse (~30%)` | 1–200 | 5% to 30% overlap levels |
| `dense (~70%)` | 201–490 | 50% to 100% overlap levels |

Per-bucket columns: `N | recall_rot | med_rot° | med_trans | med_s_err | inlier%`

**RANSAC backends** (selectable with `--ransac`):

| Backend | Flag | Inlier metric | Notes |
|---------|------|---------------|-------|
| L2 Sim(3) | `--ransac sim3` (default) | `‖L₂ − M·L₁‖₂ < 0.1` | Fast, threshold is scale-dependent |
| Grassmannian | `--ransac grassmannian` | `arccos(\|L̂₁·L̂₂\|) < 0.15 rad` | Theoretically grounded, scale/translation-invariant; stratified sampling + Tikhonov regularization |

```bash
# Compare both RANSAC backends:
python scripts/eval.py --weights ... --dataset replica_gs --ransac sim3
python scripts/eval.py --weights ... --dataset replica_gs --ransac grassmannian
```

Results saved to `results/eval_cross_dataset/<label>.json` (overall metrics + per-bucket raw arrays).

### Mode 2 — Chess B1/B2 benchmark (`--chess`)

Builds colored point clouds from two 7-Scenes Chess sequences, extracts 9D Plücker lines, and evaluates:

- **B1** (RGBD): both clouds at the same scale — tests pure SE(3) recovery
- **B2** (RGB-only simulation): target moments scaled by 1.8 — tests Sim(3) scale recovery

```bash
python scripts/eval.py --chess \
    --weights output/joint/<date>/best_val_checkpoint.pth \
    --chess_seq1 /path/to/chess/seq-01 \
    --chess_seq3 /path/to/chess/seq-03
```

### Mode 3 — Synthetic hypothesis test (`--hypothesis`)

Generates controlled SE(3) and Sim(3) synthetic test scenes and checks:
- **H1**: scale recovery — `med_scale_err < 0.10` on Sim(3) test scenes
- **H2**: rotation accuracy — `med_rot < 5°` on SE(3) test scenes

```bash
python scripts/eval.py --hypothesis \
    --weights output/joint/<date>/best_val_checkpoint.pth \
    [--se3_weights ../PlueckerNet/.../best_val_checkpoint_real.pth]
```

### All eval flags

```
--weights          Checkpoint path (required for all modes)
--dataset          Comma-separated val split names             [default: all four]
--data_dir         Dataset root                                [default: ./dataset]
--ransac           sim3 | grassmannian                         [default: sim3]
--out_dir          Output directory for JSON results
--label            Human-readable label for the run
--chess            Run Chess B1/B2 benchmark
--chess_seq1       Path to Chess seq-01
--chess_seq3       Path to Chess seq-03
--hypothesis       Run synthetic hypothesis test
--se3_weights      SE3-Net weights for hypothesis comparison   [optional]
--n_scenes         Scenes per condition for hypothesis test    [default: 200]
```

---

## Results

All Sim(3) models evaluated with Grassmannian RANSAC (`--ransac grassmannian`) unless noted.
`recall_rot` = fraction of scenes with rotation error < 20°. `med_s_err` = median |log(s_est/s_gt)|.

### se3real_sim3_valid — 823 scenes, random Sim(3) scale (s ∈ [0.1, 10])

This is the primary Sim(3) benchmark. Scale error is only meaningful here since s ≠ 1.
Original PlueckerNet is evaluated here too — it was trained on the same structured3D/semantic3D data, so it can find correspondences. Scale error is marked `—` because it does not estimate scale.

| Model | RANSAC | recall_rot | med_rot° | med_trans | med_s_err | inlier% |
|-------|--------|-----------|----------|-----------|-----------|---------|
| Original PlueckerNet (SE3) | L2 (original) | 0.175 | 180.0° | — | — | 58.6% |
| Original PlueckerNet (SE3) | Grassmannian | 0.877 | 8.15° | 2.04 | — | 58.6% |
| se3real, no cosine (ep 91) | L2 | 0.247 | 85.4° | — | 0.232 | 68.4% |
| se3real, no cosine (ep 91) | **Grassmannian** | 0.909 | 7.6° | 2.04 | 0.027 | 68.4% |
| se3real, cosine (ep 208) | Grassmannian | 0.883 | 7.81° | 2.02 | 0.026 | 69.0% |
| Joint, cosine (ep 297) | Grassmannian | 0.898 | 7.98° | 2.045 | 0.024 | 67.2% |
| Joint + dustbin (ep 82) | Grassmannian | 0.875 | 7.85° | 2.037 | 0.030 | 67.5% |

> The L2 → Grassmannian jump (0.247 → 0.909 recall, 85° → 7.6° med_rot) is purely a RANSAC solver change — same network weights. The 2-line L2 solver fails on structured3D/semantic3D parallel lines (rank-1 cross-covariance → 180° degenerate rotation). Grassmannian samples 3 lines stratified by dominant axis, eliminating this.

### replica_gs_valid — 400 real RGBD scenes, s ≈ 1

| Model | Overlap | N | recall_rot | med_rot° | med_trans | inlier% |
|-------|---------|---|-----------|----------|-----------|---------|
| Original PlueckerNet (SE3) | overall | 400 | 0.005 | 135.5° | 2.25 | 0.2% ❌ |
| se3real, no cosine (ep 91) | overall | 400 | 0.007 | 137.0° | 2.14 | 0.2% ❌ |
| se3real, no cosine (ep 91) | no overlap | 39 | 0.000 | 180.0° | — | 0.0% ❌ |
| se3real, no cosine (ep 91) | sparse (~30%) | 142 | 0.007 | 132.2° | 2.00 | 0.0% ❌ |
| se3real, no cosine (ep 91) | dense (~70%) | 219 | 0.009 | 131.2° | 2.24 | 0.4% ❌ |
| se3real, cosine (ep 208) | overall | 400 | 0.007 | 135.7° | 2.20 | 0.2% ❌ |
| se3real, cosine (ep 208) | no overlap | 39 | 0.000 | 180.0° | — | 0.0% ❌ |
| se3real, cosine (ep 208) | sparse (~30%) | 142 | 0.007 | 130.1° | 2.17 | 0.0% ❌ |
| se3real, cosine (ep 208) | dense (~70%) | 219 | 0.009 | 130.6° | 2.20 | 0.4% ❌ |
| Joint, cosine (ep 297) | overall | 400 | 0.682 | 0.01° | 0.000 | 79.7% |
| Joint, cosine (ep 297) | no overlap | 39 | 0.000 | 180.0° | — | 0.0% |
| Joint, cosine (ep 297) | sparse (~30%) | 142 | 0.690 | 0.01° | 0.000 | 70.2% |
| Joint, cosine (ep 297) | dense (~70%) | 219 | 0.799 | 0.01° | 0.000 | 100.0% |
| Joint + dustbin (ep 82) | overall | 400 | 0.720 | 0.01° | 0.000 | 79.5% |
| Joint + dustbin (ep 82) | no overlap | 39 | 0.000 | 180.0° | — | 0.0% |
| Joint + dustbin (ep 82) | sparse (~30%) | 142 | 0.676 | 0.01° | 0.000 | 69.8% |
| Joint + dustbin (ep 82) | dense (~70%) | 219 | 0.877 | 0.00° | 0.000 | 99.9% |

### 7scenes_gs_valid — 300 real RGBD scenes, s ≈ 1

| Model | Overlap | N | recall_rot | med_rot° | med_trans | inlier% |
|-------|---------|---|-----------|----------|-----------|---------|
| Original PlueckerNet (SE3) | overall | 300 | 0.007 | 141.0° | 2.18 | 0.0% ❌ |
| se3real, no cosine (ep 91) | overall | 300 | 0.000 | 139.9° | — | 0.1% ❌ |
| se3real, no cosine (ep 91) | no overlap | 37 | 0.000 | 180.0° | — | 0.0% ❌ |
| se3real, no cosine (ep 91) | sparse (~30%) | 114 | 0.000 | 131.5° | 2.14 | 0.0% ❌ |
| se3real, no cosine (ep 91) | dense (~70%) | 149 | 0.000 | 134.7° | 2.17 | 0.1% ❌ |
| se3real, cosine (ep 208) | overall | 300 | 0.003 | 144.6° | 2.04 | 0.1% ❌ |
| se3real, cosine (ep 208) | no overlap | 37 | 0.000 | 180.0° | — | 0.0% ❌ |
| se3real, cosine (ep 208) | sparse (~30%) | 114 | 0.000 | 127.4° | 1.93 | 0.0% ❌ |
| se3real, cosine (ep 208) | dense (~70%) | 149 | 0.007 | 142.8° | 2.11 | 0.1% ❌ |
| Joint, cosine (ep 297) | overall | 300 | 0.647 | 0.01° | 0.000 | 76.0% |
| Joint, cosine (ep 297) | no overlap | 37 | 0.000 | 180.0° | — | 0.0% |
| Joint, cosine (ep 297) | sparse (~30%) | 114 | 0.728 | 0.01° | 0.000 | 69.3% |
| Joint, cosine (ep 297) | dense (~70%) | 149 | 0.745 | 0.01° | 0.000 | 100.0% |
| Joint + dustbin (ep 82) | overall | 300 | 0.657 | 0.01° | 0.000 | 75.7% |
| Joint + dustbin (ep 82) | no overlap | 37 | 0.000 | 180.0° | — | 0.0% |
| Joint + dustbin (ep 82) | sparse (~30%) | 114 | 0.702 | 0.01° | 0.000 | 68.6% |
| Joint + dustbin (ep 82) | dense (~70%) | 149 | 0.785 | 0.01° | 0.000 | 100.0% |

> Both the original PlueckerNet and the se3real specialist fail completely on replica_gs and 7scenes_gs — same ~135–141° rotation and <0.2% inlier ratio regardless of overlap level. Both were trained exclusively on structured3D/semantic3D data and do not generalize to GlueStick world-space line parameterization. Only the joint model (trained on all three domains) is expected to fill these rows.

---

## Dependencies

### Python environment

```bash
conda activate torch5090
# Python 3.11, PyTorch 2.6, CUDA, numpy 2.x
pip install tensorboardX easydict scipy
```

### PlueckerNet (required)

`../PlueckerNet/` must exist alongside this repo. Model, config, and utility files are imported from there directly:

```
../PlueckerNet/
├── model/model_plucker.py   # PluckerNetKnn
├── config.py
└── lib/
    ├── loss.py
    ├── timer.py
    ├── file.py
    ├── utils.py
    └── transformations.py
```

### GlueStick (for dataset generation only)

```
/home/rueyday/scale-aware-cross-modal-registration/GlueStick
```

Run on **CPU only** (`SPWireframeDescriptor.to('cpu')`). Only `['lines']` output is used.

---

## Potential Extensions

### Color descriptors (9D Plücker + LAB)

Append per-line mean LAB color to the 6D Plücker vector, giving 9D input `[m₀,m₁,m₂, d₀,d₁,d₂, L*,A*,B*]`. Color provides an additional discriminative signal for matching lines in scenes with strong chromatic structure.

**Dataset generation:** `scripts/generate_replica_gs_dataset.py` and `scripts/generate_7scenes_gs_dataset.py` already support color — the mean LAB color of each 3D line's reprojected pixels is computed during pool construction. Enable with the `--add_colors` flag (or by setting `ADD_COLORS=True` in the generator).

**Training:** pass `--in_channel 9` to `train.py`. The network architecture is unchanged; the KNN encoder simply receives 9-channel input instead of 6.

**Important:** 9D checkpoints should not be evaluated on 6D (colorless) datasets. Neutral LAB padding `[50, 0, 0]` gives zero discriminative signal from the color channel and degrades performance. Use the 6D joint checkpoint for colorless inputs.

**Prior results:**
- joint_color (epoch 345): 98.22% inlier ratio on replica_gs_color_valid, 98.23% on 7scenes_gs_color_valid
- ~1.7% cost vs the 6D joint model on the same data

### Dustbin token

A learnable dustbin row and column are appended to the assignment matrix (SuperGlue-style), allowing the network to explicitly route unmatched lines to the dustbin rather than forcing them onto wrong correspondences. This improves robustness at low overlap.

Enable with `--dustbin`. Best used as a fine-tuning step from a converged joint checkpoint:

```bash
python train.py --dustbin \
    --pretrain output/joint/<date>/best_val_checkpoint.pth \
    --lr 2e-4 --name dustbin_ft
```

Implementation: `sim3/model_dustbin.py` (PluckerNetKnnDustbin), `sim3/trainer_dustbin.py`.

### Cosine learning rate schedule

`--cosine_lr` replaces ExponentialLR with `CosineAnnealingWarmRestarts(T_0=50, T_mult=2, eta_min=1e-6)`. Useful for long training runs where the default decay bottoms out too early.

### Grassmannian RANSAC (available now via `--ransac grassmannian`)

The Grassmannian solver (`sim3/ransac_grassmannian.py`) uses the principal angle between Plücker lines as the inlier metric — a proper geodesic distance on the Grassmannian G(1,5). This is scale- and translation-invariant, unlike the L2 threshold used by the default solver (whose effective sensitivity varies with scene scale).

Additional improvements over the default solver:
- **Direction sign handling** — flips source directions before Procrustes when they point away from target, correctly handling undirected line ambiguity
- **Stratified sampling** — bins lines by dominant axis before RANSAC sampling to prevent degenerate all-parallel minimal sets
- **Tikhonov regularization** — prevents scale collapse to near-zero on near-parallel scenes (e.g. chessboard corridors)
- **Scale prior seeding** — bootstraps from a full-set Procrustes + fixed-scale translation solve before the random RANSAC loop
- **Moment-magnitude refinement** — final scale estimate uses `median(‖m₂‖ / ‖R·m₁‖)` rather than re-running the joint LS, which is more robust to translation noise
