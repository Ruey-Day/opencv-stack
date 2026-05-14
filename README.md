# ScalePlueckerNet

**ScalePlueckerNet** extends [PlueckerNet](https://github.com/Liumouliu/PlueckerNet) (Liu et al., CVPR 2021) from **SE(3)** to **Sim(3)** — jointly recovering rotation R, translation t, *and scale s* from Plücker line correspondences.

| Part | What it does |
|------|-------------|
| **1 — Failure analysis** | Proves analytically and verifies experimentally that the SE(3) Plücker solver structurally fails when scale is unknown. |
| **2 — Sim(3) training** | Extends PlueckerNet with a new closed-form Sim(3) RANSAC solver and a modified trainer that jointly recovers scale, rotation, and translation. |

## Research context

PlueckerNet learns to match 3D line correspondences between two scenes using Plücker coordinates. Its RANSAC back-end then recovers the relative SE(3) pose. A natural extension is **Sim(3)** — the similarity group that adds uniform scale — which arises in monocular SLAM, scale-ambiguous reconstruction, and multi-session mapping.

**Key design insight:** the correspondence network does not need to change. The Sinkhorn matching learns scale-agnostic features (directions `d` are unit vectors under Sim(3); relative moment structure within each point set is preserved up to a global scale). Only the RANSAC back-end needs to be extended to Sim(3). The scale is then recovered analytically from the matched line correspondences.

**Critical implementation note:** moment vectors `m` must **not** be normalized before feeding to the network. Their magnitude encodes scene scale; normalizing them destroys the only signal that makes scale estimation possible.

---

## Model Architecture

### Full inference pipeline

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
               │  │KNN encoder│  │    x[:,:3,:] moments  → KNN graph → Conv2d → (B,64,N)
               │  │  per cloud│  │    x[:,3:,:] directions→ KNN graph → Conv2d → (B,64,N)
               │  └─────┬─────┘  │    concat + MLP → (B,128,N)
               │        │        │
               │  ┌─────▼─────┐  │
               │  │Spatial GNN│  │    12 layers alternating:
               │  │self+cross │  │      self-attention  (within one cloud)
               │  │ ×6 each   │  │      cross-attention (between clouds)
               │  └─────┬─────┘  │    each layer: MultiHeadedAttention(4 heads) + MLP residual
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

### Input format

Lines are represented as 6D Plücker vectors in **[m, d] order** (moment first):

```
L = [m₀  m₁  m₂  d₀  d₁  d₂]     m = p × d  (moment),  d = unit direction
```

### Network layers (PluckerNetKnn)

| Layer | Input | Output | Notes |
|-------|-------|--------|-------|
| KNN graph conv — moments | `(B, 3, N)` | `(B, 64, N)` | `get_graph_feature` + Conv2d + MLP |
| KNN graph conv — directions | `(B, 3, N)` | `(B, 64, N)` | same structure |
| MLP merge | `(B, 128, N)` | `(B, 128, N)` | concat → linear → BN → ReLU |
| Self-attention ×6 | `(B, 128, N)` | `(B, 128, N)` | MultiHead(4 heads, dim=32) + MLP residual |
| Cross-attention ×6 | `(B, 128, N₁)` + `(B, 128, N₂)` | same | queries from one cloud, keys/values from the other |
| Pairwise L2 distance | `(B, 128, N₁)`, `(B, 128, N₂)` | `(B, N₁, N₂)` | |
| Sinkhorn OT | `(B, N₁, N₂)` | `(B, N₁, N₂)` | 30 iterations, temperature λ=0.1 |

**Output:** `prob_matrix` — soft doubly-stochastic matrix where `prob_matrix[b, i, j]` is the probability that line `i` in cloud 1 corresponds to line `j` in cloud 2.

### Why the network does not need to change for Sim(3)

Under Sim(3), directions transform as `d′ = Rd` (scale-invariant), so the KNN graph structure is **identical** in both views. The GNN's self-attention sees the same neighbourhood layout; its cross-attention learns to match lines with the same rotated direction and consistent moment ratio. The network never sees scale explicitly — it only learns which pairs are geometrically consistent. Scale is then recovered analytically by the RANSAC back-end from the moment magnitudes of the matched pairs.

### Sim(3) RANSAC — minimal solver

Given `R` from direction SVD (same as SE(3) RANSAC), scale and translation are solved jointly from **2 line pairs** (6 equations, 4 unknowns):

```
s · Rm₁ᵢ  −  [d₂ᵢ×] · t  =  m₂ᵢ       for i = 1, 2

⎡ Rm₁₁  −skew(d₂₁) ⎤ ⎡ s ⎤   ⎡ m₂₁ ⎤
⎢                   ⎥ ⎢   ⎥ = ⎢     ⎥     A ∈ ℝ⁶ˣ⁴,  x ∈ ℝ⁴
⎣ Rm₁₂  −skew(d₂₂) ⎦ ⎣ t ⎦   ⎣ m₂₂ ⎦

→  x = lstsq(A, b)     (closed form, no iteration)
```

Any hypothesis with `s ≤ 0` is rejected. The best hypothesis is refined on all inliers.

---

## Part 1 — Why SE(3) PlueckerNet fails on Sim(3)

### Background: Plücker coordinates

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
- **Translation is badly biased** — the solver absorbs the moment mismatch `(s−1)·Rm₁` into a wrong translation estimate.

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

## Part 2 — Sim(3)-aware PlueckerNet

### Sim(3) solver (details)

After recovering `R` from direction pairs via SVD, the per-line moment equation is:

```
m₂  =  s · R m₁  +  t × d₂
```

Let `m₁′ = R m₁`. Rearranging per correspondence `i`:

```
[ m₁′ᵢ  |  −[d₂ᵢ×] ]  [ s  ]  =  m₂ᵢ        (3 equations × 4 unknowns)
                         [ t  ]
```

Two line pairs give **6 equations for 4 unknowns** → solved by least squares. Any `s ≤ 0` hypothesis is rejected.

### Sim(3) motion matrix (RANSAC scoring)

```
M_sim3 = [ s·R    [t×]·R ]     L₂  =  M_sim3 · L₁
          [  0       R   ]
```

Inlier criterion: `‖L₂ − M_sim3 · L₁‖₂ < threshold`.

---

## Dataset Format

Each split is a directory of 6 pickle files (lists of numpy arrays):

| File | Shape per sample | dtype |
|------|-----------------|-------|
| `matches.pkl` | `(2, n_inliers)` — row 0 = src indices, row 1 = tgt indices | int32 |
| `plucker1.pkl` | `(n_lines, 6)` | float32 |
| `plucker2.pkl` | `(n_lines, 6)` | float32 |
| `R_gt.pkl` | `(3, 3)` | float32 |
| `t_gt.pkl` | `(3, 1)` | float32 |
| `s_gt.pkl` | scalar | float32 |

DataLoader path: `<data_dir>/<dataset>_train/` and `<dataset>_valid/`.

All Plücker lines use **[m, d] order** — moment first, direction last.

### Data sources

| Dataset | Source | Format |
|---------|--------|--------|
| `semantic3D` | Original PlueckerNet (Semantic3D outdoor LiDAR) — converted [d,m]→[m,d], s_gt=1.0 | 6D |
| `structured3D` | Original PlueckerNet (Structured3D indoor synthetic) — converted [d,m]→[m,d], s_gt=1.0 | 6D |
| `replica_gs` | Replica RGBD, world-space GlueStick line detection | 6D |
| `7scenes_gs` | 7-Scenes RGBD, world-space GlueStick line detection | 6D |
| `joint` | All four sources combined and shuffled | 6D |

Generate with:
```bash
# SE3 real datasets (converts from PlueckerNet format, adds scale augmentation)
python scripts/convert_se3_datasets.py

# Replica GlueStick (requires Replica RGBD data and GlueStick)
python scripts/generate_replica_gs_dataset.py

# 7-Scenes GlueStick (requires 7-Scenes data and GlueStick)
python scripts/generate_7scenes_gs_dataset.py

# Combine all into joint split
python scripts/combine_joint_dataset.py
```

---

## Training

Entry point: `train.py`

```
Extensions over original PlueckerNet (all opt-in via flags):

  CORE (always active)
    Sim(3) scale recovery: network learns R, t, AND scale s jointly.
    Sim(3) RANSAC is used for validation instead of SE(3).

  --dataset         Training data source:
                      semantic3D | structured3D | replica_gs | 7scenes_gs | joint (default)

  --dustbin         Learnable dustbin token (SuperGlue-style) for partial-overlap robustness.

  --cosine_lr       CosineAnnealingWarmRestarts instead of ExponentialLR.

  --in_channel 9    Plücker+LAB color (default: 6 = geometry only).
```

### Basic usage

```bash
conda activate torch5090
cd /home/rueyday/scale-aware-PlueckerNet

# Train on all datasets (joint), geometry-only (default):
python train.py

# Train on a single dataset:
python train.py --dataset semantic3D

# Fine-tune with dustbin from a joint checkpoint:
python train.py --dustbin \
    --pretrain output/joint/<date>/best_val_checkpoint.pth --lr 2e-4

# Resume a run:
python train.py --resume output/joint/<date>/checkpoint.pth
```

### All flags

```
--dataset        semantic3D | structured3D | replica_gs | 7scenes_gs | joint  [default: joint]
--data_dir       path to dataset root                                           [default: ./dataset]
--epochs         training epochs                                                [default: 400]
--batch          batch size                                                     [default: 32]
--lr             learning rate                                                  [default: 5e-4]
--gpu            GPU index                                                      [default: 0]
--workers        DataLoader workers                                             [default: 8]
--in_channel     6 (geometry only) or 9 (Plücker+LAB color)                    [default: 6]
--dustbin        enable learnable dustbin token
--cosine_lr      use CosineAnnealingWarmRestarts instead of ExponentialLR
--pretrain       warm-start from checkpoint (strict=False)
--resume         resume from checkpoint
```

Checkpoints and TensorBoard logs: `output/<dataset>/<date>/`

### Validation metrics

| Metric | Description |
|--------|-------------|
| `recall_rot` | Fraction of scenes with rotation error < 20° |
| `med_rot` | Median rotation error (degrees) |
| `med_trans` | Median translation error |
| `med_scale_err` | Median log-ratio scale error `\|log(ŝ / s)\|` |
| `avg_inlier_ratio` | Average % of top-K correspondence candidates that are true inliers |

The primary training metric is `avg_inlier_ratio`.

---

## Evaluation

Entry point: `scripts/eval.py`

Evaluates a checkpoint on one or more dataset validation splits and reports rotation error, translation error, and inlier ratio.

```bash
# Evaluate on all four datasets (default):
python scripts/eval.py \
    --weights output/joint/<date>/best_val_checkpoint.pth

# Evaluate on a specific dataset:
python scripts/eval.py \
    --weights output/joint/<date>/best_val_checkpoint.pth \
    --dataset replica_gs

# All flags:
python scripts/eval.py --help
```

Output: per-dataset table with `recall_rot`, `med_rot`, `med_trans`, `avg_inlier_ratio`.

---

## Dependencies

### Python environment

```bash
conda activate torch5090
# Python 3.11, PyTorch 2.6, CUDA, numpy 2.x
pip install tensorboardX easydict
```

### PlueckerNet (required)

`../PlueckerNet/` must exist (cloned alongside this repo). Model, config, and utility files are imported from there directly.

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

### GlueStick (for dataset generation)

```
/home/rueyday/scale-aware-cross-modal-registration/GlueStick
```

GlueStick runs on **CPU only** — always use `.to('cpu')` for `SPWireframeDescriptor`. Only the line detection output (`['lines']`) is used; GlueStick matching is not used.

---

## References

- **PlueckerNet** — Liu et al., *"PlueckerNet: Learn to Register 3D Line Reconstructions"*, CVPR 2021. [GitHub](https://github.com/Liumouliu/PlueckerNet)
- **7-Scenes dataset** — Shotton et al., *"Scene Coordinate Regression Forests for Camera Relocalization in RGB-D Images"*, CVPR 2013.
- **Sim(3) in SLAM** — Strasdat et al., *"Scale Drift-Aware Large Scale Monocular SLAM"*, RSS 2010.
