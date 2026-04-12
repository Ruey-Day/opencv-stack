# scale-aware-PlueckerNet

This repo extends [PlueckerNet](https://github.com/Liumouliu/PlueckerNet) (Liu et al., CVPR 2021) from **SE(3)** to **Sim(3)** — adding support for unknown scale in 3D line registration. It has two parts:

| Part | What it does |
|------|-------------|
| **1 — Failure analysis** | Proves analytically and verifies experimentally on real RGBD data that the SE(3) Plücker solver structurally fails when scale is unknown. |
| **2 — Sim(3) training** | Extends PlueckerNet with a new closed-form Sim(3) solver, synthetic data generator, and modified trainer that jointly recovers scale, rotation, and translation. |

---

## Research context

PlueckerNet learns to match 3D line correspondences between two scenes using Plücker coordinates. Its RANSAC back-end then recovers the relative SE(3) pose. A natural extension is **Sim(3)** — the similarity group that adds uniform scale — which arises in monocular SLAM, scale-ambiguous reconstruction, and multi-session mapping.

To our knowledge no prior work explicitly extends a Plücker-coordinate matching network to Sim(3). The gap is real: the SE(3) solver fails structurally (not just numerically) when scale differs, and the fix requires a new solver rather than a better network.

**Key design insight:** the correspondence network does not need to change. The Sinkhorn matching learns scale-agnostic features (directions `d` are unit vectors under Sim(3); relative moment structure within each point set is preserved up to a global scale). Only the RANSAC back-end needs to be extended to Sim(3). The scale is then recovered analytically from the matched line correspondences.

**Critical implementation note:** moment vectors `m` must **not** be normalized before feeding to the network. Their magnitude encodes scene scale; normalizing them destroys the only signal that makes scale estimation possible.

---

## Part 1 — Why SE(3) PlueckerNet fails on Sim(3)

### Background: Plücker coordinates

A 3D line through point `p` with unit direction `d` is encoded as:

```
L = (m, d)    m = p × d   (moment vector)
```

The 6-vector `[m₀ m₁ m₂ d₀ d₁ d₂]` is coordinate-free and used directly as network input.

### Transformation laws

| Group | Direction | Moment |
|-------|-----------|--------|
| **SE(3)** `(R, t)` | `d′ = R d` | `m′ = R m + t × R d` |
| **Sim(3)** `(s, R, t)` | `d′ = R d` | `m′ = s · R m + t × R d` |

Derivation of the Sim(3) moment law: a line through `p` with direction `d` maps to a line through `sRp + t` with direction `Rd`. The new moment is:

```
m′ = (sRp + t) × Rd = s(Rp × Rd) + t × Rd = s·R(p × d) + t × Rd = s·Rm + t × d′
```

### Why the SE(3) solver fails

The SE(3) solver assumes `m₂ = R m₁ + t × R d₁` and minimises the residual:

```
e  =  m₂  −  R m₁  −  t × R d₁
   =  (s − 1) · R m₁        for a Sim(3)-transformed pair
```

This residual is **non-zero for any s ≠ 1** and **cannot be made zero by any choice of t**. It grows linearly with `|s − 1|` and with the magnitude of the moment vectors (i.e., the distance of lines from the origin).

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

**Dependencies:** `numpy`, `scipy`, `matplotlib`, `opencv-python`
**Dataset:** [7-Scenes Chess](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) — intrinsics fx=fy=525, cx=319.5, cy=239.5; depth scale 1 mm/unit.

---

## Part 2 — Sim(3)-aware PlueckerNet

### Sim(3) solver

After recovering `R` from direction pairs via SVD (identical to the SE(3) case), the
per-line moment equation is:

```
m₂  =  s · R m₁  +  t × d₂
```

Let `m₁′ = R m₁`. Rearranging:

```
s · m₁′  −  [d₂×] · t  =  m₂
```

Written as a linear system per correspondence `i`:

```
[ m₁′ᵢ  |  −[d₂ᵢ×] ]  [ s  ]  =  m₂ᵢ        (3 equations × 4 unknowns)
                         [ t  ]
```

Two line pairs give **6 equations for 4 unknowns** → solved by least squares. This is the **minimum sample** for Sim(3) once `R` is fixed: the same 2 pairs used for `R` also determine `(s, t)` uniquely. Any `s ≤ 0` hypothesis is rejected.

### Sim(3) motion matrix (RANSAC scoring)

```
M_sim3 = [ s·R    [t×]·R ]     L₂  =  M_sim3 · L₁
          [  0       R   ]
```

Inlier criterion: `‖L₂ − M_sim3 · L₁‖₂ < threshold`.

### Network architecture

`PluckerNetKnn` from the original PlueckerNet is **reused unchanged**:

- KNN graph convolution on direction and moment channels separately
- Spatial attentional GNN (self + cross attention, 6 layers each)
- Sinkhorn optimal transport to produce a soft correspondence matrix
- BCE loss on the correspondence matrix

The loss is purely on correspondences — no pose or scale supervision during training. Scale is recovered analytically at inference by the Sim(3) RANSAC.

### Training data

Pure synthetic Plücker line sets. Each scene:
- **Inlier lines:** direction-clustered 3D lines (see below), with a Sim(3) transform applied
- **Outlier lines:** direction-clustered lines appended to each set with no correspondence
- **Scale:** log-uniform in `[0.3, 3.0]` — equal coverage of compression and expansion
- **Rotation:** uniform on SO(3) (via QR decomposition)
- **Translation:** uniform in `[−1.5, 1.5]³`
- **Lines per scene:** 100 inliers + 30 outliers = 130 total (fixed — all scenes must have the same size so PyTorch's default collate can batch them)

Dataset sizes: 5000 train / 500 validation scenes.

#### Why direction-clustered lines (critical design decision)

The model's KNN path `x[:,3:,:]` computes nearest neighbours by **direction similarity**. For this to give consistent local context across both views, the KNN neighbourhoods must survive the Sim(3) transform.

Under `d′ = R·d`, lines that share a similar direction in cloud 1 share the same rotated direction in cloud 2 — their KNN neighbourhood is **exactly preserved**. This is verified empirically: for a matched pair, 11/10 KNN neighbours overlap between views.

With fully random directions (the naive approach), the direction-KNN neighbourhood of each line in view 1 has no overlap with its neighbourhood in view 2. The GNN sees incoherent local context and cannot learn. Experimentally, random-direction data plateaued at 4% inlier ratio regardless of how long training ran.

The fix (`make_direction_clustered_lines`): group lines around `n_dir_clusters=10` anchor directions with a small angular spread (~8.6°). With 100 inliers and 10 clusters, each line has ~9 within-cluster KNN neighbours — consistent across both views and informative for the GNN.

### Validation metrics

| Metric | Description |
|--------|-------------|
| `recall_rot` | Fraction of scenes with rotation error < 20° |
| `med_rot` | Median rotation error (degrees) |
| `med_trans` | Median translation error |
| `med_scale_err` | Median log-ratio scale error `|log(ŝ / s)|` |
| `avg_inlier_ratio` | Average % of top-K correspondence candidates that are true inliers |

### Training results

Run: `output/sim3_synthetic/2026-04-12/` — 400 epochs, batch size 12, lr=1e-3, gamma=0.99.

| Checkpoint | avg_inlier_ratio | recall_rot | med_rot | med_scale_err |
|---|---|---|---|---|
| `best_val_checkpoint.pth` (epoch 124) | **54.8%** | 1.000 | 0.00° | 0.000 |
| `checkpoint.pth` (epoch 400, final) | ~51% | 1.000 | 0.00° | 0.000 |

Training trajectory summary:

| Epoch | avg_inlier_ratio | Notes |
|-------|-----------------|-------|
| 0 (pre-train) | 0.9% | random initialisation |
| 5 | 7.9% | recall_rot hits 1.000 for first time |
| 17 | 27.4% | surpasses old (random-direction) run's all-time best of 4% |
| 42 | 41.9% | approaches original SE3 PlueckerNet benchmark (43.3%) |
| 55 | 46.4% | surpasses SE3 benchmark |
| **124** | **54.8%** | **peak — best_val_checkpoint.pth saved** |
| 125 | 5.6% | training collapse (single bad gradient step) |
| 400 | ~51% | recovered but never beat epoch 124 |

**Note on training collapse:** The model peaked at epoch 124 then a single epoch's gradient update pushed weights into a bad region (silent — no loss spike visible). For future runs, add gradient clipping before `optimizer.step()`:
```python
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
```

#### Comparison with original SE3 PlueckerNet

| Model | Dataset | avg_inlier_ratio | recall_rot |
|-------|---------|-----------------|------------|
| SE3 PlueckerNet (pre-trained) | Semantic3D (real) | 43.3% | — |
| **Sim3 PlueckerNet (ours)** | Synthetic direction-clustered | **54.8%** | **1.000** |

**Important caveat:** these are not apples-to-apples. The original SE3 model is evaluated on real noisy Semantic3D line reconstructions with variable scene sizes (up to 3000 lines) using RANSAC threshold 0.5. Our Sim3 model is trained and evaluated on simpler synthetic data (130 fixed lines, threshold 0.1). The comparison shows the architecture can handle Sim3, but a fair benchmark would require applying scale augmentation to the real SE3 data.

### Training commands

```bash
conda activate torch5090
cd /home/rueyday/scale-aware-PlueckerNet

# Install extra dependencies (once)
pip install tensorboardX easydict

# Step 1 — generate synthetic dataset (~1 min)
# n_dir_clusters=10 with n_inliers=100 gives 10 lines/cluster,
# matching net_knn=10 so each line's KNN neighbourhood is its whole cluster.
python generate_sim3_dataset.py --out_dir ./dataset --n_train 5000 --n_valid 500 --n_inliers 100 --n_outliers 30

# Step 2 — train in tmux (400 epochs, ~3 hours on a single GPU)
tmux new-session -d -s training "python3 train_sim3.py 2>&1 | tee output/train.log"
tail -f output/train.log
```

Checkpoints and TensorBoard logs: `output/sim3_synthetic/<date>/`

To resume from the best checkpoint with a lower LR (recommended after a collapse):
```python
# in train_sim3.py, set:
configs.resume  = "./output/sim3_synthetic/<date>/best_val_checkpoint.pth"
configs.train_lr = 1e-4   # 10× lower than original
```

### Requires

`../PlueckerNet/` must exist (the original repo, cloned alongside this one). The model, config, and utility files are imported from there directly — no duplication.

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

---

## Codebase walkthrough

### File layout

```
scale-aware-PlueckerNet/
│
│  ── Part 1: failure analysis ──────────────────────────────────────────
├── chess_plueckernet_demo.py     RGBD → voxel → lines → Plücker → register
├── load_scene.py                 habitat-sim scene loader (separate experiment)
├── results/                      pre-generated figures (5 PNGs)
│
│  ── Part 2: Sim(3) training ───────────────────────────────────────────
├── generate_sim3_dataset.py      synthetic Sim(3) Plücker line data generator
├── train_sim3.py                 training entry point
└── sim3/
    ├── ransac.py                 Sim(3) RANSAC — joint (s, R, t) estimation
    ├── dataloader.py             extends original loader to include s_gt
    └── trainer.py                training loop + Sim(3) validation
```

---

### `generate_sim3_dataset.py` — data generation

Generates all training and validation data from scratch. Each call produces two splits (train/valid), each saved as a folder of six `.pkl` files.

**Data format** — each `.pkl` is a Python list of N arrays, one per scene:

| File | Shape per scene | Content |
|------|----------------|---------|
| `plucker1.pkl` | `(n_lines, 6)` float32 | Cloud 1 lines: `[m₀ m₁ m₂ d₀ d₁ d₂]` |
| `plucker2.pkl` | `(n_lines, 6)` float32 | Cloud 2 lines (Sim3-transformed inliers + independent outliers) |
| `matches.pkl` | `(2, n_inliers)` int32 | `[src_indices; tgt_indices]` — which rows of plucker1/plucker2 are true correspondences |
| `R_gt.pkl` | `(3, 3)` float32 | Ground-truth rotation |
| `t_gt.pkl` | `(3, 1)` float32 | Ground-truth translation |
| `s_gt.pkl` | scalar float32 | Ground-truth scale |

**Why fixed scene size?** PyTorch's default `collate_fn` stacks arrays into batched tensors; this requires all scenes in a batch to have the same shape. Every scene has exactly `n_inliers + n_outliers = 130` lines, so the correspondence matrix is always `(130, 130)` and Plücker arrays are always `(130, 6)`.

**Key functions:**

`random_rotation()` — samples a uniformly random rotation from SO(3) using QR decomposition of a random Gaussian matrix. The sign fix on the diagonal of R ensures det(Q)=+1.

`make_direction_clustered_lines(n, n_dir_clusters, dir_spread)` — the primary line generator. Creates `n_dir_clusters` random unit-vector anchors on the sphere, then generates lines whose directions are small Gaussian perturbations (`dir_spread=0.15 rad ≈ 8.6°`) around each anchor. Point positions are drawn uniformly from `[−pos_range, pos_range]³`. The moment is `m = p × d`.

`make_lines(n)` — fully random lines (kept for reference). Not used in current datasets.

`make_clustered_lines(n, n_clusters)` — spatial-anchor clustering (deprecated). Lines near the same position anchor share similar moments but random directions. This was the first attempt and failed: after Sim3 the `t×d′` term scatters lines in moment space since each line has a different direction. Kept for backward compatibility.

`apply_sim3(lines, s, R, t)` — transforms a set of Plücker lines by `(s, R, t)`:
```
d′ = R d
m′ = s · R m + t × d′
```
Operates on `(n, 6)` arrays in `[m, d]` format.

`generate_scene(n_inliers, n_outliers, scale_range, n_dir_clusters)` — assembles one scene:
1. Generates `n_inliers` direction-clustered lines (cloud 1 inliers)
2. Samples a random Sim3 transform; scale is log-uniform over `[0.3, 3.0]`
3. Applies transform to get cloud 2 inliers
4. Generates independent outlier lines for each cloud (fewer direction clusters so they look structurally different from inliers)
5. Concatenates inliers + outliers and shuffles both clouds independently
6. Recovers the post-shuffle inlier indices via `argsort` and stores them as the `matches` array

`generate_split(n_scenes, out_dir, ...)` — calls `generate_scene` N times and serialises the results to `.pkl` files.

---

### `sim3/ransac.py` — Sim(3) RANSAC solver

This is the core geometric solver, replacing `lib/ransac_l2l.py` from the original PlueckerNet.

**Input:** `plucker1`, `plucker2` — `(6, n)` arrays of candidate correspondences in `[m; d]` format (columns are lines).

**Two-stage estimation — same structure as SE(3) RANSAC:**

**Stage 1 — rotation from directions** (`estimate_rotation`):

Directions are scale-invariant (`d′ = R d`), so R can be recovered identically to the SE(3) case. Build the cross-covariance matrix and decompose:
```
M = Σᵢ d₂ᵢ d₁ᵢᵀ   →   M = U Σ Vᵀ   →   R = U Vᵀ
```
If `det(R) < 0` (reflection), flip the last column of U.

**Stage 2 — scale and translation from moments** (`solve_scale_translation`):

Given R, substitute `m₁′ = R m₁` and rearrange the moment equation per line `i`:
```
s · m₁′ᵢ  −  [d₂ᵢ×] · t  =  m₂ᵢ
```
Stack all lines into a `(3n × 4)` linear system `A x = b` where `x = [s, tₓ, t_y, t_z]ᵀ`:
```
A[3i:3i+3, 0]  = m₁′ᵢ          ← coefficient of s
A[3i:3i+3, 1:] = −skew(d₂ᵢ)   ← coefficient of t  (t×d = −[d×]t)
b[3i:3i+3]     = m₂ᵢ
```
Solved by `np.linalg.lstsq`. Two line pairs give 6 equations for 4 unknowns — uniquely determined. Any hypothesis with `s ≤ 0` is discarded.

**RANSAC loop** (`run_ransac_sim3`):
1. Sample 2 correspondences, estimate `(s, R, t)` via the above
2. Score all N candidates: inlier if `‖L₂ − M_sim3 · L₁‖₂ < threshold`
3. Keep the hypothesis with most inliers
4. Refine on the full inlier set using overdetermined LS (`best_fit_sim3`)

**Scoring** (`sim3_motion_matrix`, `score_sim3`): builds the 6×6 Sim(3) motion matrix and computes the L2 residual on the full Plücker 6-vector:
```
M_sim3 = [ s·R    [t×]·R ]     residual = ‖L₂ − M_sim3 · L₁‖₂
          [  0       R   ]
```

---

### `sim3/dataloader.py` — dataset loader

A minimal extension of the original `PluckerData3D_precompute`. The only change is loading the extra `s_gt.pkl` file and returning `s_gt` as a sixth element from `__getitem__`.

**Key detail — sparse to dense matches:** The stored `matches` array is sparse: shape `(2, n_inliers)` listing index pairs. `__getitem__` converts this to a dense binary matrix of shape `(n1, n2)` with 1s at correspondence positions. This dense matrix is what the BCE loss and the `InlierProb` diagnostic both operate on.

```python
matches = np.zeros([n1, n2], dtype=np.float32)
matches[matches_ind[0, :], matches_ind[1, :]] = 1.0
```

**Batching:** Because all scenes have the same `(130, 130)` correspondence matrix and `(130, 6)` Plücker arrays, PyTorch's default `collate_fn` stacks them without any custom collation. The validation loader uses `batch_size=1` (RANSAC operates per-scene); the training loader uses `batch_size=12`.

---

### `sim3/trainer.py` — training and validation

**`Sim3Trainer.__init__`** — loads `PluckerNetKnn` (from `../PlueckerNet/model/model_plucker.py`), sets up Adam optimiser, ExponentialLR scheduler (`gamma=0.99`), and TensorBoard writer. Can resume from a checkpoint by restoring epoch, model weights, optimiser state, and scheduler state.

**`_train_epoch`** — standard supervised training:
1. Loads a batch `(matches, plucker1, plucker2, R_gt, t_gt, s_gt)`
2. Passes `(plucker1, plucker2)` to the model → `prob_matrix, prior1, prior2`
3. Computes `TotalLoss` (BCE on the correspondence matrix)
4. Backward + Adam step
5. Logs `total_loss` and `InlierProb` to TensorBoard

`s_gt` is **not** used during training — scale supervision is not needed because the network only learns correspondences; scale is recovered analytically at inference.

**`InlierProb` diagnostic:**
```python
batch_prob_loss = ((1 - 2 * matches) * prob_matrix).sum(...).mean()
```
For non-inlier pairs (matches=0): contributes `+prob_matrix` (positive). For inlier pairs (matches=1): contributes `−prob_matrix` (negative). As training progresses, this value should decrease from ~+1 (network ignores inliers) toward 0 and below (network assigns mass to inlier pairs).

**`_valid_epoch`** — evaluation pipeline per scene:
1. Forward pass → `prob_matrix`
2. Select top-k=100 candidate pairs from the flattened probability matrix
3. Compute `inlier_ratio` = fraction of selected pairs that are true matches
4. Pass top-k Plücker lines to `run_ransac_sim3` (threshold=0.1)
5. Compute `err_q` (rotation angle), `err_t` (translation L2), `err_s` (log-ratio of scales)

**`_recalls`** — aggregates per-scene results:
- `recall_rot` = mean of cumulative rotation histogram at bins 0°–20° (4 bins of 5°). Value of 1.0 means every scene was solved within 20°.
- `med_rot`, `med_trans`, `med_scale_err` — median errors across scenes
- `avg_inlier_ratio` — mean inlier ratio, the primary training metric

**Checkpoint saving:** Two checkpoints are maintained: `checkpoint.pth` (every epoch, overwritten) and `best_val_checkpoint.pth` (saved only when `avg_inlier_ratio` improves). Always use `best_val_checkpoint.pth` for inference.

**`_normalize_moments` (unused):** A static method that normalises moment magnitudes by the mean norm of cloud 1 and applies the same scale to cloud 2. This was an early attempt to help the network see the scale ratio — it was removed because it was applied inconsistently (only in one view during training) and made performance worse. The method still exists in case it's needed for future experiments.

---

### `train_sim3.py` — entry point

Thin wrapper that:
1. Adds `../PlueckerNet` to `sys.path` so all original model/config/lib files are importable without copying
2. Calls `get_config()` from the original repo to get the full config namespace, then overrides the fields relevant to Sim3 training
3. Creates `DataLoader`s and hands them to `Sim3Trainer`

The model name (`configs.model_nb`) is set to today's date, so each run creates a new folder under `output/sim3_synthetic/`.

---

### Inherited from `../PlueckerNet/` (not duplicated)

| File | Role |
|------|------|
| `model/model_plucker.py` | `PluckerNetKnn` — the full network architecture |
| `config.py` | All hyperparameter defaults via argparse |
| `lib/loss.py` | `TotalLoss` — BCE on the correspondence matrix |
| `lib/utils.py` | `load_model` — imports `PluckerNetKnn` by name |
| `lib/timer.py` | `AverageMeter`, `Timer` — training timing utilities |
| `lib/file.py` | `ensure_dir` |

**`PluckerNetKnn` architecture (from `model/model_plucker.py`):**

```
Input: (B, N, 6) Plücker lines  [m₀ m₁ m₂ d₀ d₁ d₂]
         ↓
conv_in_seq_direction_moment_knn
  ├── KNN on x[:, :3, :]  (moments)  → conv_direction → mlp_direction → (B, 64, N)
  └── KNN on x[:, 3:, :]  (directions) → conv_moment → mlp_moment   → (B, 64, N)
         ↓  concat + mlp_merged
       (B, 128, N)
         ↓
SpatialAttentionalGNN  [self, cross, self, cross, self, cross, self, cross, self, cross, self, cross]
  — 6 self-attention layers (within one cloud)
  — 6 cross-attention layers (between cloud 1 and cloud 2)
  — each layer: MultiHeadedAttention (4 heads) + MLP residual
         ↓
pairwiseL2Dist  →  (B, N1, N2) feature distance matrix
         ↓
prob_mat_sinkhorn  (Sinkhorn optimal transport, 30 iterations)
         ↓
Output: prob_matrix (B, N1, N2)  — soft doubly-stochastic correspondence matrix
        prior1, prior2            — per-line matchability priors
```

**Naming note:** Despite the variable names `x_knn_direction` and `x_knn_moment` in the source, the KNN paths are actually applied to `x[:, :3, :]` (moments) and `x[:, 3:, :]` (directions) respectively — the names in the code are swapped relative to the data content. Our data follows the `[m, d]` format used by the original PlueckerNet and its SE(3) RANSAC (`ransac_l2l.py` uses `plucker[3:, :]` for directions and `plucker[:3, :]` for moments, confirming the convention).

---

## Known issues and future directions

### Synthetic vs real data gap
The synthetic direction-clustered data is structurally simpler than real 3D line reconstructions. The most impactful next step is **scale-augmenting real SE3 pairs** (multiply one cloud's moments by a random `s`) to get Sim3 pairs with real geometric structure:
```python
s = np.exp(np.random.uniform(np.log(0.3), np.log(3.0)))
plucker2[:, :3] *= s   # scale moments; directions unchanged
```

### Training collapse
A single bad gradient step at epoch 125 dropped performance from 54.8% → 5.6%. Fix: **gradient clipping**.
```python
# in sim3/trainer.py _train_epoch, before optimizer.step():
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
```

### No explicit scale signal in the network
The scale `s` is only visible to the network implicitly as a moment magnitude ratio (`‖m₂‖ ≈ s·‖m₁‖`). Injecting `log(‖m₂‖_mean / ‖m₁‖_mean)` as an explicit global feature into the GNN's cross-attention layers would give the network a direct scale hint.

### End-to-end scale supervision
Training supervises correspondences only; scale is recovered post-hoc by RANSAC. Adding a differentiable Sim3 solver in the forward pass and a geometric loss `L_sim3(ŝ, R̂, t̂)` would close the loop and likely improve hard-scene performance.

### Evaluation protocol
The original SE3 PlueckerNet uses:
- **recall_rot** = mean of cumulative rotation histogram at bins 0–20° (4 bins of 5°)
- **RANSAC threshold** = 0.5 for real data (Semantic3D/Structured3D), 0.1 for synthetic
- **Top-k** = min(100, N₁ × N₂) candidate pairs selected from the probability matrix
- **Dataset** = Semantic3D (outdoor LiDAR) or Structured3D (indoor synthetic)

Our Sim3 evaluation uses the same metrics but with synthetic data and threshold 0.1.

---

## References

- **PlueckerNet** — Liu et al., *"PlueckerNet: Learn to Register 3D Line Reconstructions"*, CVPR 2021. [GitHub](https://github.com/Liumouliu/PlueckerNet)
- **7-Scenes dataset** — Shotton et al., *"Scene Coordinate Regression Forests for Camera Relocalization in RGB-D Images"*, CVPR 2013.
- **Sim(3) in SLAM** — Strasdat et al., *"Scale Drift-Aware Large Scale Monocular SLAM"*, RSS 2010.
