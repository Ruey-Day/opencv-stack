# ScalePluckerNet

**ScalePluckerNet** extends [PlueckerNet](https://github.com/Liumouliu/PlueckerNet) (Liu et al., CVPR 2021) from **SE(3)** to **Sim(3)** — jointly recovering rotation R, translation t, *and scale s* from Plücker line correspondences. The primary application is cross-modal SLAM-map registration: matching monocular SLAM line landmarks (scale-ambiguous) to metric RGBD map lines.

---

## Repository Layout

```
ScalePluckerNet/
├── sim3/
│   ├── dataloader.py          # Sim3PluckerData (pkl) + LiveSim3PluckerData (live .db)
│   ├── trainer.py             # Sim3Trainer — Sim(3) RANSAC validation
│   ├── pair_generator.py      # Submap + symmetric pair generation from SLAM .db maps
│   ├── ransac.py              # Sim(3) RANSAC — L2 residual in Plücker space
│   ├── ransac_grassmannian.py # Sim(3) RANSAC — Grassmannian + LO-RANSAC (default)
│   └── __init__.py
│
├── scripts/
│   ├── generate_submap_dataset.py  # Generate offline pkl splits from SLAM .db files
│   ├── combine_datasets.py         # Merge multiple pkl splits into one
│   ├── prep_fastcamo_for_slam.py   # Convert FastCaMo-Real sequences for Structure-PLP-SLAM
│   ├── run_fastcamo_slam.sh        # Run SLAM on all 12 FastCaMo sequences
│   ├── make_sim3_from_se3.py       # Apply random scale to SE3 pkl splits → Sim3 variants
│   └── eval_se3_vs_sim3.py         # Evaluate SE3 pretrained model on SE3 vs Sim3 data
│
├── train.py                   # Unified training entry point (standard + live mode)
│
├── dataset/
│   ├── 7scenes_train/         # 5400 submap pairs from 7-Scenes SLAM maps
│   ├── 7scenes_valid/         # 100 submap pairs (held-out: heads seq02, stairs seq04)
│   ├── fastcamo_train/        # 1800 submap pairs from FastCaMo-Real SLAM maps
│   ├── fastcamo_valid/        # 100 submap pairs (held-out: studio, lounge_1)
│   ├── main_train/            # 7200 combined (7scenes + fastcamo) — current training set
│   ├── main_valid/            # 200 combined — current validation set
│   └── replica_train/         # Replica RGBD (held aside as unseen final evaluation)
│       replica_valid/
│
└── output/
    ├── joint/2026-05-17/      # Best joint pretrained model (avg_inlier_ratio=92.1%)
    └── main/main-finetune/    # Current training run (fine-tune from joint on main dataset)
```

Parent repo `../PlueckerNet/` must exist alongside this repo — all entry points add it to `sys.path` automatically.

---

## Dependencies

```bash
conda activate torch5090
```

Python 3.11, PyTorch 2.6, CUDA. All scripts must be run in `torch5090`.

`../PlueckerNet/` must exist at the same directory level as this repo.

---

## Plücker Line Format

**All Sim3 code uses `[m, d]` order** — moment first, direction last:

```
line = [m0, m1, m2, d0, d1, d2]   shape (6,) or (N, 6)
```

Transformation law under Sim(3) with scale s, rotation R, translation t:
```
d' = R d
m' = s·R·m + t × d'
```

The original PlueckerNet uses `[d, m]` order (direction first). Use `md_to_dm()` when calling into the SE(3) PlueckerNet.

---

## Data Sources & SLAM Maps

### Training strategy

- **Training:** 7-Scenes + FastCaMo-Real (Structure-PLP-SLAM maps → submap pairs)
- **Validation (during training):** held-out sequences from 7-Scenes and FastCaMo
- **Final evaluation (unseen):** Replica RGBD — never seen during training

### FastCaMo-Real pipeline

FastCaMo-Real is an Azure Kinect RGB-D dataset (640×480, 30 fps) with 12 indoor sequences.

**Step 1 — prepare images for SLAM:**

```bash
python scripts/prep_fastcamo_for_slam.py /mnt/crucial/rueyday/data/fastcamo/apartment_1
# Converts RGBA→RGB in-place, int32 depth→uint16 in-place, writes rgb.txt + depth.txt
```

**Step 2 — run Structure-PLP-SLAM:**

```bash
bash scripts/run_fastcamo_slam.sh
# Saves: Structure-PLP-SLAM/fastcamo_{name}_map.db
```

Camera config: `Structure-PLP-SLAM/example/tum_rgbd/FastCaMo_rgbd.yaml`
(fx=fy=320.0, cx=319.5, cy=239.5, depthmap_factor=1000.0)

**SLAM line extractor settings** (modified from defaults for more landmarks):
- `_min_line_length = 0.025` (was 0.08) — allows lines as short as 16 px
- `lineLength >= 20` (was 45) — keep shorter detected segments

**FastCaMo SLAM results** (line landmarks in final .db map):

| Sequence | Lines | Status |
|----------|-------|--------|
| apartment_1 | 227 | ✅ train |
| gym | 96 | ✅ train |
| lounge_2 | 48 | ✅ train |
| meeting_room | 139 | ✅ train |
| stairwell | 47 | ✅ train |
| workshop_1 | 131 | ✅ train |
| studio | 269 | ✅ val |
| lounge_1 | 135 | ✅ val |
| lab | 44 | ⚠️ 94% frames corrupted (disk-full extraction) — excluded |
| office | 6 | ❌ below minimum (SUBMAP_N_MIN=30) |
| workshop_2 | 26 | ❌ below minimum |
| apartment_2 | — | ❌ SLAM segfault (OpenCV assertion in ZMQ publisher) |

### 7-Scenes maps

46 Structure-PLP-SLAM `.db` files covering chess, fire, heads, office, pumpkin, redkitchen, stairs sequences. Val held-out: `heads_seq02`, `stairs_seq04`.

---

## Dataset Generation (submap pairs)

### Submap pair format

Each training sample is an **asymmetric submap registration** pair:

- `plucker1` — small monocular submap (30–120 lines, arbitrary Sim(3) scale)
- `plucker2` — large metric SLAM map (context lines, metric scale)
- All `plucker1` lines have a GT correspondence in `plucker2`
- Coverage: 5–70% of `plucker2` overlaps the submap (adaptive; see below)

### Pair generator fix (coverage enforcement)

Small pools (< 86 lines) cannot generate 30-line submaps at the original COVERAGE_MAX=0.35.
The fix in `sim3/pair_generator.py::generate_submap_pair` adaptively raises coverage for small pools:

```python
min_cov_needed = SUBMAP_N_MIN / len(big_pool)
cov_lo = max(COVERAGE_MIN, min_cov_needed)
cov_hi = max(COVERAGE_MAX, min_cov_needed * 1.2)
coverage_frac = np.random.uniform(cov_lo, cov_hi)
```

Without this fix, 72% of main_train samples had < 30 lines in plucker1 (mean=22.6).
After fix: min=30, mean=36.2, 0% below threshold.

### Generate datasets

```bash
# FastCaMo
python scripts/generate_submap_dataset.py --dataset fastcamo --pairs_per_train 300 --pairs_per_val 50

# 7-Scenes
python scripts/generate_submap_dataset.py --dataset 7scenes --pairs_per_train 300 --pairs_per_val 50

# Combine into main
python scripts/combine_datasets.py --inputs 7scenes fastcamo --output main
```

| Split | Pairs | Source |
|-------|-------|--------|
| `main_train` | 7200 | 5400 (7-Scenes) + 1800 (FastCaMo) |
| `main_valid` | 200 | 100 + 100 |
| `fastcamo_train` | 1800 | 6 FastCaMo maps × 300 |
| `fastcamo_valid` | 100 | studio + lounge_1 × 50 |
| `7scenes_train` | 5400 | 18 maps × 300 |
| `7scenes_valid` | 100 | heads_seq02 + stairs_seq04 × 50 |

### Dataset pkl format

| File | Shape per sample | dtype |
|------|-----------------|-------|
| `matches.pkl` | `(2, n_inliers)` | int32 |
| `plucker1.pkl` | `(n1, 6)` variable | float32 |
| `plucker2.pkl` | `(n2, 6)` variable | float32 |
| `R_gt.pkl` | `(3, 3)` | float32 |
| `t_gt.pkl` | `(3, 1)` | float32 |
| `s_gt.pkl` | scalar | float32 |

---

## Training

### Current run: main-finetune

```bash
nohup /home/rueyday/miniconda3/envs/torch5090/bin/python train.py \
    --dataset main \
    --pretrain output/joint/2026-05-17/best_val_checkpoint.pth \
    --lr 1e-4 --gamma 0.999 --epochs 500 \
    --batch 1 --iter_size 32 \
    --ransac grassmannian \
    --name main-finetune \
    >> output/main_finetune.log 2>&1 &
```

Monitor: `tail -f output/main_finetune.log`

### Metric interpretation on main_valid

The evaluator picks top-100 matches (`k=100`). With average n1≈36 GT inliers per sample, the theoretical ceiling is ~36% avg_inlier_ratio (all GT matches in top-100). Random baseline ≈ 1%.

### General training flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `joint` | pkl split name under `./dataset/` |
| `--pretrain` | — | Warm-start checkpoint (`strict=False`) |
| `--resume` | — | Resume (restores optimizer + scheduler) |
| `--epochs` | `1000` | |
| `--batch` | `1` | Use 1 for variable-length pairs |
| `--iter_size` | `32` | Gradient accumulation; effective batch = `batch × iter_size` |
| `--lr` | `5e-4` | |
| `--gamma` | `0.99` | ExponentialLR per epoch; `1.0` = constant |
| `--ransac` | `grassmannian` | Val RANSAC: `sim3` \| `grassmannian` |
| `--name` | today | Checkpoint dir: `output/<dataset>/<name>/` |

---

## Sim(3) RANSAC

Two backends in `sim3/ransac.py` and `sim3/ransac_grassmannian.py`.

**Minimal solver (both backends):**
1. Estimate R from direction pairs via SVD (Wahba problem)
2. Solve for (s, t) jointly from moment equations via 3n×4 least squares

**Grassmannian backend** (default, `--ransac grassmannian`) adds:
- Stratified direction-bin sampling to avoid degenerate all-parallel minimal sets
- LO-RANSAC: up to 10 local-optimisation re-fits on growing inlier sets

```python
R, t, s, inlier_mask, n_inliers = ransac_sim3(
    plucker1.T,   # (6, N) — columns are lines
    plucker2.T,
    inlier_threshold=0.3,
)
```

---

## Known Pitfalls

**Small pools and submap size.** Pools with < 86 lines require adaptive coverage (> 35%) to generate 30-line submaps. The `generate_submap_pair` function handles this automatically since the 2026-06-11 fix.

**`torch.load` weights_only error on resume.** Checkpoints store `EasyDict`. Fix: `torch.load(..., weights_only=False)`. Already applied in `sim3/trainer.py`.

**InlierProb sign.** Starts positive at init, should go negative as training converges. Negative means the network assigns more mass to true inliers.

**SE(3) solver fails when scale ≠ 1.** The original PlueckerNet RANSAC absorbs `(s−1)·Rm₁` into a spurious translation, producing rotation errors up to ~90°. Core motivation for this Sim(3) extension.

---

## SE(3) Pretrained Model Fails on Sim(3) Data

**Script:** `scripts/eval_se3_vs_sim3.py`

**Setup:** Apply `scripts/make_sim3_from_se3.py` to the original PlueckerNet validation splits to produce Sim3 variants (random scale s ∈ [0.1, 10], log-uniform). Run the PlueckerNet pretrained checkpoints on SE3 and Sim3 versions of the same scenes. Metric: `avg_inlier_ratio` (fraction of top-100 predicted correspondences that are GT matches).

| Split | avg inlier ratio | med inlier ratio | n |
|-------|-----------------|-----------------|-----|
| semantic3D  SE3  (s=1, original) | **46.18%** | 42.50% | 298 |
| semantic3D  Sim3 (random s) | **30.54%** | 23.50% | 298 |
| structured3D SE3  (s=1, original) | **81.44%** | 85.00% | 525 |
| structured3D Sim3 (random s) | **47.06%** | 49.00% | 525 |

**Drop:** −15.6 pp on semantic3D, −34.4 pp on structured3D. The same GT matches exist in both versions — only a scale factor is added to plucker1 moments. The pretrained SE(3) model nearly halves its matching quality on structured3D under scale variation, confirming the core motivation for the Sim(3) extension.
