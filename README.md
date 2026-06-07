# ScalePluckerNet

**ScalePluckerNet** extends [PlueckerNet](https://github.com/Liumouliu/PlueckerNet) (Liu et al., CVPR 2021) from **SE(3)** to **Sim(3)** — jointly recovering rotation R, translation t, *and scale s* from Plücker line correspondences. The primary application is cross-modal SLAM-map registration: matching monocular SLAM line landmarks (scale-ambiguous) to metric RGBD reconstructions.

[Website](https://rueyday.github.io/ScalePluckerNet/) &nbsp;·&nbsp; [Dataset (Dropbox)](https://www.dropbox.com/scl/fo/34o03nsdztz3fpxrwzhty/ALP0MX8KOvdEDx8fg_Wfd9I?rlkey=qzo08vwuqo4jwt5nrrsffb6t3&st=9xaxvt1b&dl=0) &nbsp;·&nbsp; [Model weights (Dropbox)](https://www.dropbox.com/scl/fo/1knswbb20t9pjug00vim7/ALfsafw208mQSmSyzAvnjIU?rlkey=spq78nh6ofobjsk1abuhy86ry&st=06gqdibi&dl=0) &nbsp;·&nbsp; [Google Colab](https://colab.research.google.com/drive/1_AWdfnJjmsteVT_lM4dakYTecn1gpRc_?usp=sharing)

---

## Repository Layout

```
ScalePlueckerNet/
├── sim3/
│   ├── dataloader.py          # Sim3PluckerData (pkl) + LiveSim3PluckerData (live .db)
│   ├── trainer.py             # Sim3Trainer — Sim(3) RANSAC validation, curriculum hook
│   ├── pair_generator.py      # Live pair generation from SLAM .db maps + curriculum schedule
│   ├── ransac.py              # Sim(3) RANSAC — L2 residual in Plücker space
│   ├── ransac_grassmannian.py # Sim(3) RANSAC — L2 metric + stratified sampling + LO-RANSAC (default)
│   └── __init__.py
│
├── scripts/
│   ├── convert_se3_datasets.py          # Step 1: convert semantic3D/structured3D pkl format
│   ├── generate_se3real_sim3_dataset.py # Step 2a: scale-augment SE3 real datasets
│   ├── generate_replica_gs_dataset.py   # Step 2b: Replica RGBD world-space GlueStick lines
│   ├── generate_7scenes_gs_dataset.py   # Step 2c: 7-Scenes RGBD world-space GlueStick lines
│   ├── _pair_gen.py                     # Shared pair generation utilities
│   ├── combine_joint_dataset.py         # Step 3: merge all sources into joint split
│   ├── generate_val.py                  # Generate slam_map_valid from .db files
│   └── eval.py                          # Evaluation entry point
│
├── train.py                   # Unified training entry point (standard + live mode)
│
├── dataset/
│   ├── slam_map_valid/        # 800 scenes — SLAM-map cross-modal pairs (200 lines, scale 0.1–9.7)
│   ├── 7scenes_valid/         # 800 scenes — 7-Scenes GlueStick world-space pairs
│   ├── replica_valid/         # 800 scenes — Replica GlueStick world-space pairs
│   └── maps/                  # 71 × Structure-PLP-SLAM .db map files (7-Scenes)
│
├── output/
│   ├── joint/2026-05-17/      # Best joint model (recall_rot=0.999, avg_inlier_ratio=92.1%)
│   ├── slam_map/2026-05-20-slam-maps-v14/  # Best SLAM-map fine-tune
│   └── joint/scratch-v5/     # From-scratch live training (in progress)
│
└── results/                   # Evaluation JSON outputs and figures
```

Parent repo `../PlueckerNet/` must exist alongside this repo — all entry points add it to `sys.path` automatically.

---

## Dependencies

### Conda environment

```bash
conda env create -f environment.yml   # Ubuntu 24.04, RTX 5090
conda activate torch5090
```

Python 3.11, PyTorch 2.6, CUDA. All scripts must be run inside `torch5090` — the base env has a numpy 1.x/2.x mismatch with cv2.

### PlueckerNet

`../PlueckerNet/` must exist at the same directory level as this repo.

### GlueStick (offline dataset generation only)

Required at `/home/rueyday/scale-aware-cross-modal-registration/GlueStick`. Run on **CPU only** (`SPWireframeDescriptor.to('cpu')`). Only the `['lines']` output is used.

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

## Dataset Pipeline

There are two data modes: **offline pkl** (pre-generated static splits) and **live** (on-the-fly from SLAM `.db` map files).

### Offline pkl datasets (joint model)

```bash
# Step 1 — convert existing SE3 pkl datasets to [m,d] format
python scripts/convert_se3_datasets.py

# Step 2 — generate Sim(3) variants from each source
python scripts/generate_se3real_sim3_dataset.py   # → dataset/se3real_sim3_{train,valid}/
python scripts/generate_replica_gs_dataset.py &   # → dataset/replica_gs_{train,valid}/
python scripts/generate_7scenes_gs_dataset.py &   # → dataset/7scenes_gs_{train,valid}/

# Step 3 — merge into joint split (filters degenerate scenes automatically)
python scripts/combine_joint_dataset.py
# → dataset/joint_{train,valid}/
```

After filtering (scale < 0.1 or < 5 GT inliers removed):

| Split | Scenes |
|-------|--------|
| `joint_train` | 24,999 |
| `joint_valid` | 1,447 |

### Live datasets (SLAM-map model)

No pre-generation needed. Pass `.db` map files directly to `train.py --mode live`.

Currently available `.db` maps under `../Structure-PLP-SLAM/`:

| Source | Files | Notes |
|--------|-------|-------|
| 7-Scenes | 46 | chess, fire, heads, office, pumpkin, redkitchen, stairs |
| Replica | 9 | office0–4 (GT), room0–1 (map + mono) |
| TUM RGB-D | 16 | freiburg1 desk, freiburg2 xyz, freiburg3 long office |
| KITTI | 1 | sequence 00 (outdoor) |
| S3E | 16+ | multi-robot indoor sequences (see below) |
| **Total** | **88+** | |

### S3E multi-robot dataset

[S3E](https://github.com/PengYu-Team/S3E) is a large-scale multi-robot SLAM dataset with three robots (Alpha, Bob, Carol) each carrying synchronized stereo cameras, IMU, and LiDAR. 18 sequences across two versions cover diverse indoor/outdoor environments.

**Processing pipeline** (`scripts/process_s3e.py`):

1. Extract stereo frames from ROS2 `.db3` bags via CDR message parsing (JPEG → grayscale PNG, EuRoC format)
2. Run `run_euroc_slam_with_line` to produce a `.db` map file per robot-sequence
3. Delete temp frames

```bash
# Process all indoor sequences for all robots:
python scripts/process_s3e.py \
    --sequences S3E_Laboratory_1 S3E_Laboratory_2 S3E_Laboratory_3 S3E_Laboratory_4 \
                S3E_Library_1 S3E_Teaching_Building_1 S3E_Dormitory_1 \
    --skip-existing

# Note: S3E_Tunnel_1 and S3E_Library_2 are LiDAR-only bags (no stereo cameras) — skipped automatically
```

**Per-robot calibration:** Each robot has distinct stereo intrinsics. Using Alpha's calibration for Bob/Carol leaves line triangulation with wrong rectification, producing near-zero line landmarks. Three separate YAML configs are provided:

| Robot | Config | fx | focal_x_baseline |
|-------|--------|----|-----------------|
| Alpha | `scripts/S3E_stereo.yaml` | 1175.51 | 423.18 |
| Bob | `scripts/S3E_stereo_bob.yaml` | 1200.35 | 432.13 |
| Carol | `scripts/S3E_stereo_carol.yaml` | 1192.07 | 429.14 |

Post-rectification parameters computed via `cv2.stereoRectify` from `S3Ev1/Calibration/{alpha,bob,carol}.yaml`. S3E_stereo.py uses per-robot YAML automatically.

**Usable maps** (≥ 30 line landmarks, needed for submap training):

| Sequence | Alpha | Bob | Carol |
|----------|-------|-----|-------|
| Laboratory 1–4 | ✓ | ✓ (recalib) | ✓ (recalib) |
| Library 1 | ✓ | ✓ (recalib) | ✓ (recalib) |
| Teaching Building 1 | ✓ | ✓ (recalib) | ✓ (recalib) |
| Dormitory 1 | ✓ | ✓ (recalib) | ✓ (recalib) |

**Structure-PLP-SLAM bug fixes** required for S3E maps to save correctly:
- `landmark_line.cc`: added null guard for `_ref_keyfrm` in `update_information()` and `to_json()`
- `map_database.cc`: removed `lm_line->update_information()` call inside `to_json()` — it deadlocks when `_mtx_observations` is held and `keyframe._mtx_pos` is contested; the call updates only the distance cache which is not serialized anyway

The `dataset/maps/` folder contains an additional 71 7-Scenes `.db` files used for validation set generation.

### Validation set generation

```bash
# Regenerate slam_map_valid from .db files (800 pairs, seed=42):
python scripts/generate_val.py \
    --db dataset/maps/*.db \
    --n 800 --name slam_map
```

### Dataset format (pkl splits)

Each split is a directory of 6 pickle files (lists of numpy arrays):

| File | Shape per sample | dtype |
|------|-----------------|-------|
| `matches.pkl` | `(2, n_inliers)` — row 0 = src indices, row 1 = tgt indices | int32 |
| `plucker1.pkl` | `(N_TOTAL, 6)` | float32 |
| `plucker2.pkl` | `(N_TOTAL, 6)` | float32 |
| `R_gt.pkl` | `(3, 3)` | float32 |
| `t_gt.pkl` | `(3, 1)` | float32 |
| `s_gt.pkl` | scalar (`0.0` = zero-overlap / no valid pose) | float32 |

**Live pairs have variable line counts** — no fixed cap. Sizes are determined by pool size, overlap fraction, and a random outlier ratio. Use `batch_size=1` (default) with `iter_size=32` for gradient accumulation, or `batch_size>1` with the automatic `variable_collate` zero-padding.

---

## Training

### Standard mode (offline pkl)

```bash
# Train on pre-generated pkl dataset:
python train.py --dataset joint --batch 1 --iter_size 32 --lr 5e-4 --gamma 0.99

# Resume a run:
python train.py --dataset joint --resume output/joint/<name>/checkpoint.pth
```

### Live mode — symmetric pairs (original)

```bash
# From scratch, all available maps:
python train.py --mode live \
    --db_train ../Structure-PLP-SLAM/*.db \
    --val_dataset slam_map \
    --lr 5e-4 --gamma 0.99 --epochs 1000

# Fine-tune from joint checkpoint:
python train.py --mode live \
    --db_train ../Structure-PLP-SLAM/*.db \
    --val_dataset slam_map \
    --lr 1e-5 --gamma 1.0 --epochs 300 \
    --pretrain output/joint/2026-05-17/best_val_checkpoint.pth
```

### Live mode — submap pairs (new asymmetric scenario)

Registers a small monocular submap (30–120 lines, arbitrary scale) against a large
metric SLAM map (100–500 lines). Only 5–35 % of the big map overlaps with the submap;
the remainder are realistic context lines. Pairs are variable size — no fixed cap.

```bash
# Train from scratch with alternating attention model:
python train.py --mode live --submap --model alt \
    --db_train ../Structure-PLP-SLAM/*.db \
    --val_dataset slam_map \
    --lr 5e-4 --gamma 0.99 --epochs 1000

# Warm-start from the joint checkpoint:
python train.py --mode live --submap --model alt \
    --db_train ../Structure-PLP-SLAM/*.db \
    --val_dataset slam_map \
    --lr 1e-4 --gamma 0.995 --epochs 1000 \
    --pretrain output/joint/2026-05-17/best_val_checkpoint.pth
```

### All training flags

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `standard` | `standard` = pkl datasets; `live` = on-the-fly from `.db` files |
| `--dataset` | `joint` | `joint \| slam_map \| replica_gs \| 7scenes_gs \| se3real_sim3` |
| `--val_dataset` | same as `--dataset` | Override validation split |
| `--data_dir` | `./dataset` | Dataset root |
| `--n_lines` | `700` | Lines per scene after subsampling (use `200` for live/slam_map mode) |
| `--n_inliers` | `490` | GT inliers per scene (use `60` for live/slam_map mode) |
| `--db_train` | — | `[live]` Path(s) to `.db` map files; shell globs expanded |
| `--db_val` | same as `--db_train` | `[live]` `.db` files for validation; ignored if static split exists |
| `--epoch_size` | `16000` | `[live]` Pairs generated per epoch |
| `--val_size` | `400` | `[live]` Val pairs when no pkl split exists |
| `--inter_map_ratio` | `0.3` | `[live]` Fraction of cross-map pairs |
| `--submap` | off | `[live]` Use asymmetric submap generator (big-map→small-submap) |
| `--epochs` | `1000` | |
| `--batch` | `1` | Batch size. Variable-length pairs require `batch=1` or `variable_collate` (auto) |
| `--iter_size` | `32` | Gradient accumulation steps; effective batch = `batch × iter_size` |
| `--lr` | `5e-4` | Initial learning rate |
| `--gamma` | `0.99` | ExponentialLR decay per epoch; use `1.0` to disable |
| `--cosine_lr` | off | CosineAnnealingWarmRestarts(T_0=50, T_mult=2) instead of ExponentialLR |
| `--gpu` | `0` | |
| `--workers` | `8` | Reduce to 4 when running multiple jobs |
| `--in_channel` | `6` | `6` = geometry only; `9` = Plücker + LAB color |
| `--ransac` | `grassmannian` | Validation RANSAC: `sim3` \| `grassmannian` |
| `--metric` | `avg_inlier_ratio` | Checkpoint criterion: `avg_inlier_ratio` only |
| `--pretrain` | — | Warm-start from checkpoint (`strict=False`) |
| `--resume` | — | Resume from checkpoint (restores optimizer + scheduler) |
| `--name` | today's date | Run name; controls checkpoint path `output/<dataset>/<name>/` |
| `--model` | `knn` | Architecture: `knn` = original PluckerNetKnn; `alt` = asymmetric alternating attention |
| `--alt_n_blocks` | `3` | `[alt]` Number of alternating attention blocks (3 → same param count as `knn`) |

### Curriculum learning (live mode)

`LiveSim3PluckerData` supports an adaptive overlap curriculum driven by the model's current `avg_inlier_ratio` on the validation set. At the start of each epoch the trainer calls `dataset.set_curriculum_phase(curriculum_ir / 100.0)`:

| Phase fraction | Avg overlap | Description |
|----------------|-------------|-------------|
| 0.0 (IR = 0%) | 0.60 | Dense pairs dominate — strong gradient signal for random init |
| 0.3 (IR = 30%) | 0.51 | Balanced |
| 1.0 (IR = 100%) | 0.41 | Full distribution including zero-overlap (hardest) |

This is automatic when using `--mode live`. No flag needed.

---

## Architecture Variants

### Baseline: `PluckerNetKnn` (`--model knn`, default)

The original architecture from Liu et al. CVPR 2021, extended to Sim(3). A shared `SpatialAttentionalGNN` alternates self- and cross-attention layers over both Plücker sets. Each layer uses a **single** `AttentionalPropagation` module whose weights are shared between the SLAM-side and metric-side updates, making the model symmetric with respect to the two inputs.

### Asymmetric Alternating Attention: `PluckerNetKnnAlt` (`--model alt`)

Inspired by the geometric alternating attention in FUSER (Jiang et al., CVPR 2026). Replaces the shared GNN with `AsymmetricAlternatingGNN`, which stacks `n_blocks` blocks each containing **four independent** `AttentionalPropagation` modules:

| Module | Query | Key/Value | Purpose |
|--------|-------|-----------|---------|
| `self0` | SLAM lines | SLAM lines | Within-source geometric context |
| `self1` | metric lines | metric lines | Within-target geometric context |
| `cross0` | SLAM lines | metric lines | Source reads target for scale/pose signal |
| `cross1` | metric lines | SLAM lines | Target reads source to localise matches |

**Why it matters:** SLAM lines (noisy, arbitrary scale, sparse) and metric map lines (accurate, metric, dense) are structurally different. Sharing weights forces the network to use the same representation for both roles. Separate modules let each direction learn its own statistical pattern.

**Parameter parity:** `--alt_n_blocks 3` gives 3 × 4 = 12 `AttentionalPropagation` modules, identical to the 12 in the default `knn` model — same parameter count (2.24M), different architecture.

```bash
# Fair comparison (same param count):
python train.py --dataset joint --model alt

# More expressive (2× GNN params):
python train.py --dataset joint --model alt --alt_n_blocks 6
```

---

## Evaluation

```bash
# Evaluate on one or more val splits:
python scripts/eval.py \
    --checkpoint output/joint/2026-05-17/best_val_checkpoint.pth \
    --dataset slam_map,7scenes,replica \
    --ransac grassmannian

# Single dataset, custom threshold:
python scripts/eval.py \
    --checkpoint output/slam_map/2026-05-20-slam-maps-v14/best_val_checkpoint.pth \
    --dataset slam_map --threshold 0.5

# Grassmannian RANSAC backend:
python scripts/eval.py --checkpoint ... --dataset slam_map --ransac grassmannian
```

Results saved to `results/eval/<label>.json`. When multiple datasets are passed, a summary table is printed at the end.

**Overlap buckets:** `no_overlap (0%)` = 0 GT inliers; `sparse (~30%)` = 1–200 inliers; `dense (~70%)` = 201+ inliers.

### Eval flags

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | required | Path to `.pth` checkpoint |
| `--dataset` | `slam_map` | Comma-separated val split names |
| `--data_dir` | `./dataset` | Dataset root |
| `--ransac` | `sim3` | `sim3` \| `grassmannian` |
| `--threshold` | `0.3` | RANSAC inlier threshold |
| `--n_iter` | `500` | RANSAC iterations |
| `--max_pairs` | `800` | Max val scenes to evaluate |
| `--out_dir` | `results/eval` | Directory for JSON output |
| `--label` | auto | Human-readable label for output filename |

---

## Training Run History

### 2026-05-12 — Joint 6D v1 *(data leakage — do not use)*

- Dataset: `joint_train` (35,658 scenes = Replica GS + 7-Scenes GS + TUM RGB-D GS + se3real)
- **TUM RGB-D validation scenes leaked into training.** Metrics are inflated.
- Apparent best: recall_rot=0.999, avg_inlier_ratio=92.1% — not trustworthy.
- Checkpoint at `output/joint/2026-05-12/` is archived but should not be used for evaluation.

### 2026-05-14 — Joint 6D v2 *(clean baseline)*

- Dataset: `joint_train` (27,658 scenes) — TUM RGB-D removed to fix leakage.
- `ExponentialLR`, `avg_inlier_ratio` checkpoint criterion, standard Sim(3) RANSAC for validation.
- Best checkpoint: `output/joint/2026-05-14/best_val_checkpoint.pth` (epoch 297)
- recall_rot=0.481, avg_inlier_ratio=72.246% on `joint_valid` (1,523 scenes)

### 2026-05-17 — Joint 6D v3 *(current best joint model)*

Three improvements over v2:

1. **Degenerate scene filter** — removed scenes with GT scale < 0.1 or < 5 GT inliers → 24,999 train / 1,447 valid scenes
2. **Checkpoint criterion → `recall_rot`** — saves the epoch with the best RANSAC success rate, not just correspondence quality
3. **Grassmannian RANSAC for validation** — scale/translation-invariant inlier test; stratified direction-bin sampling; Tikhonov scale regularisation

- Dataset: `joint_train` (24,999 scenes, filtered)
- Warm-started from 2026-05-14 checkpoint with `--cosine_lr`
- Best checkpoint: `output/joint/2026-05-17/best_val_checkpoint.pth`
- **recall_rot=0.999, avg_inlier_ratio=92.112%** on `joint_valid` (1,447 scenes)

### 2026-05-19/20 — SLAM-map fine-tuning chain (v8 → v14)

Iterative fine-tuning from the joint/2026-05-17 checkpoint on live SLAM-map pairs (71 `.db` files, 7-Scenes only). Each version starts from the previous best.

| Run | LR | Weights from |
|-----|----|--------------|
| v1–v7 | 1e-3 → 1e-4 | joint/2026-05-17 |
| v8 | 1e-4 | v5 |
| v9–v13 | 1e-4 → 1e-5 | v8 |
| **v14** | **1e-5** | **v13** |

- All runs: `normalize_n_lines=200`, `ransac_type=sim3`, `exp_gamma=1.0` (constant LR)
- Best checkpoint: **`output/slam_map/2026-05-20-slam-maps-v14/best_val_checkpoint.pth`**
- Val set: `slam_map_valid` (800 scenes, 200 lines, scale range 0.10–9.73, median 8 inliers)

### 2026-06-04 — submap-knn (live submap training)

First run with the asymmetric **submap registration** scenario: plucker1 = small monocular submap (30–120 lines, arbitrary scale), plucker2 = large metric SLAM map (100–500 lines). Key changes:

- `--submap` flag: uses `generate_submap_pair` instead of symmetric pairs
- `--model knn`: original PluckerNetKnn (alt model was slower with no benefit)
- `epoch_size=32000` (1000 steps/epoch), `iter_size=32`
- Live validation from 4 held-out TUM RGB-D freiburg maps
- Warm-started from `slam_map/2026-05-20-slam-maps-v14`
- **74 → 78 `.db` map files** as S3E Alpha maps were added mid-run

| Phase | Epochs | Best avg_inlier_ratio | Notes |
|-------|--------|----------------------|-------|
| submap-knn (run4) | 0–32 | 22.4% (ep 23) | initial run, epoch_size=16000 |
| submap-knn-s3e (run5) | 0–228 | **24.845%** (ep 156) | epoch_size=32000, 51 pools |

Checkpoint: `output/joint/2026-06-04-submap-knn-s3e/best_val_checkpoint.pth`

### 2026-06-07 — submap-knn-s3e-v2 *(current run)*

Restarted from 24.845% checkpoint after diagnosing S3E calibration issue:

- **Per-robot calibration fix**: Bob and Carol maps re-generated with correct intrinsics (previous maps used Alpha's calibration for all robots, producing near-zero line pools)
- Pretrain: `output/joint/2026-06-04-submap-knn-s3e/best_val_checkpoint.pth` (24.845%)
- `lr=5e-6` (reduced from 1e-5 given prior decay to 3.2e-6)
- **88 `.db` map files** once Bob/Carol S3E maps complete processing

Automated restart watcher (`scripts/restart_after_s3e.sh`) will launch a final run (`2026-06-07-submap-knn-s3e-full`) once all Bob/Carol maps are ready.

Monitor: `tail -f output/submap_run6.log`

### 2026-05-27/28 — scratch-v5 *(from-scratch live training, in progress)*

First attempt to train entirely from random initialization on live SLAM-map data with the full set of 74 `.db` files (7-Scenes + Replica + TUM RGB-D + KITTI).

**Key changes introduced for this run:**
- Curriculum learning: overlap distribution adapts from dense→sparse as `avg_inlier_ratio` improves
- `gamma=0.99` LR decay (previous fine-tuning used `gamma=1.0`)
- All 74 `.db` files (vs 71 7-Scenes-only in the fine-tuning runs)
- `--metric avg_inlier_ratio` for checkpoint selection
- `--cosine_lr` after plateau at epoch 588 to attempt LR reset

**Phase 1** (epochs 0–147, ExponentialLR, lr=5e-4→~3e-4):
- Launched: 2026-05-27
- Weights: random init

**Phase 2** (epochs 148–815, ExponentialLR continued):
- Restarted with `--metric avg_inlier_ratio` using phase-1 weights as pretrain
- Best at epoch 588: **avg_inlier_ratio=18.06%**, recall_rot≈0.29
- LR decayed to ~1.4e-7 by epoch 815 — model stalled

**Phase 3** (epoch 0+, CosineAnnealingWarmRestarts, lr reset to 5e-4) *(current — epoch 408 as of 2026-05-29)*
- Restarted from epoch-588 best checkpoint with `--cosine_lr`
- Best so far: epoch 336, **avg_inlier_ratio=18.421%**, recall_rot≈0.30
- Recent epochs (400–408) plateauing ~17.5%; best not yet beaten
- Checkpoint: `output/joint/scratch-v5/best_val_checkpoint.pth`
- Monitor: `tail -f output/scratch_v5.log`

```bash
# Current training command:
nohup /home/rueyday/miniconda3/envs/torch5090/bin/python train.py \
    --mode live \
    --db_train /home/rueyday/scale-aware-cross-modal-registration/Structure-PLP-SLAM/*.db \
    --val_dataset slam_map \
    --lr 5e-4 --gamma 0.99 --batch 32 --epochs 1000 \
    --metric avg_inlier_ratio --cosine_lr \
    --pretrain output/joint/scratch-v5/best_val_checkpoint.pth \
    --name scratch-v5 >> output/scratch_v5.log 2>&1 &
```

---

## Key Results

### Joint model — cross-dataset evaluation

Best checkpoint: `output/joint/2026-05-17/best_val_checkpoint.pth`

| Val set | recall_rot | avg_inlier_ratio |
|---------|-----------|-----------------|
| joint_valid (1,447 scenes) | **0.999** | **92.11%** |

### SLAM-map model — val set breakdown

Best checkpoint: `output/slam_map/2026-05-20-slam-maps-v14/best_val_checkpoint.pth`
Val set: `slam_map_valid` (800 scenes, 200 lines/side, scale 0.1–9.7×)

| Val split | Lines | Inliers (median) |
|-----------|-------|-----------------|
| slam_map_valid | 200 | 8 (~4%) |
| 7scenes_valid | 200 | 5 (~2.5%) |
| replica_valid | 200 | 18 (~9%) |

### From-scratch training — scratch-v5

Val set: `slam_map_valid`. Comparison once phase 3 completes:

| Model | avg_inlier_ratio | recall_rot | Notes |
|-------|-----------------|-----------|-------|
| Joint/2026-05-17 (pretrained) | 92.11% | 0.999 | pkl joint data |
| SLAM-map v14 (fine-tuned) | — | — | fine-tuned from joint |
| scratch-v5 best so far | **18.421%** | 0.30 | epoch 336, from random init, phase 3 ongoing |

The scratch-v5 gap vs pretrained confirms that basic Sim(3) geometry reasoning (learned on the joint pkl data) is the primary contribution of pretraining. The curriculum and LR reset (phase 3) are ongoing attempts to close this gap without pkl data.

---

## Sim(3) RANSAC

### L2 backend (`sim3/ransac.py`)

Minimal 2-correspondence solver:
1. Estimate R from direction pairs via SVD
2. Solve for (s, t) jointly from moment equations via 3n×4 least squares

```python
s, R, t, n_inliers, mask = run_ransac_sim3(
    plucker1.T,   # (6, N) — columns are lines
    plucker2.T,
    inlier_threshold=0.1,
)
# Returns (None, None, None, 0, None) on failure
```

### Grassmannian backend (`sim3/ransac_grassmannian.py`)

Structurally richer solver that shares the same L2 inlier metric as `ransac.py` but adds:
- **Stratified direction-bin sampling** — partitions correspondences by dominant direction axis and draws one line per bin, preventing degenerate all-parallel minimal sets
- **Iterative local optimisation (LO-RANSAC)** — after the main loop, re-fits `(R, s, t)` on the current inlier set up to `lo_iters=10` times, each time expanding the inlier set; uses the full joint LS solver so refinement is not biased by translation
- More iterations by default (`n_iter=5000`, `early_exit_iters=200`)

Default for training validation and evaluation. On the seq02→seq01 benchmark (GT scale 2.7947) it achieves **0.0% scale error** vs 1.7% for the L2 backend at the same inlier count.

```python
R, t, s, inlier_mask, n_inliers = ransac_sim3(
    plucker1.T,   # (6, N) — columns are lines
    plucker2.T,
    inlier_threshold=0.3,
)
# s=0.0 / n_inliers=0 on failure
```

---

## Known Pitfalls

**`gamma=0.99` is too aggressive for 1000 epochs.** LR reaches ~1e-7 by epoch 800. Use `--cosine_lr` for long runs, or `--gamma 0.999`.

**`normalize_n_lines` is a downsampling cap, not a target size.** It only fires when a scene has *more* lines than the threshold. Live-generated pairs always have exactly 200 lines — `normalize_n_lines` is a no-op for them regardless of value.

**Curriculum only applies in live mode.** The `set_curriculum_phase` hook on `LiveSim3PluckerData` is silently ignored for standard pkl datasets (the trainer checks `hasattr`).

**`torch.load` weights_only error on resume.** Checkpoints store an `EasyDict` config object. PyTorch 2.6 changed the default to `weights_only=True`. Fix: `torch.load(..., weights_only=False)`. Already applied in `sim3/trainer.py`.

**InlierProb sign.** `InlierProb` starts positive (~+1) at random init and goes negative as training progresses. Negative is correct — it means the network assigns more mass to true inliers. Target: approaching −1.

**SE(3) solver fails when scale ≠ 1.** The original PlueckerNet RANSAC absorbs `(s−1)·Rm₁` into a spurious translation, giving wrong rotation error up to ~90°. This is the core motivation for the Sim(3) extension.

---

## `ransac_grassmannian.py` — Bug Fixes (2026-05-29)

Four bugs were identified and fixed. All are now resolved in the current code.

**1. Grassmannian inlier metric was scale-blind (critical)**

The original inlier test normalized both Plücker vectors to unit norm before computing `arccos(|L1·L2|)`. Normalizing erases the moment magnitude, which carries the scale information — a hypothesis with `s=0.36` scores identically to `s=2.8` whenever the normalized directions align. Result: RANSAC consistently returned inverted or near-zero scales.

*Fix:* Replaced with the unnormalized L2 residual `‖transform(L1) − L2‖` (same metric as `ransac.py`). The threshold parameter was renamed from `inlier_angle_rad` to `inlier_threshold`.

**2. 3-line minimal solver with direction sign-flip produced wrong scales (critical)**

`solve_rotation` flipped source directions `d1` to align signs with `d2` before computing the cross-covariance. The subsequent `solve_translation_scale` then computed `Rd1 = R @ d1_original` (not sign-flipped), so for any line that had been flipped, `Rd1 ≈ -d2` — the wrong sign on the `t × direction` term. With a 35% inlier rate, the 3-line solver produced the correct scale only 13% of the time vs 47% for the 2-line solver.

*Fix:* Replaced the custom 3-line solver with `model_estimate_sim3` from `ransac.py` (2-line, no sign-flip, uses `d2` directly). Dropped `min_sample` from 3 to 2.

**3. LO scale estimator was translation-biased (significant)**

The local-optimisation loop estimated scale as `median(‖m2‖ / ‖R·m1‖)`. This formula is only unbiased when `t = 0`; with large translation the `t × (R·d)` term adds to the moment and the bias can exceed 200%.

*Fix:* Replaced with `solve_translation_scale(L1_inliers, L2_inliers, R)` — the full joint LS that correctly accounts for the translation term.

**4. Sign flip in `solve_translation_scale` was mathematically wrong (minor)**

When `lstsq` returned `s < 0`, the code did `s = -s; t = -t`. Negating both satisfies `−m2 = s·R·m1 + t×d` — not the actual constraint. In practice the inlier check rejected these hypotheses, so the impact was limited.

*Fix:* Removed the flip. The RANSAC loop's existing `s <= 0 → skip` guard handles it correctly.

**Net effect after all fixes:**

| Metric | Before | After |
|--------|--------|-------|
| Scale estimate (seq02→seq01, GT=2.7947) | 0.02–0.20 (inverted/near-zero) | **2.7955 (0.0% error)** |
| Inliers | 17 | **71** |
| L2 backend for comparison | 2.8431 (1.7% error), 71 inliers | — |
