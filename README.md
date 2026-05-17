# ScalePluckerNet

**ScalePluckerNet** extends [PlueckerNet](https://github.com/Liumouliu/PlueckerNet) (Liu et al., CVPR 2021) from **SE(3)** to **Sim(3)** — jointly recovering rotation R, translation t, *and scale s* from Plücker line correspondences.

[Website](https://rueyday.github.io/ScalePluckerNet/) &nbsp;·&nbsp; [Dataset (Dropbox)](https://www.dropbox.com/scl/fo/34o03nsdztz3fpxrwzhty/ALP0MX8KOvdEDx8fg_Wfd9I?rlkey=qzo08vwuqo4jwt5nrrsffb6t3&st=9xaxvt1b&dl=0) &nbsp;·&nbsp; [Model weights (Dropbox)](https://www.dropbox.com/scl/fo/1knswbb20t9pjug00vim7/ALfsafw208mQSmSyzAvnjIU?rlkey=spq78nh6ofobjsk1abuhy86ry&st=06gqdibi&dl=0) &nbsp;·&nbsp; [Google Colab](https://colab.research.google.com/drive/1_AWdfnJjmsteVT_lM4dakYTecn1gpRc_?usp=sharing)

## Repository Layout

```
ScalePlueckerNet/
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
│   ├── convert_se3_datasets.py          # Step 1: convert semantic3D/structured3D
│   ├── generate_se3real_sim3_dataset.py # Step 2a: scale-augment SE3 datasets
│   ├── generate_replica_gs_dataset.py   # Step 2b: Replica RGBD world-space lines
│   ├── generate_7scenes_gs_dataset.py   # Step 2c: 7-Scenes RGBD world-space lines
│   ├── _pair_gen.py                     # Shared pair generation utilities
│   ├── combine_joint_dataset.py         # Step 3: merge all sources into joint split
│   └── eval.py                          # Evaluation entry point
│
├── train.py                   # Training entry point
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

## Dependencies

### Conda environment

```bash
conda env create -f environment.yml # Conda env I used to run with ubuntu 24.04 and RTX 5090
```

### PlueckerNet

`../PlueckerNet/` must exist alongside this repo.

### GlueStick (dataset generation only)

```
/home/rueyday/scale-aware-cross-modal-registration/GlueStick
```

Run on **CPU only** (`SPWireframeDescriptor.to('cpu')`). Only `['lines']` output is used.

## Dataset Pipeline

```bash
python scripts/convert_se3_datasets.py
# reads  ../PlueckerNet/dataset/{semantic3D,structured3D}_{train,valid}/
# writes ./dataset/{semantic3D,structured3D}_{train,valid}/

python scripts/generate_se3real_sim3_dataset.py
# output: dataset/se3real_sim3_{train,valid}/

python scripts/generate_replica_gs_dataset.py &
python scripts/generate_7scenes_gs_dataset.py &

python scripts/combine_joint_dataset.py
# output: dataset/joint_{train,valid}/
```

### Dataset format

Each split is a directory of 6 pickle files:

| File | Shape per sample | dtype |
|------|-----------------|-------|
| `matches.pkl` | `(2, n_inliers)` — row 0 = src indices, row 1 = tgt indices | int32 |
| `plucker1.pkl` | `(N_TOTAL, 6)` | float32 |
| `plucker2.pkl` | `(N_TOTAL, 6)` | float32 |
| `R_gt.pkl` | `(3, 3)` | float32 |
| `t_gt.pkl` | `(3, 1)` | float32 |
| `s_gt.pkl` | scalar (`0.0` = zero-overlap, no valid pose) | float32 |

## Training

```bash
# Train on all datasets (default):
python train.py

# Single source:
python train.py --dataset se3real_sim3

# Resume:
python train.py --resume output/joint/<date>/checkpoint.pth

# With dustbin from a joint checkpoint:
python train.py --dustbin \
    --pretrain output/joint/<date>/best_val_checkpoint.pth --lr 2e-4

# Multiple simultaneous runs (use --name to avoid checkpoint clashes):
python train.py --dataset joint --name run_a
```

### Training Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `joint` | `semantic3D \| structured3D \| replica_gs \| 7scenes_gs \| se3real_sim3 \| joint` |
| `--data_dir` | `./dataset` | Dataset root |
| `--epochs` | 400 | |
| `--batch` | 32 | |
| `--lr` | 5e-4 | |
| `--gpu` | 0 | |
| `--workers` | 8 | Reduce to 4 when running multiple jobs |
| `--in_channel` | 6 | `6` = geometry only, `9` = Plücker + LAB color |
| `--dustbin` | off | Learnable dustbin token (SuperGlue-style) |
| `--cosine_lr` | off | CosineAnnealingWarmRestarts instead of ExponentialLR |
| `--pretrain` | — | Warm-start from checkpoint (`strict=False`) |
| `--resume` | — | Resume from checkpoint |
| `--name` | today's date | Prevents checkpoint clashes when running multiple jobs |

Checkpoints: `output/<dataset>/<name>/`

## Evaluation

Evaluates on one or more validation splits; reports per-scene metrics broken down by overlap level.

```bash
python scripts/eval.py \
    --weights output/joint/2026-05-14/best_val_checkpoint.pth \
    --dataset se3real_sim3,replica_gs,7scenes_gs \
    --ransac grassmannian
```

Results saved to `results/eval_cross_dataset/<label>.json`.

**Overlap buckets:** `no_overlap (0%)` = 0 inliers; `sparse (~30%)` = 1–200 inliers; `dense (~70%)` = 201–490 inliers.

**RANSAC backends:**

| Backend | Flag | Notes |
|---------|------|-------|
| L2 Sim(3) | `--ransac sim3` (default) | Fast; threshold is scale-dependent |
| Grassmannian | `--ransac grassmannian` | Scale/translation-invariant; stratified sampling + Tikhonov regularization |

### Eval Flags

```
--weights              Checkpoint path (required)
--dataset              Comma-separated val splits               [default: se3real_sim3,replica_gs,7scenes_gs]
--data_dir             Dataset root                             [default: ./dataset]
--ransac               sim3 | grassmannian                      [default: sim3]
--out_dir              Output directory for JSON                [default: results/eval_cross_dataset]
--label                Human-readable run label
```
