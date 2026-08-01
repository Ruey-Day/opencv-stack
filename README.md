# ScalePluckerNet — SCALAR

**Sim(3) registration of 3D line maps across sensor modalities.** ScalePluckerNet
is a scale-invariant Plücker line matcher (unmodified
[PlueckerNet](https://github.com/Liumouliu/PlueckerNet) architecture, trained on
fully synthetic SLAM-calibrated Sim(3) data); paired with the line-based Sim(3)
estimator in `lib/sim3_solver.py` it registers a scale-ambiguous monocular SLAM
map onto a metric RGB-D or LiDAR line map — recovering scale, rotation, and
translation in ~0.3 s per map pair.

**Explanation, theory, and full results live on the
[project page](https://rueyday.github.io/scale-aware-cross-modal-registration/).**
A runnable [Colab demo notebook](https://github.com/rueyday/scale-aware-cross-modal-registration/blob/main/docs/scalar_demo.ipynb)
registers a real bundled 7-Scenes map pair end-to-end on CPU. Both live in the
parent [SCALAR repo](https://github.com/rueyday/scale-aware-cross-modal-registration);
this README covers only the codebase.

---

## Setup

```bash
conda activate torch5090        # Python 3.11, PyTorch 2.6, CUDA
pip install easydict msgpack    # + numpy, torch, matplotlib
```

The repo is self-contained — network and solver both live in `lib/`
(`lib/model.py`, `lib/sim3_solver.py`); no external service or dataset is
required to register two maps.

## Layout

```
lib/sim3_solver.py     THE Sim(3) estimator — one self-contained file
                       (also contains the max-inlier RANSAC baseline)
lib/model.py           PluckerNetKnn network (unchanged from PlueckerNet)
lib/{trainer,loss,dataloader,utils}.py   training machinery
generate_synthetic.py  calibrated Sim(3) pair generator (13 scenarios)
train.py               training entry point
register.py            register two SLAM .db line maps (the demo entry point)
output/                checkpoints (best: output/synthetic_v6/.../snap_ep8.pth)
```

## Quickstart — register two maps

```bash
python register.py \
    --db_src  mono_map.db \
    --db_tgt  metric_map.db \
    --checkpoint output/synthetic_v6/synthetic_v6/best_val_checkpoint.pth
# add --ransac to run the classical max-inlier baseline instead
```

Library use:

```python
from lib.sim3_solver import Sim3Solver
solver = Sim3Solver((q1, q2), (r1, r2))     # (N,3) segment endpoints each
s, R, t, info = solver.register(prob=prob)  # prob = matcher output (optional)
```

`register()`'s bare defaults are the shipped method; every flag is an ablation
(see the docstring). `prob=None` runs the correspondence-free variant.

## Training

```bash
# 1) generate data (200k train / 2k valid; ~2 h, CPU)
python generate_synthetic.py --name synthetic_v6 --workers 12

# 2) train (fine-tuning from a previous checkpoint via --pretrain)
python train.py --dataset synthetic_v6 --epochs 120 --batch 1 --iter_size 32 \
    --lr 2e-4 --gamma 0.99 --workers 4 --name synthetic_v6 \
    --pretrain output/synthetic_v5/synthetic_v5/best_val_checkpoint.pth

# resume an interrupted run (restores optimizer + schedule):
python train.py ... --resume output/synthetic_v6/synthetic_v6/checkpoint.pth
```

Checkpoint selection: the trainer's synthetic validation metric (P@100) is NOT
predictive of real-map registration — select checkpoints on the real benchmark
(below), never on trainer val.

## Evaluation (real benchmarks, parent repo)

The benchmark data and runners live in the parent repository:

```bash
# 37-pair 7-Scenes benchmark (variants: unified | corrfree | arb | m_* ablations)
python tools/bench_sim3_solver.py <ckpt> unified

# matcher quality vs GT correspondences (P@200 + registration success)
python tools/eval_matcher_gt.py <ckpt> mytag

# KITTI LiDAR -> camera
python tools/eval_kitti_sim3solver.py <ckpt> --seqs 06,07 \
    --query camlines --ref lidarv3 --matcher v4
```

Numbers do not reproduce bit-exactly across sessions (GPU-numerics-sensitive
matcher forward); certify A/B comparisons within one session.

## Plücker conventions

All code here uses `[m, d]` order (moment first); the original PlueckerNet
uses `[d, m]`. Sim(3) law: `d' = R d`, `m' = s·R·m + t × d'`. Lines are
sign-ambiguous (`[m,d] ~ [-m,-d]`); solver and generator handle canonical
signs explicitly — three separate signs exist (query canon, pair, ref canon).
