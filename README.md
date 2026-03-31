# scale-aware-PlueckerNet

A demonstration that **Pluecker-coordinate line registration** (as used in [PlueckerNet](https://github.com/Liumouliu/PlueckerNet)) succeeds for **SE(3)** transformations but **fails for SIM(3)** (SE(3) + unknown scale). The failure mode is structural — not implementation-specific — and is derived analytically and verified experimentally on real RGBD data.

---

## What this repo does

### 1. Build a 3D voxel from chess RGBD frames

30 frames are loaded from the [7-Scenes Chess](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) dataset (`seq-01`). Each depth frame is back-projected into world space using the pinhole camera model and the provided 4×4 camera-to-world pose, then merged into a single voxel-downsampled point cloud (2 cm grid → ~162k points).

> **The point cloud is used only for scene reconstruction and visualization.**
> The registration itself operates entirely on 3D line features.

### 2. Extract 3D line segments

For each seed point, its k=25 nearest neighbours are assembled and their covariance matrix decomposed with PCA. A patch is kept as a line segment if its **linearity** score (λ₁ − λ₂)/λ₁ > 0.70, meaning the neighbourhood is elongated rather than planar or volumetric. This yields ~60 oriented 3D line segments (midpoint + unit direction) directly from the point cloud geometry.

### 3. Represent lines as Pluecker coordinates

Each 3D line is converted to a **6D Pluecker coordinate**:

```
L = (m, d)    where  d = unit direction,  m = p × d  (moment vector)
```

`p` is any point on the line. The full 6D vector `[m₀ m₁ m₂ d₀ d₁ d₂]` encodes both the orientation and position of the line in a coordinate-free way. This is the same representation used by PlueckerNet.

### 4. Apply known transformations

Two transformed line sets are created from the reference:

| Transform | Formula for direction | Formula for moment |
|-----------|----------------------|-------------------|
| **SE(3)** `(R, t)` | `d′ = R d` | `m′ = R m + t × R d` |
| **SIM(3)** `(s, R, t)` | `d′ = R d` | `m′ = s · R m + t × R d` |

Ground-truth: R = 27°, t = [0.35, −0.20, 0.15] m, s = 1.45.

### 5. Register with the SE(3) Pluecker solver

The solver (RANSAC rotation from direction pairs → closed-form translation via least-squares on the Pluecker constraint) is applied to both the SE(3) and SIM(3)-transformed line sets.

---

## Why PlueckerNet fails on SIM(3)

The SE(3) Pluecker constraint is:

```
m₂  =  R m₁  +  t × R d₁
```

Under SIM(3) with scale `s`, the moment becomes:

```
m₂  =  s · R m₁  +  t × R d₁
```

The solver residual is:

```
e  =  m₂  −  R m₁  −  t × R d₁
   =  (s − 1) · R m₁      ≠ 0   for  s ≠ 1
```

This residual **cannot be made zero by any choice of t**. It grows linearly with `|s − 1|` and is proportional to the magnitude of the moment vectors. Crucially:

- **Rotation is still recoverable** — unit direction vectors `d` are scale-invariant, so the RANSAC rotation step is unaffected.
- **Translation is badly biased** — the moment mismatch is absorbed into a wrong `t` estimate.

---

## Results

| | Rotation error | Translation error | Pluecker residual |
|---|---|---|---|
| **SE(3)** | **0.000°** | **0.000 m** | **0.000** ✓ |
| **SIM(3)** | 0.000° | **0.858 m** | **0.630** ✗ |

### Figure 1 — 3D voxel overview

The point cloud stitched from 30 RGBD frames, with SE(3) and SIM(3) variants shown alongside.

![fig1](results/fig1_voxel_overview.png)

### Figure 2 — SE(3) registration (success)

Before / after alignment and error metrics. The aligned cloud perfectly overlaps the source.

![fig2](results/fig2_SE3_registration.png)

### Figure 3 — SIM(3) registration (failure)

Same solver, wrong model. Rotation is correct but translation is wrong — the aligned cloud is visibly shifted.

![fig3](results/fig3_SIM3_registration_failure.png)

### Figure 4 — Analytical explanation

Left: theoretical Pluecker residual as a function of scale `s` (zero only at s=1). Right: side-by-side error summary.

![fig4](results/fig4_why_sim3_fails.png)

### Figure 5 — Pluecker line sets in 3D

The actual primitives used for registration: oriented line segments drawn from their reconstructed midpoints. The point cloud is shown faintly as spatial context only.

![fig5](results/fig5_pluecker_lines_3d.png)

---

## Running the demo

```bash
# activate an environment with numpy, scipy, matplotlib, opencv-python
conda activate depth_anything

# optionally point to a different chess sequence
export CHESS_DATA_DIR=/path/to/chess/seq-01

python chess_plueckernet_demo.py
# figures written to results/
```

**Dependencies:** `numpy`, `scipy`, `matplotlib`, `opencv-python`
**Dataset:** [7-Scenes Chess](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) — camera intrinsics fx=fy=525, cx=319.5, cy=239.5; depth scale 1 mm/unit.

---

## File layout

```
scale-aware-PlueckerNet/
├── chess_plueckernet_demo.py   # full pipeline: RGBD → voxel → lines → Pluecker → register
├── results/                    # pre-generated figures (5 PNGs)
├── load_scene.py               # habitat-sim scene loader (separate experiment)
└── old_practice_code/          # earlier CV learning notes
```

---

## References

- **PlueckerNet** — Liu et al., *"PlueckerNet: Learn to Register 3D Line Reconstructions"*, CVPR 2021. [GitHub](https://github.com/Liumouliu/PlueckerNet)
- **7-Scenes dataset** — Shotton et al., CVPR 2013.
