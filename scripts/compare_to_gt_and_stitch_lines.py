#!/usr/bin/env python3
"""Compare SLAM output to Replica ground truth and visualize stitched line cloud.

This script expects:
- SLAM output from run_semantic_object_slam.py
- object_sequence.json with per-frame Plucker lines [m0,m1,m2,d0,d1,d2]
- Replica traj.txt (one 4x4 matrix per line, flattened row-major)

Outputs:
- Console trajectory error summary after Sim(3) alignment (ATE-like metrics)
- Figure with:
  1) Trajectory overlay (estimated aligned to GT)
  2) Stitched line cloud overlay (estimated vs GT world coordinates)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slam_json", default="./results/replica_object_slam_state.json")
    parser.add_argument("--sequence_json", default="/home/rueyday/data/Replica/object_sequence.json")
    parser.add_argument("--traj_txt", default="/home/rueyday/data/Replica/room2/traj.txt")
    parser.add_argument("--max_frames", type=int, default=120)
    parser.add_argument("--max_lines_per_frame", type=int, default=12)
    parser.add_argument("--line_half_length", type=float, default=0.08)
    parser.add_argument("--save_png", default="./results/replica_gt_compare_stitched.png")
    parser.add_argument("--no_show", action="store_true")
    return parser.parse_args()


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sorted_est_frames(slam_state: dict) -> List[Tuple[int, dict]]:
    frames = []
    for k, v in slam_state.get("frames", {}).items():
        try:
            frames.append((int(k), v))
        except ValueError:
            continue
    frames.sort(key=lambda x: x[0])
    return frames


def frame_to_world_from_est(frame_payload: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (A, b, center) for x_w = A @ x_f + b.

    SLAM stores world_to_frame as x_f = s R x_w + t.
    Inverse gives x_w = (1/s) R^T (x_f - t).
    """
    s = float(frame_payload.get("s", 1.0))
    R = np.asarray(frame_payload.get("R", np.eye(3)), dtype=np.float64).reshape(3, 3)
    t = np.asarray(frame_payload.get("t", [0, 0, 0]), dtype=np.float64).reshape(3, 1)
    s = max(s, 1e-9)

    A = (1.0 / s) * R.T
    b = (-(1.0 / s) * (R.T @ t)).reshape(3)
    center = b.copy()  # frame origin transformed to world
    return A, b, center


def parse_traj(path: str) -> List[np.ndarray]:
    mats: List[np.ndarray] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            vals = line.strip().split()
            if not vals:
                continue
            arr = np.asarray([float(v) for v in vals], dtype=np.float64)
            if arr.size != 16:
                continue
            mats.append(arr.reshape(4, 4))
    return mats


def gt_centers_for_mode(gt_mats: List[np.ndarray], mode: str) -> np.ndarray:
    """Compute world centers for a chosen trajectory convention.

    mode='w2c': gt matrix is world_to_cam. Camera center Cw = -R^T t
    mode='c2w': gt matrix is cam_to_world. Camera center Cw = t
    """
    centers = []
    for M in gt_mats:
        R = M[:3, :3]
        t = M[:3, 3]
        if mode == "w2c":
            C = -(R.T @ t)
        elif mode == "c2w":
            C = t
        else:
            raise ValueError(mode)
        centers.append(C)
    return np.asarray(centers, dtype=np.float64)


def umeyama_align(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Find Sim(3) mapping src -> dst via Umeyama."""
    assert src.shape == dst.shape
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    X = src - mu_src
    Y = dst - mu_dst

    cov = (Y.T @ X) / n
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt
    var_src = np.sum(X * X) / n
    c = float(np.trace(np.diag(D) @ S) / max(var_src, 1e-12))
    t = mu_dst - c * (R @ mu_src)
    return c, R, t


def apply_sim3_points(pts: np.ndarray, c: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (c * (R @ pts.T)).T + t.reshape(1, 3)


def line_point_and_dir(line_md: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """From Plucker [m,d], recover one point on line and unit direction."""
    m = line_md[:3]
    d = line_md[3:]
    dn = np.linalg.norm(d)
    if dn < 1e-12:
        d = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        dn = 1.0
    d = d / dn
    p = np.cross(d, m) / max(np.dot(d, d), 1e-12)
    return p, d


def extract_frame_object_map(sequence: list) -> Dict[int, List[np.ndarray]]:
    fmap: Dict[int, List[np.ndarray]] = {}
    for fr in sequence:
        fid = int(fr.get("frame_id", -1))
        if fid < 0:
            continue
        lines_for_frame = []
        for obj in fr.get("objects", []):
            lines = np.asarray(obj.get("plucker_lines", []), dtype=np.float64)
            if lines.ndim == 2 and lines.shape[1] == 6:
                lines_for_frame.append(lines)
        if lines_for_frame:
            fmap[fid] = lines_for_frame
    return fmap


def gt_frame_to_world(gt_mats: List[np.ndarray], frame_id: int, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    M = gt_mats[frame_id]
    if mode == "c2w":
        A = M[:3, :3]
        b = M[:3, 3]
    else:
        R = M[:3, :3]
        t = M[:3, 3]
        A = R.T
        b = -(R.T @ t)
    return A, b


def build_stitched_clouds(
    est_frames: List[Tuple[int, dict]],
    frame_objects: Dict[int, List[np.ndarray]],
    gt_mats: List[np.ndarray],
    gt_mode: str,
    max_frames: int,
    max_lines_per_frame: int,
    half_len: float,
) -> Tuple[np.ndarray, np.ndarray]:
    est_pts = []
    gt_pts = []

    count = 0
    for fid, payload in est_frames:
        if fid not in frame_objects:
            continue
        if fid >= len(gt_mats):
            continue

        A_est, b_est, _ = frame_to_world_from_est(payload)
        A_gt, b_gt = gt_frame_to_world(gt_mats, fid, gt_mode)

        lines_cat = np.concatenate(frame_objects[fid], axis=0)
        if lines_cat.shape[0] > max_lines_per_frame:
            idx = np.linspace(0, lines_cat.shape[0] - 1, max_lines_per_frame).astype(int)
            lines_cat = lines_cat[idx]

        for line in lines_cat:
            p, d = line_point_and_dir(line)
            p1 = p - half_len * d
            p2 = p + half_len * d

            est_pts.append(A_est @ p1 + b_est)
            est_pts.append(A_est @ p2 + b_est)
            gt_pts.append(A_gt @ p1 + b_gt)
            gt_pts.append(A_gt @ p2 + b_gt)

        count += 1
        if count >= max_frames:
            break

    if not est_pts:
        return np.zeros((0, 3)), np.zeros((0, 3))

    return np.asarray(est_pts), np.asarray(gt_pts)


def evaluate_and_plot(
    slam_state: dict,
    sequence: list,
    gt_mats: List[np.ndarray],
    max_frames: int,
    max_lines_per_frame: int,
    line_half_length: float,
    save_png: str,
    no_show: bool,
):
    est_frames = sorted_est_frames(slam_state)
    est_frames = [(fid, p) for fid, p in est_frames if fid < len(gt_mats)]
    if len(est_frames) < 3:
        raise ValueError("Not enough overlapping frames between SLAM and GT")

    # Build estimated centers in world coordinates.
    est_centers = []
    frame_ids = []
    for fid, payload in est_frames:
        _, _, c = frame_to_world_from_est(payload)
        est_centers.append(c)
        frame_ids.append(fid)
    est_centers = np.asarray(est_centers, dtype=np.float64)

    gt_c2w = gt_centers_for_mode(gt_mats, "c2w")[frame_ids]
    gt_w2c = gt_centers_for_mode(gt_mats, "w2c")[frame_ids]

    # Choose GT convention that best aligns to estimated trajectory.
    c1, R1, t1 = umeyama_align(est_centers, gt_c2w)
    a1 = apply_sim3_points(est_centers, c1, R1, t1)
    rmse1 = float(np.sqrt(np.mean(np.sum((a1 - gt_c2w) ** 2, axis=1))))

    c2, R2, t2 = umeyama_align(est_centers, gt_w2c)
    a2 = apply_sim3_points(est_centers, c2, R2, t2)
    rmse2 = float(np.sqrt(np.mean(np.sum((a2 - gt_w2c) ** 2, axis=1))))

    if rmse1 <= rmse2:
        gt_mode = "c2w"
        gt_centers = gt_c2w
        c, R, t = c1, R1, t1
        est_aligned = a1
        rmse = rmse1
    else:
        gt_mode = "w2c"
        gt_centers = gt_w2c
        c, R, t = c2, R2, t2
        est_aligned = a2
        rmse = rmse2

    per_frame_err = np.linalg.norm(est_aligned - gt_centers, axis=1)
    med = float(np.median(per_frame_err))
    p90 = float(np.percentile(per_frame_err, 90))

    frame_objects = extract_frame_object_map(sequence)
    est_cloud, gt_cloud = build_stitched_clouds(
        est_frames,
        frame_objects,
        gt_mats,
        gt_mode,
        max_frames=max_frames,
        max_lines_per_frame=max_lines_per_frame,
        half_len=line_half_length,
    )

    if est_cloud.shape[0] > 0:
        est_cloud_aligned = apply_sim3_points(est_cloud, c, R, t)
    else:
        est_cloud_aligned = est_cloud

    print("==== Trajectory Comparison ====")
    print(f"Overlapping frames: {len(frame_ids)}")
    print(f"GT convention selected: {gt_mode}")
    print(f"Alignment scale: {c:.6f}")
    print(f"RMSE (m): {rmse:.6f}")
    print(f"Median (m): {med:.6f}")
    print(f"P90 (m): {p90:.6f}")

    fig = plt.figure(figsize=(16, 7))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot(gt_centers[:, 0], gt_centers[:, 1], gt_centers[:, 2], color="black", linewidth=2.0, label="GT")
    ax1.plot(
        est_aligned[:, 0],
        est_aligned[:, 1],
        est_aligned[:, 2],
        color="#d62728",
        linewidth=1.5,
        alpha=0.9,
        label="Estimated (aligned)",
    )
    ax1.set_title("Trajectory: GT vs Estimated")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.legend(loc="best")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    if gt_cloud.shape[0] > 0:
        n = min(12000, gt_cloud.shape[0])
        idx = np.linspace(0, gt_cloud.shape[0] - 1, n).astype(int)
        ax2.scatter(gt_cloud[idx, 0], gt_cloud[idx, 1], gt_cloud[idx, 2], s=1.0, alpha=0.35, c="black", label="GT stitched")
    if est_cloud_aligned.shape[0] > 0:
        n = min(12000, est_cloud_aligned.shape[0])
        idx = np.linspace(0, est_cloud_aligned.shape[0] - 1, n).astype(int)
        ax2.scatter(
            est_cloud_aligned[idx, 0],
            est_cloud_aligned[idx, 1],
            est_cloud_aligned[idx, 2],
            s=1.0,
            alpha=0.35,
            c="#1f77b4",
            label="Estimated stitched (aligned)",
        )
    ax2.set_title("Stitched Line Cloud Overlay")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.legend(loc="best")

    fig.suptitle(f"GT Comparison | RMSE={rmse:.4f} m | Median={med:.4f} m | P90={p90:.4f} m", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_png:
        out_dir = os.path.dirname(os.path.abspath(save_png))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(save_png, dpi=180)
        print(f"Saved figure: {save_png}")

    if not no_show:
        plt.show()


def main():
    args = parse_args()
    for p in [args.slam_json, args.sequence_json, args.traj_txt]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    slam_state = load_json(args.slam_json)
    sequence = load_json(args.sequence_json)
    gt_mats = parse_traj(args.traj_txt)

    evaluate_and_plot(
        slam_state=slam_state,
        sequence=sequence,
        gt_mats=gt_mats,
        max_frames=args.max_frames,
        max_lines_per_frame=args.max_lines_per_frame,
        line_half_length=args.line_half_length,
        save_png=args.save_png,
        no_show=args.no_show,
    )


if __name__ == "__main__":
    main()
