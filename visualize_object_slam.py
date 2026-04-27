#!/usr/bin/env python3
"""Desktop visualizer for semantic object-centric SLAM output.

Reads the JSON produced by run_semantic_object_slam.py and plots:
1) 3D frame trajectory (from world_to_frame translations)
2) Per-frame scale and edge support
3) Track observation coverage

Usage:
  python visualize_object_slam.py \
    --input_json ./results/replica_object_slam_state.json \
    --save_png ./results/replica_object_slam_viz.png
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _sorted_frame_items(frames_dict: Dict[str, dict]) -> List[Tuple[int, dict]]:
    items = []
    for k, v in frames_dict.items():
        try:
            fid = int(k)
        except ValueError:
            continue
        items.append((fid, v))
    items.sort(key=lambda x: x[0])
    return items


def _build_edge_degree_map(edges: List[dict]) -> Dict[int, int]:
    deg: Dict[int, int] = {}
    for e in edges:
        i = int(e.get("frame_i", -1))
        j = int(e.get("frame_j", -1))
        if i >= 0:
            deg[i] = deg.get(i, 0) + 1
        if j >= 0:
            deg[j] = deg.get(j, 0) + 1
    return deg


def _extract_series(state: dict):
    frame_items = _sorted_frame_items(state.get("frames", {}))
    if not frame_items:
        raise ValueError("No frames found in input JSON")

    frame_ids = []
    xyz = []
    scales = []

    for fid, frame_data in frame_items:
        t = np.asarray(frame_data.get("t", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
        if t.shape[0] != 3:
            t = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        s = float(frame_data.get("s", 1.0))

        frame_ids.append(fid)
        xyz.append(t)
        scales.append(s)

    xyz = np.asarray(xyz, dtype=np.float32)
    scales = np.asarray(scales, dtype=np.float32)

    edge_degree = _build_edge_degree_map(state.get("edges", []))
    edge_support = np.asarray([edge_degree.get(fid, 0) for fid in frame_ids], dtype=np.int32)

    return frame_ids, xyz, scales, edge_support


def plot_state(state: dict, title_prefix: str = "Semantic Object-Centric SLAM"):
    frame_ids, xyz, scales, edge_support = _extract_series(state)
    tracks = state.get("tracks", {})

    fig = plt.figure(figsize=(16, 10))

    # 1) 3D trajectory
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    colors = np.linspace(0.0, 1.0, len(frame_ids))
    sc = ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, cmap="viridis", s=18)
    ax1.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], linewidth=1.2, alpha=0.7)
    ax1.set_title("Frame Trajectory (3D)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    cbar = fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("time index")

    # 2) Scale over time
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(frame_ids, scales, color="#d62728", linewidth=1.6)
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax2.set_title("Estimated Scale per Frame")
    ax2.set_xlabel("frame id")
    ax2.set_ylabel("scale s")
    ax2.grid(alpha=0.25)

    # 3) Edge support over time
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(frame_ids, edge_support, color="#1f77b4", linewidth=1.6)
    ax3.set_title("Frame Graph Edge Support")
    ax3.set_xlabel("frame id")
    ax3.set_ylabel("incident edges")
    ax3.grid(alpha=0.25)

    # 4) Track observation coverage
    ax4 = fig.add_subplot(2, 2, 4)
    track_labels = []
    obs_counts = []
    for tid, td in sorted(tracks.items(), key=lambda kv: kv[0]):
        track_labels.append(tid)
        obs_counts.append(int(td.get("num_observations", 0)))

    if obs_counts:
        x = np.arange(len(obs_counts))
        ax4.bar(x, obs_counts, color="#2ca02c", alpha=0.85)
        ax4.set_xticks(x)
        ax4.set_xticklabels(track_labels, rotation=30, ha="right", fontsize=8)
    else:
        ax4.text(0.5, 0.5, "No tracks", ha="center", va="center", transform=ax4.transAxes)

    ax4.set_title("Track Observation Coverage")
    ax4.set_xlabel("track id")
    ax4.set_ylabel("num observations")
    ax4.grid(alpha=0.2, axis="y")

    n_edges = len(state.get("edges", []))
    fig.suptitle(
        f"{title_prefix} | frames={len(frame_ids)} tracks={len(tracks)} edges={n_edges}",
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        default="./results/replica_object_slam_state.json",
        help="Path to SLAM output JSON",
    )
    parser.add_argument(
        "--save_png",
        default=None,
        help="Optional path to save a PNG of the visualizer output",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Do not open desktop window (useful on headless servers)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_json):
        raise FileNotFoundError(args.input_json)

    with open(args.input_json, "r", encoding="utf-8") as f:
        state = json.load(f)

    fig = plot_state(state)

    if args.save_png:
        out_dir = os.path.dirname(os.path.abspath(args.save_png))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(args.save_png, dpi=180)
        print(f"Saved figure: {args.save_png}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
