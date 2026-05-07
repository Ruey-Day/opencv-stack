#!/usr/bin/env python3
"""Run semantic object-centric SLAM using fixed Sim(3) Pluecker matching.

Input JSON format:
[
  {
    "frame_id": 0,
    "objects": [
      {
        "semantic_label": "chair",
        "instance_id": "17",
        "plucker_lines": [[m0,m1,m2,d0,d1,d2], ...]
      }
    ]
  }
]
"""

import argparse
import json
import os
import sys
from typing import List

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from semantic_slam import (
    ObjectObservation,
    SemanticObjectCentricSLAM,
    SemanticSLAMConfig,
    Sim3PlueckerMatcher,
)
from semantic_slam.matching_adapter import MatcherConfig


def _load_frames(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError("Input must be a JSON list of frame dictionaries")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True, help="Sequence with semantic object Plucker lines")
    parser.add_argument("--output_json", required=True, help="Output SLAM state JSON")
    parser.add_argument("--weights", required=True, help="PlueckerNet checkpoint path")
    parser.add_argument("--plueckernet_dir", default=None, help="Optional path to external PlueckerNet repo")
    parser.add_argument("--device", default=None, help="torch device (e.g. cuda, cpu)")
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--ransac_iterations", type=int, default=200)
    parser.add_argument("--ransac_threshold", type=float, default=0.1)
    parser.add_argument("--min_lines", type=int, default=8)
    parser.add_argument("--min_inlier_count", type=int, default=2)
    parser.add_argument("--min_inlier_ratio", type=float, default=5.0)
    args = parser.parse_args()

    if not os.path.exists(args.input_json):
        raise FileNotFoundError(args.input_json)
    if not os.path.exists(args.weights):
        raise FileNotFoundError(args.weights)

    matcher = Sim3PlueckerMatcher(
        weights_path=args.weights,
        plueckernet_dir=args.plueckernet_dir,
        device=args.device,
        config=MatcherConfig(
            topk=args.topk,
            ransac_max_iterations=args.ransac_iterations,
            ransac_inlier_threshold=args.ransac_threshold,
        ),
    )
    slam = SemanticObjectCentricSLAM(
        matcher=matcher,
        config=SemanticSLAMConfig(
            min_lines_per_object=args.min_lines,
            min_inlier_count=args.min_inlier_count,
            min_inlier_ratio=args.min_inlier_ratio,
            use_instance_id_for_data_association=True,
        ),
    )

    frames = _load_frames(args.input_json)
    frames = sorted(frames, key=lambda x: int(x["frame_id"]))

    for frame in frames:
        frame_id = int(frame["frame_id"])
        obs_batch = []
        for obj in frame.get("objects", []):
            lines = np.asarray(obj.get("plucker_lines", []), dtype=np.float32)
            if lines.size == 0:
                continue
            obs_batch.append(
                ObjectObservation(
                    semantic_label=str(obj["semantic_label"]),
                    instance_id=str(obj["instance_id"]) if "instance_id" in obj else None,
                    plucker_lines=lines,
                )
            )
        slam.process_frame(frame_id, obs_batch)

    state = slam.export_state()
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    print(
        f"Processed {len(frames)} frames, built {len(state['tracks'])} tracks, "
        f"and {len(state['edges'])} frame edges."
    )


if __name__ == "__main__":
    main()
