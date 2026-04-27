from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Sim3Estimate:
    """Similarity transform x' = s R x + t and its quality metrics."""

    s: float
    R: np.ndarray
    t: np.ndarray
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    score: float = 0.0


@dataclass
class ObjectObservation:
    """Single object observation for one frame."""

    semantic_label: str
    plucker_lines: np.ndarray
    instance_id: Optional[str] = None


@dataclass
class TrackState:
    """Persistent map object anchored by first observation lines."""

    track_id: str
    semantic_label: str
    canonical_lines: np.ndarray
    first_frame_id: int
    last_frame_id: int
    per_frame_object_to_frame: Dict[int, Sim3Estimate] = field(default_factory=dict)


@dataclass
class FrameState:
    """Global frame pose in the SLAM map frame."""

    frame_id: int
    world_to_frame: Sim3Estimate


@dataclass
class RelativeEdge:
    """Graph edge between two frames built from shared object tracks."""

    frame_i: int
    frame_j: int
    object_track_id: str
    transform_i_to_j: Sim3Estimate
