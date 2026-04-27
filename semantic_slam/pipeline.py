from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .matching_adapter import Sim3PlueckerMatcher
from .sim3_ops import compose_sim3, fuse_relative_edges, identity_sim3, relative_sim3
from .types import FrameState, ObjectObservation, RelativeEdge, Sim3Estimate, TrackState


@dataclass
class SemanticSLAMConfig:
    min_lines_per_object: int = 8
    min_inlier_count: int = 2
    min_inlier_ratio: float = 5.0
    use_instance_id_for_data_association: bool = True


class SemanticObjectCentricSLAM:
    """Incremental object-centric SLAM using the fixed Sim(3) matcher as backend."""

    def __init__(self, matcher: Sim3PlueckerMatcher, config: Optional[SemanticSLAMConfig] = None):
        self.matcher = matcher
        self.config = config or SemanticSLAMConfig()

        self.tracks: Dict[str, TrackState] = {}
        self.frames: Dict[int, FrameState] = {}
        self.edges: List[RelativeEdge] = []

    def process_frame(self, frame_id: int, observations: List[ObjectObservation]) -> FrameState:
        if frame_id in self.frames:
            raise ValueError(f"frame_id={frame_id} already processed")

        if not self.frames:
            self.frames[frame_id] = FrameState(frame_id=frame_id, world_to_frame=identity_sim3())
        else:
            prev_id = max(self.frames.keys())
            self.frames[frame_id] = FrameState(
                frame_id=frame_id,
                world_to_frame=self.frames[prev_id].world_to_frame,
            )

        relative_candidates: List[Sim3Estimate] = []

        for obs in observations:
            obs_lines = np.asarray(obs.plucker_lines, dtype=np.float32)
            if obs_lines.ndim != 2 or obs_lines.shape[1] != 6:
                continue
            if obs_lines.shape[0] < self.config.min_lines_per_object:
                continue

            track = self._associate_track(obs)
            if track is None:
                new_id = self._create_track(frame_id, obs, obs_lines)
                track = self.tracks[new_id]

            est = self.matcher.estimate_sim3(track.canonical_lines, obs_lines)
            if est.inlier_count < self.config.min_inlier_count or est.inlier_ratio < self.config.min_inlier_ratio:
                continue

            previous_obs_frame = track.last_frame_id if track.per_frame_object_to_frame else None
            track.per_frame_object_to_frame[frame_id] = est
            track.last_frame_id = frame_id

            if (
                previous_obs_frame is not None
                and previous_obs_frame != frame_id
                and previous_obs_frame in track.per_frame_object_to_frame
            ):
                T_obj_to_prev = track.per_frame_object_to_frame[previous_obs_frame]
                T_obj_to_curr = track.per_frame_object_to_frame[frame_id]
                T_prev_to_curr = relative_sim3(T_obj_to_prev, T_obj_to_curr)
                relative_candidates.append(T_prev_to_curr)
                self.edges.append(
                    RelativeEdge(
                        frame_i=previous_obs_frame,
                        frame_j=frame_id,
                        object_track_id=track.track_id,
                        transform_i_to_j=T_prev_to_curr,
                    )
                )

        if relative_candidates:
            prev_frame_id = max(fid for fid in self.frames.keys() if fid != frame_id)
            fused_prev_to_curr = fuse_relative_edges(relative_candidates)
            world_to_prev = self.frames[prev_frame_id].world_to_frame
            self.frames[frame_id].world_to_frame = compose_sim3(world_to_prev, fused_prev_to_curr)

        return self.frames[frame_id]

    def export_state(self) -> dict:
        return {
            "frames": {
                str(fid): self._sim3_to_json(frame.world_to_frame)
                for fid, frame in sorted(self.frames.items(), key=lambda kv: kv[0])
            },
            "tracks": {
                tid: {
                    "semantic_label": tr.semantic_label,
                    "first_frame_id": tr.first_frame_id,
                    "last_frame_id": tr.last_frame_id,
                    "num_observations": len(tr.per_frame_object_to_frame),
                }
                for tid, tr in self.tracks.items()
            },
            "edges": [
                {
                    "frame_i": e.frame_i,
                    "frame_j": e.frame_j,
                    "object_track_id": e.object_track_id,
                    "transform_i_to_j": self._sim3_to_json(e.transform_i_to_j),
                }
                for e in self.edges
            ],
        }

    def _associate_track(self, obs: ObjectObservation) -> Optional[TrackState]:
        if self.config.use_instance_id_for_data_association and obs.instance_id is not None:
            track_id = f"{obs.semantic_label}:{obs.instance_id}"
            return self.tracks.get(track_id)

        candidates = [t for t in self.tracks.values() if t.semantic_label == obs.semantic_label]
        if not candidates:
            return None

        # If instance_id is absent, nearest-neighbor association by matcher score.
        best_track = None
        best_score = -np.inf
        obs_lines = np.asarray(obs.plucker_lines, dtype=np.float32)
        for track in candidates:
            est = self.matcher.estimate_sim3(track.canonical_lines, obs_lines)
            score = est.inlier_ratio
            if est.inlier_count >= self.config.min_inlier_count and score > best_score:
                best_score = score
                best_track = track
        return best_track

    def _create_track(self, frame_id: int, obs: ObjectObservation, obs_lines: np.ndarray) -> str:
        suffix = obs.instance_id if obs.instance_id is not None else str(len(self.tracks))
        track_id = f"{obs.semantic_label}:{suffix}"
        while track_id in self.tracks:
            track_id = f"{track_id}_dup"

        track = TrackState(
            track_id=track_id,
            semantic_label=obs.semantic_label,
            canonical_lines=obs_lines.copy(),
            first_frame_id=frame_id,
            last_frame_id=frame_id,
        )
        track.per_frame_object_to_frame[frame_id] = identity_sim3()
        self.tracks[track_id] = track
        return track_id

    @staticmethod
    def _sim3_to_json(est: Sim3Estimate) -> dict:
        return {
            "s": float(est.s),
            "R": np.asarray(est.R).reshape(3, 3).tolist(),
            "t": np.asarray(est.t).reshape(3).tolist(),
            "inlier_count": int(est.inlier_count),
            "inlier_ratio": float(est.inlier_ratio),
            "score": float(est.score),
        }
