from typing import Iterable, List

import numpy as np

from .types import Sim3Estimate


def identity_sim3() -> Sim3Estimate:
    return Sim3Estimate(
        s=1.0,
        R=np.eye(3, dtype=np.float32),
        t=np.zeros((3, 1), dtype=np.float32),
        inlier_count=0,
        inlier_ratio=0.0,
        score=0.0,
    )


def invert_sim3(T: Sim3Estimate) -> Sim3Estimate:
    R_inv = T.R.T.astype(np.float32)
    s_inv = float(1.0 / T.s)
    t_inv = (-s_inv * (R_inv @ T.t)).astype(np.float32)
    return Sim3Estimate(s=s_inv, R=R_inv, t=t_inv)


def compose_sim3(T_a: Sim3Estimate, T_b: Sim3Estimate) -> Sim3Estimate:
    """Return T = T_b o T_a (apply T_a, then T_b)."""
    s = float(T_b.s * T_a.s)
    R = (T_b.R @ T_a.R).astype(np.float32)
    t = (T_b.s * (T_b.R @ T_a.t) + T_b.t).astype(np.float32)
    return Sim3Estimate(
        s=s,
        R=R,
        t=t,
        inlier_count=min(T_a.inlier_count, T_b.inlier_count),
        inlier_ratio=min(T_a.inlier_ratio, T_b.inlier_ratio),
        score=min(T_a.score, T_b.score),
    )


def relative_sim3(T_i: Sim3Estimate, T_j: Sim3Estimate) -> Sim3Estimate:
    """Return T_i_to_j given object-to-frame transforms T_i and T_j."""
    return compose_sim3(invert_sim3(T_i), T_j)


def weighted_average_rotations(rotations: Iterable[np.ndarray], weights: Iterable[float]) -> np.ndarray:
    M = np.zeros((3, 3), dtype=np.float64)
    for R, w in zip(rotations, weights):
        M += float(w) * R
    U, _, Vt = np.linalg.svd(M)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    return R_avg.astype(np.float32)


def fuse_relative_edges(edges: List[Sim3Estimate]) -> Sim3Estimate:
    if not edges:
        return identity_sim3()

    weights = np.array([max(1e-6, e.inlier_ratio) for e in edges], dtype=np.float64)
    weights /= weights.sum()

    log_scales = np.array([np.log(max(1e-6, e.s)) for e in edges], dtype=np.float64)
    s = float(np.exp(np.sum(weights * log_scales)))

    rotations = [e.R for e in edges]
    R = weighted_average_rotations(rotations, weights)

    t_stack = np.stack([e.t.reshape(3) for e in edges], axis=0)
    t = np.sum(weights[:, None] * t_stack, axis=0).reshape(3, 1).astype(np.float32)

    return Sim3Estimate(
        s=s,
        R=R,
        t=t,
        inlier_count=int(np.sum([e.inlier_count for e in edges])),
        inlier_ratio=float(np.mean([e.inlier_ratio for e in edges])),
        score=float(np.mean([e.score for e in edges])),
    )
