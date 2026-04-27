import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from sim3.ransac import run_ransac_sim3
from .types import Sim3Estimate


@dataclass
class MatcherConfig:
    topk: int = 200
    ransac_max_iterations: int = 200
    ransac_inlier_threshold: float = 0.1


class Sim3PlueckerMatcher:
    """Inference-only adapter around the existing PlueckerNet model + Sim(3) RANSAC."""

    def __init__(
        self,
        weights_path: str,
        plueckernet_dir: Optional[str] = None,
        device: Optional[str] = None,
        config: Optional[MatcherConfig] = None,
    ):
        self.weights_path = weights_path
        self.config = config or MatcherConfig()

        if plueckernet_dir is None:
            plueckernet_dir = os.path.join(os.path.dirname(__file__), "..", "..", "PlueckerNet")
        self.plueckernet_dir = os.path.abspath(plueckernet_dir)
        if self.plueckernet_dir not in sys.path:
            sys.path.insert(0, self.plueckernet_dir)

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self._load_model()

    def _load_model(self):
        try:
            from easydict import EasyDict as edict
            from model.model_plucker import PluckerNetKnn
        except ImportError as exc:
            raise ImportError(
                "Could not import PlueckerNet. Make sure ../PlueckerNet exists or pass --plueckernet-dir."
            ) from exc

        cfg = edict(
            {
                "net_nchannel": 128,
                "GNN_layers": ["self", "cross"] * 6,
                "net_lambda": 0.1,
                "net_maxiter": 30,
                "net_topK": 200,
            }
        )
        model = PluckerNetKnn(cfg).to(self.device)
        # PyTorch 2.6 defaults torch.load(..., weights_only=True), which can
        # reject checkpoints containing trusted metadata objects (e.g. EasyDict).
        # We explicitly disable weights-only mode for local trusted checkpoints.
        try:
            checkpoint = torch.load(
                self.weights_path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            # Backward compatibility with older torch versions.
            checkpoint = torch.load(self.weights_path, map_location=self.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model

    @torch.no_grad()
    def estimate_sim3(self, lines_src: np.ndarray, lines_tgt: np.ndarray) -> Sim3Estimate:
        if lines_src.shape[0] < 2 or lines_tgt.shape[0] < 2:
            return Sim3Estimate(s=1.0, R=np.eye(3, dtype=np.float32), t=np.zeros((3, 1), dtype=np.float32))

        src = torch.from_numpy(lines_src.astype(np.float32)).unsqueeze(0).to(self.device)
        tgt = torch.from_numpy(lines_tgt.astype(np.float32)).unsqueeze(0).to(self.device)

        prob, _, _ = self.model(src, tgt)

        k = min(self.config.topk, prob.shape[-2] * prob.shape[-1])
        scores, flat_idx = torch.topk(prob.flatten(start_dim=-2), k=k, dim=-1, largest=True, sorted=True)

        idx_src = (flat_idx // prob.size(-1)).squeeze(0).cpu().numpy()
        idx_tgt = (flat_idx % prob.size(-1)).squeeze(0).cpu().numpy()
        top_scores = scores.squeeze(0).cpu().numpy()

        plucker_src_topk = lines_src[idx_src, :].T
        plucker_tgt_topk = lines_tgt[idx_tgt, :].T

        s, R, t, inlier_count, _ = run_ransac_sim3(
            plucker_src_topk,
            plucker_tgt_topk,
            max_iterations=self.config.ransac_max_iterations,
            inlier_threshold=self.config.ransac_inlier_threshold,
        )

        if s is None or R is None or t is None:
            return Sim3Estimate(s=1.0, R=np.eye(3, dtype=np.float32), t=np.zeros((3, 1), dtype=np.float32))

        return Sim3Estimate(
            s=float(s),
            R=R.astype(np.float32),
            t=t.astype(np.float32),
            inlier_count=int(inlier_count),
            inlier_ratio=float(100.0 * inlier_count / max(1, k)),
            score=float(np.mean(top_scores)),
        )
