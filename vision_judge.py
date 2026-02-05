"""
VLM as calibrator (which ROI is the clock?) and appeals court (classify ambiguous segments).
CPU baseline: stub (no optional deps). Optional backends: llama.cpp multimodal, etc.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

# RoiBox = (x, y, w, h)
RoiBox = Tuple[int, int, int, int]


class IVisionJudge:
    """Optional VLM: calibrator + appeals court. Stub = no-op when disabled."""

    def which_roi_is_clock(
        self,
        frames_bgr: List[np.ndarray],
        candidate_rois: List[RoiBox],
    ) -> Optional[Tuple[int, float]]:
        """
        Which candidate (0-based index) contains the game clock? Return (index, confidence) or None.
        """
        raise NotImplementedError

    def classify_segment(
        self,
        frames_bgr: List[np.ndarray],
        context: str,
    ) -> str:
        """
        Classify clip: "live" | "replay" | "commercial" | "unknown".
        Used for appeals court on ambiguous segments only.
        """
        raise NotImplementedError


class StubVisionJudge(IVisionJudge):
    """No VLM; all queries return None / unknown. Zero optional deps."""

    def which_roi_is_clock(
        self,
        frames_bgr: List[np.ndarray],
        candidate_rois: List[RoiBox],
    ) -> Optional[Tuple[int, float]]:
        return None

    def classify_segment(
        self,
        frames_bgr: List[np.ndarray],
        context: str,
    ) -> str:
        return "unknown"


def get_vision_judge(config: Any) -> IVisionJudge:
    """
    Factory: return stub if vlm disabled or backend unavailable; else return backend implementation.
    config: object with .vlm (VlmConfig or None) or dict-like .get("vlm", {}).
    """
    vlm = getattr(config, "vlm", None)
    if vlm is None:
        if hasattr(config, "get") and callable(getattr(config, "get")):
            vlm = config.get("vlm")
        if vlm is None:
            return StubVisionJudge()
    enabled = getattr(vlm, "enabled", False) if not isinstance(vlm, dict) else vlm.get("enabled", False)
    if not enabled:
        return StubVisionJudge()
    backend = getattr(vlm, "backend", "none") if not isinstance(vlm, dict) else (vlm.get("backend") or "none")
    if backend == "none":
        return StubVisionJudge()
    if backend == "llama_cpp":
        try:
            from backends.llama_mm import LlamaMMJudge
            model_path = getattr(vlm, "model_path", "") if not isinstance(vlm, dict) else (vlm.get("model_path") or "")
            max_queries = getattr(vlm, "max_clock_calibration_queries", 20) if not isinstance(vlm, dict) else (vlm.get("max_clock_calibration_queries") or 20)
            n_gpu_layers = getattr(vlm, "n_gpu_layers", 0) if not isinstance(vlm, dict) else vlm.get("n_gpu_layers", 0)
            return LlamaMMJudge(model_path=model_path, max_queries=int(max_queries), n_gpu_layers=int(n_gpu_layers))
        except Exception:
            return StubVisionJudge()
    return StubVisionJudge()
