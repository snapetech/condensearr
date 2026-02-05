"""
Optional VLM backend: llama.cpp multimodal (CPU or GPU via offload).
Uses llama-cpp-python with MoondreamChatHandler for vision (e.g. Moondream2).
Calibrator: "which candidate ROI is the game clock?" (few queries).
Appeals: "live / replay / commercial?" for ambiguous segments (capped).
"""
from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from vision_judge import IVisionJudge, RoiBox

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

# Optional: llama-cpp-python with vision (pip install llama-cpp-python huggingface-hub)
_llama = None
_moondream_handler = None
try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import MoondreamChatHandler
    _llama = Llama
    _moondream_handler = MoondreamChatHandler
except ImportError:
    pass


def _is_repo_id(model_path: str) -> bool:
    """True if model_path looks like a HuggingFace repo id (org/name) rather than a local path."""
    if not model_path or not isinstance(model_path, str):
        return False
    p = Path(model_path).expanduser()
    if p.is_absolute() or p.exists():
        return False
    return "/" in model_path and "\\" not in model_path


class LlamaMMJudge(IVisionJudge):
    """
    VLM judge using Moondream2 (HuggingFace repo or local GGUF + mmproj).
    CPU-only if n_gpu_layers=0; set n_gpu_layers=-1 to offload to GPU.
    """

    def __init__(
        self,
        model_path: str = "",
        *,
        n_gpu_layers: int = 0,
        max_queries: int = 20,
        max_appeals_queries: int = 500,
    ):
        self._model_path_raw = (model_path or "").strip()
        self._model_path_resolved: Optional[Path] = None
        if self._model_path_raw and not _is_repo_id(self._model_path_raw):
            p = Path(self._model_path_raw).expanduser().resolve()
            if p.exists():
                self._model_path_resolved = p
        self.n_gpu_layers = n_gpu_layers
        self.max_clock_queries = max(1, min(max_queries, 30))
        self.max_appeals_queries = max(0, max_appeals_queries)
        self._model = None
        self._use_chat_handler = False
        self._query_count = 0

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if _llama is None:
            return False
        repo_id = self._model_path_raw if _is_repo_id(self._model_path_raw) else None
        try:
            if repo_id and _moondream_handler is not None:
                # Load Moondream2 from HuggingFace with chat handler (vision)
                chat_handler = _moondream_handler.from_pretrained(
                    repo_id=repo_id,
                    filename="*mmproj*",
                )
                self._model = _llama.from_pretrained(
                    repo_id=repo_id,
                    filename="*text-model*",
                    chat_handler=chat_handler,
                    n_ctx=2048,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=False,
                )
                self._use_chat_handler = True
                return True
            if self._model_path_resolved and self._model_path_resolved.exists():
                # Local path: try plain load (legacy single-GGUF or text-only fallback)
                self._model = _llama(
                    model_path=str(self._model_path_resolved),
                    n_ctx=512,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=False,
                )
                return True
        except Exception:
            pass
        return False

    def _image_to_path(self, img_bgr: np.ndarray, prefix: str = "crop") -> Optional[Path]:
        if cv2 is None or img_bgr is None or img_bgr.size == 0:
            return None
        fd, path = tempfile.mkstemp(suffix=".png", prefix=prefix)
        try:
            import os
            cv2.imwrite(path, img_bgr)
            return Path(path)
        except Exception:
            import os
            try:
                os.close(fd)
                os.unlink(path)
            except Exception:
                pass
            return None

    def _image_to_data_uri(self, image_path: Path) -> str:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def _prompt_with_image(self, image_path: Path, prompt: str, max_tokens: int = 32) -> str:
        if not self._ensure_model():
            return ""
        self._query_count += 1
        # Prefer chat completion with image (data URI works with MoondreamChatHandler)
        try:
            image_url = self._image_to_data_uri(image_path)
            out = self._model.create_chat_completion(
                messages=[
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": prompt},
                    ]},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            if out and "choices" in out and len(out["choices"]) > 0:
                c = out["choices"][0].get("message", {})
                return (c.get("content") or "").strip()
        except Exception:
            pass
        try:
            out = self._model.create_completion(
                prompt=prompt,
                image_path=str(image_path),
                max_tokens=max_tokens,
                temperature=0.0,
            )
            if out and "choices" in out and len(out["choices"]) > 0:
                return (out["choices"][0].get("text") or "").strip()
        except (TypeError, Exception):
            pass
        return ""

    def which_roi_is_clock(
        self,
        frames_bgr: List[np.ndarray],
        candidate_rois: List[RoiBox],
    ) -> Optional[Tuple[int, float]]:
        if not frames_bgr or not candidate_rois or cv2 is None:
            return None
        if not self._ensure_model():
            return None
        H, W = frames_bgr[0].shape[:2]
        for idx, (x, y, w, h) in enumerate(candidate_rois):
            if self._query_count >= self.max_clock_queries:
                break
            crop = frames_bgr[0][y : y + h, x : x + w]
            if crop is None or crop.size == 0:
                continue
            path = self._image_to_path(crop, prefix="clock_cand")
            if path is None:
                continue
            try:
                prompt = "Does this image show a sports game clock or scoreboard? Answer only: yes or no."
                ans = self._prompt_with_image(path, prompt).lower()
                if "yes" in ans:
                    return (idx, 0.9)
            finally:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass
        return None

    def classify_segment(
        self,
        frames_bgr: List[np.ndarray],
        context: str,
    ) -> str:
        if not frames_bgr or self._query_count >= self.max_appeals_queries:
            return "unknown"
        if not self._ensure_model() or cv2 is None:
            return "unknown"
        # Use middle frame
        mid = frames_bgr[len(frames_bgr) // 2] if frames_bgr else None
        if mid is None or mid.size == 0:
            return "unknown"
        path = self._image_to_path(mid, prefix="appeal")
        if path is None:
            return "unknown"
        try:
            prompt = "Is this from live game play, a replay, or a commercial? Answer with exactly one word: live, replay, or commercial."
            ans = self._prompt_with_image(path, prompt).lower()
            for label in ("live", "replay", "commercial"):
                if label in ans:
                    return label
            return "unknown"
        finally:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
