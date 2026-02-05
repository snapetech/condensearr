"""
Propose candidate scorebug/clock ROIs from a list of frames (CV only, CPU).
Used as input for OCR validation or VLM calibrator ("which candidate is the clock?").
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

# ROI as (x, y, w, h) to avoid coupling to condensearr.Roi
RoiBox = Tuple[int, int, int, int]


def propose_scorebug_candidates(
    frames_bgr: List[np.ndarray],
    *,
    min_area_frac: float = 0.002,
    max_area_frac: float = 0.08,
    max_candidates: int = 12,
) -> List[RoiBox]:
    """
    Propose up to max_candidates bounding boxes that are likely scorebug/HUD regions
    (persistent, edge-dense). Returns list of (x, y, w, h). Requires opencv-python.
    """
    if cv2 is None or len(frames_bgr) < 12:
        return []

    H, W = frames_bgr[0].shape[:2]
    gray_stack = np.stack(
        [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr], axis=0
    ).astype(np.float32)

    med = np.median(gray_stack, axis=0).astype(np.uint8)
    var = np.var(gray_stack, axis=0).astype(np.float32)

    edges = cv2.Canny(med, 60, 180).astype(np.float32) / 255.0
    edges = cv2.GaussianBlur(edges, (0, 0), sigmaX=2.0)

    stability = 1.0 / (var + 8.0)
    stability = stability / (np.max(stability) + 1e-6)
    score = stability * edges
    score = cv2.GaussianBlur(score, (0, 0), sigmaX=3.0)

    thr = float(np.quantile(score, 0.995))
    mask = (score >= thr).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = float(W * H)
    candidates: List[Tuple[float, RoiBox]] = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = float(w * h)
        if area < img_area * min_area_frac or area > img_area * max_area_frac:
            continue
        pad = int(max(6, 0.06 * max(w, h)))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(W, x + w + pad)
        y1 = min(H, y + h + pad)
        box = (x0, y0, x1 - x0, y1 - y0)
        cx = x0 + (x1 - x0) / 2.0
        cy = y0 + (y1 - y0) / 2.0
        dx = min(cx, W - cx) / float(W)
        dy = min(cy, H - cy) / float(H)
        corner_bias = 1.0 - (dx + dy)
        size_bias = min(1.0, area / (img_area * 0.03))
        candidates.append((0.65 * corner_bias + 0.35 * size_bias, box))

    candidates.sort(key=lambda t: t[0], reverse=True)
    return [box for _, box in candidates[:max_candidates]]
