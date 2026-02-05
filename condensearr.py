#!/usr/bin/env python3
"""
condensearr.py

Local-file "condensed game" builder that fuses multiple signals:

Video intelligence
  - Motion energy from low-res grayscale frames (mean absolute difference)

Audio intelligence
  - RMS loudness envelope
  - Spectral flux (onsets / rapid changes)
  - Whistle-like detector (high-frequency narrowband peakness)

Clock intelligence (optional, Strategy B)
  - OCR of scorebug clock to infer "clock running" probability
  - Optional AUTO-DETECT of scorebug location (requires opencv-python)

Synthesis
  - Builds three cutlists:
      A) Action-only (Strategy A)
      B) Clock-only (running-clock mask)
      C) Fused (default): hard events + clock-weighted action
  - Picks C as the render by default, and writes diagnostics.

This script is for processing local recordings you have rights to process.
It does not fetch streams and does not bypass DRM.
Runs as fast as disk/CPU allow (no real-time throttling).

Requirements:
  - ffmpeg + ffprobe on PATH
Python:
  - numpy (required)
Optional:
  - pytesseract + system tesseract-ocr (for OCR clock)
  - opencv-python (for auto-detect scorebug ROI and debug images)

Install:
  pip install numpy pytesseract opencv-python

Example:
  python condensearr.py input.mkv --config condensearr_config.example.json --target-minutes 18 --out condensed.mkv

Arr/Tdarr automation:
  - Use --out-dir DIR to write output to a directory (e.g. Tdarr cache or a library).
  - Set CONDENSEARR_CONFIG to a default config path so the CLI needs no --config.
  - Use --min-duration SEC to skip short files (exit 0 without processing); useful for filters.
  See docs/ARR_INTEGRATION.md for Tdarr and *Arr custom script examples.

ROI:
  If you don't want to hand-pick a clock ROI, set:
    "ocr": { "auto_detect": true, ... }
  and install opencv-python.

License: MIT. Copyright (c) 2025 snapetech. See LICENSE in the repo root.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    import cv2
except Exception:
    cv2 = None


TIME_RE = re.compile(r'(\d{1,3})\s*[:.]\s*(\d{2})')


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def _kill_prior_instances() -> None:
    """Kill other running condensearr.py processes (not this one)."""
    try:
        cp = subprocess.run(
            ["pgrep", "-f", "condensearr.py"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return
    mypid = os.getpid()
    for line in (cp.stdout or "").strip().splitlines():
        try:
            pid = int(line.strip())
            if pid != mypid:
                os.kill(pid, signal.SIGTERM)
                eprint(f"Sent SIGTERM to prior condensearr pid {pid}")
        except (ValueError, ProcessLookupError, PermissionError):
            pass


class ToolError(RuntimeError):
    pass


def which_or_die(name: str) -> str:
    p = shutil.which(name)
    if not p:
        raise ToolError(f"Missing required tool on PATH: {name}")
    return p


# When True, run ffmpeg/ffprobe under nice + ionice and use lower default parallelism.
_play_nice: bool = True


def _nice_prefix() -> List[str]:
    """Prefix for subprocess to run with low CPU priority and idle I/O (Linux)."""
    if not _play_nice:
        return []
    prefix: List[str] = []
    if shutil.which("nice"):
        prefix.extend(["nice", "-n", "19"])
    if sys.platform == "linux" and shutil.which("ionice"):
        prefix.extend(["ionice", "-c", "3"])  # idle I/O
    return prefix


def run(cmd: List[str], *, capture: bool = False, check: bool = True, text: bool = True) -> subprocess.CompletedProcess:
    full_cmd = _nice_prefix() + cmd
    eprint("RUN:", " ".join(full_cmd))
    return subprocess.run(full_cmd, stdout=subprocess.PIPE if capture else None, stderr=subprocess.PIPE if capture else None, check=check, text=text)


def ffprobe_duration(input_path: Path) -> float:
    which_or_die("ffprobe")
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]
    cp = run(cmd, capture=True)
    try:
        return float(cp.stdout.strip())
    except Exception as ex:
        raise ToolError(f"ffprobe failed to parse duration: {cp.stdout!r}") from ex


def ffprobe_video_fps(input_path: Path) -> float:
    which_or_die("ffprobe")
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]
    cp = run(cmd, capture=True)
    rate = cp.stdout.strip()
    if not rate:
        return 30.0
    if "/" in rate:
        num, den = rate.split("/", 1)
        try:
            return float(num) / float(den)
        except Exception:
            return 30.0
    try:
        return float(rate)
    except Exception:
        return 30.0


def robust_z(x: np.ndarray) -> np.ndarray:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad < 1e-9:
        return (x - med) * 0.0
    return (x - med) / (1.4826 * mad)


def smooth_ema(x: np.ndarray, alpha: float) -> np.ndarray:
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc = alpha * float(x[i]) + (1.0 - alpha) * acc
        y[i] = acc
    return y


def resample_to_dt(x: np.ndarray, x_dt: float, dt: float, duration: float, *, mode: str = "nearest") -> np.ndarray:
    n = int(math.ceil(duration / dt))
    y = np.zeros(n, dtype=np.float32)
    if len(x) == 0:
        return y

    if mode not in ("nearest", "mean", "max"):
        mode = "nearest"

    if mode == "nearest":
        for i in range(n):
            t = i * dt
            j = int(round(t / x_dt))
            if 0 <= j < len(x):
                y[i] = float(x[j])
        return y

    # aggregate samples that fall into each dt bin
    counts = np.zeros(n, dtype=np.int32)
    if mode == "mean":
        for j, v in enumerate(x):
            t = j * x_dt
            i = int(t / dt)
            if 0 <= i < n:
                y[i] += float(v)
                counts[i] += 1
        for i in range(n):
            if counts[i] > 0:
                y[i] = y[i] / float(counts[i])
        return y

    # mode == "max"
    y[:] = 0.0
    have = np.zeros(n, dtype=np.uint8)
    for j, v in enumerate(x):
        t = j * x_dt
        i = int(t / dt)
        if 0 <= i < n:
            if not have[i] or float(v) > float(y[i]):
                y[i] = float(v)
                have[i] = 1
    return y


def extract_audio_pcm_s16le(input_path: Path, tmpdir: Path, *, sample_rate: int = 16000) -> Path:
    which_or_die("ffmpeg")
    out = tmpdir / "audio.s16le"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-threads", "0",
        "-i", str(input_path),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "s16le",
        str(out),
    ]
    run(cmd, capture=True)
    return out


def compute_audio_features(raw_pcm_path: Path, *, sample_rate: int, hop_seconds: float) -> Tuple[Dict[str, np.ndarray], float]:
    """
    Returns dict of features at hop_seconds:
      - rms
      - flux
      - whistle
    """
    data = raw_pcm_path.read_bytes()
    if len(data) == 0:
        return {"rms": np.zeros(0, dtype=np.float32), "flux": np.zeros(0, dtype=np.float32), "whistle": np.zeros(0, dtype=np.float32)}, hop_seconds

    x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

    hop = int(sample_rate * hop_seconds)
    if hop <= 0:
        hop = 1

    # Window length: short enough to catch onsets/whistles, long enough for stability
    win = int(sample_rate * min(0.08, max(0.02, hop_seconds)))
    if win < 256:
        win = 256
    if win > 2048:
        win = 2048
    if win > hop * 4:
        win = hop * 4

    n = len(x)
    m = max(0, 1 + (n - win) // hop)
    if m <= 0:
        return {"rms": np.zeros(0, dtype=np.float32), "flux": np.zeros(0, dtype=np.float32), "whistle": np.zeros(0, dtype=np.float32)}, hop_seconds

    # Hann window
    w = 0.5 - 0.5 * np.cos(2.0 * math.pi * np.arange(win) / float(win))
    w = w.astype(np.float32)

    nfft = 1
    while nfft < win:
        nfft *= 2

    # Frequency bins
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
    # Whistle band approx (sports whistles often 2.5-4.5kHz)
    band_lo = 2500.0
    band_hi = 4500.0
    band = np.where((freqs >= band_lo) & (freqs <= band_hi))[0]
    if len(band) < 3:
        band = np.where((freqs >= 1800.0) & (freqs <= 5000.0))[0]

    rms = np.empty(m, dtype=np.float32)
    flux = np.empty(m, dtype=np.float32)
    whistle = np.empty(m, dtype=np.float32)

    prev_mag = None

    for i in range(m):
        s = i * hop
        seg = x[s:s + win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        segw = seg * w

        # RMS
        rms[i] = float(np.sqrt(np.mean(segw * segw)))

        # Spectrum magnitude
        spec = np.fft.rfft(segw, n=nfft)
        mag = np.abs(spec).astype(np.float32)

        # Spectral flux (L1 on positive differences)
        if prev_mag is None:
            flux[i] = 0.0
        else:
            d = mag - prev_mag
            d[d < 0.0] = 0.0
            flux[i] = float(np.mean(d))
        prev_mag = mag

        # Whistle-like score: high-band energy ratio * peakness
        tot = float(np.mean(mag) + 1e-9)
        b = mag[band]
        b_mean = float(np.mean(b) + 1e-9)
        b_max = float(np.max(b))
        ratio = float(b_mean / tot)
        peakness = float(b_max / b_mean)
        whistle[i] = ratio * peakness

    return {"rms": rms, "flux": flux, "whistle": whistle}, hop_seconds


def extract_gray_frames_raw(input_path: Path, *, fps: float, width: int, tmpdir: Path) -> Tuple[Path, int, int, float]:
    which_or_die("ffmpeg")
    raw_path = tmpdir / "frames.gray.raw"
    vf = f"fps={fps},scale={width}:-1:flags=bilinear,format=gray"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-threads", "0",
        "-i", str(input_path),
        "-an",
        "-vf", vf,
        "-f", "rawvideo",
        str(raw_path),
    ]
    run(cmd, capture=True)

    png = tmpdir / "one.png"
    cmd2 = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-threads", "0",
        "-ss", "0",
        "-i", str(input_path),
        "-frames:v", "1",
        "-vf", f"scale={width}:-1:flags=bilinear",
        str(png),
    ]
    run(cmd2, capture=True)
    out_w, out_h = ffprobe_image_dims(png)
    return raw_path, out_w, out_h, fps


def ffprobe_image_dims(png_path: Path) -> Tuple[int, int]:
    which_or_die("ffprobe")
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        str(png_path),
    ]
    cp = run(cmd, capture=True)
    s = cp.stdout.strip()
    if "x" not in s:
        raise ToolError(f"Could not read image dims for {png_path}: {s!r}")
    w, h = s.split("x", 1)
    return int(w), int(h)


def compute_motion_energy(raw_path: Path, *, w: int, h: int, fps: float) -> Tuple[np.ndarray, float]:
    frame_bytes = w * h
    data = raw_path.read_bytes()
    if len(data) < frame_bytes:
        return np.zeros(0, dtype=np.float32), 1.0 / fps
    n_frames = len(data) // frame_bytes
    buf = np.frombuffer(data[:n_frames * frame_bytes], dtype=np.uint8).reshape((n_frames, h, w))
    if n_frames < 2:
        return np.zeros(n_frames, dtype=np.float32), 1.0 / fps
    diffs = np.abs(buf[1:].astype(np.int16) - buf[:-1].astype(np.int16))
    energy = diffs.mean(axis=(1, 2)).astype(np.float32)
    energy = np.concatenate([np.array([0.0], dtype=np.float32), energy])
    return energy, 1.0 / fps


@dataclass
class Roi:
    x: int
    y: int
    w: int
    h: int


@dataclass
class OcrConfig:
    auto_detect: bool = False
    clock_roi: Optional[Roi] = None
    fps: float = 2.0
    psm: int = 7
    whitelist: str = "0123456789:."
    invert: bool = False
    threshold: Optional[int] = None
    scale: float = 2.0
    min_hit_rate: float = 0.15


@dataclass
class AudioWeights:
    w_rms: float = 0.35
    w_flux: float = 0.20
    w_whistle: float = 0.10
    w_motion: float = 0.35
    hard_whistle_z: float = 2.6


@dataclass
class CombineConfig:
    dt: float = 0.5
    target_minutes: float = 18.0
    padding_pre: float = 2.0
    padding_post: float = 4.0
    min_segment: float = 2.0
    merge_gap: float = 1.25
    hard_peak_z: float = 2.0
    require_ocr: bool = False
    max_segments: int = 220
    clock_weight: float = 1.0
    render_mode: str = "fused"  # fused | action | clock


@dataclass
class RenderConfig:
    crf: int = 20
    preset: str = "veryfast"
    audio_bitrate: str = "192k"
    container: str = "mkv"
    threads: int = 0  # 0 = ffmpeg default (auto)
    parallel_jobs: int = 0  # 0 = auto (cpu_count)


@dataclass
class AppConfig:
    ocr: Optional[OcrConfig]
    audio_weights: AudioWeights
    combine: CombineConfig
    render: RenderConfig


def parse_config(path: Path) -> AppConfig:
    cfg = json.loads(path.read_text(encoding="utf-8"))

    ocr_cfg = None
    if "ocr" in cfg and cfg["ocr"] is not None:
        o = cfg["ocr"]
        roi = o.get("clock_roi", None)
        clock_roi = Roi(**roi) if roi else None
        ocr_cfg = OcrConfig(
            auto_detect=bool(o.get("auto_detect", False)),
            clock_roi=clock_roi,
            fps=float(o.get("fps", 2.0)),
            psm=int(o.get("psm", 7)),
            whitelist=str(o.get("whitelist", "0123456789:.")),
            invert=bool(o.get("invert", False)),
            threshold=o.get("threshold", None),
            scale=float(o.get("scale", 2.0)),
            min_hit_rate=float(o.get("min_hit_rate", 0.15)),
        )

    aw = cfg.get("audio_weights", {})
    audio_weights = AudioWeights(
        w_rms=float(aw.get("w_rms", 0.35)),
        w_flux=float(aw.get("w_flux", 0.20)),
        w_whistle=float(aw.get("w_whistle", 0.10)),
        w_motion=float(aw.get("w_motion", 0.35)),
        hard_whistle_z=float(aw.get("hard_whistle_z", 2.6)),
    )

    c = cfg.get("combine", {})
    combine = CombineConfig(
        dt=float(c.get("dt", 0.5)),
        target_minutes=float(c.get("target_minutes", 18.0)),
        padding_pre=float(c.get("padding_pre", 2.0)),
        padding_post=float(c.get("padding_post", 4.0)),
        min_segment=float(c.get("min_segment", 2.0)),
        merge_gap=float(c.get("merge_gap", 1.25)),
        hard_peak_z=float(c.get("hard_peak_z", 2.0)),
        require_ocr=bool(c.get("require_ocr", False)),
        max_segments=int(c.get("max_segments", 220)),
        clock_weight=float(c.get("clock_weight", 1.0)),
        render_mode=str(c.get("render_mode", "fused")),
    )

    r = cfg.get("render", {})
    ncpu = max(1, (os.cpu_count() or 4))
    render = RenderConfig(
        crf=int(r.get("crf", 20)),
        preset=str(r.get("preset", "veryfast")),
        audio_bitrate=str(r.get("audio_bitrate", "192k")),
        container=str(r.get("container", "mkv")),
        threads=int(r.get("threads", 0)),
        parallel_jobs=int(r.get("parallel_jobs", 0)) or ncpu,
    )
    return AppConfig(ocr=ocr_cfg, audio_weights=audio_weights, combine=combine, render=render)


def parse_time_str(s: str) -> Optional[int]:
    m = TIME_RE.search(s)
    if not m:
        return None
    mm = int(m.group(1))
    ss = int(m.group(2))
    if ss >= 60:
        return None
    return mm * 60 + ss


def ocr_one_image(img_gray: np.ndarray, cfg: OcrConfig) -> Optional[str]:
    if pytesseract is None:
        return None
    config = f"--psm {cfg.psm} -c tessedit_char_whitelist={cfg.whitelist}"
    try:
        txt = pytesseract.image_to_string(img_gray, config=config)
    except Exception:
        return None
    txt = txt.strip()
    if not txt:
        return None
    cleaned = re.sub(r'[^0-9:\.\s]', '', txt).strip()
    return cleaned if cleaned else None


def extract_roi_frames(input_path: Path, tmpdir: Path, roi: Roi, *, fps: float, scale: float, invert: bool, threshold: Optional[int]) -> List[Path]:
    which_or_die("ffmpeg")
    outdir = tmpdir / "roi"
    outdir.mkdir(parents=True, exist_ok=True)
    outpat = outdir / "roi_%06d.png"
    filters = [f"crop={roi.w}:{roi.h}:{roi.x}:{roi.y}"]
    if scale and scale != 1.0:
        filters.append(f"scale=iw*{scale}:ih*{scale}:flags=bilinear")
    filters.append("format=gray")
    if invert:
        filters.append("negate")
    if threshold is not None:
        filters.append(f"threshold={threshold}")
    vf = ",".join(filters)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i", str(input_path),
        "-an",
        "-vf", f"fps={fps},{vf}",
        str(outpat),
    ]
    run(cmd, capture=True)
    return sorted(outdir.glob("roi_*.png"))


def infer_clock_running(times: List[Optional[int]], fps: float, duration: float) -> Tuple[np.ndarray, float]:
    """
    Return running_prob at OCR sampling points and a confidence score in [0,1].
    Confidence is based on:
      - parse rate
      - consistency of decreasing time when running
    """
    n = len(times)
    if n == 0:
        return np.zeros(0, dtype=np.float32), 0.0

    parsed = [t for t in times if t is not None]
    parse_rate = len(parsed) / float(n)

    run_prob = np.zeros(n, dtype=np.float32)
    last = None
    last_i = None
    good_steps = 0
    total_steps = 0

    for i, t in enumerate(times):
        if t is None:
            continue
        if last is not None and last_i is not None:
            dt = (i - last_i) / fps
            expected = last - int(round(dt))
            if abs(t - expected) <= 1:
                run_prob[i] = 1.0
                good_steps += 1
            total_steps += 1
        last = t
        last_i = i

    run_prob = smooth_ema(run_prob, alpha=0.35)
    consistency = (good_steps / float(max(1, total_steps)))
    conf = float(max(0.0, min(1.0, 0.55 * parse_rate + 0.45 * consistency)))
    return run_prob, conf


def mask_from_running_prob(run_prob: np.ndarray, fps: float, duration: float) -> np.ndarray:
    out_n = int(math.ceil(duration))
    up = np.zeros(out_n, dtype=np.float32)
    if len(run_prob) == 0:
        return up
    for sec in range(out_n):
        idx = int(round(sec * fps))
        if 0 <= idx < len(run_prob):
            up[sec] = float(run_prob[idx])
    return up


def segments_from_mask(mask: np.ndarray, dt: float, min_len: float) -> List[Tuple[float, float]]:
    segs: List[Tuple[float, float]] = []
    in_seg = False
    s = 0
    for i, v in enumerate(mask):
        if v and not in_seg:
            in_seg = True
            s = i
        if in_seg and (not v or i == len(mask) - 1):
            e = i if not v else i + 1
            start = s * dt
            end = e * dt
            if end - start >= min_len:
                segs.append((start, end))
            in_seg = False
    return segs


def merge_segments(segs: List[Tuple[float, float]], gap: float) -> List[Tuple[float, float]]:
    if not segs:
        return []
    segs = sorted(segs, key=lambda x: x[0])
    out = [segs[0]]
    for a, b in segs[1:]:
        la, lb = out[-1]
        if a <= lb + gap:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out


def pad_segments(segs: List[Tuple[float, float]], pre: float, post: float, duration: float) -> List[Tuple[float, float]]:
    out = []
    for a, b in segs:
        out.append((max(0.0, a - pre), min(duration, b + post)))
    return out


def total_len(segs: List[Tuple[float, float]]) -> float:
    return sum(max(0.0, b - a) for a, b in segs)


def compute_cut_quality_stats(
    action_z: np.ndarray,
    hard_event_mask: np.ndarray,
    dt: float,
    duration: float,
    segs: List[Tuple[float, float]],
) -> Dict[str, float]:
    """
    Compute stats used to validate that kept segments have higher action than cut
    and that hard events (whistles, peaks) are mostly inside kept segments.
    """
    n = len(action_z)
    if n == 0:
        return {
            "kept_action_mean": 0.0,
            "cut_action_mean": 0.0,
            "action_ratio_kept_vs_cut": 1.0,
            "hard_events_total": 0,
            "hard_events_in_kept": 0,
            "fraction_hard_in_kept": 1.0,
            "max_gap_seconds": 0.0,
        }
    in_kept = np.zeros(n, dtype=np.uint8)
    for a, b in segs:
        i0 = max(0, int(a / dt))
        i1 = min(n, int(b / dt) + 1)
        if i0 < i1:
            in_kept[i0:i1] = 1
    kept_action = action_z[in_kept == 1]
    cut_action = action_z[in_kept == 0]
    kept_mean = float(np.mean(kept_action)) if len(kept_action) > 0 else 0.0
    cut_mean = float(np.mean(cut_action)) if len(cut_action) > 0 else 0.0
    ratio = kept_mean / (cut_mean + 1e-9) if cut_mean != 0 else (10.0 if kept_mean > 0 else 1.0)
    hard_total = int(np.sum(hard_event_mask))
    hard_in_kept = int(np.sum(hard_event_mask * in_kept))
    frac_hard = (hard_in_kept / hard_total) if hard_total > 0 else 1.0
    # Max gap between consecutive kept segments
    sorted_segs = sorted(segs, key=lambda x: x[0])
    max_gap = 0.0
    for i in range(1, len(sorted_segs)):
        gap = sorted_segs[i][0] - sorted_segs[i - 1][1]
        if gap > max_gap:
            max_gap = gap
    return {
        "kept_action_mean": kept_mean,
        "cut_action_mean": cut_mean,
        "action_ratio_kept_vs_cut": ratio,
        "hard_events_total": hard_total,
        "hard_events_in_kept": hard_in_kept,
        "fraction_hard_in_kept": frac_hard,
        "max_gap_seconds": max_gap,
    }


def write_edl(segs: List[Tuple[float, float]], path: Path) -> None:
    lines = []
    for a, b in segs:
        lines.append(f"{a:.3f} {b:.3f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def binary_search_quantile(action_z: np.ndarray, base_mask: np.ndarray, dt: float, target_seconds: float, min_len: float, merge_gap: float, max_iter: int = 18) -> Tuple[float, List[Tuple[float, float]]]:
    lo = 0.50
    hi = 0.985
    best_q = hi
    best_segs: List[Tuple[float, float]] = []
    best_err = float("inf")
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        thr = float(np.quantile(action_z, mid))
        m = (action_z >= thr).astype(np.uint8)
        m = (m * (base_mask > 0.5).astype(np.uint8)).astype(np.uint8)
        segs = merge_segments(segments_from_mask(m, dt, min_len), merge_gap)
        L = total_len(segs)
        err = abs(L - target_seconds)
        if err < best_err:
            best_err = err
            best_q = mid
            best_segs = segs
        if L > target_seconds:
            lo = mid
        else:
            hi = mid
    return best_q, best_segs


def pick_segments(
    action_z: np.ndarray,
    hard_event_mask: np.ndarray,
    clock_mask_dt: Optional[np.ndarray],
    clock_conf: float,
    cfg: CombineConfig,
    duration: float
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Returns dict with cutlists:
      - action
      - clock
      - fused
    """
    dt = cfg.dt
    target_seconds = cfg.target_minutes * 60.0

    ones = np.ones_like(action_z, dtype=np.float32)

    # Action-only: pick top quantile to hit target.
    q_a, segs_a = binary_search_quantile(action_z, ones, dt, target_seconds, cfg.min_segment, cfg.merge_gap)

    # Clock-only: just running-clock as segments, then trimmed to target if needed by taking earliest/highest-action chunks.
    segs_c: List[Tuple[float, float]] = []
    if clock_mask_dt is not None and len(clock_mask_dt) > 0:
        segs_c = merge_segments(segments_from_mask((clock_mask_dt > 0.5).astype(np.uint8), dt, cfg.min_segment), cfg.merge_gap)
        # If clock-only is longer than target (usually), reduce by selecting subsegments with highest action.
        if total_len(segs_c) > target_seconds:
            # score each segment by mean action, keep best until target
            scored = []
            for a, b in segs_c:
                i0 = int(a / dt)
                i1 = int(b / dt)
                if i1 <= i0:
                    continue
                score = float(np.mean(action_z[i0:i1]))
                scored.append((score, (a, b)))
            scored.sort(key=lambda t: t[0], reverse=True)
            keep = []
            acc = 0.0
            for _, seg in scored:
                L = seg[1] - seg[0]
                if acc + L <= target_seconds:
                    keep.append(seg)
                    acc += L
                if acc >= target_seconds * 0.995:
                    break
            segs_c = merge_segments(sorted(keep, key=lambda x: x[0]), cfg.merge_gap)

    # Fused: hard events + clock-weighted fill.
    base = ones
    if clock_mask_dt is not None and len(clock_mask_dt) > 0:
        # Degrade gracefully: if OCR confidence is low, base leans toward "ones" (Strategy A)
        w = max(0.0, min(1.0, cfg.clock_weight * clock_conf))
        base = (1.0 - w) * ones + w * clock_mask_dt.astype(np.float32)

    hard_segs = merge_segments(segments_from_mask(hard_event_mask.astype(np.uint8), dt, cfg.min_segment), cfg.merge_gap)
    remaining_target = max(0.0, target_seconds - total_len(hard_segs))
    q_f, fill = binary_search_quantile(action_z, base, dt, remaining_target, cfg.min_segment, cfg.merge_gap)

    segs_f = merge_segments(hard_segs + fill, cfg.merge_gap)

    # Pad + merge + cap
    segs_a = merge_segments(pad_segments(segs_a, cfg.padding_pre, cfg.padding_post, duration), cfg.merge_gap)
    segs_c = merge_segments(pad_segments(segs_c, cfg.padding_pre, cfg.padding_post, duration), cfg.merge_gap) if segs_c else []
    segs_f = merge_segments(pad_segments(segs_f, cfg.padding_pre, cfg.padding_post, duration), cfg.merge_gap)

    if len(segs_a) > cfg.max_segments:
        segs_a = segs_a[:cfg.max_segments]
    if len(segs_c) > cfg.max_segments:
        segs_c = segs_c[:cfg.max_segments]
    if len(segs_f) > cfg.max_segments:
        segs_f = segs_f[:cfg.max_segments]

    return {"action": segs_a, "clock": segs_c, "fused": segs_f}


def _encode_one_segment(
    input_path: Path,
    seg: Tuple[float, float],
    part: Path,
    render: RenderConfig,
) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-ss", f"{seg[0]:.3f}",
        "-to", f"{seg[1]:.3f}",
        "-i", str(input_path),
        "-map", "0:v:0?",
        "-map", "0:a:0?",
        "-c:v", "libx264",
        "-crf", str(render.crf),
        "-preset", render.preset,
        "-c:a", "aac",
        "-b:a", render.audio_bitrate,
        "-movflags", "+faststart",
        "-avoid_negative_ts", "make_zero",
    ]
    if render.threads > 0:
        cmd.extend(["-threads", str(render.threads)])
    cmd.append(str(part))
    run(cmd, capture=True)


def render_concat_segments(input_path: Path, segs: List[Tuple[float, float]], out_path: Path, render: RenderConfig, tmpdir: Path) -> None:
    which_or_die("ffmpeg")
    parts_dir = tmpdir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    jobs = max(1, render.parallel_jobs) if render.parallel_jobs > 0 else max(1, (os.cpu_count() or 4))
    part_paths: List[Optional[Path]] = [None] * len(segs)

    def do_one(idx: int, a: float, b: float) -> int:
        part = parts_dir / f"part_{idx:04d}.{render.container}"
        _encode_one_segment(input_path, (a, b), part, render)
        return idx

    if jobs <= 1:
        for idx, (a, b) in enumerate(segs):
            part = parts_dir / f"part_{idx:04d}.{render.container}"
            _encode_one_segment(input_path, (a, b), part, render)
            part_paths[idx] = part
    else:
        eprint(f"Encoding {len(segs)} segments with {jobs} parallel jobs...")
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {executor.submit(do_one, idx, a, b): idx for idx, (a, b) in enumerate(segs)}
            for fut in as_completed(futures):
                idx = fut.result()
                part_paths[idx] = parts_dir / f"part_{idx:04d}.{render.container}"

    part_paths = [p for p in part_paths if p is not None]
    part_paths.sort(key=lambda p: p.name)

    concat_list = tmpdir / "concat.txt"
    lines = [f"file '{p.as_posix()}'" for p in part_paths]
    concat_list.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cmd2 = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        str(out_path),
    ]
    run(cmd2, capture=True)


def make_preview_frame(input_path: Path, out_png: Path, *, t: float) -> None:
    which_or_die("ffmpeg")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-ss", f"{t:.3f}",
        "-i", str(input_path),
        "-frames:v", "1",
        str(out_png),
    ]
    run(cmd, capture=True)


def ffmpeg_sample_pngs(input_path: Path, out_dir: Path, *, fps: float, max_frames: int, scale_w: int) -> List[Path]:
    which_or_die("ffmpeg")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pat = out_dir / "f_%06d.png"
    vf = f"fps={fps},scale={scale_w}:-1:flags=bilinear"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i", str(input_path),
        "-an",
        "-vf", vf,
        "-frames:v", str(max_frames),
        str(out_pat),
    ]
    run(cmd, capture=True)
    return sorted(out_dir.glob("f_*.png"))


def ocr_time_hit_rate(frames_bgr: List[np.ndarray], roi: Roi, cfg: OcrConfig) -> float:
    if pytesseract is None:
        return 0.0
    hits = 0
    total = 0
    for img in frames_bgr:
        h, w = img.shape[:2]
        x0 = max(0, min(w - 1, roi.x))
        y0 = max(0, min(h - 1, roi.y))
        x1 = max(1, min(w, roi.x + roi.w))
        y1 = max(1, min(h, roi.y + roi.h))
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue

        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if cfg.scale and cfg.scale != 1.0:
            g = cv2.resize(g, None, fx=cfg.scale, fy=cfg.scale, interpolation=cv2.INTER_LINEAR)
        g = cv2.GaussianBlur(g, (3, 3), 0)

        if cfg.invert:
            g = 255 - g

        if cfg.threshold is None:
            _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, bw = cv2.threshold(g, int(cfg.threshold), 255, cv2.THRESH_BINARY)

        s = ocr_one_image(bw, cfg)
        t = parse_time_str(s) if s else None
        total += 1
        if t is not None:
            hits += 1
    if total == 0:
        return 0.0
    return hits / float(total)


def autodetect_scorebug_roi(
    input_path: Path,
    tmpdir: Path,
    *,
    sample_fps: float = 1.0,
    max_frames: int = 160,
    scale_w: int = 640,
    min_area_frac: float = 0.002,
    max_area_frac: float = 0.08,
    candidates_keep: int = 12,
    ocr_cfg: Optional[OcrConfig] = None,
    debug_dir: Optional[Path] = None
) -> Optional[Roi]:
    """
    Detect a persistent, text/edge-dense overlay region (scorebug), then OCR-validate for time-like text.
    Requires opencv-python. OCR validation requires pytesseract.
    """
    if cv2 is None:
        return None

    frames_dir = tmpdir / "auto_frames"
    paths = ffmpeg_sample_pngs(input_path, frames_dir, fps=sample_fps, max_frames=max_frames, scale_w=scale_w)
    frames = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)

    if len(frames) < 12:
        return None

    H, W = frames[0].shape[:2]
    gray_stack = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames], axis=0).astype(np.float32)

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
    candidates: List[Tuple[float, Roi]] = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = float(w * h)
        if area < img_area * min_area_frac:
            continue
        if area > img_area * max_area_frac:
            continue

        pad = int(max(6, 0.06 * max(w, h)))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(W, x + w + pad)
        y1 = min(H, y + h + pad)
        roi = Roi(x=x0, y=y0, w=x1 - x0, h=y1 - y0)

        cx = roi.x + roi.w * 0.5
        cy = roi.y + roi.h * 0.5
        dx = min(cx, W - cx) / float(W)
        dy = min(cy, H - cy) / float(H)
        corner_bias = 1.0 - (dx + dy)
        size_bias = min(1.0, area / (img_area * 0.03))
        candidates.append((0.65 * corner_bias + 0.35 * size_bias, roi))

    candidates.sort(key=lambda t: t[0], reverse=True)
    candidates = candidates[:candidates_keep]

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "median.png"), med)
        score_img = (255.0 * (score / (np.max(score) + 1e-9))).astype(np.uint8)
        cv2.imwrite(str(debug_dir / "score.png"), score_img)
        cv2.imwrite(str(debug_dir / "mask.png"), mask)

    if not candidates:
        return None

    if pytesseract is None or ocr_cfg is None:
        return candidates[0][1]

    best_roi = None
    best_hit = -1.0
    for idx, (_, roi) in enumerate(candidates):
        hit = ocr_time_hit_rate(frames[:30], roi, ocr_cfg)
        if debug_dir is not None:
            x0, y0, w, h = roi.x, roi.y, roi.w, roi.h
            crop = frames[0][y0:y0+h, x0:x0+w]
            if crop is not None and crop.size > 0:
                cv2.imwrite(str(debug_dir / f"cand_{idx:02d}_hit_{hit:.2f}.png"), crop)
        if hit > best_hit:
            best_hit = hit
            best_roi = roi

    if best_hit < ocr_cfg.min_hit_rate:
        return None
    return best_roi


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a condensed cut from a local sports recording by fusing video/audio/OCR signals.")
    ap.add_argument("input", type=str, help="Input video file path (local).")
    ap.add_argument("--config", type=str, default="", help="JSON config path; default from CONDENSEARR_CONFIG env if set.")
    ap.add_argument("--target-minutes", type=float, default=None, help="Override target minutes.")
    ap.add_argument("--out", type=str, default="", help="Output file path.")
    ap.add_argument("--out-dir", type=str, default="", help="Output directory (filename = input stem + .condensed.<ext>). For Arr/Tdarr.")
    ap.add_argument("--min-duration", type=float, default=0, help="Skip if source duration < this many seconds (exit 0). For Arr filters.")
    ap.add_argument("--workdir", type=str, default="", help="Working directory (temp by default).")
    ap.add_argument("--require-ocr", action="store_true", help="Fail if OCR isn't available or can't produce usable signal.")
    ap.add_argument("--calibrate", action="store_true", help="Write a preview frame PNG and print ROI helper commands.")
    ap.add_argument("--clock-roi", type=str, default="", help="Override clock ROI as x,y,w,h.")
    ap.add_argument("--ocr-fps", type=float, default=None, help="Override OCR sampling fps.")
    ap.add_argument("--auto-roi", action="store_true", help="Auto-detect scorebug ROI (requires opencv-python; OCR recommended).")
    ap.add_argument("--debug-dir", type=str, default="", help="Directory for debug images (auto-ROI artifacts, optional).")
    ap.add_argument("--emit-diagnostics", type=str, default="", help="Write diagnostics JSON (includes validation stats for validate_condensed.py).")
    ap.add_argument("--jobs", type=int, default=0, help="Parallel segment encode jobs (0 = use all cores, or 2 when --nice).")
    ap.add_argument("--no-nice", action="store_true", help="Do not run ffmpeg under nice/ionice; use full CPU and I/O (faster, noisier).")
    ap.add_argument("--kill-prior", action="store_true", help="Kill any existing condensearr.py and their ffmpeg children before starting.")
    args = ap.parse_args()

    global _play_nice
    _play_nice = not args.no_nice
    if _play_nice:
        eprint("Playing nice: low CPU priority (nice 19), idle I/O (ionice -c 3), capped parallel jobs")

    if args.kill_prior:
        _kill_prior_instances()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        eprint(f"Input not found: {input_path}")
        return 2

    config_path = (args.config or os.environ.get("CONDENSEARR_CONFIG", "")).strip()
    if config_path:
        cfg = parse_config(Path(config_path).expanduser().resolve())
    else:
        cfg = AppConfig(ocr=None, audio_weights=AudioWeights(), combine=CombineConfig(), render=RenderConfig())

    if args.target_minutes is not None:
        cfg.combine.target_minutes = float(args.target_minutes)

    if args.jobs is not None and args.jobs > 0:
        cfg.render.parallel_jobs = args.jobs
    elif cfg.render.parallel_jobs <= 0:
        cfg.render.parallel_jobs = max(1, (os.cpu_count() or 4))
    if _play_nice and args.jobs <= 0:
        cfg.render.parallel_jobs = min(2, cfg.render.parallel_jobs)
        if cfg.render.threads <= 0:
            cfg.render.threads = 2

    if args.require_ocr:
        cfg.combine.require_ocr = True

    # Enable OCR from CLI when no config supplies it (so --auto-roi uses all optionals)
    if (args.auto_roi or args.clock_roi) and cfg.ocr is None:
        cfg.ocr = OcrConfig(auto_detect=bool(args.auto_roi), clock_roi=None)
        if args.clock_roi:
            x, y, w, h = [int(x.strip()) for x in args.clock_roi.split(",")]
            cfg.ocr.clock_roi = Roi(x=x, y=y, w=w, h=h)

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    elif args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (input_path.stem + ".condensed." + cfg.render.container)
    else:
        out_path = input_path.with_name(input_path.stem + ".condensed." + cfg.render.container)

    workdir = Path(args.workdir).expanduser().resolve() if args.workdir else None
    if workdir:
        workdir.mkdir(parents=True, exist_ok=True)
        tmpctx = tempfile.TemporaryDirectory(dir=str(workdir))
    else:
        tmpctx = tempfile.TemporaryDirectory()

    debug_dir = Path(args.debug_dir).expanduser().resolve() if args.debug_dir else None

    with tmpctx as td:
        tmpdir = Path(td)
        duration = ffprobe_duration(input_path)
        video_fps = ffprobe_video_fps(input_path)
        eprint(f"Duration: {duration:.2f}s, Video FPS: {video_fps:.3f}")

        if args.min_duration and duration < args.min_duration:
            eprint(f"Duration {duration:.0f}s < --min-duration {args.min_duration:.0f}s; skipping (exit 0).")
            return 0

        if args.calibrate:
            preview = tmpdir / "preview.png"
            make_preview_frame(input_path, preview, t=min(300.0, duration * 0.2))
            eprint(f"Wrote preview frame: {preview}")
            eprint("ROI helper:")
            eprint("  Use an image viewer to get pixel coords for the clock area.")
            eprint("  Then set in config or pass --clock-roi x,y,w,h")
            eprint("ffplay crop helper:")
            eprint(f"  ffplay -vf crop=w:h:x:y {input_path}")
            return 0

        dt = cfg.combine.dt
        n = int(math.ceil(duration / dt))

        # 1) Video motion + 2) Audio extraction in parallel (independent; not real-time, decode as fast as disk/CPU allow)
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_frames = ex.submit(
                extract_gray_frames_raw,
                input_path,
                fps=max(1.0, min(6.0, 1.0 / dt)),
                width=160,
                tmpdir=tmpdir,
            )
            fut_audio = ex.submit(extract_audio_pcm_s16le, input_path, tmpdir, sample_rate=16000)
            frames_raw, fw, fh, mfps = fut_frames.result()
            audio_raw = fut_audio.result()

        motion, motion_dt = compute_motion_energy(frames_raw, w=fw, h=fh, fps=mfps)
        motion_r = resample_to_dt(motion.astype(np.float32), motion_dt, dt, duration)

        # Audio intelligence signals (audio_raw already extracted above)
        # compute at a finer hop, then aggregate to dt
        audio_hop = min(0.10, dt)
        feats, feats_dt = compute_audio_features(audio_raw, sample_rate=16000, hop_seconds=audio_hop)
        rms_r = resample_to_dt(feats["rms"].astype(np.float32), feats_dt, dt, duration, mode="mean")
        flux_r = resample_to_dt(feats["flux"].astype(np.float32), feats_dt, dt, duration, mode="mean")
        whistle_r = resample_to_dt(feats["whistle"].astype(np.float32), feats_dt, dt, duration, mode="max")

        # Normalize all
        m_z = smooth_ema(robust_z(motion_r), alpha=0.25)
        rms_z = smooth_ema(robust_z(rms_r), alpha=0.25)
        flux_z = smooth_ema(robust_z(flux_r), alpha=0.25)
        whistle_z = smooth_ema(robust_z(whistle_r), alpha=0.25)

        aw = cfg.audio_weights
        action_z = aw.w_motion * m_z + aw.w_rms * rms_z + aw.w_flux * flux_z + aw.w_whistle * whistle_z
        action_z = smooth_ema(action_z.astype(np.float32), alpha=0.20)

        hard_event_mask = (action_z >= cfg.combine.hard_peak_z).astype(np.uint8)
        hard_event_mask = np.maximum(hard_event_mask, (whistle_z >= aw.hard_whistle_z).astype(np.uint8))

        # 3) OCR clock running signal (optional)
        clock_mask_dt = None
        clock_conf = 0.0
        used_roi = None

        if cfg.ocr is None:
            eprint("OCR disabled (no config with 'ocr' section); using action-only weighting.")

        if cfg.ocr is not None:
            ocr_cfg = dataclasses.replace(cfg.ocr)

            if args.ocr_fps is not None:
                ocr_cfg.fps = float(args.ocr_fps)

            if args.clock_roi:
                x, y, w, h = [int(x.strip()) for x in args.clock_roi.split(",")]
                ocr_cfg.clock_roi = Roi(x=x, y=y, w=w, h=h)

            if args.auto_roi:
                ocr_cfg.auto_detect = True

            if ocr_cfg.auto_detect and (ocr_cfg.clock_roi is None):
                if cv2 is None:
                    msg = "opencv-python not installed; cannot auto-detect ROI."
                    if cfg.combine.require_ocr:
                        raise ToolError(msg)
                    eprint("WARN:", msg)
                else:
                    eprint("Auto-detecting scorebug ROI...")
                    used_roi = autodetect_scorebug_roi(
                        input_path,
                        tmpdir,
                        sample_fps=1.0,
                        max_frames=160,
                        scale_w=640,
                        ocr_cfg=ocr_cfg,
                        debug_dir=debug_dir
                    )
                    if used_roi is None:
                        msg = "Auto ROI detection failed to find a confident clock region."
                        if cfg.combine.require_ocr:
                            raise ToolError(msg)
                        eprint("WARN:", msg)
                    else:
                        ocr_cfg.clock_roi = used_roi
                        eprint(f"Auto ROI: x={used_roi.x} y={used_roi.y} w={used_roi.w} h={used_roi.h}")

            if ocr_cfg.clock_roi is not None:
                if pytesseract is None:
                    msg = "pytesseract not installed; OCR unavailable."
                    if cfg.combine.require_ocr:
                        raise ToolError(msg)
                    eprint("WARN:", msg)
                else:
                    roi_frames = extract_roi_frames(
                        input_path,
                        tmpdir,
                        ocr_cfg.clock_roi,
                        fps=ocr_cfg.fps,
                        scale=ocr_cfg.scale,
                        invert=ocr_cfg.invert,
                        threshold=ocr_cfg.threshold
                    )
                    times: List[Optional[int]] = []
                    for p in roi_frames:
                        if cv2 is None:
                            break
                        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            times.append(None)
                            continue
                        s = ocr_one_image(img, ocr_cfg)
                        t = parse_time_str(s) if s else None
                        times.append(t)

                    run_prob, clock_conf = infer_clock_running(times, ocr_cfg.fps, duration)
                    run_1hz = mask_from_running_prob(run_prob, ocr_cfg.fps, duration)
                    clock_mask_dt = resample_to_dt(run_1hz.astype(np.float32), 1.0, dt, duration, mode="nearest")

                    usable = float(np.mean(clock_mask_dt > 0.5)) if len(clock_mask_dt) else 0.0
                    eprint(f"OCR clock confidence ~ {clock_conf:.3f}, running fraction ~ {usable:.3f}")
                    if (clock_conf < 0.10 or usable < 0.01) and cfg.combine.require_ocr:
                        raise ToolError("OCR produced no usable running-clock signal; check ROI or enable auto ROI and debug output.")

        # 4) Build cutlists and render selected mode
        cutlists = pick_segments(action_z, hard_event_mask, clock_mask_dt, clock_conf, cfg.combine, duration)

        # Select render list
        mode = cfg.combine.render_mode.strip().lower()
        if mode not in ("fused", "action", "clock"):
            mode = "fused"
        if mode == "clock" and not cutlists["clock"]:
            eprint("WARN: clock mode requested but no clock cutlist available; falling back to fused.")
            mode = "fused"

        segs = cutlists[mode]

        eprint(f"Cutlist lengths (sec): action={total_len(cutlists['action']):.1f}, clock={total_len(cutlists['clock']):.1f}, fused={total_len(cutlists['fused']):.1f}")
        eprint(f"Rendering mode: {mode}, segments={len(segs)}, total={total_len(segs):.1f}s target={cfg.combine.target_minutes*60:.1f}s")

        # Write EDLs
        edl_dir = tmpdir / "edl"
        edl_dir.mkdir(parents=True, exist_ok=True)
        write_edl(cutlists["action"], edl_dir / "cutlist.action.edl")
        if cutlists["clock"]:
            write_edl(cutlists["clock"], edl_dir / "cutlist.clock.edl")
        write_edl(cutlists["fused"], edl_dir / "cutlist.fused.edl")
        eprint(f"Wrote EDLs: {edl_dir}")
        # Persist fused EDL next to output for validation (validate_condensed.py)
        fused_edl_path = out_path.parent / (out_path.stem + ".fused.edl")
        write_edl(segs, fused_edl_path)
        eprint(f"Wrote fused EDL for validation: {fused_edl_path}")

        # Diagnostics (includes validation stats for programmatic quality checks)
        if args.emit_diagnostics:
            validation_stats = compute_cut_quality_stats(
                action_z, hard_event_mask, dt, duration, segs
            )
            diag = {
                "duration_seconds": duration,
                "dt": dt,
                "target_minutes": cfg.combine.target_minutes,
                "clock_confidence": clock_conf,
                "clock_present": bool(clock_mask_dt is not None),
                "auto_roi_used": bool(used_roi is not None),
                "auto_roi": dataclasses.asdict(used_roi) if used_roi is not None else None,
                "segments": {
                    "action": len(cutlists["action"]),
                    "clock": len(cutlists["clock"]),
                    "fused": len(cutlists["fused"]),
                },
                "length_seconds": {
                    "action": total_len(cutlists["action"]),
                    "clock": total_len(cutlists["clock"]),
                    "fused": total_len(cutlists["fused"]),
                },
                "render_mode": mode,
                "validation": {
                    **validation_stats,
                    "edl_segments": [[round(a, 3), round(b, 3)] for a, b in segs],
                },
            }
            Path(args.emit_diagnostics).expanduser().resolve().write_text(json.dumps(diag, indent=2) + "\n", encoding="utf-8")

        # Render
        render_concat_segments(input_path, segs, out_path, cfg.render, tmpdir)
        eprint(f"Output: {out_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ToolError as ex:
        eprint("ERROR:", ex)
        raise SystemExit(1)
