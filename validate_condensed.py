#!/usr/bin/env python3
"""
validate_condensed.py — Programmatic validation of a condensed cut.

Proves:
  1. Structural: output duration matches EDL; has video+audio; no missing segments.
  2. Content: kept segments have higher action than cut; hard events (whistles/peaks)
     are mostly inside kept segments; no unreasonably large gaps.

Usage:
  python validate_condensed.py --output condensed.mkv --edl cutlist.fused.edl [--diagnostics diag.json]
  python validate_condensed.py --output condensed.mkv --edl cutlist.fused.edl --diagnostics diag.json --strict

With --diagnostics (from condensearr --emit-diagnostics): runs content-quality checks.
With --strict: fail on warnings (e.g. max_gap too large, ratio below 1.5).
Exit: 0 = all checks passed, 1 = validation failed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def probe_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
    return float(r.stdout.strip())


def probe_streams(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=codec_type,codec_name",
        "-of", "json",
        str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
    return json.loads(r.stdout)


def load_edl(path: Path) -> list[tuple[float, float]]:
    segs = []
    for line in path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            segs.append((float(parts[0]), float(parts[1])))
    return segs


def edl_total_seconds(segs: list[tuple[float, float]]) -> float:
    return sum(max(0.0, b - a) for a, b in segs)


def run_structural_checks(output: Path, edl: Path) -> list[str]:
    errors = []
    segs = load_edl(edl)
    expected_duration = edl_total_seconds(segs)
    if not output.exists():
        errors.append(f"Output file does not exist: {output}")
        return errors
    try:
        actual_duration = probe_duration(output)
    except Exception as e:
        errors.append(f"Failed to probe output duration: {e}")
        return errors
    # Allow 1% or 10s tolerance (encoding/keyframe boundaries can shift slightly)
    tol = max(10.0, expected_duration * 0.01)
    if abs(actual_duration - expected_duration) > tol:
        errors.append(
            f"Duration mismatch: output={actual_duration:.2f}s, EDL sum={expected_duration:.2f}s (delta={actual_duration - expected_duration:.2f}s, tolerance={tol:.1f}s)"
        )
    try:
        streams = probe_streams(output)
        types = [s.get("codec_type") for s in streams.get("streams", [])]
        if "video" not in types:
            errors.append("Output has no video stream")
        if "audio" not in types:
            errors.append("Output has no audio stream")
    except Exception as e:
        errors.append(f"Failed to probe streams: {e}")
    return errors


def run_content_checks(diagnostics: Path, strict: bool) -> list[str]:
    errors = []
    warnings = []
    data = json.loads(diagnostics.read_text())
    val = data.get("validation") or {}
    if not val:
        warnings.append("Diagnostics has no 'validation' block (run condensearr with --emit-diagnostics)")
        return errors if not strict else warnings

    # Kept segments should have higher action than cut
    ratio = val.get("action_ratio_kept_vs_cut", 1.0)
    min_ratio = 1.5 if strict else 1.2
    if ratio < min_ratio:
        errors.append(
            f"Action ratio (kept/cut) = {ratio:.3f} (expected >= {min_ratio}); kept segments are not clearly higher-action than cut"
        )
    else:
        eprint(f"  action ratio kept/cut = {ratio:.3f} (ok)")

    # Hard events (whistles, peaks) should mostly be in kept segments
    frac = val.get("fraction_hard_in_kept", 1.0)
    hard_total = val.get("hard_events_total", 0)
    hard_in_kept = val.get("hard_events_in_kept", 0)
    min_frac = 0.95 if strict else 0.90
    if hard_total > 0 and frac < min_frac:
        errors.append(
            f"Only {frac:.1%} of hard events ({hard_in_kept}/{hard_total}) are in kept segments (expected >= {min_frac:.0%})"
        )
    else:
        eprint(f"  hard events in kept: {hard_in_kept}/{hard_total} ({frac:.1%}) (ok)")

    # No single gap larger than ~15 min (watchability: we didn't drop a whole quarter)
    max_gap = val.get("max_gap_seconds", 0)
    gap_limit = 900  # 15 min
    if max_gap > gap_limit:
        msg = f"Largest gap between kept segments = {max_gap/60:.1f} min (>{gap_limit/60:.0f} min)"
        if strict:
            errors.append(msg)
        else:
            warnings.append(msg)
    else:
        eprint(f"  max gap between segments = {max_gap/60:.1f} min (ok)")

    if strict and warnings:
        errors.extend(warnings)
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a condensed cut (structural + optional content quality).")
    ap.add_argument("--output", "-o", required=True, type=Path, help="Condensed output video (e.g. condensed.mkv)")
    ap.add_argument("--edl", "-e", required=True, type=Path, help="EDL used for the cut (e.g. cutlist.fused.edl)")
    ap.add_argument("--diagnostics", "-d", type=Path, default=None, help="JSON from condensearr --emit-diagnostics (enables content checks)")
    ap.add_argument("--strict", action="store_true", help="Treat warnings as failures; require higher ratio and fraction")
    args = ap.parse_args()

    errors = []

    eprint("Structural checks (output vs EDL)...")
    errors.extend(run_structural_checks(args.output, args.edl))

    if args.diagnostics and args.diagnostics.exists():
        eprint("Content-quality checks (from diagnostics)...")
        errors.extend(run_content_checks(args.diagnostics, args.strict))
    elif args.diagnostics:
        errors.append(f"Diagnostics file not found: {args.diagnostics}")

    if errors:
        eprint("\nValidation FAILED:")
        for e in errors:
            eprint("  -", e)
        return 1
    eprint("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
