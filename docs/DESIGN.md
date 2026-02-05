# Condensearr design and logic

**MIT License.** Copyright (c) 2025 snapetech. See [LICENSE](../LICENSE) in the repo root.

This document describes how condensearr chooses what to keep and what to cut, and how validation works.

## Overview

The pipeline:

1. **Probe** — Get source duration and video FPS.
2. **Extract** — In parallel: low-res grayscale frames (for motion) and mono PCM audio (for loudness/onsets/whistle).
3. **Signals** — Compute motion energy, RMS, spectral flux, whistle-like score; optionally OCR scorebug for “clock running” mask.
4. **Fusion** — Build action score; detect hard events (whistles + peaks); optionally weight by clock; pick segments to hit target length.
5. **Render** — Encode segments (ffmpeg), concat, write output and fused EDL.

No real-time throttling; everything runs as fast as disk/CPU allow.

## Condensing model

**Default: variable-length (remove most optional removals).** **Optional: target length.**

- **No target (default):** `target_minutes` is 0 or omitted. We keep segments where action is above a configurable quantile (default **median**, `no_target_quantile: 0.5`). We remove the most obviously removable (low-action) content; output length is whatever that gives.
- **Target length (option):** When `target_minutes > 0` (config or `--target-minutes N`), we condense **to** that many minutes via binary search on the action quantile.
- **Example (target):** 106-minute game with `--target-minutes 18` → ~18-minute cut.
- **Example (no target):** Same game with default config → above-median action (and hard events); output length varies (e.g. ~45–60 min).
- **When source is shorter than target:** We get as close as possible; prefer a target ≤ source length for short clips.

## Signals

### Video: motion energy

- Frames at low FPS (e.g. 2 fps), scaled to 160px width, grayscale.
- Per-frame “motion” = mean absolute difference from previous frame.
- Resampled to a fixed time grid (e.g. 0.5 s) and robust-normalized (median/MAD) then smoothed (EMA).
- Result: `motion_z` — higher where there’s more change (plays, cuts, camera moves).

### Audio: RMS, flux, whistle

- Mono 16 kHz PCM.
- **RMS** — short-window loudness.
- **Spectral flux** — L1 norm of positive changes in magnitude spectrum (onsets).
- **Whistle** — energy in a high band (e.g. 2.5–4.5 kHz) relative to total, times peakiness (whistle-like transients).
- All resampled to the same grid and robust-normalized, then combined with configurable weights into an “audio” contribution to action.

### Combined action score

- `action_z = w_motion * motion_z + w_rms * rms_z + w_flux * flux_z + w_whistle * whistle_z` (default weights in config).
- Smoothed again. Used to decide which half-second bins are “action” vs “dead”.

### Hard events

- **Peaks** — Bins where `action_z` exceeds a threshold (e.g. 2.0 in z-space).
- **Whistles** — Bins where `whistle_z` exceeds a higher threshold (e.g. 2.6).
- Union of these is the “hard event” mask: segments containing these are preferred in the final cutlist.

### Optional: clock (OCR)

- If OCR is enabled, extract a scorebug ROI (manual or auto-detected), sample frames at low FPS, run tesseract for time-like text.
- Infer “clock running” probability over time; resample to the same grid.
- Used as a **weight** on the action score (Strategy C / fused): when the clock is “running”, action is weighted more; when it’s not (commercial, timeout), action is down-weighted so we don’t keep as much of that.

## Segment selection

- **Target length** — Optional. If `target_minutes > 0`, we aim for that many minutes of output. If 0 or omitted (default), we use a single quantile (default median) so we keep “above median action” — variable output length.
- **Strategy A (action-only):** If target set: binary search on quantile to hit target length. If no target: keep bins with action above `no_target_quantile` (default 0.5). No clock.
- **Strategy B (clock-only):** Segments where “clock running” is high. If target set and clock segments exceed it, trim to target by keeping highest-action subsegments. If no target, keep all clock-running segments.
- **Strategy C (fused, default):** Hard events are always kept. If target set: fill remaining length by binary search on clock-weighted action. If no target: add segments from clock-weighted action above `no_target_quantile`. Then pad (pre/post) and merge.

Result: one ordered list of `(start_sec, end_sec)` segments — the “fused” cutlist.

## Render

- Each segment is encoded with ffmpeg (libx264, AAC) into a part file.
- Parts are concatenated (stream copy) into the final file.
- The fused EDL is written next to the output as `<output>.fused.edl` for validation and auditing.

## Validation

**Structural (validate_condensed.py, always):**

- Output duration is within 1% or 10 s of the sum of EDL segment lengths (allows for encoding/keyframe drift).
- Output has at least one video and one audio stream.

**Content (with diagnostics from condensearr --emit-diagnostics):**

- **Action ratio** — Mean action in kept segments vs mean action in cut regions. We require `kept_mean / cut_mean >= 1.2` (or 1.5 in `--strict`) so the cut is justified by the same signal we used to build it.
- **Hard events in kept** — Fraction of hard-event bins that fall inside kept segments. We require ≥ 90% (or 95% in `--strict`) so we’re not dropping whistles/peaks.
- **Max gap** — Largest gap between two consecutive kept segments. In `--strict`, we fail if this exceeds 15 minutes (catches “dropped a whole quarter” mistakes).

Diagnostics are produced during the run (action_z and hard_event_mask are already in memory), so no second pass is needed to validate content.

## Performance and “nice” mode

- **Analysis** — Audio and motion extraction run in parallel (two threads); ffmpeg calls use `-threads 0` for multi-core decode.
- **Encode** — By default we run with `nice -n 19` and `ionice -c 3` (Linux) and cap at 2 parallel segment jobs and 2 threads per ffmpeg so the machine stays responsive. Use `--no-nice` and `--jobs N` for maximum speed when the box is dedicated.

## File layout

- **condensearr.py** — Main script; CLI, pipeline, and EDL/diagnostics output.
- **validate_condensed.py** — Standalone validator (structural + optional content from diagnostics).
- **docs/DESIGN.md** — This file.
- **docs/ARR_INTEGRATION.md** — Tdarr and *Arr integration.
