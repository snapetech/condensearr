# Method matrix reference results

This document records **reference results** from running the [method matrix](METHOD_MATRIX.md) so you know what to compare against when re-running or tuning.

## NBA clip (10 min) — LAL @ NYK, 2026-02-01 (variable-length) — **VLM enabled**

**Source:** 10-minute clip from `NBA_20260201_LAL @ NYK_1080p60_NBC.mkv` (full game ~106 min).  
**Mode:** **Variable-length** (no target). Default config: `target_minutes: 0`, `no_target_quantile: 0.5` — we keep segments with action above the median and remove the most obviously removable (low-action) content; output length is whatever that yields.  
**Run:** Analysis-only (no encode). Condensearr default weights + `condensearr_config.full.json` base. **VLM enabled** (`vlm.enabled: true`, `vlm.backend: "llama_cpp"`, `vlm.model_path: "vikhyatk/moondream2"`) — on this real sports clip the VLM can pick the scorebug/clock ROI so OCR locks onto it.

### Results table

| Method | Segments | Length (s) | Action ratio (kept vs cut) | Hard events in kept |
|--------|----------|------------|----------------------------|----------------------|
| motion | 10 | 369.6 | -1.85 | **100%** |
| rms | 11 | 392.1 | -0.18 | 5.3% |
| flux | 11 | 397.0 | -0.29 | 0% |
| whistle | 17 | 421.5 | -2.07 | **100%** |
| motion_rms | 9 | 370.6 | -1.10 | **100%** |
| motion_flux | 12 | 387.1 | -1.39 | 96.7% |
| motion_whistle | 14 | 387.5 | -4.10 | **100%** |
| rms_flux | 10 | 408.6 | -0.17 | 5.3% |
| rms_whistle | 12 | 399.0 | -0.93 | **100%** |
| flux_whistle | 13 | 400.5 | -1.02 | **100%** |
| motion_rms_flux | 10 | 386.1 | -0.78 | 73.7% |
| motion_rms_whistle | 10 | 384.6 | -1.94 | **100%** |
| motion_flux_whistle | 13 | 410.5 | -2.00 | **100%** |
| rms_flux_whistle | 13 | 409.0 | -0.57 | **100%** |
| **all_action** | **8** | **370.6** | -1.20 | **100%** |
| **clock** | **8** | **370.6** | -1.20 | **100%** |
| **fused** | **8** | **370.6** | -1.20 | **100%** |

### What to reference

- **Fused / all_action / clock** give **8 segments**, **~370.6 s** kept, and **100%** of hard events in kept. Use this as the baseline when re-running the matrix on the same or similar input with variable-length (no target).
- Single-signal **rms** and **flux** show low hard-events-in-kept (0–5%); that is expected and confirms fused is the right default.
- Action ratios can be negative on this clip; for full-length games with variable-length or with a target, expect ratio ≥ 1.2 when the cut is good.

### How to reproduce

```bash
# Create a 10-min clip from your copy of the game (optional)
ffmpeg -y -i "NBA_20260201_LAL @ NYK_1080p60_NBC.mkv" -t 600 -c copy nba_clip_10min.mkv

# Run matrix (analysis-only) on the NBA clip with VLM enabled (condensearr_config.full.json)
.venv/bin/python3 run_matrix.py matrix_nba/nba_clip_10min.mkv matrix_nba

# Optional: run with a target length (e.g. 5 min) for fixed-length comparison
.venv/bin/python3 run_matrix.py matrix_nba/nba_clip_10min.mkv matrix_nba --target-minutes 5
```

Then compare `matrix_nba/comparison.csv` to the table above.

---

## Fixture run with VLM enabled (method matrix)

**Source:** Generated 120 s fixture (`run_matrix.py matrix_out --gen-fixture`).  
**Config:** `condensearr_config.full.json` with **VLM enabled** (`vlm.enabled: true`, `vlm.backend: "llama_cpp"`, `vlm.model_path: "vikhyatk/moondream2"`). Backend: llama-cpp-python with `MoondreamChatHandler` (vision); model loaded from HuggingFace when needed for clock ROI calibration.  
**Run:** Analysis-only. All 17 combos completed successfully.

**Note:** The fixture is synthetic (smptebars); it has no real scorebug. The VLM runs (asks “which candidate is the clock?”) but never gets a “yes,” so the pipeline falls back to OCR-based ROI selection—same outcome as with VLM off. To see the VLM **help**, run the matrix (or a single run) on **real sports footage** with a visible clock/scorebug; then the VLM can pick the correct ROI and lock OCR onto it.

| Method       | Segments | Length (s) | Action ratio (kept vs cut) | Hard events in kept |
|-------------|----------|------------|----------------------------|----------------------|
| motion      | 1        | 120.0      | 1.0                        | 100% (7/7)           |
| rms         | 3        | 89.5       | -0.41                      | 74% (20/27)          |
| flux        | 2        | 71.5       | -73.1                      | 100% (52/52)         |
| whistle     | 5        | 91.5       | -4.08                      | 100% (14/14)         |
| motion_rms  | 3        | 89.5       | -0.41                      | 72% (18/25)          |
| motion_flux | 2        | 71.5       | -73.1                      | 100% (49/49)         |
| motion_whistle | 5     | 91.5       | -4.08                      | 100% (14/14)         |
| rms_flux    | 4        | 90.0       | -3.98                      | 100%                 |
| rms_whistle | 4        | 96.0       | -0.70                      | 72%                  |
| flux_whistle | 2       | 78.5       | -47.8                      | 100%                 |
| motion_rms_flux | 4    | 90.0       | -3.99                      | 100%                 |
| motion_rms_whistle | 4  | 96.0       | -0.70                      | 70%                  |
| motion_flux_whistle | 2 | 78.5       | -47.7                      | 100%                 |
| rms_flux_whistle | 3  | 78.5       | -6.64                      | 100%                 |
| all_action  | 4        | 89.5       | -48.0                      | 100%                 |
| clock       | 4        | 89.5       | -48.0                      | 100%                 |
| fused       | 4        | 89.5       | -48.0                      | 100%                 |

### How to reproduce (VLM)

```bash
cd /path/to/condensearr
python3 -m venv .venv
.venv/bin/pip install numpy opencv-python llama-cpp-python huggingface-hub
# Ensure condensearr_config.full.json has vlm.enabled: true, vlm.backend: "llama_cpp", vlm.model_path: "vikhyatk/moondream2"
.venv/bin/python3 run_matrix.py matrix_out --gen-fixture
```

Then compare `matrix_out/comparison.csv` to the table above.
