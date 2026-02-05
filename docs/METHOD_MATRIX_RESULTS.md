# Method matrix reference results

This document records **reference results** from running the [method matrix](METHOD_MATRIX.md) so you know what to compare against when re-running or tuning.

## NBA clip (10 min) — LAL @ NYK, 2026-02-01 (variable-length)

**Source:** 10-minute clip from `NBA_20260201_LAL @ NYK_1080p60_NBC.mkv` (full game ~106 min).  
**Mode:** **Variable-length** (no target). Default config: `target_minutes: 0`, `no_target_quantile: 0.5` — we keep segments with action above the median and remove the most obviously removable (low-action) content; output length is whatever that yields.  
**Run:** Analysis-only (no encode). Condensearr default weights + `condensearr_config.full.json` base.

### Results table

| Method | Segments | Length (s) | Action ratio (kept vs cut) | Hard events in kept |
|--------|----------|------------|----------------------------|----------------------|
| motion | 10 | 369.6 | -1.85 | **100%** (117/117) |
| rms | 11 | 392.1 | -0.18 | 5.3% (1/19) |
| flux | 11 | 397.0 | -0.29 | **0%** (0/19) |
| whistle | 17 | 421.5 | -2.07 | **100%** (33/33) |
| motion_rms | 9 | 370.6 | -1.10 | **100%** (42/42) |
| motion_flux | 12 | 387.1 | -1.39 | 96.7% (29/30) |
| motion_whistle | 14 | 387.5 | -4.10 | **100%** (30/30) |
| rms_flux | 10 | 408.6 | -0.17 | 5.3% (1/19) |
| rms_whistle | 12 | 399.0 | -0.93 | **100%** (12/12) |
| flux_whistle | 13 | 400.5 | -1.02 | **100%** (13/13) |
| motion_rms_flux | 10 | 386.1 | -0.78 | 73.7% (14/19) |
| motion_rms_whistle | 10 | 384.6 | -1.94 | **100%** (10/10) |
| motion_flux_whistle | 13 | 410.5 | -2.00 | **100%** (13/13) |
| rms_flux_whistle | 13 | 409.0 | -0.57 | **100%** (13/13) |
| **all_action** | **8** | **370.6** | -1.20 | **100%** |
| **clock** | **8** | **370.6** | -1.20 | **100%** |
| **fused** | **8** | **370.6** | -1.20 | **100%** |

### What to reference

- **Fused / all_action / clock** should give **8 segments**, **~370.6 s** kept, and **100%** of hard events in kept. Use this as the baseline when re-running the matrix on the same or similar input with variable-length (no target).
- Single-signal **rms** and **flux** show low hard-events-in-kept (0–5%); that is expected and confirms fused is the right default.
- Action ratios can be negative on this clip; for full-length games with variable-length or with a target, expect ratio ≥ 1.2 when the cut is good.

### How to reproduce

```bash
# Create a 10-min clip from your copy of the game (optional)
ffmpeg -y -i "NBA_20260201_LAL @ NYK_1080p60_NBC.mkv" -t 600 -c copy nba_clip_10min.mkv

# Run matrix (analysis-only) with variable-length default (no target)
python3 run_matrix.py nba_clip_10min.mkv matrix_nba

# Optional: run with a target length (e.g. 5 min) for fixed-length comparison
python3 run_matrix.py nba_clip_10min.mkv matrix_nba --target-minutes 5
```

Then compare `matrix_nba/comparison.csv` to the table above.
