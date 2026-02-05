# Method matrix reference results

This document records **reference results** from running the [method matrix](METHOD_MATRIX.md) so you know what to compare against when re-running or tuning.

## NBA clip (10 min) — LAL @ NYK, 2026-02-01

**Source:** 10-minute clip from `NBA_20260201_LAL @ NYK_1080p60_NBC.mkv` (full game ~106 min).  
**Target length:** 18 min (config default); clip is 10 min so kept length is below target.  
**Run:** Analysis-only (no encode). Condensearr default weights + `condensearr_config.full.json` base.

### Results table

| Method | Segments | Length (s) | Action ratio (kept vs cut) | Hard events in kept |
|--------|----------|------------|----------------------------|----------------------|
| motion | 10 | 369.1 | -1.86 | **100%** (117/117) |
| rms | 11 | 392.1 | -0.18 | 5.3% (1/19) |
| flux | 11 | 396.5 | -0.29 | **0%** (0/19) |
| whistle | 17 | 421.0 | -2.07 | **100%** (33/33) |
| motion_rms | 9 | 370.6 | -1.10 | **100%** (42/42) |
| motion_flux | 11 | 378.1 | -1.49 | 96.7% (29/30) |
| motion_whistle | 14 | 387.5 | -4.10 | **100%** (30/30) |
| rms_flux | 10 | 408.6 | -0.17 | 5.3% (1/19) |
| rms_whistle | 12 | 398.5 | -0.93 | **100%** (12/12) |
| flux_whistle | 13 | 400.5 | -1.02 | **100%** (13/13) |
| motion_rms_flux | 10 | 386.1 | -0.78 | 73.7% (14/19) |
| motion_rms_whistle | 10 | 384.1 | -1.95 | **100%** (10/10) |
| motion_flux_whistle | 13 | 410.0 | -2.01 | **100%** (13/13) |
| rms_flux_whistle | 13 | 408.5 | -0.57 | **100%** (13/13) |
| **all_action** | **8** | **370.6** | -1.20 | **100%** (8/8) |
| **clock** | **8** | **370.6** | -1.20 | **100%** |
| **fused** | **8** | **370.6** | -1.20 | **100%** |

### What to reference

- **Fused / all_action / clock** should give **8 segments**, **~370.6 s** kept, and **100%** of hard events in kept. Use this as the baseline when re-running the matrix on the same or similar input.
- Single-signal **rms** and **flux** show low hard-events-in-kept (0–5%); that is expected and confirms fused is the right default.
- Action ratios can be negative on short clips when target length is close to or above source length; for full-length games expect ratio ≥ 1.2 when configured correctly.

### How to reproduce

```bash
# Create a 10-min clip from your copy of the game (optional)
ffmpeg -y -i "NBA_20260201_LAL @ NYK_1080p60_NBC.mkv" -t 600 -c copy nba_clip_10min.mkv

# Run matrix (analysis-only)
python3 run_matrix.py nba_clip_10min.mkv matrix_nba
```

Then compare `matrix_nba/comparison.csv` to the table above.
