# Condensearr

**Condensearr** turns long sports (or other) recordings into shorter “condensed” cuts by keeping high-action segments and dropping low-action stretches. It uses video motion, audio (loudness, onsets, whistle-like peaks), and optional scorebug OCR to build a single fused cutlist and encode one output file.

- **Local only** — processes files you already have; no streaming, no DRM bypass.
- **CLI-first** — scriptable, non-interactive, exit codes for automation.
- **Validation** — optional programmatic checks that the encode is correct and that kept segments are higher-action than cut regions.

## Requirements

- **ffmpeg** and **ffprobe** on `PATH`
- **Python 3.9+**
- **numpy** (required)
- Optional: **pytesseract** + system **tesseract-ocr** (for clock OCR)
- Optional: **opencv-python** (for auto scorebug detection)

## Install

```bash
pip install numpy
# Optional for OCR / auto-ROI:
pip install pytesseract opencv-python
```

## Quick start

```bash
# Basic: output next to input as <stem>.condensed.mkv
python condensearr.py /path/to/recording.mkv

# Explicit output and target length
python condensearr.py /path/to/recording.mkv --out /path/to/condensed.mkv --target-minutes 18

# With config and validation artifacts
python condensearr.py /path/to/recording.mkv --config config.json --out condensed.mkv \
  --emit-diagnostics diagnostics.json
python validate_condensed.py --output condensed.mkv --edl condensed.fused.edl --diagnostics diagnostics.json

# All optionals (OCR/clock, diagnostics, opencv + pytesseract required)
python condensearr.py /path/to/recording.mkv --config condensearr_config.full.json \
  --out condensed.mkv --emit-diagnostics diagnostics.json
```

## Options (summary)

| Option | Description |
|--------|-------------|
| `input` | Input video path (positional). |
| `--out` | Output file path. |
| `--out-dir DIR` | Write output as `DIR/<stem>.condensed.mkv` (for Tdarr/Arr). |
| `--config PATH` | JSON config; default from `CONDENSEARR_CONFIG` env if set. |
| `--target-minutes N` | Target condensed length in minutes. |
| `--min-duration SEC` | Skip if source duration < SEC (exit 0). For automation filters. |
| `--jobs N` | Parallel encode jobs (0 = auto; “nice” mode caps at 2). |
| `--no-nice` | Don’t use nice/ionice; use full CPU. |
| `--emit-diagnostics PATH` | Write JSON with validation stats and EDL. |
| `--auto-roi` | Enable OCR with auto scorebug detection (no config needed if ocr not in config). |
| `--clock-roi x,y,w,h` | Use this region for clock OCR (overrides config). |
| `--debug-dir DIR` | Write auto-ROI debug images to DIR. |
| `--kill-prior` | Kill other running condensearr processes before starting. |

Default behavior runs with low CPU priority (nice 19) and idle I/O (ionice -c 3) and caps parallel encode jobs so the system stays responsive.

## Validation

**validate_condensed.py** checks:

1. **Structural** — Output duration matches the EDL (within tolerance); file has video and audio.
2. **Content** (with `--diagnostics`) — Kept segments have higher action than cut; most hard events (whistles/peaks) fall inside kept segments; optional max-gap check.

Condensearr writes `<output>.fused.edl` beside the output so you can validate without keeping the temp dir. See [DESIGN.md](docs/DESIGN.md#validation) for the logic.

## All optional / advanced features

To turn on **everything** (OCR clock weighting, auto scorebug detection, diagnostics):

1. Install optional deps: `pip install pytesseract opencv-python` (and a Tesseract binary).
2. Run with the full config: `--config condensearr_config.full.json --emit-diagnostics diagnostics.json`.

If the log says *"Auto ROI detection failed to find a confident clock region"*, OCR is enabled but no clock region passed the threshold. You can:

- Lower `ocr.auto_detect.min_hit_rate` in the config (e.g. to `0.05`), or  
- Use `--debug-dir /tmp/condensearr_debug` to inspect candidate regions, then set `--clock-roi x,y,w,h` or add `clock_roi` to the config.

## Automation (Tdarr / *Arr)

- Use `--out-dir`, `CONDENSEARR_CONFIG`, and `--min-duration` for unattended runs.
- See **[docs/ARR_INTEGRATION.md](docs/ARR_INTEGRATION.md)** for Tdarr Run CLI and Sonarr/Radarr custom script examples.

## Testing

Install all dependencies (including optional) and run tests:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt pytesseract opencv-python pytest
.venv/bin/pytest tests/ -v
```

Or with unittest (no pytest): `python3 -m unittest discover -s tests -v`

## Docs

- **[docs/DESIGN.md](docs/DESIGN.md)** — Design and signal logic (motion, audio, OCR, fusion, validation).
- **[docs/ARR_INTEGRATION.md](docs/ARR_INTEGRATION.md)** — Tdarr and *Arr integration.
- **[docs/LEGAL.md](docs/LEGAL.md)** — Legal audit: why MIT, dependency compatibility, sharp corners.

## License

**MIT License.** Copyright (c) 2025 snapetech. See [LICENSE](LICENSE) for full terms.
