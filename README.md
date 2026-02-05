# Condensearr

**Condensearr** turns long sports (and similar) recordings into shorter condensed cuts by keeping high-action segments and dropping low-action stretches. It fuses video motion, audio (loudness, onsets, whistle-like peaks), and optional scorebug OCR into one cutlist and encodes a single output file.

- **Local only** — processes files you already have; no streaming, no DRM bypass.
- **CLI-first** — scriptable, non-interactive, exit codes for automation.
- **Validation** — optional checks that the encode matches the EDL and that kept segments are higher-action than cut regions.

## Requirements

- **ffmpeg** and **ffprobe** on `PATH`
- **Python 3.9+**
- **numpy** (required)
- Optional: **pytesseract** + system **tesseract-ocr** (for clock OCR)
- Optional: **opencv-python** (for auto scorebug detection and VLM ROI candidates)
- Optional: **llama-cpp-python** + **huggingface-hub** (for VLM clock ROI calibrator)

## Install

```bash
pip install numpy
# Optional for OCR / auto-ROI:
pip install pytesseract opencv-python
# Optional for VLM (clock ROI calibrator on real sports footage):
pip install opencv-python llama-cpp-python huggingface-hub
```

## Quick start

```bash
# Minimal: output next to input as <stem>.condensed.mkv
python3 condensearr.py /path/to/recording.mkv

# Recommended: full config + diagnostics (default = variable-length; add --target-minutes 18 for fixed length)
python3 condensearr.py /path/to/recording.mkv \
  --config condensearr_config.full.json \
  --out condensed.mkv \
  --emit-diagnostics diagnostics.json

# Validate after encode
python3 validate_condensed.py --output condensed.mkv --diagnostics diagnostics.json
```

For production runs, use **fused** mode (default) with the full config so motion, audio, and optional clock OCR are all used. See [Best way to run](#best-way-to-run) below. **Default:** variable-length — we remove the most obviously removable (low-action) content. **Optional:** pass `--target-minutes N` to condense to a fixed length (e.g. 18 min). See [docs/DESIGN.md](docs/DESIGN.md#condensing-model).

## Options (summary)

| Option | Description |
|--------|-------------|
| `input` | Input video path (positional). |
| `--out` | Output file path. |
| `--out-dir DIR` | Write output as `DIR/<stem>.condensed.<ext>` (for Tdarr/Arr). |
| `--config PATH` | JSON config; default from `CONDENSEARR_CONFIG` env if set. |
| `--target-minutes N` | Optional. Target output length in minutes; if omitted, variable-length (remove low-action only). |
| `--min-duration SEC` | Skip if source duration < SEC (exit 0). For automation. |
| `--jobs N` | Parallel encode jobs (0 = auto; “nice” mode caps at 2). |
| `--no-nice` | Don’t use nice/ionice; use full CPU. |
| `--emit-diagnostics PATH` | Write JSON with validation stats and EDL. |
| `--analysis-only` | Run signals and segment selection only; write EDL and diagnostics, skip encode. |
| `--auto-roi` | Enable OCR with auto scorebug detection (if ocr not in config). |
| `--clock-roi x,y,w,h` | Use this region for clock OCR (overrides config). |
| `--debug-dir DIR` | Write auto-ROI debug images to DIR. |
| `--kill-prior` | Kill other running condensearr processes before starting. |

Default behavior uses low CPU priority (nice 19) and idle I/O (ionice -c 3) so the system stays responsive.

## Best way to run

For real games, use **fused** (default) with the full config and validate afterward:

1. **Config:** `--config condensearr_config.full.json`. Add `--auto-roi` or `--clock-roi x,y,w,h` if the clock isn’t in the config.
2. **Length:** Omit `--target-minutes` for variable-length, or pass `--target-minutes 18` to condense to 18 minutes.
3. **Diagnostics:** `--emit-diagnostics path/to/diag.json` so you can run the validator.
4. **Validate:** `python3 validate_condensed.py --output <out>.mkv --diagnostics diag.json`. Expect action ratio ≥ 1.2 and fraction of hard events in kept ≥ 0.9.

Single-signal methods (motion-only, RMS-only, etc.) are for comparison only; they often miss whistles/peaks. The method matrix (below) proves fused is the right default.

## Method matrix

You can run **all method combinations** (single-signal through fused) and compare diagnostics to confirm fused is best:

```bash
# With a generated 120s fixture (no input needed)
python3 run_matrix.py matrix_out --gen-fixture

# With your own video (analysis-only by default; use --render to encode each combo)
python3 run_matrix.py /path/to/game.mkv matrix_out
```

Outputs: `matrix_out/comparison.csv`, `comparison.json`, and per-combo `diag_<id>.json`. Use `--limit N` to run only the first N combos (e.g. for a fast smoke test).

**Reference results:** An [NBA clip (10 min) results table](docs/METHOD_MATRIX_RESULTS.md) is in the repo so you can compare your runs. Fused/all_action/clock should give 100% of hard events in kept segments; single-signal rms/flux often do not.

- **[docs/METHOD_MATRIX.md](docs/METHOD_MATRIX.md)** — How to run the matrix, interpret results, and what they tell us.
- **[docs/METHOD_MATRIX_RESULTS.md](docs/METHOD_MATRIX_RESULTS.md)** — NBA clip reference table and how to reproduce.

## Validation

**validate_condensed.py** checks:

1. **Structural** — Output duration matches the EDL (within tolerance); file has video and audio.
2. **Content** (with `--diagnostics`) — Kept segments have higher action than cut; most hard events (whistles/peaks) fall inside kept segments; optional max-gap check.

Condensearr writes `<output>.fused.edl` next to the output so you can validate without keeping the temp dir. See [docs/DESIGN.md](docs/DESIGN.md#validation) for the logic.

## All optional / advanced features

To enable OCR clock weighting, auto scorebug detection, and diagnostics:

1. Install: `pip install pytesseract opencv-python` (and a Tesseract binary).
2. Run: `--config condensearr_config.full.json --emit-diagnostics diagnostics.json`.

If the log says *"Auto ROI detection failed to find a confident clock region"*, you can lower `ocr.auto_detect.min_hit_rate` in the config, use `--debug-dir` to inspect candidates, or set `--clock-roi x,y,w,h` (or `clock_roi` in config).

## Automation (Tdarr / *Arr)

Use `--out-dir`, `CONDENSEARR_CONFIG`, and `--min-duration` for unattended runs. See **[docs/ARR_INTEGRATION.md](docs/ARR_INTEGRATION.md)** for Tdarr Run CLI and Sonarr/Radarr custom script examples.

## Testing

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt pytesseract opencv-python pytest
.venv/bin/pytest tests/ -v
```

Or without pytest: `python3 -m unittest discover -s tests -v`

Tests include: CLI help and imports, optional deps, `compute_cut_quality_stats`, and the **method matrix** (matrix JSON structure and a one-combo smoke run with `run_matrix.py --gen-fixture --limit 1`).

## Optional: VLM as clock ROI calibrator

**Fully optional** (like OCR and opencv): no VLM dependency by default. When `vlm.enabled` is false or the backend is unavailable, a stub is used and the pipeline is unchanged.

**What it does:** On real sports footage, when auto-detecting the scorebug/clock region, the pipeline can ask a **vision-language model** “which of these candidate regions shows the game clock?” and lock OCR onto that ROI. The VLM is used as a **calibrator** (dozens of frame-queries), not as the main signal. On synthetic fixture (e.g. `--gen-fixture`) there is no real scorebug, so the VLM never picks one and the result is the same as with VLM off.

### VLM install

Use a venv so system Python stays clean:

```bash
python3 -m venv .venv
.venv/bin/pip install numpy opencv-python llama-cpp-python huggingface-hub
# Optional for OCR: .venv/bin/pip install pytesseract
```

No local GGUF download needed: the backend can load **Moondream2** from HuggingFace (`vikhyatk/moondream2`) on first use.

### VLM config

In `condensearr_config.full.json` (or your config), set:

```json
"vlm": {
  "enabled": true,
  "backend": "llama_cpp",
  "model_path": "vikhyatk/moondream2",
  "max_clock_calibration_queries": 20,
  "n_gpu_layers": 0
}
```

- **model_path** — `"vikhyatk/moondream2"` loads from HuggingFace; or a local path to a multimodal GGUF.
- **n_gpu_layers** — `0` = CPU only; `-1` = full GPU offload (if llama-cpp-python was built with GPU).

### Run with VLM

Same as usual; the full config enables VLM when the above is set:

```bash
.venv/bin/python3 condensearr.py /path/to/game.mkv --config condensearr_config.full.json --emit-diagnostics diag.json
.venv/bin/python3 run_matrix.py matrix_nba/nba_clip_10min.mkv matrix_nba
```

If the VLM backend fails to load (missing package or wrong API), the pipeline falls back to OCR-only ROI selection with no error. See **[docs/VLM.md](docs/VLM.md)** for design, config keys, and module boundaries.

## Docs

- **[docs/DESIGN.md](docs/DESIGN.md)** — Design and signal logic (motion, audio, OCR, fusion, validation).
- **[docs/VLM.md](docs/VLM.md)** — Optional VLM (calibrator + appeals court), CPU/GPU, backends.
- **[docs/METHOD_MATRIX.md](docs/METHOD_MATRIX.md)** — Method matrix: how to run, interpret, and what the results mean.
- **[docs/METHOD_MATRIX_RESULTS.md](docs/METHOD_MATRIX_RESULTS.md)** — NBA clip reference results table.
- **[docs/ARR_INTEGRATION.md](docs/ARR_INTEGRATION.md)** — Tdarr and *Arr integration.
- **[docs/LEGAL.md](docs/LEGAL.md)** — Legal: MIT, dependency compatibility, sharp corners.

## License

**MIT License.** Copyright (c) 2025 snapetech. See [LICENSE](LICENSE) for full terms.
