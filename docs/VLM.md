# VLM-assisted ROI and appeals (optional)

Condensearr can use a **local vision-language model (VLM)** as:

1. **Calibrator** — When auto-detecting the scorebug/clock ROI, ask the VLM “which of these candidate regions shows the game clock?” so we lock ROI with few queries (dozens of frames, not the whole game).
2. **Appeals court** *(planned)* — On ambiguous segments (e.g. near threshold), optionally ask the VLM “live / replay / commercial?” and cap total queries (e.g. &lt;500 per game).

The VLM is **not** the main signal: motion, audio, and OCR remain the backbone. The VLM only calibrates ROI and (later) reviews ambiguous segments so the CPU budget stays in **single-digit minutes per game** with **&lt;500 frame-queries** for the VLM step.

## Design

- **Fully optional:** No VLM dependency by default. When `vlm.enabled` is false (default) or the `vlm` section is omitted, the vision judge is a stub and the pipeline behaves exactly as without VLM. Same pattern as optional OCR and opencv.
- **CPU-only baseline:** Everything works with OCR + CV. When VLM is enabled, `n_gpu_layers: 0` keeps it CPU-only.
- **Optional VLM:** Set `vlm.enabled: true` and `vlm.backend: "llama_cpp"` with a path to a multimodal GGUF (e.g. Moondream2). The backend uses **llama-cpp-python** if installed and built with multimodal support.
- **GPU on top:** Set `vlm.n_gpu_layers: -1` (or a positive number) to offload layers to GPU; `0` = CPU only.

## Config

```json
"vlm": {
  "enabled": false,
  "backend": "llama_cpp",
  "model_path": "/path/to/moondream2-xxx.gguf",
  "max_clock_calibration_queries": 20,
  "max_appeals_queries": 500,
  "n_gpu_layers": 0
}
```

- **enabled** — Turn on VLM (calibrator when `ocr.auto_detect` is true; appeals later).
- **backend** — `"none"` (stub) or `"llama_cpp"`.
- **model_path** — Path to multimodal GGUF (e.g. [Moondream2 GGUF](https://huggingface.co/ggml-org/moondream2-20250414-GGUF)).
- **max_clock_calibration_queries** — Cap VLM calls when choosing which candidate ROI is the clock.
- **max_appeals_queries** — Cap for appeals court (future).
- **n_gpu_layers** — `0` = CPU only; `-1` = full GPU offload (when llama-cpp-python is built with GPU).

## Modules

| Module | Role |
|--------|------|
| **roi_proposals.py** | CV-only: propose candidate scorebug boxes from frames (opencv). Used by both OCR path and VLM calibrator. |
| **vision_judge.py** | Interface `IVisionJudge` (which_roi_is_clock, classify_segment), stub implementation, factory `get_vision_judge(cfg)`. |
| **backends/llama_mm.py** | Optional: llama-cpp-python multimodal backend; loads GGUF, runs image + prompt, returns (index, confidence) or classification. |

When VLM is enabled and `ocr.auto_detect` is true, the pipeline:

1. Loads frames (same as before).
2. Gets candidate ROIs from `roi_proposals.propose_scorebug_candidates(frames)`.
3. Asks the vision judge “which candidate is the clock?” (up to `max_clock_calibration_queries`).
4. If the judge returns an index, we use that ROI for OCR. Otherwise we fall back to OCR-only selection (current behavior).

## Installing the optional VLM backend

- **llama-cpp-python** with multimodal support: install from source or use a build that includes vision (e.g. `pip install llama-cpp-python` and ensure the installed backend supports `image_path` or the chat image API for your version).
- **Model:** Download a small multimodal GGUF (e.g. Moondream2) and set `vlm.model_path` to the file.

If the backend fails to load (missing package or wrong API), the factory returns the stub and the pipeline falls back to OCR-only ROI selection.

## Licensing

Moondream is Apache-2.0. If you ship a binary or ship model weights, ensure you comply with the model’s license and redistribution terms.
