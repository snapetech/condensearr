# Condensearr + Tdarr / *Arr automation

Condensearr is **CLI-ready** for automation: non-interactive, exit 0 on success and non-zero on failure, and supports env-based config and output conventions that Tdarr and *Arr custom scripts expect.

## What’s required for Arr use

| Requirement | Status |
|-------------|--------|
| Single command with input path | ✅ `condensearr.py <input> [--out <path>]` |
| Predictable output path | ✅ `--out <path>` or `--out-dir <dir>` (writes `<dir>/<stem>.condensed.mkv`) |
| Config without CLI | ✅ `CONDENSEARR_CONFIG=/path/to/config.json` |
| Skip short files (filter) | ✅ `--min-duration 3600` → exit 0 without processing if duration < 1h |
| Exit codes | ✅ 0 = success, 1 = error, 2 = input not found |
| No prompts | ✅ Fully non-interactive |

## Tdarr (Run CLI plugin)

Tdarr can run condensearr via the **Run CLI** flow plugin.

1. **Custom CLI path:** your Python 3 binary, e.g. `/usr/bin/python3`.
2. **CLI arguments:** path to script + input + output. Tdarr exposes the current file and an optional output path. Example (adjust paths):

   ```text
   /path/to/condensearr.py "{{{args.inputFileObj.file}}}"
   --out-dir "{{{args.outputFilePath}}}" --min-duration 3600
   ```

   Or if Tdarr gives you a single output file path:

   ```text
   /path/to/condensearr.py "{{{args.inputFileObj.file}}}" --out "{{{args.outputFilePath}}}"
   ```

3. **Environment (server/worker):** set `CONDENSEARR_CONFIG` to your default config so you don’t need `--config` in the arguments.
4. **Filter:** use Tdarr’s flow filters so this only runs on the right library (e.g. “Sports” or path contains “NBA”). Optionally use `--min-duration 3600` so files under 1 hour are skipped with exit 0.

**Example Run CLI setup (conceptual):**

- **Use Custom CLI Path:** Yes  
- **Custom CLI Path:** `/usr/bin/python3`  
- **CLI Arguments:**  
  `/path/to/condensearr.py "{{{args.inputFileObj.file}}}" --out-dir "{{{args.cacheDir}}}" --min-duration 3600`  
- **Does Command Create Output File?** Yes  
- **Output File Path:** `${cacheDir}/${fileName}.condensed.mkv`  
- **Output File Becomes Working File?** Yes (if you want the condensed file to replace/follow in the pipeline)

(Exact variable names may differ; check Tdarr’s Run CLI docs for the current templating.)

## Sonarr / Radarr custom script

These typically call a script and pass the file path (or set env vars like `SONARR_EPISODEFILE_PATH`). Use a small wrapper that forwards that path to condensearr.

**Example wrapper (e.g. `condensearr-sonarr.sh`):**

```bash
#!/bin/bash
# Usage: condensearr-sonarr.sh /path/to/video.mkv
# Or: Sonarr sets SONARR_EPISODEFILE_PATH; use that if no arg.
INPUT="${1:-$SONARR_EPISODEFILE_PATH}"
OUT_DIR="${CONDENSEARR_OUT_DIR:-/path/to/condensed/library}"

export CONDENSEARR_CONFIG="${CONDENSEARR_CONFIG:-$HOME/.config/condensearr/config.json}"

exec python3 /path/to/condensearr.py "$INPUT" \
  --out-dir "$OUT_DIR" \
  --min-duration 3600
```

- **Exit 0:** success (or skipped because of `--min-duration`).  
- **Exit non-zero:** failure; Sonarr/Radarr will treat it as a script error.

Point Sonarr/Radarr’s “Custom Script” at this wrapper and (if applicable) set the env vars in the app’s script settings or systemd/service environment.

## Validation after run

To validate the condensed file and (optionally) quality stats:

```bash
python3 /path/to/validate_condensed.py \
  --output "$OUT_PATH" \
  --edl condensed.fused.edl \
  --diagnostics diagnostics.json
```

Condensearr writes `<output>.fused.edl` next to the output. Use `--emit-diagnostics` when calling condensearr if you want content-quality checks in the validator. Run `validate_condensed.py` from the repo root (or ensure it’s on `PATH`).

## Summary

- **Ready for Arr use:** yes, with the options above.
- **Tdarr:** use Run CLI with `python3`, path to `condensearr.py`, and `--out` or `--out-dir` plus optional `--min-duration` and `CONDENSEARR_CONFIG`.
- **Sonarr/Radarr:** use a wrapper script that passes the file path and (optionally) `CONDENSEARR_OUT_DIR` and `CONDENSEARR_CONFIG` into condensearr.
