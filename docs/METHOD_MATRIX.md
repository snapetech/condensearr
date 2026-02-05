# Method matrix testing

Run **all method combinations** (single-signal → fused) and compare diagnostics.

**Reference results:** See [METHOD_MATRIX_RESULTS.md](METHOD_MATRIX_RESULTS.md) for the NBA clip results table to compare against when re-running the matrix.

## Quick run (with generated fixture)

```bash
python3 run_matrix.py matrix_out --gen-fixture
```

- Generates `matrix_out/fixture.mkv` (120s test pattern + tone) if missing.
- Runs 17 combos in **analysis-only** mode (no encode).
- Writes `matrix_out/comparison.csv`, `matrix_out/comparison.json`, and per-combo `diag_<id>.json`.

## Run with your own video

```bash
python3 run_matrix.py /path/to/game.mkv matrix_out
```

Use `--render` to run full encode for each combo (slower).

## Matrix contents

| Kind | Combos |
|------|--------|
| Single signal | motion, rms, flux, whistle |
| Pairs | motion+rms, motion+flux, motion+whistle, rms+flux, rms+whistle, flux+whistle |
| Triples | motion+rms+flux, motion+rms+whistle, motion+flux+whistle, rms+flux+whistle |
| All signals (action) | default weights, render_mode=action |
| Clock only | default weights, render_mode=clock (needs OCR) |
| Fused | default weights, render_mode=fused |

## Outputs

- **comparison.csv** — One row per combo: id, name, ok, render_mode, action_ratio_kept_vs_cut, fraction_hard_in_kept, segment_count, length_seconds.
- **comparison.json** — Same data plus full diagnostics path and analysis_only flag.
- **diag_<id>.json** — Full diagnostics for that combo (validation stats, segment counts, EDL segment list).
- **config_<id>.json** — Config used for that combo.

## Interpreting results

- **action_ratio_kept_vs_cut** — Mean action in kept segments vs cut; we want ≥ 1.2 (validators use this). Synthetic fixture can yield odd ratios. On short clips or when target length is close to source length, ratios can be negative or odd because the “cut” region is small or has different signal (e.g. loud ads).
- **fraction_hard_in_kept** — Fraction of hard-event bins (whistles/peaks) inside kept segments; we want ≥ 0.9.
- Compare **fused** (and **all_action**) to single-signal rows: fused should be at least as good on these metrics for real sports footage.

## What the results table tells us

- **Fused is the right default.** In the NBA clip matrix, **fused** (and **all_action** / **clock**) kept **100% of hard events** with the fewest segments (8) and near-target length. Single-signal methods often did worse: **RMS only** and **flux only** kept only 0–5% of hard events (whistles/peaks), so they would drop most key moments. That confirms we should **not** run with a single signal for real games.
- **Configuration is sound.** Using default weights (motion + RMS + flux + whistle) and **render_mode: fused** is correct. The matrix validates that the combined method captures what single methods miss.
- **When to re-evaluate.** Re-tune or re-run the matrix if: (1) you change signal logic or weights, (2) you add new signals, or (3) validation fails on real runs (e.g. `validate_condensed.py --diagnostics` fails action ratio or hard-events-in-kept). Negative action ratios on a **short** clip (e.g. 10 min with 18 min target) are not enough by themselves to re-evaluate; run validation on a **full-length** game and check ratio ≥ 1.2 and fraction_hard_in_kept ≥ 0.9.

## Best method to run the script

For **production** (real games):

1. Use **fused** (default). Do not use a single-signal method or `render_mode: action` alone unless you are doing comparison tests.
2. Use the full config with OCR if you have a clock/scorebug:  
   `--config condensearr_config.full.json`  
   Optionally add `--auto-roi` or `--clock-roi x,y,w,h` if not in config.
3. Emit diagnostics so you can validate:  
   `--emit-diagnostics path/to/diag.json`
4. Example:

   ```bash
   python3 condensearr.py /path/to/game.mkv \
     --config condensearr_config.full.json \
     --emit-diagnostics game_diag.json \
     --out /path/to/game.condensed.mkv
   ```

5. After encoding, run `validate_condensed.py --output game.condensed.mkv --diagnostics game_diag.json` and fix config or logic if it fails.
