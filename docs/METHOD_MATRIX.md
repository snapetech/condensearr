# Method matrix testing

Run **all method combinations** (single-signal → fused) and compare diagnostics.

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

- **action_ratio_kept_vs_cut** — Mean action in kept segments vs cut; we want ≥ 1.2 (validators use this). Synthetic fixture can yield odd ratios.
- **fraction_hard_in_kept** — Fraction of hard-event bins inside kept segments; we want ≥ 0.9.
- Compare **fused** (and **all_action**) to single-signal rows: fused should be at least as good on these metrics for real sports footage.
