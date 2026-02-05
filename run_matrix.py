#!/usr/bin/env python3
"""
Run the method matrix: run condensearr for each combo (single-signal through fused),
collect diagnostics, and emit a comparison report.

Usage:
  python3 run_matrix.py INPUT_VIDEO OUTPUT_DIR [--matrix PATH] [--base-config PATH] [--render]
  --analysis-only (default): run signals + segment selection only, no encode.
  --render: run full encode for each combo (slower).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MATRIX_PATH = ROOT / "tests" / "method_matrix.json"
DEFAULT_BASE_CONFIG = ROOT / "condensearr_config.full.json"


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def merge_combo_config(base: dict, combo: dict) -> dict:
    out = json.loads(json.dumps(base))
    if "audio_weights" in combo:
        out.setdefault("audio_weights", {}).update(combo["audio_weights"])
    if "render_mode" in combo:
        out.setdefault("combine", {})["render_mode"] = combo["render_mode"]
    return out


def run_condensearr(
    input_path: Path,
    out_dir: Path,
    config_path: Path,
    diag_path: Path,
    analysis_only: bool,
    out_file: Path,
) -> bool:
    cmd = [
        sys.executable,
        str(ROOT / "condensearr.py"),
        str(input_path),
        "--config",
        str(config_path),
        "--emit-diagnostics",
        str(diag_path),
        "--out",
        str(out_file),
        "--no-nice",
    ]
    if analysis_only:
        cmd.append("--analysis-only")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, cwd=str(ROOT))
        if r.returncode != 0:
            print(r.stderr or r.stdout, file=sys.stderr)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"Timeout: {config_path.name}", file=sys.stderr)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Run method matrix and compare diagnostics.")
    ap.add_argument("input", type=str, nargs="?", default="", help="Input video path (or use --gen-fixture to create one in out_dir).")
    ap.add_argument("out_dir", type=str, help="Output directory for configs, diagnostics, and report.")
    ap.add_argument("--matrix", type=str, default="", help=f"Method matrix JSON (default: {MATRIX_PATH}).")
    ap.add_argument("--base-config", type=str, default="", help="Base config JSON (default: condensearr_config.full.json or minimal).")
    ap.add_argument("--render", action="store_true", help="Run full encode for each combo (default: analysis-only).")
    ap.add_argument("--gen-fixture", action="store_true", help="Generate a short test video in out_dir and use it (for CI/demo).")
    ap.add_argument("--limit", type=int, default=0, help="Run only first N combos (0 = all). For CI/smoke test.")
    ap.add_argument("--target-minutes", type=float, default=0, help="Override target_minutes in config (e.g. 5 for a 10-min clip so we actually condense).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.gen_fixture:
        fixture = out_dir / "fixture.mkv"
        if not fixture.exists():
            print("Generating fixture video (120s)...")
            subprocess.run(
                [
                    "ffmpeg", "-y", "-f", "lavfi", "-i", "smptebars=duration=120:size=320x180:rate=4",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=120",
                    "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", "-shortest", str(fixture),
                ],
                capture_output=True,
                timeout=60,
                check=True,
            )
        input_path = fixture
    else:
        if not args.input:
            print("Provide input video path or use --gen-fixture.", file=sys.stderr)
            return 2
        input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 2

    matrix_path = Path(args.matrix).expanduser().resolve() if args.matrix else MATRIX_PATH
    if not matrix_path.exists():
        print(f"Matrix not found: {matrix_path}", file=sys.stderr)
        return 2

    base_config_path = Path(args.base_config).expanduser().resolve() if args.base_config else DEFAULT_BASE_CONFIG
    if base_config_path.exists():
        base_config = load_json(base_config_path)
    else:
        base_config = {
            "audio_weights": {"w_motion": 0.35, "w_rms": 0.35, "w_flux": 0.20, "w_whistle": 0.10},
            "combine": {"target_minutes": 0, "render_mode": "fused"},
            "render": {},
        }
    if args.target_minutes > 0:
        base_config.setdefault("combine", {})["target_minutes"] = args.target_minutes
        print(f"Overriding target_minutes to {args.target_minutes}")

    matrix = load_json(matrix_path)
    combos = matrix.get("combos", [])
    if not combos:
        print("No combos in matrix.", file=sys.stderr)
        return 2
    if args.limit > 0:
        combos = combos[: args.limit]
        print(f"Limited to first {len(combos)} combos (--limit {args.limit}).")

    analysis_only = not args.render
    print(f"Running {len(combos)} combos (analysis_only={analysis_only})...")

    results = []
    for i, combo in enumerate(combos):
        cid = combo.get("id", f"combo_{i}")
        name = combo.get("name", cid)
        config = merge_combo_config(base_config, combo)
        config_path = out_dir / f"config_{cid}.json"
        save_json(config_path, config)
        diag_path = out_dir / f"diag_{cid}.json"
        out_file = out_dir / f"out_{cid}.mkv"

        ok = run_condensearr(
            input_path, out_dir, config_path, diag_path, analysis_only, out_file
        )
        row = {"id": cid, "name": name, "ok": ok}
        if diag_path.exists():
            try:
                diag = load_json(diag_path)
                val = diag.get("validation", {})
                row["action_ratio_kept_vs_cut"] = val.get("action_ratio_kept_vs_cut")
                row["fraction_hard_in_kept"] = val.get("fraction_hard_in_kept")
                row["hard_events_in_kept"] = val.get("hard_events_in_kept")
                row["hard_events_total"] = val.get("hard_events_total")
                segs = diag.get("segments", {})
                lens = diag.get("length_seconds", {})
                mode = diag.get("render_mode", "")
                row["render_mode"] = mode
                row["segment_count"] = segs.get(mode)
                row["length_seconds"] = lens.get(mode)
                row["duration_seconds"] = diag.get("duration_seconds")
            except Exception as e:
                row["error"] = str(e)
        else:
            row["error"] = "no diagnostics"
        results.append(row)
        print(f"  {cid}: ok={ok} ratio={row.get('action_ratio_kept_vs_cut')} hard_kept={row.get('fraction_hard_in_kept')}")

    # Comparison report
    report = {
        "input": str(input_path),
        "analysis_only": analysis_only,
        "combos": results,
    }
    save_json(out_dir / "comparison.json", report)

    # CSV for easy eyeballing
    csv_path = out_dir / "comparison.csv"
    headers = ["id", "name", "ok", "render_mode", "action_ratio_kept_vs_cut", "fraction_hard_in_kept", "segment_count", "length_seconds"]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in results:
            f.write(",".join(str(r.get(h, "")) for h in headers) + "\n")
    print(f"Wrote {out_dir / 'comparison.json'} and {csv_path}")

    return 0 if all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
