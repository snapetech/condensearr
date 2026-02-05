"""
Tests for method matrix: matrix JSON structure and run_matrix.py smoke run.
Run with: python3 -m pytest tests/test_method_matrix.py -v
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MATRIX_PATH = ROOT / "tests" / "method_matrix.json"


class TestMethodMatrixJson(unittest.TestCase):
    def test_matrix_file_exists(self) -> None:
        self.assertTrue(MATRIX_PATH.exists(), f"Matrix file missing: {MATRIX_PATH}")

    def test_matrix_has_expected_structure(self) -> None:
        with open(MATRIX_PATH, encoding="utf-8") as f:
            data = json.load(f)
        combos = data.get("combos", [])
        self.assertGreaterEqual(len(combos), 17, "Matrix should have at least 17 combos")
        for i, c in enumerate(combos):
            self.assertIn("id", c, f"Combo {i} missing 'id'")
            self.assertIn("render_mode", c, f"Combo {i} missing 'render_mode'")
            self.assertIn(c["render_mode"], ("action", "clock", "fused"), f"Combo {i} invalid render_mode")
            aw = c.get("audio_weights", {})
            for k in ("w_motion", "w_rms", "w_flux", "w_whistle"):
                self.assertIn(k, aw, f"Combo {i} audio_weights missing {k}")

    def test_fused_combo_present(self) -> None:
        with open(MATRIX_PATH, encoding="utf-8") as f:
            data = json.load(f)
        ids = [c["id"] for c in data["combos"]]
        self.assertIn("fused", ids, "Matrix should include 'fused' combo")


class TestRunMatrixCLI(unittest.TestCase):
    def test_run_matrix_help(self) -> None:
        r = subprocess.run(
            [sys.executable, str(ROOT / "run_matrix.py"), "--help"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=10,
        )
        self.assertEqual(r.returncode, 0, msg=r.stderr or r.stdout)
        self.assertIn("out_dir", r.stdout)
        self.assertIn("--gen-fixture", r.stdout)
        self.assertIn("--limit", r.stdout)


class TestRunMatrixSmoke(unittest.TestCase):
    """Run matrix with --gen-fixture and --limit 1 to verify pipeline (one combo)."""

    def test_run_matrix_one_combo(self) -> None:
        with tempfile.TemporaryDirectory(prefix="condensearr_matrix_") as tmp:
            out_dir = Path(tmp) / "out"
            r = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "run_matrix.py"),
                    str(out_dir),
                    "--gen-fixture",
                    "--limit",
                    "1",
                ],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
                timeout=120,
            )
            self.assertEqual(r.returncode, 0, msg=r.stderr or r.stdout)
            comparison = out_dir / "comparison.json"
            self.assertTrue(comparison.exists(), "comparison.json should be written")
            with open(comparison, encoding="utf-8") as f:
                report = json.load(f)
            combos = report.get("combos", [])
            self.assertEqual(len(combos), 1, "Should have exactly one combo result")
            self.assertTrue(combos[0].get("ok"), "Single combo run should succeed")
