"""
Smoke tests: CLI --help, imports, and optional deps.
Run with: python3 -m pytest tests/ -v
Or:       python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

# Repo root
ROOT = Path(__file__).resolve().parent.parent


class TestCLI(unittest.TestCase):
    def test_condensearr_help(self) -> None:
        r = subprocess.run(
            [sys.executable, str(ROOT / "condensearr.py"), "--help"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=10,
        )
        self.assertEqual(r.returncode, 0, msg=r.stderr or r.stdout)
        self.assertIn("input", r.stdout)
        self.assertIn("--out", r.stdout)

    def test_validate_condensed_help(self) -> None:
        r = subprocess.run(
            [sys.executable, str(ROOT / "validate_condensed.py"), "--help"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=10,
        )
        self.assertEqual(r.returncode, 0, msg=r.stderr or r.stdout)
        self.assertIn("--output", r.stdout)
        self.assertIn("--edl", r.stdout)


class TestImports(unittest.TestCase):
    def test_condensearr_imports(self) -> None:
        """Main script imports and has expected pieces."""
        sys.path.insert(0, str(ROOT))
        try:
            import condensearr as m  # type: ignore[import-not-found]
            self.assertTrue(hasattr(m, "run"))
            self.assertTrue(hasattr(m, "compute_cut_quality_stats"))
            self.assertTrue(hasattr(m, "total_len"))
        finally:
            if str(ROOT) in sys.path:
                sys.path.remove(str(ROOT))

    def test_validate_condensed_imports(self) -> None:
        sys.path.insert(0, str(ROOT))
        try:
            import validate_condensed as m  # type: ignore[import-not-found]
            self.assertTrue(hasattr(m, "run_structural_checks"))
            self.assertTrue(hasattr(m, "load_edl"))
        finally:
            if str(ROOT) in sys.path:
                sys.path.remove(str(ROOT))


class TestOptionals(unittest.TestCase):
    """Optional deps: only pass if installed (pytesseract, opencv)."""

    def test_numpy_imported(self) -> None:
        sys.path.insert(0, str(ROOT))
        try:
            import condensearr as m  # type: ignore[import-not-found]
            self.assertIsNotNone(m.np, "numpy should be required and present")
        finally:
            if str(ROOT) in sys.path:
                sys.path.remove(str(ROOT))

    def test_pytesseract_optional(self) -> None:
        """If pytesseract is installed, condensearr should see it (or None)."""
        sys.path.insert(0, str(ROOT))
        try:
            import condensearr as m  # type: ignore[import-not-found]
            # Module may set pytesseract to None if import fails
            self.assertTrue(hasattr(m, "pytesseract"))
        finally:
            if str(ROOT) in sys.path:
                sys.path.remove(str(ROOT))

    def test_cv2_optional(self) -> None:
        """If opencv (cv2) is installed, condensearr should see it (or None)."""
        sys.path.insert(0, str(ROOT))
        try:
            import condensearr as m  # type: ignore[import-not-found]
            self.assertTrue(hasattr(m, "cv2"))
        finally:
            if str(ROOT) in sys.path:
                sys.path.remove(str(ROOT))


class TestComputeCutQualityStats(unittest.TestCase):
    """Unit test for pure function with numpy."""

    def test_empty_returns_safe_defaults(self) -> None:
        import numpy as np

        sys.path.insert(0, str(ROOT))
        try:
            from condensearr import compute_cut_quality_stats  # type: ignore[import-not-found]

            out = compute_cut_quality_stats(
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.uint8),
                dt=0.5,
                duration=10.0,
                segs=[],
            )
            self.assertEqual(out["hard_events_total"], 0)
            self.assertEqual(out["fraction_hard_in_kept"], 1.0)
        finally:
            if str(ROOT) in sys.path:
                sys.path.remove(str(ROOT))

    def test_kept_has_higher_action_ratio(self) -> None:
        import numpy as np

        sys.path.insert(0, str(ROOT))
        try:
            from condensearr import compute_cut_quality_stats  # type: ignore[import-not-found]

            # 20 bins at dt=0.5 -> 10s; keep bins 5-9 (high), cut rest (low)
            action_z = np.array(
                [0.0] * 5 + [2.0] * 5 + [0.0] * 10,
                dtype=np.float32,
            )
            hard = np.zeros(20, dtype=np.uint8)
            hard[7] = 1
            segs = [(2.5, 5.0)]  # bins 5-9
            out = compute_cut_quality_stats(action_z, hard, dt=0.5, duration=10.0, segs=segs)
            self.assertGreater(out["action_ratio_kept_vs_cut"], 1.0)
            self.assertEqual(out["hard_events_in_kept"], 1)
            self.assertEqual(out["hard_events_total"], 1)
        finally:
            if str(ROOT) in sys.path:
                sys.path.remove(str(ROOT))
