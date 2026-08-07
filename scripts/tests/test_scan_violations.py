"""Offline tests for scripts/scan_violations.py (P3 workstream 2A, the batch driver).

The fuse + report stages run for real over the committed fixtures (genuine integration); the GFW
stage (needs a token) is monkeypatched to drop in a canned vessel-records file so the re-fuse path is
exercised without network. Pure stdlib. Run:
    python scripts/tests/test_scan_violations.py
    python -m unittest scripts.tests.test_scan_violations
"""
import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = TESTS_DIR.parent
FIXTURES = TESTS_DIR / "fixtures"
sys.path.insert(0, str(SCRIPTS_DIR))

import scan_violations as scan  # noqa: E402


class _Args:
    """Minimal stand-in for parse_args output."""
    def __init__(self, **kw):
        self.run = self.ais = self.vessels = None
        self.gfw = False
        self.gfw_start = self.gfw_end = self.wdpa_id = self.dt_minutes = self.weights = None
        self.detect = False
        self.no_report = False
        self.__dict__.update(kw)


class TestScanDriver(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.run = Path(self._td.name) / "run"
        shutil.copytree(FIXTURES / "run", self.run)
        self.ais = FIXTURES / "ais_positions.json"
        self.vessels = FIXTURES / "gfw_vessels.json"

    def tearDown(self):
        self._td.cleanup()

    def test_fuse_then_report_real(self):
        summary = scan.run_pipeline(_Args(run=self.run, ais=self.ais))
        self.assertEqual(summary["stages"], ["fuse", "report"])
        # fuse output exists and is valid
        doc = json.loads((self.run / "violation_events.json").read_text())
        self.assertEqual(doc["counts"]["in_scope"], 2)
        # report output exists (CSV + HTML) in the run dir
        self.assertTrue(list(self.run.glob("violations_report_*.csv")))
        self.assertTrue(list(self.run.glob("violations_report_*.html")))

    def test_gfw_enrich_reFuses_with_vessels(self):
        # Monkeypatch the GFW stage to write a canned vessels file (as a real gfw_vessels run would).
        calls = []

        def fake_gfw(events_json, out_path, start=None, end=None, wdpa_id=None):
            calls.append({"events": Path(events_json).name, "wdpa_id": wdpa_id})
            Path(out_path).write_text((FIXTURES / "gfw_vessels.json").read_text())

        orig = scan.stage_gfw_vessels
        scan.stage_gfw_vessels = fake_gfw
        try:
            summary = scan.run_pipeline(_Args(run=self.run, ais=self.ais, gfw=True, wdpa_id="555622064"))
        finally:
            scan.stage_gfw_vessels = orig

        self.assertEqual(summary["stages"], ["fuse", "gfw_vessels", "refuse", "report"])
        self.assertEqual(calls[0]["events"], "violation_events.json")   # fed the first fuse output
        self.assertEqual(calls[0]["wdpa_id"], "555622064")              # region passed through
        # final (re-fused) events are GFW-enriched
        doc = json.loads((self.run / "violation_events.json").read_text())
        self.assertEqual(doc["counts"]["gfw_enriched"], 1)
        matched = next(e for e in doc["violation_events"] if e["ais_status"] == "matched")
        self.assertEqual(matched["vessel_name"], "NEPTUNE")
        self.assertIn("GFW", matched["evidence"])

    def test_given_vessels_skips_gfw_even_with_flag(self):
        summary = scan.run_pipeline(_Args(run=self.run, ais=self.ais, vessels=self.vessels, gfw=True))
        self.assertNotIn("gfw_vessels", summary["stages"])   # --vessels supplied -> no fetch
        doc = json.loads((self.run / "violation_events.json").read_text())
        self.assertEqual(doc["counts"]["gfw_enriched"], 1)   # but still enriched from the given file

    def test_no_report_flag(self):
        summary = scan.run_pipeline(_Args(run=self.run, ais=self.ais, no_report=True))
        self.assertEqual(summary["stages"], ["fuse"])
        self.assertFalse(list(self.run.glob("violations_report_*.csv")))

    def test_gfw_skip_when_no_matched_mmsis(self):
        # An all-dark run has no matched MMSIs, so gfw_vessels raises "No MMSIs"; the driver
        # must log + skip the enrich (no refuse) and still produce the report, not crash.
        def raising_gfw(events_json, out_path, start=None, end=None, wdpa_id=None):
            raise RuntimeError("No MMSIs.")
        orig = scan.stage_gfw_vessels
        scan.stage_gfw_vessels = raising_gfw
        try:
            summary = scan.run_pipeline(_Args(run=self.run, ais=self.ais, gfw=True))
        finally:
            scan.stage_gfw_vessels = orig
        self.assertEqual(summary["stages"], ["fuse", "report"])
        self.assertTrue(list(self.run.glob("violations_report_*.csv")))

    def test_console_output_survives_legacy_codepage(self):
        # Windows consoles default to legacy code pages; the driver's summary print crashed on
        # U+2192 under cp1252. Drive main() through a strict cp1252 stdout to lock console-safety.
        stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
        with contextlib.redirect_stdout(stream):
            rc = scan.main(["--run", str(self.run), "--ais", str(self.ais)])
        stream.flush()
        self.assertFalse(rc)

    def test_requires_run_and_ais(self):
        with self.assertRaises(RuntimeError):
            scan.run_pipeline(_Args(run=None, ais=self.ais))
        with self.assertRaises(RuntimeError):
            scan.run_pipeline(_Args(run=self.run, ais=None))   # nothing to fuse


if __name__ == "__main__":
    unittest.main(verbosity=2)
