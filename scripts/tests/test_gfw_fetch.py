"""Offline tests for scripts/gfw_fetch.py (the GFW 4WINGS effort ingester).

Pure stdlib, no token / no network. Exercises the parts correctness depends on: the
argv-driven parse_args, date validation + range resolution, the gear filter clause, the
request assembly, and the hardened atomic JSON writer (including tmp cleanup on failure).
The HTTP POST itself is a thin requests shell (lazy-imported in main) and is not tested here.
Run either of:
    python scripts/tests/test_gfw_fetch.py
    python -m unittest scripts.tests.test_gfw_fetch
"""
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import gfw_fetch as gf  # noqa: E402


class TestParseArgs(unittest.TestCase):
    def test_defaults(self):
        a = gf.parse_args([])
        self.assertEqual(a.region_id, 8466)
        self.assertEqual(a.region_dataset, "public-eez-areas")
        self.assertEqual(a.gear, "trawlers")
        self.assertIsNone(a.start)
        self.assertIsNone(a.end)

    def test_overrides(self):
        a = gf.parse_args(["--region-id", "42", "--start", "2026-01-01",
                           "--end", "2026-01-02", "--gear", "trawlers,longliners"])
        self.assertEqual(a.region_id, 42)
        self.assertEqual(a.start, "2026-01-01")
        self.assertEqual(a.gear, "trawlers,longliners")


class TestDateHandling(unittest.TestCase):
    def test_validate_date_passthrough_none(self):
        self.assertIsNone(gf.validate_date("start", None))

    def test_validate_date_ok(self):
        self.assertEqual(gf.validate_date("start", "2026-07-25"), "2026-07-25")

    def test_validate_date_rejects_garbage(self):
        with self.assertRaises(ValueError):
            gf.validate_date("start", "07/25/2026")
        with self.assertRaises(ValueError):
            gf.validate_date("end", "2026-13-40")

    def test_resolve_defaults_to_yesterday(self):
        args = gf.parse_args([])
        start, end = gf.resolve_date_range(args)
        yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
        self.assertEqual(start, yesterday)
        self.assertEqual(end, yesterday)

    def test_resolve_explicit_range(self):
        args = gf.parse_args(["--start", "2026-01-01", "--end", "2026-01-31"])
        self.assertEqual(gf.resolve_date_range(args), ("2026-01-01", "2026-01-31"))

    def test_resolve_rejects_inverted_range(self):
        args = gf.parse_args(["--start", "2026-02-01", "--end", "2026-01-01"])
        with self.assertRaises(ValueError):
            gf.resolve_date_range(args)


class TestGearFilter(unittest.TestCase):
    def test_single(self):
        self.assertEqual(gf.gear_filter("trawlers"), "geartype in ('trawlers')")

    def test_multi_trims_whitespace(self):
        self.assertEqual(
            gf.gear_filter(" trawlers , longliners "),
            "geartype in ('trawlers', 'longliners')",
        )

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            gf.gear_filter("  , ,")


class TestBuildRequest(unittest.TestCase):
    def test_shape(self):
        args = gf.parse_args(["--region-id", "8466", "--gear", "trawlers"])
        params, payload = gf.build_request(args, "2026-01-01", "2026-01-02")
        self.assertEqual(params["date-range"], "2026-01-01,2026-01-02")
        self.assertEqual(params["datasets[0]"], "public-global-fishing-effort:latest")
        self.assertEqual(params["filters[0]"], "geartype in ('trawlers')")
        self.assertEqual(payload["region"], {"dataset": "public-eez-areas", "id": 8466})


class TestAtomicWrite(unittest.TestCase):
    def test_writes_and_roundtrips(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "sub" / "gfw_report_x.json"  # parent auto-created
            gf.write_json_atomic(out, {"ok": True, "n": 3})
            self.assertEqual(json.loads(out.read_text()), {"ok": True, "n": 3})
            self.assertFalse(out.with_name(out.name + ".tmp").exists())

    def test_cleans_tmp_on_serialization_failure(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "gfw_report_x.json"
            unserializable = {"bad": object()}
            with self.assertRaises(TypeError):
                gf.write_json_atomic(out, unserializable)
            self.assertFalse(out.exists())
            self.assertFalse(out.with_name(out.name + ".tmp").exists())

    def test_overwrite_is_atomic(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "gfw_report_x.json"
            gf.write_json_atomic(out, {"v": 1})
            gf.write_json_atomic(out, {"v": 2})
            self.assertEqual(json.loads(out.read_text()), {"v": 2})


if __name__ == "__main__":
    unittest.main()
