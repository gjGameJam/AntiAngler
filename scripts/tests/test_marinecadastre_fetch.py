"""Offline tests for scripts/marinecadastre_fetch.py (PHASE3_PLAN Workstream 3C).

Pure stdlib — builds a synthetic MarineCadastre CSV and asserts the row normalization, AOI/time
filtering, and a normalize → write → fuse_violations round trip (proving the output is the AIS
contract). The archive download is a do-from-home shell and is not tested here. Run:
    python scripts/tests/test_marinecadastre_fetch.py
    python -m unittest scripts.tests.test_marinecadastre_fetch
"""
import csv
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

import marinecadastre_fetch as mc  # noqa: E402
import fuse_violations as fv        # noqa: E402

MC_HEADER = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "VesselName", "IMO"]


def write_csv(path, rows):
    with Path(path).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(MC_HEADER)
        for r in rows:
            w.writerow(r)
    return path


class TestTimeAndRow(unittest.TestCase):
    def test_parse_mc_time(self):
        self.assertEqual(mc.parse_mc_time("2026-07-14T10:02:00"), "2026-07-14T10:02:00Z")
        self.assertEqual(mc.parse_mc_time("2026-07-14 10:02:00"), "2026-07-14T10:02:00Z")  # space sep
        self.assertEqual(mc.parse_mc_time("2026-07-14T10:02:00Z"), "2026-07-14T10:02:00Z")
        self.assertIsNone(mc.parse_mc_time(""))
        self.assertIsNone(mc.parse_mc_time("garbage"))

    def test_normalize_row_maps_columns(self):
        row = {"mmsi": "211476060", "basedatetime": "2026-07-14T10:02:00",
               "lat": "37.0004", "lon": "25.0004", "sog": "0.3", "cog": "180",
               "vesselname": "NEPTUNE", "imo": "9012345"}
        rec = mc.normalize_row(row)
        self.assertEqual(rec["mmsi"], "211476060")
        self.assertEqual(rec["lat"], 37.0004)
        self.assertEqual(rec["lon"], 25.0004)
        self.assertEqual(rec["timestamp"], "2026-07-14T10:02:00Z")
        self.assertEqual(rec["sog"], 0.3)
        self.assertEqual(rec["cog"], 180.0)
        self.assertEqual(rec["name"], "NEPTUNE")

    def test_normalize_row_rejects_bad(self):
        self.assertIsNone(mc.normalize_row({"mmsi": "1", "lat": "37", "lon": "25"}))          # no time
        self.assertIsNone(mc.normalize_row({"mmsi": "1", "basedatetime": "2026-07-14T00:00:00",
                                            "lat": "999", "lon": "25"}))                       # bad lat
        self.assertIsNone(mc.normalize_row({}))


class TestFilterAndStats(unittest.TestCase):
    def test_bbox_and_time_filter(self):
        rows = [
            {"mmsi": "1", "basedatetime": "2026-07-14T10:00:00", "lat": "37.0", "lon": "25.0"},   # keep
            {"mmsi": "2", "basedatetime": "2026-07-14T10:00:00", "lat": "10.0", "lon": "10.0"},   # out area
            {"mmsi": "3", "basedatetime": "2026-07-13T10:00:00", "lat": "37.0", "lon": "25.0"},   # out window
            {"mmsi": "4", "lat": "37.0", "lon": "25.0"},                                          # skipped (no time)
        ]
        recs, stats = mc.normalize_rows(rows, bbox=(24.9, 36.9, 26.1, 38.1),
                                        start="2026-07-14T00:00:00Z", end="2026-07-14T23:59:59Z")
        self.assertEqual([r["mmsi"] for r in recs], ["1"])
        self.assertEqual(stats["outside_area"], 1)
        self.assertEqual(stats["outside_window"], 1)
        self.assertEqual(stats["skipped"], 1)

    def test_resolve_bbox_conflict_and_none(self):
        class A:
            bbox = [24.9, 36.9, 26.1, 38.1]
            wdpa_id = wdpa_pid = None
            gpkg = None
            label = None
        self.assertEqual(mc.resolve_bbox(A())[0], (24.9, 36.9, 26.1, 38.1))
        A.wdpa_id = 5
        with self.assertRaises(RuntimeError):
            mc.resolve_bbox(A())
        A.bbox = None
        A.wdpa_id = None
        self.assertEqual(mc.resolve_bbox(A())[0], None)   # no AOI -> None (keep all)


class TestMainAndFuseRoundTrip(unittest.TestCase):
    def test_csv_to_contract_then_fuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv_path = write_csv(tmp / "AIS_2026_07_14.csv", [
                ["211476060", "2026-07-14T10:02:00", "37.0004", "25.0004", "0.3", "180", "NEPTUNE", "9012345"],
                ["636019888", "2026-07-14T10:01:00", "36.95", "24.95", "11.4", "65", "FAR", ""],
                ["999", "2026-07-14T10:00:00", "10.0", "10.0", "0", "0", "OUTSIDE", ""],  # outside AOI
            ])
            out = tmp / "ais.json"
            mc.main(["--csv", str(csv_path), "--bbox", "24.9", "36.9", "26.1", "38.1", "--out", str(out)])
            doc = json.loads(out.read_text())
            self.assertEqual(doc["source"], "marinecadastre")
            self.assertEqual(doc["position_count"], 2)          # the OUTSIDE row filtered out
            self.assertEqual({p["mmsi"] for p in doc["positions"]}, {"211476060", "636019888"})

            # the contract file drives fuse_violations end to end
            run = tmp / "run"
            shutil.copytree(FIXTURES / "run", run)
            fv.main(["--run", str(run), "--ais", str(out)])
            events = {e["detection_id"]: e for e in
                      json.loads((run / "violation_events.json").read_text())["violation_events"]}
            self.assertEqual(events["det_00000"]["ais_status"], "matched")   # sat on the NEPTUNE ping
            self.assertEqual(events["det_00000"]["mmsi"], "211476060")
            self.assertEqual(events["det_00001"]["ais_status"], "dark")

    def test_requires_csv(self):
        class _A:
            csv = None; date = None; bbox = [24.9, 36.9, 26.1, 38.1]
            wdpa_id = wdpa_pid = None; gpkg = None; label = None; start = end = out = None
        with self.assertRaises(RuntimeError):
            mc.main(["--bbox", "24.9", "36.9", "26.1", "38.1"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
