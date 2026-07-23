"""Offline tests for scripts/gfw_vessels.py (ROADMAP Phase 3 step 3).

Pure stdlib, no token / no network - exercises the GFW response NORMALIZERS (the part correctness
depends on) with synthetic GFW-shaped payloads, the MMSI collectors, and a normalize -> vessel
record -> fuse_violations enrichment round trip. The HTTP layer is a thin shell and is not tested
here (it needs a live token). Run either of:
    python scripts/tests/test_gfw_vessels.py
    python -m unittest scripts.tests.test_gfw_vessels
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import gfw_vessels as gv  # noqa: E402
import fuse_violations as fv  # noqa: E402


class TestMmsiCollectors(unittest.TestCase):
    def test_from_events_takes_matched_only(self):
        doc = {"violation_events": [
            {"detection_id": "e0", "ais_status": "matched", "mmsi": "111"},
            {"detection_id": "e1", "ais_status": "dark", "mmsi": None},
            {"detection_id": "e2", "ais_status": "matched", "mmsi": "222"},
        ]}
        self.assertEqual(gv.mmsis_from_events(doc), ["111", "222"])

    def test_from_ais_takes_all(self):
        doc = {"positions": [{"mmsi": "111"}, {"mmsi": 222}, {"lon": 1}]}
        self.assertEqual(gv.mmsis_from_ais(doc), ["111", "222"])

    def test_collect_dedups_and_orders(self):
        class A:
            mmsi = "111, 222, 111"
            from_events = None
            from_ais = None
        self.assertEqual(gv.collect_mmsis(A()), ["111", "222"])


class TestIdentity(unittest.TestCase):
    def test_self_reported_info(self):
        resp = {"entries": [{"selfReportedInfo": [
            {"vesselId": "gfw-abc", "ssvid": "368207620", "shipname": "MARIA", "flag": "ESP",
             "callsign": "EA1234", "imo": "1234567", "geartypes": ["trawlers"]}], "registryInfo": []}]}
        idv = gv.normalize_identity(resp, "368207620")
        self.assertEqual(idv["vessel_id"], "gfw-abc")
        self.assertEqual(idv["shipname"], "MARIA")
        self.assertEqual(idv["flag"], "ESP")
        self.assertEqual(idv["geartype"], "trawlers")
        self.assertEqual(idv["imo"], "1234567")
        self.assertEqual(idv["callsign"], "EA1234")

    def test_picks_entry_matching_the_mmsi(self):
        resp = {"entries": [{"selfReportedInfo": [
            {"vesselId": "v-A", "ssvid": "111", "shipname": "A"},
            {"vesselId": "v-B", "ssvid": "222", "shipname": "B"}]}]}
        self.assertEqual(gv.normalize_identity(resp, "222")["shipname"], "B")

    def test_geartype_dict_shape(self):
        resp = {"entries": [{"selfReportedInfo": [
            {"ssvid": "111", "geartypes": [{"name": "drifting_longlines"}]}]}]}
        self.assertEqual(gv.normalize_identity(resp, "111")["geartype"], "drifting_longlines")

    def test_empty_response_yields_stub(self):
        idv = gv.normalize_identity({"entries": []}, "111")
        self.assertEqual(idv["mmsi"], "111")
        self.assertIsNone(idv["vessel_id"])
        self.assertIsNone(idv["shipname"])


class TestFishing(unittest.TestCase):
    def test_sum_durations_and_probability(self):
        resp = {"entries": [
            {"type": "fishing", "start": "2026-07-14T08:00:00Z", "end": "2026-07-14T10:30:00Z"},  # 2.5h
            {"type": "fishing", "durationHours": 1.5},                                             # 1.5h
            {"type": "port_visit", "start": "2026-07-14T00:00:00Z", "end": "2026-07-14T06:00:00Z"},  # ignored
        ]}
        s = gv.summarize_fishing(resp)
        self.assertAlmostEqual(s["fishing_hours"], 4.0, places=3)
        self.assertEqual(s["fishing_event_count"], 2)
        self.assertEqual(s["fishing_probability"], 1.0)

    def test_no_fishing_is_zero_not_none(self):
        s = gv.summarize_fishing({"entries": []})
        self.assertEqual(s["fishing_hours"], 0.0)
        self.assertEqual(s["fishing_probability"], 0.0)

    def test_nested_time_shape(self):
        resp = {"entries": [{"type": "fishing",
                             "start": {"timestamp": "2026-07-14T08:00:00Z"},
                             "end": {"timestamp": "2026-07-14T09:00:00Z"}}]}
        self.assertAlmostEqual(gv.summarize_fishing(resp)["fishing_hours"], 1.0, places=3)


class TestInsights(unittest.TestCase):
    def test_iuu_nested_true(self):
        resp = {"vessels": [{"vesselId": "gfw-abc",
                             "iuuIndicator": {"valueInThePeriod": True},
                             "gap": {"aisOff": [{"x": 1}, {"x": 2}]},
                             "apparentFishing": {"eventsInNoTakeMpas": 3}}]}
        ins = gv.normalize_insights(resp, vessel_id="gfw-abc")
        self.assertTrue(ins["iuu_listed"])
        self.assertEqual(ins["ais_off_events"], 2)
        self.assertEqual(ins["fishing_in_no_take_mpa"], 3)

    def test_iuu_false_and_missing(self):
        resp = {"vessels": [{"vesselId": "gfw-abc",
                             "iuuIndicator": {"valuesInThePeriod": [{"result": "false"}]}}]}
        ins = gv.normalize_insights(resp, vessel_id="gfw-abc")
        self.assertIsNone(ins["iuu_listed"])          # nothing truthy -> not found
        self.assertIsNone(ins["ais_off_events"])
        self.assertIsNone(ins["fishing_in_no_take_mpa"])


class TestRegionFiltering(unittest.TestCase):
    """MPA-specific fishing_probability (PHASE3_PLAN 3A) — client-side position filtering."""

    def test_point_in_bbox(self):
        bbox = (24.9, 36.9, 26.1, 38.1)
        self.assertTrue(gv.point_in_bbox(25.0, 37.0, bbox))
        self.assertFalse(gv.point_in_bbox(30.0, 37.0, bbox))

    def test_event_position_shapes(self):
        self.assertEqual(gv.event_position({"position": {"lat": 37.0, "lon": 25.0}}), (25.0, 37.0))
        self.assertEqual(gv.event_position({"lat": 37.0, "lon": 25.0}), (25.0, 37.0))
        self.assertIsNone(gv.event_position({"type": "fishing"}))   # no position -> None

    def test_region_filter_keeps_only_in_mpa_events(self):
        bbox = (24.9, 36.9, 26.1, 38.1)
        resp = {"entries": [
            {"type": "fishing", "durationHours": 2.0, "position": {"lat": 37.0, "lon": 25.0}},   # in
            {"type": "fishing", "durationHours": 5.0, "position": {"lat": 10.0, "lon": 10.0}},   # out
        ]}
        s = gv.summarize_fishing(resp, region_bbox=bbox)
        self.assertTrue(s["region_filtered"])
        self.assertEqual(s["fishing_event_count"], 1)
        self.assertAlmostEqual(s["fishing_hours"], 2.0, places=3)
        self.assertEqual(s["fishing_probability"], 1.0)
        # a vessel that only fished OUTSIDE the MPA -> probability 0 under region filtering
        out_only = gv.summarize_fishing({"entries": [
            {"type": "fishing", "durationHours": 5.0, "position": {"lat": 10.0, "lon": 10.0}}]}, region_bbox=bbox)
        self.assertEqual(out_only["fishing_probability"], 0.0)

    def test_region_filter_falls_back_when_no_positions(self):
        # region requested but events carry no position -> skip filter (degrade to window prior)
        resp = {"entries": [{"type": "fishing", "durationHours": 3.0}]}
        s = gv.summarize_fishing(resp, region_bbox=(24.9, 36.9, 26.1, 38.1))
        self.assertFalse(s["region_filtered"])
        self.assertEqual(s["fishing_probability"], 1.0)
        self.assertAlmostEqual(s["fishing_hours"], 3.0, places=3)

    def test_no_region_is_unchanged(self):
        resp = {"entries": [{"type": "fishing", "durationHours": 1.0, "position": {"lat": 0, "lon": 0}}]}
        s = gv.summarize_fishing(resp)   # no region_bbox
        self.assertFalse(s["region_filtered"])
        self.assertEqual(s["fishing_event_count"], 1)

    def test_resolve_region_bbox_and_conflict(self):
        class A:
            region_bbox = [24.9, 36.9, 26.1, 38.1]
            wdpa_id = wdpa_pid = None
            gpkg = None
        bbox, label = gv.resolve_region_bbox(A())
        self.assertEqual(bbox, (24.9, 36.9, 26.1, 38.1))
        self.assertEqual(label, "bbox")
        A.wdpa_id = 555
        with self.assertRaises(RuntimeError):   # both provided
            gv.resolve_region_bbox(A())

    def test_resolve_region_bbox_none(self):
        class A:
            region_bbox = None
            wdpa_id = wdpa_pid = None
            gpkg = None
        self.assertEqual(gv.resolve_region_bbox(A()), (None, None))


class TestWindowAndRecord(unittest.TestCase):
    def test_resolve_window_defaults_7_days(self):
        start, end = gv.resolve_window(None, "2026-07-22")
        self.assertEqual(end, "2026-07-22")
        self.assertEqual(start, "2026-07-15")

    def test_build_record_merges(self):
        rec = gv.build_vessel_record(
            {"mmsi": "111", "vessel_id": "v", "shipname": "N", "flag": "F",
             "geartype": "trawlers", "imo": "9", "callsign": "C"},
            {"fishing_hours": 2.0, "fishing_event_count": 1, "fishing_probability": 1.0},
            {"iuu_listed": True, "ais_off_events": 1, "fishing_in_no_take_mpa": 2})
        self.assertEqual(rec["mmsi"], "111")
        self.assertEqual(rec["fishing_hours"], 2.0)
        self.assertTrue(rec["iuu_listed"])
        self.assertEqual(rec["source"], "gfw")

    def test_build_record_without_insights(self):
        rec = gv.build_vessel_record({"mmsi": "111", "vessel_id": None, "shipname": None,
                                      "flag": None, "geartype": None, "imo": None, "callsign": None},
                                     {"fishing_hours": None, "fishing_event_count": None,
                                      "fishing_probability": None}, None)
        self.assertNotIn("iuu_listed", rec)   # insights omitted entirely when not requested


class TestEndToEndIntoFuse(unittest.TestCase):
    """Normalize synthetic GFW payloads -> vessel record -> vessels file -> fuse enrichment."""

    def test_gfw_record_enriches_matched_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # Build a real vessel record from the pure normalizers (as gfw_vessels.main would).
            identity = gv.normalize_identity(
                {"entries": [{"selfReportedInfo": [
                    {"vesselId": "gfw-abc", "ssvid": "111", "shipname": "MARIA", "flag": "ESP",
                     "geartypes": ["trawlers"], "imo": "1234567"}]}]}, "111")
            fishing = gv.summarize_fishing(
                {"entries": [{"type": "fishing", "durationHours": 4.0}]})
            record = gv.build_vessel_record(identity, fishing, None)
            (tmp / "vessels.json").write_text(json.dumps({"vessels": [record]}))

            run = tmp / "run"
            run.mkdir()
            (run / "detections.json").write_text(json.dumps({
                "count": 1, "scene": {"datetime": "2026-07-14T10:00:00Z"},
                "wdpa": {"WDPAID": 555, "NAME": "Test MPA"},
                "detections": [{"detection_id": "det_00000", "confidence": 0.8,
                                "centroid_wgs84": [25.0, 37.0],
                                "bbox_wgs84": [25.0, 37.0, 25.001, 37.001],
                                "inside_mpa": True, "chip": "c.tif"}]}))
            (run / "manifest.json").write_text(json.dumps({"provider": "pc"}))
            (tmp / "ais.json").write_text(json.dumps({"positions": [
                {"mmsi": 111, "lon": 25.0, "lat": 37.0, "timestamp": "2026-07-14T10:00:00Z", "sog": 0.1}]}))

            fv.main(["--run", str(run), "--ais", str(tmp / "ais.json"),
                     "--vessels", str(tmp / "vessels.json")])
            e = json.loads((run / "violation_events.json").read_text())["violation_events"][0]
            self.assertEqual(e["ais_status"], "matched")
            self.assertEqual(e["vessel_name"], "MARIA")
            self.assertEqual(e["gear_type"], "trawlers")
            self.assertEqual(e["fishing_hours"], 4.0)
            self.assertAlmostEqual(e["violation_score"], 0.8, places=4)
            self.assertIn("GFW", e["evidence"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
