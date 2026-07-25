"""Offline tests for scripts/fuse_violations.py (ROADMAP Phase 3 verification cases).

Pure stdlib, no network / no ML deps - builds a synthetic run dir + AIS file in a temp dir and
asserts the dark/matched verdicts. Run either of:
    python scripts/tests/test_fuse_violations.py
    python -m unittest scripts.tests.test_fuse_violations
"""
import csv
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import fuse_violations as fv  # noqa: E402

SCENE_DT = "2026-07-14T10:00:00Z"


def make_run(tmp, detections, wdpa=None, provider="pc"):
    run = Path(tmp) / "run"
    run.mkdir(parents=True, exist_ok=True)
    for i, d in enumerate(detections):
        d.setdefault("detection_id", f"det_{i:05d}")
        d.setdefault("chip", "chips/chip_r00000_c00000.tif")
        d.setdefault("class", "boat")
    doc = {"run_dir": "run", "count": len(detections),
           "scene": {"item_id": "SYNTH", "datetime": SCENE_DT},
           "wdpa": wdpa, "detections": detections}
    (run / "detections.json").write_text(json.dumps(doc))
    (run / "manifest.json").write_text(json.dumps({"provider": provider, "collection": ""}))
    return run


def make_ais(tmp, pings):
    path = Path(tmp) / "ais.json"
    path.write_text(json.dumps({"positions": pings}))
    return path


def make_vessels(tmp, vessels):
    path = Path(tmp) / "vessels.json"
    path.write_text(json.dumps({"vessels": vessels}))
    return path


def make_ais_csv(tmp, rows, header=("mmsi", "lon", "lat", "timestamp", "sog", "cog")):
    path = Path(tmp) / "ais.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    return path


def run_fuse(run, ais, extra=None):
    argv = ["--run", str(run), "--ais", str(ais)] + (extra or [])
    fv.main(argv)
    return json.loads((run / "violation_events.json").read_text())


def by_detection(doc):
    return {e["detection_id"]: e for e in doc["violation_events"]}


class TestHelpers(unittest.TestCase):
    def test_haversine_known_distance(self):
        # 0.001 deg of latitude ~ 111.2 m
        d = fv.haversine_m(0.0, 0.0, 0.0, 0.001)
        self.assertAlmostEqual(d, 111.19, delta=0.5)

    def test_parse_timestamp_forms(self):
        z = fv.parse_timestamp("2026-07-14T10:00:00Z")
        off = fv.parse_timestamp("2026-07-14T12:00:00+02:00")
        naive = fv.parse_timestamp("2026-07-14 10:00:00")
        epoch = fv.parse_timestamp(z.timestamp())
        self.assertEqual(z, off)          # same instant, different notation
        self.assertEqual(z, naive)        # naive assumed UTC
        self.assertEqual(z, epoch)        # epoch seconds
        self.assertEqual(z.tzinfo, timezone.utc)


class TestVerificationCases(unittest.TestCase):
    """The three cases the roadmap requires, plus an infeasible-ping control."""

    def test_matched_must_not_flag_dark(self):
        # A detection sitting on a near-coincident AIS ping (~70 m, dt=0) must be MATCHED, not dark.
        run = make_run(self._tmp, [{"confidence": 0.9, "centroid_wgs84": [25.0, 37.0],
                                    "bbox_wgs84": [25.0, 37.0, 25.001, 37.001], "inside_mpa": True}])
        ais = make_ais(self._tmp, [{"mmsi": 111, "lon": 25.0005, "lat": 37.0005,
                                    "timestamp": SCENE_DT, "sog": 0.1, "cog": 90}])
        e = by_detection(run_fuse(run, ais))["det_00000"]
        self.assertEqual(e["ais_status"], "matched")
        self.assertEqual(e["mmsi"], "111")
        self.assertLess(e["ais_match"]["distance_m"], 500)

    def test_known_dark_must_flag(self):
        # No AIS ping anywhere near -> DARK.
        run = make_run(self._tmp, [{"confidence": 0.8, "centroid_wgs84": [25.1, 37.1],
                                    "bbox_wgs84": [25.1, 37.1, 25.101, 37.101], "inside_mpa": True}])
        ais = make_ais(self._tmp, [{"mmsi": 999, "lon": 20.0, "lat": 40.0,
                                    "timestamp": SCENE_DT, "sog": 5, "cog": 0}])
        e = by_detection(run_fuse(run, ais))["det_00000"]
        self.assertEqual(e["ais_status"], "dark")
        self.assertIsNone(e["mmsi"])
        self.assertIsNone(e["ais_match"])
        self.assertEqual(e["evidence"], ["Sentinel-2"])

    def test_fast_mover_20min_off_still_matches(self):
        # The velocity-gate crux: a ping 20 min before acquisition, ~10 km away, at 20 kn.
        # A fixed small radius would miss it; the feasible-movement envelope (12.8 km) catches it.
        run = make_run(self._tmp, [{"confidence": 0.7, "centroid_wgs84": [25.2, 37.0],
                                    "bbox_wgs84": [25.2, 37.0, 25.201, 37.001], "inside_mpa": True}])
        ais = make_ais(self._tmp, [{"mmsi": 333, "lon": 25.3125, "lat": 37.0,
                                    "timestamp": "2026-07-14T09:40:00Z", "sog": 20, "cog": 270}])
        e = by_detection(run_fuse(run, ais))["det_00000"]
        self.assertEqual(e["ais_status"], "matched")
        self.assertEqual(e["mmsi"], "333")
        self.assertAlmostEqual(e["ais_match"]["dt_seconds"], 1200.0, delta=1)
        self.assertGreater(e["ais_match"]["distance_m"], 9000)   # far beyond any fixed small radius
        self.assertLessEqual(e["ais_match"]["distance_m"], e["ais_match"]["reach_m"])

    def test_infeasible_ping_stays_dark(self):
        # A ping exists 5 min off but ~5 km away at only 5 kn (reach ~1.27 km) -> still DARK.
        run = make_run(self._tmp, [{"confidence": 0.6, "centroid_wgs84": [25.5, 37.5],
                                    "bbox_wgs84": [25.5, 37.5, 25.501, 37.501], "inside_mpa": True}])
        ais = make_ais(self._tmp, [{"mmsi": 444, "lon": 25.5, "lat": 37.545,  # ~5 km north
                                    "timestamp": "2026-07-14T09:55:00Z", "sog": 5, "cog": 180}])
        e = by_detection(run_fuse(run, ais))["det_00000"]
        self.assertEqual(e["ais_status"], "dark")

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self._tmp = self._td.name

    def tearDown(self):
        self._td.cleanup()


class TestScopingAndScoring(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self._tmp = self._td.name

    def tearDown(self):
        self._td.cleanup()

    def test_outside_mpa_excluded_by_default(self):
        run = make_run(self._tmp, [
            {"confidence": 0.9, "centroid_wgs84": [25.0, 37.0],
             "bbox_wgs84": [25.0, 37.0, 25.001, 37.001], "inside_mpa": False},
        ], wdpa={"WDPAID": 555, "NAME": "Test MPA"})
        ais = make_ais(self._tmp, [])
        doc = run_fuse(run, ais)
        self.assertEqual(doc["counts"]["in_scope"], 0)   # outside-MPA detection skipped
        # ...but --all-detections includes it
        doc2 = run_fuse(run, ais, ["--all-detections"])
        self.assertEqual(doc2["counts"]["in_scope"], 1)
        self.assertEqual(doc2["violation_events"][0]["mpa_name"], "Test MPA")

    def test_matched_weight_and_dark_only(self):
        run = make_run(self._tmp, [
            {"confidence": 0.8, "centroid_wgs84": [25.0, 37.0],
             "bbox_wgs84": [25.0, 37.0, 25.001, 37.001], "inside_mpa": True},   # will match
            {"confidence": 0.8, "centroid_wgs84": [25.1, 37.1],
             "bbox_wgs84": [25.1, 37.1, 25.101, 37.101], "inside_mpa": True},   # dark
        ])
        ais = make_ais(self._tmp, [{"mmsi": 111, "lon": 25.0, "lat": 37.0,
                                    "timestamp": SCENE_DT, "sog": 0.1, "cog": 0}])
        doc = run_fuse(run, ais, ["--matched-weight", "0.25"])
        events = {e["ais_status"]: e for e in doc["violation_events"]}
        self.assertAlmostEqual(events["dark"]["violation_score"], 0.8, places=4)
        self.assertAlmostEqual(events["matched"]["violation_score"], 0.2, places=4)  # 0.8 * 0.25
        # dark-only drops the matched event
        doc2 = run_fuse(run, ais, ["--dark-only"])
        self.assertEqual(doc2["counts"]["emitted"], 1)
        self.assertTrue(all(e["ais_status"] == "dark" for e in doc2["violation_events"]))

    def test_geojson_has_only_dark_points(self):
        run = make_run(self._tmp, [
            {"confidence": 0.8, "centroid_wgs84": [25.0, 37.0],
             "bbox_wgs84": [25.0, 37.0, 25.001, 37.001], "inside_mpa": True},   # match
            {"confidence": 0.8, "centroid_wgs84": [25.1, 37.1],
             "bbox_wgs84": [25.1, 37.1, 25.101, 37.101], "inside_mpa": True},   # dark
        ])
        ais = make_ais(self._tmp, [{"mmsi": 111, "lon": 25.0, "lat": 37.0,
                                    "timestamp": SCENE_DT, "sog": 0.1, "cog": 0}])
        run_fuse(run, ais)
        gj = json.loads((run / "violation_events.geojson").read_text())
        self.assertEqual(len(gj["features"]), 1)
        self.assertEqual(gj["features"][0]["properties"]["ais_status"], "dark")


class TestVesselEnrichment(unittest.TestCase):
    """The optional --vessels (GFW) enrichment join, ROADMAP Phase 3 step 3. A matched detection
    at [25.0,37.0] on MMSI 111 + a dark detection at [25.1,37.1]."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self._tmp = self._td.name
        self._run = make_run(self._tmp, [
            {"confidence": 0.8, "centroid_wgs84": [25.0, 37.0],
             "bbox_wgs84": [25.0, 37.0, 25.001, 37.001], "inside_mpa": True},   # matches 111
            {"confidence": 0.8, "centroid_wgs84": [25.1, 37.1],
             "bbox_wgs84": [25.1, 37.1, 25.101, 37.101], "inside_mpa": True},   # dark
        ], wdpa={"WDPAID": 555, "NAME": "Test MPA"})
        self._ais = make_ais(self._tmp, [{"mmsi": 111, "lon": 25.0, "lat": 37.0,
                                          "timestamp": SCENE_DT, "sog": 0.1, "cog": 0}])

    def tearDown(self):
        self._td.cleanup()

    def _events(self, vessels, extra=None):
        vf = make_vessels(self._tmp, vessels)
        doc = run_fuse(self._run, self._ais, ["--vessels", str(vf)] + (extra or []))
        return {e["detection_id"]: e for e in doc["violation_events"]}, doc

    def test_fishing_vessel_scores_up_and_identity_filled(self):
        ev, _ = self._events([{"mmsi": "111", "vessel_id": "gfw-abc", "shipname": "MARIA",
                               "flag": "ESP", "geartype": "trawlers", "imo": "1234567",
                               "fishing_hours": 3.5, "fishing_probability": 1.0, "iuu_listed": False}])
        m = ev["det_00000"]
        self.assertEqual(m["ais_status"], "matched")
        self.assertEqual(m["vessel_name"], "MARIA")
        self.assertEqual(m["flag"], "ESP")
        self.assertEqual(m["gear_type"], "trawlers")
        self.assertEqual(m["imo"], "1234567")
        self.assertEqual(m["fishing_hours"], 3.5)
        self.assertEqual(m["gfw_vessel_id"], "gfw-abc")
        self.assertIn("GFW", m["evidence"])
        self.assertAlmostEqual(m["violation_score"], 0.8, places=4)   # fishing prob 1.0 -> full weight

    def test_non_fishing_vessel_keeps_matched_floor(self):
        ev, _ = self._events([{"mmsi": "111", "shipname": "IDLE", "fishing_probability": 0.0}])
        self.assertAlmostEqual(ev["det_00000"]["violation_score"], 0.2, places=4)  # 0.8 * matched_weight

    def test_partial_probability_interpolates(self):
        ev, _ = self._events([{"mmsi": "111", "fishing_probability": 0.5}])
        # weight = 0.25 + (1.0-0.25)*0.5 = 0.625 ; score = 0.8 * 0.625 = 0.5
        self.assertAlmostEqual(ev["det_00000"]["violation_score"], 0.5, places=4)

    def test_iuu_escalates_even_without_fishing(self):
        ev, _ = self._events([{"mmsi": "111", "fishing_probability": 0.0, "iuu_listed": True}])
        self.assertAlmostEqual(ev["det_00000"]["violation_score"], 0.8, places=4)  # escalated to full
        self.assertTrue(ev["det_00000"]["iuu_listed"])

    def test_dark_event_unaffected_by_vessels(self):
        ev, doc = self._events([{"mmsi": "111", "fishing_probability": 1.0}])
        d = ev["det_00001"]
        self.assertEqual(d["ais_status"], "dark")
        self.assertAlmostEqual(d["violation_score"], 0.8, places=4)   # dark = presence, unchanged
        self.assertIsNone(d["fishing_probability"])
        self.assertEqual(doc["counts"]["gfw_enriched"], 1)

    def test_no_matching_record_is_noop(self):
        # vessels file has a different MMSI -> matched event stays at the AIS-only floor, no identity
        ev, _ = self._events([{"mmsi": "999", "shipname": "OTHER", "fishing_probability": 1.0}])
        m = ev["det_00000"]
        self.assertAlmostEqual(m["violation_score"], 0.2, places=4)
        self.assertIsNone(m["vessel_name"])
        self.assertNotIn("GFW", m["evidence"])


class TestAisInputAndEdges(unittest.TestCase):
    """The AIS input contract (CSV, field aliases, bad rows) + the velocity-gate fallback and edge
    cases — the paths the four verification cases don't isolate."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self._tmp = self._td.name

    def tearDown(self):
        self._td.cleanup()

    def _one_det(self, ais_path, conf=0.9, centroid=(25.0, 37.0), extra=None):
        run = make_run(self._tmp, [{"confidence": conf, "centroid_wgs84": list(centroid),
                                    "bbox_wgs84": [centroid[0], centroid[1], centroid[0] + 0.001, centroid[1] + 0.001],
                                    "inside_mpa": True}])
        doc = run_fuse(run, ais_path, extra)
        return doc, by_detection(doc)

    def test_csv_ais_input(self):
        # The documented --ais .csv contract (header row), not just JSON.
        csvp = make_ais_csv(self._tmp, [["111", "25.0005", "37.0005", "2026-07-14T10:00:00Z", "0.1", "90"]])
        _, ev = self._one_det(csvp)
        self.assertEqual(ev["det_00000"]["ais_status"], "matched")
        self.assertEqual(ev["det_00000"]["mmsi"], "111")

    def test_missing_sog_uses_max_speed_fallback(self):
        # Ping ~5 km away, 10 min before acquisition, with NO sog -> matches ONLY via the 25 kn
        # max-speed envelope (a fixed small radius, or sog=0, would miss it). This isolates the fallback.
        ais = make_ais(self._tmp, [{"mmsi": 222, "lon": 25.05625, "lat": 37.0,
                                    "timestamp": "2026-07-14T09:50:00Z"}])   # sog omitted
        _, ev = self._one_det(ais)
        m = ev["det_00000"]
        self.assertEqual(m["ais_status"], "matched")
        self.assertIsNone(m["ais_match"]["sog"])                 # ping carried no sog
        self.assertGreater(m["ais_match"]["distance_m"], 4000)   # far beyond the 500 m slack — only speed*dt reaches

    def test_missing_sog_beyond_reach_is_dark(self):
        # Same ping but ~12 km away -> beyond the 25 kn / 10 min feasible envelope -> dark.
        ais = make_ais(self._tmp, [{"mmsi": 222, "lon": 25.135, "lat": 37.0,
                                    "timestamp": "2026-07-14T09:50:00Z"}])
        _, ev = self._one_det(ais)
        self.assertEqual(ev["det_00000"]["ais_status"], "dark")

    def test_field_aliases_and_skips_bad_rows(self):
        # Contract tolerates MMSI/longitude/latitude/time_utc/Sog; an unusable row is skipped, not fatal.
        ais = make_ais(self._tmp, [
            {"MMSI": 333, "longitude": 25.0005, "latitude": 37.0005,
             "time_utc": "2026-07-14T10:00:00Z", "Sog": 0.1},
            {"foo": 1}])
        _, ev = self._one_det(ais)
        self.assertEqual(ev["det_00000"]["ais_status"], "matched")
        self.assertEqual(ev["det_00000"]["mmsi"], "333")

    def test_min_conf_drops_low_confidence(self):
        run = make_run(self._tmp, [
            {"confidence": 0.9, "centroid_wgs84": [25.0, 37.0],
             "bbox_wgs84": [25.0, 37.0, 25.001, 37.001], "inside_mpa": True},
            {"confidence": 0.1, "centroid_wgs84": [25.2, 37.2],
             "bbox_wgs84": [25.2, 37.2, 25.201, 37.201], "inside_mpa": True}])
        doc = run_fuse(run, make_ais(self._tmp, []), ["--min-conf", "0.5"])
        self.assertEqual(doc["counts"]["in_scope"], 1)   # the 0.1 detection dropped before fusing

    def test_empty_ais_all_dark(self):
        doc, ev = self._one_det(make_ais(self._tmp, []))
        self.assertEqual(ev["det_00000"]["ais_status"], "dark")
        self.assertAlmostEqual(ev["det_00000"]["violation_score"], 0.9, places=4)

    def test_missing_scene_datetime_raises(self):
        run = Path(self._tmp) / "run_nodt"
        run.mkdir()
        (run / "detections.json").write_text(json.dumps({
            "count": 1, "scene": {}, "wdpa": None,
            "detections": [{"detection_id": "det_00000", "confidence": 0.9, "centroid_wgs84": [25, 37],
                            "bbox_wgs84": [25, 37, 25.001, 37.001], "inside_mpa": True, "chip": "c.tif"}]}))
        (run / "manifest.json").write_text(json.dumps({"provider": "pc"}))
        with self.assertRaises(RuntimeError):
            fv.main(["--run", str(run), "--ais", str(make_ais(self._tmp, []))])


if __name__ == "__main__":
    unittest.main(verbosity=2)
