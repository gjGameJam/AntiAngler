"""Offline tests for scripts/geofence_ais.py (AIS-track MPA geofencing — the "reported" status).

Pure stdlib, no network / key / geopandas. The point-in-polygon geometry is deliberately dependency-
free so it is testable here; only the WDPA GeoPackage *lookup* needs geopandas and is not exercised.
Covers: ray-casting correctness (incl. holes and concave shapes), the presence gate, the behaviour-
driven scoring tiers, and the contract guarantee that report_violations.py can rank the output.
Run either of:
    python scripts/tests/test_geofence_ais.py
    python -m unittest scripts.tests.test_geofence_ais
"""
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import geofence_ais as ga  # noqa: E402

T0 = datetime(2026, 7, 14, 10, 0, 0, tzinfo=timezone.utc)

# A 1x1 degree square MPA at the origin, used by most tests.
SQUARE = [{"exterior": [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], "holes": []}]
MPA = {"WDPAID": 12345.0, "WDPA_PID": "12345", "NAME": "Test Reserve"}


def ping(mmsi, lon, lat, minutes, sog=None, name=None):
    return {"mmsi": mmsi, "lon": lon, "lat": lat,
            "timestamp": (T0 + timedelta(minutes=minutes)).isoformat().replace("+00:00", "Z"),
            "sog": sog, "name": name}


def args(**over):
    argv = ["--ais", "x.json", "--bbox", "0", "0", "1", "1"]
    for key, val in over.items():
        argv += [f"--{key.replace('_', '-')}", str(val)]
    return ga.parse_args(argv)


def track(raw):
    return [ga.normalize_ping(p) for p in raw]


class TestRayCasting(unittest.TestCase):
    def test_inside_and_outside_a_square(self):
        ring = SQUARE[0]["exterior"]
        self.assertTrue(ga.point_in_ring(0.5, 0.5, ring))
        self.assertFalse(ga.point_in_ring(1.5, 0.5, ring))   # east
        self.assertFalse(ga.point_in_ring(-0.5, 0.5, ring))  # west
        self.assertFalse(ga.point_in_ring(0.5, 1.5, ring))   # north
        self.assertFalse(ga.point_in_ring(0.5, -0.5, ring))  # south

    def test_concave_shape(self):
        """A C-shape: the notch must read as OUTSIDE even though it is within the bounding box —
        this is exactly what a bbox test would get wrong on a real coastline."""
        c_shape = [(0, 0), (3, 0), (3, 1), (1, 1), (1, 2), (3, 2), (3, 3), (0, 3)]
        self.assertTrue(ga.point_in_ring(0.5, 1.5, c_shape))    # in the spine
        self.assertFalse(ga.point_in_ring(2.0, 1.5, c_shape))   # in the notch

    def test_hole_is_excluded(self):
        donut = [{"exterior": [(0, 0), (4, 0), (4, 4), (0, 4)],
                  "holes": [[(1, 1), (3, 1), (3, 3), (1, 3)]]}]
        self.assertTrue(ga.point_in_polygon(0.5, 0.5, donut[0]))   # in the ring
        self.assertFalse(ga.point_in_polygon(2.0, 2.0, donut[0]))  # in the hole

    def test_multipolygon_any_part(self):
        two = [{"exterior": [(0, 0), (1, 0), (1, 1), (0, 1)], "holes": []},
               {"exterior": [(5, 5), (6, 5), (6, 6), (5, 6)], "holes": []}]
        self.assertTrue(ga.point_in_any(0.5, 0.5, two))
        self.assertTrue(ga.point_in_any(5.5, 5.5, two))
        self.assertFalse(ga.point_in_any(3.0, 3.0, two))

    def test_closed_ring_with_repeated_first_vertex(self):
        """Shapely emits closed rings (last == first); the wraparound edge must not double-count."""
        closed = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
        self.assertTrue(ga.point_in_ring(0.5, 0.5, closed))
        self.assertFalse(ga.point_in_ring(1.5, 0.5, closed))

    def test_bbox_polygon_matches_ring_semantics(self):
        poly = ga.bbox_polygon((0.0, 0.0, 1.0, 1.0))
        self.assertTrue(ga.point_in_any(0.5, 0.5, poly))
        self.assertFalse(ga.point_in_any(1.5, 0.5, poly))


class TestPresenceGate(unittest.TestCase):
    def test_only_vessels_inside_are_reported(self):
        pings = track([ping("inside", 0.5, 0.5, 0), ping("inside", 0.6, 0.5, 10),
                       ping("outside", 5.0, 5.0, 0), ping("outside", 5.1, 5.0, 10)])
        events = ga.geofence(pings, SQUARE, MPA, args())
        self.assertEqual([e["mmsi"] for e in events], ["inside"])

    def test_partial_track_counts_only_inside_pings(self):
        """A vessel that enters and leaves: entry/exit bracket the IN-MPA pings, not the whole track."""
        pings = track([ping("1", 5.0, 5.0, 0),      # outside, before
                       ping("1", 0.5, 0.5, 30),     # inside
                       ping("1", 0.6, 0.6, 90),     # inside
                       ping("1", 5.0, 5.0, 120)])   # outside, after
        event = ga.geofence(pings, SQUARE, MPA, args())[0]
        self.assertEqual(event["ais_match"]["pings_inside"], 2)
        self.assertEqual(event["entry_time"], "2026-07-14T10:30:00Z")
        self.assertEqual(event["exit_time"], "2026-07-14T11:30:00Z")
        self.assertAlmostEqual(event["duration_hours"], 1.0, places=4)

    def test_min_pings_inside_suppresses_boundary_jitter(self):
        pings = track([ping("1", 0.5, 0.5, 0), ping("2", 0.5, 0.5, 0), ping("2", 0.6, 0.5, 10)])
        self.assertEqual(len(ga.geofence(pings, SQUARE, MPA, args())), 2)
        strict = ga.geofence(pings, SQUARE, MPA, args(min_pings_inside=2))
        self.assertEqual([e["mmsi"] for e in strict], ["2"])


class TestScoringTiers(unittest.TestCase):
    """Presence is certain; BEHAVIOUR sets the weight. This is the decision the whole script hangs on."""

    def _event(self, vessel=None, **over):
        pings = track([ping("1", 0.5, 0.5, 0), ping("1", 0.6, 0.6, 30)])
        vessels = {"1": vessel} if vessel else {}
        return ga.geofence(pings, SQUARE, MPA, args(**over), vessels)[0]

    def test_no_behaviour_evidence_sits_at_the_transit_floor(self):
        event = self._event()
        self.assertAlmostEqual(event["violation_score"], 0.10, places=4)
        self.assertIsNone(event["fishing_probability"])

    def test_null_probability_is_treated_as_no_evidence_not_zero(self):
        """ais_fishing returns null for a thin track; that must not be read as 'definitely transiting'
        any differently from having no record at all — both sit at the floor."""
        self.assertAlmostEqual(
            self._event({"mmsi": "1", "fishing_probability": None})["violation_score"], 0.10, places=4)

    def test_fishing_like_movement_scores_up(self):
        event = self._event({"mmsi": "1", "fishing_probability": 1.0, "source": "ais-heuristic"})
        self.assertAlmostEqual(event["violation_score"], 1.0, places=4)

    def test_probability_interpolates(self):
        # w = 0.10 + (1.0-0.10)*0.5 = 0.55 ; presence 1.0
        event = self._event({"mmsi": "1", "fishing_probability": 0.5})
        self.assertAlmostEqual(event["violation_score"], 0.55, places=4)

    def test_iuu_escalates_even_without_fishing(self):
        event = self._event({"mmsi": "1", "fishing_probability": 0.0, "iuu_listed": True})
        self.assertAlmostEqual(event["violation_score"], 1.0, places=4)
        self.assertTrue(event["iuu_listed"])

    def test_transit_weight_is_tunable(self):
        self.assertAlmostEqual(self._event(transit_weight=0.0)["violation_score"], 0.0, places=4)

    def test_events_ranked_and_min_score_filters(self):
        pings = track([ping("transit", 0.5, 0.5, 0), ping("transit", 0.6, 0.5, 30),
                       ping("fishing", 0.7, 0.7, 0), ping("fishing", 0.71, 0.7, 30)])
        vessels = {"fishing": {"mmsi": "fishing", "fishing_probability": 1.0}}
        events = ga.geofence(pings, SQUARE, MPA, args(), vessels)
        self.assertEqual([e["mmsi"] for e in events], ["fishing", "transit"])
        self.assertEqual([e["event_id"] for e in events], ["evt_00000", "evt_00001"])
        top = ga.geofence(pings, SQUARE, MPA, args(min_score=0.5), vessels)
        self.assertEqual([e["mmsi"] for e in top], ["fishing"])


class TestEventShape(unittest.TestCase):
    def test_matches_the_violation_events_schema(self):
        vessels = {"1": {"mmsi": "1", "shipname": "MARIA", "flag": "ESP", "geartype": "trawlers",
                         "imo": "1234567", "fishing_hours": 2.0, "fishing_probability": 0.8,
                         "source": "ais-heuristic"}}
        pings = track([ping("1", 0.5, 0.5, 0, sog=2.0), ping("1", 0.6, 0.6, 30, sog=2.5)])
        event = ga.geofence(pings, SQUARE, MPA, args(), vessels)[0]
        self.assertEqual(event["ais_status"], "reported")
        self.assertIsNone(event["detection_id"])     # no imagery backs this event
        self.assertIsNone(event["chip"])
        self.assertEqual(event["presence_confidence"], 1.0)
        self.assertEqual(event["mpa_id"], 12345.0)
        self.assertEqual(event["mpa_name"], "Test Reserve")
        self.assertEqual(event["vessel_name"], "MARIA")
        self.assertEqual(event["flag"], "ESP")
        # duration IS derivable here, unlike the instantaneous satellite path
        self.assertIsNotNone(event["duration_hours"])

    def test_evidence_never_misattributes_the_fishing_source(self):
        pings = track([ping("1", 0.5, 0.5, 0), ping("1", 0.6, 0.6, 30)])
        heur = ga.geofence(pings, SQUARE, MPA, args(),
                           {"1": {"mmsi": "1", "fishing_probability": 0.9,
                                  "source": "ais-heuristic"}})[0]
        self.assertEqual(heur["evidence"], ["AIS", "AIS-heuristic"])
        gfw = ga.geofence(pings, SQUARE, MPA, args(),
                          {"1": {"mmsi": "1", "fishing_probability": 0.9, "source": "gfw"}})[0]
        self.assertEqual(gfw["evidence"], ["AIS", "GFW"])
        bare = ga.geofence(pings, SQUARE, MPA, args())[0]
        self.assertEqual(bare["evidence"], ["AIS"])

    def test_vessel_name_falls_back_to_the_ais_name(self):
        pings = track([ping("1", 0.5, 0.5, 0, name="SEA STAR"), ping("1", 0.6, 0.6, 30)])
        self.assertEqual(ga.geofence(pings, SQUARE, MPA, args())[0]["vessel_name"], "SEA STAR")


class TestEndToEnd(unittest.TestCase):
    def test_output_is_rankable_by_report_violations(self):
        """The contract that justifies reusing the schema: the existing report reads it unchanged."""
        import report_violations as rv
        positions = [ping("1", 0.5, 0.5, 0, sog=2.0), ping("1", 0.6, 0.6, 30, sog=2.0),
                     ping("2", 9.0, 9.0, 0, sog=12.0)]
        with tempfile.TemporaryDirectory() as td:
            ais_path = Path(td) / "ais.json"
            ais_path.write_text(json.dumps({"positions": positions}), encoding="utf-8")
            out_path = Path(td) / "geofence.json"
            rc = ga.main(["--ais", str(ais_path), "--bbox", "0", "0", "1", "1",
                          "--out", str(out_path)])
            self.assertEqual(rc, 0)

            doc = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(doc["counts"]["reported"], 1)     # vessel 2 is outside
            self.assertEqual(doc["sensor"], "AIS")
            self.assertTrue(out_path.with_suffix(".geojson").exists())

            rows = rv.collect_rows([out_path])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["ais_status"], "reported")
            self.assertEqual(rows[0]["mmsi"], "1")
            # --dark-only correctly excludes cooperative vessels
            self.assertEqual(rv.collect_rows([out_path], dark_only=True), [])

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            ais_path = Path(td) / "ais.json"
            ais_path.write_text(json.dumps([ping("1", 0.5, 0.5, 0)]), encoding="utf-8")
            out_path = Path(td) / "nope.json"
            self.assertEqual(ga.main(["--ais", str(ais_path), "--bbox", "0", "0", "1", "1",
                                      "--out", str(out_path), "--dry-run"]), 0)
            self.assertFalse(out_path.exists())


class TestArgValidation(unittest.TestCase):
    def test_requires_an_area(self):
        with self.assertRaises(RuntimeError):
            ga.main(["--ais", "x.json"])

    def test_rejects_both_area_forms(self):
        with self.assertRaises(RuntimeError):
            ga.resolve_area(ga.parse_args(["--ais", "x.json", "--wdpa-id", "1",
                                           "--bbox", "0", "0", "1", "1"]))

    def test_rejects_inverted_bbox(self):
        with self.assertRaises(RuntimeError):
            ga.resolve_area(ga.parse_args(["--ais", "x.json", "--bbox", "1", "1", "0", "0"]))

    def test_rejects_transit_weight_above_fishing_weight(self):
        with tempfile.TemporaryDirectory() as td:
            ais_path = Path(td) / "ais.json"
            ais_path.write_text(json.dumps([ping("1", 0.5, 0.5, 0)]), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                ga.main(["--ais", str(ais_path), "--bbox", "0", "0", "1", "1",
                         "--transit-weight", "2.0", "--fishing-weight", "1.0"])

    def test_requires_ais(self):
        with self.assertRaises(RuntimeError):
            ga.main(["--bbox", "0", "0", "1", "1"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
