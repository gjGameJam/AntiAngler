"""Offline tests for scripts/ais_fishing.py (the local speed/sinuosity fishing heuristic).

Pure stdlib, no network / key / ML deps. Synthetic tracks are built to isolate each behaviour the
heuristic claims to separate: a trawler working a small box, a cargo ship transiting straight, a
moored vessel, an AIS gap, and the thin-evidence cases that must return null rather than 0.0.
Run either of:
    python scripts/tests/test_ais_fishing.py
    python -m unittest scripts.tests.test_ais_fishing
"""
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import ais_fishing as af  # noqa: E402

T0 = datetime(2026, 7, 14, 10, 0, 0, tzinfo=timezone.utc)


def ping(mmsi, lon, lat, minutes, sog=None, name=None):
    return {"mmsi": mmsi, "lon": lon, "lat": lat,
            "timestamp": (T0 + timedelta(minutes=minutes)).isoformat().replace("+00:00", "Z"),
            "sog": sog, "name": name}


def args(**over):
    """parse_args with an --ais placeholder, so tests exercise the real defaults."""
    argv = ["--ais", "x.json"]
    for key, val in over.items():
        argv += [f"--{key.replace('_', '-')}", str(val)]
    return af.parse_args(argv)


def track(raw_pings):
    """Raw dicts -> the normalized, time-sorted track assess_track expects."""
    return [af.normalize_ping(p) for p in raw_pings]


class TestSpeedBand(unittest.TestCase):
    def test_reported_sog_preferred_over_derived(self):
        a = af.normalize_ping(ping("1", 0.0, 0.0, 0, sog=3.0))
        b = af.normalize_ping(ping("1", 0.0, 0.0, 10, sog=3.0))
        # positions identical -> derived speed would be 0; the reported SOG must win
        self.assertAlmostEqual(af.interval_speed_kn(a, b), 3.0)

    def test_derived_when_sog_missing(self):
        # 1 nautical mile (1852 m) of longitude at the equator in 60 min = 1 kn
        a = af.normalize_ping(ping("1", 0.0, 0.0, 0))
        b = af.normalize_ping(ping("1", 1852.0 / 111_320.0, 0.0, 60))
        self.assertAlmostEqual(af.interval_speed_kn(a, b), 1.0, places=1)

    def test_zero_duration_interval_is_none(self):
        a = af.normalize_ping(ping("1", 0.0, 0.0, 0))
        b = af.normalize_ping(ping("1", 0.1, 0.0, 0))
        self.assertIsNone(af.interval_speed_kn(a, b))


class TestBehaviourSeparation(unittest.TestCase):
    """The whole point of the heuristic: working != transiting != moored."""

    def test_trawler_working_a_box_scores_high(self):
        # 3 kn (in band) around a small square -> large path, tiny net displacement
        box = [(0.00, 0.00), (0.02, 0.00), (0.02, 0.02), (0.00, 0.02), (0.00, 0.00)]
        pings = [ping("1", lon, lat, i * 20, sog=3.0) for i, (lon, lat) in enumerate(box)]
        rec = af.assess_track("1", track(pings), args())
        self.assertLess(rec["straightness"], 0.1)          # returned near its start
        self.assertGreater(rec["fishing_probability"], 0.8)
        self.assertFalse(rec["insufficient_data"])

    def test_cargo_transiting_straight_scores_low(self):
        # 14 kn (above band) in a straight line -> straightness 1.0, band fraction 0
        pings = [ping("2", 0.1 * i, 0.0, i * 20, sog=14.0) for i in range(5)]
        rec = af.assess_track("2", track(pings), args())
        self.assertAlmostEqual(rec["straightness"], 1.0, places=3)
        self.assertLess(rec["fishing_probability"], 0.05)
        self.assertEqual(rec["fishing_hours"], 0.0)

    def test_moored_vessel_is_not_fishing(self):
        """Below the band floor: anchored/moored, which must not read as working the gear."""
        pings = [ping("3", 0.0, 0.0, i * 20, sog=0.1) for i in range(5)]
        rec = af.assess_track("3", track(pings), args())
        self.assertEqual(rec["fishing_hours"], 0.0)
        # no measurable path -> straightness unknown, so the band term alone drives it
        self.assertIsNone(rec["straightness"])
        self.assertAlmostEqual(rec["fishing_probability"], 0.0)

    def test_slow_straight_transit_lands_in_the_middle(self):
        """Documents a known limit: a slow straight run is in-band but not sinuous, so it scores
        mid-range rather than being confidently classified either way."""
        pings = [ping("4", 0.02 * i, 0.0, i * 20, sog=3.0) for i in range(5)]
        rec = af.assess_track("4", track(pings), args())
        self.assertAlmostEqual(rec["fishing_probability"], 0.5, places=2)


class TestGapHandling(unittest.TestCase):
    def test_long_gap_excluded_from_both_numerator_and_denominator(self):
        """An AIS gap is unobserved time: it must not invent fishing hours, nor dilute the score."""
        pings = [ping("1", 0.0, 0.0, 0, sog=3.0),
                 ping("1", 0.001, 0.0, 20, sog=3.0),
                 ping("1", 5.0, 5.0, 20 + 600, sog=3.0)]   # 10 h later, far away
        rec = af.assess_track("1", track(pings), args())
        self.assertEqual(rec["gaps_skipped"], 1)
        self.assertEqual(rec["intervals_used"], 1)
        self.assertAlmostEqual(rec["observed_hours"], 20 / 60, places=3)   # only the 20-min interval

    def test_gap_threshold_is_tunable(self):
        pings = track([ping("1", 0.0, 0.0, 0, sog=3.0),
                       ping("1", 0.001, 0.0, 90, sog=3.0)])   # a single 90-minute interval
        self.assertEqual(af.assess_track("1", pings, args())["intervals_used"], 0)
        self.assertEqual(af.assess_track("1", pings, args(max_gap_minutes=120))["intervals_used"], 1)


class TestInsufficientEvidenceIsNullNotZero(unittest.TestCase):
    """'We could not tell' and 'it was not fishing' are different claims. fuse_violations treats a
    null fishing_probability as no-information (score stays at matched_weight), so this matters."""

    def test_too_few_pings(self):
        rec = af.assess_track("1", track([ping("1", 0.0, 0.0, 0, sog=3.0),
                                          ping("1", 0.001, 0.0, 30, sog=3.0)]), args())
        self.assertTrue(rec["insufficient_data"])
        self.assertIsNone(rec["fishing_probability"])
        self.assertIsNone(rec["fishing_hours"])

    def test_span_too_short(self):
        pings = [ping("1", 0.0, 0.0, i, sog=3.0) for i in range(4)]   # 3 min span
        rec = af.assess_track("1", track(pings), args())
        self.assertTrue(rec["insufficient_data"])
        self.assertIsNone(rec["fishing_probability"])

    def test_probability_none_when_no_usable_intervals(self):
        self.assertIsNone(af.fishing_probability(
            {"observed_seconds": 0.0, "fishing_seconds": 0.0, "straightness": None}, 0.5))


class TestContractOutput(unittest.TestCase):
    """The record must satisfy the vessel-records contract fuse_violations.py --vessels reads."""

    def test_record_has_contract_keys_and_honest_source(self):
        pings = [ping("123456789", 0.001 * i, 0.0, i * 20, sog=3.0, name="MARIA") for i in range(5)]
        rec = af.assess_track("123456789", track(pings), args())
        for key in ("mmsi", "vessel_id", "shipname", "flag", "geartype", "imo", "callsign",
                    "fishing_hours", "fishing_probability", "source"):
            self.assertIn(key, rec)
        self.assertEqual(rec["mmsi"], "123456789")
        self.assertEqual(rec["shipname"], "MARIA")
        # identity is unknowable from positions alone - must be null, never guessed
        self.assertIsNone(rec["flag"])
        self.assertIsNone(rec["geartype"])
        self.assertIsNone(rec["imo"])
        # and it must NEVER claim GFW provenance
        self.assertEqual(rec["source"], "ais-heuristic")

    def test_fuse_violations_reads_it_and_tags_evidence_honestly(self):
        """The integration guard: a heuristic record must enrich like a GFW one but be labelled
        AIS-heuristic, not GFW."""
        import fuse_violations as fv
        rec = {"mmsi": "111", "fishing_probability": 1.0, "source": "ais-heuristic"}
        self.assertEqual(fv.vessel_evidence_label(rec), "AIS-heuristic")
        self.assertEqual(fv.vessel_evidence_label({"source": "gfw"}), "GFW")
        self.assertEqual(fv.vessel_evidence_label({}), "GFW")   # back-compat for pre-source files

    def test_ranked_and_min_probability_filter(self):
        work = [ping("1", lon, lat, i * 20, sog=3.0) for i, (lon, lat)
                in enumerate([(0, 0), (0.02, 0), (0.02, 0.02), (0, 0.02), (0, 0)])]
        transit = [ping("2", 0.1 * i, 1.0, i * 20, sog=14.0) for i in range(5)]
        ranked = af.assess_all([p for p in track(work + transit)], args())
        self.assertEqual([r["mmsi"] for r in ranked], ["1", "2"])   # most fishing-like first
        filtered = af.assess_all([p for p in track(work + transit)], args(min_probability=0.5))
        self.assertEqual([r["mmsi"] for r in filtered], ["1"])


class TestLoadAndMain(unittest.TestCase):
    def test_end_to_end_writes_readable_contract(self):
        work = [ping("1", lon, lat, i * 20, sog=3.0) for i, (lon, lat)
                in enumerate([(0, 0), (0.02, 0), (0.02, 0.02), (0, 0.02), (0, 0)])]
        with tempfile.TemporaryDirectory() as td:
            ais_path = Path(td) / "ais.json"
            ais_path.write_text(json.dumps({"positions": work}), encoding="utf-8")
            out_path = Path(td) / "fishing.json"
            rc = af.main(["--ais", str(ais_path), "--out", str(out_path)])
            self.assertEqual(rc, 0)
            doc = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(doc["source"], "ais-heuristic")
            self.assertEqual(doc["vessel_count"], 1)
            # and fuse_violations' own loader accepts it
            import fuse_violations as fv
            by_mmsi = fv.load_vessels(out_path)
            self.assertIn("1", by_mmsi)
            self.assertGreater(by_mmsi["1"]["fishing_probability"], 0.8)

    def test_dry_run_writes_nothing(self):
        work = [ping("1", 0.001 * i, 0.0, i * 20, sog=3.0) for i in range(5)]
        with tempfile.TemporaryDirectory() as td:
            ais_path = Path(td) / "ais.json"
            ais_path.write_text(json.dumps(work), encoding="utf-8")   # bare-list form
            out_path = Path(td) / "nope.json"
            self.assertEqual(af.main(["--ais", str(ais_path), "--out", str(out_path), "--dry-run"]), 0)
            self.assertFalse(out_path.exists())

    def test_rejects_inverted_speed_band(self):
        with tempfile.TemporaryDirectory() as td:
            ais_path = Path(td) / "ais.json"
            ais_path.write_text(json.dumps([ping("1", 0.0, 0.0, 0, sog=3.0)]), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                af.main(["--ais", str(ais_path), "--fishing-speed-min", "6",
                         "--fishing-speed-max", "2"])

    def test_missing_ais_is_a_clear_error(self):
        with self.assertRaises(RuntimeError):
            af.main([])


if __name__ == "__main__":
    unittest.main(verbosity=2)
