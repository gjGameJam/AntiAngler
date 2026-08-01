"""Offline tests for scripts/alert_violations.py (the severity digest — the alerting interface).

Pure stdlib, no network. Covers the severity bucketing (including the IUU override and the unscored
case), the filters, MPA grouping/ranking, both output formats, and the opt-in --exit-code gate.
Run either of:
    python scripts/tests/test_alert_violations.py
    python -m unittest scripts.tests.test_alert_violations
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import alert_violations as av  # noqa: E402


def event(score=0.5, status="dark", **over):
    base = {
        "event_id": over.pop("event_id", "evt_00000"),
        "detection_id": "det_00000",
        "ais_status": status,
        "mmsi": None, "vessel_name": None, "flag": None, "iuu_listed": None,
        "mpa_id": 1.0, "mpa_name": "Test Reserve",
        "centroid_wgs84": [-119.38, 34.03],
        "chip": "chips/chip_r0_c0.tif",
        "duration_hours": None,
        "presence_confidence": score,
        "fishing_probability": None,
        "violation_score": score,
        "evidence": ["Sentinel-2"],
    }
    base.update(over)
    return base


def write_events(tmpdir, events, name="violation_events.json", run_dir="runX"):
    path = Path(tmpdir) / name
    path.write_text(json.dumps({"run_dir": run_dir, "violation_events": events}), encoding="utf-8")
    return path


def args(*paths, **over):
    argv = [str(p) for p in paths]
    for key, val in over.items():
        flag = f"--{key.replace('_', '-')}"
        argv += [flag] if val is True else [flag, str(val)]
    return av.parse_args(argv)


class TestSeverityBucketing(unittest.TestCase):
    def test_default_tiers(self):
        self.assertEqual(av.severity_of(event(0.95)), "critical")
        self.assertEqual(av.severity_of(event(0.80)), "critical")   # boundary is inclusive
        self.assertEqual(av.severity_of(event(0.79)), "high")
        self.assertEqual(av.severity_of(event(0.50)), "high")
        self.assertEqual(av.severity_of(event(0.49)), "medium")
        self.assertEqual(av.severity_of(event(0.25)), "medium")
        self.assertEqual(av.severity_of(event(0.24)), "low")

    def test_iuu_forces_critical_regardless_of_score(self):
        """An IUU listing is a standing fact about the vessel, not a property of this sighting."""
        self.assertEqual(av.severity_of(event(0.01, iuu_listed=True)), "critical")

    def test_unscored_event_is_low_never_paged(self):
        self.assertEqual(av.severity_of(event(None)), "low")

    def test_thresholds_are_tunable(self):
        self.assertEqual(av.severity_of(event(0.30), critical=0.3, high=0.2, medium=0.1), "critical")

    def test_transit_floor_stays_below_the_default_medium_floor(self):
        """geofence_ais' 0.10 transit weight must not page anyone by default — the whole reason
        cooperative transits are emitted rather than suppressed."""
        self.assertEqual(av.severity_of(event(0.10, status="reported")), "low")


class TestOrdering(unittest.TestCase):
    def test_at_or_above(self):
        self.assertTrue(av.at_or_above("critical", "medium"))
        self.assertTrue(av.at_or_above("medium", "medium"))
        self.assertFalse(av.at_or_above("low", "medium"))


class TestFiltering(unittest.TestCase):
    def _alerts(self, events, **over):
        with tempfile.TemporaryDirectory() as td:
            path = write_events(td, events)
            return av.build_alerts([path], args(path, **over))

    def test_min_severity_floor_applies(self):
        alerts, counts, scanned = self._alerts([event(0.9), event(0.6), event(0.3), event(0.05)])
        self.assertEqual(scanned, 4)
        self.assertEqual(len(alerts), 3)                     # the 0.05 is below medium
        self.assertEqual(counts["critical"], 1)
        self.assertEqual(counts["high"], 1)
        self.assertEqual(counts["medium"], 1)

    def test_min_severity_can_be_raised(self):
        alerts, _, _ = self._alerts([event(0.9), event(0.6), event(0.3)], min_severity="critical")
        self.assertEqual(len(alerts), 1)

    def test_dark_only(self):
        alerts, _, _ = self._alerts(
            [event(0.9, status="dark"), event(0.9, status="matched"), event(0.9, status="reported")],
            dark_only=True)
        self.assertEqual([a["event"]["ais_status"] for a in alerts], ["dark"])

    def test_top_truncates_after_ranking(self):
        alerts, _, _ = self._alerts([event(0.3, event_id="a"), event(0.9, event_id="b"),
                                     event(0.6, event_id="c")], top=2)
        self.assertEqual([a["event"]["event_id"] for a in alerts], ["b", "c"])

    def test_ranked_by_severity_then_score(self):
        alerts, _, _ = self._alerts([event(0.3), event(0.95), event(0.55)])
        self.assertEqual([a["severity"] for a in alerts], ["critical", "high", "medium"])


class TestGrouping(unittest.TestCase):
    def test_label_falls_back_sensibly(self):
        self.assertEqual(av.mpa_label({"mpa_name": "Kornati", "mpa_id": 7}), "Kornati")
        self.assertEqual(av.mpa_label({"mpa_name": None, "mpa_id": 7}), "MPA 7")
        self.assertEqual(av.mpa_label({"mpa_name": None, "mpa_id": None}), "(no MPA)")

    def test_named_mpa_without_id_is_not_swallowed(self):
        """Guards the operator-precedence bug: a name with no id must still group under the name."""
        self.assertEqual(av.mpa_label({"mpa_name": "Anacapa", "mpa_id": None}), "Anacapa")

    def test_groups_ordered_by_worst_alert(self):
        alerts = [{"severity": "critical", "source": "s",
                   "event": event(0.9, mpa_name="Bad Reserve")},
                  {"severity": "medium", "source": "s",
                   "event": event(0.3, mpa_name="Calm Reserve")}]
        self.assertEqual(list(av.group_by_mpa(alerts)), ["Bad Reserve", "Calm Reserve"])


class TestFormatting(unittest.TestCase):
    def _render(self, events, **over):
        with tempfile.TemporaryDirectory() as td:
            path = write_events(td, events)
            alerts, counts, scanned = av.build_alerts([path], args(path, **over))
            return (av.format_console(alerts, counts, scanned, [str(path)], "2026-07-29T10:44:00Z"),
                    av.format_markdown(alerts, counts, scanned, [str(path)], "2026-07-29T10:44:00Z"))

    def test_console_marks_critical_and_shows_context(self):
        console, _ = self._render([event(0.92, status="dark")])
        self.assertIn("CRITICAL", console)
        self.assertIn("! ", console)
        self.assertIn("Test Reserve", console)
        self.assertIn("34.03000,-119.38000", console)      # lat,lon order for pasting into a map
        self.assertIn("chips/chip_r0_c0.tif", console)

    def test_console_renders_reported_events_with_their_own_fields(self):
        console, _ = self._render([event(0.9, status="reported", mmsi="211476060",
                                         vessel_name="MARIA", fishing_probability=0.9,
                                         duration_hours=2.5, chip=None,
                                         evidence=["AIS", "AIS-heuristic"])])
        self.assertIn("reported", console)
        self.assertIn("MMSI 211476060", console)
        self.assertIn("MARIA", console)
        self.assertIn("fishing 0.90", console)
        self.assertIn("2.50 h in MPA", console)
        self.assertIn("[AIS+AIS-heuristic]", console)

    def test_iuu_is_called_out(self):
        console, markdown = self._render([event(0.1, iuu_listed=True, mmsi="1")])
        self.assertIn("IUU-LISTED", console)
        self.assertIn("CRITICAL", markdown)

    def test_empty_digest_is_explicit_not_blank(self):
        console, markdown = self._render([event(0.01)])
        self.assertIn("0 alert(s)", console)
        self.assertIn("nothing at or above the severity floor", console)
        self.assertIn("_Nothing at or above the severity floor._", markdown)

    def test_markdown_is_a_valid_table(self):
        _, markdown = self._render([event(0.92)])
        self.assertIn("# Violation alerts", markdown)
        self.assertIn("| Severity | Score | Status |", markdown)
        self.assertIn("| CRITICAL | 0.92 | dark |", markdown)
        self.assertTrue(markdown.endswith("\n"))

    def test_unscored_event_renders_without_crashing(self):
        console, markdown = self._render([event(None)], min_severity="low")
        self.assertIn("n/a", console)
        self.assertIn("n/a", markdown)


class TestMainAndInputs(unittest.TestCase):
    def test_reads_run_dirs_files_and_events_flag(self):
        with tempfile.TemporaryDirectory() as td:
            run = Path(td) / "runA"
            run.mkdir()
            write_events(run, [event(0.9)])
            other = write_events(td, [event(0.6)], name="geofence_events.json")
            parsed = args(run, events=other)
            files = av.collect_inputs(parsed)
            self.assertEqual(len(files), 2)
            self.assertTrue(str(files[0]).endswith("violation_events.json"))

    def test_deduplicates_repeated_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_events(td, [event(0.9)])
            self.assertEqual(len(av.collect_inputs(args(path, events=path))), 1)

    def test_writes_markdown_and_exit_code_gate(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_events(td, [event(0.95)])
            md = Path(td) / "alerts.md"
            self.assertEqual(av.main([str(path), "--out-markdown", str(md), "--quiet"]), 0)
            self.assertIn("CRITICAL", md.read_text(encoding="utf-8"))
            # opt-in gate: non-zero means "needs attention", not "the tool failed"
            self.assertEqual(av.main([str(path), "--quiet", "--exit-code"]), 1)

    def test_exit_code_zero_when_nothing_alerts(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_events(td, [event(0.01)])
            self.assertEqual(av.main([str(path), "--quiet", "--exit-code"]), 0)

    def test_no_inputs_is_a_clear_error(self):
        with self.assertRaises(RuntimeError):
            av.main([])

    def test_missing_file_is_a_clear_error(self):
        with self.assertRaises(RuntimeError):
            av.main(["definitely_not_here.json", "--quiet"])

    def test_rejects_inverted_thresholds(self):
        with tempfile.TemporaryDirectory() as td:
            path = write_events(td, [event(0.5)])
            with self.assertRaises(RuntimeError):
                av.main([str(path), "--critical", "0.1", "--high", "0.9", "--quiet"])

    def test_tolerates_a_bare_list_document(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bare.json"
            path.write_text(json.dumps([event(0.9)]), encoding="utf-8")
            alerts, _, scanned = av.build_alerts([path], args(path))
            self.assertEqual((len(alerts), scanned), (1, 1))


class TestMixedSources(unittest.TestCase):
    def test_fusion_and_geofence_events_rank_together(self):
        """The payoff of geofence_ais reusing the schema: one digest over both producers."""
        with tempfile.TemporaryDirectory() as td:
            fusion = write_events(td, [event(0.92, status="dark")], name="violation_events.json")
            geo = write_events(td, [event(0.75, status="reported", mmsi="1",
                                          mpa_name="Test Reserve", chip=None,
                                          evidence=["AIS", "AIS-heuristic"])],
                               name="geofence_events.json")
            alerts, counts, scanned = av.build_alerts([fusion, geo], args(fusion, events=geo))
            self.assertEqual(scanned, 2)
            self.assertEqual([a["event"]["ais_status"] for a in alerts], ["dark", "reported"])
            self.assertEqual(counts["critical"], 1)
            self.assertEqual(counts["high"], 1)
            # both land under the same MPA heading
            self.assertEqual(list(av.group_by_mpa(alerts)), ["Test Reserve"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
