"""Offline tests for scripts/sweep_digest.py (the sweep's "how many ships, where" text view).

Pure stdlib, no network. Covers the distinct-vessel de-duplication (including the overlapping-
WDPA-records case), MPA grouping/ordering, the --min-hours / --flagged-only filters, the
GFW-Insights flag join (and its honesty when no --vessels file is given), both output formats,
and the latest-file default resolution. Run either of:
    python scripts/tests/test_sweep_digest.py
    python -m unittest scripts.tests.test_sweep_digest
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import sweep_digest as sd  # noqa: E402


def ev(pid="100", mpa="Reserve A", mmsi="111", name="VESSEL A", hours=2.0,
       entry="2026-08-10T00:00:00Z", exit="2026-08-11T00:00:00Z", **over):
    base = {"wdpa_pid": pid, "wdpa_id": pid, "mpa_name": mpa, "iso3": "NOR", "iucn_cat": "Ia",
            "vessel_id": f"gfw-{mmsi}", "mmsi": mmsi, "vessel_name": name, "flag": "NOR",
            "gear_type": "TRAWLERS", "imo": None, "callsign": None, "fishing_hours": hours,
            "entry_timestamp": entry, "exit_timestamp": exit, "lat": 63.9, "lon": 8.8,
            "source": "gfw-4wings-mpa", "cells": 1}
    base.update(over)
    return base


def sweep_doc(events, errors=None, queried=10):
    return {"generated_utc": "2026-08-16T01:50:20Z", "source": "gfw-4wings-mpa",
            "window": {"start": "2026-08-07", "end": "2026-08-14", "end_exclusive": True},
            "coverage": {"index_total": 1799, "priority_count": 748, "tail_slice": 250,
                         "budget_exhausted": False},
            "mpas_queried": queried, "mpas_with_effort": len({e["wdpa_pid"] for e in events}),
            "raw_rows": len(events), "errors": errors or [], "events": events}


def args(**over):
    argv = []
    for key, val in over.items():
        flag = f"--{key.replace('_', '-')}"
        argv += [flag] if val is True else [flag, str(val)]
    return sd.parse_args(argv)


class TestSummarize(unittest.TestCase):
    def test_distinct_dedups_across_mpas(self):
        # one vessel in two different reserves + one other vessel -> 2 distinct, 3 events
        doc = sweep_doc([ev(pid="1", mpa="A", mmsi="111"),
                         ev(pid="2", mpa="B", mmsi="111", entry="2026-08-12T00:00:00Z"),
                         ev(pid="1", mpa="A", mmsi="222", name="VESSEL B")])
        s = sd.summarize(doc)
        self.assertEqual(s["distinct"], 2)
        self.assertEqual(s["event_count"], 3)
        self.assertEqual(len(s["multi"]), 1)
        self.assertFalse(s["multi"][0]["overlapping_records"])   # different entry stamps

    def test_overlapping_wdpa_records_flagged(self):
        # identical entry/exit/hours in two records -> the Steller-pair case
        doc = sweep_doc([ev(pid="1", mpa="Rookery Buffer", mmsi="111"),
                         ev(pid="2", mpa="No Transit Area", mmsi="111")])
        s = sd.summarize(doc)
        self.assertEqual(s["distinct"], 1)
        self.assertTrue(s["multi"][0]["overlapping_records"])

    def test_missing_mmsi_falls_back_and_never_merges_strangers(self):
        doc = sweep_doc([ev(mmsi=None, vessel_id=None, name=None),
                         ev(mmsi=None, vessel_id=None, name=None)])
        self.assertEqual(sd.summarize(doc)["distinct"], 2)   # row fallback: counts once each

    def test_groups_ordered_most_vessels_then_hours(self):
        doc = sweep_doc([ev(pid="1", mpa="Busy", mmsi="1"), ev(pid="1", mpa="Busy", mmsi="2"),
                         ev(pid="2", mpa="Heavy", mmsi="3", hours=50.0),
                         ev(pid="3", mpa="Light", mmsi="4", hours=1.0)])
        names = [g["mpa_name"] for g in sd.summarize(doc)["groups"]]
        self.assertEqual(names, ["Busy", "Heavy", "Light"])

    def test_events_within_group_hours_descending(self):
        doc = sweep_doc([ev(mmsi="1", hours=1.0), ev(mmsi="2", hours=9.0)])
        g = sd.summarize(doc)["groups"][0]
        self.assertEqual([e["mmsi"] for e in g["events"]], ["2", "1"])

    def test_min_hours_drops_and_reports(self):
        doc = sweep_doc([ev(mmsi="1", hours=0.1), ev(mmsi="2", hours=5.0)])
        s = sd.summarize(doc, min_hours=1.0)
        self.assertEqual(s["event_count"], 1)
        self.assertEqual(s["dropped_min_hours"], 1)
        self.assertEqual(s["distinct"], 1)


class TestFlags(unittest.TestCase):
    FLAGMAP = {"111": {"mmsi": "111", "iuu_listed": True, "ais_off_events": 3},
               "222": {"mmsi": "222", "iuu_listed": False, "ais_off_events": 0}}

    def test_flags_for_labels_honestly(self):
        self.assertEqual(sd.flags_for("111", self.FLAGMAP), ["IUU-LISTED", "AIS-OFF x3"])
        self.assertEqual(sd.flags_for("222", self.FLAGMAP), [])        # looked up, clean
        self.assertIsNone(sd.flags_for("999", self.FLAGMAP))           # unknown != clean
        self.assertIsNone(sd.flags_for("111", None))                   # no vessels file at all
        self.assertIsNone(sd.flags_for(None, self.FLAGMAP))

    def test_no_take_fishing_flag(self):
        flags = sd.flags_for("1", {"1": {"fishing_in_no_take_mpa": 2}})
        self.assertEqual(flags, ["NO-TAKE-FISHING x2"])

    def test_flagged_only_keeps_flagged_and_reports_drops(self):
        doc = sweep_doc([ev(mmsi="111"), ev(mmsi="222", name="CLEAN"), ev(mmsi="999", name="UNKNOWN")])
        s = sd.summarize(doc, flagmap=self.FLAGMAP, flagged_only=True)
        self.assertEqual(s["event_count"], 1)
        self.assertEqual(s["dropped_unflagged"], 2)   # clean AND never-looked-up both drop
        self.assertEqual(s["groups"][0]["events"][0]["mmsi"], "111")

    def test_load_vessel_flagmap_wrapped_and_bare(self):
        with tempfile.TemporaryDirectory() as tmp:
            wrapped = Path(tmp) / "w.json"
            wrapped.write_text(json.dumps({"vessels": [{"mmsi": 111, "iuu_listed": True}]}))
            self.assertIn("111", sd.load_vessel_flagmap(wrapped))
            bare = Path(tmp) / "b.json"
            bare.write_text(json.dumps([{"mmsi": "222"}]))
            self.assertIn("222", sd.load_vessel_flagmap(bare))


class TestFormatting(unittest.TestCase):
    def test_console_header_counts_and_flag_note(self):
        doc = sweep_doc([ev(mmsi="111"), ev(pid="2", mpa="B", mmsi="222", name="VESSEL B")],
                        errors=[{"wdpa_pid": "9", "mpa_name": "Errored Reserve", "status": 429}])
        a = args()
        s = sd.summarize(doc)
        text = sd.format_console(doc, s, a, Path("sweep.json"))
        self.assertIn("2 distinct vessel(s) in 2 MPA(s)", text)
        self.assertIn("unavailable - pass --vessels", text)     # no vessels file -> honest note
        self.assertIn("no GPS-spoofing field", text)            # disclaims spoofing detection
        self.assertIn("10 of 1799 MPAs queried", text)
        self.assertIn("HTTP 429  Errored Reserve", text)

    def test_console_shows_flags_and_counts_when_vessels_given(self):
        doc = sweep_doc([ev(mmsi="111"), ev(mmsi="333", name="DARKSHIP",
                                            vessel_id="gfw-333")])
        flagmap = {"111": {"iuu_listed": True}, "333": {"ais_off_events": 2}}
        a = args(vessels="v.json")
        s = sd.summarize(doc, flagmap=flagmap)
        text = sd.format_console(doc, s, a, Path("sweep.json"))
        self.assertIn("IUU-LISTED", text)
        self.assertIn("AIS-OFF x2", text)
        self.assertIn("1 IUU-listed, 1 with AIS-off gap events", text)

    def test_console_multi_mpa_footer(self):
        doc = sweep_doc([ev(pid="1", mpa="Rookery", mmsi="111"),
                         ev(pid="2", mpa="No Transit", mmsi="111")])
        text = sd.format_console(doc, sd.summarize(doc), args(), Path("s.json"))
        self.assertIn("vessels in more than one MPA:", text)
        self.assertIn("overlapping WDPA records", text)

    def test_markdown_has_tables(self):
        doc = sweep_doc([ev(mmsi="111", hours=3.5)])
        md = sd.format_markdown(doc, sd.summarize(doc), args(), Path("s.json"))
        self.assertIn("## Reserve A - NOR Ia (1 vessel(s), 3.5 h)", md)
        self.assertIn("| 3.5 | VESSEL A | 111 |", md)


class TestMainAndResolution(unittest.TestCase):
    def test_resolve_latest_in_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / "gfw_mpa_sweep_20260810T000000Z.json"
            new = Path(tmp) / "gfw_mpa_sweep_20260816T015020Z.json"
            old.write_text("{}")
            new.write_text("{}")
            self.assertEqual(sd.resolve_sweep_path(tmp), new)

    def test_resolve_missing_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                sd.resolve_sweep_path(tmp)
        with self.assertRaises(RuntimeError):
            sd.resolve_sweep_path(Path("does") / "not" / "exist.json")

    def test_flagged_only_without_vessels_raises(self):
        with self.assertRaises(RuntimeError):
            sd.main(["--flagged-only", "--quiet", "whatever.json"])

    def test_main_end_to_end_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            sweep = tmp / "gfw_mpa_sweep_20260816T015020Z.json"
            sweep.write_text(json.dumps(sweep_doc([ev(mmsi="111"), ev(mmsi="222", name="B")])))
            vessels = tmp / "vessels.json"
            vessels.write_text(json.dumps({"vessels": [{"mmsi": "111", "ais_off_events": 1}]}))
            out_txt, out_md = tmp / "digest.txt", tmp / "digest.md"
            rc = sd.main([str(sweep), "--vessels", str(vessels), "--quiet",
                          "--out", str(out_txt), "--out-markdown", str(out_md)])
            self.assertEqual(rc, 0)
            text = out_txt.read_text(encoding="utf-8")
            self.assertIn("AIS-OFF x1", text)
            self.assertIn("2 distinct vessel(s)", text)
            self.assertIn("| VESSEL A | 111 |", out_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
