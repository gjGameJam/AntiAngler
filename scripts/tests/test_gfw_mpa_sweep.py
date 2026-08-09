"""Offline tests for scripts/gfw_mpa_sweep.py (the GFW MPA intrusion sweep).

Pure stdlib, no token / no network / no geopandas. Covers the parts correctness depends
on, each of which was a real bug or a measured API quirk during the 2026-08-09 build:

  * the half-open [start, end) window — a zero-width range silently returns nothing,
  * the two-tier MPA selection and its stateless date rotation,
  * parsing 4WINGS' nested (and null-when-empty) entries envelope,
  * summing per-cell rows into one record per (MPA, vessel),
  * the lowercase gear filter, and
  * the markdown digest, including honest reporting of a truncated run.

The HTTP layer is a thin requests shell (lazy-imported) and is not tested here.
Run either of:
    python scripts/tests/test_gfw_mpa_sweep.py
    python -m unittest scripts.tests.test_gfw_mpa_sweep
"""
import json
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import gfw_mpa_sweep as ms  # noqa: E402


def mpa(pid, area=10.0, name=None, iso3="XXX", cat="Ia"):
    return {"WDPA_PID": str(pid), "WDPAID": str(pid), "NAME": name or f"MPA {pid}",
            "ISO3": iso3, "IUCN_CAT": cat, "MARINE": "2", "GIS_M_AREA": str(area)}


class TestImportIsStdlibOnly(unittest.TestCase):
    """CI installs nothing, so the module must import without requests/dotenv present."""

    def test_no_module_level_requests_import(self):
        src = (SCRIPTS_DIR / "gfw_mpa_sweep.py").read_text(encoding="utf-8")
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("import requests") or \
                    stripped.startswith("from requests"):
                self.fail("gfw_mpa_sweep.py must lazy-import requests (inside a function); "
                          f"found a module-level import: {stripped!r}")


class TestResolveWindow(unittest.TestCase):
    """GFW's date-range is HALF-OPEN [start, end): measured 2026-08-09, the window
    2026-08-03,2026-08-03 returns 0 rows while 2026-08-03,2026-08-04 returns the vessel
    that entered on 08-03. A zero-width window is an error, never a silent empty result."""

    def test_default_window_is_lagged_and_wide(self):
        args = ms.parse_args([])
        today = date(2026, 8, 9)
        start, end = ms.resolve_window(args, today=today)
        # default lag 2, width 7 -> ends 08-07, starts 07-31
        self.assertEqual(end, "2026-08-07")
        self.assertEqual(start, "2026-07-31")
        self.assertLess(start, end)

    def test_explicit_range_wins(self):
        args = ms.parse_args(["--start", "2026-01-01", "--end", "2026-01-08"])
        self.assertEqual(ms.resolve_window(args, today=date(2026, 8, 9)),
                         ("2026-01-01", "2026-01-08"))

    def test_zero_width_explicit_range_rejected(self):
        args = ms.parse_args(["--start", "2026-01-01", "--end", "2026-01-01"])
        with self.assertRaises(ValueError) as ctx:
            ms.resolve_window(args, today=date(2026, 8, 9))
        self.assertIn("empty", str(ctx.exception).lower())

    def test_inverted_range_rejected(self):
        args = ms.parse_args(["--start", "2026-02-01", "--end", "2026-01-01"])
        with self.assertRaises(ValueError):
            ms.resolve_window(args, today=date(2026, 8, 9))

    def test_zero_window_days_rejected(self):
        args = ms.parse_args(["--window-days", "0"])
        with self.assertRaises(ValueError):
            ms.resolve_window(args, today=date(2026, 8, 9))


class TestSelectMpas(unittest.TestCase):
    """A full 1799-region sweep costs ~6.5h (one request at a time), past the CI job
    ceiling — so big MPAs go every run and the small tail rotates by date."""

    def setUp(self):
        self.rows = ([mpa(f"big{i}", area=100.0 + i) for i in range(5)] +
                     [mpa(f"small{i}", area=0.1) for i in range(10)])

    def test_priority_tier_always_included_largest_first(self):
        sel, meta = ms.select_mpas(self.rows, min_area=1.0, tail_slice=0, day_ordinal=1)
        self.assertEqual(meta["priority_count"], 5)
        self.assertEqual(len(sel), 5)
        areas = [float(r["GIS_M_AREA"]) for r in sel]
        self.assertEqual(areas, sorted(areas, reverse=True), "largest MPAs first")

    def test_tail_slice_appended(self):
        sel, meta = ms.select_mpas(self.rows, min_area=1.0, tail_slice=4, day_ordinal=0)
        self.assertEqual(len(sel), 9)
        self.assertEqual(meta["tail_total"], 10)
        self.assertEqual(meta["tail_slice"], 4)
        self.assertEqual(meta["tail_cycle_runs"], 3)  # ceil(10/4)
        self.assertFalse(meta["priority_rotating"])

    def test_capacity_below_priority_tier_rotates_the_tier_itself(self):
        """The starvation guard: if the budget cannot fit the priority tier, the tier
        must ROTATE rather than being cut at the same place every run — otherwise its
        smallest members are never swept while the run still claims to cover the tier."""
        seen = set()
        for day in range(3):
            sel, meta = ms.select_mpas(self.rows, min_area=1.0, tail_slice=4,
                                       day_ordinal=day, capacity=2)
            self.assertTrue(meta["priority_rotating"])
            self.assertEqual(len(sel), 2, "must not exceed capacity")
            self.assertEqual(meta["tail_slice"], 0, "no tail when the tier does not fit")
            seen.update(r["WDPA_PID"] for r in sel)
        self.assertEqual(len(seen), 5, "3 runs of 2 must reach all 5 priority MPAs")

    def test_capacity_limits_the_tail_not_the_priority_tier(self):
        """With room for the tier but not the full tail slice, the tier stays whole."""
        sel, meta = ms.select_mpas(self.rows, min_area=1.0, tail_slice=10,
                                   day_ordinal=0, capacity=7)
        self.assertFalse(meta["priority_rotating"])
        self.assertEqual(meta["priority_count"], 5)
        self.assertEqual(meta["tail_slice"], 2)
        self.assertEqual(len(sel), 7)

    def test_capacity_none_means_unlimited(self):
        sel, meta = ms.select_mpas(self.rows, min_area=1.0, tail_slice=10,
                                   day_ordinal=0, capacity=None)
        self.assertFalse(meta["priority_rotating"])
        self.assertEqual(len(sel), 15)

    def test_rotation_advances_with_the_date_and_covers_the_tail(self):
        """Successive runs must not re-check the same tail slice forever."""
        seen = set()
        for day in range(3):
            sel, _ = ms.select_mpas(self.rows, min_area=1.0, tail_slice=4,
                                    day_ordinal=day)
            seen.update(r["WDPA_PID"] for r in sel if r["WDPA_PID"].startswith("small"))
        self.assertEqual(len(seen), 10, "3 runs of 4 should cover all 10 tail MPAs")

    def test_rotation_is_deterministic_for_a_given_day(self):
        a, _ = ms.select_mpas(self.rows, min_area=1.0, tail_slice=3, day_ordinal=7)
        b, _ = ms.select_mpas(self.rows, min_area=1.0, tail_slice=3, day_ordinal=7)
        self.assertEqual([r["WDPA_PID"] for r in a], [r["WDPA_PID"] for r in b])

    def test_min_area_zero_takes_everything_as_priority(self):
        sel, meta = ms.select_mpas(self.rows, min_area=0.0, tail_slice=0, day_ordinal=1)
        self.assertEqual(meta["priority_count"], 15)
        self.assertEqual(meta["tail_total"], 0)
        self.assertEqual(len(sel), 15)

    def test_unparseable_area_is_treated_as_zero_not_a_crash(self):
        rows = [mpa("x", area=5.0), dict(mpa("y"), GIS_M_AREA=""),
                dict(mpa("z"), GIS_M_AREA="n/a")]
        sel, meta = ms.select_mpas(rows, min_area=1.0, tail_slice=0, day_ordinal=1)
        self.assertEqual(meta["priority_count"], 1)
        self.assertEqual(meta["tail_total"], 2)
        self.assertEqual(len(sel), 1)


class TestExtractRows(unittest.TestCase):
    """4WINGS wraps rows as {"entries":[{"<dataset>:<ver>": [...]}]} and reports EMPTY as
    a list holding a dict whose single value is null — not an empty list."""

    def test_real_populated_envelope(self):
        payload = {"total": 1, "entries": [{"public-global-fishing-effort:v4.0": [
            {"shipName": "GRAND EXPLORER", "hours": 0.68},
            {"shipName": "GRAND EXPLORER", "hours": 1.43}]}]}
        self.assertEqual(len(ms.extract_rows(payload)), 2)

    def test_real_empty_envelope_yields_no_rows(self):
        payload = {"total": 1, "entries": [{"public-global-fishing-effort:v4.0": None}]}
        self.assertEqual(ms.extract_rows(payload), [])

    def test_malformed_payloads_yield_no_rows_rather_than_raising(self):
        """One bad region must never abort a multi-hundred-region sweep."""
        for payload in (None, {}, {"entries": None}, {"entries": []}, {"entries": [None]},
                        {"entries": [{"k": "not-a-list"}]}, "nonsense", 42):
            self.assertEqual(ms.extract_rows(payload), [], repr(payload))


class TestNormalizeEvent(unittest.TestCase):
    ROW = {"callsign": "HOA4632", "entryTimestamp": "2026-08-03T05:00:00Z",
           "exitTimestamp": "2026-08-03T09:00:00Z", "flag": "PAN",
           "geartype": "OTHER_PURSE_SEINES", "hours": 0.6836111111111111,
           "imo": "9080950", "lat": 79.6, "lon": 18.399999618530273,
           "mmsi": "352005711", "shipName": "GRAND EXPLORER",
           "vesselId": "d738540dd-d4b7-56e8-2c3c-97ac0e0a35ff"}

    def test_maps_the_live_response_shape(self):
        ev = ms.normalize_event(self.ROW, mpa("1334", area=36858.0,
                                              name="Nordaust-Svalbard", iso3="SJM",
                                              cat="Ib"))
        self.assertEqual(ev["wdpa_pid"], "1334")
        self.assertEqual(ev["mpa_name"], "Nordaust-Svalbard")
        self.assertEqual(ev["iucn_cat"], "Ib")
        self.assertEqual(ev["mmsi"], "352005711")
        self.assertEqual(ev["vessel_name"], "GRAND EXPLORER")
        self.assertEqual(ev["flag"], "PAN")
        self.assertAlmostEqual(ev["fishing_hours"], 0.6836111, places=5)
        self.assertEqual(ev["source"], ms.SOURCE)

    def test_missing_fields_become_none_not_crashes(self):
        ev = ms.normalize_event({}, mpa("1"))
        self.assertIsNone(ev["mmsi"])
        self.assertIsNone(ev["vessel_name"])
        self.assertEqual(ev["fishing_hours"], 0.0)

    def test_non_numeric_hours_does_not_raise(self):
        ev = ms.normalize_event({"hours": "not-a-number"}, mpa("1"))
        self.assertEqual(ev["fishing_hours"], 0.0)


class TestAggregateEvents(unittest.TestCase):
    """4WINGS returns one row per vessel PER SPATIAL CELL, so a single vessel in a single
    MPA arrives split across rows. Live check: GALOPIN came back in 8 cells totalling
    7.59 h, GRAND EXPLORER in 2 totalling 2.11 h."""

    def _ev(self, pid, vid, hours, entry=None, exit_=None, **kw):
        base = ms.normalize_event(
            {"vesselId": vid, "hours": hours, "entryTimestamp": entry,
             "exitTimestamp": exit_, "shipName": "V", "mmsi": "1"}, mpa(pid))
        base.update(kw)
        return base

    def test_sums_hours_per_mpa_and_vessel(self):
        out = ms.aggregate_events([self._ev("1334", "v1", 0.6836111111111111),
                                   self._ev("1334", "v1", 1.4283333333333332)])
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0]["fishing_hours"], 2.1119444, places=5)
        self.assertEqual(out[0]["cells"], 2)

    def test_same_vessel_in_two_mpas_stays_two_events(self):
        out = ms.aggregate_events([self._ev("1", "v1", 1.0), self._ev("2", "v1", 2.0)])
        self.assertEqual(len(out), 2)

    def test_takes_widest_entry_exit_span(self):
        out = ms.aggregate_events([
            self._ev("1", "v1", 1.0, entry="2026-08-03T09:00:00Z",
                     exit_="2026-08-03T10:00:00Z"),
            self._ev("1", "v1", 1.0, entry="2026-08-03T05:00:00Z",
                     exit_="2026-08-03T09:00:00Z")])
        self.assertEqual(out[0]["entry_timestamp"], "2026-08-03T05:00:00Z")
        self.assertEqual(out[0]["exit_timestamp"], "2026-08-03T10:00:00Z")

    def test_sorted_by_hours_descending(self):
        out = ms.aggregate_events([self._ev("1", "a", 1.0), self._ev("2", "b", 9.0),
                                   self._ev("3", "c", 5.0)])
        self.assertEqual([e["fishing_hours"] for e in out], [9.0, 5.0, 1.0])

    def test_min_hours_filters_after_summing_not_before(self):
        """Two 0.6h cells make a 1.2h event; a 1.0h floor must keep it."""
        out = ms.aggregate_events([self._ev("1", "v1", 0.6), self._ev("1", "v1", 0.6)],
                                  min_hours=1.0)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0]["fishing_hours"], 1.2, places=6)

    def test_missing_identity_backfilled_from_a_later_cell(self):
        a = self._ev("1", "v1", 1.0)
        a["flag"] = None
        b = self._ev("1", "v1", 1.0)
        b["flag"] = "ESP"
        self.assertEqual(ms.aggregate_events([a, b])[0]["flag"], "ESP")


class TestGearFilter(unittest.TestCase):
    """Filter values are LOWERCASE. Measured: 'other_purse_seines' matched 2 rows while
    'OTHER_PURSE_SEINES' matched 0, even though the response reports uppercase."""

    def test_lowercases_values(self):
        self.assertEqual(ms.gear_filter("OTHER_PURSE_SEINES"),
                         "geartype in ('other_purse_seines')")

    def test_multi_trims_whitespace(self):
        self.assertEqual(ms.gear_filter(" trawlers , purse_seines "),
                         "geartype in ('trawlers', 'purse_seines')")

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            ms.gear_filter("  ,  ")

    def test_default_sends_no_filter_at_all(self):
        """Any gear inside a no-take reserve is the signal; the inherited 'trawlers'
        default silently dropped a real OTHER_PURSE_SEINES intrusion."""
        args = ms.parse_args([])
        self.assertIsNone(args.gear)
        self.assertNotIn("filters[0]", ms.build_params(args, "2026-01-01", "2026-01-08"))


class TestBuildParams(unittest.TestCase):
    def test_core_params(self):
        p = ms.build_params(ms.parse_args([]), "2026-07-31", "2026-08-07")
        self.assertEqual(p["date-range"], "2026-07-31,2026-08-07")
        self.assertEqual(p["group-by"], "VESSEL_ID")
        self.assertEqual(p["format"], "JSON")

    def test_gear_filter_included_when_asked(self):
        p = ms.build_params(ms.parse_args(["--gear", "trawlers"]),
                            "2026-07-31", "2026-08-07")
        self.assertEqual(p["filters[0]"], "geartype in ('trawlers')")


class TestLoadMpaIndex(unittest.TestCase):
    def _write(self, text):
        d = Path(tempfile.mkdtemp())
        p = d / "mpa_index.csv"
        p.write_text(text, encoding="utf-8")
        return p

    def test_reads_rows(self):
        p = self._write("WDPA_PID,NAME,GIS_M_AREA\n1,Alpha,5.0\n2,Beta,1.0\n")
        rows = ms.load_mpa_index(p)
        self.assertEqual([r["WDPA_PID"] for r in rows], ["1", "2"])

    def test_utf8_names_survive(self):
        p = self._write("WDPA_PID,NAME,GIS_M_AREA\n46,Reserva Biológica,3.0\n")
        self.assertEqual(ms.load_mpa_index(p)[0]["NAME"], "Reserva Biológica")

    def test_blank_pid_rows_dropped(self):
        p = self._write("WDPA_PID,NAME,GIS_M_AREA\n,Nameless,5.0\n7,Real,5.0\n")
        self.assertEqual([r["WDPA_PID"] for r in ms.load_mpa_index(p)], ["7"])

    def test_limit_applies(self):
        p = self._write("WDPA_PID,NAME,GIS_M_AREA\n1,A,5\n2,B,5\n3,C,5\n")
        self.assertEqual(len(ms.load_mpa_index(p, limit=2)), 2)

    def test_missing_file_names_the_generator(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            ms.load_mpa_index(Path(tempfile.mkdtemp()) / "nope.csv")
        self.assertIn("prep_polygons.py", str(ctx.exception))

    def test_empty_index_rejected(self):
        p = self._write("WDPA_PID,NAME,GIS_M_AREA\n")
        with self.assertRaises(ValueError):
            ms.load_mpa_index(p)


class TestSummarizeMarkdown(unittest.TestCase):
    def _result(self, events=None, errors=None, **cov):
        coverage = {"min_area_km2": 1.0, "priority_total": 748, "priority_count": 748,
                    "priority_offset": 0, "priority_cycle_runs": 1,
                    "priority_rotating": False, "capacity": 750, "tail_total": 1051,
                    "tail_slice": 250, "tail_offset": 0, "tail_cycle_runs": 5,
                    "index_total": 1799, "selected": 998, "queried": 998, "skipped": 0,
                    "budget_exhausted": False}
        coverage.update(cov)
        return {"window": {"start": "2026-07-31", "end": "2026-08-07"},
                "mpas_queried": coverage["queried"],
                "mpas_with_effort": len({e["wdpa_pid"] for e in (events or [])}),
                "coverage": coverage, "errors": errors or [], "events": events or []}

    def test_empty_run_says_so(self):
        md = ms.summarize_markdown(self._result())
        self.assertIn("No fishing effort recorded", md)

    def test_lists_mpa_and_vessel(self):
        ev = ms.normalize_event(
            {"shipName": "GALOPIN", "mmsi": "224286000", "flag": "ESP",
             "geartype": "DRIFTING_LONGLINES", "hours": 7.59}, mpa("303552", 857.0,
                                                                   "Malpelo", "COL", "Ib"))
        md = ms.summarize_markdown(self._result([ev]))
        self.assertIn("Malpelo", md)
        self.assertIn("GALOPIN", md)
        self.assertIn("224286000", md)
        self.assertIn("7.59", md)

    def test_truncated_run_is_reported_not_hidden(self):
        """A budget-limited run must never read as full coverage."""
        md = ms.summarize_markdown(self._result(skipped=300, budget_exhausted=True))
        self.assertIn("300", md)
        self.assertIn("Time budget", md)

    def test_coverage_line_states_what_was_not_swept(self):
        md = ms.summarize_markdown(self._result())
        self.assertIn("748", md)
        self.assertIn("1799", md)

    def test_pipe_in_mpa_name_does_not_break_the_table(self):
        ev = ms.normalize_event({"shipName": "V", "hours": 1.0},
                                mpa("1", 5.0, "Odd | Name"))
        md = ms.summarize_markdown(self._result([ev]))
        self.assertIn("Odd \\| Name", md)


class TestWriteJsonAtomic(unittest.TestCase):
    def test_writes_and_leaves_no_tmp(self):
        d = Path(tempfile.mkdtemp())
        out = ms.write_json_atomic(d / "sub" / "sweep.json", {"events": []})
        self.assertTrue(out.exists())
        self.assertEqual(json.loads(out.read_text(encoding="utf-8")), {"events": []})
        self.assertEqual(list(out.parent.glob("*.tmp")), [])

    def test_tmp_cleaned_up_on_serialization_failure(self):
        d = Path(tempfile.mkdtemp())
        target = d / "sweep.json"
        with self.assertRaises(TypeError):
            ms.write_json_atomic(target, {"bad": object()})
        self.assertFalse(target.exists())
        self.assertEqual(list(d.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
