"""Offline tests for scripts/report_violations.py (PHASE3_PLAN Workstream 2B).

Pure stdlib — builds synthetic violation_events.json files and asserts ranking, filtering, the CSV,
and the self-contained HTML. Run:
    python scripts/tests/test_report_violations.py
    python -m unittest scripts.tests.test_report_violations
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import report_violations as rv  # noqa: E402


def events_doc(run_dir, events):
    return {"run_dir": run_dir, "violation_events": events}


def ev(event_id, score, status="dark", **kw):
    base = {"event_id": event_id, "detection_id": event_id.replace("evt", "det"),
            "ais_status": status, "violation_score": score, "presence_confidence": score,
            "mpa_name": "Seal Rocks", "mpa_id": 555, "centroid_wgs84": [25.0, 37.0],
            "evidence": ["Sentinel-2"] + (["AIS"] if status == "matched" else []),
            "mmsi": None, "vessel_name": None, "flag": None, "gear_type": None, "imo": None,
            "iuu_listed": None, "fishing_hours": None, "fishing_probability": None, "chip": "c.tif"}
    base.update(kw)
    return base


class TestRankingAndFilters(unittest.TestCase):
    def _rows(self, files, **kw):
        return rv.collect_rows(files, **kw)

    def _write(self, tmp, name, events, run="run"):
        p = Path(tmp) / name
        p.write_text(json.dumps(events_doc(run, events)))
        return p

    def test_ranked_by_score_desc_dark_first_on_tie(self):
        with tempfile.TemporaryDirectory() as tmp:
            f1 = self._write(tmp, "a.json", [ev("evt_1", 0.9, "matched"), ev("evt_2", 0.3)])
            f2 = self._write(tmp, "b.json", [ev("evt_3", 0.9, "dark"), ev("evt_4", 0.5)], run="run2")
            rows = self._rows([f1, f2])
            self.assertEqual([r["rank"] for r in rows], [1, 2, 3, 4])
            # 0.9 dark (evt_3) ranks above 0.9 matched (evt_1) on the tie
            self.assertEqual(rows[0]["event_id"], "evt_3")
            self.assertEqual(rows[1]["event_id"], "evt_1")
            self.assertEqual([r["violation_score"] for r in rows], [0.9, 0.9, 0.5, 0.3])

    def test_dark_only_and_min_score_and_top(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = self._write(tmp, "a.json",
                            [ev("evt_1", 0.9, "matched"), ev("evt_2", 0.7, "dark"),
                             ev("evt_3", 0.2, "dark")])
            self.assertEqual([r["event_id"] for r in self._rows([f], dark_only=True)],
                             ["evt_2", "evt_3"])
            self.assertEqual([r["event_id"] for r in self._rows([f], min_score=0.5)],
                             ["evt_1", "evt_2"])
            self.assertEqual([r["event_id"] for r in self._rows([f], top=1)], ["evt_1"])

    def test_flatten_pulls_identity_and_latlon(self):
        row = rv.flatten_event(
            ev("evt_1", 0.8, "matched", mmsi="211", vessel_name="NEPTUNE", flag="DEU",
               gear_type="trawlers", fishing_hours=2.4, iuu_listed=True,
               centroid_wgs84=[25.5, 37.5], evidence=["Sentinel-2", "AIS", "GFW"]), "runX")
        self.assertEqual(row["run_dir"], "runX")
        self.assertEqual(row["vessel_name"], "NEPTUNE")
        self.assertEqual(row["lon"], 25.5)
        self.assertEqual(row["lat"], 37.5)
        self.assertEqual(row["evidence"], "Sentinel-2+AIS+GFW")
        self.assertTrue(row["iuu_listed"])


class TestOutputs(unittest.TestCase):
    def test_csv_header_and_row(self):
        rowlist = [rv.flatten_event(ev("evt_1", 0.8, "matched", mmsi="211"), "run")]
        rowlist[0]["rank"] = 1
        csv_text = rv.rows_to_csv(rowlist)
        self.assertEqual(csv_text.splitlines()[0], ",".join(rv.CSV_COLUMNS))
        self.assertIn("evt_1", csv_text.splitlines()[1])

    def test_html_is_self_contained(self):
        rowlist = [rv.flatten_event(ev("evt_1", 0.8, "dark"), "run")]
        rowlist[0]["rank"] = 1
        h = rv.rows_to_html(rowlist, ["a/violation_events.json"])
        self.assertIn("MPA violation candidates", h)
        self.assertIn("<table>", h)
        self.assertNotIn("http://", h)   # zero external assets
        self.assertNotIn("https://", h)

    def test_html_escapes_values(self):
        rowlist = [rv.flatten_event(ev("evt_1", 0.5, "matched", vessel_name="<script>x</script>"), "run")]
        rowlist[0]["rank"] = 1
        h = rv.rows_to_html(rowlist, ["s"])
        self.assertNotIn("<script>x</script>", h)
        self.assertIn("&lt;script&gt;", h)

    def test_empty_report_renders(self):
        h = rv.rows_to_html([], ["s"])
        self.assertIn("No events matched", h)


class TestMainIO(unittest.TestCase):
    def test_main_writes_both_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "violation_events.json").write_text(json.dumps(
                events_doc("run", [ev("evt_1", 0.9, "dark"), ev("evt_2", 0.4, "matched", mmsi="211")])))
            rv.main(["--run", str(tmp), "--out-csv", str(tmp / "r.csv"), "--out-html", str(tmp / "r.html")])
            self.assertTrue((tmp / "r.csv").exists())
            self.assertTrue((tmp / "r.html").exists())
            self.assertEqual(len((tmp / "r.csv").read_text().splitlines()), 3)  # header + 2 rows

    def test_main_requires_input(self):
        with self.assertRaises(RuntimeError):
            rv.main([])

    def test_main_writes_non_ascii_and_iuu_utf8(self):
        # The HTML declares <meta charset="utf-8"> and carries the U+26A0 IUU marker, and real
        # vessel names are not Latin-1 — the write must be UTF-8 regardless of the locale default
        # (Windows cp1252 crashed on both).
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "violation_events.json").write_text(json.dumps(events_doc(
                "run", [ev("evt_1", 0.9, "matched", mmsi="412000000",
                           vessel_name="海丰607", iuu_listed=True)])))
            rv.main(["--run", str(tmp), "--out-csv", str(tmp / "r.csv"), "--out-html", str(tmp / "r.html")])
            html = (tmp / "r.html").read_text(encoding="utf-8")
            self.assertIn("海丰607", html)
            self.assertIn("⚠", html)   # the IUU marker survived
            self.assertIn("海丰607", (tmp / "r.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
