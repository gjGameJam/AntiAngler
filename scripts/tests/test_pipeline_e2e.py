"""End-to-end P3 contract test over COMMITTED fixtures (ROADMAP / P3 workstream 4A).

Unlike the per-module tests (which build inputs in temp dirs), this runs the fusion + GFW-enrichment
chain over the checked-in `scripts/tests/fixtures/` and asserts the exact `violation_events.json`
contract pinned in `docs/PIPELINE.md`. It guards the shared seam between detect_boats → fuse_violations
→ gfw_vessels as a unit: if the emitted key set or the dark/matched/scoring behavior drifts, this fails.

Pure stdlib, no network / key / ML deps. Run:
    python scripts/tests/test_pipeline_e2e.py
    python -m unittest scripts.tests.test_pipeline_e2e
"""
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

import fuse_violations as fv  # noqa: E402

# The exact per-event key set documented in docs/PIPELINE.md. Changing build_event's output must
# update BOTH that doc and this set (that's the point of the guard).
EVENT_KEYS = {
    "event_id", "detection_id", "ais_status", "vessel_id", "mmsi", "gfw_vessel_id",
    "vessel_name", "flag", "gear_type", "imo", "iuu_listed", "mpa_id", "mpa_name",
    "centroid_wgs84", "chip", "entry_time", "exit_time", "duration_hours",
    "distance_from_port_km", "presence_confidence", "fishing_hours", "fishing_probability",
    "violation_score", "ais_match", "evidence",
}
TOP_KEYS = {
    "generated_utc", "run_dir", "source_detections", "ais_file", "vessels_file", "sensor",
    "scene_datetime", "params", "wdpa", "ais_ping_count", "counts", "violation_events",
}


class TestPipelineE2E(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        # Copy the committed run dir so fuse writes its outputs into temp, not into git.
        self.run = Path(self._td.name) / "run"
        shutil.copytree(FIXTURES / "run", self.run)
        self.ais = FIXTURES / "ais_positions.json"
        self.vessels = FIXTURES / "gfw_vessels.json"

    def tearDown(self):
        self._td.cleanup()

    def _run(self, extra=None):
        fv.main(["--run", str(self.run), "--ais", str(self.ais)] + (extra or []))
        doc = json.loads((self.run / "violation_events.json").read_text())
        gj = json.loads((self.run / "violation_events.geojson").read_text())
        return doc, gj, {e["detection_id"]: e for e in doc["violation_events"]}

    def test_ais_only_contract(self):
        doc, gj, by_det = self._run()
        # scope: det_00002 is outside the MPA -> excluded by default
        self.assertEqual(doc["counts"], {"in_scope": 2, "dark": 1, "matched": 1,
                                         "emitted": 2, "gfw_enriched": 0})
        self.assertEqual(set(doc), TOP_KEYS)
        for e in doc["violation_events"]:
            self.assertEqual(set(e), EVENT_KEYS)
        # det_00000 sits on MMSI 211476060 (~40 m, 2 min off) -> matched, not GFW-enriched here
        m = by_det["det_00000"]
        self.assertEqual(m["ais_status"], "matched")
        self.assertEqual(m["mmsi"], "211476060")
        self.assertEqual(m["evidence"], ["Sentinel-2", "AIS"])
        self.assertIsNone(m["fishing_probability"])
        self.assertAlmostEqual(m["violation_score"], 0.2, places=4)   # 0.8 * matched_weight(0.25)
        self.assertLessEqual(m["ais_match"]["distance_m"], 500)
        # det_00001 has no nearby AIS -> dark
        d = by_det["det_00001"]
        self.assertEqual(d["ais_status"], "dark")
        self.assertAlmostEqual(d["violation_score"], 0.6, places=4)
        # geojson holds exactly the dark point
        self.assertEqual(len(gj["features"]), 1)
        self.assertEqual(gj["features"][0]["properties"]["detection_id"], "det_00001")

    def test_gfw_enriched_contract(self):
        doc, _, by_det = self._run(["--vessels", str(self.vessels)])
        self.assertEqual(doc["counts"]["gfw_enriched"], 1)
        self.assertEqual(doc["vessels_file"], str(self.vessels))
        m = by_det["det_00000"]
        self.assertEqual(m["vessel_name"], "NEPTUNE")
        self.assertEqual(m["flag"], "DEU")
        self.assertEqual(m["gear_type"], "trawlers")
        self.assertEqual(m["imo"], "9012345")
        self.assertEqual(m["gfw_vessel_id"], "gfw-abc123")
        self.assertEqual(m["fishing_hours"], 2.4)
        self.assertEqual(m["evidence"], ["Sentinel-2", "AIS", "GFW"])
        self.assertAlmostEqual(m["violation_score"], 0.8, places=4)   # fishing_probability 1.0 -> full weight
        # dark event stays untouched by the GFW join
        self.assertEqual(by_det["det_00001"]["ais_status"], "dark")
        self.assertAlmostEqual(by_det["det_00001"]["violation_score"], 0.6, places=4)

    def test_all_detections_scope_includes_outside_mpa(self):
        doc, _, by_det = self._run(["--all-detections"])
        self.assertEqual(doc["counts"]["in_scope"], 3)   # det_00002 now included
        self.assertIn("det_00002", by_det)


if __name__ == "__main__":
    unittest.main(verbosity=2)
