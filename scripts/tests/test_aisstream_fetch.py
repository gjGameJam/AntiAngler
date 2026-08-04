"""Offline tests for scripts/aisstream_fetch.py (ROADMAP Phase 3 step 2).

Pure stdlib, no key / no network - exercises the wire-format normalization (the risky part) with
captured sample AISStream messages, and a normalize -> write -> fuse_violations round trip proving
the ingester's output feeds the matcher. Run either of:
    python scripts/tests/test_aisstream_fetch.py
    python -m unittest scripts.tests.test_aisstream_fetch
"""
import json
import sys
import tempfile
import unittest
from datetime import timezone
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import aisstream_fetch as ais  # noqa: E402
import fuse_violations as fv    # noqa: E402


def sample_message(mmsi=368207620, lat=37.0005, lon=25.0005,
                   time_utc="2026-07-14 10:00:00.509865357 +0000 UTC", sog=8.2, cog=141.0,
                   name="TEST VESSEL"):
    """A realistic AISStream PositionReport frame (MetaData + nested Message.PositionReport)."""
    return {
        "MessageType": "PositionReport",
        "MetaData": {"MMSI": mmsi, "ShipName": name, "latitude": lat, "longitude": lon,
                     "time_utc": time_utc},
        "Message": {"PositionReport": {"Cog": cog, "Latitude": lat, "Longitude": lon,
                                       "Sog": sog, "TrueHeading": 140, "UserID": mmsi}},
    }


class TestTimeParsing(unittest.TestCase):
    def test_go_nanosecond_format(self):
        # The critical gotcha: 9-digit fraction + trailing " UTC" tz name.
        dt = ais.parse_aisstream_time("2023-05-10 11:46:52.509865357 +0000 UTC")
        self.assertEqual(dt.tzinfo, timezone.utc)
        self.assertEqual(dt.year, 2023)
        self.assertEqual((dt.hour, dt.minute, dt.second), (11, 46, 52))
        self.assertEqual(dt.microsecond, 509865)   # nanoseconds truncated to microseconds

    def test_no_fraction_and_iso_and_epoch(self):
        a = ais.parse_aisstream_time("2026-07-14 10:00:00 +0000 UTC")
        b = ais.parse_aisstream_time("2026-07-14T10:00:00Z")
        c = ais.parse_aisstream_time(a.timestamp())
        self.assertEqual(a, b)
        self.assertEqual(a, c)

    def test_nonzero_offset_is_converted_to_utc(self):
        # +02:00 local 12:00 == 10:00 UTC
        dt = ais.parse_aisstream_time("2026-07-14 12:00:00 +0200 UTC")
        self.assertEqual((dt.hour, dt.minute), (10, 0))

    def test_unparseable_raises(self):
        with self.assertRaises(ValueError):
            ais.parse_aisstream_time("")
        with self.assertRaises(ValueError):
            ais.parse_aisstream_time("not-a-time")


class TestFieldCleaning(unittest.TestCase):
    def test_sog_sentinel_and_range(self):
        self.assertEqual(ais.clean_sog(8.2), 8.2)
        self.assertEqual(ais.clean_sog(0), 0.0)
        self.assertIsNone(ais.clean_sog(102.3))   # not-available sentinel
        self.assertIsNone(ais.clean_sog(-1))
        self.assertIsNone(ais.clean_sog(None))
        self.assertIsNone(ais.clean_sog("junk"))

    def test_cog_sentinel_and_range(self):
        self.assertEqual(ais.clean_cog(141.0), 141.0)
        self.assertIsNone(ais.clean_cog(360.0))    # not-available sentinel
        self.assertIsNone(ais.clean_cog(400))
        self.assertIsNone(ais.clean_cog(None))


class TestBboxAndSubscribe(unittest.TestCase):
    def test_bbox_lat_lon_ordering(self):
        # GeoJSON (minLon,minLat,maxLon,maxLat) -> AISStream [[[minLat,minLon],[maxLat,maxLon]]]
        self.assertEqual(ais.bbox_to_aisstream((25.0, 37.0, 26.0, 38.0)),
                         [[[37.0, 25.0], [38.0, 26.0]]])

    def test_build_subscribe(self):
        msg = ais.build_subscribe("KEY", (25.0, 37.0, 26.0, 38.0), "PositionReport", "123, 456")
        self.assertEqual(msg["APIKey"], "KEY")
        self.assertEqual(msg["BoundingBoxes"], [[[37.0, 25.0], [38.0, 26.0]]])
        self.assertEqual(msg["FilterMessageTypes"], ["PositionReport"])
        self.assertEqual(msg["FiltersShipMMSI"], ["123", "456"])

    def test_build_subscribe_omits_empty_filters(self):
        msg = ais.build_subscribe("KEY", (25.0, 37.0, 26.0, 38.0), "", None)
        self.assertNotIn("FilterMessageTypes", msg)
        self.assertNotIn("FiltersShipMMSI", msg)

    def test_resolve_bbox_rejects_bad_and_conflicting(self):
        class A:  # minimal args stand-in
            bbox = [26.0, 38.0, 25.0, 37.0]   # inverted
            wdpa_id = wdpa_pid = None
            gpkg = Path("/nonexistent.gpkg")
            label = None
        with self.assertRaises(RuntimeError):
            ais.resolve_bbox(A())
        A.bbox = None
        with self.assertRaises(RuntimeError):   # no AOI at all
            ais.resolve_bbox(A())


class TestNormalize(unittest.TestCase):
    def test_full_message(self):
        rec = ais.normalize_aisstream_message(sample_message())
        self.assertEqual(rec["mmsi"], "368207620")
        self.assertEqual(rec["lon"], 25.0005)
        self.assertEqual(rec["lat"], 37.0005)
        self.assertEqual(rec["timestamp"], "2026-07-14T10:00:00Z")
        self.assertEqual(rec["sog"], 8.2)
        self.assertEqual(rec["cog"], 141.0)
        self.assertEqual(rec["name"], "TEST VESSEL")

    def test_sentinels_become_none(self):
        rec = ais.normalize_aisstream_message(sample_message(sog=102.3, cog=360.0))
        self.assertIsNone(rec["sog"])
        self.assertIsNone(rec["cog"])

    def test_missing_fields_skip(self):
        self.assertIsNone(ais.normalize_aisstream_message({}))
        self.assertIsNone(ais.normalize_aisstream_message({"MetaData": {}, "Message": {}}))
        bad = sample_message()
        bad["MetaData"]["time_utc"] = ""
        self.assertIsNone(ais.normalize_aisstream_message(bad))

    def test_out_of_range_latlon_skip(self):
        self.assertIsNone(ais.normalize_aisstream_message(sample_message(lat=91.0)))   # AIS lat N/A
        self.assertIsNone(ais.normalize_aisstream_message(sample_message(lon=181.0)))

    def test_class_b_position_report_fallback(self):
        # Real streams also emit StandardClassBPositionReport (smaller vessels); the normalizer must
        # fall back to it for Sog/Cog while taking mmsi/lat/lon/time from MetaData.
        msg = {"MessageType": "StandardClassBPositionReport",
               "MetaData": {"MMSI": 444, "ShipName": "CLASS-B", "latitude": 37.0, "longitude": 25.0,
                            "time_utc": "2026-07-14 10:00:00.1 +0000 UTC"},
               "Message": {"StandardClassBPositionReport": {"Sog": 6.1, "Cog": 88.0, "UserID": 444}}}
        rec = ais.normalize_aisstream_message(msg)
        self.assertEqual(rec["mmsi"], "444")
        self.assertEqual(rec["lat"], 37.0)
        self.assertEqual(rec["sog"], 6.1)
        self.assertEqual(rec["cog"], 88.0)

    def test_dedup_latest_keeps_newest_per_mmsi(self):
        recs = [
            {"mmsi": "1", "timestamp": "2026-07-14T10:00:00Z"},
            {"mmsi": "1", "timestamp": "2026-07-14T10:05:00Z"},
            {"mmsi": "2", "timestamp": "2026-07-14T10:01:00Z"},
        ]
        kept = ais.apply_dedup(recs, "latest")
        by = {r["mmsi"]: r for r in kept}
        self.assertEqual(len(kept), 2)
        self.assertEqual(by["1"]["timestamp"], "2026-07-14T10:05:00Z")
        self.assertEqual(ais.apply_dedup(recs, "none"), recs)


class _FakeTimeout(Exception):
    pass


class _FakeClosed(Exception):
    pass


class _FakeWS:
    def __init__(self, frames):
        self._frames = list(frames)
        self.sent = None
        self.closed = False

    def send(self, payload):
        self.sent = payload

    def recv(self):
        if self._frames:
            item = self._frames.pop(0)
            if item is _FakeTimeout:
                raise _FakeTimeout()
            return item
        raise _FakeClosed()   # server closed once frames are exhausted

    def close(self):
        self.closed = True


class _FakeWebsocketModule:
    WebSocketTimeoutException = _FakeTimeout
    WebSocketConnectionClosedException = _FakeClosed

    def __init__(self, frames):
        self._frames = frames
        self.last_ws = None

    def create_connection(self, url, timeout=None):
        self.last_ws = _FakeWS(self._frames)
        return self.last_ws


class TestCollectLoop(unittest.TestCase):
    """Exercise the websocket shell offline by injecting a stub 'websocket' module (it is
    lazy-imported inside collect())."""

    def _run_with_frames(self, frames, **kw):
        """Single-session behaviour: reconnect off, so 'server closed' ends the capture.
        The reconnecting path is covered by TestReconnect below."""
        fake = _FakeWebsocketModule(frames)
        sys.modules["websocket"] = fake
        kw.setdefault("reconnect", False)
        try:
            return ais.collect({"APIKey": "x"}, duration_s=30, max_messages=0, recv_timeout=0.01,
                               **kw), fake
        finally:
            sys.modules.pop("websocket", None)

    def test_happy_path_skip_and_close(self):
        frames = [
            json.dumps(sample_message(mmsi=111)),          # good -> 1 record
            "{ not json",                                  # -> errors += 1
            json.dumps({"MetaData": {}, "Message": {}}),   # unusable -> skipped += 1
            _FakeTimeout,                                  # quiet tick -> continue
            # then frames exhausted -> server-closed -> loop breaks
        ]
        (records, stats), fake = self._run_with_frames(frames)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["mmsi"], "111")
        self.assertEqual(stats["errors"], 1)
        self.assertEqual(stats["skipped"], 1)
        self.assertTrue(fake.last_ws.closed)                       # connection closed in finally
        self.assertIn("APIKey", fake.last_ws.sent)                 # subscription was sent

    def test_error_frame_raises(self):
        with self.assertRaises(RuntimeError):
            self._run_with_frames([json.dumps({"error": "Invalid API Key"})])

    def test_missing_dependency_message(self):
        sys.modules.pop("websocket", None)
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "websocket":
                raise ImportError("no websocket")
            return real_import(name, *a, **k)

        builtins.__import__ = fake_import
        try:
            with self.assertRaises(RuntimeError) as ctx:
                ais.collect({"APIKey": "x"}, 1, 0, 1)
            self.assertIn("requirements-ais.txt", str(ctx.exception))
        finally:
            builtins.__import__ = real_import


class _FakeSeqWebsocketModule:
    """Serves a DIFFERENT frame list per connection, so a reconnect is observable offline.
    Optionally raises a queued exception from create_connection to simulate a failed handshake."""
    WebSocketTimeoutException = _FakeTimeout
    WebSocketConnectionClosedException = _FakeClosed

    def __init__(self, sessions, connect_errors=()):
        self._sessions = list(sessions)
        self._connect_errors = list(connect_errors)
        self.connections = 0

    def create_connection(self, url, timeout=None):
        self.connections += 1
        if self._connect_errors:
            err = self._connect_errors.pop(0)
            if err is not None:
                raise err
        frames = self._sessions.pop(0) if self._sessions else []
        return _FakeWS(frames)


class TestReconnect(unittest.TestCase):
    """ROADMAP W8 capture semantics: --duration is wall-clock, outages are recorded not hidden,
    and an exact (mmsi, timestamp) ping is never counted twice across a reconnect."""

    def _collect(self, sessions, connect_errors=(), **kw):
        fake = _FakeSeqWebsocketModule(sessions, connect_errors)
        sys.modules["websocket"] = fake
        kw.setdefault("duration_s", 10)
        kw.setdefault("max_messages", 0)
        kw.setdefault("recv_timeout", 0.01)
        kw.setdefault("sleep", lambda _s: None)   # never actually sleep in tests
        try:
            return ais.collect({"APIKey": "x"}, **kw), fake
        finally:
            sys.modules.pop("websocket", None)

    def test_reconnects_after_drop_and_records_the_gap(self):
        (records, stats), fake = self._collect(
            [[json.dumps(sample_message(mmsi=111))],
             [json.dumps(sample_message(mmsi=222, time_utc="2026-07-14 10:00:05 +0000 UTC"))]],
            max_messages=2)
        self.assertEqual([r["mmsi"] for r in records], ["111", "222"])
        self.assertEqual(fake.connections, 2)
        self.assertEqual(stats["reconnects"], 1)
        self.assertEqual(len(stats["gaps"]), 1)
        gap = stats["gaps"][0]
        self.assertLessEqual({"from", "to", "seconds"}, set(gap))   # the gap is legible downstream

    def test_duplicate_ping_dropped_across_reconnect(self):
        dup = json.dumps(sample_message(mmsi=111))          # identical mmsi AND timestamp
        (records, stats), _ = self._collect(
            [[dup],
             [dup, json.dumps(sample_message(mmsi=222, time_utc="2026-07-14 10:00:05 +0000 UTC"))]],
            max_messages=2)
        self.assertEqual([r["mmsi"] for r in records], ["111", "222"])
        self.assertEqual(stats["duplicates"], 1)            # the replayed ping did not double-count

    def test_error_frame_is_never_retried(self):
        """A bad key/bbox must fail fast - reconnecting would just hammer the endpoint."""
        fake = _FakeSeqWebsocketModule([[json.dumps({"error": "Invalid API Key"})]])
        sys.modules["websocket"] = fake
        try:
            with self.assertRaises(RuntimeError):
                ais.collect({"APIKey": "x"}, duration_s=10, max_messages=0, recv_timeout=0.01,
                            reconnect=True, sleep=lambda _s: None)
        finally:
            sys.modules.pop("websocket", None)
        self.assertEqual(fake.connections, 1)               # did not reconnect

    def test_failed_handshake_is_retried(self):
        (records, stats), fake = self._collect(
            [[json.dumps(sample_message(mmsi=111))]],
            connect_errors=[OSError("connection refused"), None],
            max_messages=1)
        self.assertEqual(fake.connections, 2)
        self.assertEqual([r["mmsi"] for r in records], ["111"])
        self.assertEqual(len(stats["gaps"]), 1)

    def test_no_reconnect_stops_at_first_drop(self):
        (records, stats), fake = self._collect(
            [[json.dumps(sample_message(mmsi=111))], [json.dumps(sample_message(mmsi=222))]],
            reconnect=False)
        self.assertEqual([r["mmsi"] for r in records], ["111"])
        self.assertEqual(fake.connections, 1)
        self.assertEqual(stats["reconnects"], 0)

    def test_flapping_endpoint_escalates_backoff(self):
        """A connection that serves one frame then drops must NOT reset the backoff.

        Resetting on 'received a frame' would reconnect a flapping endpoint at 0s forever -
        precisely the hammering the backoff exists to prevent. stability_s is set high here so
        no session counts as stable."""
        slept = []
        sessions = [[json.dumps(sample_message(
            mmsi=100 + i, time_utc=f"2026-07-14 10:00:{i:02d} +0000 UTC"))] for i in range(5)]
        fake = _FakeSeqWebsocketModule(sessions)
        sys.modules["websocket"] = fake
        try:
            ais.collect({"APIKey": "x"}, duration_s=10, max_messages=4, recv_timeout=0.01,
                        reconnect=True, backoff_s=2.0, max_backoff_s=30.0,
                        stability_s=999.0, sleep=slept.append)
        finally:
            sys.modules.pop("websocket", None)
        self.assertGreaterEqual(len(slept), 3)
        self.assertEqual(slept[:3], [0.0, 2.0, 4.0])   # escalates rather than staying at 0

    def test_stable_session_resets_backoff(self):
        """The mirror case: a session that outlived stability_s starts the next retry at 0s."""
        slept = []
        sessions = [[json.dumps(sample_message(mmsi=100 + i,
                                               time_utc=f"2026-07-14 10:00:{i:02d} +0000 UTC"))]
                    for i in range(4)]
        fake = _FakeSeqWebsocketModule(sessions)
        sys.modules["websocket"] = fake
        try:
            ais.collect({"APIKey": "x"}, duration_s=10, max_messages=3, recv_timeout=0.01,
                        reconnect=True, backoff_s=2.0, max_backoff_s=30.0,
                        stability_s=0.0, sleep=slept.append)   # every session counts as stable
        finally:
            sys.modules.pop("websocket", None)
        self.assertTrue(slept)
        self.assertEqual(set(slept), {0.0})            # never escalates while sessions are healthy

    def test_stats_carry_connected_and_elapsed(self):
        (_, stats), _ = self._collect([[json.dumps(sample_message(mmsi=111))]], max_messages=1)
        for key in ("connected_s", "elapsed_s", "reconnects", "gaps", "duplicates", "interrupted"):
            self.assertIn(key, stats)
        self.assertGreaterEqual(stats["elapsed_s"], stats["connected_s"])   # gaps are not connected


class TestReconnectDelay(unittest.TestCase):
    """The backoff schedule mirrors scripts/_http.py's 0s/2s/4s shape, capped."""

    def test_schedule(self):
        self.assertEqual(ais.reconnect_delay(1, 2.0, 30.0), 0.0)   # first retry is immediate
        self.assertEqual(ais.reconnect_delay(2, 2.0, 30.0), 2.0)
        self.assertEqual(ais.reconnect_delay(3, 2.0, 30.0), 4.0)
        self.assertEqual(ais.reconnect_delay(4, 2.0, 30.0), 8.0)

    def test_capped(self):
        self.assertEqual(ais.reconnect_delay(99, 2.0, 30.0), 30.0)


class TestEndToEndIntoMatcher(unittest.TestCase):
    """Normalize AISStream frames -> positions file -> fuse_violations: the ingester's whole point."""

    def test_normalized_positions_drive_the_matcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # A detection sitting on the sample vessel (matched) + one with no AIS (dark).
            run = tmp / "run"
            run.mkdir()
            dets = {
                "count": 2,
                "scene": {"item_id": "SYNTH", "datetime": "2026-07-14T10:00:00Z"},
                "wdpa": {"WDPAID": 555, "NAME": "Test MPA"},
                "detections": [
                    {"detection_id": "det_00000", "confidence": 0.9, "centroid_wgs84": [25.0, 37.0],
                     "bbox_wgs84": [25.0, 37.0, 25.001, 37.001], "inside_mpa": True, "chip": "c.tif"},
                    {"detection_id": "det_00001", "confidence": 0.8, "centroid_wgs84": [25.5, 37.5],
                     "bbox_wgs84": [25.5, 37.5, 25.501, 37.501], "inside_mpa": True, "chip": "c.tif"},
                ],
            }
            (run / "detections.json").write_text(json.dumps(dets))
            (run / "manifest.json").write_text(json.dumps({"provider": "pc"}))

            # Ingester output: normalize a real-shaped AISStream frame near det_00000.
            rec = ais.normalize_aisstream_message(
                sample_message(mmsi=111, lat=37.0005, lon=25.0005, sog=0.1))
            positions = {"positions": [rec]}
            ais_path = tmp / "ais.json"
            ais_path.write_text(json.dumps(positions))

            fv.main(["--run", str(run), "--ais", str(ais_path)])
            doc = json.loads((run / "violation_events.json").read_text())
            by_det = {e["detection_id"]: e for e in doc["violation_events"]}
            self.assertEqual(by_det["det_00000"]["ais_status"], "matched")
            self.assertEqual(by_det["det_00000"]["mmsi"], "111")
            self.assertEqual(by_det["det_00001"]["ais_status"], "dark")


if __name__ == "__main__":
    unittest.main(verbosity=2)
