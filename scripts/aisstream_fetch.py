import argparse
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv   # pinned in requirements.txt for real runs
except ImportError:                  # keep the module importable (and offline-testable) without it
    def load_dotenv(*_args, **_kwargs):
        return False

# The objective of this script (AIS ingestion, ROADMAP Phase 3 step 2):
# - Connect to the AISStream.io real-time websocket (free; needs a registered API key)
# - Subscribe to a bounding box (from --bbox or an --wdpa-id/--wdpa-pid MPA lookup)
# - Collect live PositionReport messages for a bounded --duration / --max-messages
# - NORMALIZE each AISStream message into the project's source-agnostic AIS contract
#   ({mmsi, lon, lat, timestamp, sog, cog}) and write it to data/raw/ais/ as the file
#   that scripts/fuse_violations.py --ais consumes.
#
# House rule: one script per data stream. This is the AISStream *adapter* for the AIS
# position stream - it ONLY produces the normalized contract; it does not import or touch
# fuse_violations.py. A different raw-AIS source (Spire, an AIS archive) or an identity
# source (GFW Vessels/Insights) would each be its OWN sibling ingester normalizing into the
# same contract. See docs/ROADMAP.md Phase 3 and the contract in CLAUDE.md.
#
# The normalization is factored into pure functions (parse_aisstream_time,
# normalize_aisstream_message, bbox_to_aisstream, build_subscribe) so the risky wire-format
# handling is unit-tested offline (scripts/tests/test_aisstream_fetch.py) with no key and no
# network. The websocket loop is a thin shell that lazy-imports websocket-client.
#
# AISStream wire-format gotchas handled here (all verified against AISStream's docs/examples):
#   1. Bounding boxes use [latitude, longitude] corner order (the OPPOSITE of GeoJSON lon/lat):
#      [[[minLat, minLon], [maxLat, maxLon]]]  (SW corner, then NE corner).
#   2. MetaData.time_utc is a Go-format string, e.g. "2023-05-10 11:46:52.509865357 +0000 UTC",
#      with NANOSECOND (9-digit) precision that Python's parsers can't take directly - we
#      truncate the fraction to microseconds.
#   3. Sog/Cog carry AIS "not available" sentinels (Sog 102.3, Cog 360.0) and out-of-range
#      junk; those map to None, which correctly triggers fuse_violations' max-speed fallback.
#
# LIMITATION (be honest): AISStream is a LIVE feed - it only delivers positions from now
# forward, so this ingests AIS for a NEAR-REAL-TIME workflow (fetch imagery + capture AIS in
# the same window, then fuse). It cannot retrieve AIS for a historical satellite scene; that
# needs an archive/Spire/GFW source (a future sibling ingester). See ROADMAP Phase 3 step 2.

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "ais"
DEFAULT_GPKG = REPO_ROOT / "data" / "processed" / "wdpa_marine_ia_ib.gpkg"
AISSTREAM_URL = "wss://stream.aisstream.io/v0/stream"

# AIS "not available" sentinels (ITU-R M.1371): Sog 102.3 kn, Cog 360.0 deg. Real values are
# strictly below these; anything at/above (or negative) is treated as unavailable.
SOG_NA = 102.3
COG_NA = 360.0

load_dotenv(SCRIPT_DIR / ".env")
load_dotenv(REPO_ROOT / ".env")


def env_or(name, default):
    val = os.getenv(name)
    return val if val not in (None, "") else default


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Ingest live AIS positions from AISStream.io over a bbox/MPA and write them in "
                    "the normalized AIS contract for fuse_violations.py (ROADMAP Phase 3 step 2).")
    p.add_argument("--bbox", type=float, nargs=4, metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
                   default=None, help="AOI in WGS84 lon/lat. Mutually exclusive with --wdpa-id/--wdpa-pid.")
    p.add_argument("--wdpa-id", type=float, default=(lambda v: float(v) if v else None)(env_or("AIS_WDPA_ID", None)),
                   help="Select an MPA bbox by WDPAID from the GeoPackage (needs geopandas).")
    p.add_argument("--wdpa-pid", default=env_or("AIS_WDPA_PID", None),
                   help="Select an MPA bbox by WDPA_PID from the GeoPackage.")
    p.add_argument("--gpkg", type=Path, default=Path(env_or("AIS_GPKG", str(DEFAULT_GPKG))),
                   help="MPA GeoPackage to resolve --wdpa-id/--wdpa-pid against.")
    p.add_argument("--label", default=env_or("AIS_LABEL", None),
                   help="Short label for the output filename. Defaults to the MPA name or 'bbox'.")
    p.add_argument("--duration", type=float, default=float(env_or("AIS_DURATION", 60.0)),
                   help="Seconds to collect before writing. Default 60. Ctrl-C stops early and still writes.")
    p.add_argument("--max-messages", type=int, default=int(env_or("AIS_MAX_MESSAGES", 0)),
                   help="Stop after this many position records (0 = unlimited within --duration).")
    p.add_argument("--message-types", default=env_or("AIS_MESSAGE_TYPES", "PositionReport"),
                   help="Comma-separated AISStream message types to subscribe to (server-side filter). "
                        "Default PositionReport (the moving-vessel positions the matcher needs).")
    p.add_argument("--mmsi", default=env_or("AIS_MMSI", None),
                   help="Optional comma-separated MMSI allowlist (AISStream FiltersShipMMSI).")
    p.add_argument("--dedup", choices=("none", "latest"), default=env_or("AIS_DEDUP", "none"),
                   help="'none' (default) keeps every ping (the matcher wants multiple pings per "
                        "vessel across the window); 'latest' keeps only the newest ping per MMSI.")
    p.add_argument("--recv-timeout", type=float, default=float(env_or("AIS_RECV_TIMEOUT", 5.0)),
                   help="Per-receive socket timeout (s). Bounds how long a quiet feed blocks before "
                        "the deadline is re-checked. Default 5.")
    p.add_argument("--no-reconnect", action="store_true",
                   default=env_or("AIS_NO_RECONNECT", "").lower() in ("1", "true", "yes"),
                   help="Stop at the first dropped connection instead of reconnecting. Default is to "
                        "reconnect until --duration elapses; --duration is WALL-CLOCK either way, so "
                        "an outage shortens the data collected rather than extending the capture.")
    p.add_argument("--reconnect-backoff", type=float, default=float(env_or("AIS_RECONNECT_BACKOFF", 2.0)),
                   help="Base seconds for reconnect backoff (0s on the first retry, then 2s/4s/8s... "
                        "capped by --max-backoff). Default 2.")
    p.add_argument("--max-backoff", type=float, default=float(env_or("AIS_MAX_BACKOFF", 30.0)),
                   help="Cap on the reconnect backoff delay (s). Default 30.")
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve the bbox + build the subscription and print them, but do NOT connect "
                        "or write. Offline sanity check for the AOI/key wiring.")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- pure: AOI / subscribe

def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-") or "bbox"


def bbox_to_aisstream(bbox):
    """(minLon, minLat, maxLon, maxLat) WGS84 -> AISStream BoundingBoxes: a single box given as
    two [latitude, longitude] corners [[[minLat, minLon], [maxLat, maxLon]]]. NOTE the lat/lon
    order is the reverse of GeoJSON - this is the #1 AISStream footgun."""
    min_lon, min_lat, max_lon, max_lat = bbox
    return [[[min_lat, min_lon], [max_lat, max_lon]]]


def build_subscribe(api_key, bbox, message_types, mmsis=None):
    """Build the AISStream subscription message. APIKey + BoundingBoxes are required;
    FilterMessageTypes / FiltersShipMMSI are optional server-side filters."""
    msg = {"APIKey": api_key, "BoundingBoxes": bbox_to_aisstream(bbox)}
    types = [t.strip() for t in str(message_types).split(",") if t.strip()]
    if types:
        msg["FilterMessageTypes"] = types
    if mmsis:
        ids = [m.strip() for m in str(mmsis).split(",") if m.strip()]
        if ids:
            msg["FiltersShipMMSI"] = ids
    return msg


def resolve_bbox(args):
    """bbox from --bbox or an MPA lookup (mirrors sat_fetch.resolve_bbox). Returns (bbox, label, wdpa_meta)."""
    has_bbox = args.bbox is not None
    has_wdpa = args.wdpa_id is not None or args.wdpa_pid is not None
    if has_bbox and has_wdpa:
        raise RuntimeError("Provide either --bbox or --wdpa-id/--wdpa-pid, not both.")
    if not has_bbox and not has_wdpa:
        raise RuntimeError("Provide an AOI: --bbox MINLON MINLAT MAXLON MAXLAT or --wdpa-id/--wdpa-pid.")
    if has_bbox:
        bbox = tuple(float(x) for x in args.bbox)
        min_lon, min_lat, max_lon, max_lat = bbox
        if not (min_lon < max_lon and min_lat < max_lat):
            raise RuntimeError(f"Bad --bbox (need minLon<maxLon and minLat<maxLat): {bbox}")
        return bbox, (args.label or "bbox"), None
    bbox, name, meta = bbox_from_wdpa(args.gpkg, args.wdpa_id, args.wdpa_pid)
    return bbox, (args.label or slugify(name)), meta


def bbox_from_wdpa(gpkg, wdpa_id, wdpa_pid):
    import geopandas as gpd  # heavy dep; only for the MPA-lookup path
    if not Path(gpkg).exists():
        raise RuntimeError(f"GeoPackage not found: {gpkg}")
    gdf = gpd.read_file(gpkg)
    if wdpa_id is not None:
        sel = gdf[gdf["WDPAID"] == float(wdpa_id)]
        key = f"WDPAID={wdpa_id}"
    else:
        sel = gdf[gdf["WDPA_PID"].astype(str) == str(wdpa_pid)]
        key = f"WDPA_PID={wdpa_pid}"
    if sel.empty:
        raise RuntimeError(f"No MPA matching {key} in {gpkg}")
    row = sel.iloc[0]
    minx, miny, maxx, maxy = (float(v) for v in sel.total_bounds)
    meta = {"WDPAID": float(row["WDPAID"]) if row.get("WDPAID") is not None else None,
            "WDPA_PID": str(row["WDPA_PID"]), "NAME": str(row["NAME"])}
    return (minx, miny, maxx, maxy), meta["NAME"], meta


# --------------------------------------------------------------------------- pure: time / fields

_AIS_TIME_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})[ T]"
    r"(?P<time>\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<frac>\d+))?"
    r"\s*(?P<offset>[+-]\d{2}:?\d{2}|Z)?")


def parse_aisstream_time(value):
    """Parse AISStream's Go-format MetaData.time_utc into an aware UTC datetime.

    Handles e.g. "2023-05-10 11:46:52.509865357 +0000 UTC" (nanosecond fraction, trailing tz
    name), plain ISO-8601, and epoch seconds. The nanosecond fraction is TRUNCATED to
    microseconds (Python's max). Raises ValueError on anything unparseable.
    """
    if value is None or value == "":
        raise ValueError("empty timestamp")
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    s = str(value).strip()
    m = _AIS_TIME_RE.match(s)
    if not m:
        # last resort: epoch seconds as a bare number
        return datetime.fromtimestamp(float(s), tz=timezone.utc)
    frac = (m.group("frac") or "")[:6].ljust(6, "0") if m.group("frac") else "000000"
    offset = m.group("offset")
    if offset in (None, "", "Z"):
        off = "+00:00"
    elif ":" in offset:
        off = offset
    else:
        off = offset[:3] + ":" + offset[3:]   # +0000 -> +00:00
    iso = f"{m.group('date')}T{m.group('time')}.{frac}{off}"
    return datetime.fromisoformat(iso).astimezone(timezone.utc)


def clean_sog(value):
    """Speed over ground in knots, or None if missing / the 102.3 not-available sentinel / junk."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v < 0 or v >= SOG_NA:
        return None
    return round(v, 2)


def clean_cog(value):
    """Course over ground in degrees, or None if missing / the 360.0 not-available sentinel / junk."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v < 0 or v >= COG_NA:
        return None
    return round(v, 2)


def normalize_aisstream_message(msg):
    """One AISStream message dict -> a normalized AIS contract record
    {mmsi, lon, lat, timestamp, sog, cog, name?}, or None if it is not a usable position.

    Prefers MetaData (present on every message and carrying the authoritative lat/lon/time),
    falling back to the inner PositionReport. Returns None (skip) rather than raising, so a
    malformed frame never aborts a collection run.
    """
    if not isinstance(msg, dict):
        return None
    meta = msg.get("MetaData") or {}
    body = msg.get("Message") or {}
    report = body.get("PositionReport") or body.get("StandardClassBPositionReport") or {}

    mmsi = meta.get("MMSI") or report.get("UserID")
    lat = meta.get("latitude", report.get("Latitude"))
    lon = meta.get("longitude", report.get("Longitude"))
    ts = meta.get("time_utc")
    if mmsi in (None, "") or lat is None or lon is None or ts in (None, ""):
        return None
    try:
        when = parse_aisstream_time(ts)
        lat = float(lat)
        lon = float(lon)
    except (ValueError, TypeError, OverflowError, OSError):
        return None
    # AIS lat/lon not-available sentinels (91 / 181) and plain out-of-range -> skip
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        return None
    rec = {"mmsi": str(mmsi), "lon": round(lon, 7), "lat": round(lat, 7),
           "timestamp": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
           "sog": clean_sog(report.get("Sog")), "cog": clean_cog(report.get("Cog"))}
    name = meta.get("ShipName")
    if name and str(name).strip():
        rec["name"] = str(name).strip()   # extra identity field; the contract ignores unknown keys
    return rec


# --------------------------------------------------------------------------- IO / collection

def write_atomic(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def apply_dedup(records, mode):
    """mode 'latest' -> keep only the newest ping per MMSI; 'none' -> keep all (order preserved)."""
    if mode != "latest":
        return records
    latest = {}
    for r in records:
        prev = latest.get(r["mmsi"])
        if prev is None or r["timestamp"] >= prev["timestamp"]:
            latest[r["mmsi"]] = r
    return list(latest.values())


def reconnect_delay(consecutive_failures, backoff_s, max_backoff_s):
    """0s on the first retry, then exponential backoff capped at max_backoff_s.

    Mirrors scripts/_http.py's 0s/2s/4s shape so a websocket outage is not hammered any harder
    than an HTTP one. Pure function so the schedule is testable without sleeping."""
    if consecutive_failures <= 1:
        return 0.0
    return min(backoff_s * (2 ** (consecutive_failures - 2)), max_backoff_s)


def collect(subscribe_msg, duration_s, max_messages, recv_timeout,
            reconnect=True, backoff_s=2.0, max_backoff_s=30.0, stability_s=30.0, sleep=None):
    """Websocket shell: connect, subscribe, and collect normalized position records until the
    deadline / max_messages / Ctrl-C, reconnecting across dropped connections.

    Capture semantics (decided 2026-08-03, ROADMAP W8):
      * `--duration` is **wall-clock**: the deadline is fixed at entry, so a reconnect gap eats
        into the window rather than extending it. A capture therefore still ends when the caller
        expects, which is what makes it safe to pair against a known satellite overpass time.
      * Outages are **recorded, not hidden**: the returned stats carry `connected_s`, a `gaps`
        list and a `reconnects` count, so a consumer can see the capture is holey and where.
        This is the same honesty property `ais_fishing.py` applies to AIS gaps (unobserved time
        is excluded rather than assumed idle).
      * Exact duplicates are dropped on the `(mmsi, timestamp)` key, since a reconnect can in
        principle re-deliver a ping. This is distinct from `--dedup latest`, which is about
        keeping one ping per vessel and is applied later in `apply_dedup`.
      * The backoff resets only after a session that stayed up for `stability_s`, so a flapping
        endpoint (one frame, then drop, repeatedly) escalates the delay instead of being
        reconnected at 0s forever.

    An AISStream `error` frame (bad key / bad bbox) still fails fast and is never retried.
    Returns (records, stats). Lazy-imports websocket-client so the module (and its offline
    tests) import without the dependency installed."""
    try:
        import websocket  # websocket-client; see requirements-ais.txt
    except ImportError as exc:
        raise RuntimeError("websocket-client is not installed. Install the AIS deps: "
                           "pip install -r requirements-ais.txt") from exc

    sleep = sleep or time.sleep
    records = []
    seen_keys = set()
    seen_frames = skipped = errors = duplicates = reconnects = 0
    gaps = []
    connected_s = 0.0
    consecutive_failures = 0
    interrupted = False

    started_monotonic = time.monotonic()
    deadline = started_monotonic + duration_s
    disconnected_at = None          # (monotonic, utc) of the last drop, for gap accounting

    def _utc():
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    while time.monotonic() < deadline:
        if max_messages and len(records) >= max_messages:
            break

        # ---- (re)connect ------------------------------------------------------------------
        ws = None
        try:
            ws = websocket.create_connection(AISSTREAM_URL, timeout=recv_timeout)
            ws.send(json.dumps(subscribe_msg))   # AISStream requires the sub within 3 s of connect
        except KeyboardInterrupt:
            interrupted = True
            break
        except Exception as exc:                 # connect/handshake failure -> treat as an outage
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass
            if not reconnect:
                raise
            consecutive_failures += 1
            if disconnected_at is None:
                disconnected_at = (time.monotonic(), _utc())
            delay = reconnect_delay(consecutive_failures, backoff_s, max_backoff_s)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            print(f"  AIS: connect failed ({type(exc).__name__}); retrying in {delay:.0f}s")
            try:
                sleep(min(delay, remaining))
            except KeyboardInterrupt:
                interrupted = True
                break
            continue

        # a connection was established -> close any open gap
        if disconnected_at is not None:
            gap_monotonic, gap_from = disconnected_at
            gaps.append({"from": gap_from, "to": _utc(),
                         "seconds": round(time.monotonic() - gap_monotonic, 1)})
            disconnected_at = None
            reconnects += 1
        session_start = time.monotonic()

        # ---- receive until this session dies or the deadline passes -----------------------
        try:
            while time.monotonic() < deadline:
                if max_messages and len(records) >= max_messages:
                    break
                try:
                    raw = ws.recv()
                except websocket.WebSocketTimeoutException:
                    continue   # quiet feed; loop back and re-check the deadline
                except websocket.WebSocketConnectionClosedException:
                    break
                if not raw:
                    continue
                seen_frames += 1
                try:
                    msg = json.loads(raw)
                except (ValueError, TypeError):
                    errors += 1
                    continue
                if isinstance(msg, dict) and msg.get("error"):
                    # bad key / bbox -> fail fast, never retry
                    raise RuntimeError(f"AISStream error: {msg['error']}")
                rec = normalize_aisstream_message(msg)
                if rec is None:
                    skipped += 1
                    continue
                key = (rec["mmsi"], rec["timestamp"])
                if key in seen_keys:
                    duplicates += 1
                    continue
                seen_keys.add(key)
                records.append(rec)
        except KeyboardInterrupt:
            interrupted = True
        finally:
            connected_s += time.monotonic() - session_start
            try:
                ws.close()
            except Exception:
                pass

        if interrupted:
            print("\n  AIS: interrupted; writing what was collected")
            break
        if max_messages and len(records) >= max_messages:
            break
        if time.monotonic() >= deadline:
            break
        if not reconnect:
            print("  AIS: websocket closed by server; stopping early")
            break

        # Session ended before the deadline -> open a gap and go round again.
        # Reset the backoff only if the session was STABLE, not merely because it delivered a
        # frame: a flapping endpoint that serves one frame and drops would otherwise reset the
        # counter every cycle and be reconnected at 0s delay forever.
        if time.monotonic() - session_start >= stability_s:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
        disconnected_at = (time.monotonic(), _utc())
        delay = reconnect_delay(consecutive_failures, backoff_s, max_backoff_s)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        print(f"  AIS: connection dropped; reconnecting in {delay:.0f}s "
              f"({remaining / 60:.1f} min left in the window)")
        try:
            sleep(min(delay, remaining))
        except KeyboardInterrupt:
            interrupted = True
            break

    # a capture that ended mid-outage still owes the record a closed gap
    if disconnected_at is not None:
        gap_monotonic, gap_from = disconnected_at
        gaps.append({"from": gap_from, "to": _utc(),
                     "seconds": round(time.monotonic() - gap_monotonic, 1)})

    elapsed_s = time.monotonic() - started_monotonic
    return records, {
        "frames": seen_frames,
        "skipped": skipped,
        "errors": errors,
        "duplicates": duplicates,
        "reconnects": reconnects,
        "gaps": gaps,
        "connected_s": round(connected_s, 1),
        "elapsed_s": round(elapsed_s, 1),
        "interrupted": interrupted,
    }


def main(argv=None):
    args = parse_args(argv)
    bbox, label, wdpa_meta = resolve_bbox(args)

    api_key = os.getenv("AISSTREAM_API_KEY")
    subscribe_msg = build_subscribe(api_key or "<AISSTREAM_API_KEY>", bbox, args.message_types, args.mmsi)

    print("AIS INGEST (AISStream):")
    print(f"  bbox(lon/lat)={tuple(round(v, 5) for v in bbox)}  label={label}")
    print(f"  aisstream_bbox(lat/lon)={bbox_to_aisstream(bbox)}")
    print(f"  message_types={args.message_types}  mmsi_filter={args.mmsi or '-'}")
    print(f"  duration={args.duration}s  max_messages={args.max_messages or 'unlimited'}  dedup={args.dedup}")

    if args.dry_run:
        printable = dict(subscribe_msg, APIKey=("<set>" if api_key else "<MISSING AISSTREAM_API_KEY>"))
        print("  DRY RUN — subscription message (not sent):")
        print("    " + json.dumps(printable))
        return 0

    if not api_key:
        raise RuntimeError("Missing AISSTREAM_API_KEY (put it in a .env at the repo root or scripts/). "
                           "Register free at https://aisstream.io/. Use --dry-run to test wiring without a key.")

    start = datetime.now(timezone.utc)
    print(f"  connecting {AISSTREAM_URL} at {start.strftime('%Y-%m-%dT%H:%M:%SZ')} ...")
    records, stats = collect(subscribe_msg, args.duration, args.max_messages, args.recv_timeout,
                             reconnect=not args.no_reconnect,
                             backoff_s=args.reconnect_backoff, max_backoff_s=args.max_backoff)
    kept = apply_dedup(records, args.dedup)
    unique_mmsi = len({r["mmsi"] for r in kept})
    print(f"  frames={stats['frames']}  positions={len(records)}  "
          f"kept={len(kept)} ({unique_mmsi} vessels)  skipped={stats['skipped']}  errors={stats['errors']}"
          + (f"  duplicates={stats['duplicates']}" if stats["duplicates"] else ""))
    # Say plainly when the window was not fully covered - a holey capture that looks complete is
    # the failure mode this reporting exists to prevent (ROADMAP W8).
    if stats["reconnects"] or stats["gaps"]:
        lost = sum(g["seconds"] for g in stats["gaps"])
        print(f"  connection: {stats['reconnects']} reconnect(s), {len(stats['gaps'])} gap(s), "
              f"{lost:.0f}s disconnected -> connected {stats['connected_s']:.0f}s "
              f"of {stats['elapsed_s']:.0f}s elapsed")
    if stats["interrupted"]:
        print(f"  NOTE: capture ended early ({stats['elapsed_s']:.0f}s of the requested "
              f"{args.duration:.0f}s) - the record below says so.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_path = OUTPUT_DIR / f"ais_positions_{ts}_{slugify(label)}.json"
    doc = {
        "generated_utc": ts,
        "source": "aisstream",
        "bbox_wgs84": [round(v, 7) for v in bbox],
        "wdpa": wdpa_meta,
        "collected_from_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_s": args.duration,              # REQUESTED window (wall-clock)
        # --- what actually happened (ROADMAP W8) ---------------------------------------------
        # duration_s alone used to be written unconditionally, so a capture truncated by a dropped
        # socket or Ctrl-C still claimed the full window. These fields make a holey capture legible:
        # a consumer can check whether the moment it cares about (e.g. scene.datetime) was covered.
        "elapsed_s": stats["elapsed_s"],          # wall-clock actually spent
        "connected_s": stats["connected_s"],      # of which genuinely connected
        "reconnects": stats["reconnects"],
        "gaps": stats["gaps"],                    # [{from, to, seconds}] - unobserved intervals
        "truncated": bool(stats["interrupted"]),  # Ctrl-C / early stop
        "message_types": args.message_types,
        "dedup": args.dedup,
        "duplicates_dropped": stats["duplicates"],
        "frames_seen": stats["frames"],
        "position_count": len(kept),
        "unique_mmsi": unique_mmsi,
        "positions": kept,   # <- the normalized AIS contract fuse_violations.py --ais reads
    }
    write_atomic(out_path, json.dumps(doc, indent=2))
    print(f"\nWrote {len(kept)} positions -> {out_path}")
    print(f"Next: python scripts/fuse_violations.py --run <run> --ais {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
