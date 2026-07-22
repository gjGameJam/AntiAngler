import argparse
import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

# The objective of this script (pipeline stage 4 - "fuse", ROADMAP Phase 3):
# - Read a detect_boats.py run directory (detections.json + the sibling manifest.json)
# - Read a NORMALIZED AIS positions file (our own source-agnostic schema; see below)
# - For each IN-MPA detection, decide dark vs matched with a VELOCITY-AWARE gate:
#     a satellite acquisition is instantaneous and almost never coincides with an AIS
#     ping, so a fixed match radius is wrong - it over-flags fast vessels and under-flags
#     during long AIS gaps. Instead each candidate ping gets a feasible-movement envelope
#     reach = speed * |t_detection - t_ping| + slack, and it matches iff the great-circle
#     distance from the detection to the ping is <= reach. This is the "feasible movement
#     envelope" the operational systems use (Skylight; SAR<->AIS match ~1 km at coincidence).
# - An UNMATCHED in-MPA detection is a DARK candidate - the product signal (detection - AIS).
#   A MATCHED one is an identified (cooperative) vessel, still a candidate violation inside a
#   no-take IUCN Ia/Ib MPA, but not a dark one.
# - OPTIONALLY (--vessels, ROADMAP Phase 3 step 3), a normalized GFW vessel-records file from
#   scripts/gfw_vessels.py enriches the MATCHED events by MMSI: it fills identity (name/flag/
#   gear/imo) and fishing_hours/fishing_probability, and lifts the violation_score when GFW shows
#   the identified vessel was FISHING in the MPA (or is IUU-listed) - a confirmed fishing violation
#   scores as high as a dark one. This is a SEPARATE optional join (gfw_vessels.py is its own
#   ingester); with no --vessels the verdicts and scores are unchanged (the GFW fields stay null).
# - Emit one violation_events.json (+ .geojson of dark points for QGIS) per run, using the
#   violation_events schema specified in CLAUDE.md. Atomic writes (.tmp -> replace), same as
#   every other stage.
#
# This is the "dark-vessel = detection - AIS" reframe: it inverts the metric from the
# detector's uncatchable RECALL (capped by the 10 m GSD) to MISMATCH PRECISION, which the join
# improves - a false positive that happens to sit on a matching AIS track is filtered out, and
# a real dark vessel with no AIS survives. See docs/ROADMAP.md Phase 3.
#
# House rule: one script per data stream. This is the FUSION stage; it only READS the run dir
# and a normalized AIS file. Live AIS ingestion (AISStream websocket, GFW Vessels/Insights,
# Skylight) belongs in its own ingester module that normalizes into the schema below - NOT here.
# Pure stdlib on purpose (no torch/rasterio/shapely): inside_mpa + centroid_wgs84 are already
# computed in detections.json, so the matcher runs anywhere and is testable offline.
#
# ---------------------------------------------------------------------------
# Normalized AIS input contract (--ais FILE, .json or .csv)
# ---------------------------------------------------------------------------
# JSON: either a bare list, or {"positions": [ ... ]}, of objects:
#   { "mmsi": 123456789,          # int or str, the vessel identity
#     "lon": 12.34, "lat": 56.78, # WGS84 decimal degrees
#     "timestamp": "2026-07-14T10:23:45Z",  # ISO-8601 UTC (Z or +00:00), or epoch seconds
#     "sog": 8.2,                 # OPTIONAL speed over ground, KNOTS (null/missing/0 -> max-speed cap)
#     "cog": 141.0 }              # OPTIONAL course over ground, DEGREES (carried, not used in the gate)
# CSV: a header row with columns mmsi,lon,lat,timestamp[,sog,cog] (order-independent).
#
# Mapping from a live feed to this contract (done by the future ingester, documented here so the
# field choice is legible) - AISStream PositionReport:
#   mmsi <- MetaData.MMSI ; lat/lon <- MetaData.latitude/longitude (or Message.PositionReport.
#   Latitude/Longitude) ; timestamp <- MetaData.time_utc ; sog <- Message.PositionReport.Sog
#   (knots) ; cog <- Message.PositionReport.Cog (degrees). AISStream sentinels 511 (heading) /
#   valued Sog are passed through as-is; a Sog of 0 or absent triggers the max-speed fallback.

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

EARTH_RADIUS_M = 6_371_008.8   # mean Earth radius (IUGG), metres
KNOTS_TO_M_S = 0.514444        # 1 knot = 0.514444 m/s


def env_or(name, default):
    val = os.getenv(name)
    return val if val not in (None, "") else default


def truthy(val):
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Fuse a detect_boats run against AIS positions -> dark-vessel violation_events "
                    "(ROADMAP Phase 3). Velocity-aware, pure stdlib.")
    p.add_argument("--run", type=Path, default=env_or("FUSE_RUN", None),
                   help="A detect_boats run directory (contains detections.json + manifest.json).")
    p.add_argument("--ais", type=Path, default=env_or("FUSE_AIS", None),
                   help="Normalized AIS positions file (.json or .csv). See the module header for the "
                        "schema: [{mmsi, lon, lat, timestamp, sog?, cog?}, ...].")
    p.add_argument("--dt-minutes", type=float, default=float(env_or("FUSE_DT_MINUTES", 30.0)),
                   help="Time half-window: only AIS pings within +/- this many minutes of the scene "
                        "acquisition are considered. Default 30 (SAR/optical acquisition rarely "
                        "coincides with a ping).")
    p.add_argument("--slack-m", type=float, default=float(env_or("FUSE_SLACK_M", 500.0)),
                   help="Additive slack (metres) on the feasible-movement envelope, absorbing AIS "
                        "position error + detection geolocation error. Default 500 (operational "
                        "SAR<->AIS coincident-match thresholds are ~1 km).")
    p.add_argument("--max-speed-kn", type=float, default=float(env_or("FUSE_MAX_SPEED_KN", 25.0)),
                   help="Fallback speed cap (knots) used to size the reach when a ping's sog is "
                        "missing or zero. Default 25 (a fast vessel could plausibly reach this far).")
    p.add_argument("--min-conf", type=float, default=float(env_or("FUSE_MIN_CONF", 0.0)),
                   help="Drop detections below this confidence before fusing. Default 0.0 (keep all; "
                        "the detector's operating point already thresholds).")
    p.add_argument("--all-detections", action="store_true",
                   default=truthy(env_or("FUSE_ALL_DETECTIONS", "")),
                   help="Consider ALL detections, not just inside_mpa=true. Use for bbox runs with no "
                        "MPA polygon (inside_mpa is null) or to audit the full scene.")
    p.add_argument("--dark-only", action="store_true", default=truthy(env_or("FUSE_DARK_ONLY", "")),
                   help="Emit only DARK events (drop AIS-matched, identified vessels). Default off: "
                        "matched in-MPA vessels are still candidate violations, kept with lower score.")
    p.add_argument("--matched-weight", type=float, default=float(env_or("FUSE_MATCHED_WEIGHT", 0.25)),
                   help="Violation-score multiplier for an AIS-MATCHED (cooperative, identified) "
                        "event. Dark events score at full presence_confidence; matched ones are "
                        "down-weighted by this factor for ranking. Default 0.25.")
    p.add_argument("--vessels", type=Path, default=env_or("FUSE_VESSELS", None),
                   help="OPTIONAL normalized GFW vessel-records file (from scripts/gfw_vessels.py) to "
                        "enrich MATCHED events by MMSI: fills vessel_name/flag/gear_type/imo and "
                        "fishing_hours/fishing_probability, and lifts violation_score toward "
                        "--fishing-weight when GFW shows the identified vessel was fishing (or is "
                        "IUU-listed). Absent -> no enrichment, identical to the AIS-only behavior.")
    p.add_argument("--fishing-weight", type=float, default=float(env_or("FUSE_FISHING_WEIGHT", 1.0)),
                   help="Violation-score multiplier for a matched vessel that GFW confirms was FISHING "
                        "in the MPA (fishing_probability 1.0). The matched score interpolates from "
                        "--matched-weight (probability 0) to this (probability 1). Default 1.0 — a "
                        "confirmed-fishing identified vessel in a no-take MPA is as strong a signal as "
                        "a dark one. Only used when --vessels supplies fishing_probability.")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- IO / parsing

def load_detections(run_dir):
    dpath = run_dir / "detections.json"
    if not dpath.exists():
        raise RuntimeError(f"No detections.json in {run_dir} - run detect_boats.py (or sar_seed.py) first.")
    return json.loads(dpath.read_text())


def load_vessels(vessels_path):
    """Load a normalized GFW vessel-records file (scripts/gfw_vessels.py output) into a dict keyed
    by string MMSI. Accepts a bare list or {"vessels": [...]}. Each record may carry shipname/flag/
    geartype/imo/fishing_hours/fishing_probability/iuu_listed. Absent path -> {} (no enrichment).
    Later records for the same MMSI win (the writer already dedups; this is just defensive)."""
    if vessels_path is None:
        return {}
    path = Path(vessels_path)
    if not path.exists():
        raise RuntimeError(f"Vessels file not found: {path}")
    doc = json.loads(path.read_text())
    rows = doc.get("vessels", []) if isinstance(doc, dict) else doc
    by_mmsi = {}
    for row in rows:
        mmsi = row.get("mmsi", row.get("ssvid", row.get("MMSI")))
        if mmsi in (None, ""):
            continue
        by_mmsi[str(mmsi)] = row
    return by_mmsi


def load_manifest(run_dir):
    """Optional - only for the evidence/modality tag (provider/collection). Absent is fine."""
    mpath = run_dir / "manifest.json"
    if not mpath.exists():
        return {}
    try:
        return json.loads(mpath.read_text())
    except (ValueError, OSError):
        return {}


def parse_timestamp(value):
    """Parse an AIS timestamp to an aware UTC datetime. Accepts ISO-8601 (with 'Z' or offset;
    naive is assumed UTC) or epoch seconds (int/float/numeric string)."""
    if value is None or value == "":
        raise ValueError("empty timestamp")
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    s = str(value).strip()
    # epoch seconds as a bare number
    try:
        return datetime.fromtimestamp(float(s), tz=timezone.utc)
    except ValueError:
        pass
    iso = s.replace("Z", "+00:00")
    # tolerate a trailing 'UTC' and space-separated date/time
    iso = iso.replace(" UTC", "").replace("UTC", "").strip()
    if " " in iso and "T" not in iso:
        iso = iso.replace(" ", "T", 1)
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _coerce_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_ping(raw):
    """One normalized AIS record from a raw dict. Returns None if it lacks the required fields
    (mmsi + lat/lon + a parseable timestamp), so a partial feed degrades gracefully."""
    mmsi = raw.get("mmsi", raw.get("MMSI"))
    lon = _coerce_float(raw.get("lon", raw.get("longitude", raw.get("Longitude"))))
    lat = _coerce_float(raw.get("lat", raw.get("latitude", raw.get("Latitude"))))
    ts = raw.get("timestamp", raw.get("time_utc", raw.get("time")))
    if mmsi in (None, "") or lon is None or lat is None or ts in (None, ""):
        return None
    try:
        when = parse_timestamp(ts)
    except (ValueError, OverflowError, OSError):
        return None
    sog = _coerce_float(raw.get("sog", raw.get("Sog", raw.get("speed"))))
    cog = _coerce_float(raw.get("cog", raw.get("Cog", raw.get("course"))))
    return {"mmsi": str(mmsi), "lon": lon, "lat": lat, "when": when, "sog": sog, "cog": cog}


def load_ais(ais_path):
    """Load + normalize the AIS positions file (.json or .csv). Returns a list of normalized pings
    sorted by time. Raises if the file is missing; warns (prints) on unusable rows."""
    if not ais_path.exists():
        raise RuntimeError(f"AIS file not found: {ais_path}")
    suffix = ais_path.suffix.lower()
    if suffix == ".csv":
        with ais_path.open(newline="") as fh:
            raw_rows = list(csv.DictReader(fh))
    else:
        doc = json.loads(ais_path.read_text())
        if isinstance(doc, dict):
            raw_rows = doc.get("positions") or doc.get("pings") or doc.get("data") or []
        elif isinstance(doc, list):
            raw_rows = doc
        else:
            raise RuntimeError(f"Unrecognized AIS JSON shape in {ais_path} (want a list or {{positions:[...]}}).")
    pings, skipped = [], 0
    for row in raw_rows:
        ping = normalize_ping(row)
        if ping is None:
            skipped += 1
        else:
            pings.append(ping)
    if skipped:
        print(f"  AIS: skipped {skipped} unusable rows (missing mmsi/lat/lon/timestamp)")
    pings.sort(key=lambda p: p["when"])
    return pings


# --------------------------------------------------------------------------- geo / matching

def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres between two WGS84 points."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def match_detection(det, pings, scene_dt, dt_window_s, slack_m, max_speed_kn):
    """Velocity-aware association of one detection to the AIS pings.

    For each ping within +/- dt_window_s of the acquisition, the vessel could have travelled
    reach = speed * |t_scene - t_ping| + slack from where it pinged (use the ping's own sog;
    fall back to max_speed_kn when sog is missing or zero). It matches iff the great-circle
    distance is <= reach. Returns the best match (smallest distance-minus-reach margin, i.e.
    the most comfortably-inside-envelope ping) or None. A vessel stationary at the acquisition
    instant (dt=0) collapses reach to just the slack - correct, it must be right there.
    """
    lon, lat = det["centroid_wgs84"]
    best = None
    for ping in pings:
        dt_s = abs((scene_dt - ping["when"]).total_seconds())
        if dt_s > dt_window_s:
            continue
        speed_kn = ping["sog"] if (ping["sog"] and ping["sog"] > 0) else max_speed_kn
        reach = speed_kn * KNOTS_TO_M_S * dt_s + slack_m
        dist = haversine_m(lon, lat, ping["lon"], ping["lat"])
        if dist <= reach:
            margin = dist - reach   # <=0; more negative = more comfortably inside the envelope
            if best is None or margin < best["margin"]:
                best = {"mmsi": ping["mmsi"], "distance_m": round(dist, 1),
                        "dt_seconds": round(dt_s, 1), "reach_m": round(reach, 1),
                        "margin": margin, "sog": ping["sog"], "cog": ping["cog"],
                        "ping_wgs84": [round(ping["lon"], 7), round(ping["lat"], 7)],
                        "ping_time": ping["when"].strftime("%Y-%m-%dT%H:%M:%SZ")}
    if best is not None:
        best.pop("margin", None)
    return best


# --------------------------------------------------------------------------- output

def write_atomic(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def modality_evidence(manifest, detections_doc):
    """Human-readable sensor tag for the evidence list, from the manifest provider/collection."""
    provider = (manifest.get("provider") or "").lower()
    if provider == "s1":
        return "Sentinel-1"
    if provider == "pc":
        return "Sentinel-2"
    collection = (manifest.get("collection") or "").lower()
    if "sentinel-1" in collection:
        return "Sentinel-1"
    if "sentinel-2" in collection:
        return "Sentinel-2"
    return "satellite"


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def matched_score(presence, matched_weight, fishing_weight, fishing_probability, iuu_listed):
    """Violation-score weight for a MATCHED (identified) vessel. With no GFW fishing info the weight
    is matched_weight (cooperative transit, down-weighted). When GFW supplies fishing_probability the
    weight interpolates matched_weight -> fishing_weight (a confirmed-fishing vessel in a no-take MPA
    is as strong a signal as a dark one). An IUU-listed vessel escalates to fishing_weight."""
    w = matched_weight
    if fishing_probability is not None:
        w = matched_weight + (fishing_weight - matched_weight) * _clamp01(fishing_probability)
    if iuu_listed:
        w = max(w, fishing_weight)
    return None if presence is None else round(presence * w, 4)


def build_event(idx, det, match, scene_dt_iso, mpa, sensor, matched_weight,
                vessel=None, fishing_weight=1.0):
    """Map one fused detection onto the CLAUDE.md violation_events schema.

    A single satellite acquisition is instantaneous, so entry_time == exit_time == the scene
    datetime and duration_hours is null (multi-pass track reconstruction is future work).

    Enrichment (only for a MATCHED event when a GFW vessel record is supplied via --vessels):
    fills vessel_name/flag/gear_type/imo and fishing_hours/fishing_probability from GFW, adds "GFW"
    to evidence, and lifts violation_score per matched_score(). With no vessel record the matched
    score stays presence*matched_weight and the GFW fields stay null - unchanged from AIS-only.
    """
    is_dark = match is None
    presence = det.get("confidence")

    # GFW enrichment fields (matched + vessel record only; else all None)
    vessel_name = flag = gear_type = imo = fishing_hours = fishing_probability = None
    iuu_listed = gfw_vessel_id = None
    if not is_dark and vessel:
        vessel_name = vessel.get("shipname") or vessel.get("name")
        flag = vessel.get("flag")
        gear_type = vessel.get("geartype") or vessel.get("gear_type")
        imo = vessel.get("imo")
        fishing_hours = vessel.get("fishing_hours")
        fishing_probability = vessel.get("fishing_probability")
        iuu_listed = vessel.get("iuu_listed")
        gfw_vessel_id = vessel.get("vessel_id") or vessel.get("vesselId")

    if is_dark:
        score = None if presence is None else round(presence * 1.0, 4)
    else:
        score = matched_score(presence, matched_weight, fishing_weight, fishing_probability, iuu_listed)

    evidence = [sensor] if is_dark else [sensor, "AIS"]
    if not is_dark and vessel:
        evidence.append("GFW")
    return {
        "event_id": f"evt_{idx:05d}",
        "detection_id": det.get("detection_id"),
        "ais_status": "dark" if is_dark else "matched",
        "vessel_id": None if is_dark else match["mmsi"],
        "mmsi": None if is_dark else match["mmsi"],
        "gfw_vessel_id": gfw_vessel_id,
        "vessel_name": vessel_name,
        "flag": flag,
        "gear_type": gear_type,
        "imo": imo,
        "iuu_listed": iuu_listed,
        "mpa_id": (mpa or {}).get("WDPAID"),
        "mpa_name": (mpa or {}).get("NAME"),
        "centroid_wgs84": det.get("centroid_wgs84"),
        "chip": det.get("chip"),
        "entry_time": scene_dt_iso,
        "exit_time": scene_dt_iso,
        "duration_hours": None,
        "distance_from_port_km": None,
        "presence_confidence": presence,
        "fishing_hours": fishing_hours,
        "fishing_probability": fishing_probability,
        "violation_score": score,
        "ais_match": match,          # null when dark; the matched ping details when matched
        "evidence": evidence,
    }


def build_geojson(events):
    """Point FeatureCollection of the DARK events (the product signal) for QGIS, mirroring
    detections.geojson."""
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature",
             "geometry": {"type": "Point", "coordinates": e["centroid_wgs84"]},
             "properties": {"event_id": e["event_id"], "detection_id": e["detection_id"],
                            "ais_status": e["ais_status"], "mpa_name": e["mpa_name"],
                            "presence_confidence": e["presence_confidence"],
                            "violation_score": e["violation_score"]}}
            for e in events if e["ais_status"] == "dark" and e["centroid_wgs84"]
        ],
    }


# --------------------------------------------------------------------------- orchestration

def fuse(detections_doc, pings, scene_dt, args, sensor, vessels_by_mmsi=None):
    """Core, pure function (no IO): returns (events, stats). Testable directly.
    vessels_by_mmsi (optional) enriches matched events by MMSI (from --vessels / gfw_vessels.py)."""
    dt_window_s = args.dt_minutes * 60.0
    mpa = detections_doc.get("wdpa")
    vessels_by_mmsi = vessels_by_mmsi or {}

    candidates = []
    for det in detections_doc.get("detections", []):
        if det.get("confidence") is not None and det["confidence"] < args.min_conf:
            continue
        if not args.all_detections and det.get("inside_mpa") is not True:
            continue
        candidates.append(det)

    events = []
    dark = matched = 0
    for det in candidates:
        match = match_detection(det, pings, scene_dt, dt_window_s, args.slack_m, args.max_speed_kn)
        if match is None:
            dark += 1
        else:
            matched += 1
        if args.dark_only and match is not None:
            continue
        events.append((det, match))

    # rank by violation_score desc (dark first at equal presence), assign event ids
    events.sort(key=lambda dm: (dm[1] is not None, -(dm[0].get("confidence") or 0.0)))
    scene_dt_iso = scene_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    built, enriched = [], 0
    for i, (det, match) in enumerate(events):
        vessel = vessels_by_mmsi.get(match["mmsi"]) if match else None
        if vessel:
            enriched += 1
        built.append(build_event(i, det, match, scene_dt_iso, mpa, sensor, args.matched_weight,
                                 vessel=vessel, fishing_weight=args.fishing_weight))
    stats = {"in_scope": len(candidates), "dark": dark, "matched": matched,
             "emitted": len(built), "enriched": enriched}
    return built, stats


def main(argv=None):
    args = parse_args(argv)
    if args.run is None:
        raise RuntimeError("Provide a run dir: --run data/raw/sentinel2/<run>/")
    if args.ais is None:
        raise RuntimeError("Provide a normalized AIS file: --ais positions.json (see module header for schema).")
    run_dir = args.run.resolve()

    detections_doc = load_detections(run_dir)
    manifest = load_manifest(run_dir)
    sensor = modality_evidence(manifest, detections_doc)

    scene = detections_doc.get("scene") or {}
    scene_dt_raw = scene.get("datetime")
    if not scene_dt_raw:
        raise RuntimeError("detections.json has no scene.datetime - cannot build the AIS time window. "
                           "Re-run detect_boats.py (it records scene.datetime from the manifest).")
    scene_dt = parse_timestamp(scene_dt_raw)

    pings = load_ais(args.ais.resolve())
    vessels_by_mmsi = load_vessels(args.vessels)

    print("FUSE:")
    print(f"  run={run_dir}")
    print(f"  ais={args.ais}  pings={len(pings)}")
    if args.vessels:
        print(f"  vessels={args.vessels}  gfw_records={len(vessels_by_mmsi)}")
    print(f"  scene_datetime={scene_dt.strftime('%Y-%m-%dT%H:%M:%SZ')}  sensor={sensor}")
    print(f"  dt-window=+/-{args.dt_minutes}min  slack={args.slack_m}m  max-speed={args.max_speed_kn}kn")
    print(f"  detections={detections_doc.get('count')}  "
          f"scope={'all' if args.all_detections else 'inside_mpa'}")

    events, stats = fuse(detections_doc, pings, scene_dt, args, sensor, vessels_by_mmsi)
    print(f"  in scope: {stats['in_scope']}  ->  DARK {stats['dark']}  |  matched {stats['matched']}"
          f"  (GFW-enriched {stats['enriched']})")

    doc = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_dir": run_dir.name,
        "source_detections": "detections.json",
        "ais_file": str(args.ais),
        "vessels_file": str(args.vessels) if args.vessels else None,
        "sensor": sensor,
        "scene_datetime": scene_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "params": {"dt_minutes": args.dt_minutes, "slack_m": args.slack_m,
                   "max_speed_kn": args.max_speed_kn, "min_conf": args.min_conf,
                   "all_detections": args.all_detections, "dark_only": args.dark_only,
                   "matched_weight": args.matched_weight, "fishing_weight": args.fishing_weight},
        "wdpa": detections_doc.get("wdpa"),
        "ais_ping_count": len(pings),
        "counts": {"in_scope": stats["in_scope"], "dark": stats["dark"],
                   "matched": stats["matched"], "emitted": stats["emitted"],
                   "gfw_enriched": stats["enriched"]},
        "violation_events": events,
    }
    write_atomic(run_dir / "violation_events.json", json.dumps(doc, indent=2))
    write_atomic(run_dir / "violation_events.geojson", json.dumps(build_geojson(events), indent=2))
    print(f"\nWrote {len(events)} violation_events ({stats['dark']} dark) -> "
          f"{run_dir / 'violation_events.json'} (+ .geojson of dark points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
