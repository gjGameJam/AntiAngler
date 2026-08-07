import argparse
import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv   # optional; only used to pick up GEO_* knobs from .env
except ImportError:                  # keep the module importable (and offline-testable) without it
    def load_dotenv(*_args, **_kwargs):
        return False

# The objective of this script (AIS-track MPA geofencing, P3):
# - Read a normalized AIS positions file (the fuse_violations.py --ais contract) and an MPA polygon
# - Flag every vessel whose AIS positions fall INSIDE the no-take MPA, independent of any satellite
#   detection - the COOPERATIVE counterpart to the dark-vessel signal
# - Emit the same violation_events.json schema fuse_violations.py writes, so report_violations.py and
#   alert_violations.py rank AIS-only events alongside detection-based ones with no special-casing
#
# WHY this exists: fuse_violations answers "is this DETECTION on an AIS track?" - it can only see
# vessels the detector found. A vessel transmitting from inside a strict reserve is evidence on its
# own, and it is evidence we already have, for free, in every AIS file we capture. This closes the
# other half of the geofencing capability (CLAUDE.md's capability table: "intersect positions with MPA
# geofences" was only partial - detections were tagged, AIS-reported vessels were not).
#
# THE THREE STATUSES, once this exists:
#   dark      - detected by satellite, NOT transmitting AIS        (fuse_violations)
#   matched   - detected by satellite AND transmitting AIS         (fuse_violations)
#   reported  - transmitting AIS from inside the MPA, no detection (this script)
# "reported" is the weakest signal of the three: the vessel is being transparent, and many MPAs permit
# innocent passage. So it is EMITTED but scored by BEHAVIOUR, not by presence alone - a straight-line
# transit sits at --transit-weight, while a vessel whose movement looks like working gear scores up
# toward --fishing-weight. Ranking, not filtering, is the house pattern (report_violations --min-score
# hides the transits; the record still holds them).
#
# BEHAVIOUR comes from the vessel-records contract via --vessels: either ais_fishing.py (movement
# heuristic, no key) or gfw_vessels.py (the GFW API). With NO --vessels every vessel sits at the
# transit floor, which is the honest default - we have no behavioural evidence.
#
# Point-in-polygon is PURE STDLIB (ray casting, holes handled), so the geometry is offline-testable;
# only LOADING the WDPA polygon needs geopandas, and that is isolated in load_mpa_polygons() and
# lazy-imported. The --bbox path needs nothing at all.
#
# House rule: an ANALYSIS sibling. It never imports fuse_violations.py (the matcher), ais_fishing.py or
# any ingester; the small parsing/geo helpers are deliberately duplicated (docs/AUDIT.md #7).

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "ais_geofence"
DEFAULT_GPKG = REPO_ROOT / "data" / "processed" / "wdpa_marine_ia_ib.gpkg"

EARTH_RADIUS_M = 6_371_008.8   # mean Earth radius (IUGG), metres - same constant as fuse_violations

load_dotenv(SCRIPT_DIR / ".env")
load_dotenv(REPO_ROOT / ".env")


def env_or(name, default):
    val = os.getenv(name)
    return val if val not in (None, "") else default


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Flag vessels transmitting AIS from inside a no-take MPA (the cooperative "
                    "counterpart to dark-vessel detection). Emits the violation_events schema.")
    p.add_argument("--ais", type=Path, default=env_or("GEO_AIS", None),
                   help="Normalized AIS positions file (.json or .csv) - the fuse_violations --ais contract.")

    area = p.add_argument_group("MPA (provide one)")
    area.add_argument("--wdpa-id", type=float,
                      default=(lambda v: float(v) if v else None)(env_or("GEO_WDPA_ID", None)),
                      help="Select an MPA by WDPAID from the GeoPackage (true point-in-polygon).")
    area.add_argument("--wdpa-pid", default=env_or("GEO_WDPA_PID", None),
                      help="Select an MPA by WDPA_PID from the GeoPackage.")
    area.add_argument("--bbox", type=float, nargs=4, metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
                      default=None,
                      help="Rectangular AOI instead of an MPA polygon (stdlib-only; no GeoPackage).")
    area.add_argument("--gpkg", type=Path, default=Path(env_or("GEO_GPKG", str(DEFAULT_GPKG))),
                      help="WDPA GeoPackage for the --wdpa-id/--wdpa-pid lookup.")

    p.add_argument("--vessels", type=Path, default=env_or("GEO_VESSELS", None),
                   help="Vessel-records file (ais_fishing.py or gfw_vessels.py) supplying the "
                        "behaviour signal that scores a vessel above the transit floor.")

    score = p.add_argument_group("scoring (presence is certain; BEHAVIOUR sets the weight)")
    score.add_argument("--transit-weight", type=float, default=float(env_or("GEO_TRANSIT_WEIGHT", 0.10)),
                       help="Weight for a vessel with no behavioural evidence of fishing - i.e. an "
                            "apparent transit. Default 0.10 (below fuse_violations' 0.25 matched "
                            "floor, because there is no imagery corroborating it).")
    score.add_argument("--fishing-weight", type=float, default=float(env_or("GEO_FISHING_WEIGHT", 1.0)),
                       help="Weight at fishing_probability 1.0. Default 1.0.")
    score.add_argument("--min-score", type=float, default=float(env_or("GEO_MIN_SCORE", 0.0)),
                       help="Drop events below this score. Default 0.0 (emit all; rank, don't filter).")

    p.add_argument("--min-pings-inside", type=int, default=int(env_or("GEO_MIN_PINGS_INSIDE", 1)),
                   help="Vessels need at least this many in-MPA positions. Default 1. Raise it to "
                        "suppress single-ping GPS noise on the boundary.")
    p.add_argument("--out", type=Path, default=env_or("GEO_OUT", None),
                   help="Output path. Default data/raw/ais_geofence/geofence_events_<UTC>.json.")
    p.add_argument("--dry-run", action="store_true", help="Print the result but write no file.")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- AIS input (the contract)

def parse_timestamp(value):
    """Parse an AIS timestamp to an aware UTC datetime (ISO-8601 or epoch seconds).
    Mirrors fuse_violations.parse_timestamp - same contract."""
    if value is None or value == "":
        raise ValueError("empty timestamp")
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    s = str(value).strip()
    try:
        return datetime.fromtimestamp(float(s), tz=timezone.utc)
    except ValueError:
        pass
    iso = s.replace("Z", "+00:00").replace(" UTC", "").replace("UTC", "").strip()
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
    """One normalized AIS record, or None if it lacks mmsi/lat/lon/timestamp."""
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
    name = raw.get("name", raw.get("VesselName", raw.get("shipname")))
    return {"mmsi": str(mmsi), "lon": lon, "lat": lat, "when": when,
            "sog": _coerce_float(raw.get("sog", raw.get("Sog", raw.get("speed")))),
            "cog": _coerce_float(raw.get("cog", raw.get("Cog", raw.get("course")))),
            "name": name or None}


def load_ais(ais_path):
    """Load + normalize the AIS positions file (.json or .csv), sorted by time."""
    path = Path(ais_path)
    if not path.exists():
        raise RuntimeError(f"AIS file not found: {path}")
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig", errors="replace") as fh:
            raw_rows = list(csv.DictReader(fh))
    else:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(doc, dict):
            raw_rows = doc.get("positions") or doc.get("pings") or doc.get("data") or []
        elif isinstance(doc, list):
            raw_rows = doc
        else:
            raise RuntimeError(f"Unrecognized AIS JSON shape in {path} (want a list or {{positions:[...]}}).")
    pings, skipped = [], 0
    for row in raw_rows:
        ping = normalize_ping(row)
        if ping is None:
            skipped += 1
        else:
            pings.append(ping)
    pings.sort(key=lambda p: p["when"])
    return pings, skipped


def load_vessels(vessels_path):
    """Vessel-records file (ais_fishing.py or gfw_vessels.py) -> {mmsi: record}. None -> {}."""
    if vessels_path is None:
        return {}
    path = Path(vessels_path)
    if not path.exists():
        raise RuntimeError(f"Vessels file not found: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = doc.get("vessels", []) if isinstance(doc, dict) else doc
    by_mmsi = {}
    for row in rows:
        mmsi = row.get("mmsi", row.get("ssvid", row.get("MMSI")))
        if mmsi not in (None, ""):
            by_mmsi[str(mmsi)] = row
    return by_mmsi


# --------------------------------------------------------------------------- geometry (pure stdlib)

def point_in_ring(lon, lat, ring):
    """Ray-casting point-in-polygon for one closed ring of (lon, lat) vertices.

    Counts crossings of a ray cast along +longitude. Points exactly on an edge are not guaranteed
    either way (the classic ambiguity) - irrelevant here, since an MPA boundary is far coarser than
    AIS position noise and --min-pings-inside exists for the boundary-jitter case."""
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        # does the edge straddle the horizontal line through the point?
        if (y1 > lat) != (y2 > lat):
            x_at = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if x_at > lon:
                inside = not inside
    return inside


def point_in_polygon(lon, lat, polygon):
    """polygon = {"exterior": [(lon,lat), ...], "holes": [[...], ...]}. Inside the exterior and
    outside every hole (an MPA can legitimately have an excluded inner zone)."""
    if not point_in_ring(lon, lat, polygon["exterior"]):
        return False
    return not any(point_in_ring(lon, lat, hole) for hole in polygon.get("holes", ()))


def point_in_any(lon, lat, polygons):
    """True if the point is inside any polygon (a WDPA record is often a MultiPolygon)."""
    return any(point_in_polygon(lon, lat, poly) for poly in polygons)


def bbox_polygon(bbox):
    """A --bbox as the same polygon structure, so both AOI paths share one code path."""
    minlon, minlat, maxlon, maxlat = bbox
    return [{"exterior": [(minlon, minlat), (maxlon, minlat), (maxlon, maxlat), (minlon, maxlat)],
             "holes": []}]


def shapely_to_polygons(geom):
    """A shapely (Multi)Polygon -> the plain-tuple structure the stdlib test uses. Keeping the
    conversion here means every downstream geometry call is dependency-free and unit-testable."""
    polys = []
    parts = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
    for part in parts:
        if part.is_empty:
            continue
        polys.append({
            "exterior": [(float(x), float(y)) for x, y in part.exterior.coords],
            "holes": [[(float(x), float(y)) for x, y in ring.coords] for ring in part.interiors],
        })
    return polys


def load_mpa_polygons(gpkg, wdpa_id, wdpa_pid):
    """Look up the MPA and return (polygons, meta). The ONLY geopandas-dependent function here."""
    import geopandas as gpd   # heavy dep; only needed for the MPA-lookup path

    if not Path(gpkg).exists():
        raise RuntimeError(f"GeoPackage not found: {gpkg} (run scripts/prep_polygons.py first)")
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
    geom = sel.geometry.union_all() if hasattr(sel.geometry, "union_all") else sel.geometry.unary_union
    meta = {"WDPAID": float(row["WDPAID"]) if row.get("WDPAID") is not None else None,
            "WDPA_PID": str(row["WDPA_PID"]), "NAME": str(row["NAME"])}
    return shapely_to_polygons(geom), meta


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres between two WGS84 points."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


# --------------------------------------------------------------------------- events

def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


VESSEL_SOURCE_EVIDENCE = {"gfw": "GFW", "ais-heuristic": "AIS-heuristic"}


def vessel_evidence_label(vessel):
    """Evidence tag for a joined vessel record, from its `source`. Mirrors
    fuse_violations.vessel_evidence_label so an event never misattributes its fishing signal."""
    src = str(vessel.get("source") or "").strip().lower()
    if not src:
        return "GFW"
    return VESSEL_SOURCE_EVIDENCE.get(src, src.upper())


def reported_score(transit_weight, fishing_weight, fishing_probability, iuu_listed, presence=1.0):
    """Weight for a REPORTED (cooperative, undetected) vessel.

    Mirrors fuse_violations.matched_score, one tier lower: with no behavioural evidence the weight is
    transit_weight (an apparent transit, and many MPAs permit innocent passage); as fishing_probability
    rises it interpolates toward fishing_weight; an IUU-listed vessel escalates outright. A NULL
    probability means 'we could not tell', which is no evidence, so it stays at the floor."""
    w = transit_weight
    if fishing_probability is not None:
        w = transit_weight + (fishing_weight - transit_weight) * _clamp01(fishing_probability)
    if iuu_listed:
        w = max(w, fishing_weight)
    return round(presence * w, 4)


def build_event(idx, mmsi, inside, mpa, vessel, transit_weight, fishing_weight):
    """One vessel's in-MPA presence as a violation_events record.

    Unlike the satellite path, AIS gives MULTIPLE timestamped positions, so entry_time/exit_time and
    duration_hours are genuinely computable here - the one place the original proposal's duration term
    is not null. presence_confidence is 1.0 because the vessel is self-reporting its position; the
    doubt in this event is about the BEHAVIOUR being a violation, and that lives in the weight."""
    fishing_probability = fishing_hours = iuu_listed = None
    vessel_name = flag = gear_type = imo = gfw_vessel_id = None
    if vessel:
        fishing_probability = vessel.get("fishing_probability")
        fishing_hours = vessel.get("fishing_hours")
        iuu_listed = vessel.get("iuu_listed")
        vessel_name = vessel.get("shipname") or vessel.get("name")
        flag = vessel.get("flag")
        gear_type = vessel.get("geartype") or vessel.get("gear_type")
        imo = vessel.get("imo")
        gfw_vessel_id = vessel.get("vessel_id") or vessel.get("vesselId")
    if vessel_name is None:
        vessel_name = next((p["name"] for p in inside if p.get("name")), None)

    first, last = inside[0], inside[-1]
    duration_h = round((last["when"] - first["when"]).total_seconds() / 3600.0, 4)
    centroid = [round(sum(p["lon"] for p in inside) / len(inside), 6),
                round(sum(p["lat"] for p in inside) / len(inside), 6)]

    evidence = ["AIS"]
    if vessel:
        evidence.append(vessel_evidence_label(vessel))

    return {
        "event_id": f"evt_{idx:05d}",
        "detection_id": None,                    # no satellite detection - this IS the distinction
        "ais_status": "reported",                # vs "dark" / "matched" from fuse_violations
        "vessel_id": str(mmsi),
        "mmsi": str(mmsi),
        "gfw_vessel_id": gfw_vessel_id,
        "vessel_name": vessel_name,
        "flag": flag,
        "gear_type": gear_type,
        "imo": imo,
        "iuu_listed": iuu_listed,
        "mpa_id": (mpa or {}).get("WDPAID"),
        "mpa_name": (mpa or {}).get("NAME"),
        "centroid_wgs84": centroid,
        "chip": None,                            # no imagery backs this event
        "entry_time": first["when"].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "exit_time": last["when"].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_hours": duration_h,            # computable here, unlike the instantaneous scene
        "distance_from_port_km": None,
        "presence_confidence": 1.0,              # self-reported position
        "fishing_hours": fishing_hours,
        "fishing_probability": fishing_probability,
        "violation_score": reported_score(transit_weight, fishing_weight,
                                          fishing_probability, iuu_listed),
        "ais_match": {
            "mmsi": str(mmsi),
            "pings_inside": len(inside),
            "first_inside": first["when"].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "last_inside": last["when"].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "track_km_inside": round(sum(
                haversine_m(a["lon"], a["lat"], b["lon"], b["lat"])
                for a, b in zip(inside, inside[1:])) / 1000.0, 4),
            "sog_first": first.get("sog"),
            "sog_last": last.get("sog"),
        },
        "evidence": evidence,
    }


def geofence(pings, polygons, mpa, args, vessels_by_mmsi=None):
    """Every vessel with >= --min-pings-inside positions inside the MPA, as ranked events."""
    vessels_by_mmsi = vessels_by_mmsi or {}
    inside_by_mmsi = {}
    for ping in pings:
        if point_in_any(ping["lon"], ping["lat"], polygons):
            inside_by_mmsi.setdefault(ping["mmsi"], []).append(ping)

    events = []
    for mmsi, inside in inside_by_mmsi.items():
        if len(inside) < args.min_pings_inside:
            continue
        events.append(build_event(0, mmsi, inside, mpa, vessels_by_mmsi.get(str(mmsi)),
                                  args.transit_weight, args.fishing_weight))
    events = [e for e in events if e["violation_score"] >= args.min_score]
    events.sort(key=lambda e: -e["violation_score"])
    for i, event in enumerate(events):      # event_id encodes rank, as in fuse_violations
        event["event_id"] = f"evt_{i:05d}"
    return events


def build_geojson(events):
    """Point FeatureCollection of the reported vessels (QGIS), mirroring the dark .geojson."""
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": e["centroid_wgs84"]},
            "properties": {k: v for k, v in e.items() if k != "centroid_wgs84"},
        } for e in events],
    }


# --------------------------------------------------------------------------- IO / orchestration

def write_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return path


def resolve_area(args):
    """(polygons, mpa_meta, label) from --wdpa-id/--wdpa-pid or --bbox."""
    has_wdpa = args.wdpa_id is not None or args.wdpa_pid is not None
    if has_wdpa and args.bbox:
        raise RuntimeError("Provide --wdpa-id/--wdpa-pid OR --bbox, not both.")
    if has_wdpa:
        polygons, meta = load_mpa_polygons(args.gpkg, args.wdpa_id, args.wdpa_pid)
        return polygons, meta, meta["NAME"]
    if args.bbox:
        minlon, minlat, maxlon, maxlat = args.bbox
        if minlon >= maxlon or minlat >= maxlat:
            raise RuntimeError(f"--bbox must be minLon minLat maxLon maxLat, got {args.bbox}")
        return bbox_polygon(args.bbox), None, "bbox"
    raise RuntimeError("Provide an area: --wdpa-id <id> (true polygon) or --bbox <minLon minLat maxLon maxLat>.")


def main(argv=None):
    args = parse_args(argv)
    if not args.ais:
        raise RuntimeError("Provide --ais <normalized AIS positions file> "
                           "(from aisstream_fetch.py or marinecadastre_fetch.py).")
    if args.transit_weight > args.fishing_weight:
        raise RuntimeError(f"--transit-weight ({args.transit_weight}) must not exceed "
                           f"--fishing-weight ({args.fishing_weight}).")

    polygons, mpa, label = resolve_area(args)
    pings, skipped = load_ais(args.ais)
    if skipped:
        print(f"  AIS: skipped {skipped} unusable rows (missing mmsi/lat/lon/timestamp)")
    if not pings:
        raise RuntimeError(f"No usable AIS positions in {args.ais}.")
    vessels_by_mmsi = load_vessels(args.vessels)

    events = geofence(pings, polygons, mpa, args, vessels_by_mmsi)
    scored = sum(1 for e in events if e["fishing_probability"] is not None)

    print("AIS GEOFENCE (cooperative vessels inside the MPA):")
    print(f"  ais={args.ais}  pings={len(pings)}  vessels={len({p['mmsi'] for p in pings})}")
    print(f"  area={label}" + (f"  WDPAID={mpa['WDPAID']}" if mpa else ""))
    print(f"  vessels inside: {len(events)}  (behaviour-scored {scored}, "
          f"{len(events) - scored} at the transit floor)")
    if not vessels_by_mmsi:
        print("  NOTE: no --vessels file, so every event sits at the transit floor "
              f"({args.transit_weight}). Run ais_fishing.py to score by behaviour.")
    for event in events[:10]:
        print(f"    {event['mmsi']:>10s}  score={event['violation_score']:.2f}  "
              f"pings_inside={event['ais_match']['pings_inside']}  "
              f"duration_h={event['duration_hours']}  name={event['vessel_name']}")
    if len(events) > 10:
        print(f"    ... and {len(events) - 10} more")

    if args.dry_run:
        print("\nDRY RUN - nothing written.")
        return 0

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_path = Path(args.out) if args.out else OUTPUT_DIR / f"geofence_events_{ts}.json"
    doc = {
        "generated_utc": ts,
        "run_dir": None,                 # not tied to an imagery run - AIS alone produced this
        "source_detections": None,
        "ais_file": str(args.ais),
        "vessels_file": str(args.vessels) if args.vessels else None,
        "sensor": "AIS",
        "scene_datetime": None,
        "params": {"transit_weight": args.transit_weight, "fishing_weight": args.fishing_weight,
                   "min_pings_inside": args.min_pings_inside, "min_score": args.min_score},
        "wdpa": mpa,
        "ais_ping_count": len(pings),
        "counts": {"vessels_seen": len({p["mmsi"] for p in pings}), "reported": len(events),
                   "behaviour_scored": scored},
        "violation_events": events,      # <- the same schema report_violations.py ranks
    }
    write_atomic(out_path, json.dumps(doc, indent=2))
    write_atomic(out_path.with_suffix(".geojson"), json.dumps(build_geojson(events), indent=2))
    print(f"\nWrote {len(events)} violation_events -> {out_path} (+ .geojson)")
    print(f"Next: python scripts/report_violations.py --events {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
