import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

# The objective of this script (historical AIS ingestion, PHASE3_PLAN Workstream 3C):
# - Normalize a NOAA MarineCadastre archive AIS CSV into the project's neutral AIS contract
#   ({mmsi, lon, lat, timestamp, sog?, cog?}) that fuse_violations.py --ais reads
# - Spatially filter to an AOI (--bbox / --wdpa-id) and optionally a time window (--start/--end),
#   because the national daily files are huge and a scene only needs its own AOI + acquisition window
# - Write data/raw/ais/ais_positions_<UTC>_<label>.json — same output dir/shape as aisstream_fetch
#
# WHY this exists: AISStream (aisstream_fetch.py) is LIVE-only, so historical satellite scenes can't be
# fused from it. MarineCadastre is a FREE HISTORICAL archive (US coastal waters, daily CSVs), so this
# unblocks fusing past scenes over US MPAs offline. It is a sibling AIS ingester (house rule): it only
# produces the contract; it never imports fuse_violations.py. Other historical sources (Spire, etc.)
# would each be their own sibling normalizing into the same contract.
#
# MarineCadastre CSV columns (standard AIS_YYYY_MM_DD.csv): MMSI, BaseDateTime, LAT, LON, SOG, COG,
# Heading, VesselName, IMO, CallSign, VesselType, Status, Length, Width, Draft, Cargo, TransceiverClass.
# The mapping is: mmsi←MMSI, lat←LAT, lon←LON, timestamp←BaseDateTime (naive local-less ISO → UTC),
# sog←SOG (knots), cog←COG (degrees), name←VesselName. Column matching is case-insensitive + aliased,
# and the risky bits (row normalization, time parse, AOI/time filter) are pure + offline-tested
# (scripts/tests/test_marinecadastre_fetch.py) with a synthetic CSV — no download needed.
#
# The actual archive DOWNLOAD (coast.noaa.gov zips, large) is a thin do-from-home shell (--date/--url,
# lazy `requests`); the here-runnable path is `--csv <local file>`.

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "ais"
DEFAULT_GPKG = REPO_ROOT / "data" / "processed" / "wdpa_marine_ia_ib.gpkg"
# MarineCadastre daily archive URL pattern (for the optional do-from-home download).
MC_URL_TMPL = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{year}_{month:02d}_{day:02d}.zip"


def env_or(name, default):
    val = os.getenv(name)
    return val if val not in (None, "") else default


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Normalize a NOAA MarineCadastre archive AIS CSV into the AIS contract "
                    "fuse_violations.py --ais reads, filtered to an AOI (PHASE3_PLAN 3C).")
    p.add_argument("--csv", type=Path, nargs="*", default=None,
                   help="Local MarineCadastre AIS CSV file(s) to normalize (the here-runnable path).")
    p.add_argument("--date", default=env_or("MC_DATE", None),
                   help="YYYY-MM-DD — download that day's national archive first (do-from-home; large).")
    p.add_argument("--bbox", type=float, nargs=4, metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
                   default=None, help="AOI to keep positions inside. Mutually exclusive with --wdpa-id.")
    p.add_argument("--wdpa-id", type=float, default=(lambda v: float(v) if v else None)(env_or("MC_WDPA_ID", None)),
                   help="Resolve the AOI bbox from the GeoPackage by WDPAID (needs geopandas).")
    p.add_argument("--wdpa-pid", default=env_or("MC_WDPA_PID", None),
                   help="Resolve the AOI bbox from the GeoPackage by WDPA_PID.")
    p.add_argument("--gpkg", type=Path, default=Path(env_or("MC_GPKG", str(DEFAULT_GPKG))),
                   help="MPA GeoPackage to resolve --wdpa-id/--wdpa-pid against.")
    p.add_argument("--start", default=env_or("MC_START", None),
                   help="ISO-8601 UTC window start (inclusive). Keep pings at/after this.")
    p.add_argument("--end", default=env_or("MC_END", None),
                   help="ISO-8601 UTC window end (inclusive). Keep pings at/before this.")
    p.add_argument("--label", default=env_or("MC_LABEL", None),
                   help="Filename label. Default: the MPA name or 'marinecadastre'.")
    p.add_argument("--out", type=Path, default=env_or("MC_OUT", None),
                   help="Output path. Default data/raw/ais/ais_positions_<UTC>_<label>.json.")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- pure: geo / time / row

def slugify(text):
    import re
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-") or "marinecadastre"


def point_in_bbox(lon, lat, bbox):
    min_lon, min_lat, max_lon, max_lat = bbox
    return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat


def parse_mc_time(value):
    """MarineCadastre BaseDateTime -> ISO-8601 UTC 'Z' string, or None. Accepts 'YYYY-MM-DDTHH:MM:SS'
    (naive → assumed UTC), a space separator, a trailing 'Z'/offset, or epoch seconds."""
    if value in (None, ""):
        return None
    s = str(value).strip()
    try:  # epoch seconds
        return datetime.fromtimestamp(float(s), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        pass
    iso = s.replace(" ", "T", 1).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get(row_lc, *keys):
    for k in keys:
        v = row_lc.get(k)
        if v not in (None, ""):
            return v
    return None


def _coerce_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_row(row_lc):
    """A MarineCadastre CSV row (dict with LOWERCASED keys) -> a contract record, or None if it lacks
    mmsi/lat/lon/timestamp or the coords are out of range."""
    mmsi = _get(row_lc, "mmsi")
    lat = _coerce_float(_get(row_lc, "lat", "latitude"))
    lon = _coerce_float(_get(row_lc, "lon", "longitude"))
    ts = parse_mc_time(_get(row_lc, "basedatetime", "timestamp", "time"))
    if mmsi in (None, "") or lat is None or lon is None or ts is None:
        return None
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        return None
    sog = _coerce_float(_get(row_lc, "sog", "speed"))
    cog = _coerce_float(_get(row_lc, "cog", "course"))
    # AIS not-available sentinels (same as AISStream): SOG 102.3 kn / COG 360 deg mean "no
    # value", not a speed. First real fusion (2026-07-26): a moored vessel's raw 102.3 built a
    # bogus 92 km feasible envelope; null makes the matcher fall back to --max-speed-kn.
    if sog is not None and sog >= 102.2:
        sog = None
    if cog is not None and cog >= 360.0:
        cog = None
    rec = {"mmsi": str(mmsi), "lon": round(lon, 7), "lat": round(lat, 7), "timestamp": ts,
           "sog": None if sog is None else round(sog, 2),
           "cog": None if cog is None else round(cog, 2)}
    name = _get(row_lc, "vesselname", "name", "shipname")
    if name:
        rec["name"] = str(name).strip()
    return rec


def normalize_rows(rows, bbox=None, start=None, end=None):
    """Iterate raw CSV row dicts -> normalized records inside the AOI + time window. Returns
    (records, stats). start/end are ISO strings compared lexically (both are UTC 'Z' → safe)."""
    records, skipped, out_area, out_time = [], 0, 0, 0
    for raw in rows:
        row_lc = {str(k).strip().lower(): v for k, v in raw.items()}
        rec = normalize_row(row_lc)
        if rec is None:
            skipped += 1
            continue
        if bbox is not None and not point_in_bbox(rec["lon"], rec["lat"], bbox):
            out_area += 1
            continue
        if (start and rec["timestamp"] < start) or (end and rec["timestamp"] > end):
            out_time += 1
            continue
        records.append(rec)
    records.sort(key=lambda r: r["timestamp"])
    return records, {"skipped": skipped, "outside_area": out_area, "outside_window": out_time}


def resolve_bbox(args):
    has_bbox = args.bbox is not None
    has_wdpa = args.wdpa_id is not None or args.wdpa_pid is not None
    if has_bbox and has_wdpa:
        raise RuntimeError("Provide either --bbox or --wdpa-id/--wdpa-pid, not both.")
    if has_bbox:
        b = tuple(float(x) for x in args.bbox)
        if not (b[0] < b[2] and b[1] < b[3]):
            raise RuntimeError(f"Bad --bbox (need minLon<maxLon and minLat<maxLat): {b}")
        return b, (args.label or "marinecadastre")
    if has_wdpa:
        return bbox_from_wdpa(args.gpkg, args.wdpa_id, args.wdpa_pid)
    return None, (args.label or "marinecadastre")


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
    minx, miny, maxx, maxy = (float(v) for v in sel.total_bounds)
    return (minx, miny, maxx, maxy), str(sel.iloc[0]["NAME"])


# --------------------------------------------------------------------------- IO

def write_atomic(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def iter_csv_rows(csv_paths):
    """Yield row dicts from one or more CSVs, streaming (national files are large)."""
    for path in csv_paths:
        with Path(path).open(newline="", encoding="utf-8-sig", errors="replace") as fh:
            yield from csv.DictReader(fh)


def download_day(date_str):
    """Do-from-home: download + unzip a MarineCadastre daily archive to a temp CSV. Thin shell."""
    import io
    import tempfile
    import zipfile
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests needed for --date download (pip install -r requirements.txt)") from exc
    y, m, d = (int(x) for x in date_str.split("-"))
    url = MC_URL_TMPL.format(year=y, month=m, day=d)
    print(f"  downloading {url} …")
    resp = requests.get(url, timeout=600)
    resp.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
    if not names:
        raise RuntimeError(f"No CSV inside {url}")
    tmpdir = Path(tempfile.mkdtemp())
    out = tmpdir / names[0].split("/")[-1]
    out.write_bytes(zf.read(names[0]))
    return [out]


def main(argv=None):
    args = parse_args(argv)
    bbox, label = resolve_bbox(args)

    csv_paths = list(args.csv or [])
    if args.date and not csv_paths:
        csv_paths = download_day(args.date)   # do-from-home
    if not csv_paths:
        raise RuntimeError("Provide --csv <MarineCadastre file> (or --date YYYY-MM-DD to download).")
    for p in csv_paths:
        if not Path(p).exists():
            raise RuntimeError(f"CSV not found: {p}")

    print("MARINECADASTRE AIS:")
    print(f"  csv={[str(p) for p in csv_paths]}")
    print(f"  bbox={tuple(round(v,5) for v in bbox) if bbox else None}  "
          f"window={args.start or '-'}..{args.end or '-'}  label={label}")

    records, stats = normalize_rows(iter_csv_rows(csv_paths), bbox, args.start, args.end)
    unique = len({r["mmsi"] for r in records})
    print(f"  kept={len(records)} ({unique} vessels)  skipped={stats['skipped']}  "
          f"outside_area={stats['outside_area']}  outside_window={stats['outside_window']}")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_path = Path(args.out) if args.out else OUTPUT_DIR / f"ais_positions_{ts}_{slugify(label)}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "generated_utc": ts,
        "source": "marinecadastre",
        "bbox_wgs84": [round(v, 7) for v in bbox] if bbox else None,
        "window": {"start": args.start, "end": args.end},
        "position_count": len(records),
        "unique_mmsi": unique,
        "positions": records,   # <- the AIS contract fuse_violations.py --ais reads
    }
    write_atomic(out_path, json.dumps(doc, indent=2))
    print(f"\nWrote {len(records)} positions -> {out_path}")
    print(f"Next: python scripts/fuse_violations.py --run <run> --ais {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
