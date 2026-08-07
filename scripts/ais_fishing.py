import argparse
import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv   # optional; only used to pick up AISF_* knobs from .env
except ImportError:                  # keep the module importable (and offline-testable) without it
    def load_dotenv(*_args, **_kwargs):
        return False

# The objective of this script (LOCAL fishing inference, P3 / design-proposal
# "infer fishing activity via GFW *or local ML*"):
# - Read a normalized AIS positions file (the same contract fuse_violations.py --ais reads)
# - Infer, per MMSI, whether the vessel's MOVEMENT looks like fishing rather than transit
# - Emit the SAME vessel-records contract gfw_vessels.py produces, so fuse_violations.py --vessels
#   consumes it unchanged - only the `source` field differs ("ais-heuristic" vs "gfw")
#
# WHY this exists: today the only fishing signal is GFW (gfw_vessels.py), which needs a GFW_API_TOKEN,
# a network round-trip per MMSI, and only covers vessels GFW tracks. This gives a $0, offline, no-key
# fallback so the fishing term in violation_score works for ANY vessel we have positions for.
#
# HONESTY - this is a COARSE HEURISTIC, not GFW's model and not "local ML". It scores two movement
# features that separate working from transiting:
#   1. SPEED BAND - fishing gear is worked at low but non-zero speed (roughly 1-5 kn for trawlers).
#      Below the band the vessel is moored/anchored/adrift; above it, it is steaming.
#   2. SINUOSITY - a working vessel covers a lot of track inside a small area, so
#      straightness = net_displacement / path_length is LOW. A transiting vessel runs straight
#      (straightness -> 1). This is the standard movement-ecology straightness index.
# Known false positives it CANNOT distinguish: a vessel drifting with engines off, and a slow transit,
# both sit in the speed band. Treat the output as a prior to rank on, never as proof of fishing - which
# is exactly how violation_score already treats GFW's fishing_probability.
#
# House rule: one script per data stream, and this is an ANALYSIS sibling, not an ingester - it reads
# the AIS contract and writes the vessel-records contract. It never imports fuse_violations.py (the
# matcher) or gfw_vessels.py; the small parsing/geo helpers below are deliberately duplicated (see
# docs/AUDIT.md #7 on the standalone-script trade-off).
#
# Everything here is PURE STDLIB and offline-tested (scripts/tests/test_ais_fishing.py) - no network,
# no key, no ML deps.

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "ais_fishing"

EARTH_RADIUS_M = 6_371_008.8   # mean Earth radius (IUGG), metres - same constant as fuse_violations
KN_TO_MS = 1852.0 / 3600.0     # 1 knot = 1852 m/h

load_dotenv(SCRIPT_DIR / ".env")
load_dotenv(REPO_ROOT / ".env")


def env_or(name, default):
    val = os.getenv(name)
    return val if val not in (None, "") else default


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Infer fishing-like behaviour from AIS movement; writes the vessel-records "
                    "contract fuse_violations.py --vessels reads (source: ais-heuristic).")
    p.add_argument("--ais", type=Path, default=env_or("AISF_AIS", None), required=False,
                   help="Normalized AIS positions file (.json or .csv) - the fuse_violations --ais contract.")
    p.add_argument("--out", type=Path, default=env_or("AISF_OUT", None),
                   help="Output path. Default data/raw/ais_fishing/ais_fishing_<UTC>.json.")

    band = p.add_argument_group("speed band (knots) - the 'working the gear' window")
    band.add_argument("--fishing-speed-min", type=float,
                      default=float(env_or("AISF_SPEED_MIN", 1.0)),
                      help="Below this the vessel is moored/anchored/adrift, not fishing. Default 1.0 kn.")
    band.add_argument("--fishing-speed-max", type=float,
                      default=float(env_or("AISF_SPEED_MAX", 5.0)),
                      help="Above this the vessel is steaming, not fishing. Default 5.0 kn.")

    qual = p.add_argument_group("data-sufficiency gates (below these, probability is null, not 0)")
    qual.add_argument("--min-pings", type=int, default=int(env_or("AISF_MIN_PINGS", 3)),
                      help="Vessels with fewer positions get a null probability. Default 3.")
    qual.add_argument("--min-span-minutes", type=float, default=float(env_or("AISF_MIN_SPAN", 10.0)),
                      help="Vessels observed for less than this get a null probability. Default 10 min.")
    qual.add_argument("--max-gap-minutes", type=float, default=float(env_or("AISF_MAX_GAP", 60.0)),
                      help="An interval longer than this is UNOBSERVED time: it is excluded from both "
                           "fishing_hours and the observed total rather than assumed. Default 60 min.")

    p.add_argument("--sinuosity-weight", type=float, default=float(env_or("AISF_SINUOSITY_WEIGHT", 0.5)),
                   help="Blend between the two features in [0,1]: 0 = speed band only, 1 = sinuosity "
                        "only. Default 0.5 (equal weight).")
    p.add_argument("--min-probability", type=float, default=float(env_or("AISF_MIN_PROB", 0.0)),
                   help="Drop vessels scoring below this from the output. Default 0.0 (keep all).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the per-vessel assessment but write no file.")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- AIS input (the contract)

def parse_timestamp(value):
    """Parse an AIS timestamp to an aware UTC datetime. Accepts ISO-8601 (with 'Z' or offset; naive is
    assumed UTC) or epoch seconds. Mirrors fuse_violations.parse_timestamp - same contract."""
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
    """One normalized AIS record from a raw dict, or None if it lacks mmsi/lat/lon/timestamp."""
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
    name = raw.get("name", raw.get("VesselName", raw.get("shipname")))
    return {"mmsi": str(mmsi), "lon": lon, "lat": lat, "when": when,
            "sog": sog, "cog": cog, "name": name or None}


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


def group_by_mmsi(pings):
    """{mmsi: [ping, ...]} preserving the time order of the input."""
    tracks = {}
    for ping in pings:
        tracks.setdefault(ping["mmsi"], []).append(ping)
    return tracks


# --------------------------------------------------------------------------- movement features

def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres between two WGS84 points."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def interval_speed_kn(a, b):
    """Speed over the a->b interval in knots.

    Prefers the vessel's own reported SOG (mean of the two endpoints) and falls back to the speed
    derived from the positions. The ingesters already map the AIS 'not available' sentinel (102.3 kn)
    to null, so a None here genuinely means unreported."""
    sogs = [s for s in (a.get("sog"), b.get("sog")) if s is not None]
    if sogs:
        return sum(sogs) / len(sogs)
    dt_s = (b["when"] - a["when"]).total_seconds()
    if dt_s <= 0:
        return None
    metres = haversine_m(a["lon"], a["lat"], b["lon"], b["lat"])
    return (metres / dt_s) / KN_TO_MS


def track_metrics(track, speed_min, speed_max, max_gap_s):
    """Per-vessel movement summary over consecutive ping intervals.

    Intervals longer than ``max_gap_s`` are UNOBSERVED (an AIS gap, or two unrelated visits) and are
    dropped from both the numerator and the denominator - counting them as fishing would invent hours
    the data never showed, and counting them as not-fishing would dilute a real signal."""
    observed_s = fishing_s = path_m = 0.0
    used = skipped_gaps = 0
    for a, b in zip(track, track[1:]):
        dt_s = (b["when"] - a["when"]).total_seconds()
        if dt_s <= 0:
            continue
        if dt_s > max_gap_s:
            skipped_gaps += 1
            continue
        speed = interval_speed_kn(a, b)
        if speed is None:
            continue
        observed_s += dt_s
        path_m += haversine_m(a["lon"], a["lat"], b["lon"], b["lat"])
        if speed_min <= speed <= speed_max:
            fishing_s += dt_s
        used += 1

    net_m = haversine_m(track[0]["lon"], track[0]["lat"],
                        track[-1]["lon"], track[-1]["lat"]) if len(track) >= 2 else 0.0
    # straightness = net / path. 1.0 = dead straight (transit), ->0 = working a small area.
    # With no measurable path there is no evidence either way, so it stays None (not 0, which would
    # read as "maximally sinuous" and inflate the score).
    straightness = min(1.0, net_m / path_m) if path_m > 0 else None
    span_s = (track[-1]["when"] - track[0]["when"]).total_seconds() if len(track) >= 2 else 0.0
    return {"observed_seconds": observed_s, "fishing_seconds": fishing_s,
            "path_m": path_m, "net_m": net_m, "straightness": straightness,
            "span_seconds": span_s, "intervals_used": used, "gaps_skipped": skipped_gaps}


def fishing_probability(metrics, sinuosity_weight):
    """Blend the two movement features into a [0,1] prior, or None when the evidence is too thin.

    p = (1-w)*time_in_speed_band + w*(1-straightness). Both terms are fractions, so p is one too.
    Returns None (unknown) rather than 0.0 when there are no usable intervals - "we could not tell"
    and "it was not fishing" are different claims, and the score treats a null as no-information."""
    if metrics["observed_seconds"] <= 0:
        return None
    band = metrics["fishing_seconds"] / metrics["observed_seconds"]
    straightness = metrics["straightness"]
    if straightness is None:          # no measurable path: fall back to the speed band alone
        return round(max(0.0, min(1.0, band)), 4)
    w = max(0.0, min(1.0, sinuosity_weight))
    p = (1.0 - w) * band + w * (1.0 - straightness)
    return round(max(0.0, min(1.0, p)), 4)


def assess_track(mmsi, track, args):
    """One vessel-record for the vessel-records contract, plus the heuristic's own diagnostics.

    Identity fields (shipname aside) are null by construction: AIS positions carry no flag/gear/IMO.
    That is exactly how fuse_violations already treats a partial record - it fills what it has."""
    max_gap_s = args.max_gap_minutes * 60.0
    metrics = track_metrics(track, args.fishing_speed_min, args.fishing_speed_max, max_gap_s)

    enough = (len(track) >= args.min_pings
              and metrics["span_seconds"] >= args.min_span_minutes * 60.0
              and metrics["intervals_used"] > 0)
    prob = fishing_probability(metrics, args.sinuosity_weight) if enough else None
    hours = round(metrics["fishing_seconds"] / 3600.0, 4) if enough else None

    name = next((p["name"] for p in track if p.get("name")), None)
    return {
        # --- the vessel-records contract fuse_violations.py --vessels reads ---
        "mmsi": str(mmsi),
        "vessel_id": None,
        "shipname": name,
        "flag": None,
        "geartype": None,
        "imo": None,
        "callsign": None,
        "fishing_hours": hours,
        "fishing_probability": prob,
        # --- heuristic diagnostics (extra keys; the reader ignores what it does not know) ---
        "ping_count": len(track),
        "observed_hours": round(metrics["observed_seconds"] / 3600.0, 4),
        "span_hours": round(metrics["span_seconds"] / 3600.0, 4),
        "straightness": None if metrics["straightness"] is None else round(metrics["straightness"], 4),
        "path_km": round(metrics["path_m"] / 1000.0, 4),
        "net_km": round(metrics["net_m"] / 1000.0, 4),
        "intervals_used": metrics["intervals_used"],
        "gaps_skipped": metrics["gaps_skipped"],
        "insufficient_data": not enough,
        # Never "gfw": fuse_violations tags evidence from this field, so a locally-derived signal must
        # not claim GFW provenance.
        "source": "ais-heuristic",
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


def assess_all(pings, args):
    """Every vessel in the file, ranked most-fishing-like first (nulls last)."""
    records = [assess_track(mmsi, track, args) for mmsi, track in group_by_mmsi(pings).items()]
    records = [r for r in records
               if r["fishing_probability"] is None or r["fishing_probability"] >= args.min_probability]
    records.sort(key=lambda r: (r["fishing_probability"] is None, -(r["fishing_probability"] or 0.0)))
    return records


def main(argv=None):
    args = parse_args(argv)
    if not args.ais:
        raise RuntimeError("Provide --ais <normalized AIS positions file> "
                           "(from aisstream_fetch.py or marinecadastre_fetch.py).")
    if args.fishing_speed_min >= args.fishing_speed_max:
        raise RuntimeError(f"--fishing-speed-min ({args.fishing_speed_min}) must be below "
                           f"--fishing-speed-max ({args.fishing_speed_max}).")

    pings, skipped = load_ais(args.ais)
    if skipped:
        print(f"  AIS: skipped {skipped} unusable rows (missing mmsi/lat/lon/timestamp)")
    if not pings:
        raise RuntimeError(f"No usable AIS positions in {args.ais}.")

    records = assess_all(pings, args)
    scored = [r for r in records if r["fishing_probability"] is not None]

    print("AIS FISHING (local heuristic - a prior, not proof):")
    print(f"  ais={args.ais}  pings={len(pings)}  vessels={len(records)}")
    print(f"  speed band={args.fishing_speed_min}-{args.fishing_speed_max} kn  "
          f"sinuosity weight={args.sinuosity_weight}  max gap={args.max_gap_minutes} min")
    print(f"  scored={len(scored)}  insufficient data={len(records) - len(scored)}")
    for rec in records[:10]:
        prob = "n/a " if rec["fishing_probability"] is None else f"{rec['fishing_probability']:.2f}"
        straight = "n/a" if rec["straightness"] is None else f"{rec['straightness']:.2f}"
        print(f"    {rec['mmsi']:>10s}  p={prob}  fishing_h={rec['fishing_hours']}  "
              f"straightness={straight}  pings={rec['ping_count']}")
    if len(records) > 10:
        print(f"    ... and {len(records) - 10} more")

    if args.dry_run:
        print("\nDRY RUN - nothing written.")
        return 0

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_path = Path(args.out) if args.out else OUTPUT_DIR / f"ais_fishing_{ts}.json"
    doc = {
        "generated_utc": ts,
        "source": "ais-heuristic",
        "ais_file": str(args.ais),
        "method": "speed-band + sinuosity over AIS movement (coarse prior, not GFW's model)",
        "params": {
            "fishing_speed_min_kn": args.fishing_speed_min,
            "fishing_speed_max_kn": args.fishing_speed_max,
            "sinuosity_weight": args.sinuosity_weight,
            "max_gap_minutes": args.max_gap_minutes,
            "min_pings": args.min_pings,
            "min_span_minutes": args.min_span_minutes,
        },
        "ping_count": len(pings),
        "vessel_count": len(records),
        "scored_count": len(scored),
        "vessels": records,   # <- the vessel-records contract fuse_violations.py --vessels reads
    }
    write_atomic(out_path, json.dumps(doc, indent=2))
    print(f"\nWrote {len(records)} vessel records -> {out_path}")
    print(f"Next: python scripts/fuse_violations.py --run <run> --ais {args.ais} --vessels {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
