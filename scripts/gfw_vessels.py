import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv   # pinned in requirements.txt for real runs
except ImportError:                  # keep the module importable (and offline-testable) without it
    def load_dotenv(*_args, **_kwargs):
        return False

# The objective of this script (GFW per-vessel ingestion, ROADMAP Phase 3 step 3):
# - Take a set of MMSIs (explicit --mmsi, or pulled from a fuse_violations violation_events.json
#   via --from-events, or an AIS positions file via --from-ais)
# - Resolve each to GFW vessel IDENTITY (Vessels search: vesselId, shipname, flag, geartype, imo)
# - Summarize its recent FISHING activity (Events API: fishing_hours + a coarse fishing_probability)
# - OPTIONALLY (--insights) pull GFW Vessel Insights (IUU-list flag, AIS-off/gap events, apparent
#   fishing in no-take MPAs)
# - NORMALIZE all of that into the project's vessel-records contract and write it to
#   data/raw/gfw_vessels/ - the file scripts/fuse_violations.py --vessels reads to ENRICH the
#   matched (AIS-identified) events (identity + fishing_hours/fishing_probability) and lift their
#   violation_score when the identified vessel was actually fishing in the MPA.
#
# House rule: one script per data stream. This is the GFW per-VESSEL identity/activity ingester -
# its OWN stream, distinct from gfw_fetch.py (which pulls the coarse GRIDDED 4wings fishing-effort
# DENSITY, not per-vessel positions). It never imports fuse_violations.py; it only produces the
# normalized contract. See docs/ROADMAP.md Phase 3.
#
# The risky GFW response handling is factored into PURE functions (normalize_identity,
# summarize_fishing, normalize_insights, the MMSI collectors) and unit-tested offline
# (scripts/tests/test_gfw_vessels.py) with synthetic GFW-shaped payloads - no token, no network.
# The HTTP layer is a thin shell mirroring gfw_fetch.py's proven Bearer-token/.env auth.
#
# IMPORTANT (honesty): the exact GFW v3 request paths/params and response shapes below follow
# GFW's documented v3 API but were NOT verified against the live service here (no token in this
# environment) - same situation gfw_fetch.py was written in. The PARSERS are deliberately tolerant
# (field aliases, graceful None) so a small shape drift degrades to missing fields rather than a
# crash; verify/adjust the endpoints on the first real run with a GFW_API_TOKEN. Use --dry-run to
# inspect the resolved MMSIs + planned requests with neither a token nor the network.
#
# Terminology (GFW): ssvid = source-specific vessel id; for AIS, ssvid == MMSI. vesselId is GFW's
# internal identity id (MMSI+IMO combination). Identity lives in selfReportedInfo / registryInfo.

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "gfw_vessels"
GFW_BASE = "https://gateway.api.globalfishingwatch.org"

load_dotenv(SCRIPT_DIR / ".env")
load_dotenv(REPO_ROOT / ".env")


def env_or(name, default):
    val = os.getenv(name)
    return val if val not in (None, "") else default


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Resolve MMSIs to GFW identity + fishing activity (+ optional insights) and write "
                    "the normalized vessel-records file fuse_violations.py --vessels enriches with "
                    "(ROADMAP Phase 3 step 3).")
    src = p.add_argument_group("MMSI source (provide at least one)")
    src.add_argument("--mmsi", default=env_or("GFWV_MMSI", None),
                     help="Comma-separated MMSIs to look up.")
    src.add_argument("--from-events", type=Path, default=env_or("GFWV_FROM_EVENTS", None),
                     help="A fuse_violations violation_events.json; pulls the MATCHED events' MMSIs "
                          "(exactly the vessels an enrichment pass would use).")
    src.add_argument("--from-ais", type=Path, default=env_or("GFWV_FROM_AIS", None),
                     help="A normalized AIS positions file; pulls every MMSI seen.")
    p.add_argument("--start", default=env_or("GFWV_START", None),
                   help="YYYY-MM-DD activity-window start. Default: 7 days before --end. Set this to "
                        "bracket the scene date so fishing_probability reflects the relevant window.")
    p.add_argument("--end", default=env_or("GFWV_END", None),
                   help="YYYY-MM-DD activity-window end. Default: today UTC.")
    p.add_argument("--insights", action="store_true", default=bool(env_or("GFWV_INSIGHTS", "")),
                   help="Also call the GFW Vessel Insights API (IUU-list flag, AIS-off/gap events, "
                        "apparent fishing in no-take MPAs). Optional; off by default.")
    p.add_argument("--dataset-identity", default=env_or("GFWV_DATASET_IDENTITY",
                                                        "public-global-vessel-identity:latest"))
    p.add_argument("--dataset-events", default=env_or("GFWV_DATASET_EVENTS",
                                                     "public-global-fishing-events:latest"))
    p.add_argument("--limit", type=int, default=int(env_or("GFWV_LIMIT", 5)),
                   help="Max identity search results to consider per MMSI. Default 5.")
    p.add_argument("--out", type=Path, default=env_or("GFWV_OUT", None),
                   help="Output path. Default data/raw/gfw_vessels/gfw_vessels_<UTC>.json.")
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve MMSIs + print the planned requests, but do NOT call GFW or write. "
                        "Needs neither a token nor the network.")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- pure: MMSI collection

def mmsis_from_events(doc):
    """MATCHED MMSIs from a fuse_violations violation_events.json (dark events have mmsi=null)."""
    events = doc.get("violation_events", []) if isinstance(doc, dict) else (doc or [])
    out = []
    for e in events:
        mmsi = e.get("mmsi")
        if mmsi not in (None, ""):
            out.append(str(mmsi))
    return out


def mmsis_from_ais(doc):
    """Every MMSI in a normalized AIS positions file (list or {positions:[...]})."""
    rows = doc.get("positions", []) if isinstance(doc, dict) else (doc or [])
    out = []
    for r in rows:
        mmsi = r.get("mmsi", r.get("MMSI"))
        if mmsi not in (None, ""):
            out.append(str(mmsi))
    return out


def collect_mmsis(args):
    """Union of MMSIs from --mmsi / --from-events / --from-ais, de-duplicated, order preserved."""
    seen, ordered = set(), []
    def add(values):
        for v in values:
            s = str(v).strip()
            if s and s not in seen:
                seen.add(s)
                ordered.append(s)
    if args.mmsi:
        add([m for m in str(args.mmsi).split(",") if m.strip()])
    if args.from_events:
        add(mmsis_from_events(json.loads(Path(args.from_events).read_text())))
    if args.from_ais:
        add(mmsis_from_ais(json.loads(Path(args.from_ais).read_text())))
    return ordered


# --------------------------------------------------------------------------- pure: normalization

def _first(*values):
    for v in values:
        if v not in (None, "", []):
            return v
    return None


def _geartype(entry):
    """geartype can be a bare string, a list of strings, or a list of {name,...} across GFW's
    selfReported/combinedSources shapes. Return a single readable string or None."""
    g = _first(entry.get("geartype"), entry.get("geartypes"), entry.get("vesselType"))
    if isinstance(g, list) and g:
        g = g[0]
    if isinstance(g, dict):
        g = g.get("name") or g.get("value")
    return str(g) if g not in (None, "") else None


def normalize_identity(search_json, mmsi):
    """GFW Vessels-search response -> a compact identity dict for one MMSI. Tolerant of the
    selfReportedInfo/registryInfo entry shapes; picks the entry whose ssvid matches the MMSI (else
    the first). Returns {mmsi, vessel_id, shipname, flag, geartype, imo, callsign} (fields may be None)."""
    entries = search_json.get("entries", search_json) if isinstance(search_json, dict) else search_json
    if not isinstance(entries, list):
        entries = []
    # flatten candidate identity blocks (self-reported first, then registry)
    blocks = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        sri = e.get("selfReportedInfo") or []
        reg = e.get("registryInfo") or []
        blocks.extend(b for b in sri if isinstance(b, dict))
        blocks.extend(b for b in reg if isinstance(b, dict))
        if not sri and not reg:
            blocks.append(e)   # some shapes put identity fields directly on the entry
    if not blocks:
        return {"mmsi": str(mmsi), "vessel_id": None, "shipname": None, "flag": None,
                "geartype": None, "imo": None, "callsign": None}
    match = next((b for b in blocks if str(b.get("ssvid", b.get("mmsi", ""))) == str(mmsi)), blocks[0])
    return {
        "mmsi": str(mmsi),
        "vessel_id": _first(match.get("vesselId"), match.get("id")),
        "shipname": _first(match.get("shipname"), match.get("name")),
        "flag": match.get("flag"),
        "geartype": _geartype(match),
        "imo": _first(match.get("imo"), match.get("imoNumber")),
        "callsign": match.get("callsign"),
    }


def _parse_iso(value):
    if value in (None, ""):
        return None
    s = str(value).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _event_time(value):
    """A GFW event start/end may be an ISO string or a nested {timestamp/time/date} dict."""
    if isinstance(value, dict):
        value = _first(value.get("timestamp"), value.get("time"), value.get("date"))
    return _parse_iso(value)


def event_duration_hours(entry):
    """Hours for one fishing event, from an explicit durationHours field, else start/end timestamps."""
    dur = entry.get("durationHours", entry.get("duration_hours"))
    if dur is not None:
        try:
            return max(0.0, float(dur))
        except (TypeError, ValueError):
            return 0.0
    start = _event_time(entry.get("start"))
    end = _event_time(entry.get("end"))
    if start and end and end >= start:
        return (end - start).total_seconds() / 3600.0
    return 0.0


def summarize_fishing(events_json):
    """GFW Events response -> {fishing_hours, fishing_event_count, fishing_probability}. Counts only
    fishing-type events. fishing_probability is a COARSE, transparent prior: 1.0 if the vessel had
    any fishing event in the queried window, else 0.0 (None only if there were no entries at all to
    judge from is impossible here - an empty list is a real 'no fishing' answer, so 0.0)."""
    entries = events_json.get("entries", events_json) if isinstance(events_json, dict) else events_json
    if not isinstance(entries, list):
        return {"fishing_hours": None, "fishing_event_count": None, "fishing_probability": None}
    fishing = [e for e in entries if isinstance(e, dict)
               and str(e.get("type", "fishing")).lower() == "fishing"]
    hours = round(sum(event_duration_hours(e) for e in fishing), 3)
    return {"fishing_hours": hours, "fishing_event_count": len(fishing),
            "fishing_probability": 1.0 if fishing else 0.0}


def _any_truthy(o):
    """Is there any truthy signal anywhere in this scalar/dict/list subtree? (bool True, a
    'true'/'yes'/'1' string, or a positive number - e.g. a 'listed N times' count.)"""
    if isinstance(o, bool):
        return o
    if isinstance(o, (int, float)):
        return o > 0
    if isinstance(o, str):
        return o.strip().lower() in ("true", "yes", "1")
    if isinstance(o, dict):
        return any(_any_truthy(v) for v in o.values())
    if isinstance(o, list):
        return any(_any_truthy(v) for v in o)
    return False


def _deep_find_bool(obj, key_substrings):
    """Search a nested dict/list for an indicator whose key contains any of key_substrings and whose
    (possibly nested) value is truthy. Returns True/None (None = not found; heuristic, verify live)."""
    found = None
    def walk(o):
        nonlocal found
        if found:
            return
        if isinstance(o, dict):
            for k, v in o.items():
                if any(s in str(k).lower() for s in key_substrings) and _any_truthy(v):
                    found = True
                    return
                walk(v)
        elif isinstance(o, list):
            for item in o:
                walk(item)
    walk(obj)
    return found


def normalize_insights(insights_json, mmsi=None, vessel_id=None):
    """GFW Insights response -> {iuu_listed, ais_off_events, fishing_in_no_take_mpa}. Very tolerant:
    the Insights v3 shape is nested and evolving, so this searches defensively and returns None for
    any indicator it can't confidently read rather than guessing wrong."""
    doc = insights_json
    if isinstance(insights_json, dict):
        vessels = insights_json.get("vessels")
        if isinstance(vessels, list) and vessels:
            doc = next((v for v in vessels
                        if str(v.get("vesselId", "")) == str(vessel_id)
                        or str(v.get("ssvid", "")) == str(mmsi)), vessels[0])
    iuu = _deep_find_bool(doc, ["iuu"])
    ais_off = _find_count(doc, ["aisoff", "ais_off", "gap"])
    fishing_mpa = _find_count(doc, ["notakempa", "no_take", "notake"])
    return {"iuu_listed": iuu,
            "ais_off_events": ais_off,
            "fishing_in_no_take_mpa": fishing_mpa}


def _find_count(obj, key_substrings):
    """Search a nested structure for an integer count under a key containing any substring; also
    counts the length of a matching list. Returns an int or None."""
    result = None
    def walk(o):
        nonlocal result
        if isinstance(o, dict):
            for k, v in o.items():
                kl = str(k).lower()
                if any(s in kl for s in key_substrings):
                    if isinstance(v, bool):
                        result = (result or 0) + (1 if v else 0)
                    elif isinstance(v, (int, float)):
                        result = (result or 0) + int(v)
                    elif isinstance(v, list):
                        result = (result or 0) + len(v)
                walk(v)
        elif isinstance(o, list):
            for item in o:
                walk(item)
    walk(obj)
    return result


def build_vessel_record(identity, fishing, insights):
    """Merge the pure sub-results into one normalized vessel record (the contract fuse reads)."""
    rec = dict(identity)
    rec.update({
        "fishing_hours": fishing.get("fishing_hours"),
        "fishing_event_count": fishing.get("fishing_event_count"),
        "fishing_probability": fishing.get("fishing_probability"),
    })
    if insights is not None:
        rec.update({
            "iuu_listed": insights.get("iuu_listed"),
            "ais_off_events": insights.get("ais_off_events"),
            "fishing_in_no_take_mpa": insights.get("fishing_in_no_take_mpa"),
        })
    rec["source"] = "gfw"
    return rec


# --------------------------------------------------------------------------- IO / HTTP shell

def write_atomic(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def resolve_window(start, end):
    end_d = end or datetime.now(timezone.utc).date().isoformat()
    if start:
        return start, end_d
    end_date = datetime.fromisoformat(end_d).date() if end else datetime.now(timezone.utc).date()
    return (end_date - timedelta(days=7)).isoformat(), end_d


def _requests():
    try:
        import requests   # pinned in requirements.txt
        return requests
    except ImportError as exc:
        raise RuntimeError("requests is not installed (pip install -r requirements.txt)") from exc


def _headers(token):
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def search_vessel(token, mmsi, dataset, limit):
    """GET /v3/vessels/search?query=<mmsi> - identity resolution. Thin shell; parsing is
    normalize_identity(). Verify endpoint/params against GFW v3 docs on first real run."""
    requests = _requests()
    params = {"query": str(mmsi), "datasets[0]": dataset, "limit": limit}
    r = requests.get(f"{GFW_BASE}/v3/vessels/search", headers=_headers(token), params=params, timeout=120)
    r.raise_for_status()
    return r.json()


def fishing_events(token, vessel_id, dataset, start, end):
    """GET /v3/events (fishing) for one vesselId in [start,end]. Parsing is summarize_fishing()."""
    requests = _requests()
    params = {"datasets[0]": dataset, "vessels[0]": vessel_id,
              "start-date": start, "end-date": end, "limit": 500}
    r = requests.get(f"{GFW_BASE}/v3/events", headers=_headers(token), params=params, timeout=120)
    r.raise_for_status()
    return r.json()


def vessel_insights(token, vessel_id, dataset_identity, start, end):
    """POST /v3/insights/vessels for one vesselId. Parsing is normalize_insights()."""
    requests = _requests()
    body = {"includes": ["FISHING", "GAP", "VESSEL-IDENTITY-IUU"],
            "startDate": start, "endDate": end,
            "vessels": [{"vesselId": vessel_id, "datasetId": dataset_identity}]}
    r = requests.post(f"{GFW_BASE}/v3/insights/vessels", headers=_headers(token), json=body, timeout=120)
    r.raise_for_status()
    return r.json()


def resolve_one(token, mmsi, args, start, end):
    """Full per-MMSI resolution: identity -> fishing -> (optional) insights -> merged record.
    Any single sub-call failing is caught and logged so one bad MMSI doesn't abort the batch."""
    identity = normalize_identity(search_vessel(token, mmsi, args.dataset_identity, args.limit), mmsi)
    vessel_id = identity.get("vessel_id")
    fishing = {"fishing_hours": None, "fishing_event_count": None, "fishing_probability": None}
    insights = None
    if vessel_id:
        try:
            fishing = summarize_fishing(fishing_events(token, vessel_id, args.dataset_events, start, end))
        except Exception as exc:   # noqa: BLE001 - one vessel's failure must not kill the run
            print(f"  MMSI {mmsi}: fishing lookup failed: {exc}")
        if args.insights:
            try:
                insights = normalize_insights(
                    vessel_insights(token, vessel_id, args.dataset_identity, start, end), mmsi, vessel_id)
            except Exception as exc:   # noqa: BLE001
                print(f"  MMSI {mmsi}: insights lookup failed: {exc}")
    return build_vessel_record(identity, fishing, insights)


def main(argv=None):
    args = parse_args(argv)
    mmsis = collect_mmsis(args)
    if not mmsis:
        raise RuntimeError("No MMSIs. Provide --mmsi 1,2,3 and/or --from-events <violation_events.json> "
                           "and/or --from-ais <positions.json>.")
    start, end = resolve_window(args.start, args.end)

    print("GFW VESSELS:")
    print(f"  mmsis={len(mmsis)}  window={start}..{end}  insights={args.insights}")
    print(f"  identity_dataset={args.dataset_identity}  events_dataset={args.dataset_events}")

    if args.dry_run:
        print("  DRY RUN — planned per-MMSI requests (not sent):")
        print(f"    GET  {GFW_BASE}/v3/vessels/search?query=<mmsi>&datasets[0]={args.dataset_identity}")
        print(f"    GET  {GFW_BASE}/v3/events?datasets[0]={args.dataset_events}&vessels[0]=<vesselId>"
              f"&start-date={start}&end-date={end}")
        if args.insights:
            print(f"    POST {GFW_BASE}/v3/insights/vessels  (vesselId, {start}..{end})")
        print(f"  mmsis: {', '.join(mmsis)}")
        return 0

    token = os.getenv("GFW_API_TOKEN")
    if not token:
        raise RuntimeError("Missing GFW_API_TOKEN (put it in a .env at the repo root or scripts/). "
                           "Use --dry-run to inspect the plan without a token.")
    token = token.strip()

    records = []
    for mmsi in mmsis:
        try:
            rec = resolve_one(token, mmsi, args, start, end)
        except Exception as exc:   # noqa: BLE001 - identity failure for one MMSI: log + emit a stub
            print(f"  MMSI {mmsi}: identity lookup failed: {exc}")
            rec = build_vessel_record({"mmsi": str(mmsi), "vessel_id": None, "shipname": None,
                                       "flag": None, "geartype": None, "imo": None, "callsign": None},
                                      {"fishing_hours": None, "fishing_event_count": None,
                                       "fishing_probability": None}, None)
        records.append(rec)
        print(f"  {mmsi}: name={rec.get('shipname')} flag={rec.get('flag')} "
              f"gear={rec.get('geartype')} fishing_h={rec.get('fishing_hours')}")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_path = Path(args.out) if args.out else OUTPUT_DIR / f"gfw_vessels_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "generated_utc": ts,
        "source": "gfw",
        "window": {"start": start, "end": end},
        "insights": args.insights,
        "vessel_count": len(records),
        "vessels": records,   # <- the vessel-records contract fuse_violations.py --vessels reads
    }
    write_atomic(out_path, json.dumps(doc, indent=2))
    print(f"\nWrote {len(records)} vessel records -> {out_path}")
    print(f"Next: python scripts/fuse_violations.py --run <run> --ais <ais.json> --vessels {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
