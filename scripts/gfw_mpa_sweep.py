"""Sweep every marine IUCN Ia/Ib MPA against GFW fishing effort — the nightly watch.

This is the third GFW data stream in the repo, and it is deliberately its own script
(house rule: one script per data stream):

  * ``gfw_fetch.py``   — coarse GRIDDED effort density over ONE region (an EEZ).
  * ``gfw_vessels.py`` — per-vessel identity/activity for a KNOWN set of MMSIs.
  * ``gfw_mpa_sweep.py`` (this) — asks, for EVERY protected area we care about, "was any
    fishing vessel inside it during this window?" and answers with named vessels.

Why this shape, measured 2026-08-09 against the live v3 API:

1. **GFW resolves our MPAs directly.** The context layer ``public-mpa-all`` is "MPAs
   (WDPA)" sourced from Protected Planet, and its ``idProperty`` is ``SITE_PID`` — the
   same ``WDPA_PID`` already in ``data/processed/wdpa_marine_ia_ib.gpkg``. So a region is
   ``{"dataset": "public-mpa-all", "id": <WDPA_PID>}`` with no mapping table.
   (``public-mpa-no-take`` would be a better filter but answers 403 on our token.)

2. **``date-range`` is HALF-OPEN [start, end).** A vessel entered Nordaust-Svalbard at
   2026-08-03T05:00Z; ``2026-08-03,2026-08-03`` returned 0 rows and
   ``2026-08-03,2026-08-04`` returned it. A zero-width window is silently empty, never an
   error — the bug that made every nightly ``gfw_fetch.py`` cron run fetch nothing.

3. **Do not filter by gear.** ``geartype`` filter values are LOWERCASE while the response
   reports them uppercase, and the inherited ``trawlers`` filter dropped a real
   ``OTHER_PURSE_SEINES`` intrusion. In a no-take reserve *any* fishing gear is the
   signal, so the default sends no filter at all. ``--gear`` is kept for ad-hoc use.

4. **With ``group-by=VESSEL_ID`` the response is per-vessel, not a grid** — shipName,
   mmsi, flag, geartype, hours, entry/exit timestamps. That is an intrusion record.

Rows arrive per vessel *per spatial cell*, so the same vessel appears several times in one
MPA; ``aggregate_events`` sums the hours and takes the widest entry/exit span.

Output: ``data/raw/gfw_mpa/gfw_mpa_sweep_<UTC>.json`` plus an optional markdown digest
for ``$GITHUB_STEP_SUMMARY``. Reads the committed geometry-free ``mpa_index.csv`` so it
needs neither geopandas nor the gitignored WDPA shapefiles.

``--dry-run`` prints the plan with neither a token nor the network.
"""

import argparse
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is optional at import time
    def load_dotenv(*_args, **_kwargs):
        return False

import _http  # stdlib-only at import time; lazy-imports requests inside build_session

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_INDEX = REPO_ROOT / "data" / "processed" / "mpa_index.csv"
OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "gfw_mpa"

GFW_REPORT_URL = "https://gateway.api.globalfishingwatch.org/v3/4wings/report"
MPA_REGION_DATASET = "public-mpa-all"
SOURCE = "gfw-4wings-mpa"

# Above this share of failed regions the sweep is not trustworthy, so it exits non-zero
# and the workflow's failure notification fires. A handful of individual 5xx/timeouts on a
# 1799-region sweep is normal and must NOT page anyone.
DEFAULT_MAX_ERROR_RATE = 0.25

# ⚠️ ONE REQUEST AT A TIME. Measured 2026-08-09: strictly sequential calls returned 15/15
# HTTP 200, while ANY concurrency produced 429 "Too Many Requests" — 6 workers failed 4/5
# regions, and even 2 workers paced to 0.29 req/s failed 9/12. Pacing does not help; the
# limit is on in-flight requests, not on rate. The quota is not the issue (the same
# responses carry x-ratelimit-daily-limit-requests: 50000 against ~170 used). Raising this
# does not make the sweep faster, it makes it fail.
DEFAULT_WORKERS = 1

# Cold requests average ~12-14s regardless of window width (1-day, 7-day and 30-day
# windows all measured the same), so a region costs ~13s and the schedule is:
#   all 1799 MPAs -> ~390 min, past GitHub Actions' 6h job ceiling
#   the 748 MPAs >= 1 km2 -> ~162 min
# Hence the two-tier design in select_mpas().
DEFAULT_MIN_AREA_KM2 = 1.0
DEFAULT_MAX_MINUTES = 300.0


def load_env():
    """Load .env from scripts/ then the repo root (repo root wins on conflict)."""
    load_dotenv(SCRIPT_DIR / ".env")
    load_dotenv(REPO_ROOT / ".env")


def env_or(name, default):
    val = os.getenv(name)
    return val if val not in (None, "") else default


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Sweep marine Ia/Ib MPAs for GFW fishing effort; write intrusion events.")
    p.add_argument("--index", type=Path, default=Path(env_or("GFW_MPA_INDEX", DEFAULT_INDEX)),
                   help=f"MPA index CSV from prep_polygons.py (default {DEFAULT_INDEX}).")
    p.add_argument("--start", default=env_or("GFW_MPA_START", None),
                   help="YYYY-MM-DD, INCLUSIVE. Overrides --lag-days/--window-days.")
    p.add_argument("--end", default=env_or("GFW_MPA_END", None),
                   help="YYYY-MM-DD, EXCLUSIVE (GFW's date-range is half-open).")
    p.add_argument("--lag-days", type=int, default=int(env_or("GFW_MPA_LAG_DAYS", 2)),
                   help="Days to stay behind today, absorbing GFW's publication lag "
                        "(default 2). The window ENDS at today - lag.")
    p.add_argument("--window-days", type=int, default=int(env_or("GFW_MPA_WINDOW_DAYS", 7)),
                   help="Width of the window in days (default 7). Must be >= 1; a "
                        "zero-width range returns nothing from GFW.")
    p.add_argument("--gear", default=env_or("GFW_MPA_GEAR", None),
                   help="Comma-separated LOWERCASE geartypes. Default: no filter at all, "
                        "which is what you want (any gear in a no-take MPA is the signal).")
    p.add_argument("--limit", type=int, default=None,
                   help="Query only the first N MPAs (smoke-testing).")
    p.add_argument("--min-area", type=float,
                   default=float(env_or("GFW_MPA_MIN_AREA", DEFAULT_MIN_AREA_KM2)),
                   help=f"MPAs with GIS_M_AREA >= this (km2) are swept EVERY run "
                        f"(default {DEFAULT_MIN_AREA_KM2}). Smaller ones form the rotating "
                        f"tail — the median Ia/Ib MPA is 0.42 km2, below what GFW's "
                        f"AIS-derived effort can resolve. 0 sweeps everything every run.")
    p.add_argument("--tail-slice", type=int, default=int(env_or("GFW_MPA_TAIL_SLICE", 250)),
                   help="How many below-threshold MPAs to include per run, rotating by "
                        "date so the whole tail is covered over successive runs "
                        "(default 250). 0 disables the tail.")
    p.add_argument("--max-minutes", type=float,
                   default=float(env_or("GFW_MPA_MAX_MINUTES", DEFAULT_MAX_MINUTES)),
                   help=f"Stop sweeping after this many minutes and report what was "
                        f"skipped (default {DEFAULT_MAX_MINUTES}). Guards the GitHub "
                        f"Actions 6h job ceiling. 0 disables the budget.")
    p.add_argument("--workers", type=int, default=int(env_or("GFW_MPA_WORKERS",
                                                             DEFAULT_WORKERS)),
                   help="In-flight requests. KEEP AT 1: the GFW token rejects any "
                        "concurrency with 429 (measured), so raising this fails the sweep "
                        "rather than speeding it up.")
    p.add_argument("--min-hours", type=float, default=float(env_or("GFW_MPA_MIN_HOURS", 0.0)),
                   help="Drop aggregated events below this many fishing hours (default 0).")
    p.add_argument("--dataset", default=env_or("GFW_MPA_DATASET",
                                               "public-global-fishing-effort:latest"))
    p.add_argument("--spatial-resolution", default=env_or("GFW_MPA_SPATIAL_RESOLUTION", "LOW"))
    p.add_argument("--temporal-resolution", default=env_or("GFW_MPA_TEMPORAL_RESOLUTION",
                                                           "ENTIRE"))
    p.add_argument("--out", type=Path, default=None,
                   help="Output JSON (default data/raw/gfw_mpa/gfw_mpa_sweep_<UTC>.json).")
    p.add_argument("--summary-md", type=Path, default=None,
                   help="Also write a markdown digest here (e.g. $GITHUB_STEP_SUMMARY).")
    p.add_argument("--max-error-rate", type=float,
                   default=float(env_or("GFW_MPA_MAX_ERROR_RATE", DEFAULT_MAX_ERROR_RATE)),
                   help=f"Exit non-zero if more than this share of regions fail "
                        f"(default {DEFAULT_MAX_ERROR_RATE}).")
    p.add_argument("--timeout", type=float, default=float(env_or("GFW_MPA_TIMEOUT", 120)))
    p.add_argument("--retries", type=int, default=int(env_or("GFW_MPA_RETRIES",
                                                             _http.DEFAULT_TOTAL)))
    p.add_argument("--retry-backoff", type=float,
                   default=float(env_or("GFW_MPA_RETRY_BACKOFF", _http.DEFAULT_BACKOFF_FACTOR)))
    p.add_argument("--insecure-tls", action="store_true",
                   help="Skip TLS verification (corporate/AV interception). Last resort.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan and exit. Needs no token and no network.")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- pure helpers

def resolve_window(args, today=None):
    """Return the half-open (start, end) window as ISO dates. ``end`` is EXCLUSIVE.

    Explicit --start/--end win. Otherwise the window ends ``lag_days`` before today (GFW
    publishes effort with a lag) and is ``window_days`` wide."""
    today = today or datetime.now(timezone.utc).date()
    if args.start and args.end:
        start, end = args.start, args.end
    else:
        if args.window_days < 1:
            raise ValueError(
                f"--window-days must be >= 1, got {args.window_days}: GFW's date-range is "
                f"half-open [start, end), so a zero-width window always returns no rows.")
        if args.lag_days < 0:
            raise ValueError(f"--lag-days must be >= 0, got {args.lag_days}")
        end_date = today - timedelta(days=args.lag_days)
        start = (end_date - timedelta(days=args.window_days)).isoformat()
        end = end_date.isoformat()
    if start >= end:
        raise ValueError(
            f"window [{start}, {end}) is empty or inverted; GFW would return no rows. "
            f"--end is EXCLUSIVE and must be strictly after --start.")
    return start, end


def load_mpa_index(path, limit=None):
    """Read the geometry-free MPA index CSV into a list of dicts (stdlib only)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"MPA index not found: {path}\n"
            "Generate it with `python scripts/prep_polygons.py` (it is written beside the "
            "GeoPackage). It is committed, so a fresh clone should already have it.")
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("WDPA_PID") or "").strip()]
    if not rows:
        raise ValueError(f"MPA index {path} has no usable rows (needs a WDPA_PID column).")
    return rows[:limit] if limit else rows


def _area_km2(row):
    try:
        return float(row.get("GIS_M_AREA") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def select_mpas(rows, min_area=DEFAULT_MIN_AREA_KM2, tail_slice=250, day_ordinal=None):
    """Split the index into the always-swept tier plus a rotating slice of the tail.

    A full 1799-region sweep costs ~6.5h (one request at a time at ~13s each), past the
    GitHub Actions job ceiling — so the run cannot cover everything every night. Rather
    than truncate arbitrarily:

      * **priority tier** — every MPA with ``GIS_M_AREA >= min_area``, swept on EVERY run,
        largest first. At the 1 km2 default that is 748 of 1799 (~162 min).
      * **rotating tail** — the sub-threshold remainder (median Ia/Ib MPA is 0.42 km2,
        below what GFW's AIS effort resolves anyway) sliced ``tail_slice`` at a time, the
        offset advancing with the date so successive runs cover the whole tail and none is
        never checked.

    The rotation is derived from the date, not from a stored cursor, so the job stays
    stateless — nothing to commit back, and a missed night does not desynchronize it.

    Returns ``(selected, meta)``; ``meta`` records the split so the run can report its own
    coverage instead of implying it looked everywhere."""
    day_ordinal = (datetime.now(timezone.utc).date().toordinal()
                   if day_ordinal is None else day_ordinal)
    priority = sorted([r for r in rows if _area_km2(r) >= min_area],
                      key=lambda r: -_area_km2(r))
    tail = [r for r in rows if _area_km2(r) < min_area]

    take = max(0, min(tail_slice, len(tail)))
    offset = (day_ordinal * take) % len(tail) if tail and take else 0
    rotated = (tail[offset:] + tail[:offset])[:take] if take else []

    return priority + rotated, {
        "min_area_km2": min_area,
        "priority_count": len(priority),
        "tail_total": len(tail),
        "tail_slice": len(rotated),
        "tail_offset": offset,
        "tail_cycle_runs": (-(-len(tail) // take) if take else None),
    }


def gear_filter(gear):
    """``geartype in ('a', 'b')``. Values are LOWERCASE — uppercase silently matches nothing."""
    items = [g.strip().lower() for g in gear.split(",") if g.strip()]
    if not items:
        raise ValueError("--gear resolved to an empty list; omit the flag for no filter")
    return "geartype in (" + ", ".join(f"'{g}'" for g in items) + ")"


def build_params(args, start, end):
    """Query params for the 4WINGS report. Pure, so the offline suite can assert on it."""
    params = {
        "datasets[0]": args.dataset,
        "date-range": f"{start},{end}",
        "group-by": "VESSEL_ID",
        "spatial-resolution": args.spatial_resolution,
        "temporal-resolution": args.temporal_resolution,
        "format": "JSON",
    }
    if args.gear:
        params["filters[0]"] = gear_filter(args.gear)
    return params


def extract_rows(payload):
    """Pull vessel rows out of 4WINGS' nested envelope.

    Shape is ``{"entries": [{"<dataset>:<ver>": [row, ...]}]}`` and an EMPTY result is
    ``{"entries": [{"<dataset>:<ver>": null}]}`` — note it is a list holding a dict whose
    single value is null, not an empty list. Anything unexpected yields no rows rather
    than raising: one malformed region must not abort a 1799-region sweep."""
    rows = []
    if not isinstance(payload, dict):
        return rows
    for entry in payload.get("entries") or []:
        if isinstance(entry, list):
            rows.extend(r for r in entry if isinstance(r, dict))
        elif isinstance(entry, dict):
            for value in entry.values():
                if isinstance(value, list):
                    rows.extend(r for r in value if isinstance(r, dict))
    return rows


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_event(row, mpa):
    """One raw GFW row + its MPA -> a flat intrusion record."""
    return {
        "wdpa_pid": str(mpa.get("WDPA_PID", "")).strip(),
        "wdpa_id": str(mpa.get("WDPAID", "")).strip() or None,
        "mpa_name": (mpa.get("NAME") or "").strip() or None,
        "iso3": (mpa.get("ISO3") or "").strip() or None,
        "iucn_cat": (mpa.get("IUCN_CAT") or "").strip() or None,
        "vessel_id": row.get("vesselId"),
        "mmsi": str(row["mmsi"]).strip() if row.get("mmsi") not in (None, "") else None,
        "vessel_name": (row.get("shipName") or "").strip() or None,
        "flag": (row.get("flag") or "").strip() or None,
        "gear_type": (row.get("geartype") or "").strip() or None,
        "imo": (row.get("imo") or "").strip() or None,
        "callsign": (row.get("callsign") or "").strip() or None,
        "fishing_hours": _num(row.get("hours")) or 0.0,
        "entry_timestamp": row.get("entryTimestamp"),
        "exit_timestamp": row.get("exitTimestamp"),
        "lat": _num(row.get("lat")),
        "lon": _num(row.get("lon")),
        "source": SOURCE,
    }


def aggregate_events(events, min_hours=0.0):
    """Collapse per-cell rows to one record per (MPA, vessel).

    4WINGS returns a row per vessel per spatial cell, so a single vessel in a single MPA
    can appear several times with the hours split between cells. Sum the hours and take
    the widest entry/exit span. Sorted by hours desc so the digest leads with the worst."""
    merged = {}
    for ev in events:
        key = (ev["wdpa_pid"], ev.get("vessel_id") or ev.get("mmsi") or ev.get("vessel_name"))
        cur = merged.get(key)
        if cur is None:
            merged[key] = dict(ev, cells=1)
            continue
        cur["fishing_hours"] += ev["fishing_hours"]
        cur["cells"] += 1
        for field, pick in (("entry_timestamp", min), ("exit_timestamp", max)):
            a, b = cur.get(field), ev.get(field)
            cur[field] = pick(a, b) if a and b else (a or b)
        for field in ("mmsi", "vessel_name", "flag", "gear_type", "imo", "callsign",
                      "lat", "lon"):
            if cur.get(field) is None:
                cur[field] = ev.get(field)
    out = [e for e in merged.values() if e["fishing_hours"] >= min_hours]
    return sorted(out, key=lambda e: (-e["fishing_hours"], e["wdpa_pid"]))


def summarize_markdown(result, top_mpas=40, top_vessels=5):
    """Render the run as a markdown digest for $GITHUB_STEP_SUMMARY.

    Bounded on purpose: a full sweep can touch 1799 regions and the step summary has a
    hard size limit, so the tail is reported as a count, never silently dropped."""
    w = result["window"]
    cov = result.get("coverage", {})
    lines = ["# GFW MPA sweep", "",
             f"**Window** `{w['start']}` → `{w['end']}` (end exclusive)  ",
             f"**MPAs queried** {result['mpas_queried']}  ",
             f"**MPAs with fishing effort** {result['mpas_with_effort']}  ",
             f"**Vessel events** {len(result['events'])}  ",
             f"**Failed regions** {len(result['errors'])}", ""]

    if cov:
        tail = (f"plus {cov.get('tail_slice', 0)} of {cov.get('tail_total', 0)} smaller "
                f"ones (rotating; the tail is fully covered every "
                f"{cov.get('tail_cycle_runs')} runs)"
                if cov.get("tail_slice") else "and no smaller ones")
        lines += [f"_Coverage: all {cov.get('priority_count', 0)} MPAs ≥ "
                  f"{cov.get('min_area_km2')} km² {tail}, out of "
                  f"{cov.get('index_total', 0)} in the index._", ""]
        if cov.get("budget_exhausted"):
            lines += [f"> ⚠️ **Time budget reached — {cov.get('skipped')} selected MPAs "
                      f"were not queried in this run.**", ""]

    events = result["events"]
    if not events:
        lines += ["No fishing effort recorded inside any Ia/Ib marine MPA in this window.",
                  ""]
    else:
        by_mpa = {}
        for ev in events:
            by_mpa.setdefault(ev["wdpa_pid"], []).append(ev)
        ranked = sorted(by_mpa.items(),
                        key=lambda kv: -sum(e["fishing_hours"] for e in kv[1]))
        lines += ["## MPAs with effort", "",
                  "| MPA | ISO3 | IUCN | Vessels | Fishing hours |",
                  "|---|---|---|---:|---:|"]
        for pid, evs in ranked[:top_mpas]:
            head = evs[0]
            name = (head["mpa_name"] or f"WDPA_PID {pid}").replace("|", "\\|")
            hours = sum(e["fishing_hours"] for e in evs)
            lines.append(f"| {name} | {head['iso3'] or '—'} | {head['iucn_cat'] or '—'} "
                         f"| {len(evs)} | {hours:.2f} |")
        if len(ranked) > top_mpas:
            lines.append(f"| _…and {len(ranked) - top_mpas} more MPAs_ | | | | |")
        lines.append("")

        lines += ["## Vessels", ""]
        for pid, evs in ranked[:top_mpas]:
            head = evs[0]
            name = (head["mpa_name"] or f"WDPA_PID {pid}").replace("|", "\\|")
            lines.append(f"**{name}** (`{pid}`)")
            lines.append("")
            lines.append("| Vessel | MMSI | Flag | Gear | Hours | Entry | Exit |")
            lines.append("|---|---|---|---|---:|---|---|")
            for ev in evs[:top_vessels]:
                lines.append(
                    f"| {(ev['vessel_name'] or '—').replace('|', chr(92) + '|')} "
                    f"| {ev['mmsi'] or '—'} | {ev['flag'] or '—'} "
                    f"| {(ev['gear_type'] or '—').lower()} | {ev['fishing_hours']:.2f} "
                    f"| {ev['entry_timestamp'] or '—'} | {ev['exit_timestamp'] or '—'} |")
            if len(evs) > top_vessels:
                lines.append(f"| _…and {len(evs) - top_vessels} more_ | | | | | | |")
            lines.append("")

    if result["errors"]:
        lines += ["## Failed regions", "",
                  "| WDPA_PID | MPA | Problem |", "|---|---|---|"]
        for err in result["errors"][:20]:
            lines.append(f"| {err['wdpa_pid']} | {(err.get('mpa_name') or '—')} "
                         f"| {err['error']} |")
        if len(result["errors"]) > 20:
            lines.append(f"| _…and {len(result['errors']) - 20} more_ | | |")
        lines.append("")

    lines.append("> Effort is GFW's AIS-derived apparent fishing. Presence in an MPA is a "
                 "candidate to investigate, not a proven violation — many reserves permit "
                 "innocent passage, and some permit regulated fishing.")
    return "\n".join(lines) + "\n"


def write_json_atomic(path, data):
    """Serialize to a sibling .tmp, fsync, then rename over the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return path


# ------------------------------------------------------------------------------- network

_local = threading.local()


def _session(policy, headers, verify):
    """One requests.Session per worker thread. Sessions are not documented as
    thread-safe, so sharing one across the pool is avoided."""
    sess = getattr(_local, "session", None)
    if sess is None:
        sess = _http.build_session(policy, verify=verify, headers=headers)
        _local.session = sess
    return sess


def query_mpa(mpa, params, policy, headers, timeout, verify):
    """Query one MPA. Returns (events, error_or_None) — never raises for one region."""
    pid = str(mpa.get("WDPA_PID", "")).strip()
    try:
        resp = _session(policy, headers, verify).post(
            GFW_REPORT_URL, params=params,
            json={"region": {"dataset": MPA_REGION_DATASET, "id": pid}},
            timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - one region must not abort the sweep
        return [], {"wdpa_pid": pid, "mpa_name": mpa.get("NAME"),
                    "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    if resp.status_code != 200:
        return [], {"wdpa_pid": pid, "mpa_name": mpa.get("NAME"),
                    "status": resp.status_code,
                    "error": f"HTTP {resp.status_code}: {resp.text[:160]}"}
    try:
        payload = resp.json()
    except ValueError:
        return [], {"wdpa_pid": pid, "mpa_name": mpa.get("NAME"),
                    "error": "response was not JSON"}
    return [normalize_event(r, mpa) for r in extract_rows(payload)], None


def sweep(mpas, params, policy, headers, timeout, verify, workers=DEFAULT_WORKERS,
          deadline=None, progress_every=50):
    """Query each MPA in turn; collect events, per-region errors, and anything skipped.

    Runs in chunks of ``workers`` so the time budget can be checked between chunks. With
    the default ``workers=1`` this is a plain sequential loop, which is the only mode the
    GFW token tolerates — see DEFAULT_WORKERS. The chunked shape is kept so a token with a
    higher concurrency allowance can use it without a second code path.

    ``deadline`` is a ``time.monotonic()`` value. Whatever is left when it passes comes
    back as ``skipped`` so the caller reports partial coverage rather than implying the
    whole list was checked."""
    events, errors, done = [], [], 0
    step = max(1, workers)
    i = 0
    with ThreadPoolExecutor(max_workers=step) as pool:
        while i < len(mpas):
            if deadline is not None and time.monotonic() >= deadline:
                skipped = mpas[i:]
                print(f"  time budget reached — skipping {len(skipped)} remaining MPAs",
                      flush=True)
                return events, errors, skipped
            chunk = mpas[i:i + step]
            i += len(chunk)
            for fut in [pool.submit(query_mpa, m, params, policy, headers, timeout, verify)
                        for m in chunk]:
                got, err = fut.result()
                events.extend(got)
                if err:
                    errors.append(err)
                done += 1
            if progress_every and done % progress_every == 0:
                print(f"  {done}/{len(mpas)} MPAs queried, "
                      f"{len(events)} raw rows, {len(errors)} errors", flush=True)
    return events, errors, []


# ---------------------------------------------------------------------------------- main

def main(argv=None):
    load_env()
    args = parse_args(argv)
    start, end = resolve_window(args)
    all_rows = load_mpa_index(args.index)
    mpas, coverage = select_mpas(all_rows, args.min_area, args.tail_slice)
    if args.limit:
        mpas = mpas[:args.limit]
    params = build_params(args, start, end)

    print("GFW MPA SWEEP")
    print(f"  window:   [{start}, {end})   <- end is EXCLUSIVE")
    print(f"  index:    {len(all_rows)} MPAs ({args.index})")
    print(f"  selected: {coverage['priority_count']} >= {coverage['min_area_km2']} km2 "
          f"+ {coverage['tail_slice']} of {coverage['tail_total']} smaller (rotating, "
          f"offset {coverage['tail_offset']}, full cycle every "
          f"{coverage['tail_cycle_runs']} runs)")
    if args.limit:
        print(f"  --limit:  truncated to the first {len(mpas)}")
    print(f"  region:   {MPA_REGION_DATASET} / WDPA_PID")
    print(f"  gear:     {args.gear or 'ALL (no filter)'}")
    print(f"  workers:  {args.workers}"
          + ("" if args.workers == 1 else "  WARNING: >1 triggers HTTP 429 on this token"))
    print(f"  budget:   {args.max_minutes or 'none'} min "
          f"(~{len(mpas) * 13 / 60:.0f} min estimated at ~13s/MPA)")

    if args.dry_run:
        print("\n--dry-run: no token read, no request sent. Params:")
        print(json.dumps(params, indent=2))
        for m in mpas[:5]:
            print(f"  would query {m['WDPA_PID']:>12}  {m.get('NAME', '')[:52]}")
        if len(mpas) > 5:
            print(f"  … and {len(mpas) - 5} more")
        return 0

    token = (os.getenv("GFW_API_TOKEN") or "").strip()
    if not token:
        raise SystemExit(
            "Missing GFW_API_TOKEN. Put it in a .env file at the repo root or in "
            "scripts/, or export it (see .env.example).")

    policy = _http.retry_policy(total=args.retries, backoff_factor=args.retry_backoff)
    print(f"  {_http.describe(policy)}\n")
    headers = {"Authorization": f"Bearer {token}",
               "Content-Type": "application/json", "Accept": "application/json"}
    verify = False if args.insecure_tls else None
    if args.insecure_tls:
        # ASCII only: the Windows console is cp1252 and a non-ASCII print raises
        # UnicodeEncodeError. The markdown digest is written UTF-8 and may use emoji.
        print("  WARNING: TLS verification DISABLED (--insecure-tls)\n")

    started = datetime.now(timezone.utc)
    deadline = (time.monotonic() + args.max_minutes * 60) if args.max_minutes else None
    raw_events, errors, skipped = sweep(mpas, params, policy, headers, args.timeout,
                                        verify, args.workers, deadline)
    events = aggregate_events(raw_events, args.min_hours)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()

    result = {
        "generated_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": SOURCE,
        "window": {"start": start, "end": end, "end_exclusive": True},
        "query": {"region_dataset": MPA_REGION_DATASET, "gear": args.gear,
                  "dataset": args.dataset,
                  "spatial_resolution": args.spatial_resolution,
                  "temporal_resolution": args.temporal_resolution},
        # Coverage is reported, never implied: a run that ran out of budget says so and
        # names how many MPAs it did not reach (cf. aisstream_fetch's capture accounting).
        "coverage": dict(coverage,
                         index_total=len(all_rows),
                         selected=len(mpas),
                         queried=len(mpas) - len(skipped),
                         skipped=len(skipped),
                         budget_exhausted=bool(skipped)),
        "mpas_queried": len(mpas) - len(skipped),
        "mpas_with_effort": len({e["wdpa_pid"] for e in events}),
        "raw_rows": len(raw_events),
        "elapsed_s": round(elapsed, 1),
        "errors": errors,
        "events": events,
    }

    out = args.out or (OUTPUT_DIR /
                       f"gfw_mpa_sweep_{started.strftime('%Y-%m-%dT%H-%M-%SZ')}.json")
    write_json_atomic(Path(out), result)

    print(f"\nSwept {result['mpas_queried']} MPAs in {elapsed / 60:.0f} min: "
          f"{result['mpas_with_effort']} with effort, {len(events)} vessel events, "
          f"{len(errors)} failed regions"
          + (f", {len(skipped)} SKIPPED (time budget)" if skipped else ""))
    for ev in events[:10]:
        print(f"  {ev['fishing_hours']:7.2f} h  {(ev['mpa_name'] or ev['wdpa_pid'])[:40]:42s}"
              f" {(ev['vessel_name'] or '?')[:24]:26s} {ev['flag'] or '?'}")
    print(f"\nWrote {Path(out)}")

    if args.summary_md:
        Path(args.summary_md).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_md, "a", encoding="utf-8") as f:
            f.write(summarize_markdown(result))
        print(f"Wrote digest -> {args.summary_md}")

    queried = result["mpas_queried"]
    error_rate = len(errors) / queried if queried else 1.0
    if error_rate > args.max_error_rate:
        print(f"\nFAILED: {len(errors)}/{queried} regions errored "
              f"({error_rate:.0%} > --max-error-rate {args.max_error_rate:.0%}); "
              f"the sweep is incomplete.\nIf these are HTTP 429, check --workers is 1: "
              f"the GFW token rejects concurrent requests.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
