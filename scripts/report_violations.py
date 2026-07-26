import argparse
import csv
import html
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

# The objective of this script (P3 operator-facing output, PHASE3_PLAN Workstream 2B):
# - Read one or more fuse_violations `violation_events.json` files (or run dirs) and aggregate their
#   events into a single RANKED table (by violation_score, dark first at equal score)
# - Write a CSV (for spreadsheets / further analysis) and a self-contained HTML page (zero external
#   assets — same hardened, offline style as review_server.py) an operator can open to see the top
#   candidate violations with their evidence and vessel identity.
#
# It is a pure read-only reporter over the stage-4 output — it never fetches, fuses, or mutates the
# run. Pure stdlib (json/csv/html), so it runs and is tested anywhere. The event schema it consumes
# is pinned in docs/PIPELINE.md (`violation_events.json`).
#
# House rule: this is a *view* over violation_events; it does no fusion or ingestion of its own.

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Columns emitted to CSV (full detail). HTML shows a readable subset of these.
CSV_COLUMNS = [
    "rank", "violation_score", "ais_status", "mpa_name", "mpa_id", "run_dir",
    "event_id", "detection_id", "presence_confidence", "mmsi", "vessel_name", "flag",
    "gear_type", "imo", "iuu_listed", "fishing_hours", "fishing_probability",
    "evidence", "lat", "lon", "chip",
]


def env_or(name, default):
    val = os.getenv(name)
    return val if val not in (None, "") else default


def truthy(val):
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Aggregate fuse_violations violation_events into a ranked CSV + HTML report "
                    "(PHASE3_PLAN Workstream 2B). Pure stdlib, read-only.")
    p.add_argument("inputs", nargs="*", type=Path,
                   help="violation_events.json files and/or run dirs (each dir's violation_events.json "
                        "is read). Also accepted via --run / --events.")
    p.add_argument("--run", type=Path, nargs="*", default=None,
                   help="Run dirs to read <dir>/violation_events.json from.")
    p.add_argument("--events", type=Path, nargs="*", default=None,
                   help="Explicit violation_events.json paths.")
    p.add_argument("--out-csv", type=Path, default=env_or("REPORT_OUT_CSV", None),
                   help="CSV output path. Default: violations_report_<UTC>.csv next to the first input.")
    p.add_argument("--out-html", type=Path, default=env_or("REPORT_OUT_HTML", None),
                   help="HTML output path. Default: violations_report_<UTC>.html next to the first input.")
    p.add_argument("--top", type=int, default=int(env_or("REPORT_TOP", 0)),
                   help="Keep only the top N rows by score (0 = all).")
    p.add_argument("--min-score", type=float, default=float(env_or("REPORT_MIN_SCORE", 0.0)),
                   help="Drop events below this violation_score. Default 0.0.")
    p.add_argument("--dark-only", action="store_true", default=truthy(env_or("REPORT_DARK_ONLY", "")),
                   help="Report only DARK events (the strongest product signal).")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- pure: load / flatten / rank

def resolve_inputs(args):
    """Collect event-file paths from positionals + --run (dir → dir/violation_events.json) + --events."""
    paths = []
    for item in list(args.inputs) + list(args.run or []):
        item = Path(item)
        paths.append(item / "violation_events.json" if item.is_dir() else item)
    paths += [Path(e) for e in (args.events or [])]
    # de-dup, preserve order
    seen, out = set(), []
    for p in paths:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def load_events(path):
    """Read one violation_events.json → (run_label, [events]). Tolerates a bare list."""
    doc = json.loads(Path(path).read_text())
    if isinstance(doc, list):
        return Path(path).parent.name, doc
    return doc.get("run_dir") or Path(path).parent.name, doc.get("violation_events", [])


def flatten_event(event, run_label):
    """One violation_events entry → a flat row dict (CSV_COLUMNS minus the assigned 'rank')."""
    centroid = event.get("centroid_wgs84") or [None, None]
    lon, lat = (centroid + [None, None])[:2]
    evidence = event.get("evidence") or []
    return {
        "violation_score": event.get("violation_score"),
        "ais_status": event.get("ais_status"),
        "mpa_name": event.get("mpa_name"),
        "mpa_id": event.get("mpa_id"),
        "run_dir": run_label,
        "event_id": event.get("event_id"),
        "detection_id": event.get("detection_id"),
        "presence_confidence": event.get("presence_confidence"),
        "mmsi": event.get("mmsi"),
        "vessel_name": event.get("vessel_name"),
        "flag": event.get("flag"),
        "gear_type": event.get("gear_type"),
        "imo": event.get("imo"),
        "iuu_listed": event.get("iuu_listed"),
        "fishing_hours": event.get("fishing_hours"),
        "fishing_probability": event.get("fishing_probability"),
        "evidence": "+".join(str(e) for e in evidence),
        "lat": lat,
        "lon": lon,
        "chip": event.get("chip"),
    }


def collect_rows(event_files, dark_only=False, min_score=0.0, top=0):
    """Load all files, flatten, filter, rank by score (dark first at equal score), assign rank."""
    rows = []
    for path in event_files:
        run_label, events = load_events(path)
        for ev in events:
            rows.append(flatten_event(ev, run_label))
    if dark_only:
        rows = [r for r in rows if r["ais_status"] == "dark"]
    rows = [r for r in rows if (r["violation_score"] or 0.0) >= min_score]
    # rank: score desc; at equal score, dark before matched; stable by run/event id
    rows.sort(key=lambda r: (-(r["violation_score"] or 0.0), r["ais_status"] != "dark",
                             str(r["run_dir"]), str(r["event_id"])))
    if top and top > 0:
        rows = rows[:top]
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def rows_to_csv(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


# --------------------------------------------------------------------------- HTML (self-contained)

def _fmt(v):
    return "" if v is None else html.escape(str(v))


def _score_cell(score):
    s = 0.0 if score is None else max(0.0, min(1.0, float(score)))
    pct = round(s * 100)
    label = "" if score is None else f"{score:.3f}"
    return (f'<div class="bar"><span style="width:{pct}%"></span></div>'
            f'<span class="score">{label}</span>')


def rows_to_html(rows, sources):
    n = len(rows)
    darks = sum(1 for r in rows if r["ais_status"] == "dark")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    head = (f'<tr><th>#</th><th>score</th><th>status</th><th>MPA</th><th>vessel</th>'
            f'<th>flag</th><th>gear</th><th>fishing h</th><th>IUU</th><th>evidence</th>'
            f'<th>lat, lon</th><th>run</th></tr>')
    body_rows = []
    for r in rows:
        cls = "dark" if r["ais_status"] == "dark" else "matched"
        vessel = r["vessel_name"] or (f'MMSI {r["mmsi"]}' if r["mmsi"] else "—")
        latlon = "" if r["lat"] is None else f'{r["lat"]:.4f}, {r["lon"]:.4f}'
        iuu = "⚠︎" if r["iuu_listed"] else ""
        body_rows.append(
            f'<tr class="{cls}"><td>{r["rank"]}</td><td class="scorecell">{_score_cell(r["violation_score"])}</td>'
            f'<td><span class="pill {cls}">{_fmt(r["ais_status"])}</span></td>'
            f'<td>{_fmt(r["mpa_name"])}</td><td>{_fmt(vessel)}</td><td>{_fmt(r["flag"])}</td>'
            f'<td>{_fmt(r["gear_type"])}</td><td>{_fmt(r["fishing_hours"])}</td><td class="iuu">{iuu}</td>'
            f'<td>{_fmt(r["evidence"])}</td><td class="mono">{_fmt(latlon)}</td><td class="mono">{_fmt(r["run_dir"])}</td></tr>')
    src = html.escape(", ".join(str(s) for s in sources))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MPA violation candidates</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 14px/1.45 system-ui, sans-serif; margin: 1.5rem; }}
  h1 {{ font-size: 1.25rem; margin: 0 0 .25rem; }}
  .meta {{ color: #888; margin-bottom: 1rem; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ text-align: left; padding: .4rem .55rem; border-bottom: 1px solid #8883; vertical-align: middle; }}
  th {{ font-weight: 600; border-bottom: 2px solid #8886; position: sticky; top: 0; background: Canvas; }}
  tr.dark {{ background: color-mix(in srgb, crimson 8%, transparent); }}
  td.scorecell {{ white-space: nowrap; min-width: 120px; }}
  .bar {{ display: inline-block; width: 70px; height: 8px; border-radius: 4px; background: #8883;
          overflow: hidden; vertical-align: middle; margin-right: .4rem; }}
  .bar span {{ display: block; height: 100%; background: crimson; }}
  .score {{ font-variant-numeric: tabular-nums; }}
  .pill {{ font-size: .75rem; padding: .05rem .4rem; border-radius: 999px; }}
  .pill.dark {{ background: crimson; color: white; }}
  .pill.matched {{ background: #8884; }}
  .iuu {{ color: crimson; text-align: center; }}
  .mono {{ font-family: ui-monospace, monospace; font-size: .85em; }}
  .empty {{ color: #888; padding: 1rem 0; }}
</style></head><body>
<h1>MPA violation candidates</h1>
<div class="meta">{n} event(s) · {darks} dark · generated {generated}<br>sources: {src}</div>
{('<table><thead>' + head + '</thead><tbody>' + ''.join(body_rows) + '</tbody></table>') if rows else '<div class="empty">No events matched the filters.</div>'}
</body></html>
"""


# --------------------------------------------------------------------------- IO

def write_atomic(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def main(argv=None):
    args = parse_args(argv)
    event_files = resolve_inputs(args)
    if not event_files:
        raise RuntimeError("No inputs. Pass violation_events.json files / run dirs (positional, "
                           "--run, or --events).")
    missing = [str(p) for p in event_files if not Path(p).exists()]
    if missing:
        raise RuntimeError(f"Not found: {', '.join(missing)}")

    rows = collect_rows(event_files, args.dark_only, args.min_score, args.top)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    base = Path(event_files[0]).parent
    out_csv = Path(args.out_csv) if args.out_csv else base / f"violations_report_{ts}.csv"
    out_html = Path(args.out_html) if args.out_html else base / f"violations_report_{ts}.html"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(out_csv, rows_to_csv(rows))
    write_atomic(out_html, rows_to_html(rows, event_files))

    darks = sum(1 for r in rows if r["ais_status"] == "dark")
    print("REPORT:")
    print(f"  inputs={len(event_files)}  events={len(rows)} ({darks} dark)"
          f"  {'dark-only ' if args.dark_only else ''}min-score={args.min_score}"
          f"{f'  top={args.top}' if args.top else ''}")
    if rows:
        top = rows[0]
        print(f"  top: score={top['violation_score']} {top['ais_status']} "
              f"{top['vessel_name'] or top['mmsi'] or '—'} @ {top['mpa_name']}")
    print(f"  -> {out_csv}")
    print(f"  -> {out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
