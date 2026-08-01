import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv   # optional; only used to pick up ALERT_* knobs from .env
except ImportError:                  # keep the module importable (and offline-testable) without it
    def load_dotenv(*_args, **_kwargs):
        return False

# The objective of this script (the alerting interface — the design proposal's final promised output):
# - Read violation_events.json files (from fuse_violations.py and/or geofence_ais.py) or run dirs
# - Assign each event a SEVERITY from its violation_score (with an IUU listing forcing critical)
# - Print a ranked, MPA-grouped digest an operator can act on, and optionally write it as markdown
#
# WHY this exists separately from report_violations.py: that is a full RECORD (every event, every
# column, CSV + HTML for review and archive). This answers a different question — "what needs
# attention right now?" — so it filters to a severity floor, groups by MPA, and fits in a terminal or
# a cron mail. Same input contract, different question.
#
# It is a VIEW: read-only, no fusion, no ingestion, no scoring of its own. violation_score is computed
# upstream (fuse_violations / geofence_ais); this only BUCKETS it. If a threshold looks wrong, the fix
# is usually the upstream weight, not the threshold here.
#
# Handles all three statuses without special-casing: "dark" (detected, no AIS), "matched" (detected +
# AIS) and "reported" (AIS inside the MPA, no detection). Unknown future statuses pass through too.
#
# --exit-code makes it usable as an automation gate (non-zero when something at/above the floor
# exists), which is off by default so interactive runs never look like failures.
#
# Pure stdlib, offline-tested (scripts/tests/test_alert_violations.py). The event loader is
# deliberately duplicated rather than imported from report_violations.py: the two views ask different
# questions of the same pinned schema, and coupling them would mean a change to the report's table
# breaks alerting (see docs/AUDIT.md #7 on the standalone-script trade-off).

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Severity floors on violation_score. Ordered most severe first; an event takes the first tier it
# clears. Defaults align with the upstream weights: a dark detection carries its full presence
# confidence, so a confident dark hit lands critical, while an apparent transit (geofence transit
# floor 0.10) stays below the default medium floor and does not page anyone.
SEVERITIES = ("critical", "high", "medium", "low")

load_dotenv(SCRIPT_DIR / ".env")
load_dotenv(REPO_ROOT / ".env")


def env_or(name, default):
    val = os.getenv(name)
    return val if val not in (None, "") else default


def truthy(val):
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Ranked severity digest of violation_events — the 'what needs attention now' "
                    "view over fuse_violations / geofence_ais output.")
    p.add_argument("inputs", nargs="*", type=Path,
                   help="violation_events.json files and/or run dirs (each dir's violation_events.json).")
    p.add_argument("--run", type=Path, nargs="*", default=None,
                   help="Run dirs to read <dir>/violation_events.json from.")
    p.add_argument("--events", type=Path, nargs="*", default=None,
                   help="Explicit violation_events.json paths (e.g. a geofence_events_*.json).")

    tiers = p.add_argument_group("severity thresholds (violation_score floors)")
    tiers.add_argument("--critical", type=float, default=float(env_or("ALERT_CRITICAL", 0.80)))
    tiers.add_argument("--high", type=float, default=float(env_or("ALERT_HIGH", 0.50)))
    tiers.add_argument("--medium", type=float, default=float(env_or("ALERT_MEDIUM", 0.25)))

    p.add_argument("--min-severity", choices=SEVERITIES, default=env_or("ALERT_MIN_SEVERITY", "medium"),
                   help="Only alert at or above this tier. Default medium (suppresses apparent transits).")
    p.add_argument("--dark-only", action="store_true", default=truthy(env_or("ALERT_DARK_ONLY", "")),
                   help="Only dark events — the strongest product signal.")
    p.add_argument("--top", type=int, default=int(env_or("ALERT_TOP", 0)),
                   help="Keep only the N highest-scoring alerts (0 = all).")
    p.add_argument("--out-markdown", type=Path, default=env_or("ALERT_OUT_MD", None),
                   help="Also write the digest as markdown to this path.")
    p.add_argument("--quiet", action="store_true", help="Suppress the console digest (use with --out-markdown).")
    p.add_argument("--exit-code", action="store_true", default=truthy(env_or("ALERT_EXIT_CODE", "")),
                   help="Return exit status 1 when any alert is emitted, so a cron/CI job can gate on it. "
                        "Off by default so an interactive run never looks like a failure.")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- input

def collect_inputs(args):
    """Event-file paths from positionals + --run (dir -> dir/violation_events.json) + --events."""
    paths = []
    for item in list(args.inputs or []) + list(args.run or []):
        item = Path(item)
        paths.append(item / "violation_events.json" if item.is_dir() else item)
    paths += [Path(p) for p in (args.events or [])]
    seen, ordered = set(), []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            ordered.append(path)
    return ordered


def load_events(path):
    """Read one violation_events.json -> (source_label, [events]). Tolerates a bare list."""
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"Events file not found: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(doc, list):
        return path.parent.name, doc
    return doc.get("run_dir") or path.parent.name, doc.get("violation_events", [])


# --------------------------------------------------------------------------- severity

def severity_of(event, critical=0.80, high=0.50, medium=0.25):
    """Bucket one event. An IUU-listed vessel is critical regardless of score — an entry on the IUU
    list is a standing fact about the vessel, not a property of this sighting's confidence."""
    if event.get("iuu_listed"):
        return "critical"
    score = event.get("violation_score")
    if score is None:
        return "low"          # unscored: surface it only at --min-severity low, never page on it
    if score >= critical:
        return "critical"
    if score >= high:
        return "high"
    if score >= medium:
        return "medium"
    return "low"


def at_or_above(severity, floor):
    """True if `severity` is at least as severe as `floor` (SEVERITIES is most-severe-first)."""
    return SEVERITIES.index(severity) <= SEVERITIES.index(floor)


def build_alerts(event_files, args):
    """Load, bucket, filter and rank. Returns (alerts, counts_by_severity, scanned_total)."""
    alerts, scanned = [], 0
    for path in event_files:
        source, events = load_events(path)
        for event in events:
            scanned += 1
            severity = severity_of(event, args.critical, args.high, args.medium)
            if not at_or_above(severity, args.min_severity):
                continue
            if args.dark_only and event.get("ais_status") != "dark":
                continue
            alerts.append({"severity": severity, "source": source, "event": event})

    alerts.sort(key=lambda a: (SEVERITIES.index(a["severity"]),
                               -(a["event"].get("violation_score") or 0.0),
                               str(a["event"].get("event_id"))))
    if args.top and args.top > 0:
        alerts = alerts[: args.top]
    counts = {sev: sum(1 for a in alerts if a["severity"] == sev) for sev in SEVERITIES}
    return alerts, counts, scanned


# --------------------------------------------------------------------------- formatting

def _describe(event):
    """The one-line 'what is it' fragment, adapted to which evidence the event actually has."""
    status = event.get("ais_status") or "?"
    bits = [status]
    mmsi = event.get("mmsi")
    if mmsi:
        bits.append(f"MMSI {mmsi}")
    name = event.get("vessel_name")
    if name:
        bits.append(str(name))
    flag = event.get("flag")
    if flag:
        bits.append(f"({flag})")
    prob = event.get("fishing_probability")
    if prob is not None:
        bits.append(f"fishing {float(prob):.2f}")
    if event.get("iuu_listed"):
        bits.append("IUU-LISTED")
    return "  ".join(bits)


def _location(event):
    centroid = event.get("centroid_wgs84") or []
    if len(centroid) >= 2 and centroid[0] is not None and centroid[1] is not None:
        return f"{centroid[1]:.5f},{centroid[0]:.5f}"      # lat,lon — the order operators paste into a map
    return "-"


def mpa_label(event):
    """Group heading for one event: the MPA name, else its id, else the bbox-run case."""
    name = event.get("mpa_name")
    if name:
        return str(name)
    mpa_id = event.get("mpa_id")
    return f"MPA {mpa_id}" if mpa_id is not None else "(no MPA)"


def group_by_mpa(alerts):
    """{mpa_label: [alert, ...]} ordered by each group's most severe / highest-scoring alert.
    `alerts` arrives already ranked, so each group's first entry is its worst."""
    groups = {}
    for alert in alerts:
        groups.setdefault(mpa_label(alert["event"]), []).append(alert)
    return dict(sorted(groups.items(),
                       key=lambda kv: (SEVERITIES.index(kv[1][0]["severity"]),
                                       -(kv[1][0]["event"].get("violation_score") or 0.0))))


def format_console(alerts, counts, scanned, sources, generated):
    lines = []
    total = len(alerts)
    summary = ", ".join(f"{counts[s]} {s}" for s in SEVERITIES if counts[s])
    lines.append(f"ALERTS  {generated}   {total} alert(s)" + (f": {summary}" if summary else ""))
    lines.append(f"scanned {scanned} event(s) from {len(sources)} source(s)")
    if not alerts:
        lines.append("")
        lines.append("  nothing at or above the severity floor.")
        return "\n".join(lines)
    for label, group in group_by_mpa(alerts).items():
        lines.append("")
        lines.append(label)
        for alert in group:
            event = alert["event"]
            marker = "!" if alert["severity"] == "critical" else " "
            score = event.get("violation_score")
            score_s = "  n/a" if score is None else f"{float(score):.2f}"
            lines.append(f"  {marker} {alert['severity'].upper():<8s}  score {score_s}  {_describe(event)}")
            detail = [_location(event)]
            if event.get("chip"):
                detail.append(str(event["chip"]))
            if event.get("duration_hours") is not None:
                detail.append(f"{float(event['duration_hours']):.2f} h in MPA")
            evidence = event.get("evidence") or []
            if evidence:
                detail.append("[" + "+".join(str(e) for e in evidence) + "]")
            lines.append(f"    {'  '.join(detail)}")
    return "\n".join(lines)


def format_markdown(alerts, counts, scanned, sources, generated):
    lines = [f"# Violation alerts — {generated}", ""]
    summary = ", ".join(f"**{counts[s]}** {s}" for s in SEVERITIES if counts[s]) or "none"
    lines.append(f"{len(alerts)} alert(s): {summary}. Scanned {scanned} event(s) from "
                 f"{len(sources)} source(s).")
    lines.append("")
    if not alerts:
        lines.append("_Nothing at or above the severity floor._")
        return "\n".join(lines) + "\n"
    for label, group in group_by_mpa(alerts).items():
        lines.append(f"## {label}")
        lines.append("")
        lines.append("| Severity | Score | Status | Vessel | Location (lat,lon) | Evidence |")
        lines.append("|---|---|---|---|---|---|")
        for alert in group:
            event = alert["event"]
            score = event.get("violation_score")
            score_s = "n/a" if score is None else f"{float(score):.2f}"
            vessel = " ".join(str(x) for x in
                              (event.get("mmsi") or "", event.get("vessel_name") or "") if x).strip() or "—"
            evidence = "+".join(str(e) for e in (event.get("evidence") or [])) or "—"
            lines.append(f"| {alert['severity'].upper()} | {score_s} | "
                         f"{event.get('ais_status') or '?'} | {vessel} | {_location(event)} | {evidence} |")
        lines.append("")
    lines.append(f"_Sources: {', '.join(sources)}_")
    return "\n".join(lines) + "\n"


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


# --------------------------------------------------------------------------- orchestration

def main(argv=None):
    args = parse_args(argv)
    if not (args.critical >= args.high >= args.medium):
        raise RuntimeError(f"Thresholds must satisfy critical >= high >= medium, got "
                           f"{args.critical} / {args.high} / {args.medium}.")

    event_files = collect_inputs(args)
    if not event_files:
        raise RuntimeError("No inputs. Pass violation_events.json files / run dirs (positional, "
                           "--run <dir> or --events <file>).")

    alerts, counts, scanned = build_alerts(event_files, args)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sources = [str(p) for p in event_files]

    if not args.quiet:
        print(format_console(alerts, counts, scanned, sources, generated))

    if args.out_markdown:
        out = write_atomic(Path(args.out_markdown),
                           format_markdown(alerts, counts, scanned, sources, generated))
        if not args.quiet:
            print(f"\n-> {out}")

    # Opt-in automation gate: non-zero says "something needs attention", not "the tool failed".
    return 1 if (args.exit_code and alerts) else 0


if __name__ == "__main__":
    raise SystemExit(main())
