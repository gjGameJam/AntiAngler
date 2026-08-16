import argparse
import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv   # optional; only used to pick up DIGEST_* knobs from .env
except ImportError:                  # keep the module importable (and offline-testable) without it
    def load_dotenv(*_args, **_kwargs):
        return False

# The objective of this script (the sweep's "how many ships, and where" view):
# - Read one gfw_mpa_sweep_*.json (the nightly per-(MPA, vessel) intrusion records)
# - Count DISTINCT vessels (the events list double-counts a vessel seen in two MPAs — including
#   the degenerate case of two overlapping WDPA records over the same water) and group them by
#   MPA, worst-first, as a plain-text digest
# - OPTIONALLY (--vessels, a gfw_vessels.py --insights output) flag each vessel with the GFW
#   Insights signals: IUU listing and AIS-off gap events (intentional AIS disabling)
#
# WHY this exists separately from the sweep's own end-of-run summary: that prints coverage
# (what was reached); this answers the operator question — how many named vessels, in which
# protected areas, and which of them look shady beyond mere presence. Read-only, same house
# pattern as report_violations.py / alert_violations.py: a VIEW over an upstream file, no
# fetching, no scoring of its own.
#
# HONESTY (the "spoofing" question): GFW's public Insights API does NOT expose GPS-location
# spoofing. The closest public signals are the AIS-off gap events (a vessel deliberately going
# dark) and the IUU list. This digest labels them exactly that — "AIS-OFF" / "IUU-LISTED" —
# and never claims spoofing. Without a --vessels file those flags are UNAVAILABLE, and the
# header says so rather than implying every vessel is clean.
#
# Pure stdlib, offline-tested (scripts/tests/test_sweep_digest.py). The small helpers are
# deliberately duplicated rather than imported from the other views (docs/AUDIT.md #7 — the
# views ask different questions of different files; coupling them breaks one when the other
# moves).

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SWEEP_DIR = REPO_ROOT / "data" / "raw" / "gfw_mpa"

load_dotenv(SCRIPT_DIR / ".env")
load_dotenv(REPO_ROOT / ".env")


def env_or(name, default):
    val = os.getenv(name)
    return val if val not in (None, "") else default


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Plain-text digest of a gfw_mpa_sweep run: distinct vessels, grouped by MPA, "
                    "with optional GFW-Insights IUU / AIS-off flags per vessel.")
    p.add_argument("sweep", nargs="?", type=Path, default=None,
                   help="A gfw_mpa_sweep_*.json, or a directory of them (latest taken). "
                        "Default: the latest file in data/raw/gfw_mpa/.")
    p.add_argument("--vessels", type=Path, default=env_or("DIGEST_VESSELS", None),
                   help="A gfw_vessels.py vessel-records file (run it with --from-sweep <sweep> "
                        "--insights) to flag IUU-listed vessels and AIS-off gap events. Without "
                        "it those flags are reported as unavailable, not as clean.")
    p.add_argument("--min-hours", type=float, default=float(env_or("DIGEST_MIN_HOURS", 0.0)),
                   help="Drop events with less apparent fishing than this (hours). The header "
                        "reports how many were dropped, so a filtered digest is legible as one.")
    p.add_argument("--flagged-only", action="store_true",
                   help="Only vessels with an IUU or AIS-off flag (requires --vessels).")
    p.add_argument("--out", type=Path, default=env_or("DIGEST_OUT", None),
                   help="Also write the console text to this path.")
    p.add_argument("--out-markdown", type=Path, default=env_or("DIGEST_OUT_MD", None),
                   help="Also write the digest as markdown to this path.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress the console digest (use with --out/--out-markdown).")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- input

def resolve_sweep_path(arg):
    """The sweep file to digest: an explicit file, the latest in an explicit dir, or the
    latest in data/raw/gfw_mpa/. Filenames carry a UTC timestamp, so max(name) is newest."""
    base = Path(arg) if arg else SWEEP_DIR
    if base.is_file():
        return base
    if base.is_dir():
        candidates = sorted(base.glob("gfw_mpa_sweep_*.json"))
        if candidates:
            return candidates[-1]
        raise RuntimeError(f"No gfw_mpa_sweep_*.json in {base}")
    raise RuntimeError(f"Sweep path not found: {base}")


def load_vessel_flagmap(path):
    """{mmsi: vessel_record} from a gfw_vessels.py vessel-records file (or a bare list)."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = doc.get("vessels", []) if isinstance(doc, dict) else (doc or [])
    out = {}
    for r in rows:
        mmsi = r.get("mmsi")
        if mmsi not in (None, ""):
            out[str(mmsi)] = r
    return out


# --------------------------------------------------------------------------- pure: flags

def flags_for(mmsi, flagmap):
    """GFW-Insights flags for one vessel, honestly labeled.
    Returns None when the vessel was never looked up (unknown != clean), else a list —
    empty when it WAS looked up and nothing flagged."""
    if flagmap is None or mmsi in (None, ""):
        return None
    rec = flagmap.get(str(mmsi))
    if rec is None:
        return None
    flags = []
    if rec.get("iuu_listed"):
        flags.append("IUU-LISTED")
    off = rec.get("ais_off_events")
    if off:
        flags.append(f"AIS-OFF x{int(off)}")
    no_take = rec.get("fishing_in_no_take_mpa")
    if no_take:
        flags.append(f"NO-TAKE-FISHING x{int(no_take)}")
    return flags


# --------------------------------------------------------------------------- pure: summarize

def vessel_key(event, index):
    """Identity key for de-duplicating a vessel across MPAs: MMSI, else GFW vessel_id, else
    name+flag, else the event's own index (counts once, never merges strangers)."""
    for field in ("mmsi", "vessel_id"):
        val = event.get(field)
        if val not in (None, ""):
            return f"{field}:{val}"
    name, flag = event.get("vessel_name"), event.get("flag")
    if name not in (None, ""):
        return f"name:{name}|{flag}"
    return f"row:{index}"


def summarize(doc, min_hours=0.0, flagmap=None, flagged_only=False):
    """Group a sweep doc's events by MPA and de-duplicate vessels across MPAs.

    Returns {groups, distinct, event_count, dropped_min_hours, dropped_unflagged,
    multi, total_hours}; each group is {wdpa_pid, mpa_name, iso3, iucn_cat, total_hours,
    events} worst-first (most vessels, then most hours), events hours-descending, each
    event annotated with "flags" (see flags_for). `multi` lists vessels seen in >1 MPA,
    with identical entry/exit/hours everywhere marked as overlapping WDPA records rather
    than genuine multi-reserve activity."""
    events = doc.get("events", []) or []
    kept, dropped_min, dropped_unflagged = [], 0, 0
    for i, e in enumerate(events):
        hours = e.get("fishing_hours") or 0.0
        if hours < min_hours:
            dropped_min += 1
            continue
        flags = flags_for(e.get("mmsi"), flagmap)
        if flagged_only and not flags:
            dropped_unflagged += 1
            continue
        kept.append((i, dict(e, flags=flags)))

    groups = {}
    for i, e in kept:
        pid = str(e.get("wdpa_pid"))
        g = groups.setdefault(pid, {"wdpa_pid": pid, "mpa_name": e.get("mpa_name"),
                                    "iso3": e.get("iso3"), "iucn_cat": e.get("iucn_cat"),
                                    "events": []})
        g["events"].append(e)
    for g in groups.values():
        g["events"].sort(key=lambda e: -(e.get("fishing_hours") or 0.0))
        g["total_hours"] = sum(e.get("fishing_hours") or 0.0 for e in g["events"])
    ordered = sorted(groups.values(),
                     key=lambda g: (-len(g["events"]), -g["total_hours"], g["mpa_name"] or ""))

    by_vessel = {}
    for i, e in kept:
        by_vessel.setdefault(vessel_key(e, i), []).append(e)
    multi = []
    for key, occurrences in by_vessel.items():
        if len(occurrences) < 2:
            continue
        stamps = {(e.get("entry_timestamp"), e.get("exit_timestamp"),
                   round(e.get("fishing_hours") or 0.0, 6)) for e in occurrences}
        first = occurrences[0]
        multi.append({"mmsi": first.get("mmsi"), "vessel_name": first.get("vessel_name"),
                      "mpas": [e.get("mpa_name") for e in occurrences],
                      "overlapping_records": len(stamps) == 1})

    return {"groups": ordered,
            "distinct": len(by_vessel),
            "event_count": len(kept),
            "dropped_min_hours": dropped_min,
            "dropped_unflagged": dropped_unflagged,
            "multi": sorted(multi, key=lambda m: str(m.get("vessel_name"))),
            "total_hours": sum(g["total_hours"] for g in ordered)}


# --------------------------------------------------------------------------- formatting

NO_FLAGS_NOTE = ("unavailable - pass --vessels <gfw_vessels_*.json> (from gfw_vessels.py "
                 "--from-sweep <sweep> --insights). AIS-off gaps / IUU listing are GFW's "
                 "public 'going dark' signals; the public API has no GPS-spoofing field.")


def _flag_counts(summary):
    """(looked_up, iuu, ais_off) across the kept events, counting each vessel line once."""
    looked_up = iuu = ais_off = 0
    for g in summary["groups"]:
        for e in g["events"]:
            flags = e.get("flags")
            if flags is None:
                continue
            looked_up += 1
            if any(f.startswith("IUU") for f in flags):
                iuu += 1
            if any(f.startswith("AIS-OFF") for f in flags):
                ais_off += 1
    return looked_up, iuu, ais_off


def _flags_line(summary, vessels_path):
    if vessels_path is None:
        return NO_FLAGS_NOTE
    looked_up, iuu, ais_off = _flag_counts(summary)
    return (f"{iuu} IUU-listed, {ais_off} with AIS-off gap events "
            f"({looked_up} of {summary['event_count']} event vessel(s) looked up in {vessels_path})")


def _date(ts):
    return str(ts)[:10] if ts else "?"


def _vessel_line(e):
    name = e.get("vessel_name") or "(unnamed)"
    mmsi = e.get("mmsi") or "-"
    hours = e.get("fishing_hours") or 0.0
    lat, lon = e.get("lat"), e.get("lon")
    where = f"{lat:.1f},{lon:.1f}" if lat is not None and lon is not None else "-"
    parts = [f"{hours:7.1f} h  {name:<22s} MMSI {mmsi:<10s} {e.get('flag') or '-':<4s}"
             f" {e.get('gear_type') or '-':<18s} {_date(e.get('entry_timestamp'))}"
             f"..{_date(e.get('exit_timestamp'))}  {where}"]
    if e.get("flags"):
        parts.append("  ** " + ", ".join(e["flags"]))
    return "".join(parts)


def _header_lines(doc, summary, args, sweep_path):
    cov = doc.get("coverage") or {}
    win = doc.get("window") or {}
    errors = doc.get("errors") or []
    lines = [f"GFW MPA SWEEP DIGEST  {doc.get('generated_utc', '?')}  ({sweep_path})",
             f"window    {win.get('start', '?')} -> {win.get('end', '?')}"
             + (" (end exclusive)" if win.get("end_exclusive") else ""),
             f"coverage  {doc.get('mpas_queried', '?')} of {cov.get('index_total', '?')} MPAs "
             f"queried ({cov.get('priority_count', '?')} priority + {cov.get('tail_slice', '?')} "
             f"tail); {len(errors)} error(s); budget exhausted: "
             f"{'YES' if cov.get('budget_exhausted') else 'no'}",
             f"vessels   {summary['distinct']} distinct vessel(s) in {len(summary['groups'])} "
             f"MPA(s) - {summary['event_count']} (MPA, vessel) event(s), "
             f"{summary['total_hours']:.1f} h apparent fishing",
             f"flags     {_flags_line(summary, args.vessels)}"]
    if summary["dropped_min_hours"]:
        lines.append(f"filtered  {summary['dropped_min_hours']} event(s) below "
                     f"--min-hours {args.min_hours:g} dropped")
    if summary["dropped_unflagged"]:
        lines.append(f"filtered  {summary['dropped_unflagged']} unflagged event(s) dropped "
                     f"(--flagged-only)")
    return lines


def format_console(doc, summary, args, sweep_path):
    lines = _header_lines(doc, summary, args, sweep_path)
    for g in summary["groups"]:
        lines.append("")
        lines.append(f"{g['mpa_name']}  [{g['iso3']} {g['iucn_cat']}  wdpa {g['wdpa_pid']}]  - "
                     f"{len(g['events'])} vessel(s), {g['total_hours']:.1f} h")
        for e in g["events"]:
            lines.append("  " + _vessel_line(e))
    if summary["multi"]:
        lines.append("")
        lines.append("vessels in more than one MPA:")
        for m in summary["multi"]:
            note = (" (identical hours - overlapping WDPA records, count once)"
                    if m["overlapping_records"] else "")
            lines.append(f"  {m.get('vessel_name') or '(unnamed)'}  MMSI {m.get('mmsi') or '-'}: "
                         + ", ".join(str(x) for x in m["mpas"]) + note)
    errors = doc.get("errors") or []
    if errors:
        lines.append("")
        lines.append("not swept this run (errored; priority-tier MPAs retry next run):")
        for err in errors:
            lines.append(f"  HTTP {err.get('status', '?')}  {err.get('mpa_name', '?')} "
                         f"(wdpa {err.get('wdpa_pid', '?')})")
    return "\n".join(lines)


def format_markdown(doc, summary, args, sweep_path):
    lines = [f"# GFW MPA sweep digest - {doc.get('generated_utc', '?')}", ""]
    lines += [line.rstrip() for line in _header_lines(doc, summary, args, sweep_path)[1:]]
    lines.append("")
    for g in summary["groups"]:
        lines.append(f"## {g['mpa_name']} - {g['iso3']} {g['iucn_cat']} "
                     f"({len(g['events'])} vessel(s), {g['total_hours']:.1f} h)")
        lines.append("")
        lines.append("| Hours | Vessel | MMSI | Flag | Gear | In MPA | Position (lat,lon) | Flags |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for e in g["events"]:
            lat, lon = e.get("lat"), e.get("lon")
            where = f"{lat:.1f},{lon:.1f}" if lat is not None and lon is not None else "-"
            flags = e.get("flags")
            flags_s = ", ".join(flags) if flags else ("-" if flags is not None else "n/a")
            lines.append(f"| {e.get('fishing_hours') or 0.0:.1f} | {e.get('vessel_name') or '-'} "
                         f"| {e.get('mmsi') or '-'} | {e.get('flag') or '-'} "
                         f"| {e.get('gear_type') or '-'} "
                         f"| {_date(e.get('entry_timestamp'))}..{_date(e.get('exit_timestamp'))} "
                         f"| {where} | {flags_s} |")
        lines.append("")
    if summary["multi"]:
        lines.append("**Vessels in more than one MPA:** " + "; ".join(
            f"{m.get('vessel_name') or '(unnamed)'} ({', '.join(str(x) for x in m['mpas'])}"
            + (" - overlapping records" if m["overlapping_records"] else "") + ")"
            for m in summary["multi"]))
        lines.append("")
    errors = doc.get("errors") or []
    if errors:
        lines.append(f"**Not swept this run ({len(errors)} errored):** " + "; ".join(
            f"{err.get('mpa_name', '?')} (HTTP {err.get('status', '?')})" for err in errors))
        lines.append("")
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
    if args.flagged_only and not args.vessels:
        raise RuntimeError("--flagged-only needs --vessels: the IUU/AIS-off flags come from a "
                           "gfw_vessels.py --insights file, and filtering on flags we never "
                           "fetched would silently show nothing.")

    sweep_path = resolve_sweep_path(args.sweep)
    doc = json.loads(sweep_path.read_text(encoding="utf-8"))
    flagmap = load_vessel_flagmap(args.vessels) if args.vessels else None
    summary = summarize(doc, min_hours=args.min_hours, flagmap=flagmap,
                        flagged_only=args.flagged_only)

    text = format_console(doc, summary, args, sweep_path)
    if not args.quiet:
        print(text)
    if args.out:
        out = write_atomic(Path(args.out), text + "\n")
        if not args.quiet:
            print(f"\n-> {out}")
    if args.out_markdown:
        out = write_atomic(Path(args.out_markdown), format_markdown(doc, summary, args, sweep_path))
        if not args.quiet:
            print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
