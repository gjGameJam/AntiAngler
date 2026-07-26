import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

# The objective of this script (P3 batch driver, PHASE3_PLAN Workstream 2A):
# - Sequence the Phase-3 stages for one run dir / MPA end to end, reusing each stage's main(argv)
#   (every P3 script is argv-guarded + import-safe). It is a THIN ORCHESTRATOR: no fusion or scoring
#   logic of its own — just ordering, argv plumbing, and a per-stage log + final summary.
#
# Pipeline (flags pick the stages):
#   [--detect] detect_boats  ->  fuse_violations --ais  ->  [--gfw] gfw_vessels --from-events +
#   re-fuse --vessels  ->  [--report] report_violations
#
# The fuse + report stages are pure stdlib and run anywhere (and are exercised offline here). The
# front-end stages are do-from-home: --detect needs the ML stack (torch/ultralytics/rasterio) and is
# lazy-imported only when used; a real --gfw run needs GFW_API_TOKEN; producing the run dir itself
# (sat_fetch) and capturing live AIS (aisstream_fetch) happen upstream. See docs/PHASE3_PLAN.md.
#
# House rule: a driver, not a data stream. It imports the stages; it never reimplements them.

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def env_or(name, default):
    val = os.getenv(name)
    return val if val not in (None, "") else default


def truthy(val):
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Batch-drive the Phase-3 stages (detect -> fuse -> GFW-enrich -> report) for a run "
                    "dir (PHASE3_PLAN Workstream 2A). Thin orchestrator over each stage's main(argv).")
    p.add_argument("--run", type=Path, default=env_or("SCAN_RUN", None),
                   help="A detect_boats run dir (contains detections.json; or use --detect to make it).")
    p.add_argument("--ais", type=Path, default=env_or("SCAN_AIS", None),
                   help="Normalized AIS positions file for the fusion stage (required to fuse).")
    p.add_argument("--vessels", type=Path, default=env_or("SCAN_VESSELS", None),
                   help="Pre-existing GFW vessel-records file to enrich with (skips the --gfw fetch).")
    p.add_argument("--gfw", action="store_true", default=truthy(env_or("SCAN_GFW", "")),
                   help="After the first fuse, run gfw_vessels on the matched MMSIs and re-fuse with "
                        "--vessels (needs GFW_API_TOKEN). Ignored if --vessels is already given.")
    p.add_argument("--gfw-start", default=env_or("SCAN_GFW_START", None),
                   help="YYYY-MM-DD fishing-window start passed to gfw_vessels (default: its own).")
    p.add_argument("--gfw-end", default=env_or("SCAN_GFW_END", None),
                   help="YYYY-MM-DD fishing-window end passed to gfw_vessels.")
    p.add_argument("--wdpa-id", default=env_or("SCAN_WDPA_ID", None),
                   help="Pass through to gfw_vessels --wdpa-id so fishing_probability is MPA-specific.")
    p.add_argument("--dt-minutes", default=env_or("SCAN_DT_MINUTES", None),
                   help="Pass through to fuse_violations --dt-minutes.")
    p.add_argument("--detect", action="store_true", default=truthy(env_or("SCAN_DETECT", "")),
                   help="Run detect_boats first (needs the ML stack; lazy-imported). Do-from-home.")
    p.add_argument("--weights", default=env_or("SCAN_WEIGHTS", None),
                   help="Weights for --detect (else detect_boats' default).")
    p.add_argument("--no-report", action="store_true", default=truthy(env_or("SCAN_NO_REPORT", "")),
                   help="Skip the ranked CSV/HTML report stage (on by default).")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- stages (thin wrappers)

def stage_detect(run, weights):
    """Run detect_boats over the run dir. Lazy-imported (heavy ML deps); do-from-home."""
    import detect_boats  # noqa: PLC0415 - heavy (torch/rasterio); only when --detect
    argv = ["--run", str(run)]
    if weights:
        argv += ["--weights", str(weights)]
    detect_boats.main(argv) if hasattr(detect_boats, "main") else _run_module_main(detect_boats, argv)


def stage_fuse(run, ais, vessels=None, dt_minutes=None):
    import fuse_violations
    argv = ["--run", str(run), "--ais", str(ais)]
    if vessels:
        argv += ["--vessels", str(vessels)]
    if dt_minutes is not None:
        argv += ["--dt-minutes", str(dt_minutes)]
    fuse_violations.main(argv)


def stage_gfw_vessels(events_json, out_path, start=None, end=None, wdpa_id=None):
    import gfw_vessels
    argv = ["--from-events", str(events_json), "--out", str(out_path)]
    if start:
        argv += ["--start", str(start)]
    if end:
        argv += ["--end", str(end)]
    if wdpa_id:
        argv += ["--wdpa-id", str(wdpa_id)]
    gfw_vessels.main(argv)


def stage_report(run):
    import report_violations
    report_violations.main(["--run", str(run)])


def _run_module_main(mod, argv):
    raise RuntimeError(f"{mod.__name__} has no main(argv); cannot drive it.")


# --------------------------------------------------------------------------- orchestration

def run_pipeline(args):
    """Sequence the stages per the flags. Returns a summary dict (stages run + output paths).
    Pure control-flow over the stage wrappers so it is unit-testable by monkeypatching them."""
    if args.run is None:
        raise RuntimeError("Provide a run dir: --run <detect_boats run dir>")
    run = Path(args.run).resolve()
    summary = {"run": str(run), "stages": [], "outputs": {}}

    if args.detect:
        print("[scan] detect_boats ...")
        stage_detect(run, args.weights)
        summary["stages"].append("detect")

    if args.ais is None:
        raise RuntimeError("Provide --ais <positions.json> to fuse (or only --detect to stop at stage 2).")

    print("[scan] fuse_violations ...")
    stage_fuse(run, args.ais, args.vessels, args.dt_minutes)
    summary["stages"].append("fuse")
    events_json = run / "violation_events.json"
    summary["outputs"]["violation_events"] = str(events_json)

    # GFW enrich: build a vessels file from the matched MMSIs, then re-fuse. Skipped if --vessels
    # was supplied (already enriched) or --gfw not set.
    if args.gfw and not args.vessels:
        vessels_out = run / "gfw_vessels.json"
        print("[scan] gfw_vessels (--from-events) ...")
        stage_gfw_vessels(events_json, vessels_out, args.gfw_start, args.gfw_end, args.wdpa_id)
        summary["stages"].append("gfw_vessels")
        summary["outputs"]["gfw_vessels"] = str(vessels_out)
        print("[scan] fuse_violations (re-fuse --vessels) ...")
        stage_fuse(run, args.ais, vessels_out, args.dt_minutes)
        summary["stages"].append("refuse")

    if not args.no_report:
        print("[scan] report_violations ...")
        stage_report(run)
        summary["stages"].append("report")

    return summary


def main(argv=None):
    args = parse_args(argv)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"SCAN VIOLATIONS ({started}):")
    summary = run_pipeline(args)
    print(f"\n[scan] done - stages: {' -> '.join(summary['stages'])}")
    for name, path in summary["outputs"].items():
        print(f"       {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
