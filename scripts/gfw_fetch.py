import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

# python-dotenv is soft-imported so this module can be imported (and its pure
# helpers unit-tested) without the dep installed; `requests` is lazy-imported
# inside main() for the same reason (keeps `import gfw_fetch` stdlib-only).
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is optional at import time
    def load_dotenv(*_args, **_kwargs):
        return False

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "gfw"

GFW_REPORT_URL = "https://gateway.api.globalfishingwatch.org/v3/4wings/report"


def load_env():
    """Load .env from scripts/ then the repo root (repo root wins on conflict)."""
    load_dotenv(SCRIPT_DIR / ".env")
    load_dotenv(REPO_ROOT / ".env")


def env_or(name, default):
    val = os.getenv(name)
    return val if val not in (None, "") else default


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Fetch a GFW 4WINGS fishing-effort report.")
    p.add_argument("--region-id", type=int,
                   default=int(env_or("GFW_REGION_ID", 8466)))
    p.add_argument("--region-dataset",
                   default=env_or("GFW_REGION_DATASET", "public-eez-areas"))
    p.add_argument("--start", default=env_or("GFW_START", None),
                   help="YYYY-MM-DD. Defaults to yesterday UTC.")
    p.add_argument("--end", default=env_or("GFW_END", None),
                   help="YYYY-MM-DD. Defaults to yesterday UTC.")
    p.add_argument("--gear", default=env_or("GFW_GEAR", "trawlers"),
                   help="Comma-separated list, e.g. 'trawlers,longliners'.")
    p.add_argument("--group-by", default=env_or("GFW_GROUP_BY", "VESSEL_ID"))
    p.add_argument("--spatial-resolution",
                   default=env_or("GFW_SPATIAL_RESOLUTION", "LOW"))
    p.add_argument("--temporal-resolution",
                   default=env_or("GFW_TEMPORAL_RESOLUTION", "ENTIRE"))
    p.add_argument("--format", default=env_or("GFW_FORMAT", "JSON"))
    p.add_argument("--dataset",
                   default=env_or("GFW_DATASET", "public-global-fishing-effort:latest"))
    p.add_argument("--timeout", type=float, default=float(env_or("GFW_TIMEOUT", 120)),
                   help="HTTP timeout in seconds (default 120).")
    return p.parse_args(argv)


def validate_date(label, value):
    """Return value if it is a YYYY-MM-DD calendar date, else raise ValueError.

    None passes through (the caller substitutes the yesterday-UTC default)."""
    if value is None:
        return None
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except (ValueError, TypeError):
        raise ValueError(f"--{label} must be YYYY-MM-DD, got {value!r}")
    return value


def resolve_date_range(args):
    yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    start = validate_date("start", args.start) or yesterday
    end = validate_date("end", args.end) or yesterday
    if start > end:
        raise ValueError(f"--start ({start}) must not be after --end ({end})")
    return start, end


def gear_filter(gear):
    items = [g.strip() for g in gear.split(",") if g.strip()]
    if not items:
        raise ValueError("--gear resolved to an empty list; pass at least one gear type")
    quoted = ", ".join(f"'{g}'" for g in items)
    return f"geartype in ({quoted})"


def build_request(args, start, end):
    """Assemble (params, payload) for the 4WINGS report POST. Pure, testable."""
    params = {
        "datasets[0]": args.dataset,
        "date-range": f"{start},{end}",
        "group-by": args.group_by,
        "spatial-resolution": args.spatial_resolution,
        "temporal-resolution": args.temporal_resolution,
        "format": args.format,
        "filters[0]": gear_filter(args.gear),
    }
    payload = {
        "region": {
            "dataset": args.region_dataset,
            "id": args.region_id,
        }
    }
    return params, payload


def write_json_atomic(path, data):
    """Write JSON to path atomically: serialize to a sibling .tmp, fsync, then
    os-level rename over the target. The tmp is cleaned up on any failure
    (including Ctrl-C), so the output dir never holds a partial file."""
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


def main(argv=None):
    import requests  # lazy: keeps `import gfw_fetch` free of the network dep

    load_env()
    args = parse_args(argv)
    start, end = resolve_date_range(args)

    token = os.getenv("GFW_API_TOKEN")
    if not token or not token.strip():
        raise SystemExit(
            "Missing GFW_API_TOKEN. Put it in a .env file at the repo root or in "
            "scripts/, or export it (see .env.example)."
        )
    token = token.strip()

    params, payload = build_request(args, start, end)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    print("REQUEST:")
    print(f"  region={args.region_dataset}/{args.region_id}")
    print(f"  date-range={start},{end}")
    print(f"  gear={args.gear}")
    print(f"  dataset={args.dataset}")

    response = requests.post(
        GFW_REPORT_URL, headers=headers, params=params, json=payload, timeout=args.timeout
    )

    print(f"\nFINAL URL: {response.request.url}")
    print(f"STATUS:    {response.status_code}")

    # Raise before writing anything, so a non-2xx never leaves a file behind.
    response.raise_for_status()
    data = response.json()

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_path = write_json_atomic(OUTPUT_DIR / f"gfw_report_{timestamp}.json", data)

    print(f"\nWrote {out_path.stat().st_size} bytes -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
