import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "gfw"

load_dotenv(SCRIPT_DIR / ".env")
load_dotenv(REPO_ROOT / ".env")


def env_or(name, default):
    val = os.getenv(name)
    return val if val not in (None, "") else default


def parse_args():
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
    return p.parse_args()


def resolve_date_range(args):
    yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    return args.start or yesterday, args.end or yesterday


def gear_filter(gear):
    items = [g.strip() for g in gear.split(",") if g.strip()]
    quoted = ", ".join(f"'{g}'" for g in items)
    return f"geartype in ({quoted})"


args = parse_args()
start, end = resolve_date_range(args)

token = os.getenv("GFW_API_TOKEN")
if not token:
    raise RuntimeError("Missing GFW_API_TOKEN")
token = token.strip()

url = "https://gateway.api.globalfishingwatch.org/v3/4wings/report"

params = {
    "datasets[0]": args.dataset,
    "date-range": f"{start},{end}",
    "group-by": args.group_by,
    "spatial-resolution": args.spatial_resolution,
    "temporal-resolution": args.temporal_resolution,
    "format": args.format,
    "filters[0]": gear_filter(args.gear),
}

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}

payload = {
    "region": {
        "dataset": args.region_dataset,
        "id": args.region_id,
    }
}

print("REQUEST:")
print(f"  region={args.region_dataset}/{args.region_id}")
print(f"  date-range={start},{end}")
print(f"  gear={args.gear}")
print(f"  dataset={args.dataset}")

response = requests.post(url, headers=headers, params=params, json=payload, timeout=120)

print(f"\nFINAL URL: {response.request.url}")
print(f"STATUS:    {response.status_code}")

response.raise_for_status()
data = response.json()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
out_path = OUTPUT_DIR / f"gfw_report_{timestamp}.json"
tmp_path = out_path.with_suffix(".json.tmp")
tmp_path.write_text(json.dumps(data, indent=2))
tmp_path.replace(out_path)

print(f"\nWrote {out_path.stat().st_size} bytes -> {out_path}")
