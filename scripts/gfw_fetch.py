import os
import requests
from pathlib import Path
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent

load_dotenv(SCRIPT_DIR / ".env")
load_dotenv(SCRIPT_DIR.parent / ".env")

token = os.getenv("GFW_API_TOKEN")

if not token:
    raise RuntimeError("Missing GFW_API_TOKEN")

token = token.strip()

url = "https://gateway.api.globalfishingwatch.org/v3/4wings/report"

params = {
    "datasets[0]": "public-global-fishing-effort:latest",
    "date-range": "2024-01-01,2024-03-01",
    "group-by": "VESSEL_ID",
    "spatial-resolution": "LOW",
    "temporal-resolution": "ENTIRE",
    "format": "JSON",
    "filters[0]": "geartype in ('trawlers')"
}

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}

payload = {
    "region": {
        "dataset": "public-eez-areas",
        "id": 8466
    }
}

response = requests.post(
    url,
    headers=headers,
    params=params,
    json=payload,
    timeout=120
)

print("FINAL URL:")
print(response.request.url)

print("\nHEADERS SENT:")
print(response.request.headers)

print("\nSTATUS:")
print(response.status_code)

print("\nBODY:")
print(response.text[:5000])

response.raise_for_status()

data = response.json()

print("\nSUCCESS")
print(data.keys())
