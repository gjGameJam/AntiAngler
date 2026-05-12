import os
import requests
import json
from datetime import datetime, timedelta

def fetch_dark_vessels():
    # In GitHub, we don't need .env files; we use env vars directly
    api_token = os.getenv("GFW_API_TOKEN")
    base_url = "https://gateway.globalfishingwatch.org/v3"
    
    headers = {"Authorization": f"Bearer {api_token}"}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3)
    
    params = {
        "start-date": start_date.strftime('%Y-%m-%d'),
        "end-date": end_date.strftime('%Y-%m-%d'),
        "bbox": "-18,10,-14,15",
        "format": "json"
    }

    print(f"Requesting data for {start_date.date()}...")
    response = requests.get(f"{base_url}/datasets/vessel-detections-s1:latest/data", 
                            headers=headers, params=params)
    
    if response.status_code == 200:
        data = response.json()
        os.makedirs("scripts/data", exist_ok=True)
        with open("scripts/data/report.json", "w") as f:
            json.dump(data, f)
        print("Success!")
    else:
        print(f"Failed: {response.status_code} - {response.text}")

if __name__ == "__main__":
    fetch_dark_vessels()