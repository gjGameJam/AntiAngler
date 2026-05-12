import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def fetch_dark_vessels():
    # 1. Fetch your token securely from GitHub Action Secrets
    # Ensure you have mapped secrets.GFW_API_TOKEN to this environment variable in your workflow.yml
    api_token = os.environ.get("GFW_API_TOKEN") 
    
    if not api_token:
        raise ValueError("API Token missing! Please set the GFW_API_TOKEN environment variable.")

    # 2. Set the required headers per the API Documentation
    headers = {
        "Authorization": f"Bearer {api_token}",
        # Override the default python-requests user agent
        "User-Agent": "AntiAngler-Pipeline/1.0 (Contact: your-email@example.com)", 
        "Accept": "application/json"
    }

    params = {
        "start-date": "2026-05-09",
        "end-date": "2026-05-12",
        "bbox": "-18,10,-14,15",
        "format": "json"
    }
    
    base_url = "https://gateway.globalfishingwatch.org/v3"
    endpoint = f"{base_url}/datasets/vessel-detections-s1:latest/data"

    # 3. Use a Session with a retry strategy for pipeline stability
    session = requests.Session()
    session.headers.update(headers)
    
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))

    # 4. Make the request
    try:
        response = session.get(endpoint, params=params, timeout=30)
        response.raise_for_status() # Raises an exception for 4xx and 5xx status codes
        
        data = response.json()
        print(f"Successfully fetched {len(data.get('entries', []))} records.")
        return data

    except requests.exceptions.SSLError as e:
        print(f"SSL handshake dropped by WAF/Gateway: {e}")
        raise
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
        raise

if __name__ == "__main__":
    fetch_dark_vessels()