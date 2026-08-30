import os
import sys
import time
import requests
from dotenv import load_dotenv

# Add the project root to sys.path so we can import sibling packages
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.database.db_operations import init_db, insert_records

# 1. Load environment variables
load_dotenv()
API_KEY = os.getenv("CPCB_API_KEY")
RESOURCE_ID = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"

def fetch_delhi_aqi(max_retries=3):
    """Fetches real-time AQI data from data.gov.in for Delhi stations."""
    if not API_KEY:
        raise ValueError("Missing CPCB_API_KEY in .env file.")

    url = f"https://api.data.gov.in/resource/{RESOURCE_ID}"
    params = {
        "api-key": API_KEY,
        "format": "json",
        "filters[state]": "Delhi",
        "limit": 10
    }

    # 2. Try fetching data multiple times if the server fails
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for attempt in range(max_retries):
        print(f"Attempt {attempt + 1} of {max_retries}...")
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e}")
            if attempt < max_retries - 1:
                print("Waiting 5 seconds before retrying...")
                time.sleep(5)
                continue
            else:
                print("Max retries reached. Moving on.")
                return []

        # 3. Check status and parse JSON
        if response.status_code == 200:
            data = response.json()
            records = data.get("records", [])
            return records
        else:
            print(f"Request failed with status {response.status_code}")
            if attempt < max_retries - 1:
                print("Waiting 5 seconds before retrying...")
                time.sleep(5)
            else:
                print("Max retries reached. Moving on.")
                return []

if __name__ == "__main__":
    # Step A – Make sure the database and table exist
    print("=" * 50)
    print("STEP 1: Initialising the database...")
    print("=" * 50)
    init_db()

    # Step B – Fetch live AQI data from the CPCB / data.gov.in API
    print()
    print("=" * 50)
    print("STEP 2: Fetching AQI data from the API...")
    print("=" * 50)
    records = fetch_delhi_aqi()
    print(f"Fetched {len(records)} records.")

    # Step C – Insert the fetched records into SQLite
    if records:
        print()
        print("=" * 50)
        print("STEP 3: Inserting records into the database...")
        print("=" * 50)
        insert_records(records)
        print()
        print("Pipeline complete! Data saved to data/aqi_history.db")
    else:
        print("No records fetched – nothing to insert.")