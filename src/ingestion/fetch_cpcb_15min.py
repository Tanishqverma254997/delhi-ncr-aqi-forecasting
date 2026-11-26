"""
fetch_cpcb_15min.py
Downloads the 15-minute air quality Parquet file from the india-cpcb-aqi
GitHub repository, filters for Delhi-NCR stations, resamples to 1-hour
intervals, and saves to the aqi_hourly table in aqi_history.db.
"""

import os
import sqlite3
import tempfile

import pandas as pd
import requests

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
DB_PATH = os.path.join(BASE_DIR, "data", "aqi_history.db")
DATA_DIR = os.path.join(BASE_DIR, "data")

# ── Download config ─────────────────────────────────────────────────────────
PARQUET_URL = (
    "https://github.com/Vonter/india-cpcb-aqi/releases/download/{year}/cpcb-air-quality-{year}.parquet"
)
YEARS = [2022, 2023, 2024, 2025]

# ── Delhi-NCR filter ────────────────────────────────────────────────────────
DELHI_STATES = ["Delhi"]

# ── Columns we care about ──────────────────────────────────────────────────
AQ_COLS = ["PM2.5 (µg/m³)", "PM10 (µg/m³)", "NO2 (µg/m³)", "CO (mg/m³)",
           "SO2 (µg/m³)", "Ozone (µg/m³)"]
WEATHER_COLS = ["AT (°C)", "RH (%)", "WS (m/s)", "WD (deg)",
                "RF (mm)", "BP (mmHg)", "SR (W/mt2)"]
META_COLS = ["Station ID", "State", "City", "Station Name", "Timestamp"]

KEEP_COLS = META_COLS + AQ_COLS + WEATHER_COLS

# Rename to cleaner column names
RENAME_MAP = {
    "Station ID": "station_id",
    "State": "state",
    "City": "city",
    "Station Name": "station",
    "Timestamp": "timestamp",
    "PM2.5 (µg/m³)": "pm25",
    "PM10 (µg/m³)": "pm10",
    "NO2 (µg/m³)": "no2",
    "CO (mg/m³)": "co",
    "SO2 (µg/m³)": "so2",
    "Ozone (µg/m³)": "ozone",
    "AT (°C)": "temperature",
    "RH (%)": "humidity",
    "WS (m/s)": "wind_speed",
    "WD (deg)": "wind_direction",
    "RF (mm)": "rainfall",
    "BP (mmHg)": "pressure",
    "SR (W/mt2)": "solar_radiation",
}


def download_parquet(year: int) -> pd.DataFrame:
    """Download the Parquet file for a given year and return as DataFrame."""
    url = PARQUET_URL.format(year=year)
    local_path = os.path.join(DATA_DIR, f"cpcb-air-quality-{year}.parquet")

    if os.path.exists(local_path):
        print(f"  Using cached {local_path}")
    else:
        print(f"  Downloading {url} ...")
        resp = requests.get(url, timeout=600, stream=True)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"  Saved {local_path} ({size_mb:.1f} MB)")

    df = pd.read_parquet(local_path)
    print(f"  Raw rows for {year}: {len(df)}")
    return df


def filter_delhi(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only Delhi-NCR stations."""
    available_cols = [c for c in KEEP_COLS if c in df.columns]
    missing = set(KEEP_COLS) - set(available_cols)
    if missing:
        print(f"  Warning: missing columns {missing}")

    df = df[available_cols].copy()
    df = df[df["State"].isin(DELHI_STATES)].reset_index(drop=True)
    print(f"  Delhi rows: {len(df)}")
    return df


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 15-min data to 1-hour intervals per station."""
    df = df.rename(columns=RENAME_MAP)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # Convert to IST
    df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df = df.sort_values(["station", "timestamp"]).reset_index(drop=True)

    numeric_cols = [c for c in df.columns
                    if c not in ["station_id", "state", "city", "station", "timestamp"]]

    # Group by station + hour and aggregate
    df["_hour"] = df["timestamp"].dt.floor("h")

    agg_dict = {col: "mean" for col in numeric_cols if col in df.columns}
    agg_dict["station_id"] = "first"
    agg_dict["state"] = "first"
    agg_dict["city"] = "first"

    hourly = df.groupby(["station", "_hour"]).agg(agg_dict).reset_index()
    hourly = hourly.rename(columns={"_hour": "timestamp"})

    # Round numeric columns
    for col in numeric_cols:
        if col in hourly.columns:
            hourly[col] = hourly[col].round(2)

    print(f"  Resampled to {len(hourly)} hourly rows.")
    return hourly


def save_to_db(df: pd.DataFrame):
    """Save the hourly data to the aqi_hourly table."""
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("aqi_hourly", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Saved {len(df)} rows to 'aqi_hourly' table.")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    all_dfs = []
    for year in YEARS:
        print(f"\n--- Processing {year} ---")
        raw = download_parquet(year)
        delhi = filter_delhi(raw)
        hourly = resample_hourly(delhi)
        all_dfs.append(hourly)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(["station", "timestamp"]).reset_index(drop=True)
    print(f"\nTotal: {len(combined)} hourly rows across {combined['station'].nunique()} stations.")

    print("\n-- Sample rows --")
    print(combined.head(5).to_string(index=False))
    print(f"\nColumns: {list(combined.columns)}")

    save_to_db(combined)
