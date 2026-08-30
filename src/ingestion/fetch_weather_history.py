"""
fetch_weather_history.py
Fetches historical hourly weather data from the Open-Meteo Archive API
for Delhi, resamples it to 6-hour blocks, and saves it to the
weather_history table in aqi_history.db.
"""

import os
import sqlite3

import pandas as pd
import requests

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
DB_PATH = os.path.join(BASE_DIR, "data", "aqi_history.db")

# ── Delhi coordinates ───────────────────────────────────────────────────────
LATITUDE = 28.65
LONGITUDE = 77.30

# ── Open-Meteo config ──────────────────────────────────────────────────────
API_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = ["precipitation", "wind_speed_10m", "relative_humidity_2m"]


def get_date_range() -> tuple[str, str]:
    """Read min and max last_update from aqi_readings and return (start, end) as YYYY-MM-DD."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT last_update FROM aqi_readings", conn)
    conn.close()

    df["dt"] = pd.to_datetime(df["last_update"], format="%d-%m-%Y %H:%M:%S")
    start = df["dt"].min().strftime("%Y-%m-%d")
    end = df["dt"].max().strftime("%Y-%m-%d")
    print(f"Date range in database: {start} to {end}")
    return start, end


def fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch hourly weather data from Open-Meteo Archive API."""
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "Asia/Kolkata",
    }

    print(f"Fetching weather data from Open-Meteo for {start_date} to {end_date} ...")
    response = requests.get(API_URL, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    hourly = data["hourly"]
    df = pd.DataFrame({
        "time": pd.to_datetime(hourly["time"]),
        "precipitation": hourly["precipitation"],
        "wind_speed_10m": hourly["wind_speed_10m"],
        "relative_humidity_2m": hourly["relative_humidity_2m"],
    })
    print(f"Received {len(df)} hourly rows.")
    return df


def resample_6h(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly data into 6-hour blocks (00, 06, 12, 18)."""
    df = df.set_index("time")
    resampled = df.resample("6h").agg({
        "precipitation": "sum",
        "wind_speed_10m": "mean",
        "relative_humidity_2m": "mean",
    })
    resampled = resampled.dropna().reset_index()
    resampled = resampled.rename(columns={"time": "datetime"})
    numeric_cols = ["precipitation", "wind_speed_10m", "relative_humidity_2m"]
    resampled[numeric_cols] = resampled[numeric_cols].round(2)
    print(f"Resampled to {len(resampled)} six-hour blocks.")
    return resampled


def save_to_db(df: pd.DataFrame):
    """Save the resampled weather data to the weather_history table."""
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("weather_history", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Saved {len(df)} rows to the 'weather_history' table.")


if __name__ == "__main__":
    start, end = get_date_range()
    weather_df = fetch_weather(start, end)
    resampled_df = resample_6h(weather_df)

    print("\n-- First rows --")
    print(resampled_df.head(10).to_string(index=False))

    save_to_db(resampled_df)
