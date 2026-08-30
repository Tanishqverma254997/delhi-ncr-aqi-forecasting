"""predict_live.py

Fetches the latest engineered features from SQLite, queries the Open-Meteo
NWP Forecast API for the next 24 hours of forecasted atmospheric data
(wind, temperature, ventilation), and evaluates the 24h safety alert model.
"""

import os
import sys
import sqlite3
import joblib
import pandas as pd
import numpy as np

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ingestion.fetch_open_meteo_forecast import (
    fetch_24h_weather_forecast,
    get_forecast_at_horizon,
)

DB_PATH = os.path.join(BASE_DIR, "data", "aqi_history.db")
ALERT_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_poor_alert_model.joblib")
REG_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_aqi_model.joblib")

# ── Config ──────────────────────────────────────────────────────────────────
DEFAULT_STATION = "Anand Vihar, Delhi - DPCC"
ALERT_THRESHOLD = 0.25  # Safety threshold (Honest test: 95.88% recall, 2,552 misses vs 5,478 at 0.50)
STRICT_THRESHOLD = 0.50 # High-precision threshold (Honest test: 91.92% precision, 91.16% recall)


def get_latest_station_features(station: str) -> pd.DataFrame:
    """Query the most recent engineered row for the given station."""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT *
        FROM aqi_features
        WHERE station = ?
        ORDER BY timestamp DESC
        LIMIT 1
    """
    df = pd.read_sql_query(query, conn, params=(station,))
    conn.close()

    if df.empty:
        raise ValueError(f"No data found in aqi_features for station='{station}'.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def predict_alert(station: str = DEFAULT_STATION):
    """Run live 24h hazard forecast using honest non-leaking features and Open-Meteo context."""
    if not os.path.exists(ALERT_MODEL_PATH):
        raise FileNotFoundError(f"Alert model not found at {ALERT_MODEL_PATH}. Run train_model.py first.")

    latest_df = get_latest_station_features(station)
    row = latest_df.iloc[0]

    # Optional: fetch live Open-Meteo NWP forecast for advisory diagnostic display
    try:
        live_forecast_df = fetch_24h_weather_forecast(station)
        forecast_24h = get_forecast_at_horizon(live_forecast_df, pd.Timestamp.now(), horizon_hours=24)
    except Exception as e:
        forecast_24h = None

    # Load classifier
    clf = joblib.load(ALERT_MODEL_PATH)

    # Extract exactly the non-leaking candidate features expected by the trained model
    features = clf.feature_names_in_
    X = latest_df[features]

    # Predict hazard probability
    prob_poor_or_worse = clf.predict_proba(X)[0, 1]

    # Determine alert status
    if prob_poor_or_worse >= ALERT_THRESHOLD:
        status = "CRITICAL ALERT: POOR OR WORSE AIR EXPECTED"
        badge = "[CRITICAL ALERT]"
        action = "Limit prolonged outdoor exertion. Sensitive groups should remain indoors with air filtration."
    elif prob_poor_or_worse >= 0.15:
        status = "WATCH: ELEVATED POLLUTION RISK"
        badge = "[WATCH]"
        action = "Monitor ambient readings. Precautionary masks recommended for sensitive groups."
    else:
        status = "NORMAL: LOW RISK OF SEVERE DETERIORATION"
        badge = "[NORMAL]"
        action = "Standard outdoor activities permitted."

    # ── Display ─────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  DELHI-NCR 24-HOUR AIR QUALITY & METEOROLOGICAL SAFETY ALERT SYSTEM")
    print("=" * 75)
    print(f"  Station                     : {station}")
    print(f"  Latest Station Reading      : {row['timestamp'].strftime('%Y-%m-%d %H:%M')}")
    print(f"  Current Ambient PM2.5       : {row['pm25']:.1f} ug/m3 (Sub-Index AQI: {row['aqi']:.0f})")
    print("-" * 75)
    print(f"  {badge:<18s}        : {status}")
    print(f"  Hazard Probability (24h)    : {prob_poor_or_worse * 100:.1f}% (Alert Trigger: {ALERT_THRESHOLD * 100:.0f}%)")
    print(f"  Public Health Action        : {action}")
    print("-" * 75)
    if forecast_24h is not None:
        print("  Open-Meteo 24h Meteorological Advisory (NWP):")
        print(f"    * Forecasted 24h Wind     : {forecast_24h['wind_speed_forecast_24h']:.2f} m/s (Direction: {forecast_24h['wind_dir_forecast_24h']} deg)")
        print(f"    * Forecasted Temperature  : {forecast_24h['temp_forecast_24h']:.1f} C")
        print(f"    * Forecasted Humidity     : {forecast_24h['humidity_forecast_24h']:.1f}%")
        print(f"    * Forecasted Rainfall     : {forecast_24h['rainfall_forecast_24h']:.2f} mm")
        print("-" * 75)
    print("  Station Precursor Signals (Past Lags & Upwind Fronts):")
    print(f"    * 3-Hour PM2.5 Surge      : {row.get('pm25_diff_3', 0.0):+.1f} ug/m3")
    print(f"    * Regional NCR Mean       : {row.get('ncr_mean_pm25', 0.0):.1f} ug/m3")
    print(f"    * Upwind NW Smoke Front   : {row.get('nw_smoke_front', 0.0):.1f} ug/m3 (NW Gradient: {row.get('nw_gradient', 0.0):+.1f})")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    predict_alert()
