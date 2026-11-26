"""Forecast the target hour from the latest completed, cached station observation."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys

import joblib
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.fetch_open_meteo_forecast import fetch_24h_weather_forecast, get_forecast_at_horizon
from src.models.forecast import BUNDLE_PATH, DB_PATH, FORECAST_HORIZON, predict_batch

DEFAULT_STATION = "Anand Vihar, Delhi - DPCC"


def get_latest_station_features(station, db_path=DB_PATH, now=None):
    now = pd.Timestamp.now("Asia/Kolkata").tz_localize(None) if now is None else now
    with closing(sqlite3.connect(db_path)) as conn:
        rows = pd.read_sql_query(
            """SELECT * FROM aqi_features WHERE station = ? AND pm25 IS NOT NULL
               AND timestamp < ? ORDER BY timestamp DESC LIMIT 1""",
            conn, params=(station, str(now.floor("h"))), parse_dates=["timestamp"])
    if rows.empty:
        raise ValueError(f"No completed station-hour with a PM2.5 reading for {station!r}.")
    return rows


def forecast_station(station=DEFAULT_STATION, bundle_path=BUNDLE_PATH, db_path=DB_PATH, now=None):
    """Return a JSON-compatible result. Times are start-of-hour labels in Asia/Kolkata."""
    if not Path(bundle_path).is_file():
        raise FileNotFoundError("Run src/models/train_model.py to build the benchmarked forecast bundle.")
    now = pd.Timestamp.now("Asia/Kolkata").tz_localize(None) if now is None else now
    bundle = joblib.load(bundle_path)
    rows = get_latest_station_features(station, db_path, now)
    prediction = predict_batch(bundle, rows).iloc[0]
    row = rows.iloc[0]
    target = row.timestamp + FORECAST_HORIZON
    return {"station": station, "timezone": "Asia/Kolkata", "observation_hour": row.timestamp.isoformat(),
            "target_hour": target.isoformat(), "historical": bool(target < now.floor("h")),
            "observed_pm25": float(row.pm25), "forecast_pm25": float(prediction.pm25_forecast),
            "unit": "µg/m³", "tier": prediction.tier,
            "probabilities": {name: float(prediction[name]) for name in bundle["thresholds"]},
            "thresholds": bundle["thresholds"]}


def predict_alert(station=DEFAULT_STATION):
    result = forecast_station(station)
    print(f"{'Historical replay' if result['historical'] else 'Station forecast'} | {station}")
    print(f"Observation hour {result['observation_hour']}  ->  target hour {result['target_hour']} (IST)")
    print(f"Hourly PM2.5 forecast {result['forecast_pm25']:.1f} ug/m3 | Alert tier {result['tier']}")
    for name, probability in result["probabilities"].items():
        print(f"  {name}: {probability:.1%} (trigger {result['thresholds'][name]:.0%})")
    if not result["historical"]:
        try:
            weather = fetch_24h_weather_forecast(station)
            target = pd.Timestamp(result["target_hour"])
            if weather["timestamp"].eq(target).any():
                context = get_forecast_at_horizon(weather, pd.Timestamp(result["observation_hour"]))
                print(f"Target-hour weather: {context['temp_forecast_24h']:.1f} C, "
                      f"wind {context['wind_speed_forecast_24h']:.2f} m/s")
        except (OSError, ValueError, KeyError) as exc:
            print(f"Weather advisory unavailable: {exc}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default=DEFAULT_STATION)
    parser.add_argument("--json", action="store_true", help="Emit only the model result, without a weather request.")
    args = parser.parse_args()
    if args.json:
        print(json.dumps(forecast_station(args.station), allow_nan=False))
    else:
        predict_alert(args.station)
