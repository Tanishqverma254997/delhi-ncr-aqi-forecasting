"""fetch_open_meteo_forecast.py

Fetches 24-hour to 48-hour ahead Numerical Weather Prediction (NWP)
forecasts from Open-Meteo API for Delhi-NCR coordinates.
Provides hourly forecasted wind speed, direction, temperature, humidity,
surface pressure, and precipitation.
"""

import requests
import pandas as pd
from datetime import datetime

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Station coordinates mapping for Delhi-NCR
NCR_STATION_COORDINATES = {
    "Anand Vihar, Delhi - DPCC": (28.647, 77.316),
    "Alipur, Delhi - DPCC": (28.815, 77.153),
    "Bawana, Delhi - DPCC": (28.776, 77.051),
    "Narela, Delhi - DPCC": (28.853, 77.093),
    "Punjabi Bagh, Delhi - DPCC": (28.674, 77.131),
    "R K Puram, Delhi - DPCC": (28.563, 77.186),
    "ITO, Delhi - DPCC": (28.631, 77.249),
    "Mandir Marg, Delhi - DPCC": (28.636, 77.199),
    "DEFAULT": (28.6139, 77.2090),  # Central Delhi
}

HOURLY_PARAMS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "precipitation",
]


def fetch_24h_weather_forecast(station: str = "DEFAULT") -> pd.DataFrame:
    """Fetch 48-hour hourly weather forecast from Open-Meteo for a station.
    
    Returns a DataFrame with columns:
    [timestamp, temperature, humidity, wind_speed, wind_direction, pressure, rainfall]
    """
    coords = NCR_STATION_COORDINATES.get(station, NCR_STATION_COORDINATES["DEFAULT"])
    lat, lon = coords

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_PARAMS),
        "timezone": "Asia/Kolkata",
        "forecast_days": 3,
    }

    print(f"Querying Open-Meteo Forecast API for ({lat}, {lon}) ...")
    resp = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()["hourly"]

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(data["time"]),
        "temperature": data["temperature_2m"],
        "humidity": data["relative_humidity_2m"],
        "wind_speed": data["wind_speed_10m"],  # km/h from Open-Meteo; converted to m/s below
        "wind_direction": data["wind_direction_10m"],
        "pressure": data["surface_pressure"],
        "rainfall": data["precipitation"],
    })

    # Convert wind speed from km/h to m/s to match CPCB station sensors
    df["wind_speed"] = (df["wind_speed"] / 3.6).round(2)
    return df


def get_forecast_at_horizon(forecast_df: pd.DataFrame, current_timestamp: pd.Timestamp, horizon_hours: int = 24) -> dict:
    """Extract weather forecast exactly horizon_hours into the future."""
    target_time = current_timestamp + pd.Timedelta(hours=horizon_hours)
    
    # Match closest forecast timestamp
    idx = (forecast_df["timestamp"] - target_time).abs().idxmin()
    forecast_row = forecast_df.loc[idx]

    return {
        "forecast_time": forecast_row["timestamp"],
        "wind_speed_forecast_24h": forecast_row["wind_speed"],
        "wind_dir_forecast_24h": forecast_row["wind_direction"],
        "temp_forecast_24h": forecast_row["temperature"],
        "humidity_forecast_24h": forecast_row["humidity"],
        "pressure_forecast_24h": forecast_row["pressure"],
        "rainfall_forecast_24h": forecast_row["rainfall"],
    }


if __name__ == "__main__":
    df = fetch_24h_weather_forecast("Anand Vihar, Delhi - DPCC")
    print("\n-- Sample Forecast Output (Next 24h) --")
    print(df.head(25).to_string(index=False))
