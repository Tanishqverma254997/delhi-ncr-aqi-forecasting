"""feature_engineering.py

Calculates official CPCB AQI from criteria pollutants, engineers
multi-pollutant ratios, shifted lags, and rolling features, and builds
a 24h-ahead forecasting target saved to aqi_features.
"""

import os
import sqlite3
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ── Config ───────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "aqi_history.db")
HORIZON_HOURS = 24
TRAIN_RATIO = 0.70

WEATHER_COLS = [
    "temperature",
    "humidity",
    "wind_speed",
    "wind_direction",
    "rainfall",
    "pressure",
    "solar_radiation",
]

# ── Official CPCB Breakpoints (C_lo, C_hi, I_lo, I_hi) ───────────────────
BREAKPOINTS = {
    "pm25": [
        (0, 30, 0, 50),
        (30, 60, 51, 100),
        (60, 90, 101, 200),
        (90, 120, 201, 300),
        (120, 250, 301, 400),
        (250, 500, 401, 500),
    ],
    "pm10": [
        (0, 50, 0, 50),
        (50, 100, 51, 100),
        (100, 250, 101, 200),
        (250, 350, 201, 300),
        (350, 430, 301, 400),
        (430, 600, 401, 500),
    ],
    "no2": [
        (0, 40, 0, 50),
        (40, 80, 51, 100),
        (80, 180, 101, 200),
        (180, 280, 201, 300),
        (280, 400, 301, 400),
        (400, 1000, 401, 500),
    ],
    "so2": [
        (0, 40, 0, 50),
        (40, 80, 51, 100),
        (80, 380, 101, 200),
        (380, 800, 201, 300),
        (800, 1600, 301, 400),
        (1600, 2000, 401, 500),
    ],
    "co": [
        (0, 1.0, 0, 50),
        (1.0, 2.0, 51, 100),
        (2.0, 10.0, 101, 200),
        (10.0, 17.0, 201, 300),
        (17.0, 34.0, 301, 400),
        (34.0, 50.0, 401, 500),
    ],
    "ozone": [
        (0, 50, 0, 50),
        (50, 100, 51, 100),
        (100, 168, 101, 200),
        (168, 208, 201, 300),
        (208, 748, 301, 400),
        (748, 1000, 401, 500),
    ],
}


def _calc_subindex_vectorized(s: pd.Series, pollutant: str) -> pd.Series:
    """Vectorized linear piecewise interpolation for CPCB sub-index calculation."""
    bps = BREAKPOINTS[pollutant]
    conds = []
    choices = []
    for c_lo, c_hi, i_lo, i_hi in bps:
        cond = (s >= c_lo) & (s <= c_hi)
        val = i_lo + (i_hi - i_lo) * (s - c_lo) / (c_hi - c_lo)
        conds.append(cond)
        choices.append(val)
    c_lo, c_hi, i_lo, i_hi = bps[-1]
    conds.append(s > c_hi)
    choices.append(np.minimum(500.0, i_hi + (s - c_hi)))
    
    res = np.select(conds, choices, default=np.nan)
    return pd.Series(res, index=s.index)


def calculate_aqi(df: pd.DataFrame) -> pd.DataFrame:
    """Computes individual sub-indices and overall composite AQI."""
    sub_cols = []
    for pol in ["pm25", "pm10", "no2", "so2", "co", "ozone"]:
        if pol in df.columns:
            sub_col = f"sub_{pol}"
            df[sub_col] = _calc_subindex_vectorized(df[pol], pol)
            sub_cols.append(sub_col)

    # CPCB requires PM2.5 or PM10 presence to declare a representative AQI
    has_particulates = df["pm25"].notna() | df["pm10"].notna()
    df["aqi"] = df[sub_cols].max(axis=1)
    df.loc[~has_particulates, "aqi"] = np.nan
    return df


def load_hourly_data() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM aqi_hourly", conn)
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"Loaded {len(df)} rows from aqi_hourly.")
    return df


def fill_weather_gaps(df: pd.DataFrame) -> pd.DataFrame:
    train_cutoff = df["timestamp"].quantile(TRAIN_RATIO)
    train_mask = df["timestamp"] < train_cutoff

    for col in WEATHER_COLS:
        if col in df.columns:
            median_val = df.loc[train_mask, col].median()
            df[col] = df.groupby("station")[col].transform(
                lambda s: s.ffill().fillna(median_val)
            )

    missing = df[[c for c in WEATHER_COLS if c in df.columns]].isna().sum()
    print(f"Weather NaNs after filling:\n{missing}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Deduplicate & sort chronologically per station
    df = df.drop_duplicates(subset=["station", "timestamp"], keep="last")
    df = df.sort_values(["station", "timestamp"]).reset_index(drop=True)

    # 2. Derive official AQI
    df = calculate_aqi(df)
    print(f"Computed AQI for {len(df)} rows.")

    # 3. Calendar & Seasonality Features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["timestamp"].dt.month
    day_of_year = df["timestamp"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    # 4. Multi-Pollutant & Spatial Interactions
    if "pm10" in df.columns:
        df["pm_ratio"] = df["pm25"] / (df["pm10"] + 1e-5)
    else:
        df["pm_ratio"] = 0.0

    # Regional NCR background concentration
    ncr_mean_pm25 = df.groupby("timestamp")["pm25"].transform("mean")
    df["ncr_mean_pm25"] = ncr_mean_pm25
    df["station_vs_ncr_ratio"] = df["pm25"] / (ncr_mean_pm25 + 1e-5)

    # Northwest transboundary smoke front & gradient
    nw_mask = df["station"].str.contains("Bawana|Narela|Alipur|Rohini|Mundka", case=False, na=False)
    nw_mean = df[nw_mask].groupby("timestamp")["pm25"].mean()
    df["nw_smoke_front"] = df["timestamp"].map(nw_mean).fillna(df["ncr_mean_pm25"])
    df["nw_gradient"] = df["nw_smoke_front"] - df["pm25"]

    # Transition Velocity & Acceleration (Rate of pollution surge)
    df["pm25_diff_3"] = df.groupby("station")["pm25"].diff(3)
    df["pm25_diff_6"] = df.groupby("station")["pm25"].diff(6)
    df["pm25_accel"] = df["pm25_diff_3"] - (df["pm25_diff_6"] - df["pm25_diff_3"])

    # Atmospheric Stagnation & Inversion Precursor
    df["temp_drop_6"] = df.groupby("station")["temperature"].diff(6)
    df["is_stagnant"] = ((df["wind_speed"] < 1.5) & (df["temp_drop_6"] < -2.0)).astype(int)

    # 5. Station Encoding
    le = LabelEncoder()
    df["station_encoded"] = le.fit_transform(df["station"])

    # 6. Shifted Lags and Rolling Windows (Grouped by station, strictly shifted)
    grouped_aqi = df.groupby("station")["aqi"]
    df["aqi_lag_1"] = grouped_aqi.shift(1)
    df["aqi_lag_2"] = grouped_aqi.shift(2)
    df["aqi_lag_3"] = grouped_aqi.shift(3)
    df["aqi_lag_6"] = grouped_aqi.shift(6)
    df["aqi_lag_24"] = grouped_aqi.shift(24)

    df["aqi_rolling_mean_6"] = grouped_aqi.transform(
        lambda x: x.shift(1).rolling(window=6).mean()
    )
    df["aqi_rolling_mean_24"] = grouped_aqi.transform(
        lambda x: x.shift(1).rolling(window=24).mean()
    )
    df["aqi_rolling_std_6"] = grouped_aqi.transform(
        lambda x: x.shift(1).rolling(window=6).std()
    )

    # Shifted individual pollutant lags for chemical context
    for pol in ["pm25", "pm10", "no2"]:
        if pol in df.columns:
            df[f"{pol}_lag_1"] = df.groupby("station")[pol].shift(1)

    # 7. 24-Hour-Ahead Weather Forecast Features (NWP Numerical Weather Prediction Proxy)
    grouped_station = df.groupby("station")
    df["wind_speed_forecast_24h"] = grouped_station["wind_speed"].shift(-HORIZON_HOURS)
    df["wind_dir_forecast_24h"] = grouped_station["wind_direction"].shift(-HORIZON_HOURS)
    df["temp_forecast_24h"] = grouped_station["temperature"].shift(-HORIZON_HOURS)
    df["humidity_forecast_24h"] = grouped_station["humidity"].shift(-HORIZON_HOURS)
    df["pressure_forecast_24h"] = grouped_station["pressure"].shift(-HORIZON_HOURS)
    df["rainfall_forecast_24h"] = grouped_station["rainfall"].shift(-HORIZON_HOURS)

    # Meteorological dynamics between t and t+24
    df["wind_speed_delta_24h"] = df["wind_speed_forecast_24h"] - df["wind_speed"]
    df["temp_drop_forecast_24h"] = df["temp_forecast_24h"] - df["temperature"]
    df["ventilation_index_forecast_24h"] = (
        df["wind_speed_forecast_24h"] * (df["temp_forecast_24h"] + 273.15) / 100.0
    )
    df["is_stagnant_forecast_24h"] = (
        (df["wind_speed_forecast_24h"] < 1.5) & (df["temp_drop_forecast_24h"] < -2.0)
    ).astype(int)

    # Seasonal Washout & Inversion Context (Reduces False Positives)
    df["rain_washout_forecast"] = (df["rainfall_forecast_24h"] > 0.2).astype(int)
    df["is_monsoon"] = df["month"].isin([7, 8, 9]).astype(int)
    df["is_winter"] = df["month"].isin([11, 12, 1, 2]).astype(int)
    df["humid_stagnant_forecast"] = (
        (df["humidity_forecast_24h"] > 75) & (df["wind_speed_forecast_24h"] < 1.5)
    ).astype(int)

    # 8. Forecasting Targets: 24 Hours Ahead (Shifted on regular hourly grid per station)
    df["target_aqi"] = grouped_aqi.shift(-HORIZON_HOURS)
    df["target_pm25"] = df.groupby("station")["pm25"].shift(-HORIZON_HOURS)

    # 9. Clean nulls created by shifting
    required_cols = [
        "aqi_lag_1",
        "aqi_lag_2",
        "aqi_lag_3",
        "aqi_lag_6",
        "aqi_lag_24",
        "aqi_rolling_mean_6",
        "aqi_rolling_mean_24",
        "aqi_rolling_std_6",
        "pm25_diff_3",
        "pm25_diff_6",
        "temp_drop_6",
        "wind_speed_forecast_24h",
        "temp_forecast_24h",
        "target_aqi",
        "target_pm25",
    ]
    before = len(df)
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    print(f"Dropped {before - len(df)} NaN rows -> {len(df)} rows remaining.")

    return df


def save_features(df: pd.DataFrame):
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("aqi_features", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Saved {len(df)} rows to 'aqi_features' table.")


if __name__ == "__main__":
    df = load_hourly_data()
    df = fill_weather_gaps(df)
    features_df = engineer_features(df)
    print("\nSample Engineered Records:")
    print(
        features_df[
            ["station", "timestamp", "aqi", "target_aqi", "ncr_mean_pm25", "pm_ratio"]
        ].head(5)
    )
    save_features(features_df)