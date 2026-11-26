"""feature_engineering.py

Calculates official CPCB AQI from criteria pollutants, engineers
multi-pollutant ratios, shifted lags, and rolling features, and builds
a 24h-ahead forecasting target saved to aqi_features.
"""

import os
import sqlite3
import numpy as np
import pandas as pd
from contextlib import closing
from uuid import uuid4

# ── Config ───────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "aqi_history.db")
HORIZON_HOURS = 24

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
    choices.append(500.0)
    
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
    # Limit temporary Python objects while reading the multi-year archive.
    with closing(sqlite3.connect(DB_PATH)) as conn:
        chunks = pd.read_sql_query("SELECT * FROM aqi_hourly ORDER BY rowid", conn,
                                   parse_dates=["timestamp"], chunksize=25000)
        parts = []
        for chunk in chunks:
            for col in ["station", "station_id", "state", "city"]:
                if col in chunk:
                    chunk[col] = chunk[col].astype("category")
            numeric = chunk.columns.difference(["timestamp", "station", "station_id", "state", "city"])
            chunk[numeric] = chunk[numeric].astype("float64")
            parts.append(chunk)
    return pd.concat(parts, ignore_index=True)


def fill_weather_gaps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["station", "timestamp"], kind="stable").copy()
    cols = [col for col in WEATHER_COLS if col in df]
    df[cols] = df.groupby("station")[cols].ffill()
    return df


def iter_station_features(df: pd.DataFrame):
    """Build each station on a clock-hour grid, keeping the original observation keys."""
    df = df.drop_duplicates(["station", "timestamp"], keep="last")
    df = fill_weather_gaps(df)
    regional = df.groupby("timestamp")["pm25"].mean().sort_index().asfreq("h")
    nw = df["station"].str.contains("Bawana|Narela|Alipur|Rohini|Mundka", case=False, na=False)
    northwest = df[nw].groupby("timestamp")["pm25"].mean().reindex(regional.index).fillna(regional)
    region = pd.DataFrame({
        "ncr_mean_pm25": regional,
        "nw_smoke_front": northwest,
        "ncr_pm25_mean_24h": regional.rolling(24, min_periods=1).mean(),
        "ncr_pm25_change_24h": regional.diff(24),
    })

    for code, (station, rows) in enumerate(df.groupby("station", sort=True, observed=True)):
        observed = pd.DatetimeIndex(rows["timestamp"])
        frame = rows.set_index("timestamp").asfreq("h").join(region)
        frame = calculate_aqi(frame)
        hour = frame.index
        frame["hour"] = hour.hour
        frame["day_of_week"] = hour.dayofweek
        frame["is_weekend"] = (hour.dayofweek >= 5).astype(int)
        frame["month"] = hour.month
        frame["doy_sin"] = np.sin(2 * np.pi * hour.dayofyear / 365.25)
        frame["doy_cos"] = np.cos(2 * np.pi * hour.dayofyear / 365.25)
        frame["is_monsoon"] = hour.month.isin([7, 8, 9]).astype(int)
        frame["is_winter"] = hour.month.isin([11, 12, 1, 2]).astype(int)
        frame["station_encoded"] = code
        pm = frame["pm25"]
        frame["pm_ratio"] = pm / (frame["pm10"] + 1e-5)
        frame["station_vs_ncr_ratio"] = pm / (frame["ncr_mean_pm25"] + 1e-5)
        frame["nw_gradient"] = frame["nw_smoke_front"] - pm
        frame["pm25_diff_3"] = pm.diff(3)
        frame["pm25_diff_6"] = pm.diff(6)
        frame["pm25_accel"] = 2 * frame["pm25_diff_3"] - frame["pm25_diff_6"]
        frame["temp_drop_6"] = frame["temperature"].diff(6)
        frame["is_stagnant"] = ((frame["wind_speed"] < 1.5) & (frame["temp_drop_6"] < -2)).astype(int)

        for lag in [1, 2, 3, 6, 24]:
            frame[f"aqi_lag_{lag}"] = frame["aqi"].shift(lag)
        past = frame["aqi"].shift(1)
        frame["aqi_rolling_mean_6"] = past.rolling(6).mean()
        frame["aqi_rolling_mean_24"] = past.rolling(24).mean()
        frame["aqi_rolling_std_6"] = past.rolling(6).std()
        for pollutant in ["pm25", "pm10", "no2"]:
            frame[f"{pollutant}_lag_1"] = frame[pollutant].shift(1)

        frame["pm25_mean_24h"] = pm.rolling(24, min_periods=1).mean()
        frame["pm25_mean_72h"] = pm.rolling(72, min_periods=1).mean()
        frame["pm25_same_hour_3d"] = pd.concat([pm, pm.shift(24), pm.shift(48)], axis=1).mean(axis=1)
        frame["pm25_change_24h"] = pm.diff(24)

        # Historical diagnostic proxies and labels are excluded from the model inputs.
        for col, prefix in [("wind_speed", "wind_speed"), ("wind_direction", "wind_dir"),
                            ("temperature", "temp"), ("humidity", "humidity"),
                            ("pressure", "pressure"), ("rainfall", "rainfall")]:
            frame[f"{prefix}_forecast_24h"] = frame[col].shift(-HORIZON_HOURS)
        frame["target_aqi"] = frame["aqi"].shift(-HORIZON_HOURS)
        frame["target_pm25"] = pm.shift(-HORIZON_HOURS)
        yield frame.loc[observed].reset_index()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return labeled samples for offline callers."""
    return pd.concat([part.dropna(subset=["pm25", "target_pm25"]) for part in iter_station_features(df)],
                     ignore_index=True)


def rebuild_feature_table():
    # Keep the previous complete cache available to readers until the rebuild succeeds.
    stage = "features_" + uuid4().hex
    with closing(sqlite3.connect(DB_PATH)) as conn:
        try:
            count = 0
            for part in iter_station_features(load_hourly_data()):
                part.to_sql(stage, conn, if_exists="append", index=False, chunksize=5000)
                count += len(part)
            if not count:
                raise ValueError("No hourly station observations found.")
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("DROP TABLE IF EXISTS aqi_features")
                conn.execute(f"ALTER TABLE {stage} RENAME TO aqi_features")
                conn.execute("CREATE UNIQUE INDEX forecast_station_hour ON aqi_features(station, timestamp)")
        finally:
            with conn:
                conn.execute(f"DROP TABLE IF EXISTS {stage}")
    print(f"Cached {count:,} station-hours, including the newest unlabeled observations.")


if __name__ == "__main__":
    rebuild_feature_table()
