"""Shared forecast and alert decisions for evaluation and inference."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BUNDLE_PATH = ROOT / "models" / "forecast_bundle.joblib"
DB_PATH = ROOT / "data" / "aqi_history.db"
FORECAST_HORIZON = pd.Timedelta(hours=24)
TIER_LIMITS = {"poor": 90.0, "very_poor": 120.0, "severe": 250.0}
TIER_THRESHOLDS = dict.fromkeys(TIER_LIMITS, 0.25)
TIER_NAMES = ["NORMAL", "POOR", "VERY_POOR", "SEVERE"]

CANDIDATE_FEATURES = [
    # Current readings known at prediction time (t)
    "aqi",
    "pm25",
    "pm10",
    "no2",
    "pm_ratio",
    "ncr_mean_pm25",
    "station_vs_ncr_ratio",
    # Transition velocity & upwind front features (past t to t-6)
    "nw_smoke_front",
    "nw_gradient",
    "pm25_diff_3",
    "pm25_diff_6",
    "pm25_accel",
    "temp_drop_6",
    "is_stagnant",
    # Shifted AQI history (past lags t-1 to t-24)
    "aqi_lag_1",
    "aqi_lag_2",
    "aqi_lag_3",
    "aqi_lag_6",
    "aqi_lag_24",
    "aqi_rolling_mean_6",
    "aqi_rolling_mean_24",
    "aqi_rolling_std_6",
    # Shifted pollutant history (past lags)
    "pm25_lag_1",
    "pm10_lag_1",
    "no2_lag_1",
    # Calendar & Seasonality (known at time t)
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "doy_sin",
    "doy_cos",
    "is_monsoon",
    "is_winter",
    # Current weather parameters at time t
    "temperature",
    "humidity",
    "wind_speed",
    "wind_direction",
    "rainfall",
    "pressure",
    "solar_radiation",
    # Station identifier
    "station_encoded",
    "pm25_mean_24h", "pm25_mean_72h", "pm25_same_hour_3d", "pm25_change_24h",
    "ncr_pm25_mean_24h", "ncr_pm25_change_24h",
]


def tier_decisions(probabilities, thresholds):
    """Escalate to the highest active tier, including non-monotone classifier outputs."""
    decisions = np.zeros(len(probabilities), dtype=np.int8)
    for code, name in enumerate(TIER_LIMITS, start=1):
        decisions[probabilities[name] >= thresholds[name]] = code
    return decisions


def get_alert_tier(clf_poor, clf_verypoor, clf_severe, X, thresholds=None):
    models = [clf_poor, clf_verypoor, clf_severe]
    probabilities = pd.DataFrame({name: model.predict_proba(X)[:, 1]
                                  for name, model in zip(TIER_LIMITS, models)}, index=X.index)
    codes = tier_decisions(probabilities, TIER_THRESHOLDS if thresholds is None else thresholds)
    tiers = pd.Series(np.take(TIER_NAMES, codes), index=X.index)
    return tiers.iloc[0] if len(tiers) == 1 else tiers


def predict_batch(bundle, rows):
    X = rows[bundle["features"]].astype("float32")
    X["station_encoded"] = rows["station"].map(bundle["station_codes"]).astype("float32")
    prediction = pd.DataFrame(index=rows.index)
    correction = bundle["regressor"].predict(X)
    prediction["pm25_forecast"] = np.maximum(0, X["pm25"] + bundle["correction_weight"] * correction)
    for name, model in bundle["classifiers"].items():
        prediction[name] = model.predict_proba(X)[:, 1]
    prediction["tier"] = np.take(TIER_NAMES, tier_decisions(prediction, bundle["thresholds"]))
    return prediction
