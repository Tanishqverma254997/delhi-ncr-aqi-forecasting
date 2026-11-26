"""Train and benchmark 24-hour station forecasts on chronologically separated periods."""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.forecast import (
    BUNDLE_PATH, CANDIDATE_FEATURES, DB_PATH, FORECAST_HORIZON, TIER_LIMITS,
    TIER_NAMES, TIER_THRESHOLDS, get_alert_tier, predict_batch,
)

TARGET_REG = "target_pm25"
# Freeze the previous benchmark's boundaries so changes in eligible rows cannot move the test.
VALIDATION_START = "2024-11-07 08:00"
TEST_START = "2025-05-30 23:00"


def load_features(db_path=DB_PATH):
    cols = ["station", "timestamp", TARGET_REG, *CANDIDATE_FEATURES]
    with closing(sqlite3.connect(db_path)) as conn:
        available = {row[1] for row in conn.execute("PRAGMA table_info(aqi_features)")}
        if set(cols) - available:
            raise ValueError("Rebuild aqi_features with src/analysis/feature_engineering.py first.")
        sql = "SELECT " + ", ".join(cols) + " FROM aqi_features WHERE pm25 IS NOT NULL AND target_pm25 IS NOT NULL"
        chunks = pd.read_sql_query(sql, conn, parse_dates=["timestamp"], chunksize=25000,
                                   dtype=dict.fromkeys(CANDIDATE_FEATURES, "float32"))
        return pd.concat(chunks, ignore_index=True), CANDIDATE_FEATURES


def split_data(df, features=None, validation_start=None, test_start=None):
    times = df["timestamp"]
    val = pd.Timestamp(validation_start) if validation_start is not None else times.quantile(0.70)
    test = pd.Timestamp(test_start) if test_start is not None else times.quantile(0.85)
    target_time = times + FORECAST_HORIZON
    masks = [target_time < val, (times >= val) & (target_time < test), times >= test]
    parts = tuple(df.loc[mask].sort_values(["timestamp"], kind="stable").reset_index(drop=True) for mask in masks)
    if any(part.empty for part in parts):
        raise ValueError("Training, validation and test must each contain labeled station-hours.")
    return parts


def regression_metrics(y, pred):
    limits = [30, 60, *TIER_LIMITS.values()]
    return {"mae": float(mean_absolute_error(y, pred)), "rmse": float(root_mean_squared_error(y, pred)),
            "r2": float(r2_score(y, pred)) if len(y) > 1 else None,
            "band_accuracy": float(np.mean(np.digitize(y, limits, right=True) == np.digitize(pred, limits, right=True)))}


def score(y, point, alerts, tiers):
    y = np.asarray(y)
    result = {"regression": regression_metrics(y, point), "alerts": {}}
    for name, limit in TIER_LIMITS.items():
        tn, fp, fn, tp = confusion_matrix(y > limit, alerts[name], labels=[False, True]).ravel().tolist()
        result["alerts"][name] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1)}
    actual = np.take(TIER_NAMES, np.digitize(y, list(TIER_LIMITS.values()), right=True))
    result["tier_accuracy"] = float(accuracy_score(actual, tiers))
    result["tier_macro_f1"] = float(f1_score(actual, tiers, labels=TIER_NAMES, average="macro", zero_division=0))
    return result


def benchmark(bundle, test):
    pred = predict_batch(bundle, test)
    pm = test["pm25"]
    truth = test[TARGET_REG]
    triggers = {name: pred[name] >= th for name, th in bundle["thresholds"].items()}
    persistence = {name: pm > limit for name, limit in TIER_LIMITS.items()}
    current_tiers = np.take(TIER_NAMES, np.digitize(pm, list(TIER_LIMITS.values()), right=True))
    result = {"persistence": score(truth, pm, persistence, current_tiers),
              "forecast": score(truth, pred.pm25_forecast, triggers, pred.tier), "months": {}}
    for month, part in test.groupby(test.timestamp.dt.to_period("M")):
        result["months"][str(month)] = {"rows": len(part),
            "forecast": regression_metrics(part[TARGET_REG], pred.loc[part.index, "pm25_forecast"]),
            "persistence": regression_metrics(part[TARGET_REG], part.pm25)}
    return result


def regressor():
    return XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, reg_alpha=1, reg_lambda=2, random_state=42, n_jobs=-1)


def train_pipeline(validation_start=VALIDATION_START, test_start=TEST_START):
    df, features = load_features()
    train, val, test = split_data(df, features, validation_start, test_start)
    del df
    print(f"Rows: train={len(train):,}, validation={len(val):,}, test={len(test):,}", flush=True)
    # Fit a correction to persistence. Choose its strength using only validation errors.
    model = regressor().fit(train[features], train[TARGET_REG] - train.pm25)
    delta = model.predict(val[features]).astype("float64")
    residual = (val[TARGET_REG] - val.pm25).to_numpy()
    energy = np.dot(delta, delta)
    weight = float(np.clip(np.dot(delta, residual) / energy, 0, 1)) if energy else 0.0
    validation = {"weight": weight, "rows": len(val),
                  "forecast": regression_metrics(val[TARGET_REG], np.maximum(0, val.pm25 + weight * delta)),
                  "persistence": regression_metrics(val[TARGET_REG], val.pm25)}
    print("Validation:", json.dumps(validation), flush=True)
    fit = pd.concat([train, val], ignore_index=True)
    station_codes = fit.drop_duplicates("station").set_index("station")["station_encoded"].to_dict()
    X = fit[features]
    reg = regressor().fit(X, fit[TARGET_REG] - fit.pm25)
    classifiers = {}
    for name, limit in TIER_LIMITS.items():
        print(f"Training {name} classifier ...", flush=True)
        target = fit[TARGET_REG] > limit
        if target.nunique() != 2:
            raise ValueError(f"The training period needs both outcomes for {name}.")
        classifiers[name] = XGBClassifier(n_estimators=350, max_depth=6, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.75, min_child_weight=8, reg_alpha=1.5,
            reg_lambda=4, random_state=42, n_jobs=-1, eval_metric="logloss").fit(X, target)
    bundle = {"features": features, "station_codes": station_codes, "regressor": reg,
              "correction_weight": weight, "classifiers": classifiers, "thresholds": TIER_THRESHOLDS.copy(),
              "horizon_hours": 24}
    report = benchmark(bundle, test)
    report.update({"validation_start": str(validation_start), "test_start": str(test_start),
                   "test_end": str(test.timestamp.max()), "test_rows": len(test),
                   "stations": int(test.station.nunique()), "train_rows": len(train), "validation": validation})
    bundle["benchmark"] = report
    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=BUNDLE_PATH.parent, delete=False) as handle:
        tmp = Path(handle.name)
    try:
        joblib.dump(bundle, tmp)
        os.replace(tmp, BUNDLE_PATH)
    finally:
        tmp.unlink(missing_ok=True)
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-start", default=VALIDATION_START)
    parser.add_argument("--test-start", default=TEST_START)
    args = parser.parse_args()
    train_pipeline(args.validation_start, args.test_start)
