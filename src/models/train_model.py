"""train_model.py

Trains both:
1. 24h-ahead continuous PM2.5 regressor (XGBRegressor)
2. 24h-ahead binary "Poor or worse" safety alert classifier (XGBClassifier)

Benchmarks both against persistence baselines and saves artifacts.
"""

import os
import sqlite3
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from xgboost import XGBRegressor, XGBClassifier

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
DB_PATH = os.path.join(BASE_DIR, "data", "aqi_history.db")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── Feature & Target Definitions ───────────────────────────────────────────
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
]
TARGET_REG = "target_pm25"
POOR_THRESHOLD = 90.0  # CPCB PM2.5 threshold for Poor or worse (>90 ug/m3 -> Sub-index >200)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def load_features():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM aqi_features", conn)
    conn.close()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    features = [c for c in CANDIDATE_FEATURES if c in df.columns]

    df["target_poor_or_worse"] = (df[TARGET_REG] > POOR_THRESHOLD).astype(int)
    df["current_poor_or_worse"] = (df["pm25"] > POOR_THRESHOLD).astype(int)

    df = df.dropna(subset=features + [TARGET_REG]).reset_index(drop=True)
    print(f"Loaded {len(df)} records. Features active: {len(features)}")
    return df, features


def split_data(df: pd.DataFrame, features: list):
    train_cutoff = df["timestamp"].quantile(TRAIN_RATIO)
    val_cutoff = df["timestamp"].quantile(TRAIN_RATIO + VAL_RATIO)

    train_df = (
        df[df["timestamp"] < train_cutoff]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    val_df = (
        df[
            (df["timestamp"] >= train_cutoff) & (df["timestamp"] < val_cutoff)
        ]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    test_df = (
        df[df["timestamp"] >= val_cutoff]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    print(
        f"Temporal splits: Train={len(train_df)} ({train_df['timestamp'].min()} to {train_df['timestamp'].max()}), "
        f"Val={len(val_df)} ({val_df['timestamp'].min()} to {val_df['timestamp'].max()}), "
        f"Test={len(test_df)} ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})"
    )
    return train_df, val_df, test_df


def pm25_to_aqi_category(pm: pd.Series | np.ndarray) -> pd.Series:
    """Map PM2.5 concentration (ug/m3) to official CPCB AQI categories."""
    pm_s = pd.Series(pm)
    conditions = [
        pm_s <= 30,
        (pm_s > 30) & (pm_s <= 60),
        (pm_s > 60) & (pm_s <= 90),
        (pm_s > 90) & (pm_s <= 120),
        (pm_s > 120) & (pm_s <= 250),
        pm_s > 250,
    ]
    categories = [
        "Good",
        "Satisfactory",
        "Moderate",
        "Poor",
        "Very Poor",
        "Severe",
    ]
    return pd.Series(np.select(conditions, categories, default="Unknown"), index=pm_s.index)


def evaluate_regression(name: str, model, X: pd.DataFrame, y: pd.Series, features: list):
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = root_mean_squared_error(y, preds)
    r2 = r2_score(y, preds)

    y_cat = pm25_to_aqi_category(y.values)
    preds_cat = pm25_to_aqi_category(preds)
    cat_acc = accuracy_score(y_cat, preds_cat)

    print(f"\n-- {name} (Continuous PM2.5 Regression) --")
    print(f"  MAE          : {mae:.4f}")
    print(f"  RMSE         : {rmse:.4f}")
    print(f"  R²           : {r2:.4f}")
    print(f"  AQI Cat Acc  : {cat_acc:.4f} ({cat_acc*100:.2f}%)")

    importances = model.feature_importances_
    print("  Top 5 Features:")
    for feat, imp in sorted(zip(features, importances), key=lambda x: -x[1])[:5]:
        print(f"    {feat:22s}: {imp:.4f}")

    print("\n  Per-Category Classification Report (from Regression Predictions):")
    print(classification_report(y_cat, preds_cat, digits=4))

    return {"mae": mae, "rmse": rmse, "r2": r2, "cat_acc": cat_acc, "preds": preds}


def evaluate_persistence_regression(df: pd.DataFrame):
    preds = df["pm25"]  # Naive forecast: PM2.5 tomorrow = PM2.5 today
    y = df[TARGET_REG]

    y_cat = pm25_to_aqi_category(y.values)
    preds_cat = pm25_to_aqi_category(preds.values)
    cat_acc = accuracy_score(y_cat, preds_cat)

    print("\n-- Persistence Baseline (Continuous PM2.5 Regression) --")
    print(f"  MAE          : {mean_absolute_error(y, preds):.4f}")
    print(f"  RMSE         : {root_mean_squared_error(y, preds):.4f}")
    print(f"  R²           : {r2_score(y, preds):.4f}")
    print(f"  AQI Cat Acc  : {cat_acc:.4f} ({cat_acc*100:.2f}%)")
    print("\n  Per-Category Classification Report (Persistence Baseline):")
    print(classification_report(y_cat, preds_cat, digits=4))


def evaluate_persistence_classification(df: pd.DataFrame):
    y = df["target_poor_or_worse"]
    preds = df["current_poor_or_worse"]

    prec = precision_score(y, preds)
    rec = recall_score(y, preds)
    f1 = f1_score(y, preds)
    cm = confusion_matrix(y, preds)

    print("\n-- Persistence Baseline (Binary Poor-or-Worse Alert) --")
    print(f"  Precision    : {prec:.4f}")
    print(f"  Recall       : {rec:.4f}")
    print(f"  F1-Score     : {f1:.4f}")
    print(f"  Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")


def evaluate_classifier(name: str, clf, X: pd.DataFrame, y: pd.Series, features: list):
    probs = clf.predict_proba(X)[:, 1]
    roc_auc = roc_auc_score(y, probs)

    print(f"\n-- {name} (Binary Poor-or-Worse Alert) --")
    print(f"  ROC-AUC      : {roc_auc:.4f}")

    importances = clf.feature_importances_
    print("  Top 5 Features for Safety Alert Classifier:")
    for feat, imp in sorted(zip(features, importances), key=lambda x: -x[1])[:5]:
        print(f"    {feat:22s}: {imp:.4f}")

    print("\n  Threshold Evaluation Sweep:")
    print(f"  {'Threshold':>10s} | {'Precision':>10s} | {'Recall':>10s} | {'F1-Score':>10s} | {'TN':>8s} | {'FP':>8s} | {'FN (Misses)':>12s} | {'TP':>8s}")
    print("  " + "-" * 95)
    for th in [0.10, 0.20, 0.25, 0.28, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80]:
        preds_th = (probs >= th).astype(int)
        prec = precision_score(y, preds_th, zero_division=0)
        rec = recall_score(y, preds_th, zero_division=0)
        f1 = f1_score(y, preds_th, zero_division=0)
        cm = confusion_matrix(y, preds_th)
        tn, fp, fn, tp = cm.ravel()
        print(f"  {th:10.2f} | {prec:10.4f} | {rec:10.4f} | {f1:10.4f} | {tn:8d} | {fp:8d} | {fn:12d} | {tp:8d}")


def train_pipeline():
    df, features = load_features()
    train_df, val_df, test_df = split_data(df, features)

    X_train, y_train_reg = train_df[features], train_df[TARGET_REG]
    X_val, y_val_reg = val_df[features], val_df[TARGET_REG]
    X_test, y_test_reg = test_df[features], test_df[TARGET_REG]

    y_train_clf = train_df["target_poor_or_worse"]
    y_val_clf = val_df["target_poor_or_worse"]
    y_test_clf = test_df["target_poor_or_worse"]

    X_trainval = pd.concat([X_train, X_val])
    y_trainval_reg = pd.concat([y_train_reg, y_val_reg])
    y_trainval_clf = pd.concat([y_train_clf, y_val_clf])

    # 1. Train Continuous PM2.5 Regressor
    print("\n" + "=" * 70)
    print("1. TRAINING CONTINUOUS PM2.5 REGRESSION MODEL")
    print("=" * 70)
    reg = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
    )
    print("Fitting XGBRegressor on Train+Val ...", flush=True)
    reg.fit(X_trainval, y_trainval_reg, verbose=False)
    print("Regression training complete.")

    evaluate_persistence_regression(test_df)
    evaluate_regression("Unseen Chronological Holdout Test Set", reg, X_test, y_test_reg, features)

    # 2. Train Binary Safety Alert Classifier (with Precision-Tuned Regularization)
    print("\n" + "=" * 70)
    print("2. TRAINING BINARY 'POOR OR WORSE' SAFETY ALERT CLASSIFIER")
    print("=" * 70)
    clf = XGBClassifier(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.75,
        min_child_weight=8,
        reg_alpha=1.5,
        reg_lambda=4.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )
    print("Fitting Precision-Tuned XGBClassifier on Train+Val ...", flush=True)
    clf.fit(X_trainval, y_trainval_clf)
    print("Classification training complete.")

    evaluate_persistence_classification(test_df)
    evaluate_classifier("Unseen Chronological Holdout Test Set", clf, X_test, y_test_clf, features)

    # Save artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    reg_path = os.path.join(MODEL_DIR, "best_aqi_model.joblib")
    clf_path = os.path.join(MODEL_DIR, "best_poor_alert_model.joblib")

    joblib.dump(reg, reg_path)
    joblib.dump(clf, clf_path)
    print(f"\nSaved regression model to: {reg_path}")
    print(f"Saved classification model to: {clf_path}")


if __name__ == "__main__":
    train_pipeline()