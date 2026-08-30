"""
visualize_model.py
Plots Actual vs Predicted PM2.5 for the last 100 readings
at Anand Vihar, Delhi - DPCC and saves the chart as a PNG.
"""

import os
import sqlite3

import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DB_PATH = os.path.join(BASE_DIR, "data", "aqi_history.db")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_aqi_model.joblib")
OUTPUT_PATH = os.path.join(BASE_DIR, "aqi_forecast_chart.png")

# ── Config ──────────────────────────────────────────────────────────────────
STATION = "Anand Vihar, Delhi - DPCC"
POLLUTANT = "PM2.5"
FEATURES = ["lag_1", "rolling_mean_3", "hour"]
N_ROWS = 100


def load_recent_data() -> pd.DataFrame:
    """Load the last N_ROWS for the given station and pollutant."""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT *
        FROM aqi_features
        WHERE station = ? AND pollutant_id = ?
        ORDER BY last_update ASC
    """
    df = pd.read_sql_query(query, conn, params=(STATION, POLLUTANT))
    conn.close()
    df["last_update"] = pd.to_datetime(df["last_update"])
    return df.tail(N_ROWS).reset_index(drop=True)


def main():
    # ── 1. Load data ────────────────────────────────────────────────────
    df = load_recent_data()
    print(f"Loaded {len(df)} rows for {STATION} / {POLLUTANT}.")

    # ── 2. Load model and predict ───────────────────────────────────────
    model = joblib.load(MODEL_PATH)
    df["predicted"] = model.predict(df[FEATURES])

    # ── 3. Plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(df["last_update"], df["avg_value"], color="#1976D2",
            linewidth=1.5, marker="o", markersize=3, label="Actual PM2.5")
    ax.plot(df["last_update"], df["predicted"], color="#E53935",
            linewidth=1.5, marker="s", markersize=3, linestyle="--", label="Predicted PM2.5")

    ax.set_title(f"Actual vs Predicted PM2.5 — {STATION}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("PM2.5 Value", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
    fig.autofmt_xdate(rotation=40)
    fig.tight_layout()

    # ── 4. Save ─────────────────────────────────────────────────────────
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Chart saved to {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
