"""
visualize_model.py
Plots Actual vs Predicted PM2.5 for the last 100 readings
at Anand Vihar, Delhi - DPCC and saves the chart as a PNG.
"""

from contextlib import closing
from pathlib import Path
import sys
import os
import sqlite3

import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DB_PATH = os.path.join(BASE_DIR, "data", "aqi_history.db")
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.forecast import BUNDLE_PATH, FORECAST_HORIZON, predict_batch
OUTPUT_PATH = os.path.join(BASE_DIR, "aqi_forecast_chart.png")

# ── Config ──────────────────────────────────────────────────────────────────
STATION = "Anand Vihar, Delhi - DPCC"
POLLUTANT = "PM2.5"
N_ROWS = 100


def load_recent_data() -> pd.DataFrame:
    """Load the last N_ROWS for the given station and pollutant."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        df = pd.read_sql_query(
            """SELECT * FROM aqi_features WHERE station = ? AND pm25 IS NOT NULL
               AND target_pm25 IS NOT NULL ORDER BY timestamp DESC LIMIT ?""",
            conn, params=(STATION, N_ROWS), parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def main():
    # ── 1. Load data ────────────────────────────────────────────────────
    df = load_recent_data()
    print(f"Loaded {len(df)} rows for {STATION} / {POLLUTANT}.")

    # ── 2. Load model and predict ───────────────────────────────────────
    bundle = joblib.load(BUNDLE_PATH)
    df["predicted"] = predict_batch(bundle, df)["pm25_forecast"]
    df["target_hour"] = df.timestamp + FORECAST_HORIZON

    # ── 3. Plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(df["target_hour"], df["target_pm25"], color="#1976D2",
            linewidth=1.5, marker="o", markersize=3, label="Actual PM2.5")
    ax.plot(df["target_hour"], df["predicted"], color="#E53935",
            linewidth=1.5, marker="s", markersize=3, linestyle="--", label="Predicted PM2.5")

    ax.set_title(f"Actual vs Predicted PM2.5 — {STATION}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Forecast target hour (IST)", fontsize=11)
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
