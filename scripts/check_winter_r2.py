from contextlib import closing
from pathlib import Path
import sqlite3
import sys

import joblib
import pandas as pd
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.forecast import BUNDLE_PATH, DB_PATH

report = joblib.load(BUNDLE_PATH)["benchmark"]
with closing(sqlite3.connect(DB_PATH)) as conn:
    test_df = pd.read_sql_query(
        """SELECT timestamp, pm25, target_pm25 FROM aqi_features
           WHERE timestamp BETWEEN ? AND ? AND pm25 IS NOT NULL AND target_pm25 IS NOT NULL""",
        conn, params=(str(pd.Timestamp(report["test_start"])), report["test_end"]),
        parse_dates=["timestamp"])
print(f"Full test set: {len(test_df)} rows, {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

full_r2 = r2_score(test_df["target_pm25"], test_df["pm25"])
print(f"Persistence R^2 on FULL test set: {full_r2:.4f}")

# Peak winter subset within the test window (Oct-Dec, since test ends Dec 31)
winter_test = test_df[test_df["timestamp"].dt.month.isin([10, 11, 12])]
print(f"\nWinter-only subset: {len(winter_test)} rows")
winter_r2 = r2_score(winter_test["target_pm25"], winter_test["pm25"])
print(f"Persistence R^2 on WINTER-ONLY subset: {winter_r2:.4f}")

# Month-by-month, so no single month can hide behind the average
for m, group in test_df.groupby(test_df["timestamp"].dt.month):
    r2 = r2_score(group["target_pm25"], group["pm25"])
    print(f"  Month {m}: R^2 = {r2:.4f}  (n={len(group)})")
