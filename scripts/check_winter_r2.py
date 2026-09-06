import sqlite3
import pandas as pd
from sklearn.metrics import r2_score

DB_PATH = "data/aqi_history.db"

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query(
    "SELECT timestamp, pm25, target_pm25 FROM aqi_features WHERE target_pm25 IS NOT NULL",
    conn
)
conn.close()

df["timestamp"] = pd.to_datetime(df["timestamp"])

# Same test-set start date confirmed earlier from split_data() output
test_start = pd.Timestamp("2025-05-30 23:00:00")
test_df = df[df["timestamp"] >= test_start].copy()
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