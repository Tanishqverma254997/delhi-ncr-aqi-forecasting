import sqlite3
import pandas as pd

DB_PATH = "data/aqi_history.db"

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT timestamp, pm25 FROM aqi_hourly WHERE pm25 IS NOT NULL", conn)
conn.close()

df["timestamp"] = pd.to_datetime(df["timestamp"])

# Check across ALL years combined first
winter_mask = df["timestamp"].dt.month.isin([10, 11, 12, 1])
print(f"Overall (all years) winter Poor-or-worse rate: {(df.loc[winter_mask, 'pm25'] > 90).mean():.2%}")

# Then break it down year by year, so one unusual year can't hide behind an average
df["year"] = df["timestamp"].dt.year
for yr, group in df[winter_mask].groupby("year"):
    rate = (group["pm25"] > 90).mean()
    print(f"  {yr}: {rate:.2%}  (n={len(group)})")
    