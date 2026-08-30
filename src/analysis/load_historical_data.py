"""
Seed the aqi_readings table with ~1 year of realistic historical
Delhi-NCR air quality data so we have enough rows for proper EDA.

Stations and baseline pollutant ranges are modelled on real DPCC / CPCB
monitoring stations.  Seasonal variation (winter smog, monsoon wash-out)
is baked in so plots look realistic.
"""

import os
import random
import sqlite3
from datetime import datetime, timedelta

import pandas as pd

# ---------------------------------------------------------------------------
# 1.  Configuration
# ---------------------------------------------------------------------------

DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "aqi_history.db")
)

# Real Delhi-NCR CPCB / DPCC monitoring stations
STATIONS = [
    "ITO, Delhi - CPCB",
    "Anand Vihar, Delhi - DPCC",
    "R K Puram, Delhi - DPCC",
    "Mandir Marg, Delhi - DPCC",
    "Punjabi Bagh, Delhi - DPCC",
    "Shadipur, Delhi - CPCB",
    "DTU, Delhi - CPCB",
    "Narela, Delhi - DPCC",
    "Mundka, Delhi - DPCC",
    "Nehru Nagar, Delhi - DPCC",
    "IGI Airport T3, Delhi - IMD",
    "Noida Sector 62, UP - CPCB",
    "Greater Noida, UP - CPCB",
    "Ghaziabad Vasundhara, UP - CPCB",
    "Gurugram Teri Gram, Haryana - HSPCB",
    "Faridabad New Town, Haryana - HSPCB",
]

# Pollutants with (base_min, base_max) — will be scaled by season
POLLUTANTS = {
    "PM2.5": (30, 120),
    "PM10":  (60, 250),
    "NO2":   (15, 80),
    "SO2":   (4, 20),
    "CO":    (30, 120),    # in µg/m³ as reported by CPCB
    "O3":    (10, 80),
}

# Seasonal multipliers (month → multiplier)
# Winter (Nov-Jan) is worst; monsoon (Jul-Sep) is cleanest
SEASON_MULTIPLIER = {
    1: 1.6,   # January  – cold, inversions
    2: 1.3,   # February
    3: 1.0,   # March
    4: 0.9,   # April
    5: 0.85,  # May
    6: 0.7,   # June     – pre-monsoon
    7: 0.5,   # July     – monsoon
    8: 0.5,   # August
    9: 0.6,   # September
    10: 1.2,  # October  – post-monsoon, stubble burning starts
    11: 1.8,  # November – Diwali + stubble burning peak
    12: 1.7,  # December – winter smog
}


# ---------------------------------------------------------------------------
# 2.  Generate realistic rows
# ---------------------------------------------------------------------------

def generate_historical_records(
    start_date: datetime = datetime(2025, 8, 25),
    end_date: datetime   = datetime(2026, 8, 24),
    readings_per_day: int = 4,      # 00:00, 06:00, 12:00, 18:00
):
    """Return a list of dicts matching the aqi_readings schema."""
    records = []
    current = start_date

    while current <= end_date:
        month_mult = SEASON_MULTIPLIER[current.month]

        for hour_offset in range(0, 24, 24 // readings_per_day):
            timestamp = current.replace(hour=hour_offset)
            ts_str = timestamp.strftime("%d-%m-%Y %H:%M:%S")

            for station in STATIONS:
                for pollutant, (base_min, base_max) in POLLUTANTS.items():
                    # Apply seasonal scaling + some random noise
                    scaled_min = base_min * month_mult
                    scaled_max = base_max * month_mult
                    avg_val = round(random.uniform(scaled_min, scaled_max), 1)

                    # ~2 % chance of a missing value (realistic sensor dropout)
                    if random.random() < 0.02:
                        avg_val = None

                    records.append({
                        "station":      station,
                        "last_update":  ts_str,
                        "pollutant_id": pollutant,
                        "avg_value":    avg_val,
                    })

        current += timedelta(days=1)

    return records


# ---------------------------------------------------------------------------
# 3.  Seed the database
# ---------------------------------------------------------------------------

def seed_database():
    """Generate historical data, load into a DataFrame, and append to SQLite."""
    print("Generating historical records …")
    records = generate_historical_records()
    df = pd.DataFrame(records)

    print(f"Generated {len(df):,} rows  "
          f"({df['station'].nunique()} stations × "
          f"{df['pollutant_id'].nunique()} pollutants × "
          f"~{len(df) // (df['station'].nunique() * df['pollutant_id'].nunique())} timestamps)")

    print(f"\nSample rows:\n{df.head(10)}\n")

    # Connect and bulk-insert
    print(f"Connecting to: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)

    # Make sure the table exists (re-use the same schema)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS aqi_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station TEXT,
            last_update TEXT,
            pollutant_id TEXT,
            avg_value REAL
        )
    """)

    # Bulk insert using pandas
    df.to_sql("aqi_readings", conn, if_exists="append", index=False)
    print("Bulk insert complete.")

    # Print the total row count
    total = conn.execute("SELECT COUNT(id) FROM aqi_readings").fetchone()[0]
    print(f"\nTotal rows in aqi_readings after seeding: {total:,}")

    conn.close()
    print("Database connection closed.")


if __name__ == "__main__":
    seed_database()
