import os
import sqlite3

# Path to the SQLite database file
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "aqi_history.db")


def get_connection():
    """Returns a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    return conn


def init_db():
    """Creates the aqi_readings table if it does not already exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS aqi_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station TEXT,
            last_update TEXT,
            pollutant_id TEXT,
            avg_value REAL
        )
    """)
    conn.commit()
    conn.close()
    print("Database initialised – aqi_readings table is ready.")


def _clean_value(value):
    """Converts missing or 'NA' values to None so SQLite can store them as NULL."""
    if value is None or str(value).strip().upper() == "NA" or str(value).strip() == "":
        return None
    return value


def insert_records(records):
    """
    Inserts a list of record dictionaries into the aqi_readings table.

    Each dictionary is expected to have the keys:
      station, last_update, pollutant_id, avg_value
    Missing or 'NA' values are converted to None.
    """
    conn = get_connection()
    cursor = conn.cursor()

    rows_inserted = 0
    for record in records:
        station = _clean_value(record.get("station"))
        last_update = _clean_value(record.get("last_update"))
        pollutant_id = _clean_value(record.get("pollutant_id"))
        avg_value = _clean_value(record.get("avg_value"))

        # Convert avg_value to float if it is not None
        if avg_value is not None:
            try:
                avg_value = float(avg_value)
            except (ValueError, TypeError):
                avg_value = None

        cursor.execute(
            "INSERT INTO aqi_readings (station, last_update, pollutant_id, avg_value) VALUES (?, ?, ?, ?)",
            (station, last_update, pollutant_id, avg_value),
        )
        rows_inserted += 1

    conn.commit()
    conn.close()
    print(f"Inserted {rows_inserted} record(s) into aqi_readings.")


if __name__ == "__main__":
    # Quick smoke test
    init_db()

    sample = [
        {"station": "Test Station", "last_update": "25-08-2026 00:00:00", "pollutant_id": "PM2.5", "avg_value": "42"},
        {"station": "Test Station", "last_update": "25-08-2026 00:00:00", "pollutant_id": "PM10", "avg_value": "NA"},
    ]
    insert_records(sample)
