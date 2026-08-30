import os
import sqlite3
import pandas as pd

def load_aqi_data():
    """Loads the AQI history data from SQLite into a pandas DataFrame."""
    # Resolve the path to the database file
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "aqi_history.db"))
    
    # Check if the database file exists
    if not os.path.exists(db_path):
        print(f"Database not found at: {db_path}")
        return None

    print(f"Connecting to database at: {db_path}")
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    
    try:
        # Load the data into a pandas DataFrame
        query = "SELECT * FROM aqi_readings"
        df = pd.read_sql(query, conn)
        
        # Print the first 5 rows
        print("\nFirst 5 rows of the AQI data:")
        print(df.head())
        
        return df
    finally:
        # Close the connection
        conn.close()
        print("\nDatabase connection closed.")

if __name__ == "__main__":
    df = load_aqi_data()
    print(df.info())
