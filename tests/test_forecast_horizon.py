import unittest

import numpy as np
import pandas as pd

from src.analysis.feature_engineering import engineer_features
from src.models.train_model import FORECAST_HORIZON, split_data


class ForecastHorizonTests(unittest.TestCase):
    def test_target_uses_exact_hour_for_the_same_station(self):
        hours = pd.date_range("2025-01-01", periods=100, freq="h")
        observations = pd.DataFrame({
            "station": "Station A",
            "timestamp": hours,
            "pm25": np.arange(100) + 35,
            "pm10": 50.0,
            "no2": 5.0,
            "temperature": np.arange(100),
            "humidity": 50.0,
            "wind_speed": 1.0,
            "wind_direction": 0.0,
            "pressure": 1000.0,
            "rainfall": 0.0,
        })
        other_station = observations.assign(station="Station B", pm25=observations["pm25"] + 100)
        with_gap = pd.concat([observations.drop(index=55), other_station], ignore_index=True)

        features = engineer_features(with_gap).set_index(["station", "timestamp"])

        self.assertNotIn(("Station A", hours[31]), features.index)
        self.assertEqual(features.loc[("Station A", hours[30]), "target_pm25"], 89)
        self.assertEqual(features.loc[("Station A", hours[32]), "target_pm25"], 91)
        self.assertEqual(features.loc[("Station A", hours[32]), "temp_forecast_24h"], 56)
        self.assertEqual(features.loc[("Station B", hours[31]), "target_pm25"], 190)

    def test_targets_end_before_the_following_split(self):
        rows = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=480, freq="h")})

        train, validation, test = split_data(rows, [])

        self.assertLess(train["timestamp"].max() + FORECAST_HORIZON, validation["timestamp"].min())
        self.assertLess(validation["timestamp"].max() + FORECAST_HORIZON, test["timestamp"].min())
