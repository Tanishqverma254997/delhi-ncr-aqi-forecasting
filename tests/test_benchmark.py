from contextlib import closing
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd

from src.analysis.feature_engineering import iter_station_features
from src.models.forecast import CANDIDATE_FEATURES, predict_batch, tier_decisions, TIER_THRESHOLDS
from src.models.train_model import score
from src.predict_live import get_latest_station_features


class BenchmarkTests(unittest.TestCase):
    def test_forecast_inputs_are_causal_and_lags_use_elapsed_hours(self):
        times = pd.date_range("2025-01-01", periods=80, freq="h")
        rows = pd.DataFrame({"station": "A", "timestamp": times, "pm25": np.arange(80),
                             "pm10": 90., "no2": 10., "temperature": 20., "humidity": 70.,
                             "wind_speed": 2., "wind_direction": 40., "pressure": 950.,
                             "rainfall": 0., "solar_radiation": 250.}).drop(index=29)
        rows.loc[rows.timestamp == times[0], "temperature"] = np.nan
        features = next(iter_station_features(rows)).set_index("timestamp")
        self.assertTrue(np.isnan(features.at[times[30], "pm25_lag_1"]))
        self.assertTrue(np.isnan(features.at[times[32], "pm25_diff_3"]))
        self.assertTrue(np.isnan(features.at[times[30], "aqi_rolling_mean_6"]))
        self.assertTrue(np.isnan(features.at[times[0], "temperature"]))
        prefix = next(iter_station_features(rows[rows.timestamp <= times[35]])).set_index("timestamp")
        pd.testing.assert_frame_equal(prefix[CANDIDATE_FEATURES], features.loc[prefix.index, CANDIDATE_FEATURES])
        self.assertTrue(np.isnan(prefix.iloc[-1].target_pm25))
        self.assertEqual(features.loc[times[35], "target_pm25"], 59)

    def test_serve_latest_completed_hour_even_without_a_future_label(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.db"
            with closing(sqlite3.connect(path)) as conn, conn:
                conn.execute("CREATE TABLE aqi_features(station, timestamp, pm25, target_pm25)")
                conn.executemany("INSERT INTO aqi_features VALUES (?,?,?,?)", [
                    ("A", "2025-01-01 00:00:00", 10., 50.),
                    ("A", "2025-01-02 00:00:00", 50., None),
                    ("A", "2025-01-02 01:00:00", 80., None)])
            latest = get_latest_station_features("A", path, pd.Timestamp("2025-01-02 01:30"))
            self.assertEqual(latest.iloc[0].pm25, 50.)

    def test_operating_threshold_and_highest_active_tier(self):
        probabilities = pd.DataFrame({"poor": [0.25, 0], "very_poor": [0, 0.15], "severe": [0, 0.25]})
        self.assertEqual(tier_decisions(probabilities, TIER_THRESHOLDS).tolist(), [1, 3])

    def test_single_row_uses_bundle_schema_and_trigger_with_missing_weather(self):
        reg = Mock()
        reg.predict.return_value = np.array([-20.0])
        clf = Mock()
        clf.predict_proba.return_value = np.array([[0.7, 0.3]])
        bundle = {"features": ["pm25", "wind_speed", "rainfall", "station_encoded"],
                  "station_codes": {"A": 2}, "regressor": reg, "correction_weight": 0.5,
                  "classifiers": dict.fromkeys(TIER_THRESHOLDS, clf),
                  "thresholds": {"poor": 0.25, "very_poor": 0.4, "severe": 0.5}}
        rows = pd.DataFrame({"station": ["A"], "pm25": [100.], "wind_speed": ["0.3"],
                             "rainfall": [None], "station_encoded": [7]})
        result = predict_batch(bundle, rows).iloc[0]
        model_input = reg.predict.call_args.args[0]
        self.assertTrue((model_input.dtypes == "float32").all())
        self.assertEqual(model_input.iloc[0].station_encoded, 2)
        self.assertTrue(np.isnan(model_input.iloc[0].rainfall))
        self.assertEqual(result.pm25_forecast, 90.)
        self.assertEqual(result.tier, "POOR")

    def test_scoring_handles_absent_severe_events_and_exact_boundaries(self):
        y = np.array([90., 120.])
        labels = {"poor": [False, True], "very_poor": [False, False], "severe": [False, False]}
        result = score(y, y, labels, ["NORMAL", "POOR"])
        self.assertEqual(result["alerts"]["severe"]["tn"], 2)
        self.assertEqual(result["tier_accuracy"], 1.)
