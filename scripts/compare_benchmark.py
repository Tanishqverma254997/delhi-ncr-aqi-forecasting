"""Replay legacy and revised forecasters on the intersection of their test station-hours."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.forecast import BUNDLE_PATH, DB_PATH, TIER_NAMES, tier_decisions
from src.models.train_model import benchmark, score


def test_rows(path, columns, start, end):
    with closing(sqlite3.connect(path)) as conn:
        sql = ("SELECT " + ", ".join(columns) + " FROM aqi_features WHERE timestamp BETWEEN ? AND ? "
               "AND pm25 IS NOT NULL AND target_pm25 IS NOT NULL")
        rows = pd.read_sql_query(sql, conn, params=(str(pd.Timestamp(start)), str(pd.Timestamp(end))),
                                 parse_dates=["timestamp"]).set_index(["station", "timestamp"])
        if not rows.index.is_unique:
            raise ValueError("Multiple rows found for the same station-hour.")
        return rows


def compare(previous_db, previous_models, db_path=DB_PATH, bundle_path=BUNDLE_PATH):
    bundle = joblib.load(bundle_path)
    meta = bundle["benchmark"]
    start, end = meta["test_start"], meta["test_end"]
    reg = joblib.load(previous_models / "best_aqi_model.joblib")
    features = list(reg.feature_names_in_)
    keys = ["station", "timestamp", "target_pm25"]
    old = test_rows(previous_db, keys + features, start, end).dropna()
    new = test_rows(db_path, keys + bundle["features"], start, end)
    shared = old.index.intersection(new.index)
    if shared.empty:
        raise ValueError("The feature caches share no eligible test rows.")
    old, new = old.loc[shared], new.loc[shared].reset_index()
    scores = benchmark(bundle, new)
    del scores["months"]
    probabilities = pd.DataFrame(index=old.index)
    for name, filename in [("poor", "best_alert_poor"), ("very_poor", "best_alert_verypoor"),
                           ("severe", "best_alert_severe")]:
        clf = joblib.load(previous_models / f"{filename}.joblib")
        probabilities[name] = clf.predict_proba(old[list(clf.feature_names_in_)])[:, 1]
    # Use the thresholds from the previous live forecaster for the operational comparison.
    previous_point = reg.predict(old[features])
    for key, thresholds in [("previous", dict.fromkeys(probabilities.columns, 0.25)),
                            ("previous_benchmark", {"poor": 0.40, "very_poor": 0.40, "severe": 0.25})]:
        active = {name: probabilities[name] >= th for name, th in thresholds.items()}
        tiers = np.take(TIER_NAMES, tier_decisions(probabilities, thresholds))
        scores[key] = score(new.target_pm25, previous_point, active, tiers)
    return {"rows": len(new), "stations": int(new.station.nunique()), "start": str(new.timestamp.min()),
            "end": str(new.timestamp.max()),
            "corrected_targets": int((old.target_pm25.to_numpy() != new.target_pm25.to_numpy()).sum()),
            **scores}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-db", type=Path, required=True)
    parser.add_argument("--previous-models", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.previous_db, args.previous_models), indent=2, allow_nan=False))
