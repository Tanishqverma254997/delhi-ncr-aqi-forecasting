# Delhi-NCR 24-hour PM2.5 forecasting

The system forecasts the hourly mean PM2.5 concentration at 40 Delhi monitoring stations, 24 hours after the latest completed observation hour. Three classifiers estimate the probability that the target hour exceeds 90, 120 and 250 µg/m³. These are single-hour warning indicators. Regulatory AQI uses a different averaging period.

## Benchmark on matched station-hours

Both project versions were scored on the same 161,238 station-hours from 30 May 2025 at 23:00 through 31 December 2025 at 05:00 IST. All 40 stations are represented. Both versions used identical target values on these rows. The previous models were retrained before the changes and matched the original local model files byte for byte. Persistence predicts that each station will have its current PM2.5 reading again tomorrow.

| Holdout metric | Persistence | Previous live configuration | Revised |
| --- | ---: | ---: | ---: |
| PM2.5 mean absolute error (µg/m³) ↓ | 35.01 | 34.33 | **32.40** |
| PM2.5 root mean squared error (µg/m³) ↓ | 60.39 | 59.90 | **56.14** |
| PM2.5 R² ↑ | 0.680 | 0.685 | **0.723** |
| Six PM2.5 bands, accuracy ↑ | 54.40% | 54.16% | **56.06%** |
| Poor-or-worse recall ↑ | 89.51% | 95.90% | **96.04%** |
| Poor-or-worse false alarms ↓ | 5,927 | 9,188 | 9,335 |
| Very-poor-or-worse recall ↑ | 85.49% | 96.21% | 96.05% |
| Severe alert precision ↑ | 61.88% | 52.95% | 55.23% |
| Severe alert recall ↑ | 59.58% | 69.92% | **70.81%** |
| Severe alert F1 ↑ | 0.607 | 0.603 | **0.621** |
| Four alert tiers, exact accuracy ↑ | **79.28%** | 78.32% | 78.73% |
| Four alert tiers, macro F1 ↑ | **0.612** | 0.595 | 0.604 |

The previous live script used a 0.25 probability trigger for every tier. Its training benchmark reported tier decisions at 0.40, 0.40 and 0.25. Those earlier benchmark settings scored 79.44% four-tier accuracy on these rows. The revised benchmark and inference code both use the live 0.25 settings. The revised model missed 155 fewer severe station-hours and raised 823 fewer false severe alarms than the previous live configuration.

### What changed

| Component | Previous version | Revised version |
| --- | --- | --- |
| Feature timing | Exact 24-hour target and purged splits, with historical lags calculated by row position | Preserves exact targets and split gaps. Computes station lags on an hourly grid and adds six recent-history features, including regional trends and the station's same-hour average across three days |
| Missing observations | Dropped rows with any missing model input. Filled leading weather gaps with a training-period median | Requires observed PM2.5 at both ends for scoring. Carries weather forward within each station and lets XGBoost handle missing inputs |
| Point forecast | XGBoost predicted the absolute future concentration | XGBoost predicts the change from current PM2.5. Validation selected a correction weight of 0.7991 to minimize squared error |
| Alerts and serving | Separate model files, duplicated trigger settings, latest row from a labeled table | One bundle holds four fitted models, feature order, station codes, triggers and benchmark results. Batch scoring and station inference share the prediction function. The cache includes the latest unlabeled hours |

### Full revised holdout

Allowing missing predictors added 24,751 eligible station-hours. Across all **185,989** rows the revised model achieved MAE **31.52** and RMSE **55.08** µg/m³, compared with **34.49** and **61.00** for persistence. Its monthly MAE was below persistence in every reporting month, including the partial May window. Four-tier accuracy was **79.54%**, versus **80.05%** for persistence. The matched table above isolates the change in scores from this change in coverage. The model and training coverage changed together, so the comparison measures the complete pipeline.

The training period contains 867,758 rows. Validation contains 178,352 rows beginning 7 November 2024 at 08:00 IST. Any training or validation row whose target reaches the following split is excluded. The correction weight was fixed on validation before refitting on training plus validation. The final test period has appeared in earlier project experiments. A later observation period is still needed to measure prospective performance.

## Run locally

The benchmark was run with Python 3.13.3. From the project root, activate the virtual environment and run:

```sh
python -m pip install -r requirements.txt
python src/analysis/feature_engineering.py
python src/models/train_model.py
python src/predict_live.py --json
python -m unittest discover -s tests -v
```

Feature engineering reads `data/aqi_history.db` and rebuilds the `aqi_features` cache from `aqi_hourly`. On a clean checkout, first run `python src/ingestion/fetch_cpcb_15min.py` to populate the hourly archive. The training command writes one bundle at `models/forecast_bundle.joblib` and prints JSON metrics for each alert threshold and monthly regression results. Its default split dates reproduce the comparison above. Use `--validation-start` and `--test-start` when evaluating another period.

To replay the before-and-after comparison in this local checkout:

```sh
python scripts/compare_benchmark.py --previous-db data/benchmark_previous/holdout.db --previous-models data/benchmark_previous
```

The legacy holdout snapshot and model files are local artifacts. SQL databases and trained models are excluded from Git. The comparison command accepts other snapshots with the same schemas.

## Inference for a frontend

`python src/predict_live.py --station "Anand Vihar, Delhi - DPCC" --json` emits the observation hour, target hour, timezone, PM2.5 forecast, tier, probabilities, thresholds and a historical flag. Both timestamps denote the start of an hourly bucket in Asia/Kolkata. Issue forecasts after the observation bucket closes. The bundled local archive ends on 1 January 2026. The JSON marks a forecast whose target hour has passed as historical. A live deployment needs an ingestion job that supplies recent hourly observations and refreshes the feature cache.

<details>
<summary>Earlier project results and development notes (archived)</summary>

## 4. Results & Benchmark Evaluation

All metrics below are evaluated on the unseen chronological holdout test set (**2025-05-30 23:00:00 to 2025-12-31 05:00:00**, $N=161,279$ hourly observations across 40 stations) with all future-shifted weather leaks strictly removed.

### a & b. Continuous Regression Comparison

| Model | MAE ($\mu\text{g/m}^3$) | RMSE ($\mu\text{g/m}^3$) | $R^2$ | 6-Class AQI Category Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Persistence Baseline** ($\hat{y}_{t+24} = y_t$) | 35.0024 | 60.3863 | 0.6796 | 54.39% |
| **Trained Regressor** (XGBRegressor, 41 features) | 34.5681 | 60.4317 | 0.6791 | 54.17% |
| **Delta ($\Delta$)** | **-0.43 µg/m³** | **+0.05 µg/m³** | **-0.0005** | **-0.22%** |

### c. Real Edge Over Baseline
The continuous regression model shows **zero practical edge** over simple 24-hour persistence:
- MAE improves by only **0.43 $\mu\text{g/m}^3$** (a 1.2% relative error reduction).
- RMSE is essentially identical (+0.05 $\mu\text{g/m}^3$).
- $R^2$ is flat at **0.6791** vs **0.6796** (persistence).
- 6-class AQI category accuracy drops slightly to **54.17%** compared to **54.39%** for naive persistence.

Single-hour PM2.5 readings are subject to high-frequency micro-scale noise. Because point regression error ($\text{MAE} \approx 34.57\ \mu\text{g/m}^3$) exceeds the width of individual CPCB AQI category bands ($\sim 30\ \mu\text{g/m}^3$), point regression predictions routinely spill across adjacent category boundaries.

### d. Per-Category Performance (Regression Predictions vs Ground Truth)

| AQI Category | PM2.5 Range ($\mu\text{g/m}^3$) | Test Support ($N$) | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Good** | $0 - 30$ | 41,257 | 0.7381 | 0.4157 | 0.5318 |
| **Satisfactory** | $31 - 60$ | 41,171 | 0.4910 | 0.7164 | 0.5827 |
| **Moderate** | $61 - 90$ | 16,911 | 0.3656 | 0.3250 | 0.3441 |
| **Poor** *(Alert Target)* | $91 - 120$ | 11,280 | **0.3139** | **0.2290** | **0.2648** |
| **Very Poor** | $121 - 250$ | 33,319 | 0.5958 | 0.7310 | 0.6565 |
| **Severe** | $> 250$ | 17,341 | 0.5992 | 0.4779 | 0.5317 |

> **Critical Finding:** **"Poor" is the weakest category in the regression model**, with a precision of **0.3139** and recall of **0.2290** (F1 = 0.2648). Because "Poor" is a narrow transition band ($91–120\ \mu\text{g/m}^3$), regression predictions frequently overshoot into "Very Poor" ($121–250\ \mu\text{g/m}^3$) or undershoot into "Moderate" ($61–90\ \mu\text{g/m}^3$).

---

### e. Binary "Poor or Worse" Safety Alert Classifier

Reframing the task as a dedicated binary classification problem—*"Will air quality tomorrow cross into Poor or worse ($\text{PM2.5} > 90\ \mu\text{g/m}^3$ / $\text{AQI} > 200$)?"*—aligns the model directly with the alert objective:

- **Test Set Positive Class Support (Hazardous):** 61,940 observations (38.41% of test set).
- **Test Set Negative Class Support (Non-Hazardous):** 99,339 observations (61.59% of test set).
- **Model ROC-AUC:** **0.9792**
- **Top Predictive Features (by Split Gain):** Current PM2.5 (51.41%), PM2.5 Lag-1 (16.19%), Monsoon Seasonality (5.50%), Winter Seasonality (3.23%), Regional NCR Mean PM2.5 (3.02%), Upwind NW Smoke Front (2.52%).

#### Comparison against Persistence Baseline:
| Model / Operating Setting | Precision | Recall | F1-Score | False Negatives (Missed Danger Days) | False Positives (False Alarms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline** | 0.9034 | 0.8951 | 0.8992 | 6,498 | 5,927 |
| **XGBClassifier (Threshold = 0.50)** | 0.9192 | 0.9116 | 0.9154 | 5,478 | 4,961 |
| **XGBClassifier (Threshold = 0.25, Safety Trigger)** | **0.8658** | **0.9588** | **0.9099** | **2,552** | **9,205** |

#### Operating Threshold Sweep on Test Set ($N=161,279$):
| Threshold | Precision | Recall | F1-Score | True Negatives | False Positives | False Negatives (Misses) | True Positives |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.10 | 0.7733 | 0.9825 | 0.8654 | 81,503 | 17,836 | 1,087 | 60,853 |
| 0.20 | 0.8472 | 0.9663 | 0.9028 | 88,545 | 10,794 | 2,089 | 59,851 |
| **0.25** | **0.8658** | **0.9588** | **0.9099** | **90,134** | **9,205** | **2,552** | **59,388** |
| 0.28 | 0.8748 | 0.9539 | 0.9126 | 90,883 | 8,456 | 2,858 | 59,082 |
| 0.30 | 0.8800 | 0.9505 | 0.9139 | 91,308 | 8,031 | 3,066 | 58,874 |
| 0.35 | 0.8908 | 0.9420 | 0.9157 | 92,185 | 7,154 | 3,592 | 58,348 |
| 0.40 | 0.9018 | 0.9331 | 0.9172 | 93,043 | 6,296 | 4,141 | 57,799 |
| 0.50 | 0.9192 | 0.9116 | 0.9154 | 94,378 | 4,961 | 5,478 | 56,462 |
| 0.60 | 0.9333 | 0.8844 | 0.9082 | 95,422 | 3,917 | 7,161 | 54,779 |
| 0.70 | 0.9453 | 0.8520 | 0.8963 | 96,288 | 3,051 | 9,166 | 52,774 |
| 0.80 | 0.9575 | 0.7992 | 0.8712 | 97,142 | 2,197 | 12,436 | 49,504 |

**Alert System Decision:** Operating at **Threshold = 0.25** provides the best public health protection profile:
1. Achieves **95.88% recall** on unseen future test data.
2. Reduces un-alerted dangerous days from **6,498 (persistence)** down to **2,552** (a **60.7% reduction in missed hazardous air events**).
3. Maintains strong alert precision (**86.58%** of triggered alerts correspond to true hazardous air days).

---

## 5. Honest Limitations

1. **Category Width vs. MAE Mismatch:**
   CPCB categories are only $\sim 30\ \mu\text{g/m}^3$ wide, while point regression MAE is $34.57\ \mu\text{g/m}^3$. A single-model point regressor cannot reliably separate adjacent discrete AQI categories without substantial label spilling.
2. **Weather Features Are Time-$t$ Observations, Not True 24h Numerical Weather Predictions (NWP):**
   In historical model training (`feature_engineering.py` and `train_model.py`), all weather inputs (`temperature`, `humidity`, `wind_speed`, `rainfall`, `pressure`) represent observations recorded at time $t$. They are not 24-hour-ahead NWP forecast model runs (e.g., GFS / ECMWF). In live serving (`src/predict_live.py`), Open-Meteo is queried solely for real-time advisory context, not as trained model inputs.
3. **Seasonal Test Set Variance & Persistence $R^2$ Inflation:**
   The multi-year holdout test set (May 30 to Dec 31, 2025) spans clean Monsoon months (June–August, PM2.5 $\approx 15–30\ \mu\text{g/m}^3$) through post-monsoon and peak winter smog (November–December, PM2.5 $\approx 300–600\ \mu\text{g/m}^3$). Testing across 7 months spanning annual baseline lows to seasonal highs inflates total variance ($SS_{\text{tot}}$), resulting in a high persistence $R^2$ of **0.6796**. On winter-only test slices, persistence $R^2$ drops to $\approx 0.31$.
4. **Single-Hour Snapshot vs. 24-Hour Regulatory Average:**
   India's official CPCB AQI standard is legally defined on **24-hour rolling average concentrations**. Single-hour PM2.5 readings are an operational early-warning proxy, not statutory regulatory declarations.
5. **Feature Concentration:**
   67.59% of total classifier split importance is concentrated in just two features: current station PM2.5 (51.41%) and PM2.5 lag-1 (16.19%). While macro seasonal (11.48%) and regional spatial features (5.55%) provide the boundary adjustments needed for high recall, the model relies heavily on current pollution regime persistence.

---

## 6. What I'd Improve with More Time

1. **Integrate True Historical 24h NWP Forecast Archives:**
   Ingest historical 24-hour numerical weather prediction reanalysis (e.g., ECMWF ERA5 or IMD WRF 24h-ahead forecasts of boundary layer height, planetary boundary layer ventilation index, and surface wind vectors) to replace static time-$t$ weather observations.
2. **Realign the Target to 24-Hour Rolling Mean PM2.5:**
   Change the forecast target from single-hour $\text{PM2.5}_{t+24}$ to the statutory 24-hour average $\frac{1}{24}\sum_{k=1}^{24}\text{PM2.5}_{t+k}$, smoothing out high-frequency micro-turbulence and matching CPCB's official air quality standard.
3. **Incorporate Real-Time Upwind Satellite Fire Data:**
   Ingest daily MODIS / VIIRS active thermal hotspot counts from Punjab and Haryana to provide direct leading indicators for transboundary agricultural stubble burning plumes during October–November.
4. **Spatial Graph Neural Networks (GNN) / Wind-Directed Advection Modeling:**
   Replace static station coordinate encodings with wind-directed spatial graph convolution, explicitly routing pollution transport vectors down the prevailing atmospheric wind gradient across monitoring stations.

</details>
