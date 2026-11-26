# Project Status & Current-State Inventory
**Repository:** Delhi-NCR Air Quality Forecasting  
**Timestamp:** 2026-09-06  
**Verification Method:** Direct execution of Git commands, static code inspection, and a fresh run of the ML training pipeline.

---

## 1. Git Repository State

### Exact `git log --oneline` Output:
```text
79bd050 checkpoint: honest pipeline after fixing nowcasting, split, and weather-leak bugs
```

### Exact `git status` Output:
```text
On branch main
Your branch is up to date with 'origin/main'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	modified:   README.md
	modified:   notebooks/01_exploratory_data_analysis.ipynb

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	check_winter_stats.py
	models/best_aqi_model.joblib
	models/best_poor_alert_model.joblib
	models/random_forest_aqi.joblib
	models/xgboost_aqi.joblib

no changes added to commit (use "git add" and/or "git commit -a")
```

---

### Uncommitted Changes Diff

#### A. Diff for `README.md`
```diff
diff --git a/README.md b/README.md
index 2cb22c1..076bc92 100644
--- a/README.md
+++ b/README.md
@@ -1,25 +1,24 @@
-# Delhi-NCR Air Quality Forecasting & Alert System
+# Delhi-NCR Air Quality 24-Hour Forecasting & Safety Alert System
 
 ## 1. Problem Statement
-This system provides 24-hour-ahead air quality forecasting for 40 Central Pollution Control Board (CPCB) and DPCC continuous ambient air quality monitoring stations across Delhi-NCR. Its primary objective is to power an early-warning safety alert when ambient air quality is expected to cross into "Poor" or worse (CPCB PM2.5 $> 90\ \mu\text{g/m}^3$, corresponding to AQI sub-index $> 200$), giving residents and municipal agencies actionable advance notice before severe pollution events occur.
+This system provides 24-hour-ahead air quality forecasting for 40 Central Pollution Control Board (CPCB) and Delhi Pollution Control Committee (DPCC) continuous ambient air quality monitoring stations across Delhi-NCR. Its primary objective is to power an early-warning safety alert when ambient air quality is expected to cross into "Poor" or worse (CPCB PM2.5 $> 90\ \mu\text{g/m}^3$, corresponding to AQI sub-index $> 200$), providing municipal agencies and residents actionable advance notice before hazardous pollution episodes develop.
 
 ---
 
-## 2. Architecture
-The end-to-end pipeline consists of five sequential stages:
+## 2. Pipeline Architecture
+The end-to-end pipeline consists of five stages:
 ```
-[CPCB / Data.gov.in Ingestion]     [Open-Meteo 48h NWP API]
-       │                                     │
-       ▼                                     ▼
-[SQLite Storage (aqi_history.db)]  [src/ingestion/fetch_open_meteo_forecast.py]
-       │                                     │
-       └──────────────────┬──────────────────┘
-                          ▼
+[CPCB / Data.gov.in 15-Minute Parquets]
+       │
+       ▼
+[SQLite Storage (data/aqi_history.db - aqi_hourly)]
+       │
+       ▼
 [Feature Engineering (src/analysis/feature_engineering.py)]
-  ├── CPCB AQI sub-index calculations
-  ├── 24h-Ahead Weather Forecast Features (wind_speed_forecast_24h, ventilation_index_forecast_24h, temp_drop_forecast_24h)
-  ├── Transition velocity (pm25_diff_3, pm25_diff_6, pm25_accel) & Stagnation flags
-  ├── Northwest transboundary smoke front & gradient (nw_smoke_front, nw_gradient)
-  ├── Multi-pollutant ratios (PM2.5/PM10) & regional NCR background mean
-  └── 24h-ahead per-station target generation (target_pm25, target_poor_or_worse)
+  ├── CPCB AQI sub-index conversions
+  ├── Temporal lags (1h, 2h, 3h, 6h, 24h) & rolling window statistics (6h, 24h)
+  ├── Multi-station spatial features (NCR regional mean PM2.5, NW stubble smoke front)
+  ├── Short-term pollution dynamics (3h/6h velocity differences, 6h acceleration)
+  ├── Calendar & seasonality indicators (hour, day of week, month, day-of-year harmonics)
+  └── 24-hour per-station target generation (target_pm25 = shift(-24), target_poor_or_worse)
        │
@@ -27,10 +26,10 @@ The end-to-end pipeline consists of five sequential stages:
 [Model Training & Evaluation (src/models/train_model.py)]
-  ├── Primary: Binary Safety Classifier (XGBClassifier) -> 24h Hazard Probability
-  └── Baseline: Continuous Regressor (XGBRegressor) -> Point forecast benchmark
+  ├── Continuous PM2.5 Regressor (XGBRegressor) -> Point forecast baseline benchmark
+  └── Binary Safety Alert Classifier (XGBClassifier) -> 24h Hazard Probability
        │
        ▼
-[Inference & Safety Alert Serving (src/predict_live.py)]
-  ├── Queries Open-Meteo API for live 24h meteorological forecast (wind, temp, ventilation)
-  ├── Evaluates hazard risk probability against safety threshold (25%)
-  └── Dispatches Alert (CRITICAL ALERT / WATCH / NORMAL) with precursor diagnostics
+[Inference & Live Safety Alert Dispatch (src/predict_live.py)]
+  ├── Evaluates station readings against tuned hazard probability threshold (0.25)
+  ├── Displays Open-Meteo 24h NWP meteorological advisory context
+  └── Dispatches Alert Tier ([CRITICAL ALERT] / [WATCH] / [NORMAL]) with precursor diagnostics
 ```
@@ -90,9 +89,9 @@ All metrics below are evaluated on the unseen chronological holdout test set (**
 ### a & b. Continuous Regression Comparison
 
 | Model | MAE ($\mu\text{g/m}^3$) | RMSE ($\mu\text{g/m}^3$) | $R^2$ | 6-Class AQI Category Accuracy |
 | :--- | :---: | :---: | :---: | :---: |
-| **Persistence Baseline** ($\hat{y}_{t+24} = y_t$) | 35.0024 | 60.3863 | 0.6796 | 54.39% |
-| **Trained Regressor** (XGBRegressor, 41 features) | 34.5779 | 60.4285 | 0.6791 | 54.17% |
-| **Delta ($\Delta$)** | **-0.42 µg/m³** | **+0.04 µg/m³** | **-0.0005** | **-0.22%** |
+| **Persistence Baseline** ($\hat{y}_{t+24} = y_t$) | 35.0024 | 60.3863 | 0.6796 | 54.39% |
+| **Trained Regressor** (XGBRegressor, 41 features) | 34.5681 | 60.4317 | 0.6791 | 54.17% |
+| **Delta ($\Delta$)** | **-0.43 µg/m³** | **+0.05 µg/m³** | **-0.0005** | **-0.22%** |
 
 ### c. Real Edge Over Baseline
 The continuous regression model shows **zero practical edge** over simple 24-hour persistence:
@@ -118,5 +117,5 @@ Single-hour PM2.5 readings are subject to high-frequency micro-scale noise. Beca
 
-> **Key Takeaway:** "Poor" is the single hardest category to predict via point regression because its absolute numerical range ($91–120\ \mu\text{g/m}^3$) is narrower than the model's MAE ($34.58\ \mu\text{g/m}^3$).
+> **Critical Finding:** **"Poor" is the weakest category in the regression model**, with a precision of **0.3139** and recall of **0.2290** (F1 = 0.2648). Because "Poor" is a narrow transition band ($91–120\ \mu\text{g/m}^3$), regression predictions frequently overshoot into "Very Poor" ($121–250\ \mu\text{g/m}^3$) or undershoot into "Moderate" ($61–90\ \mu\text{g/m}^3$).
 
 ---
 
@@ -131,11 +130,11 @@ Reframing the problem as a dedicated binary classification task—*"Will air qua
 
-#### Comparison against Persistence:
-| Model / Threshold | Precision | Recall | F1-Score | False Negatives (Misses) | False Positives |
+#### Comparison against Persistence Baseline:
+| Model / Operating Setting | Precision | Recall | F1-Score | False Negatives (Missed Danger Days) | False Positives (False Alarms) |
 | :--- | :---: | :---: | :---: | :---: | :---: |
 | **Persistence Baseline** | 0.9034 | 0.8951 | 0.8992 | 6,498 | 5,927 |
 | **XGBClassifier (Threshold = 0.50)** | 0.9192 | 0.9116 | 0.9154 | 5,478 | 4,961 |
-| **XGBClassifier (Threshold = 0.25, Recommended)** | **0.8658** | **0.9588** | **0.9099** | **2,552** | **9,205** |
+| **XGBClassifier (Threshold = 0.25, Safety Trigger)** | **0.8658** | **0.9588** | **0.9099** | **2,552** | **9,205** |
 
-#### Operating Threshold Trade-Off Analysis:
-| Threshold | Precision | Recall | F1-Score | True Negatives | False Positives | False Negatives (Missed Bad Air) | True Positives |
+#### Operating Threshold Sweep on Test Set ($N=161,279$):
+| Threshold | Precision | Recall | F1-Score | True Negatives | False Positives | False Negatives (Misses) | True Positives |
 | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
@@ -150,7 +149,7 @@ Reframing the problem as a dedicated binary classification task—*"Will air qua
 
-**Operating Decision:** Threshold **`0.25`** provides the best operating point for public health protection:
-1. Achieves **95.88% recall** on unseen future holdout data.
-2. Cuts un-alerted dangerous days from **6,498 (persistence)** down to **2,552** (a **60.7% reduction in missed dangerous air days**).
-3. Preserves high alert reliability with **86.58% precision**.
+**Alert System Decision:** Operating at **Threshold = 0.25** provides the best public health protection profile:
+1. Achieves **95.88% recall** on unseen future test data.
+2. Reduces un-alerted dangerous days from **6,498 (persistence)** down to **2,552** (a **60.7% reduction in missed hazardous air events**).
+3. Maintains strong alert precision (**86.58%** of triggered alerts correspond to true hazardous air days).
 
 ---
 
 ## 5. Honest Limitations
 
 1. **Category Width vs. MAE Mismatch:**
-   CPCB categories are $\sim 30\ \mu\text{g/m}^3$ wide, but the regression MAE is $34.58\ \mu\text{g/m}^3$. Point regression models cannot reliably classify 6 discrete categories without substantial label spilling.
-2. **Current Weather Features Are Not 24h Forecasts:**
-   In `feature_engineering.py` and `train_model.py`, the weather inputs (`temperature`, `humidity`, `wind_speed`, etc.) represent observations at time $t$, not true 24-hour atmospheric forecasts for time $t+24$. While live inference queries Open-Meteo's forecast API, historical training did not train on forecasted numerical weather prediction (NWP) parameters.
-3. **Seasonal Test Set Composition:**
-   The multi-year holdout test set (June 1 to Dec 31, 2025) spans Monsoon (41.3% of rows, where PM2.5 is low and steady at $15–25\ \mu\text{g/m}^3$) and Post-Monsoon before reaching winter (30.0% of rows). Testing on 7 months spanning clean baseline lows to winter highs inflates $SS_{\text{tot}}$, lifting persistence $R^2$ to 0.6792. Evaluating solely on peak winter months yields a persistence $R^2$ of $\approx 0.3091$.
-4. **Single-Hour vs. 24-Hour Regulatory Standard:**
-   India's official CPCB AQI standard specifies **24-hour rolling average concentrations**. Single-hour PM2.5 predictions are an operational proxy, not official regulatory AQI declarations.
+   CPCB categories are only $\sim 30\ \mu\text{g/m}^3$ wide, while point regression MAE is $34.57\ \mu\text{g/m}^3$. A single-model point regressor cannot reliably separate adjacent discrete AQI categories without substantial label spilling.
+2. **Weather Features Are Time-$t$ Observations, Not True 24h Numerical Weather Predictions (NWP):**
+   In historical model training (`feature_engineering.py` and `train_model.py`), all weather inputs (`temperature`, `humidity`, `wind_speed`, `rainfall`, `pressure`) represent observations recorded at time $t$. They are not 24-hour-ahead NWP forecast model runs (e.g., GFS / ECMWF). In live serving (`src/predict_live.py`), Open-Meteo is queried solely for real-time advisory context, not as trained model inputs.
+3. **Seasonal Test Set Variance & Persistence $R^2$ Inflation:**
+   The multi-year holdout test set (May 30 to Dec 31, 2025) spans clean Monsoon months (June–August, PM2.5 $\approx 15–30\ \mu\text{g/m}^3$) through post-monsoon and peak winter smog (November–December, PM2.5 $\approx 300–600\ \mu\text{g/m}^3$). Testing across 7 months spanning annual baseline lows to seasonal highs inflates total variance ($SS_{\text{tot}}$), resulting in a high persistence $R^2$ of **0.6796**. On winter-only test slices, persistence $R^2$ drops to $\approx 0.31$.
+4. **Single-Hour Snapshot vs. 24-Hour Regulatory Average:**
+   India's official CPCB AQI standard is legally defined on **24-hour rolling average concentrations**. Single-hour PM2.5 readings are an operational early-warning proxy, not statutory regulatory declarations.
+5. **Feature Concentration:**
+   67.59% of total classifier split importance is concentrated in just two features: current station PM2.5 (51.41%) and PM2.5 lag-1 (16.19%). While macro seasonal (11.48%) and regional spatial features (5.55%) provide the boundary adjustments needed for high recall, the model relies heavily on current pollution regime persistence.
 
 ---
 
 ## 6. What I'd Improve with More Time
 
-1. **Train on True 24h Numerical Weather Prediction (NWP) Forecasts:**
-   Extract historical ECMWF / GFS 24h-ahead weather forecasts (boundary layer height, surface wind speed, temperature inversions) to replace static weather observations.
-2. **Realign the Target to 24h Rolling Mean PM2.5:**
-   Change the regression target from single-hour $\text{PM2.5}_{t+24}$ to the 24-hour average $\frac{1}{24}\sum_{k=1}^{24}\text{PM2.5}_{t+k}$, matching CPCB's statutory standard and eliminating high-frequency sensor noise.
-3. **Incorporate Wind-Directed Spatial Advection:**
-   Construct dynamic upwind station features (e.g., measuring Northwest border stations like Bawana/Narela when wind direction is $300^\circ–330^\circ$) to capture incoming agricultural stubble smoke plumes before they reach central Delhi.
-4. **Implement Asymmetric Pinball Loss (Quantile Regression):**
-   Train an 80th-percentile pinball loss regressor in XGBoost to explicitly penalize under-predictions heavier than over-predictions for conservative public health planning.
-5. **Add Event Proximity Calendars:**
-   Explicitly encode days-to-Diwali and crop-residue-burning seasonal windows rather than relying solely on month and day-of-week integer indicators.
+1. **Integrate True Historical 24h NWP Forecast Archives:**
+   Ingest historical 24-hour numerical weather prediction reanalysis (e.g., ECMWF ERA5 or IMD WRF 24h-ahead forecasts of boundary layer height, planetary boundary layer ventilation index, and surface wind vectors) to replace static time-$t$ weather observations.
+2. **Realign the Target to 24-Hour Rolling Mean PM2.5:**
+   Change the forecast target from single-hour $\text{PM2.5}_{t+24}$ to the statutory 24-hour average $\frac{1}{24}\sum_{k=1}^{24}\text{PM2.5}_{t+k}$, smoothing out high-frequency micro-turbulence and matching CPCB's official air quality standard.
+3. **Incorporate Real-Time Upwind Satellite Fire Data:**
+   Ingest daily MODIS / VIIRS active thermal hotspot counts from Punjab and Haryana to provide direct leading indicators for transboundary agricultural stubble burning plumes during October–November.
+4. **Spatial Graph Neural Networks (GNN) / Wind-Directed Advection Modeling:**
+   Replace static station coordinate encodings with wind-directed spatial graph convolution, explicitly routing pollution transport vectors down the prevailing atmospheric wind gradient across monitoring stations.
```

#### B. Diff for `notebooks/01_exploratory_data_analysis.ipynb`
```diff
diff --git a/notebooks/01_exploratory_data_analysis.ipynb b/notebooks/01_exploratory_data_analysis.ipynb
index 4d1f03c..7a766b6 100644
--- a/notebooks/01_exploratory_data_analysis.ipynb
+++ b/notebooks/01_exploratory_data_analysis.ipynb
@@ -19,14 +19,18 @@
   },
   {
    "cell_type": "code",
-   "execution_count": 24,
+   "execution_count": 1,
    "metadata": {},
    "outputs": [
     {
-     "name": "stdout",
-     "output_type": "stream",
-     "text": [
-      "Database path: c:\\Users\\tanis\\OneDrive\\Documents\\Delhi-NCR Air Quality Forecasting\\data\\aqi_history.db\n"
+     "ename": "ModuleNotFoundError",
+     "evalue": "No module named 'pandas'",
+     "output_type": "error",
+     "traceback": [
+      "\u001b[31m---------------------------------------------------------------------------\u001b[39m",
+      "\u001b[31mModuleNotFoundError\u001b[39m                       Traceback (most recent call last)",
+      "\u001b[36mCell\u001b[39m\u001b[36m \u001b[39m\u001b[32mIn[1]\u001b[39m\u001b[32m, line 3\u001b[39m\n\u001b[32m      1\u001b[39m \u001b[38;5;28;01mimport\u001b[39;00m os\n\u001b[32m      2\u001b[39m \u001b[38;5;28;01mimport\u001b[39;00m sqlite3\n\u001b[32m----> \u001b[39m\u001b[32m3\u001b[39m \u001b[38;5;28;01mimport\u001b[39;00m pandas \u001b[38;5;28;01mas\u001b[39;00m pd\n\u001b[32m      4\u001b[39m \u001b[38;5;28;01mimport\u001b[39;00m matplotlib.pyplot \u001b[38;5;28;01mas\u001b[39;00m plt\n\u001b[32m      5\u001b[39m \u001b[38;5;28;01mimport\u001b[39;00m seaborn \u001b[38;5;28;01mas\u001b[39;00m sns\n\u001b[32m      6\u001b[39m \n",
+      "\u001b[31mModuleNotFoundError\u001b[39m: No module named 'pandas'"
      ]
     }
    ],
@@ -1041,7 +1045,7 @@
  ],
  "metadata": {
   "kernelspec": {
-   "display_name": "venv",
+   "display_name": ".venv",
    "language": "python",
    "name": "python3"
   },
@@ -1055,7 +1059,7 @@
    "name": "python",
    "nbconvert_exporter": "python",
    "pygments_lexer": "ipython3",
-   "version": "3.13.3"
+   "version": "3.14.3"
   }
  },
  "nbformat": 4,
```

#### C. Untracked Files
1. `check_winter_stats.py`: Quick verification script added by user to measure seasonal baseline rates for months 10, 11, 12, 1.
2. `models/*.joblib`: Generated model binary files (`best_aqi_model.joblib`, `best_poor_alert_model.joblib`, `random_forest_aqi.joblib`, `xgboost_aqi.joblib`).

---

## 2. Configuration & Verbatim Code Inspection (`train_model.py`)

### A. Exact Target Column Name Currently in Use:
Verbatim from line 85 of [src/models/train_model.py](file:///c:/Users/tanis/OneDrive/Documents/Delhi-NCR%20Air%20Quality%20Forecasting/src/models/train_model.py#L85):
```python
TARGET_REG = "target_pm25"
```
And the derived binary classification target (lines 100-101):
```python
df["target_poor_or_worse"] = (df[TARGET_REG] > POOR_THRESHOLD).astype(int)
df["current_poor_or_worse"] = (df["pm25"] > POOR_THRESHOLD).astype(int)
```

### B. Complete Verbatim `CANDIDATE_FEATURES` List (41 Features):
Verbatim from lines 35-84 of [src/models/train_model.py](file:///c:/Users/tanis/OneDrive/Documents/Delhi-NCR%20Air%20Quality%20Forecasting/src/models/train_model.py#L35-L84):
```python
CANDIDATE_FEATURES = [
    # Current readings known at prediction time (t)
    "aqi",
    "pm25",
    "pm10",
    "no2",
    "pm_ratio",
    "ncr_mean_pm25",
    "station_vs_ncr_ratio",
    # Transition velocity & upwind front features (past t to t-6)
    "nw_smoke_front",
    "nw_gradient",
    "pm25_diff_3",
    "pm25_diff_6",
    "pm25_accel",
    "temp_drop_6",
    "is_stagnant",
    # Shifted AQI history (past lags t-1 to t-24)
    "aqi_lag_1",
    "aqi_lag_2",
    "aqi_lag_3",
    "aqi_lag_6",
    "aqi_lag_24",
    "aqi_rolling_mean_6",
    "aqi_rolling_mean_24",
    "aqi_rolling_std_6",
    # Shifted pollutant history (past lags)
    "pm25_lag_1",
    "pm10_lag_1",
    "no2_lag_1",
    # Calendar & Seasonality (known at time t)
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "doy_sin",
    "doy_cos",
    "is_monsoon",
    "is_winter",
    # Current weather parameters at time t
    "temperature",
    "humidity",
    "wind_speed",
    "wind_direction",
    "rainfall",
    "pressure",
    "solar_radiation",
    # Station identifier
    "station_encoded",
]
```

### C. Thresholds & Forecasting Horizon Values:
* `POOR_THRESHOLD` (line 86 of `train_model.py`):
  ```python
  POOR_THRESHOLD = 90.0  # CPCB PM2.5 threshold for Poor or worse (>90 ug/m3 -> Sub-index >200)
  ```
* `HORIZON_HOURS` (line 16 of [src/analysis/feature_engineering.py](file:///c:/Users/tanis/OneDrive/Documents/Delhi-NCR%20Air%20Quality%20Forecasting/src/analysis/feature_engineering.py#L16)):
  ```python
  HORIZON_HOURS = 24
  ```
  *(Note: `HORIZON_HOURS` is physically defined during feature engineering where `target_pm25` is constructed as `df.groupby("station")["pm25"].shift(-HORIZON_HOURS)`; `train_model.py` directly consumes that pre-computed column.)*

### D. Exact Split Logic (Verbatim lines 108-136):
```python
def split_data(df: pd.DataFrame, features: list):
    train_cutoff = df["timestamp"].quantile(TRAIN_RATIO)
    val_cutoff = df["timestamp"].quantile(TRAIN_RATIO + VAL_RATIO)

    train_df = (
        df[df["timestamp"] < train_cutoff]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    val_df = (
        df[
            (df["timestamp"] >= train_cutoff) & (df["timestamp"] < val_cutoff)
        ]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    test_df = (
        df[df["timestamp"] >= val_cutoff]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    print(
        f"Temporal splits: Train={len(train_df)} ({train_df['timestamp'].min()} to {train_df['timestamp'].max()}), "
        f"Val={len(val_df)} ({val_df['timestamp'].min()} to {val_df['timestamp'].max()}), "
        f"Test={len(test_df)} ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})"
    )
    return train_df, val_df, test_df
```
**Confirmation:** This logic is **strictly timestamp-value-based** (`df["timestamp"].quantile(...)` and `df["timestamp"] < train_cutoff`). It operates globally across all 40 stations concurrently at the exact same chronological boundary, and is **not positional** (`.iloc[:n]`).

---

## 3. Fresh Execution of `train_model.py`

Executed directly in `.\venv\Scripts\python.exe src/models/train_model.py`:

```text
Loaded 1075166 records. Features active: 41
Temporal splits: Train=752591 (2022-01-02 05:00:00 to 2024-11-07 08:00:00), Val=161296 (2024-11-07 09:00:00 to 2025-05-30 22:00:00), Test=161279 (2025-05-30 23:00:00 to 2025-12-31 05:00:00)

======================================================================
1. TRAINING CONTINUOUS PM2.5 REGRESSION MODEL
======================================================================
Fitting XGBRegressor on Train+Val ...
Regression training complete.

-- Persistence Baseline (Continuous PM2.5 Regression) --
  MAE          : 35.0024
  RMSE         : 60.3863
  R²           : 0.6796
  AQI Cat Acc  : 0.5439 (54.39%)

  Per-Category Classification Report (Persistence Baseline):
              precision    recall  f1-score   support

        Good     0.6422    0.6330    0.6376     41257
    Moderate     0.3203    0.3281    0.3242     16911
        Poor     0.2700    0.2744    0.2722     11280
Satisfactory     0.5152    0.5246    0.5198     41171
      Severe     0.6188    0.5958    0.6071     17341
   Very Poor     0.6333    0.6313    0.6323     33319

    accuracy                         0.5439    161279
   macro avg     0.5000    0.4979    0.4989    161279
weighted avg     0.5457    0.5439    0.5447    161279


-- Unseen Chronological Holdout Test Set (Continuous PM2.5 Regression) --
  MAE          : 34.5681
  RMSE         : 60.4317
  R²           : 0.6791
  AQI Cat Acc  : 0.5417 (54.17%)
  Top 5 Features:
    pm25                  : 0.4358
    pm25_lag_1            : 0.1730
    aqi_lag_1             : 0.0936
    aqi                   : 0.0754
    is_winter             : 0.0411

  Per-Category Classification Report (from Regression Predictions):
              precision    recall  f1-score   support

        Good     0.7381    0.4157    0.5318     41257
    Moderate     0.3656    0.3250    0.3441     16911
        Poor     0.3139    0.2290    0.2648     11280
Satisfactory     0.4910    0.7164    0.5827     41171
      Severe     0.5992    0.4779    0.5317     17341
   Very Poor     0.5958    0.7310    0.6565     33319

    accuracy                         0.5417    161279
   macro avg     0.5173    0.4825    0.4853    161279
weighted avg     0.5620    0.5417    0.5322    161279


======================================================================
2. TRAINING BINARY 'POOR OR WORSE' SAFETY ALERT CLASSIFIER
======================================================================
Fitting Precision-Tuned XGBClassifier on Train+Val ...
Classification training complete.

-- Persistence Baseline (Binary Poor-or-Worse Alert) --
  Precision    : 0.9034
  Recall       : 0.8951
  F1-Score     : 0.8992
  Confusion Matrix: TN=93412, FP=5927, FN=6498, TP=55442

-- Unseen Chronological Holdout Test Set (Binary Poor-or-Worse Alert) --
  ROC-AUC      : 0.9792
  Top 5 Features for Safety Alert Classifier:
    pm25                  : 0.5141
    pm25_lag_1            : 0.1619
    is_monsoon            : 0.0550
    is_winter             : 0.0323
    ncr_mean_pm25         : 0.0302

  Threshold Evaluation Sweep:
   Threshold |  Precision |     Recall |   F1-Score |       TN |       FP |  FN (Misses) |       TP
  -----------------------------------------------------------------------------------------------
        0.10 |     0.7733 |     0.9825 |     0.8654 |    81503 |    17836 |         1087 |    60853
        0.20 |     0.8472 |     0.9663 |     0.9028 |    88545 |    10794 |         2089 |    59851
        0.25 |     0.8658 |     0.9588 |     0.9099 |    90134 |     9205 |         2552 |    59388
        0.28 |     0.8748 |     0.9539 |     0.9126 |    90883 |     8456 |         2858 |    59082
        0.30 |     0.8800 |     0.9505 |     0.9139 |    91308 |     8031 |         3066 |    58874
        0.35 |     0.8908 |     0.9420 |     0.9157 |    92185 |     7154 |         3592 |    58348
        0.40 |     0.9018 |     0.9331 |     0.9172 |    93043 |     6296 |         4141 |    57799
        0.50 |     0.9192 |     0.9116 |     0.9154 |    94378 |     4961 |         5478 |    56462
        0.60 |     0.9333 |     0.8844 |     0.9082 |    95422 |     3917 |         7161 |    54779
        0.70 |     0.9453 |     0.8520 |     0.8963 |    96288 |     3051 |         9166 |    52774
        0.80 |     0.9575 |     0.7992 |     0.8712 |    97142 |     2197 |        12436 |    49504

Saved regression model to: C:\Users\tanis\OneDrive\Documents\Delhi-NCR Air Quality Forecasting\models\best_aqi_model.joblib
Saved classification model to: C:\Users\tanis\OneDrive\Documents\Delhi-NCR Air Quality Forecasting\models\best_poor_alert_model.joblib
```

---

### Full Feature Importance List for the Safety Alert Classifier (All 41 Features):

Extracted directly from the trained `XGBClassifier` (`models/best_poor_alert_model.joblib`):

| Rank | Feature Name | Split Gain Importance | Cumulative |
| :---: | :--- | :---: | :---: |
| 1 | `pm25` | 0.514074 (51.41%) | 51.41% |
| 2 | `pm25_lag_1` | 0.161865 (16.19%) | 67.59% |
| 3 | `is_monsoon` | 0.054986 (5.50%) | 73.09% |
| 4 | `is_winter` | 0.032267 (3.23%) | 76.32% |
| 5 | `ncr_mean_pm25` | 0.030242 (3.02%) | 79.34% |
| 6 | `aqi` | 0.026566 (2.66%) | 82.00% |
| 7 | `nw_smoke_front` | 0.025229 (2.52%) | 84.52% |
| 8 | `aqi_lag_24` | 0.019377 (1.94%) | 86.46% |
| 9 | `aqi_lag_1` | 0.014146 (1.41%) | 87.87% |
| 10 | `doy_cos` | 0.010318 (1.03%) | 88.91% |
| 11 | `hour` | 0.009213 (0.92%) | 89.83% |
| 12 | `doy_sin` | 0.008844 (0.88%) | 90.71% |
| 13 | `month` | 0.008366 (0.84%) | 91.55% |
| 14 | `pm_ratio` | 0.006856 (0.69%) | 92.23% |
| 15 | `temperature` | 0.005968 (0.60%) | 92.83% |
| 16 | `day_of_week` | 0.005830 (0.58%) | 93.41% |
| 17 | `station_vs_ncr_ratio` | 0.004990 (0.50%) | 93.91% |
| 18 | `aqi_rolling_mean_24` | 0.004767 (0.48%) | 94.39% |
| 19 | `is_weekend` | 0.004719 (0.47%) | 94.86% |
| 20 | `station_encoded` | 0.003858 (0.39%) | 95.25% |
| 21 | `is_stagnant` | 0.003598 (0.36%) | 95.61% |
| 22 | `nw_gradient` | 0.003405 (0.34%) | 95.95% |
| 23 | `wind_speed` | 0.003262 (0.33%) | 96.27% |
| 24 | `rainfall` | 0.003134 (0.31%) | 96.59% |
| 25 | `humidity` | 0.002957 (0.30%) | 96.88% |
| 26 | `temp_drop_6` | 0.002921 (0.29%) | 97.18% |
| 27 | `pm25_diff_3` | 0.002830 (0.28%) | 97.46% |
| 28 | `pressure` | 0.002676 (0.27%) | 97.73% |
| 29 | `aqi_rolling_std_6` | 0.002670 (0.27%) | 97.99% |
| 30 | `solar_radiation` | 0.002582 (0.26%) | 98.25% |
| 31 | `aqi_rolling_mean_6` | 0.002302 (0.23%) | 98.48% |
| 32 | `pm10_lag_1` | 0.002000 (0.20%) | 98.68% |
| 33 | `aqi_lag_2` | 0.001731 (0.17%) | 98.86% |
| 34 | `pm25_diff_6` | 0.001724 (0.17%) | 99.03% |
| 35 | `pm25_accel` | 0.001633 (0.16%) | 99.19% |
| 36 | `pm10` | 0.001632 (0.16%) | 99.36% |
| 37 | `aqi_lag_6` | 0.001500 (0.15%) | 99.51% |
| 38 | `aqi_lag_3` | 0.001335 (0.13%) | 99.64% |
| 39 | `wind_direction` | 0.001224 (0.12%) | 99.76% |
| 40 | `no2` | 0.001203 (0.12%) | 99.88% |
| 41 | `no2_lag_1` | 0.001200 (0.12%) | 100.00% |

---

## 4. Verification of `src/predict_live.py`

Inspection of [src/predict_live.py](file:///c:/Users/tanis/OneDrive/Documents/Delhi-NCR%20Air%20Quality%20Forecasting/src/predict_live.py):

* **Point Forecast / `predicted_pm25` Status:** 
  Confirmed **completely removed**. There is **no** `predicted_pm25` variable, **no** point regression inference call, and **no** point-forecast output line displayed to the user.
* **Current Display Fields:**
  The output only presents:
  1. `Current Ambient PM2.5` (`row['pm25']` at time $t$) and current sub-index AQI.
  2. `Hazard Probability (24h)` evaluated from `clf.predict_proba(X)[0, 1]`.
  3. Alert Tier Badge (`[CRITICAL ALERT]` at $\ge 0.25$, `[WATCH]` at $\ge 0.15$, or `[NORMAL]`) and corresponding public health actions.
  4. Open-Meteo 24h NWP Meteorological Advisory context (forecasted wind speed, direction, temperature, humidity, rainfall).
  5. Station Precursor Signals (`pm25_diff_3`, `ncr_mean_pm25`, `nw_smoke_front`, `nw_gradient`).
* **Residual Unused Line:**
  Line 27 still defines an unused path constant: `REG_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_aqi_model.joblib")`, but this variable is never loaded, called, or referenced anywhere in the script.

---

## 5. Chronological Bug History & Fixes

1. **Nowcasting Target Bug**
   * *What it was:* The training pipeline previously assigned the prediction target to current-time pollution observations ($t=0$, `shift(0)`) rather than future readings, causing the model to trivially memorize present sensor values and produce artificially inflated metrics.
   * *What the fix was:* The target calculation in `feature_engineering.py` was explicitly shifted per station by negative 24 hours (`df.groupby("station")["pm25"].shift(-24)`), forcing the model to forecast true 24-hour-ahead concentrations ($t+24$).

2. **Station-Major Split Bug**
   * *What it was:* Data partitioning previously sorted records by station name first before time (`sort_values(["station", "timestamp"])`) and applied positional slicing, which caused future timestamps from earlier alphabetical stations to leak into the training set while past timestamps from later stations were evaluated in the test set.
   * *What the fix was:* Positional slicing was replaced with a global timestamp-quantile cutoff (`df["timestamp"].quantile(...)`), strictly segmenting all 40 stations concurrently at the exact same chronological boundaries across the entire NCR network.

3. **Weather-Forecast Leak Bug**
   * *What it was:* Historical training matrices inadvertently incorporated future-shifted or forward-filled weather readings as input features, granting the model access to atmospheric parameters that would not yet exist at prediction time $t$.
   * *What the fix was:* All weather features in `feature_engineering.py` and `train_model.py` were strictly constrained to contemporaneous observations recorded at or before time $t$ (with weather gap-fills derived exclusively from training-period medians), while future weather is queried strictly at live inference time from external NWP forecasts for diagnostic display only.

---

## 6. Multi-Tier Alert System Implementation (Branch: `experiment/multi-tier-alert-system`)

**Implementation Date:** 2026-09-06 / 2026-09-07  
**Branch:** `experiment/multi-tier-alert-system` (branched from commit `06b4c8f`)  
**Motivation:** While the single binary classifier (`PM2.5 > 90`) is statistically sound, 76–78% of Delhi winter days already cross into Poor-or-worse. A single binary alert fires nearly every day for 3 months, creating alert fatigue. The solution extends the proven classifier approach into a multi-tier escalation system without modifying underlying features or `feature_engineering.py`.

### A. Git Status on Branch
```text
On branch experiment/multi-tier-alert-system
Changes not staged for commit:
	modified:   src/models/train_model.py
	modified:   src/predict_live.py
```

### B. Target Definitions & CPCB Breakpoints (`src/models/train_model.py`)
```python
POOR_THRESHOLD = 90.0        # CPCB PM2.5 threshold for Poor or worse (>90 ug/m3 -> Sub-index >200)
VERY_POOR_THRESHOLD = 120.0  # CPCB PM2.5 threshold for Very Poor or worse (>120 ug/m3 -> Sub-index >300)
SEVERE_THRESHOLD = 250.0     # CPCB PM2.5 threshold for Severe (>250 ug/m3 -> Sub-index >400)
```
Targets added in `load_features()`:
* `target_poor_or_worse = (df[TARGET_REG] > 90).astype(int)`
* `target_very_poor_or_worse = (df[TARGET_REG] > 120).astype(int)`
* `target_severe = (df[TARGET_REG] > 250).astype(int)`
And corresponding time-$t$ persistence baselines (`current_poor_or_worse`, `current_very_poor_or_worse`, `current_severe`).

### C. Independent Classifier Training & Saved Artifacts
Three independent `XGBClassifier` models were fitted with identical hyperparameters (`n_estimators=350, max_depth=6, learning_rate=0.04, subsample=0.85, colsample_bytree=0.75, min_child_weight=8, reg_alpha=1.5, reg_lambda=4.0`):
* `models/best_alert_poor.joblib` (and `models/best_poor_alert_model.joblib`)
* `models/best_alert_verypoor.joblib`
* `models/best_alert_severe.joblib`

### D. Holdout Test Set Evaluation ($N=161,279$)

#### 1. Tier 1: "Poor or Worse" Classifier ($\text{PM2.5} > 90\ \mu\text{g/m}^3$)
* **Positive Class Support:** 61,940 / 161,279 (38.41%)
* **Persistence Baseline:** Precision: 0.9034 | Recall: 0.8951 | F1: 0.8992 | (TN=93,412, FP=5,927, FN=6,498, TP=55,442)
* **Model ROC-AUC:** **0.9792**
* **Top 5 Features:** `pm25` (0.5141), `pm25_lag_1` (0.1619), `is_monsoon` (0.0550), `is_winter` (0.0323), `ncr_mean_pm25` (0.0302)
* **Threshold Sweep:**
```text
 Threshold |  Precision |     Recall |   F1-Score |       TN |       FP |  FN (Misses) |       TP
-----------------------------------------------------------------------------------------------
      0.10 |     0.7733 |     0.9825 |     0.8654 |    81503 |    17836 |         1087 |    60853
      0.20 |     0.8472 |     0.9663 |     0.9028 |    88545 |    10794 |         2089 |    59851
      0.25 |     0.8658 |     0.9588 |     0.9099 |    90134 |     9205 |         2552 |    59388
      0.28 |     0.8748 |     0.9539 |     0.9126 |    90883 |     8456 |         2858 |    59082
      0.30 |     0.8800 |     0.9505 |     0.9139 |    91308 |     8031 |         3066 |    58874
      0.35 |     0.8908 |     0.9420 |     0.9157 |    92185 |     7154 |         3592 |    58348
      0.40 |     0.9018 |     0.9331 |     0.9172 |    93043 |     6296 |         4141 |    57799
      0.50 |     0.9192 |     0.9116 |     0.9154 |    94378 |     4961 |         5478 |    56462
      0.60 |     0.9333 |     0.8844 |     0.9082 |    95422 |     3917 |         7161 |    54779
      0.70 |     0.9453 |     0.8520 |     0.8963 |    96288 |     3051 |         9166 |    52774
      0.80 |     0.9575 |     0.7992 |     0.8712 |    97142 |     2197 |        12436 |    49504
```

#### 2. Tier 2: "Very Poor or Worse" Classifier ($\text{PM2.5} > 120\ \mu\text{g/m}^3$)
* **Positive Class Support:** 50,660 / 161,279 (31.41%)
* **Persistence Baseline:** Precision: 0.8678 | Recall: 0.8549 | F1: 0.8613 | (TN=104,019, FP=6,600, FN=7,353, TP=43,307)
* **Model ROC-AUC:** **0.9739**
* **Top 5 Features:** `pm25` (0.4263), `pm25_lag_1` (0.1693), `aqi` (0.1097), `aqi_lag_1` (0.1003), `is_monsoon` (0.0273)
* **Threshold Sweep:**
```text
 Threshold |  Precision |     Recall |   F1-Score |       TN |       FP |  FN (Misses) |       TP
-----------------------------------------------------------------------------------------------
      0.10 |     0.7423 |     0.9865 |     0.8472 |    93272 |    17347 |          686 |    49974
      0.20 |     0.7907 |     0.9707 |     0.8715 |    97606 |    13013 |         1486 |    49174
      0.25 |     0.8050 |     0.9616 |     0.8764 |    98818 |    11801 |         1944 |    48716
      0.28 |     0.8124 |     0.9554 |     0.8781 |    99439 |    11180 |         2258 |    48402
      0.30 |     0.8166 |     0.9515 |     0.8789 |    99792 |    10827 |         2455 |    48205
      0.35 |     0.8280 |     0.9419 |     0.8813 |   100708 |     9911 |         2945 |    47715
      0.40 |     0.8382 |     0.9305 |     0.8820 |   101522 |     9097 |         3520 |    47140
      0.50 |     0.8593 |     0.9014 |     0.8798 |   103140 |     7479 |         4995 |    45665
      0.60 |     0.8806 |     0.8609 |     0.8706 |   104703 |     5916 |         7045 |    43615
      0.70 |     0.9022 |     0.7992 |     0.8476 |   106230 |     4389 |        10175 |    40485
      0.80 |     0.9241 |     0.6999 |     0.7966 |   107708 |     2911 |        15201 |    35459
```

#### 3. Tier 3: "Severe" Classifier ($\text{PM2.5} > 250\ \mu\text{g/m}^3$)
* **Positive Class Support:** 17,341 / 161,279 (10.75%)
* **Persistence Baseline:** Precision: 0.6188 | Recall: 0.5958 | F1: 0.6071 | (TN=137,575, FP=6,363, FN=7,010, TP=10,331)
* **Model ROC-AUC:** **0.9406**
* **Top 5 Features:** `pm25` (0.4385), `aqi` (0.1424), `pm25_lag_1` (0.1342), `aqi_lag_1` (0.0873), `is_winter` (0.0211)
* **Threshold Sweep:**
```text
 Threshold |  Precision |     Recall |   F1-Score |       TN |       FP |  FN (Misses) |       TP
-----------------------------------------------------------------------------------------------
      0.10 |     0.4105 |     0.8913 |     0.5621 |   121742 |    22196 |         1885 |    15456
      0.20 |     0.4946 |     0.7651 |     0.6008 |   130382 |    13556 |         4073 |    13268
      0.25 |     0.5309 |     0.7101 |     0.6076 |   133060 |    10878 |         5028 |    12313
      0.28 |     0.5495 |     0.6764 |     0.6064 |   134323 |     9615 |         5612 |    11729
      0.30 |     0.5621 |     0.6550 |     0.6050 |   135089 |     8849 |         5983 |    11358
      0.35 |     0.5906 |     0.6067 |     0.5985 |   136645 |     7293 |         6821 |    10520
      0.40 |     0.6161 |     0.5585 |     0.5859 |   137902 |     6036 |         7656 |     9685
      0.50 |     0.6615 |     0.4664 |     0.5471 |   139799 |     4139 |         9253 |     8088
      0.60 |     0.7098 |     0.3730 |     0.4890 |   141293 |     2645 |        10873 |     6468
      0.70 |     0.7659 |     0.2733 |     0.4029 |   142489 |     1449 |        12601 |     4740
      0.80 |     0.8457 |     0.1603 |     0.2695 |   143431 |      507 |        14562 |     2779
```

#### Performance Drop in Severe:
The Severe classifier performs meaningfully worse than Poor and Very Poor:
1. Low positive prevalence (10.75%) increases false alarm rates.
2. At threshold 0.50, recall collapses to 46.64% (missing 9,253 severe events).
3. At threshold 0.25, recall recovers to 71.01% (cutting misses from 7,010 down to 5,028), but precision drops to 53.09% (10,878 false alarms).
4. Extreme spikes $>250\ \mu\text{g/m}^3$ represent acute episodic anomalies (shallow nocturnal boundary layers, fire plumes, firecrackers) that are far harder to predict 24h out than broad regional background pollution.

### E. Combined Multi-Tier Decision System (`get_alert_tier`)
```python
TIER_THRESHOLDS = {
    "poor": 0.25,
    "very_poor": 0.25,
    "severe": 0.25,
}

def get_alert_tier(clf_poor, clf_verypoor, clf_severe, X, thresholds=None):
    if thresholds is None:
        thresholds = TIER_THRESHOLDS
    th_poor, th_vp, th_sev = thresholds["poor"], thresholds["very_poor"], thresholds["severe"]
    pred_poor = clf_poor.predict_proba(X)[:, 1] >= th_poor
    pred_vp = clf_verypoor.predict_proba(X)[:, 1] >= th_vp
    pred_sev = clf_severe.predict_proba(X)[:, 1] >= th_sev
    tiers = np.where(pred_sev, "SEVERE",
            np.where(pred_vp, "VERY_POOR",
            np.where(pred_poor, "POOR", "NORMAL")))
    return str(tiers[0]) if len(X) == 1 else pd.Series(tiers, index=X.index)
```

#### Inconsistency Audit on Holdout Test Set ($N=161,279$):
```text
  Operating Decision Thresholds : Poor=0.25, VeryPoor=0.25, Severe=0.25
  Total Test Observations       : 161,279
  Total Inconsistent Rows       : 3 (0.002%)
    - Severe=True, VeryPoor=False: 0
    - Severe=True, Poor=False    : 0
    - VeryPoor=True, Poor=False  : 3
  Resolution: Inconsistent predictions resolved to the highest active tier.
```
*Inconsistencies occurred in only **3 rows out of 161,279** (a **0.002% rate**), confirming near-perfect monotonic hierarchy across independent models.*

#### Multi-Tier Classification Performance:
```text
  Multi-Tier Classification Report (Persistence Baseline):
              precision    recall  f1-score   support

      NORMAL     0.9350    0.9403    0.9376     99339
        POOR     0.2700    0.2744    0.2722     11280
   VERY_POOR     0.6333    0.6313    0.6323     33319
      SEVERE     0.6188    0.5958    0.6071     17341

    accuracy                         0.7929    161279
   macro avg     0.6143    0.6104    0.6123    161279
weighted avg     0.7922    0.7929    0.7925    161279

  Multi-Tier Classification Report (Trained Multi-Tier System):
              precision    recall  f1-score   support

      NORMAL     0.9725    0.9073    0.9388     99339
        POOR     0.2630    0.1884    0.2195     11280
   VERY_POOR     0.5844    0.6546    0.6175     33319
      SEVERE     0.5309    0.7101    0.6076     17341

    accuracy                         0.7836    161279
   macro avg     0.5877    0.6151    0.5958    161279
weighted avg     0.7952    0.7836    0.7865    161279
```

### F. Live Multi-Tier Inference (`src/predict_live.py`)
Updated to evaluate all three tier probabilities and dispatch tiered alert badges:
* `[NORMAL]`
* `[WATCH: POOR]`
* `[ALERT: VERY POOR]`
* `[CRITICAL: SEVERE]`

#### Actual Verified Live Output:
```text
Querying Open-Meteo Forecast API for (28.647, 77.316) ...

===========================================================================
  DELHI-NCR 24-HOUR MULTI-TIER AIR QUALITY & METEOROLOGICAL ALERT SYSTEM
===========================================================================
  Station                     : Anand Vihar, Delhi - DPCC
  Latest Station Reading      : 2025-12-31 05:00
  Current Ambient PM2.5       : 613.5 ug/m3 (Sub-Index AQI: 500)
---------------------------------------------------------------------------
  Alert Tier Badge            : [CRITICAL: SEVERE]
  Alert Status                : CRITICAL EMERGENCY: SEVERE AIR QUALITY EXPECTED (PM2.5 > 250 ug/m3)
  Public Health Action        : Cease all outdoor exertion. Sensitive groups and general public must remain indoors with air filtration. Wear N95 masks outdoors.
---------------------------------------------------------------------------
  24-Hour Tier Hazard Probabilities:
    * Poor or Worse (>90 ug/m3)      : 99.2% (Trigger: 25%)
    * Very Poor or Worse (>120 ug/m3) : 98.4% (Trigger: 25%)
    * Severe (>250 ug/m3)             : 27.0% (Trigger: 25%)
---------------------------------------------------------------------------
  Open-Meteo 24h Meteorological Advisory (NWP):
    * Forecasted 24h Wind     : 0.47 m/s (Direction: 347 deg)
    * Forecasted Temperature  : 24.6 C
    * Forecasted Humidity     : 100.0%
    * Forecasted Rainfall     : 0.00 mm
---------------------------------------------------------------------------
  Station Precursor Signals (Past Lags & Upwind Fronts):
    * 3-Hour PM2.5 Surge      : +100.2 ug/m3
    * Regional NCR Mean       : 338.9 ug/m3
    * Upwind NW Smoke Front   : 338.9 ug/m3 (NW Gradient: -274.6)
===========================================================================
```

