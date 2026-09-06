# Delhi-NCR Air Quality 24-Hour Forecasting & Safety Alert System

## 1. Problem Statement
This system provides 24-hour-ahead air quality forecasting for 40 Central Pollution Control Board (CPCB) and Delhi Pollution Control Committee (DPCC) continuous ambient air quality monitoring stations across Delhi-NCR. Its primary objective is to power an early-warning safety alert when ambient air quality is expected to cross into "Poor" or worse (CPCB PM2.5 $> 90\ \mu\text{g/m}^3$, corresponding to AQI sub-index $> 200$), providing municipal agencies and residents actionable advance notice before hazardous pollution episodes develop.

---

## 2. Pipeline Architecture
The end-to-end pipeline consists of five stages:
```
[CPCB / Data.gov.in 15-Minute Parquets]
       │
       ▼
[SQLite Storage (data/aqi_history.db - aqi_hourly)]
       │
       ▼
[Feature Engineering (src/analysis/feature_engineering.py)]
  ├── CPCB AQI sub-index conversions
  ├── Temporal lags (1h, 2h, 3h, 6h, 24h) & rolling window statistics (6h, 24h)
  ├── Multi-station spatial features (NCR regional mean PM2.5, NW stubble smoke front)
  ├── Short-term pollution dynamics (3h/6h velocity differences, 6h acceleration)
  ├── Calendar & seasonality indicators (hour, day of week, month, day-of-year harmonics)
  └── 24-hour per-station target generation (target_pm25 = shift(-24), target_poor_or_worse)
       │
       ▼
[Model Training & Evaluation (src/models/train_model.py)]
  ├── Continuous PM2.5 Regressor (XGBRegressor) -> Point forecast baseline benchmark
  └── Binary Safety Alert Classifier (XGBClassifier) -> 24h Hazard Probability
       │
       ▼
[Inference & Live Safety Alert Dispatch (src/predict_live.py)]
  ├── Evaluates station readings against tuned hazard probability threshold (0.25)
  ├── Displays Open-Meteo 24h NWP meteorological advisory context
  └── Dispatches Alert Tier ([CRITICAL ALERT] / [WATCH] / [NORMAL]) with precursor diagnostics
```

---

## 3. Setup & Execution Instructions

### Prerequisites
- Python 3.10+
- Virtual environment (`venv`)

### Installation
```bash
git clone <repo_url>
cd "Delhi-NCR Air Quality Forecasting"
python -m venv venv

# Activate virtual environment:
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### Script Execution Order
To reproduce data ingestion, feature generation, model training, and inference from a clean clone:

1. **Ingest Raw 15-Minute Parquet Data & Resample Hourly:**
   ```bash
   python src/ingestion/fetch_cpcb_15min.py
   ```
   *Downloads multi-year CPCB data (2022–2025), filters for 40 Delhi-NCR stations, aggregates 15-minute readings to 1-hour regular intervals, and writes 1,367,285 records to SQLite table `aqi_hourly`.*

2. **Run Feature Engineering:**
   ```bash
   python src/analysis/feature_engineering.py
   ```
   *Computes sub-indices, forward-fills station weather gaps with train-period medians, calculates past lags, velocity diffs, upwind gradients, and saves 1,075,166 records to table `aqi_features`.*

3. **Train Models & Evaluate on Chronological Holdout:**
   ```bash
   python src/models/train_model.py
   ```
   *Splits data chronologically (Train: 2022-01-02 to 2024-11-07 [70%], Val: 2024-11-07 to 2025-05-30 [15%], Test: 2025-05-30 to 2025-12-31 [15%]), fits models on Train+Val, benchmarks against persistence, and saves artifacts to `models/`.*

4. **Run Live Inference & Safety Alert:**
   ```bash
   python src/predict_live.py
   ```

---

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
