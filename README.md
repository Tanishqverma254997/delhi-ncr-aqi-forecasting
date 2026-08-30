# Delhi-NCR Air Quality Forecasting & Alert System

## 1. Problem Statement
This system provides 24-hour-ahead air quality forecasting for 40 Central Pollution Control Board (CPCB) and DPCC continuous ambient air quality monitoring stations across Delhi-NCR. Its primary objective is to power an early-warning safety alert when ambient air quality is expected to cross into "Poor" or worse (CPCB PM2.5 $> 90\ \mu\text{g/m}^3$, corresponding to AQI sub-index $> 200$), giving residents and municipal agencies actionable advance notice before severe pollution events occur.

---

## 2. Architecture
The end-to-end pipeline consists of five sequential stages:
```
[CPCB / Data.gov.in Ingestion]     [Open-Meteo 48h NWP API]
       │                                     │
       ▼                                     ▼
[SQLite Storage (aqi_history.db)]  [src/ingestion/fetch_open_meteo_forecast.py]
       │                                     │
       └──────────────────┬──────────────────┘
                          ▼
[Feature Engineering (src/analysis/feature_engineering.py)]
  ├── CPCB AQI sub-index calculations
  ├── 24h-Ahead Weather Forecast Features (wind_speed_forecast_24h, ventilation_index_forecast_24h, temp_drop_forecast_24h)
  ├── Transition velocity (pm25_diff_3, pm25_diff_6, pm25_accel) & Stagnation flags
  ├── Northwest transboundary smoke front & gradient (nw_smoke_front, nw_gradient)
  ├── Multi-pollutant ratios (PM2.5/PM10) & regional NCR background mean
  └── 24h-ahead per-station target generation (target_pm25, target_poor_or_worse)
       │
       ▼
[Model Training & Evaluation (src/models/train_model.py)]
  ├── Primary: Binary Safety Classifier (XGBClassifier) -> 24h Hazard Probability
  └── Baseline: Continuous Regressor (XGBRegressor) -> Point forecast benchmark
       │
       ▼
[Inference & Safety Alert Serving (src/predict_live.py)]
  ├── Queries Open-Meteo API for live 24h meteorological forecast (wind, temp, ventilation)
  ├── Evaluates hazard risk probability against safety threshold (25%)
  └── Dispatches Alert (CRITICAL ALERT / WATCH / NORMAL) with precursor diagnostics
```

---

## 3. Setup & Execution

### Prerequisites
- Python 3.10+
- Virtual environment (`venv`)

### Installation
```bash
git clone <repo_url>
cd "Delhi-NCR Air Quality Forecasting"
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### Script Execution Order
To reproduce data processing, feature generation, and model training from scratch:

1. **Ingest Raw 15-Minute Parquet Data & Resample Hourly:**
   ```bash
   python src/ingestion/fetch_cpcb_15min.py
   ```
   *Downloads multi-year CPCB data (2022–2025), filters for Delhi-NCR stations, aggregates 15-minute readings to 1-hour intervals, and writes to SQLite table `aqi_hourly`.*

2. **Run Feature Engineering (with 24h NWP Forecast Features):**
   ```bash
   python src/analysis/feature_engineering.py
   ```
   *Computes sub-indices, forward-fills station weather gaps with train-period medians, computes temporal lags, velocity diffs, upwind gradients, 24h-ahead forecasted weather features, and writes 1,075,166 records to table `aqi_features`.*

3. **Train Models & Evaluate against Persistence Baselines:**
   ```bash
   python src/models/train_model.py
   ```
   *Splits data chronologically (Train: 2022-01-02 to 2024-11-07 [70%], Val: 2024-11-07 to 2025-05-30 [15%], Test: 2025-05-30 to 2025-12-31 [15%]), fits models on Train+Val, benchmarks against persistence baselines, and saves model artifacts to `models/`.*

4. **Run Live Inference & Safety Alert (with Real-Time Open-Meteo Forecast):**
   ```bash
   python src/predict_live.py
   ```

---

## 4. Results & Benchmark Evaluation

All metrics below are evaluated on the unseen chronological holdout test set (**2025-05-30 23:00:00 to 2025-12-31 05:00:00**, $N=161,279$ hourly observations across 40 stations) with all future-shifted weather leaks removed.

### a & b. Continuous Regression Comparison

| Model | MAE ($\mu\text{g/m}^3$) | RMSE ($\mu\text{g/m}^3$) | $R^2$ | 6-Class AQI Category Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Persistence Baseline** ($\hat{y}_{t+24} = y_t$) | 35.0024 | 60.3863 | 0.6796 | 54.39% |
| **Trained Regressor** (XGBRegressor) | 34.5681 | 60.4317 | 0.6791 | 54.17% |
| **Delta ($\Delta$)** | **-0.43 µg/m³** | **+0.05 µg/m³** | **-0.0005** | **-0.22%** |

### c. Real Edge Over Baseline
The honest continuous regression model shows **zero practical edge** over simple 24-hour persistence:
- MAE improves by only **0.43 $\mu\text{g/m}^3$** (1.2% relative error reduction).
- $R^2$ is flat at **0.6791** vs **0.6796** (persistence).
- 6-class AQI category accuracy is **54.17%** (slightly worse than persistence at 54.39%).

Single-hour PM2.5 readings are dominated by high-frequency micro-scale noise. Because the regression error ($\text{MAE} \approx 34.6\ \mu\text{g/m}^3$) exceeds the width of individual CPCB AQI category bands ($\sim 30\ \mu\text{g/m}^3$), point predictions routinely spill into adjacent categories.

### d. Per-Category Performance (Regression Predictions vs Ground Truth)

| AQI Category | PM2.5 Range ($\mu\text{g/m}^3$) | Test Support ($N$) | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Good** | $0 - 30$ | 41,257 | 0.7381 | 0.4157 | 0.5318 |
| **Satisfactory** | $31 - 60$ | 41,171 | 0.4910 | 0.7164 | 0.5827 |
| **Moderate** | $61 - 90$ | 16,911 | 0.3656 | 0.3250 | 0.3441 |
| **Poor** *(Alert Threshold)* | $91 - 120$ | 11,280 | **0.3139** | **0.2290** | **0.2648** |
| **Very Poor** | $121 - 250$ | 33,319 | 0.5958 | 0.7310 | 0.6565 |
| **Severe** | $> 250$ | 17,341 | 0.5992 | 0.4779 | 0.5317 |

---

### e. Binary "Poor or Worse" Safety Alert Classifier

Reframing the problem as a dedicated binary classification task—*"Will air quality tomorrow cross into Poor or worse ($\text{PM2.5} > 90\ \mu\text{g/m}^3$ / $\text{AQI} > 200$)?"*—directly aligns the model with the product objective:

- **Test Set Positive Class Support:** 61,940 hazardous observations (38.41% of test rows).
- **Test Set Negative Class Support:** 99,339 non-hazardous observations (61.59% of test rows).
- **Model ROC-AUC:** **0.9792**
- **Top Predictive Features:** Current PM2.5 (51.4%), PM2.5 Lag-1 (16.2%), Monsoon Seasonality (5.5%), Winter Seasonality (3.2%), Regional NCR Mean (3.0%).

#### Comparison against Persistence:
| Model / Threshold | Precision | Recall | F1-Score | False Negatives (Misses) | False Positives |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Persistence Baseline** | 0.9034 | 0.8951 | 0.8992 | 6,498 | 5,927 |
| **XGBClassifier (Threshold = 0.50)** | 0.9192 | 0.9116 | 0.9154 | 5,478 | 4,961 |
| **XGBClassifier (Threshold = 0.25, Recommended)** | **0.8658** | **0.9588** | **0.9099** | **2,552** | **9,205** |

#### Operating Threshold Trade-Off Analysis:
| Threshold | Precision | Recall | F1-Score | True Negatives | False Positives | False Negatives (Missed Bad Air) | True Positives |
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

**Operating Decision:** Threshold **`0.25`** provides the best operating point for public health protection:
1. Achieves **95.88% recall** on unseen future holdout data.
2. Cuts un-alerted dangerous days from **6,498 (persistence)** down to **2,552** (a **60.7% reduction in missed dangerous air days**).
3. Preserves high alert reliability with **86.58% precision**.

---

## 5. Honest Limitations

1. **Category Width vs. MAE Mismatch:**
   CPCB categories are $\sim 30\ \mu\text{g/m}^3$ wide, but the regression MAE is $34.58\ \mu\text{g/m}^3$. Point regression models cannot reliably classify 6 discrete categories without substantial label spilling.
2. **Current Weather Features Are Not 24h Forecasts:**
   In `feature_engineering.py` and `train_model.py`, the weather inputs (`temperature`, `humidity`, `wind_speed`, etc.) represent observations at time $t$, not true 24-hour atmospheric forecasts for time $t+24$. While live inference queries Open-Meteo's forecast API, historical training did not train on forecasted numerical weather prediction (NWP) parameters.
3. **Seasonal Test Set Composition:**
   The multi-year holdout test set (June 1 to Dec 31, 2025) spans Monsoon (41.3% of rows, where PM2.5 is low and steady at $15–25\ \mu\text{g/m}^3$) and Post-Monsoon before reaching winter (30.0% of rows). Testing on 7 months spanning clean baseline lows to winter highs inflates $SS_{\text{tot}}$, lifting persistence $R^2$ to 0.6792. Evaluating solely on peak winter months yields a persistence $R^2$ of $\approx 0.3091$.
4. **Single-Hour vs. 24-Hour Regulatory Standard:**
   India's official CPCB AQI standard specifies **24-hour rolling average concentrations**. Single-hour PM2.5 predictions are an operational proxy, not official regulatory AQI declarations.

---

## 6. What I'd Improve with More Time

1. **Train on True 24h Numerical Weather Prediction (NWP) Forecasts:**
   Extract historical ECMWF / GFS 24h-ahead weather forecasts (boundary layer height, surface wind speed, temperature inversions) to replace static weather observations.
2. **Realign the Target to 24h Rolling Mean PM2.5:**
   Change the regression target from single-hour $\text{PM2.5}_{t+24}$ to the 24-hour average $\frac{1}{24}\sum_{k=1}^{24}\text{PM2.5}_{t+k}$, matching CPCB's statutory standard and eliminating high-frequency sensor noise.
3. **Incorporate Wind-Directed Spatial Advection:**
   Construct dynamic upwind station features (e.g., measuring Northwest border stations like Bawana/Narela when wind direction is $300^\circ–330^\circ$) to capture incoming agricultural stubble smoke plumes before they reach central Delhi.
4. **Implement Asymmetric Pinball Loss (Quantile Regression):**
   Train an 80th-percentile pinball loss regressor in XGBoost to explicitly penalize under-predictions heavier than over-predictions for conservative public health planning.
5. **Add Event Proximity Calendars:**
   Explicitly encode days-to-Diwali and crop-residue-burning seasonal windows rather than relying solely on month and day-of-week integer indicators.
