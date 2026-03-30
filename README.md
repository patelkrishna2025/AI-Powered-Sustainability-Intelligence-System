# 🌍 AI-Powered Sustainability Intelligence System

> AI & Environmental Sustainability Analytics

---

## 📁 Project Structure

```
sustainability_project/
│
├── data/
│   ├── raw/                         # OWID CO2 raw CSVs
│   └── processed/                   # Cleaned & feature-engineered data
│
├── pipeline/
│   └── data_pipeline.py             # ETL pipeline (load → clean → engineer → export)
│
├── models/
│   ├── anomaly_detection.py         # Isolation Forest + Z-Score + LSTM Autoencoder
│   ├── prediction_model.py          # Random Forest + GBM + Exponential Smoothing
│   └── scoring_system.py            # 0–100 Sustainability Scorer
│
├── cv_module/
│   └── environmental_vision.py      # OpenCV: smoke, garbage, deforestation detection
│
├── dashboard/
│   └── app.py                       # Streamlit interactive dashboard
│
├── notebooks/
│   └── eda.ipynb                    # Exploratory Data Analysis
```

## 🚀 Quick Start

```bash
# 1. Run the dashboard
streamlit run dashboard/app.py

# 2. Access in browser at:
http://localhost:8501

```

## 🧠 System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   DATA SOURCES                           │
│  OWID CO2 · World Bank · NASA GISS · OpenAQ             │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│              DATA PIPELINE (pipeline/)                   │
│  Load → Clean → Feature Engineering → Normalise         │
└──────────────────────┬───────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌─────────────────┐       ┌──────────────────────┐
│ ANOMALY ENGINE  │       │  FORECASTING ENGINE   │
│ Isolation Forest│       │  Random Forest        │
│ Z-Score         │       │  Gradient Boosting    │
│ LSTM Autoencoder│       │  Exponential Smoothing│
└────────┬────────┘       └──────────┬────────────┘
         │                           │
         └──────────┬────────────────┘
                    │
          ┌─────────▼──────────┐
          │  SCORING ENGINE     │
          │  0–100 Score        │
          │  Weighted Pillars   │
          └─────────┬──────────┘
                    │
          ┌─────────▼──────────┐
          │  CV MODULE          │
          │  Smoke Detection    │
          │  Garbage Detection  │
          │  Deforestation      │
          └─────────┬──────────┘
                    │
          ┌─────────▼──────────┐
          │  STREAMLIT DASHBOARD│
          │  KPIs · Anomalies   │
          │  Forecasts · Map    │
          │  Leaderboard        │
          └────────────────────┘
```

---

## 🤖 AI Models

| Module | Model | Purpose |
|--------|-------|---------|
| Anomaly Detection | Isolation Forest | Multi-variate global anomalies |
| Anomaly Detection | Z-Score (rolling) | Country time-series anomalies |
| Anomaly Detection | LSTM Autoencoder | Sequence-based anomalies |
| Forecasting | Random Forest | 10-year CO2 forecast |
| Forecasting | Gradient Boosting | 10-year CO2 forecast |
| Forecasting | Exponential Smoothing | Global trend baseline |
| Scoring | Weighted Composite | 0–100 sustainability score |
| Computer Vision | OpenCV HSV + Edge | Environmental issue detection |

---

## 📊 Dashboard Features

- **KPI Cards** — Country score, emissions, alert counts
- **Gauge Chart** — Real-time sustainability score
- **Anomaly Alerts** — Critical / High / Medium severity list
- **Trend Graphs** — Multi-country CO2 time-series
- **AI Forecasting** — Ensemble 10–20 year predictions
- **CV Analysis** — Upload image → detect environmental issues
- **Global Map** — Choropleth sustainability score map
- **Leaderboard** — 190+ countries ranked A–F

---

## 📈 Datasets

| Dataset | Source | Description |
|---------|--------|-------------|
| OWID CO₂ Data | Our World in Data | CO2, GHG, energy per country (1750–2023) |
| World Bank Climate | World Bank API | Temperature, rainfall, sea level |
| NASA GISS | NASA | Global surface temperature anomalies |
| OpenAQ | openaq.org | Real-time air quality measurements |
| Global Forest Watch | GFW | Deforestation & tree cover loss |

---

## 🏆 Scoring Methodology

```
Score = Σ (weight_i × normalised_indicator_i) × 100 + trend_bonus

Indicators (weights):
  CO₂ per capita          25%  ↓ lower is better
  Total GHG per capita    20%  ↓
  CO₂ / GDP intensity     15%  ↓
  Fossil fuel ratio       15%  ↓
  Energy per capita       10%  ↓
  Methane per capita       8%  ↓
  Temperature change (GHG) 7%  ↓

Grades: A (80–100) · B (65–79) · C (50–64) · D (35–49) · F (<35)
```

---

## 🔬 Computer Vision Pipeline

```
Image/Video Frame
       │
       ├─→ SmokeDetector     (HSV grey mask + Laplacian blur + variance)
       ├─→ GarbageDetector   (Brown/dark mask + edge density + entropy)
       └─→ DeforestDetector  (NDVI proxy: green ratio vs bare soil ratio)
                │
                ▼
         DetectionResult
         { confidence, severity, bboxes, alert_message }
                │
                ▼
         Annotated Frame + Dashboard Alert
```
## Future Work & Extensions
•	*Integrate real-time air quality API (OpenAQ) for live dashboard updates
•	*Add satellite imagery pipeline using Google Earth Engine for automated deforestation monitoring
•	*Deploy pre-trained YOLOv8 model for more accurate CV environmental detection
•	*Implement Prophet or N-BEATS for improved time-series forecasting
•	*Add country-level policy recommendation engine based on score trajectory
•	*Deploy dashboard to Streamlit Cloud for public access
•	*Integrate carbon credit market data to correlate emission changes with pricing
•	*Build mobile app wrapper using Streamlit's mobile-responsive layout

## References & Data Sources
•	*Our World in Data (2024). CO2 and Greenhouse Gas Emissions. ourworldindata.org/co2-and-greenhouse-gas-emissions
•	*Liu, F.T., Ting, K.M. and Zhou, Z.H. (2008). Isolation Forest. IEEE ICDM 2008.
•	*Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12.
•	*Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.
•	*Holt, C.E. (1957). Forecasting Seasonals and Trends by Exponentially Weighted Averages.
•	*Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools.
•	*Streamlit Inc. (2024). Streamlit Documentation. docs.streamlit.io
•	*Climate Change Performance Index (2024). Climate Action Network. ccpi.org




