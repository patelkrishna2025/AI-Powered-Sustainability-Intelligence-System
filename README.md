# 🌍 AI-Powered Sustainability Intelligence System

## 📁 Project Structure

```
Project/
├── app.py                          ← Streamlit dashboard 
├── README.md                       ← This file
├── requirements.txt                ← Project dependencies
│
├── chatbot/                        ← Rule-based Chatbot module
│   └── sustainability_chatbot.py
│
├── cv_module/                      ← Computer Vision (env. detection)
│   └── environmental_vision.py
│
├── weather/                        ← Weather Prediction module
│   └── weather_predictor.py
│
├── models/                         ← ML models
│   ├── anomaly_detection.py
│   ├── prediction_model.py
│   └── scoring_system.py
│
├── pipeline/                       ← Data ingestion & processing
│   └── data_pipeline.py
│
├── data/
│   └── raw/
│       └── owid-co2-data.csv
│
├── exports/                        ← Save charts / reports here
├── assets/                         ← Static assets (CSS, logos)
└── images/
    └── profile.png
```

## 📦 requirements.txt

```txt
streamlit
pandas
numpy
plotly
matplotlib
seaborn
scikit-learn
scipy
joblib
Pillow
opencv-python-headless
```

## 🚀 How to Run

```bash
pip install streamlit pandas numpy plotly scikit-learn opencv-python pillow matplotlib seaborn
streamlit run app.py
```

## 📊 Dashboard Tabs

| # | Tab | Description |
|---|-----|-------------|
| 1 | 📊 Overview | KPI cards, country trends, gauge, key-country comparison |
| 2 | 🚨 Anomaly Detection | Isolation Forest + Z-Score alerts |
| 3 | 🔮 AI Forecasting | Random Forest + GBM ensemble forecasts |
| 4 | 👁️ CV Analysis | Environmental detection + 8 image filters |
| 5 | 🗺️ Global Map | Choropleth sustainability map |
| 6 | 🏆 Leaderboard | Ranked country scoreboard |
| 7 | 💬 Chatbot | Sustainability Q&A chatbot |
| 8 | 🌦️ Weather Prediction | AI weather forecasting & climate analysis |

## 🔧 Changes
- ✅ Tab 8 added: **Weather Prediction** — RF Climate Engine
- ✅ New folder: `weather/` with `WeatherPredictionEngine`
- ✅ Forecasts: Temperature · Precipitation · Humidity (6–36 months)
- ✅ Seasonal radar chart, extreme events detection (2σ threshold)
- ✅ Warming trend analysis (°C/year)
- ✅ Notebook: Section 10 — Weather Prediction Demo added (6 new cells)
- ✅ Notebook: Fixed hardcoded Windows paths → portable os.path

## 📓 Notebook Sections

| # | Section |
|---|---------|
| 1 | Environment Setup & Data Loading |
| 2 | Exploratory Data Analysis (EDA) |
| 3 | Feature Engineering |
| 4 | Anomaly Detection (Isolation Forest + Z-Score) |
| 5 | Predictive Forecasting (Random Forest Ensemble) |
| 6 | Sustainability Scoring System |
| 7 | Computer Vision Module Demo |
| 8 | Results Summary & Insights |
| 9 | Chatbot Demo |
| 10 | Weather Prediction Module|
