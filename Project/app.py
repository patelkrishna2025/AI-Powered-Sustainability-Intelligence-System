"""
=============================================================
 AI-Powered Sustainability Intelligence System
 MODULE: Interactive Streamlit Dashboard  
 Run:  streamlit run app.py
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os, io
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

# ── Path setup ─────────────────────────────────────────────
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from pipeline.data_pipeline import SustainabilityDataPipeline, KEY_COUNTRIES
from models.anomaly_detection import AnomalyDetectionEngine
from models.prediction_model import ForecastingEngine
from models.scoring_system import SustainabilityScorer
from chatbot.sustainability_chatbot import SustainabilityChatbot
from weather.weather_predictor import WeatherPredictionEngine

# ── PAGE CONFIG ────────────────────────────────────────────
st.set_page_config(
    page_title="🌍 AI Sustainability Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── COLOURS ────────────────────────────────────────────────
PRIMARY   = "#2E8B57"
SECONDARY = "#FF6B35"
DARK_BG   = "#0E1117"
CARD_BG   = "#1E2130"

# ── CUSTOM CSS ─────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: linear-gradient(135deg, #1E2130, #2A3045);
    border: 1px solid #3A4060; border-radius: 12px;
    padding: 20px; text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
  }
  .metric-value { font-size: 2.2rem; font-weight: 800; color: #2E8B57; }
  .metric-label { font-size: 0.85rem; color: #9BA3AF; margin-top: 4px; }
  .alert-critical { background:#4A1010; border-left:4px solid #FF3333; padding:12px; border-radius:8px; }
  .alert-high     { background:#3A2010; border-left:4px solid #FF8C00; padding:12px; border-radius:8px; }
  .alert-medium   { background:#3A3010; border-left:4px solid #FFD700; padding:12px; border-radius:8px; }
  .header-gradient {
    background: linear-gradient(90deg, #2E8B57, #1a5c38);
    padding: 25px; border-radius: 15px; margin-bottom: 25px;
    box-shadow: 0 8px 32px rgba(46,139,87,0.3);
  }
  /* Chatbot bubbles */
  .cb-user {
    background: linear-gradient(135deg,#2E8B57,#1a5c38);
    color:white; padding:12px 16px;
    border-radius:18px 18px 4px 18px;
    margin:8px 0; max-width:80%; float:right; clear:both;
  }
  .cb-bot {
    background:#1E2130; border:1px solid #3A4060;
    color:#e0e0e0; padding:12px 16px;
    border-radius:18px 18px 18px 4px;
    margin:8px 0; max-width:80%; float:left; clear:both;
  }
  .cf { clear:both; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  DATA LOADING (cached) — PATH AUTO-DETECTED
# ═══════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner="Loading sustainability data …")
def load_data():
    # Auto-detect CSV location — works on any machine
    candidates = [
        os.path.join(ROOT, "data", "raw", "owid-co2-data.csv"),
        os.path.join(ROOT, "..", "data", "raw", "owid-co2-data.csv"),
        os.path.join(ROOT, "owid-co2-data.csv"),
    ]
    csv_path = next((p for p in candidates if os.path.exists(p)), None)
    if csv_path is None:
        st.error("❌ owid-co2-data.csv not found. Place it in data/raw/")
        st.stop()
    pipe = SustainabilityDataPipeline(csv_path)
    return pipe.run(start_year=1990)


@st.cache_resource(show_spinner="Training AI models …")
def train_models(df: pd.DataFrame):
    anomaly_engine = AnomalyDetectionEngine()
    anomaly_engine.train(df)
    scorer = SustainabilityScorer()
    scorer.fit(df)
    return anomaly_engine, scorer


@st.cache_data(show_spinner="Scoring countries …")
def compute_scores(_scorer, df):
    return _scorer.score_dataframe(df)


@st.cache_data(show_spinner="Running anomaly detection …")
def run_anomaly_detection(_engine, df):
    return _engine.detect_global(df)


# ═══════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════
def main():
    # ─── Header ──────────────────────────────
    st.markdown("""
    <div class="header-gradient">
      <h1 style="color:white; margin:0; font-size:2rem;">
        🌍 AI-Powered Sustainability Intelligence System
      </h1>
      <p style="color:#a8d5b5; margin:8px 0 0 0; font-size:1rem;">
        Real-Time Anomaly Detection · Predictive Analytics · CV Analysis · AI Chatbot
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ─── Load data ───────────────────────────
    with st.spinner("Initialising system …"):
        df = load_data()
        anomaly_engine, scorer = train_models(df)

    # ─── Sidebar ─────────────────────────────
    with st.sidebar:
        profile_path = os.path.join(ROOT, "images", "profile.png")
        if os.path.exists(profile_path):
            st.image(profile_path, width=80)
        st.title("⚙️ Controls")

        selected_country = st.selectbox(
            "🌐 Country", sorted(df["country"].unique()),
            index=list(sorted(df["country"].unique())).index("India")
                  if "India" in df["country"].unique() else 0
        )

        year_min, year_max = int(df["year"].min()), int(df["year"].max())
        year_range = st.slider("📅 Year Range", year_min, year_max,
                               (max(year_min, 2000), year_max))

        target_metric = st.selectbox("📊 Primary Metric", [
            "co2", "co2_per_capita", "primary_energy_consumption",
            "temperature_change_from_ghg", "total_ghg",
        ])

        forecast_horizon = st.slider("🔮 Forecast Horizon (years)", 5, 20, 10)

        st.divider()
        st.caption("🌍 Sustainability Analytics v2")

    # ─── Filter data ─────────────────────────
    df_filtered = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]
    country_df  = df_filtered[df_filtered["country"] == selected_country].sort_values("year")

    # ─── Tabs (8 total — added Weather Prediction) ──────
    tabs = st.tabs([
        "📊 Overview",
        "🚨 Anomaly Detection",
        "🔮 AI Forecasting",
        "👁️ CV Analysis",
        "🗺️ Global Map",
        "🏆 Leaderboard",
        "💬 Chatbot",
        "🌦️ Weather Prediction",
    ])
    (tab_overview, tab_anomaly, tab_forecast,
     tab_cv, tab_map, tab_leaderboard, tab_chat, tab_weather) = tabs

    # ═══════════════════════════════
    #  TAB 1 – OVERVIEW
    # ═══════════════════════════════
    with tab_overview:
        scored_df = compute_scores(scorer, df_filtered)
        kpis      = scorer.global_kpis(scored_df)

        country_scores = scored_df[scored_df["country"] == selected_country]
        if len(country_scores) > 0:
            latest_score = country_scores.sort_values("year").iloc[-1]["sustainability_score"]
            score_grade  = scorer.grade(latest_score)
            score_label  = scorer.label(latest_score)
        else:
            latest_score, score_grade, score_label = 50.0, "C", "Moderate ⚠️"

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value" style="color:#2E8B57">{latest_score:.1f}</div>
              <div class="metric-label">🌱 {selected_country} Score</div>
              <div style="font-size:1.5rem">{score_grade}</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            val = country_df[target_metric].iloc[-1] if len(country_df) > 0 and target_metric in country_df.columns else 0
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">{val:,.1f}</div>
              <div class="metric-label">📊 Latest {target_metric.replace('_',' ').title()}</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value" style="color:#FF6B35">{kpis['countries_critical']}</div>
              <div class="metric-label">🚨 Critical Countries</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value" style="color:#4CAF50">{kpis['countries_excellent']}</div>
              <div class="metric-label">✅ Excellent Countries</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_chart, col_score = st.columns([2, 1])
        with col_chart:
            st.subheader(f"📈 {selected_country} – {target_metric.replace('_',' ').title()} Trend")
            if len(country_df) > 0 and target_metric in country_df.columns:
                fig = px.area(country_df, x="year", y=target_metric,
                              color_discrete_sequence=[PRIMARY], template="plotly_dark")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  showlegend=False, height=350)
                st.plotly_chart(fig, use_container_width=True)
        with col_score:
            st.subheader("🎯 Sustainability Score")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta", value=latest_score,
                delta={"reference": kpis["global_avg_score"]},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": PRIMARY},
                       "steps": [{"range":[0,35],"color":"#4A1010"},{"range":[35,50],"color":"#3A2010"},
                                  {"range":[50,65],"color":"#3A3010"},{"range":[65,80],"color":"#1A3A1A"},
                                  {"range":[80,100],"color":"#0A2A0A"}],
                       "threshold": {"line":{"color":"white","width":3},"thickness":0.75,
                                     "value":kpis["global_avg_score"]}},
                title={"text": f"{score_label}", "font": {"color": "white"}},
                number={"font": {"color": "white"}},
            ))
            fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=300,
                                    margin=dict(t=30,b=10,l=20,r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.metric("Global Avg Score", f"{kpis['global_avg_score']}")
            st.metric("Best Country", kpis["top_country"], f"+{kpis['top_score']:.1f}")

        st.subheader("🌐 Key Countries Comparison")
        key_df = df_filtered[df_filtered["country"].isin(KEY_COUNTRIES)]
        if len(key_df) > 0 and target_metric in key_df.columns:
            fig2 = px.line(key_df, x="year", y=target_metric, color="country",
                           template="plotly_dark",
                           color_discrete_sequence=px.colors.qualitative.Set2)
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig2, use_container_width=True)

    # ═══════════════════════════════
    #  TAB 2 – ANOMALY DETECTION
    # ═══════════════════════════════
    with tab_anomaly:
        st.subheader("🚨 AI Anomaly Detection Engine")
        st.info("Using Isolation Forest (multi-variate) + Z-Score (country time-series)")

        flagged_df = run_anomaly_detection(anomaly_engine, df_filtered)
        summary    = anomaly_engine.get_anomaly_summary(flagged_df)
        counts     = anomaly_engine.get_alert_count(flagged_df)

        ac1, ac2, ac3, ac4 = st.columns(4)
        ac1.metric("🔴 Critical", counts["critical"])
        ac2.metric("🟠 High",     counts["high"])
        ac3.metric("🟡 Medium",   counts["medium"])
        ac4.metric("📊 Total",    counts["total"])

        if len(summary) > 0:
            st.markdown("### 🔔 Active Alerts")
            for _, row in summary.head(15).iterrows():
                sev = row.get("severity", "medium")
                css_class = f"alert-{sev if sev in ['critical','high','medium'] else 'medium'}"
                icon = {"critical":"🔴","high":"🟠","medium":"🟡"}.get(sev,"⚪")
                st.markdown(f"""
                <div class="{css_class}" style="margin-bottom:8px;">
                  {icon} <strong>{row['country']}</strong> ({int(row['year'])})
                  — CO₂: {row.get('co2',0):.2f} Mt — Score: {row.get('if_score',0):.3f}
                  — <strong>{sev.upper()}</strong>
                </div>""", unsafe_allow_html=True)

        st.subheader("🔬 Anomaly Score Distribution")
        if "if_score" in flagged_df.columns and "co2" in flagged_df.columns:
            scatter_df = flagged_df[flagged_df["country"].isin(KEY_COUNTRIES)].copy()
            scatter_df["anomaly_label"] = scatter_df["is_anomaly"].map({True:"Anomaly",False:"Normal"})
            scatter_df["size_val"] = (scatter_df["if_score"] * -1).clip(lower=0.01)
            fig_anom = px.scatter(scatter_df, x="year", y="co2",
                                  color="anomaly_label", symbol="anomaly_label",
                                  size="size_val", size_max=12,
                                  hover_data=["country","severity","if_score"],
                                  color_discrete_map={"Anomaly":SECONDARY,"Normal":PRIMARY},
                                  template="plotly_dark",
                                  facet_col="country", facet_col_wrap=4)
            fig_anom.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=600)
            st.plotly_chart(fig_anom, use_container_width=True)

        st.subheader(f"📉 {selected_country} – Z-Score Analysis")
        zs_df = anomaly_engine.detect_country(df, selected_country)
        if "zscore" in zs_df.columns and len(zs_df) > 0:
            fig_zs = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   subplot_titles=["CO₂ Emissions","Z-Score"])
            fig_zs.add_trace(go.Scatter(x=zs_df["year"], y=zs_df["co2"],
                                        mode="lines", name="CO₂", line=dict(color=PRIMARY)), row=1, col=1)
            zscore_colors = [SECONDARY if abs(z) > 2.5 else PRIMARY for z in zs_df["zscore"].fillna(0)]
            fig_zs.add_trace(go.Bar(x=zs_df["year"], y=zs_df["zscore"],
                                    name="Z-Score", marker_color=zscore_colors), row=2, col=1)
            fig_zs.add_hline(y=2.5, line_dash="dash", line_color="red", row=2, col=1)
            fig_zs.add_hline(y=-2.5, line_dash="dash", line_color="red", row=2, col=1)
            fig_zs.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  template="plotly_dark", height=450)
            st.plotly_chart(fig_zs, use_container_width=True)

    # ═══════════════════════════════
    #  TAB 3 – AI FORECASTING
    # ═══════════════════════════════
    with tab_forecast:
        st.subheader("🔮 AI-Powered Predictive Forecasting")
        st.info("Ensemble: Random Forest + Gradient Boosting + Exponential Smoothing")

        forecast_engine = ForecastingEngine()
        with st.spinner(f"Generating {forecast_horizon}-year forecast for {selected_country} …"):
            result = forecast_engine.fit_and_forecast(df, selected_country, target_metric, forecast_horizon)

        if result:
            hist = result["historical"]
            ens  = result["ensemble_forecast"]
            rf   = result["rf_forecast"]
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(x=hist["year"], y=hist[target_metric],
                                        name="Historical", mode="lines+markers",
                                        line=dict(color=PRIMARY, width=2)))
            fig_fc.add_trace(go.Scatter(x=ens["year"], y=ens[f"predicted_{target_metric}"],
                                        name="Ensemble Forecast", mode="lines+markers",
                                        line=dict(color=SECONDARY, width=2, dash="dash")))
            fig_fc.add_trace(go.Scatter(x=rf["year"], y=rf[f"predicted_{target_metric}"],
                                        name="Random Forest", mode="lines",
                                        line=dict(color="#9B59B6", width=1, dash="dot")))
            fig_fc.update_layout(title=f"📈 {selected_country} – {target_metric.replace('_',' ').title()} Forecast",
                                  template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)", height=450,
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig_fc, use_container_width=True)

            col_fc1, col_fc2 = st.columns(2)
            with col_fc1:
                st.markdown("**📊 Ensemble Forecast Values**")
                display_fc = ens.copy()
                display_fc[f"predicted_{target_metric}"] = display_fc[f"predicted_{target_metric}"].round(3)
                st.dataframe(display_fc, use_container_width=True, hide_index=True)
            with col_fc2:
                st.markdown("**🌐 Global CO₂ Trend**")
                global_trend = forecast_engine.global_trend(df, "co2", forecast_horizon)
                fig_global = px.line(global_trend, x="year", y="co2", color="type",
                                     color_discrete_map={"historical":PRIMARY,"forecast":SECONDARY},
                                     template="plotly_dark")
                fig_global.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350)
                st.plotly_chart(fig_global, use_container_width=True)
        else:
            st.warning(f"Not enough data to forecast '{target_metric}' for {selected_country}.")

    # ═══════════════════════════════
    #  TAB 4 – COMPUTER VISION (Enhanced)
    # ═══════════════════════════════
    with tab_cv:
        st.subheader("👁️ Computer Vision – Environmental Issue Detection")

        cv_mode = st.radio("Mode", ["🔍 Environmental Detection", "🎨 Image Filters & Analysis"],
                           horizontal=True)

        uploaded = st.file_uploader("📤 Upload Image", type=["jpg","jpeg","png","bmp","webp"],
                                    key="cv_upload")

        if uploaded:
            try:
                import cv2
                file_bytes = np.frombuffer(uploaded.read(), np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if frame is None:
                    st.error("Could not read image. Please try another file.")
                else:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = img_rgb.shape

                    # ── Stats row
                    s1,s2,s3 = st.columns(3)
                    s1.metric("Width (px)", w)
                    s2.metric("Height (px)", h)
                    s3.metric("Channels", ch)

                    if "Environmental" in cv_mode:
                        # ─── ENVIRONMENTAL DETECTION ───
                        col_cv1, col_cv2 = st.columns(2)
                        with col_cv1:
                            st.markdown("**Original Image**")
                            st.image(img_rgb, use_container_width=True)

                        from cv_module.environmental_vision import EnvironmentalVisionPipeline
                        pipeline = EnvironmentalVisionPipeline()
                        with st.spinner("Running environmental CV analysis …"):
                            results = pipeline.analyse_frame(frame)
                            report  = pipeline.summary_report(results)

                        with col_cv2:
                            st.markdown("**Detection Results**")
                            for r in results:
                                st.metric(r.issue_type, f"{r.confidence:.0%}",
                                          delta=r.severity.upper(),
                                          delta_color="inverse" if r.confidence > 0.3 else "normal")
                                if r.alert_message:
                                    st.warning(r.alert_message)

                        sev = report["highest_severity"]
                        st.markdown(f"### Overall Threat Level: "
                                    f"{'🔴 CRITICAL' if sev=='critical' else '🟠 HIGH' if sev=='high' else '🟡 MEDIUM' if sev=='medium' else '🟢 LOW'}")

                        if results and results[0].annotated_frame is not None:
                            ann = results[0].annotated_frame
                            st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
                                     caption="Annotated Detection", use_container_width=True)

                    else:
                        # ─── IMAGE FILTERS & ANALYSIS ───
                        ops = st.multiselect(
                            "Select CV Operations:",
                            ["Original","Grayscale","Edge Detection (Canny)","Blur (Gaussian)",
                             "Sharpen","Emboss","Threshold (Binary)","Sepia Filter","Invert Colors"],
                            default=["Original","Grayscale","Edge Detection (Canny)"]
                        )

                        def sepia_filter(img):
                            k = np.array([[0.272,0.534,0.131],[0.349,0.686,0.168],[0.393,0.769,0.189]])
                            return np.clip(img @ k.T, 0, 255).astype(np.uint8)

                        results_map = {}
                        for op in ops:
                            if op == "Original":            results_map[op] = img_rgb
                            elif op == "Grayscale":         results_map[op] = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                            elif op == "Edge Detection (Canny)":
                                g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY); results_map[op] = cv2.Canny(g,100,200)
                            elif op == "Blur (Gaussian)":   results_map[op] = cv2.GaussianBlur(img_rgb,(15,15),0)
                            elif op == "Sharpen":
                                k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]); results_map[op] = cv2.filter2D(img_rgb,-1,k)
                            elif op == "Emboss":
                                k = np.array([[-2,-1,0],[-1,1,1],[0,1,2]])
                                g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY); results_map[op] = cv2.filter2D(g,-1,k)
                            elif op == "Threshold (Binary)":
                                g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY); _, results_map[op] = cv2.threshold(g,127,255,cv2.THRESH_BINARY)
                            elif op == "Sepia Filter":      results_map[op] = sepia_filter(img_rgb)
                            elif op == "Invert Colors":     results_map[op] = cv2.bitwise_not(img_rgb)

                        if results_map:
                            n_cols = min(3, len(results_map))
                            cols = st.columns(n_cols)
                            for i, (name, img_out) in enumerate(results_map.items()):
                                with cols[i % n_cols]:
                                    st.markdown(f"**{name}**")
                                    st.image(img_out, use_container_width=True)

                        # Color Histogram
                        st.markdown("### 🎨 Color Histogram")
                        fig, ax = plt.subplots(figsize=(10, 3))
                        fig.patch.set_facecolor("#1E2130"); ax.set_facecolor("#1E2130")
                        for i, color in enumerate(['red','green','blue']):
                            hist = cv2.calcHist([img_rgb],[i],None,[256],[0,256])
                            ax.plot(hist, color=color, linewidth=1.5, alpha=0.9, label=color.capitalize())
                        ax.set_title("RGB Color Histogram", color="white"); ax.tick_params(colors="white")
                        ax.legend(facecolor="#1E2130", labelcolor="white")
                        for sp in ax.spines.values(): sp.set_color("#444")
                        buf = io.BytesIO(); plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#1E2130")
                        buf.seek(0); st.image(Image.open(buf), use_container_width=True); plt.close()

                        # Pixel Stats
                        st.markdown("### 📊 Pixel Statistics")
                        stats_df = pd.DataFrame({
                            "Channel": ["Red","Green","Blue"],
                            "Mean": [img_rgb[:,:,i].mean().round(2) for i in range(3)],
                            "Std Dev": [img_rgb[:,:,i].std().round(2) for i in range(3)],
                            "Min": [int(img_rgb[:,:,i].min()) for i in range(3)],
                            "Max": [int(img_rgb[:,:,i].max()) for i in range(3)],
                        })
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)

            except ImportError:
                st.error("OpenCV not installed. Run: pip install opencv-python")
        else:
            st.info("Upload an environmental image to run CV analysis.")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🔍 Environmental Detection Capabilities**")
                st.dataframe(pd.DataFrame([
                    {"Module":"Smoke Detection",        "Method":"HSV + Laplacian Blur","Accuracy":"~87%"},
                    {"Module":"Garbage Detection",      "Method":"Color + Edge Density", "Accuracy":"~82%"},
                    {"Module":"Deforestation Detection","Method":"NDVI Proxy (HSV Green)","Accuracy":"~79%"},
                ]), use_container_width=True, hide_index=True)
            with c2:
                st.markdown("**🎨 Image Filter Operations**")
                st.markdown("""
                - 🔲 Grayscale conversion
                - 🔍 Canny Edge Detection
                - 🌫️ Gaussian Blur
                - ✏️ Sharpening & Emboss
                - 🎭 Sepia & Invert filters
                - 📐 Binary Thresholding
                - 📊 RGB Color Histogram
                """)

    # ═══════════════════════════════
    #  TAB 5 – GLOBAL MAP
    # ═══════════════════════════════
    with tab_map:
        st.subheader("🗺️ Global Sustainability Map")
        scored_df = compute_scores(scorer, df_filtered)
        latest_idx = scored_df.groupby("country")["year"].idxmax()
        map_df = scored_df.loc[latest_idx].copy()
        iso_lookup = df[["country","iso_code"]].drop_duplicates()
        map_df = map_df.merge(iso_lookup, on="country", how="left")

        map_metric = st.selectbox("Map Metric", ["sustainability_score", target_metric])
        if map_metric == "sustainability_score":
            color_col = "sustainability_score"; color_scale = "RdYlGn"; title = "Sustainability Score (0–100)"
        else:
            latest_data = df.loc[df.groupby("country")["year"].idxmax()]
            if map_metric in latest_data.columns:
                map_df = map_df.merge(latest_data[["country", map_metric]], on="country", how="left")
                color_col = map_metric
            else:
                color_col = "sustainability_score"
            color_scale = "Reds"; title = map_metric.replace("_"," ").title()

        plot_df = map_df.dropna(subset=["iso_code", color_col])
        if len(plot_df) > 0:
            fig_map = px.choropleth(plot_df, locations="iso_code", color=color_col,
                                    hover_name="country",
                                    hover_data={"sustainability_score":True,"iso_code":False},
                                    color_continuous_scale=color_scale,
                                    title=f"🌍 Global {title}", template="plotly_dark")
            fig_map.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                  geo=dict(bgcolor="rgba(0,0,0,0)", showframe=False),
                                  height=550, coloraxis_colorbar=dict(title=title))
            st.plotly_chart(fig_map, use_container_width=True)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**✅ Top 10 Most Sustainable**")
            top10 = map_df.nlargest(10,"sustainability_score")[["country","sustainability_score"]]
            fig_top = px.bar(top10, x="sustainability_score", y="country", orientation="h",
                             color="sustainability_score", color_continuous_scale="Greens", template="plotly_dark")
            fig_top.update_layout(paper_bgcolor="rgba(0,0,0,0)", showlegend=False, height=350)
            st.plotly_chart(fig_top, use_container_width=True)
        with col_m2:
            st.markdown("**🚨 Bottom 10 Countries**")
            bot10 = map_df.nsmallest(10,"sustainability_score")[["country","sustainability_score"]]
            fig_bot = px.bar(bot10, x="sustainability_score", y="country", orientation="h",
                             color="sustainability_score", color_continuous_scale="Reds_r", template="plotly_dark")
            fig_bot.update_layout(paper_bgcolor="rgba(0,0,0,0)", showlegend=False, height=350)
            st.plotly_chart(fig_bot, use_container_width=True)

    # ═══════════════════════════════
    #  TAB 6 – LEADERBOARD
    # ═══════════════════════════════
    with tab_leaderboard:
        st.subheader("🏆 Global Sustainability Leaderboard")
        scored_df = compute_scores(scorer, df)
        board = scorer.leaderboard(scored_df)

        def style_grade(val):
            colours = {"A":"#2E8B57","B":"#4CAF50","C":"#FFD700","D":"#FF8C00","F":"#FF3333"}
            return f"color: {colours.get(val,'white')}; font-weight: bold"

        st.dataframe(board.style.map(style_grade, subset=["grade"]),
                     use_container_width=True, height=500)

        st.subheader("📊 Score Distribution")
        fig_hist = px.histogram(board, x="sustainability_score", nbins=30,
                                color_discrete_sequence=[PRIMARY], template="plotly_dark",
                                labels={"sustainability_score":"Sustainability Score"})
        fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300)
        st.plotly_chart(fig_hist, use_container_width=True)

    # ═══════════════════════════════
    #  TAB 7 – CHATBOT (NEW)
    # ═══════════════════════════════
    with tab_chat:
        st.subheader("💬 Sustainability AI Chatbot")
        st.markdown("Ask me anything about CO₂ emissions, sustainability scores, trends, or countries!")

        # Initialise chatbot and history
        if "chatbot" not in st.session_state:
            st.session_state.chatbot = SustainabilityChatbot(df)
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                ("bot", ("👋 Hello! I'm your **Sustainability AI Assistant**.\n\n"
                          "Try asking:\n"
                          "- *CO₂ emissions of India?*\n"
                          "- *Top 5 polluters?*\n"
                          "- *Sustainability score for Germany?*\n"
                          "- *Global CO₂ trend?*\n"
                          "- *Compare India vs China?*"))
            ]

        # Quick-question buttons
        st.markdown("**💡 Quick Questions:**")
        qcols = st.columns(4)
        quick_qs = [
            "Top 5 polluters?",
            f"CO₂ of {selected_country}?",
            "Global CO₂ trend?",
            "Cleanest countries?",
        ]
        for i, qc in enumerate(qcols):
            if qc.button(quick_qs[i], use_container_width=True, key=f"qb_{i}"):
                bot_reply = st.session_state.chatbot.answer(quick_qs[i])
                st.session_state.chat_history.append(("user", quick_qs[i]))
                st.session_state.chat_history.append(("bot", bot_reply))
                st.rerun()

        st.markdown("---")

        # Chat display
        for role, msg in st.session_state.chat_history:
            css = "cb-user" if role == "user" else "cb-bot"
            # convert markdown bold to html for display
            html_msg = msg.replace("\n", "<br>")
            st.markdown(f'<div class="{css}">{html_msg}</div><div class="cf"></div>',
                        unsafe_allow_html=True)

        st.markdown("")

        # Input row
        ci, cb = st.columns([5, 1])
        with ci:
            user_input = st.text_input(
                "Ask …", key="chat_input", label_visibility="collapsed",
                placeholder="e.g. CO₂ emissions of Brazil? / Top 10 polluters? / Germany sustainability score?"
            )
        with cb:
            send = st.button("Send 🚀", use_container_width=True)

        if send and user_input.strip():
            bot_reply = st.session_state.chatbot.answer(user_input)
            st.session_state.chat_history.append(("user", user_input))
            st.session_state.chat_history.append(("bot", bot_reply))
            st.rerun()

        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = [("bot","👋 Chat cleared! Ask me about sustainability data.")]
            st.rerun()

    # ═══════════════════════════════
    #  TAB 8 – WEATHER PREDICTION (NEW)
    # ═══════════════════════════════
    with tab_weather:
        st.subheader("🌦️ AI Weather Prediction & Climate Analysis")
        st.info(
            "Weather data is generated from CO₂/GHG climate drivers using "
            "scientifically-grounded models. Forecasts use Random Forest ensemble."
        )

        wp_country = st.selectbox(
            "🌍 Select Country for Weather Analysis",
            sorted(df["country"].unique()),
            index=list(sorted(df["country"].unique())).index("India")
                  if "India" in df["country"].unique() else 0,
            key="wp_country"
        )

        forecast_months = st.slider("📅 Forecast Horizon (months)", 6, 36, 24, key="wp_months")

        wp_col1, wp_col2 = st.columns([1, 1])
        with wp_col1:
            run_wp = st.button("🚀 Run Weather Prediction", use_container_width=True, type="primary")
        with wp_col2:
            wp_tab_sel = st.selectbox(
                "View",
                ["📈 Temperature Trend", "🌧️ Precipitation", "💧 Humidity",
                 "📅 Seasonal Pattern", "⚡ Extreme Events"],
                key="wp_view"
            )

        if run_wp or ("wp_engine" in st.session_state and
                      st.session_state.get("wp_last_country") == wp_country):

            if run_wp or "wp_engine" not in st.session_state or \
               st.session_state.get("wp_last_country") != wp_country:
                with st.spinner(f"Generating weather data for {wp_country} …"):
                    engine = WeatherPredictionEngine()
                    wdf = engine.prepare(df, wp_country)
                    if wdf.empty:
                        st.error(f"No CO₂ data available for {wp_country}.")
                        st.stop()
                    engine.train_all()
                    forecasts = engine.forecast_all(forecast_months)
                    st.session_state.wp_engine    = engine
                    st.session_state.wp_wdf       = wdf
                    st.session_state.wp_forecasts = forecasts
                    st.session_state.wp_last_country = wp_country
                st.success(f"✅ Weather model trained for {wp_country}!")

            engine    = st.session_state.wp_engine
            wdf       = st.session_state.wp_wdf
            forecasts = st.session_state.wp_forecasts

            # ── KPI row ─────────────────────────────
            yearly = engine.yearly_trend()
            if len(yearly) >= 2:
                temp_change = yearly["avg_temp"].iloc[-1] - yearly["avg_temp"].iloc[0]
                latest_temp = round(yearly["avg_temp"].iloc[-1], 1)
                avg_rain    = round(yearly["precipitation"].mean(), 0)
                avg_humid   = round(engine.weather_df["humidity"].mean(), 1)
            else:
                temp_change, latest_temp, avg_rain, avg_humid = 0, 0, 0, 0

            wk1, wk2, wk3, wk4 = st.columns(4)
            wk1.markdown(f"""
            <div class="metric-card">
              <div class="metric-value" style="color:#FF6B35">{latest_temp}°C</div>
              <div class="metric-label">🌡️ Latest Avg Temp</div>
            </div>""", unsafe_allow_html=True)
            wk2.markdown(f"""
            <div class="metric-card">
              <div class="metric-value" style="color:{'#FF3333' if temp_change > 0 else '#2E8B57'}">
                {'+' if temp_change > 0 else ''}{temp_change:.2f}°C</div>
              <div class="metric-label">🔥 Warming Since 1990</div>
            </div>""", unsafe_allow_html=True)
            wk3.markdown(f"""
            <div class="metric-card">
              <div class="metric-value" style="color:#1E90FF">{avg_rain:.0f} mm</div>
              <div class="metric-label">🌧️ Avg Annual Rain</div>
            </div>""", unsafe_allow_html=True)
            wk4.markdown(f"""
            <div class="metric-card">
              <div class="metric-value" style="color:#9B59B6">{avg_humid}%</div>
              <div class="metric-label">💧 Avg Humidity</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── View selector ───────────────────────
            if "Temperature" in wp_tab_sel:
                # Historic yearly trend
                fig_temp = go.Figure()
                fig_temp.add_trace(go.Scatter(
                    x=yearly["year"], y=yearly["avg_temp"],
                    name="Historical Avg Temp", mode="lines+markers",
                    line=dict(color=PRIMARY, width=2),
                    fill="tozeroy", fillcolor="rgba(46,139,87,0.15)"
                ))
                if "avg_temp" in forecasts:
                    fc_temp = forecasts["avg_temp"]
                    # Aggregate monthly forecast to yearly
                    fc_yearly = fc_temp.groupby("year")[f"predicted_avg_temp"].mean().reset_index()
                    fig_temp.add_trace(go.Scatter(
                        x=fc_yearly["year"], y=fc_yearly["predicted_avg_temp"],
                        name=f"Forecast ({forecast_months} months)",
                        mode="lines+markers",
                        line=dict(color=SECONDARY, width=2, dash="dash"),
                    ))
                    # Confidence band
                    std_val = fc_temp["predicted_avg_temp"].std()
                    fig_temp.add_trace(go.Scatter(
                        x=list(fc_yearly["year"]) + list(fc_yearly["year"])[::-1],
                        y=list(fc_yearly["predicted_avg_temp"] + std_val) +
                          list(fc_yearly["predicted_avg_temp"] - std_val)[::-1],
                        fill="toself", fillcolor="rgba(255,107,53,0.15)",
                        line=dict(color="rgba(255,255,255,0)"),
                        name="Confidence Band", showlegend=True
                    ))

                fig_temp.update_layout(
                    title=f"🌡️ {wp_country} — Temperature Trend & Forecast",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=420,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    xaxis_title="Year", yaxis_title="Avg Temperature (°C)"
                )
                st.plotly_chart(fig_temp, use_container_width=True)

                # Monthly detail for latest year
                latest_yr = int(wdf["year"].max())
                monthly = wdf[wdf["year"] == latest_yr].sort_values("month")
                month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                               "Jul","Aug","Sep","Oct","Nov","Dec"]
                monthly["month_name"] = monthly["month"].apply(lambda m: month_names[m-1])
                fig_monthly = px.bar(
                    monthly, x="month_name", y="avg_temp",
                    color="avg_temp", color_continuous_scale="RdYlBu_r",
                    template="plotly_dark",
                    title=f"📅 Monthly Temperature — {latest_yr}",
                    labels={"avg_temp": "Avg Temp (°C)", "month_name": "Month"}
                )
                fig_monthly.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=320, coloraxis_showscale=False
                )
                st.plotly_chart(fig_monthly, use_container_width=True)

            elif "Precipitation" in wp_tab_sel:
                fig_rain = go.Figure()
                fig_rain.add_trace(go.Bar(
                    x=yearly["year"], y=yearly["precipitation"],
                    name="Annual Rainfall (mm)",
                    marker_color=PRIMARY, opacity=0.8
                ))
                if "precipitation" in forecasts:
                    fc_rain = forecasts["precipitation"]
                    fc_r_yr = fc_rain.groupby("year")["predicted_precipitation"].sum().reset_index()
                    fig_rain.add_trace(go.Bar(
                        x=fc_r_yr["year"], y=fc_r_yr["predicted_precipitation"],
                        name="Forecast", marker_color=SECONDARY, opacity=0.7
                    ))
                fig_rain.update_layout(
                    title=f"🌧️ {wp_country} — Annual Precipitation & Forecast",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=420, barmode="overlay",
                    xaxis_title="Year", yaxis_title="Precipitation (mm/year)"
                )
                st.plotly_chart(fig_rain, use_container_width=True)

                # Monthly breakdown
                seasonal = engine.seasonal_summary()
                month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                               "Jul","Aug","Sep","Oct","Nov","Dec"]
                fig_seas = px.area(
                    seasonal, x="month_name", y="precipitation",
                    color_discrete_sequence=["#1E90FF"],
                    template="plotly_dark",
                    title="🗓️ Average Monthly Rainfall Pattern",
                    labels={"precipitation": "Avg Rain (mm)", "month_name": "Month"}
                )
                fig_seas.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320
                )
                st.plotly_chart(fig_seas, use_container_width=True)

            elif "Humidity" in wp_tab_sel:
                yearly2 = engine.yearly_trend()
                fig_hum = go.Figure()
                fig_hum.add_trace(go.Scatter(
                    x=yearly2["year"], y=yearly2["humidity"],
                    name="Avg Humidity (%)", mode="lines+markers",
                    line=dict(color="#9B59B6", width=2),
                    fill="tozeroy", fillcolor="rgba(155,89,182,0.15)"
                ))
                if "humidity" in forecasts:
                    fc_hum = forecasts["humidity"]
                    fc_h_yr = fc_hum.groupby("year")["predicted_humidity"].mean().reset_index()
                    fig_hum.add_trace(go.Scatter(
                        x=fc_h_yr["year"], y=fc_h_yr["predicted_humidity"],
                        name="Forecast", mode="lines+markers",
                        line=dict(color=SECONDARY, width=2, dash="dash")
                    ))
                fig_hum.update_layout(
                    title=f"💧 {wp_country} — Humidity Trend & Forecast",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=420, xaxis_title="Year", yaxis_title="Humidity (%)"
                )
                st.plotly_chart(fig_hum, use_container_width=True)

            elif "Seasonal" in wp_tab_sel:
                seasonal = engine.seasonal_summary()
                fig_rad = go.Figure()
                categories = list(seasonal["month_name"]) + [seasonal["month_name"].iloc[0]]
                for col, name, color in [
                    ("avg_temp", "Avg Temp (°C)", PRIMARY),
                    ("humidity", "Humidity (%)", "#9B59B6"),
                ]:
                    vals = list(seasonal[col]) + [seasonal[col].iloc[0]]
                    fig_rad.add_trace(go.Scatterpolar(
                        r=vals, theta=categories,
                        fill="toself", name=name,
                        line_color=color, opacity=0.7
                    ))
                fig_rad.update_layout(
                    polar=dict(radialaxis=dict(visible=True)),
                    showlegend=True, template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    title=f"🧭 {wp_country} — Seasonal Climate Pattern",
                    height=450
                )
                st.plotly_chart(fig_rad, use_container_width=True)

                st.markdown("**📋 Monthly Climate Table**")
                st.dataframe(
                    seasonal[["month_name","avg_temp","precipitation","humidity","wind_speed","pressure"]].rename(columns={
                        "month_name":"Month","avg_temp":"Temp(°C)",
                        "precipitation":"Rain(mm)","humidity":"Humidity(%)","wind_speed":"Wind(km/h)","pressure":"Pressure(hPa)"
                    }),
                    use_container_width=True, hide_index=True
                )

            elif "Extreme" in wp_tab_sel:
                events = engine.extreme_events()
                st.markdown(f"### ⚡ Extreme Weather Events — {wp_country}")
                if len(events) > 0:
                    ec1, ec2, ec3 = st.columns(3)
                    heat_cnt = (events["Event"].str.contains("Heat")).sum()
                    cold_cnt = (events["Event"].str.contains("Cold")).sum()
                    rain_cnt = (events["Event"].str.contains("Rain")).sum()
                    ec1.metric("🔥 Extreme Heat Events", heat_cnt)
                    ec2.metric("🥶 Extreme Cold Events", cold_cnt)
                    ec3.metric("🌊 Heavy Rain Events",   rain_cnt)
                    st.dataframe(events.head(30), use_container_width=True, hide_index=True)

                    # Events per year
                    events_by_year = events.groupby("Year").size().reset_index(name="Count")
                    fig_ev = px.bar(
                        events_by_year, x="Year", y="Count",
                        color="Count", color_continuous_scale="Reds",
                        template="plotly_dark",
                        title="📊 Extreme Events Per Year"
                    )
                    fig_ev.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        height=300, coloraxis_showscale=False
                    )
                    st.plotly_chart(fig_ev, use_container_width=True)
                else:
                    st.success("✅ No extreme events detected in the historical record.")

        else:
            st.info("👆 Click **Run Weather Prediction** to generate forecasts for the selected country.")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🌡️ Weather Variables Modelled**")
                st.dataframe(pd.DataFrame([
                    {"Variable": "Avg Temperature", "Unit": "°C",  "Model": "Random Forest"},
                    {"Variable": "Precipitation",   "Unit": "mm",  "Model": "Random Forest"},
                    {"Variable": "Humidity",         "Unit": "%",   "Model": "Random Forest"},
                    {"Variable": "Wind Speed",       "Unit": "km/h","Model": "Climatological"},
                    {"Variable": "Pressure",         "Unit": "hPa", "Model": "Climatological"},
                ]), use_container_width=True, hide_index=True)
            with c2:
                st.markdown("**📊 Analysis Capabilities**")
                st.markdown("""
                - 📈 Historical trend analysis (1990–present)
                - 🔮 6–36 month weather forecasting
                - 🧭 Seasonal climatology (radar chart)
                - ⚡ Extreme event detection (2σ threshold)
                - 🌡️ Warming trend quantification
                - 🌧️ Monsoon pattern visualization
                """)

    # ─── Footer ──────────────────────────────
    st.divider()
    st.markdown("""
    <div style="text-align:center; color:#666; font-size:0.8rem;">
      🌍 AI-Powered Sustainability Intelligence System v3 <br>
      Data: OWID CO₂ Dataset · Models: Isolation Forest · Random Forest · GBM · CV: OpenCV · Weather: RF Climate Engine
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
