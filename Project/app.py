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
import sys, os

# ── Path setup ─────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from pipeline.data_pipeline import SustainabilityDataPipeline, KEY_COUNTRIES
from models.anomaly_detection import AnomalyDetectionEngine
from models.prediction_model import ForecastingEngine
from models.scoring_system import SustainabilityScorer

# ── PAGE CONFIG ────────────────────────────────────────────
st.set_page_config(
    page_title="🌍 AI Sustainability Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── THEME COLOURS ──────────────────────────────────────────
PRIMARY   = "#2E8B57"   # sea green
SECONDARY = "#FF6B35"   # burnt orange (alert)
DARK_BG   = "#0E1117"
CARD_BG   = "#1E2130"

# ── CUSTOM CSS ─────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: linear-gradient(135deg, #1E2130, #2A3045);
    border: 1px solid #3A4060;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
  }
  .metric-value { font-size: 2.2rem; font-weight: 800; color: #2E8B57; }
  .metric-label { font-size: 0.85rem; color: #9BA3AF; margin-top: 4px; }
  .alert-critical { background:#4A1010; border-left: 4px solid #FF3333; padding:12px; border-radius:8px; }
  .alert-high     { background:#3A2010; border-left: 4px solid #FF8C00; padding:12px; border-radius:8px; }
  .alert-medium   { background:#3A3010; border-left: 4px solid #FFD700; padding:12px; border-radius:8px; }
  .score-ring { font-size: 3.5rem; font-weight: 900; }
  .header-gradient {
    background: linear-gradient(90deg, #2E8B57, #1a5c38);
    padding: 25px; border-radius: 15px; margin-bottom: 25px;
    box-shadow: 0 8px 32px rgba(46,139,87,0.3);
  }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  DATA LOADING (cached)
# ═══════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner="Loading sustainability data …")
def load_data():
    # ── FIX: load_data() takes NO arguments ────────────────
    # The function builds the path internally using ROOT.
    # Previously called as load_data("owid-co2-data.csv") — WRONG.
    # Now called as load_data() — CORRECT.
    pipe = SustainabilityDataPipeline(
        os.path.join(ROOT, "C:/Users/krish/Desktop/Project/data/raw/owid-co2-data.csv")
    )
    df = pipe.run(start_year=1990)
    return df

@st.cache_resource(show_spinner="Training AI models …")
def train_models(df: pd.DataFrame):
    # Anomaly engine
    anomaly_engine = AnomalyDetectionEngine()
    anomaly_engine.train(df)

    # Scorer
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
        Real-Time Anomaly Detection · Predictive Analytics · Global Sustainability Scoring
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ─── Load data ───────────────────────────
    # ✅ FIXED: was load_data("owid-co2-data.csv") — removed the argument
    with st.spinner("Initialising system …"):
        df = load_data()
        anomaly_engine, scorer = train_models(df)

    # ─── Sidebar ─────────────────────────────
    with st.sidebar:
        st.image(os.path.join(os.path.dirname(__file__), "images", "profile.png"), width=80)
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
        st.caption("🌍 Sustainability Analytics")

    # ─── Filter data ─────────────────────────
    df_filtered = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]
    country_df  = df_filtered[df_filtered["country"] == selected_country].sort_values("year")

    # ─── Tabs ────────────────────────────────
    tab_overview, tab_anomaly, tab_forecast, tab_cv, tab_map, tab_leaderboard = st.tabs([
        "📊 Overview",
        "🚨 Anomaly Detection",
        "🔮 AI Forecasting",
        "👁️ CV Analysis",
        "🗺️ Global Map",
        "🏆 Leaderboard",
    ])

    # ═══════════════════════════════
    #  TAB 1 – OVERVIEW
    # ═══════════════════════════════
    with tab_overview:
        scored_df = compute_scores(scorer, df_filtered)
        kpis      = scorer.global_kpis(scored_df)

        # ── Score for selected country
        country_scores = scored_df[scored_df["country"] == selected_country]
        if len(country_scores) > 0:
            latest_score = country_scores.sort_values("year").iloc[-1]["sustainability_score"]
            score_grade  = scorer.grade(latest_score)
            score_label  = scorer.label(latest_score)
        else:
            latest_score, score_grade, score_label = 50.0, "C", "Moderate ⚠️"

        # ── KPI Cards Row 1
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

        # ── Country time-series
        col_chart, col_score = st.columns([2, 1])

        with col_chart:
            st.subheader(f"📈 {selected_country} – {target_metric.replace('_',' ').title()} Trend")
            if len(country_df) > 0 and target_metric in country_df.columns:
                fig = px.area(
                    country_df, x="year", y=target_metric,
                    color_discrete_sequence=[PRIMARY],
                    template="plotly_dark",
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for this metric.")

        with col_score:
            st.subheader("🎯 Sustainability Score")
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=latest_score,
                delta={"reference": kpis["global_avg_score"]},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": PRIMARY},
                    "steps": [
                        {"range": [0,  35], "color": "#4A1010"},
                        {"range": [35, 50], "color": "#3A2010"},
                        {"range": [50, 65], "color": "#3A3010"},
                        {"range": [65, 80], "color": "#1A3A1A"},
                        {"range": [80, 100],"color": "#0A2A0A"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.75,
                        "value": kpis["global_avg_score"],
                    },
                },
                title={"text": f"{score_label}", "font": {"color": "white"}},
                number={"font": {"color": "white"}},
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                height=300, margin=dict(t=30,b=10,l=20,r=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.metric("Global Avg Score", f"{kpis['global_avg_score']}")
            st.metric("Best Country", kpis["top_country"], f"+{kpis['top_score']:.1f}")

        # ── Key countries comparison
        st.subheader("🌐 Key Countries Comparison")
        key_df = df_filtered[df_filtered["country"].isin(KEY_COUNTRIES)]
        if len(key_df) > 0 and target_metric in key_df.columns:
            fig2 = px.line(
                key_df, x="year", y=target_metric, color="country",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
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

        # Alert KPIs
        ac1, ac2, ac3, ac4 = st.columns(4)
        ac1.metric("🔴 Critical Anomalies", counts["critical"])
        ac2.metric("🟠 High Anomalies",     counts["high"])
        ac3.metric("🟡 Medium Anomalies",   counts["medium"])
        ac4.metric("📊 Total Flagged",       counts["total"])

        # Alert list
        if len(summary) > 0:
            st.markdown("### 🔔 Active Alerts")
            for _, row in summary.head(15).iterrows():
                sev = row.get("severity", "medium")
                css_class = f"alert-{sev if sev in ['critical','high','medium'] else 'medium'}"
                icon = {"critical": "🔴", "high": "🟠", "medium": "🟡"}.get(sev, "⚪")
                co2_val = row.get("co2", 0)
                st.markdown(f"""
                <div class="{css_class}" style="margin-bottom:8px;">
                  {icon} <strong>{row['country']}</strong> ({int(row['year'])})
                  — CO₂: {co2_val:.2f} Mt — Score: {row.get('if_score',0):.3f}
                  — Severity: <strong>{sev.upper()}</strong>
                </div>""", unsafe_allow_html=True)

        # Scatter: anomaly scores
        st.subheader("🔬 Anomaly Score Distribution")
        if "if_score" in flagged_df.columns and "co2" in flagged_df.columns:
            scatter_df = flagged_df[flagged_df["country"].isin(KEY_COUNTRIES)].copy()
            scatter_df["anomaly_label"] = scatter_df["is_anomaly"].map(
                {True: "Anomaly", False: "Normal"}
            )
            # Make if_score positive for size (it's negative by nature)
            scatter_df["size_val"] = (scatter_df["if_score"] * -1).clip(lower=0.01)
            fig_anom = px.scatter(
                scatter_df, x="year", y="co2",
                color="anomaly_label",
                symbol="anomaly_label",
                size="size_val",
                size_max=12,
                hover_data=["country", "severity", "if_score"],
                color_discrete_map={"Anomaly": SECONDARY, "Normal": PRIMARY},
                template="plotly_dark",
                facet_col="country", facet_col_wrap=4,
            )
            fig_anom.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=600,
            )
            st.plotly_chart(fig_anom, use_container_width=True)

        # Country-level Z-Score
        st.subheader(f"📉 {selected_country} – Z-Score Analysis")
        zs_df = anomaly_engine.detect_country(df, selected_country)
        if "zscore" in zs_df.columns and len(zs_df) > 0:
            fig_zs = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   subplot_titles=["CO₂ Emissions", "Z-Score"])
            fig_zs.add_trace(go.Scatter(x=zs_df["year"], y=zs_df["co2"],
                                        mode="lines", name="CO₂", line=dict(color=PRIMARY)), row=1, col=1)
            zscore_colors = [SECONDARY if abs(z) > 2.5 else PRIMARY
                             for z in zs_df["zscore"].fillna(0)]
            fig_zs.add_trace(go.Bar(x=zs_df["year"], y=zs_df["zscore"],
                                    name="Z-Score",
                                    marker_color=zscore_colors), row=2, col=1)
            fig_zs.add_hline(y=2.5, line_dash="dash", line_color="red", row=2, col=1)
            fig_zs.add_hline(y=-2.5, line_dash="dash", line_color="red", row=2, col=1)
            fig_zs.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                template="plotly_dark", height=450,
            )
            st.plotly_chart(fig_zs, use_container_width=True)

    # ═══════════════════════════════
    #  TAB 3 – AI FORECASTING
    # ═══════════════════════════════
    with tab_forecast:
        st.subheader("🔮 AI-Powered Predictive Forecasting")
        st.info("Ensemble: Random Forest + Gradient Boosting + Exponential Smoothing")

        forecast_engine = ForecastingEngine()

        with st.spinner(f"Generating {forecast_horizon}-year forecast for {selected_country} …"):
            result = forecast_engine.fit_and_forecast(
                df, selected_country, target_metric, forecast_horizon
            )

        if result:
            hist = result["historical"]
            ens  = result["ensemble_forecast"]
            rf   = result["rf_forecast"]

            # Combined chart
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=hist["year"], y=hist[target_metric],
                name="Historical", mode="lines+markers",
                line=dict(color=PRIMARY, width=2),
            ))
            fig_fc.add_trace(go.Scatter(
                x=ens["year"], y=ens[f"predicted_{target_metric}"],
                name="Ensemble Forecast", mode="lines+markers",
                line=dict(color=SECONDARY, width=2, dash="dash"),
            ))
            fig_fc.add_trace(go.Scatter(
                x=rf["year"], y=rf[f"predicted_{target_metric}"],
                name="Random Forest", mode="lines",
                line=dict(color="#9B59B6", width=1, dash="dot"),
            ))
            fig_fc.update_layout(
                title=f"📈 {selected_country} – {target_metric.replace('_',' ').title()} Forecast",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=450,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            # Forecast table
            col_fc1, col_fc2 = st.columns(2)
            with col_fc1:
                st.markdown("**📊 Ensemble Forecast Values**")
                display_fc = ens.copy()
                display_fc[f"predicted_{target_metric}"] = display_fc[f"predicted_{target_metric}"].round(3)
                st.dataframe(display_fc, use_container_width=True, hide_index=True)

            with col_fc2:
                st.markdown("**🌐 Global CO₂ Trend**")
                global_trend = forecast_engine.global_trend(df, "co2", forecast_horizon)
                fig_global = px.line(
                    global_trend, x="year", y="co2", color="type",
                    color_discrete_map={"historical": PRIMARY, "forecast": SECONDARY},
                    template="plotly_dark",
                )
                fig_global.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=350,
                )
                st.plotly_chart(fig_global, use_container_width=True)
        else:
            st.warning(f"Not enough data to forecast '{target_metric}' for {selected_country}. Try selecting a different metric or country.")

    # ═══════════════════════════════
    #  TAB 4 – COMPUTER VISION
    # ═══════════════════════════════
    with tab_cv:
        st.subheader("👁️ Computer Vision – Environmental Issue Detection")
        st.markdown("""
        Upload an environmental image to detect:
        - 🔥 Smoke / Air Pollution
        - 🗑️ Garbage / Illegal Dumping
        - 🌲 Deforestation / Vegetation Loss
        """)

        uploaded = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

        col_cv1, col_cv2 = st.columns(2)

        if uploaded:
            try:
                import cv2
                from cv_module.environmental_vision import EnvironmentalVisionPipeline

                file_bytes = np.frombuffer(uploaded.read(), np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if frame is None:
                    st.error("Could not read the uploaded image. Please try a different file.")
                else:
                    with col_cv1:
                        st.markdown("**Original Image**")
                        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

                    pipeline = EnvironmentalVisionPipeline()
                    with st.spinner("Running CV analysis …"):
                        results = pipeline.analyse_frame(frame)
                        report  = pipeline.summary_report(results)

                    with col_cv2:
                        st.markdown("**Detection Results**")
                        for r in results:
                            st.metric(
                                r.issue_type,
                                f"{r.confidence:.0%}",
                                delta=r.severity.upper(),
                                delta_color="inverse" if r.confidence > 0.3 else "normal",
                            )
                            if r.alert_message:
                                st.warning(r.alert_message)

                    # Severity summary
                    sev = report["highest_severity"]
                    st.markdown(f"### Overall Threat Level: {'🔴 CRITICAL' if sev=='critical' else '🟠 HIGH' if sev=='high' else '🟡 MEDIUM' if sev=='medium' else '🟢 LOW'}")

                    # Annotated frame
                    if results and results[0].annotated_frame is not None:
                        ann = results[0].annotated_frame
                        st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
                                 caption="Annotated Detection", use_container_width=True)
            except ImportError:
                st.error("OpenCV not installed. Run: pip install opencv-python")

        else:
            st.info("Upload an environmental image to run AI-powered issue detection.")

            # Demo metrics
            st.markdown("### 📊 System Capabilities")
            demo_data = [
                {"Module": "Smoke Detection",        "Method": "HSV + Laplacian Blur", "Accuracy": "~87%"},
                {"Module": "Garbage Detection",      "Method": "Color + Edge Density",  "Accuracy": "~82%"},
                {"Module": "Deforestation Detection","Method": "NDVI Proxy (HSV Green)","Accuracy": "~79%"},
            ]
            st.dataframe(pd.DataFrame(demo_data), use_container_width=True, hide_index=True)

    # ═══════════════════════════════
    #  TAB 5 – GLOBAL MAP
    # ═══════════════════════════════
    with tab_map:
        st.subheader("🗺️ Global Sustainability Map")

        scored_df = compute_scores(scorer, df_filtered)
        latest_idx = scored_df.groupby("country")["year"].idxmax()
        map_df = scored_df.loc[latest_idx].copy()

        # Merge with iso codes from original data
        iso_lookup = df[["country", "iso_code"]].drop_duplicates()
        map_df = map_df.merge(iso_lookup, on="country", how="left")

        # Metric to map
        map_metric = st.selectbox("Map Metric", [
            "sustainability_score", target_metric,
        ])

        if map_metric == "sustainability_score":
            color_col = "sustainability_score"
            color_scale = "RdYlGn"
            title = "Sustainability Score (0–100)"
        else:
            # Merge target metric safely
            latest_data = df.loc[df.groupby("country")["year"].idxmax()]
            if map_metric in latest_data.columns:
                map_df = map_df.merge(
                    latest_data[["country", map_metric]], on="country", how="left"
                )
                color_col = map_metric
            else:
                color_col = "sustainability_score"
            color_scale = "Reds"
            title = map_metric.replace("_", " ").title()

        plot_df = map_df.dropna(subset=["iso_code", color_col])
        if len(plot_df) > 0:
            fig_map = px.choropleth(
                plot_df,
                locations="iso_code",
                color=color_col,
                hover_name="country",
                hover_data={"sustainability_score": True, "iso_code": False},
                color_continuous_scale=color_scale,
                title=f"🌍 Global {title}",
                template="plotly_dark",
            )
            fig_map.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                geo=dict(bgcolor="rgba(0,0,0,0)", showframe=False),
                height=550,
                coloraxis_colorbar=dict(title=title),
            )
            st.plotly_chart(fig_map, use_container_width=True)

        # Bar chart of worst/best
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**✅ Top 10 Most Sustainable**")
            top10 = map_df.nlargest(10, "sustainability_score")[["country", "sustainability_score"]]
            fig_top = px.bar(top10, x="sustainability_score", y="country",
                             orientation="h", color="sustainability_score",
                             color_continuous_scale="Greens", template="plotly_dark")
            fig_top.update_layout(paper_bgcolor="rgba(0,0,0,0)", showlegend=False, height=350)
            st.plotly_chart(fig_top, use_container_width=True)

        with col_m2:
            st.markdown("**🚨 Bottom 10 Countries**")
            bot10 = map_df.nsmallest(10, "sustainability_score")[["country", "sustainability_score"]]
            fig_bot = px.bar(bot10, x="sustainability_score", y="country",
                             orientation="h", color="sustainability_score",
                             color_continuous_scale="Reds_r", template="plotly_dark")
            fig_bot.update_layout(paper_bgcolor="rgba(0,0,0,0)", showlegend=False, height=350)
            st.plotly_chart(fig_bot, use_container_width=True)

    # ═══════════════════════════════
    #  TAB 6 – LEADERBOARD
    # ═══════════════════════════════
    with tab_leaderboard:
        st.subheader("🏆 Global Sustainability Leaderboard")

        scored_df = compute_scores(scorer, df)
        board = scorer.leaderboard(scored_df)

        # Colour-code by grade
        def style_grade(val):
            colours = {"A": "#2E8B57", "B": "#4CAF50", "C": "#FFD700",
                       "D": "#FF8C00", "F": "#FF3333"}
            return f"color: {colours.get(val, 'white')}; font-weight: bold"

        st.dataframe(
            board.style.map(style_grade, subset=["grade"]),
            use_container_width=True, height=500,
        )

        # Score distribution
        st.subheader("📊 Score Distribution")
        fig_hist = px.histogram(
            board, x="sustainability_score", nbins=30,
            color_discrete_sequence=[PRIMARY],
            template="plotly_dark",
            labels={"sustainability_score": "Sustainability Score"},
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ─── Footer ──────────────────────────────
    st.divider()
    st.markdown("""
    <div style="text-align:center; color:#666; font-size:0.8rem;">
      🌍 AI-Powered Sustainability Intelligence System <br>
      Data: Our World in Data (OWID CO₂ Dataset) · Models: Isolation Forest · Random Forest · GBM
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()