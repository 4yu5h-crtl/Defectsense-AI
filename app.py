"""
DefectSense-AI: Manufacturing Defect Prediction & GenAI Root Cause Analyzer
Interactive Streamlit Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.data_simulator import generate_sensor_data
from src.feature_engineering import engineer_features, get_feature_columns
from src.model_training import train_model
from src.model_inference import predict_batch, load_model, get_high_risk_readings
from src.genai_analyzer import analyze_defect, analyze_batch
from src.database import init_db, save_sensor_readings, load_sensor_readings, save_predictions

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DefectSense-AI",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        padding: 0.5rem 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .stMetric > div { text-align: center; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
</style>
""", unsafe_allow_html=True)


# ── Session State Initialization ─────────────────────────────────────────────
def init_session_state():
    if "data_generated" not in st.session_state:
        st.session_state.data_generated = False
    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False
    if "predictions_df" not in st.session_state:
        st.session_state.predictions_df = None
    if "raw_df" not in st.session_state:
        st.session_state.raw_df = None
    if "artifact" not in st.session_state:
        st.session_state.artifact = None
    if "genai_reports" not in st.session_state:
        st.session_state.genai_reports = []


init_session_state()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/factory.png", width=64)
    st.markdown("## ⚙️ Control Panel")
    st.markdown("---")

    # Data Generation
    st.markdown("### 📊 Data Simulation")
    num_records = st.slider("Number of Records", 500, 20000, 5000, step=500)
    defect_rate_pct = st.slider("Defect Rate (%)", 5, 25, 12)

    if st.button("🔄 Generate Sensor Data", use_container_width=True):
        config.DEFECT_RATE = defect_rate_pct / 100
        with st.spinner("Simulating sensor data..."):
            df = generate_sensor_data(num_records=num_records)
            st.session_state.raw_df = df
            st.session_state.data_generated = True
            st.session_state.model_trained = False
            st.session_state.predictions_df = None
            st.session_state.genai_reports = []
        st.success(f"✓ {len(df)} readings generated")

    st.markdown("---")

    # Model Training
    st.markdown("### 🤖 ML Model")
    model_type = st.selectbox("Algorithm", ["gradient_boosting", "random_forest"])

    if st.button("🧠 Train Model", use_container_width=True,
                 disabled=not st.session_state.data_generated):
        with st.spinner("Training model... (may take 1-2 min)"):
            artifact = train_model(st.session_state.raw_df, model_type=model_type)
            st.session_state.artifact = artifact
            st.session_state.model_trained = True
        st.success(f"✓ Model trained (F1: {artifact['metrics']['f1_score']:.3f})")

    st.markdown("---")

    # Predictions
    if st.button("🎯 Run Predictions", use_container_width=True,
                 disabled=not st.session_state.model_trained):
        with st.spinner("Scoring sensor data..."):
            result = predict_batch(st.session_state.raw_df, st.session_state.artifact)
            st.session_state.predictions_df = result
        defects = (result["predicted_label"] == 1).sum()
        st.success(f"✓ {defects} defects predicted")

    st.markdown("---")

    # GenAI Analysis
    st.markdown("### 🧬 GenAI Analysis")
    genai_provider = st.selectbox("Provider", ["fallback", "ollama", "gemini"])
    max_analyses = st.slider("Max Analyses", 3, 20, 10)

    if st.button("🔍 Analyze Root Causes", use_container_width=True,
                 disabled=st.session_state.predictions_df is None):
        with st.spinner("Generating root cause analyses..."):
            fi = st.session_state.artifact.get("feature_importance") if st.session_state.artifact else None
            reports = analyze_batch(
                st.session_state.predictions_df,
                feature_importance=fi,
                provider=genai_provider,
                max_items=max_analyses,
            )
            st.session_state.genai_reports = reports
        st.success(f"✓ {len(reports)} analyses generated")

    st.markdown("---")

    # Station Filter
    st.markdown("### 🏗️ Filters")
    station_filter = st.multiselect(
        "Stations",
        config.STATION_IDS,
        default=config.STATION_IDS,
    )


# ── Main Content ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🏭 DefectSense-AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Manufacturing Defect Prediction & GenAI Root Cause Analyzer</div>', unsafe_allow_html=True)

# Quick-start guide if no data
if not st.session_state.data_generated:
    st.info("👈 Use the **Control Panel** to get started:\n"
            "1. **Generate Sensor Data** → Simulate manufacturing readings\n"
            "2. **Train Model** → Build the defect prediction model\n"
            "3. **Run Predictions** → Score sensor data for defects\n"
            "4. **Analyze Root Causes** → Get AI-powered explanations")
    st.stop()

# Filter data by station
raw_df = st.session_state.raw_df
filtered_df = raw_df[raw_df["station_id"].isin(station_filter)]

# ── Tab Layout ───────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Sensor Overview", "🎯 Predictions", "🧬 GenAI Analysis",
    "📊 Model Performance", "🔧 Sensor Health"
])

# ── TAB 1: Sensor Overview ───────────────────────────────────────────────────
with tab1:
    st.markdown("### Real-Time Sensor Monitoring")

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Readings", f"{len(filtered_df):,}")
    with col2:
        st.metric("Defects", f"{filtered_df['is_defect'].sum():,}")
    with col3:
        st.metric("Defect Rate", f"{filtered_df['is_defect'].mean():.1%}")
    with col4:
        st.metric("Avg Temperature", f"{filtered_df['temperature'].mean():.1f}°C")
    with col5:
        st.metric("Avg Vibration", f"{filtered_df['vibration'].mean():.2f} mm/s")

    st.markdown("---")

    # Sensor time-series charts
    col_left, col_right = st.columns(2)

    with col_left:
        fig_temp = px.line(
            filtered_df, x="timestamp", y="temperature", color="station_id",
            title="🌡️ Temperature Over Time",
            labels={"temperature": "Temperature (°C)", "timestamp": ""},
        )
        fig_temp.update_layout(height=350, showlegend=True, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_temp, use_container_width=True)

        fig_pres = px.line(
            filtered_df, x="timestamp", y="pressure", color="station_id",
            title="💨 Pressure Over Time",
            labels={"pressure": "Pressure (PSI)", "timestamp": ""},
        )
        fig_pres.update_layout(height=350, showlegend=True, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_pres, use_container_width=True)

    with col_right:
        fig_vib = px.line(
            filtered_df, x="timestamp", y="vibration", color="station_id",
            title="📳 Vibration Over Time",
            labels={"vibration": "Vibration (mm/s)", "timestamp": ""},
        )
        fig_vib.update_layout(height=350, showlegend=True, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_vib, use_container_width=True)

        fig_cycle = px.line(
            filtered_df, x="timestamp", y="cycle_time", color="station_id",
            title="⏱️ Cycle Time Over Time",
            labels={"cycle_time": "Cycle Time (s)", "timestamp": ""},
        )
        fig_cycle.update_layout(height=350, showlegend=True, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_cycle, use_container_width=True)

    # Defect distribution
    st.markdown("### Defect Distribution by Station")
    defect_stats = filtered_df.groupby("station_id").agg(
        total=("is_defect", "count"),
        defects=("is_defect", "sum"),
    ).reset_index()
    defect_stats["defect_rate"] = defect_stats["defects"] / defect_stats["total"]

    fig_bar = px.bar(
        defect_stats, x="station_id", y="defects",
        color="defect_rate", color_continuous_scale="RdYlGn_r",
        title="Defect Count by Station",
        text="defects",
    )
    fig_bar.update_layout(height=350)
    st.plotly_chart(fig_bar, use_container_width=True)


# ── TAB 2: Predictions ──────────────────────────────────────────────────────
with tab2:
    if st.session_state.predictions_df is None:
        st.warning("Run predictions first using the sidebar.")
    else:
        pred_df = st.session_state.predictions_df
        pred_filtered = pred_df[pred_df["station_id"].isin(station_filter)]

        st.markdown("### Defect Prediction Results")

        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Predicted Defects", f"{(pred_filtered['predicted_label']==1).sum():,}")
        with col2:
            st.metric("High Risk (>70%)", f"{(pred_filtered['defect_probability']>0.7).sum():,}")
        with col3:
            st.metric("Avg Probability", f"{pred_filtered['defect_probability'].mean():.2%}")
        with col4:
            st.metric("Max Probability", f"{pred_filtered['defect_probability'].max():.2%}")

        st.markdown("---")

        # Prediction timeline
        fig_pred = px.scatter(
            pred_filtered, x="timestamp", y="defect_probability",
            color="predicted_label", color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
            size="defect_probability", hover_data=["station_id", "temperature", "vibration"],
            title="Defect Probability Timeline",
            labels={"defect_probability": "Defect Probability", "predicted_label": "Prediction"},
        )
        fig_pred.add_hline(y=0.5, line_dash="dash", line_color="orange",
                           annotation_text="Decision Threshold")
        fig_pred.update_layout(height=400)
        st.plotly_chart(fig_pred, use_container_width=True)

        # Probability distribution
        col_left, col_right = st.columns(2)
        with col_left:
            fig_hist = px.histogram(
                pred_filtered, x="defect_probability", nbins=50,
                color="predicted_label", color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
                title="Prediction Probability Distribution",
                barmode="overlay", opacity=0.7,
            )
            fig_hist.update_layout(height=350)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_right:
            prob_by_station = pred_filtered.groupby("station_id")["defect_probability"].mean().reset_index()
            fig_station = px.bar(
                prob_by_station, x="station_id", y="defect_probability",
                color="defect_probability", color_continuous_scale="RdYlGn_r",
                title="Avg Defect Probability by Station",
            )
            fig_station.update_layout(height=350)
            st.plotly_chart(fig_station, use_container_width=True)

        # High-risk table
        st.markdown("### ⚠️ High-Risk Readings")
        high_risk = get_high_risk_readings(pred_filtered, threshold=0.5)
        if len(high_risk) > 0:
            display_cols = ["station_id", "timestamp", "temperature", "vibration",
                            "pressure", "cycle_time", "defect_probability"]
            st.dataframe(
                high_risk[display_cols].head(50).style.background_gradient(
                    subset=["defect_probability"], cmap="RdYlGn_r"
                ),
                use_container_width=True, height=400,
            )
        else:
            st.success("No high-risk readings detected!")


# ── TAB 3: GenAI Analysis ───────────────────────────────────────────────────
with tab3:
    if not st.session_state.genai_reports:
        st.warning("Generate root cause analyses first using the sidebar.")
    else:
        st.markdown("### 🧬 AI-Powered Root Cause Analysis")
        st.markdown("*Intelligent explanations for predicted manufacturing defects*")
        st.markdown("---")

        for i, report in enumerate(st.session_state.genai_reports):
            confidence = report.get("confidence", 0)
            severity = "🔴" if confidence > 0.85 else "🟠" if confidence > 0.65 else "🟡"

            with st.expander(
                f"{severity} {report.get('station_id', 'N/A')} — "
                f"{report.get('timestamp', 'N/A')} — "
                f"Confidence: {confidence:.0%}",
                expanded=(i < 3),
            ):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("#### 🔍 Root Cause")
                    st.info(report.get("root_cause", "N/A"))

                    st.markdown("#### 📋 Explanation")
                    st.write(report.get("explanation", "N/A"))

                    st.markdown("#### ✅ Recommendations")
                    recs = report.get("recommendations", "").split(";")
                    for rec in recs:
                        rec = rec.strip()
                        if rec:
                            st.markdown(f"- {rec}")

                with col2:
                    # Confidence gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=confidence * 100,
                        title={"text": "Defect Confidence"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#e74c3c" if confidence > 0.8 else "#f39c12"},
                            "steps": [
                                {"range": [0, 50], "color": "#d5f5e3"},
                                {"range": [50, 75], "color": "#fdebd0"},
                                {"range": [75, 100], "color": "#fadbd8"},
                            ],
                        },
                    ))
                    fig_gauge.update_layout(height=250, margin=dict(t=40, b=0, l=30, r=30))
                    st.plotly_chart(fig_gauge, use_container_width=True)


# ── TAB 4: Model Performance ────────────────────────────────────────────────
with tab4:
    if not st.session_state.model_trained:
        st.warning("Train a model first using the sidebar.")
    else:
        metrics = st.session_state.artifact["metrics"]
        feat_imp = st.session_state.artifact["feature_importance"]

        st.markdown("### Model Performance Metrics")

        # Metrics cards
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.3f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.3f}")
        with col4:
            st.metric("F1 Score", f"{metrics['f1_score']:.3f}")
        with col5:
            st.metric("AUC-ROC", f"{metrics['auc_roc']:.3f}")

        st.markdown("---")

        col_left, col_right = st.columns(2)

        with col_left:
            # Confusion Matrix
            cm = np.array(metrics["confusion_matrix"])
            fig_cm = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Normal", "Defect"], y=["Normal", "Defect"],
                title="Confusion Matrix",
                color_continuous_scale="Blues",
            )
            fig_cm.update_layout(height=400)
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_right:
            # Feature importance
            fi_df = pd.DataFrame(feat_imp, columns=["Feature", "Importance"])
            fig_fi = px.bar(
                fi_df.head(15), x="Importance", y="Feature",
                orientation="h", title="Top 15 Feature Importances",
                color="Importance", color_continuous_scale="Viridis",
            )
            fig_fi.update_layout(height=400, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_fi, use_container_width=True)

        # CV scores
        st.markdown(f"**Cross-Validation F1:** {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")


# ── TAB 5: Sensor Health ────────────────────────────────────────────────────
with tab5:
    st.markdown("### Sensor Health Monitoring")
    st.markdown("*Real-time sensor status across all production stations*")

    sensor_metrics = {
        "temperature": {"unit": "°C", "min": 60, "max": 100, "warn": 85},
        "vibration": {"unit": "mm/s", "min": 0, "max": 10, "warn": 5},
        "pressure": {"unit": "PSI", "min": 10, "max": 50, "warn_low": 20},
        "cycle_time": {"unit": "s", "min": 30, "max": 80, "warn": 60},
    }

    for station in station_filter:
        station_data = filtered_df[filtered_df["station_id"] == station]
        if len(station_data) == 0:
            continue

        latest = station_data.iloc[-1]

        with st.expander(f"🏗️ {station}", expanded=True):
            cols = st.columns(4)
            for i, (sensor, meta) in enumerate(sensor_metrics.items()):
                val = latest[sensor]
                avg = station_data[sensor].mean()
                std = station_data[sensor].std()

                with cols[i]:
                    # Status indicator
                    if sensor == "pressure":
                        status = "🟢" if val > meta.get("warn_low", 0) else "🔴"
                    else:
                        status = "🟢" if val < meta.get("warn", 999) else "🔴"

                    st.markdown(f"**{status} {sensor.title()}**")

                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=val,
                        number={"suffix": f" {meta['unit']}"},
                        gauge={
                            "axis": {"range": [meta["min"], meta["max"]]},
                            "bar": {"color": "#2ecc71" if status == "🟢" else "#e74c3c"},
                            "steps": [
                                {"range": [meta["min"], meta["min"] + (meta["max"]-meta["min"])*0.6], "color": "#eafaf1"},
                                {"range": [meta["min"] + (meta["max"]-meta["min"])*0.6, meta["max"]], "color": "#fdedec"},
                            ],
                        },
                    ))
                    fig_gauge.update_layout(height=180, margin=dict(t=20, b=0, l=20, r=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    st.caption(f"Avg: {avg:.1f} | Std: {std:.2f}")

    # Correlation heatmap
    st.markdown("### Sensor Correlation Matrix")
    sensor_cols = ["temperature", "vibration", "pressure", "cycle_time"]
    corr = filtered_df[sensor_cols].corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f",
        title="Sensor Cross-Correlation",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
    )
    fig_corr.update_layout(height=400)
    st.plotly_chart(fig_corr, use_container_width=True)


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#999; font-size:0.85rem;'>"
    "DefectSense-AI v1.0 | Manufacturing Defect Prediction & GenAI Root Cause Analyzer"
    "</div>",
    unsafe_allow_html=True,
)
