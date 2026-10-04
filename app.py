import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Jet Engine Intelligence",
    page_icon="✈️",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    .stApp {
        background: #07111f;
        color: white;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .hero {
        padding: 30px;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #0b1d33,
            #102a43
        );
        border: 1px solid #1d405f;
        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .hero-subtitle {
        color: #9fb3c8;
        font-size: 17px;
    }

    .status-card {
        padding: 25px;
        border-radius: 18px;
        background: #0c1b2a;
        border: 1px solid #1c3a52;
        text-align: center;
    }

    .status-title {
        color: #91a8bd;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .rul-value {
        font-size: 55px;
        font-weight: 800;
        margin: 8px 0;
    }

    .rul-label {
        color: #91a8bd;
        font-size: 15px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .metric-box {
        background: #0c1b2a;
        border: 1px solid #1c3a52;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
    }

    .metric-label {
        color: #91a8bd;
        font-size: 13px;
        text-transform: uppercase;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 700;
        margin-top: 5px;
    }

    .recommendation {
        padding: 18px;
        border-radius: 14px;
        background: #0c1b2a;
        border: 1px solid #1c3a52;
        margin-top: 15px;
    }

    .footer {
        text-align: center;
        color: #6f879b;
        margin-top: 40px;
        padding: 20px;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("rul_prediction_model.pkl")


# =========================================================
# FEATURE COLUMNS
# =========================================================

feature_columns = [
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3"
]

sensor_columns = [
    f"sensor_{i}"
    for i in range(1, 22)
]

feature_columns += sensor_columns


# =========================================================
# LOAD TEST DATA
# =========================================================

columns = [
    "engine_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21"
]

test_df = pd.read_csv(
    "data/test_FD001.txt",
    sep=r"\s+",
    header=None,
    names=columns
)


# =========================================================
# CREATE ENGINE-LEVEL PREDICTIONS
# =========================================================

latest_rows = (
    test_df
    .sort_values(["engine_id", "cycle"])
    .groupby("engine_id")
    .tail(1)
    .copy()
)

latest_features = latest_rows[feature_columns]

latest_predictions = model.predict(
    latest_features
)

latest_predictions = np.clip(
    latest_predictions,
    0,
    125
)

latest_rows["Predicted_RUL"] = latest_predictions


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
✈️ JET ENGINE INTELLIGENCE
</div>

<div class="hero-subtitle">
Predictive Maintenance & Remaining Useful Life Monitoring System
</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# ENGINE SELECTOR
# =========================================================

st.markdown(
    '<div class="section-title">🛫 Engine Command Center</div>',
    unsafe_allow_html=True
)

selected_engine = st.selectbox(
    "Select Engine",
    latest_rows["engine_id"].tolist()
)


engine_data = latest_rows[
    latest_rows["engine_id"] == selected_engine
].iloc[0]


predicted_rul = float(
    engine_data["Predicted_RUL"]
)

current_cycle = int(
    engine_data["cycle"]
)


# =========================================================
# STATUS
# =========================================================

if predicted_rul <= 20:

    status = "CRITICAL"
    status_icon = "🔴"
    recommendation = (
        "Immediate inspection recommended. "
        "The engine has a low estimated remaining useful life."
    )

elif predicted_rul <= 50:

    status = "WARNING"
    status_icon = "🟠"
    recommendation = (
        "Monitor engine condition closely and "
        "plan maintenance based on operational requirements."
    )

else:

    status = "HEALTHY"
    status_icon = "🟢"
    recommendation = (
        "Engine currently shows a relatively high "
        "remaining useful life. Continue monitoring."
    )


# =========================================================
# MAIN STATUS CARD
# =========================================================

col1, col2, col3 = st.columns([1.5, 1, 1])

with col1:

    st.markdown(f"""
    <div class="status-card">

    <div class="status-title">
    ENGINE {selected_engine}
    </div>

    <div class="rul-value">
    {predicted_rul:.1f}
    </div>

    <div class="rul-label">
    ESTIMATED REMAINING CYCLES
    </div>

    <h3>
    {status_icon} {status}
    </h3>

    </div>
    """, unsafe_allow_html=True)


with col2:

    st.markdown(f"""
    <div class="metric-box">

    <div class="metric-label">
    Current Cycle
    </div>

    <div class="metric-value">
    {current_cycle}
    </div>

    </div>
    """, unsafe_allow_html=True)


with col3:

    health_percentage = min(
        100,
        (predicted_rul / 125) * 100
    )

    st.markdown(f"""
    <div class="metric-box">

    <div class="metric-label">
    Estimated Life
    </div>

    <div class="metric-value">
    {health_percentage:.0f}%
    </div>

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# RUL GAUGE
# =========================================================

st.markdown(
    '<div class="section-title">🔋 Remaining Life Indicator</div>',
    unsafe_allow_html=True
)

st.progress(
    health_percentage / 100
)

st.caption(
    f"Estimated remaining useful life: "
    f"{predicted_rul:.1f} cycles out of the 125-cycle prediction range."
)


# =========================================================
# RECOMMENDATION
# =========================================================

st.markdown(f"""
<div class="recommendation">

<b>Maintenance Recommendation</b>

<br><br>

{recommendation}

</div>
""", unsafe_allow_html=True)


# =========================================================
# SENSOR DIAGNOSTICS
# =========================================================

st.markdown(
    '<div class="section-title">📡 Sensor Diagnostics</div>',
    unsafe_allow_html=True
)

sensor_data = engine_data[
    sensor_columns
]

sensor_df = pd.DataFrame({
    "Sensor": sensor_columns,
    "Reading": sensor_data.values
})

sensor_df["Reading"] = sensor_df["Reading"].round(4)

st.dataframe(
    sensor_df,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.markdown(
    '<div class="section-title">🧠 Model Intelligence</div>',
    unsafe_allow_html=True
)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "Algorithm",
        "Gradient Boosting"
    )

with m2:
    st.metric(
        "MAE",
        "11.65 cycles"
    )

with m3:
    st.metric(
        "RMSE",
        "16.07 cycles"
    )

with m4:
    st.metric(
        "R² Score",
        "0.839"
    )


# =========================================================
# ENGINE TABLE
# =========================================================

st.markdown(
    '<div class="section-title">📊 Fleet Overview</div>',
    unsafe_allow_html=True
)

fleet_table = latest_rows[
    [
        "engine_id",
        "cycle",
        "Predicted_RUL"
    ]
].copy()

fleet_table.columns = [
    "Engine ID",
    "Latest Cycle",
    "Predicted RUL"
]

fleet_table["Predicted RUL"] = (
    fleet_table["Predicted RUL"]
    .round(2)
)

st.dataframe(
    fleet_table,
    use_container_width=True,
    hide_index=True
)

# =========================================================
# FLEET HEALTH VISUALIZATION
# =========================================================

st.markdown(
    '<div class="section-title">📈 Fleet RUL Distribution</div>',
    unsafe_allow_html=True
)

chart_data = fleet_table.set_index("Engine ID")[
    ["Predicted RUL"]
]

st.bar_chart(
    chart_data,
    height=350
)

st.caption(
    "Predicted remaining useful life across all 100 test engines."
)
# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

NASA C-MAPSS FD001 • Gradient Boosting RUL Prediction

<br>

Predictive maintenance prototype for turbofan engine health monitoring

</div>
""", unsafe_allow_html=True)