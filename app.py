import os
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="AUMP Dashboard",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
    .main > div {
        padding-top: 1.2rem;
    }

    .hero {
        padding: 1.25rem 1.4rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(124,58,237,0.18), rgba(14,165,233,0.12));
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1rem;
    }

    .hero h1 {
        margin: 0 0 0.35rem 0;
        font-size: 2.2rem;
    }

    .hero p {
        margin: 0;
        opacity: 0.9;
        line-height: 1.5;
    }

    .mini-card {
        padding: 0.95rem 1rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.03);
        margin-bottom: 0.75rem;
    }

    .mini-card h4 {
        margin: 0 0 0.35rem 0;
        font-size: 1rem;
    }

    .mini-card p {
        margin: 0;
        opacity: 0.9;
    }

    .badge-safe {
        display: inline-block;
        padding: 0.4rem 0.7rem;
        border-radius: 999px;
        font-weight: 600;
        background: rgba(34,197,94,0.15);
        color: #86efac;
        border: 1px solid rgba(34,197,94,0.35);
    }

    .badge-watch {
        display: inline-block;
        padding: 0.4rem 0.7rem;
        border-radius: 999px;
        font-weight: 600;
        background: rgba(245,158,11,0.15);
        color: #fcd34d;
        border: 1px solid rgba(245,158,11,0.35);
    }

    .badge-alert {
        display: inline-block;
        padding: 0.4rem 0.7rem;
        border-radius: 999px;
        font-weight: 600;
        background: rgba(239,68,68,0.15);
        color: #fca5a5;
        border: 1px solid rgba(239,68,68,0.35);
    }

    .footnote {
        font-size: 0.92rem;
        opacity: 0.8;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load model
# -----------------------------
MODEL_PATH = "models/model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Run `python3 src/train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

# -----------------------------
# Scenarios
# -----------------------------
preset_scenarios = {
    "Stable": [76, 75, 76, 74, 75, 76, 75, 74, 75, 76],
    "Mild Decline": [82, 81, 80, 79, 77, 76, 74, 72, 70, 68],
    "Crash Pattern": [75, 74, 73, 72, 70, 68, 65, 60, 58, 55, 52],
    "Sharp Crash": [84, 83, 82, 79, 74, 68, 60, 52, 47, 44],
}

# -----------------------------
# Helpers
# -----------------------------
def compute_risk_over_time(recent_bpm, model):
    results = []

    for i in range(4, len(recent_bpm) + 1):
        window = recent_bpm[:i]

        df = pd.DataFrame({"bpm": window})
        df["bpm_prev"] = df["bpm"].shift(1)
        df["delta"] = df["bpm"] - df["bpm_prev"]
        df["rolling_mean"] = df["bpm"].rolling(window=3).mean()
        df["rolling_std"] = df["bpm"].rolling(window=3).std()
        df["acceleration"] = df["delta"].diff()
        df["trend"] = df["rolling_mean"].diff()
        df = df.dropna()

        input_data = df[[
            "bpm",
            "delta",
            "rolling_mean",
            "rolling_std",
            "acceleration",
            "trend"
        ]].iloc[[-1]]

        prob = float(model.predict_proba(input_data)[0][1])

        results.append({
            "time": i,
            "bpm": float(window[-1]),
            "risk": prob
        })

    return pd.DataFrame(results)


def classify_status(latest_risk, alert_threshold):
    if latest_risk >= alert_threshold:
        return "ALERT", "badge-alert"
    elif latest_risk >= alert_threshold * 0.6:
        return "WATCH", "badge-watch"
    return "STABLE", "badge-safe"


def make_bpm_chart(results_df, danger_threshold):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=results_df["time"],
        y=results_df["bpm"],
        mode="lines+markers",
        name="BPM",
        line=dict(width=3),
        marker=dict(size=8)
    ))

    fig.add_hline(
        y=danger_threshold,
        line_dash="dash",
        annotation_text=f"Danger Threshold ({danger_threshold} BPM)",
        annotation_position="top left"
    )

    fig.update_layout(
        title="Heart Rate Over Time",
        xaxis_title="Time",
        yaxis_title="BPM",
        template="plotly_dark",
        height=430,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig


def make_risk_chart(results_df, alert_threshold):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=results_df["time"],
        y=results_df["risk"],
        mode="lines+markers",
        name="Crash Risk",
        line=dict(width=3),
        marker=dict(size=8)
    ))

    fig.add_hline(
        y=alert_threshold,
        line_dash="dash",
        annotation_text=f"Alert Threshold ({alert_threshold:.2f})",
        annotation_position="top left"
    )

    fig.update_layout(
        title="Predicted Crash Risk Over Time",
        xaxis_title="Time",
        yaxis_title="Risk Probability",
        template="plotly_dark",
        height=430,
        margin=dict(l=20, r=20, t=60, b=20),
        yaxis=dict(range=[0, 1])
    )
    return fig


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Controls")

scenario_choice = st.sidebar.selectbox(
    "Preset Scenario",
    ["Custom Input"] + list(preset_scenarios.keys())
)

if scenario_choice == "Custom Input":
    default_bpm = "75, 74, 73, 72, 70, 68, 65, 60, 58, 55, 52"
else:
    default_bpm = ", ".join(str(x) for x in preset_scenarios[scenario_choice])

bpm_text = st.sidebar.text_area(
    "BPM Sequence (comma-separated)",
    value=default_bpm,
    height=140
)

alert_threshold = st.sidebar.slider(
    "Alert Threshold",
    min_value=0.10,
    max_value=0.95,
    value=0.70,
    step=0.05
)

danger_threshold = st.sidebar.slider(
    "Danger Threshold (BPM)",
    min_value=40,
    max_value=80,
    value=60,
    step=1
)

run_button = st.sidebar.button("Run Analysis", use_container_width=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="hero">
    <h1>🫀 AUMP Dashboard</h1>
    <p>
        AUMP — a machine learning prototype for estimating
        short-term crash risk from recent BPM trends. This demo is intended for research-style
        simulation and visualization, not clinical decision-making.
    </p>
</div>
""", unsafe_allow_html=True)

info_col1, info_col2, info_col3 = st.columns(3)
with info_col1:
    st.markdown("""
    <div class="mini-card">
        <h4>Input</h4>
        <p>Recent BPM sequence from a preset scenario or custom values.</p>
    </div>
    """, unsafe_allow_html=True)
with info_col2:
    st.markdown("""
    <div class="mini-card">
        <h4>Model Output</h4>
        <p>Probability-like crash risk score over time using engineered temporal features.</p>
    </div>
    """, unsafe_allow_html=True)
with info_col3:
    st.markdown("""
    <div class="mini-card">
        <h4>Important Limitation</h4>
        <p>Current results reflect a model trained on simulated data and are not clinically validated.</p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Main
# -----------------------------
if run_button:
    try:
        recent_bpm = [float(x.strip()) for x in bpm_text.split(",") if x.strip() != ""]
    except ValueError:
        st.error("Please enter only numbers separated by commas.")
        st.stop()

    if len(recent_bpm) < 4:
        st.error("Please enter at least 4 BPM values.")
        st.stop()

    results_df = compute_risk_over_time(recent_bpm, model)

    if results_df.empty:
        st.error("No predictions could be generated from the current input.")
        st.stop()

    alert_time = None
    danger_time = None

    for _, row in results_df.iterrows():
        if alert_time is None and row["risk"] >= alert_threshold:
            alert_time = int(row["time"])
        if danger_time is None and row["bpm"] < danger_threshold:
            danger_time = int(row["time"])

    latest_risk = float(results_df["risk"].iloc[-1])
    latest_bpm = float(results_df["bpm"].iloc[-1])

    status_text, badge_class = classify_status(latest_risk, alert_threshold)

    st.markdown(
        f'<div class="{badge_class}">Current Status: {status_text}</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latest BPM", f"{latest_bpm:.1f}")
    m2.metric("Latest Risk", f"{latest_risk:.3f}")
    m3.metric("Alert Time", alert_time if alert_time is not None else "No alert")
    if alert_time is not None and danger_time is not None:
        lead_time = danger_time - alert_time
        m4.metric("Lead Time", lead_time)
    else:
        lead_time = None
        m4.metric("Lead Time", "N/A")

    if alert_time is not None and danger_time is not None:
        if lead_time > 0:
            st.success(f"Early warning achieved: the model triggered {lead_time} timestep(s) before BPM crossed the danger threshold.")
        elif lead_time == 0:
            st.warning("The model triggered at the same timestep that BPM crossed the danger threshold.")
        else:
            st.error(f"The model triggered {-lead_time} timestep(s) after the danger threshold was crossed.")
    else:
        st.info("A complete lead-time comparison was not available for this input sequence.")

    tab1, tab2, tab3 = st.tabs(["Dashboard", "Detailed Data", "Method & Limits"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(make_bpm_chart(results_df, danger_threshold), use_container_width=True)
        with c2:
            st.plotly_chart(make_risk_chart(results_df, alert_threshold), use_container_width=True)

    with tab2:
        display_df = results_df.copy()
        display_df["risk"] = display_df["risk"].round(4)
        st.dataframe(display_df, use_container_width=True)

        csv_data = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Results CSV",
            data=csv_data,
            file_name="risk_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )

    with tab3:
        st.markdown("""
### What the app is doing
For each new timestep, the app computes engineered features from recent BPM values:

- current BPM
- one-step BPM change (`delta`)
- short rolling average
- short rolling variability
- acceleration of BPM change
- recent trend shift

Those features are passed into the trained classifier, which returns a crash-risk probability-like score.

### What “accurate” means here
This app is **internally consistent** with your current training pipeline, but it is **not clinically validated**. The risk score reflects:
- your simulated data generation logic
- your feature engineering choices
- the behavior of the current saved model

### What would make it more scientifically credible
- real wearable or physiological data
- better event labels
- comparison across more models
- calibration analysis
- external validation
        """)

else:
    st.info("Choose a preset or enter custom BPM values in the sidebar, then click **Run Analysis**.")