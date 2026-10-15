"""Streamflow Forecasting & Flood Early-Warning Dashboard.

Run with:
    streamlit run app.py
"""
import os, json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import torch

from src.models import FeedForward, LSTMNet
from src.data import load_csv, clean

Q_MAX = 29700.0
R_MAX = 6.96
PEAK_THRESHOLD = 1500.0

st.set_page_config(
    page_title="Streamflow Forecasting AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def get_models():
    models = {}
    ff_path = "results/model_feedforward.pt"
    lstm_path = "results/model_lstm.pt"
    if os.path.exists(ff_path):
        m_ff = FeedForward(n_in=7, hidden=12)
        m_ff.load_state_dict(torch.load(ff_path, weights_only=True))
        m_ff.eval()
        models["Feed-Forward ANN"] = m_ff
    if os.path.exists(lstm_path):
        m_lstm = LSTMNet(hidden=12)
        m_lstm.load_state_dict(torch.load(lstm_path, weights_only=True))
        m_lstm.eval()
        models["LSTM Network"] = m_lstm
    return models


@st.cache_data
def get_historical_data():
    if os.path.exists("data/leaf_river.csv"):
        df = clean(load_csv("data/leaf_river.csv"))
        return df
    return None


def predict_single(models, rain_5, runoff_5):
    preds = {}
    if "Feed-Forward ANN" in models:
        feat = np.concatenate([
            np.array(rain_5[:3], dtype=float) / R_MAX,
            np.array(runoff_5[:4], dtype=float) / Q_MAX
        ])
        inp = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = models["Feed-Forward ANN"](inp).item()
        preds["Feed-Forward ANN"] = max(0.0, float(out * Q_MAX))

    if "LSTM Network" in models:
        r = np.array(rain_5[::-1], dtype=float) / R_MAX
        q = np.array(runoff_5[::-1], dtype=float) / Q_MAX
        seq = np.stack([r, q], axis=-1)
        inp = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = models["LSTM Network"](inp).item()
        preds["LSTM Network"] = max(0.0, float(out * Q_MAX))

    return preds


# ============================== SIDEBAR ==============================
st.sidebar.title("🌊 Streamflow AI")
st.sidebar.markdown("**Leaf River Basin (Collins, MS/LA)**")
st.sidebar.caption("Course Mini-Project: UE24CS352A - Machine Learning")

mode = st.sidebar.radio(
    "Select Mode",
    ["🎮 Scenario Simulator", "📅 Historical Date Explorer", "📊 Benchmarks & Hydrographs"],
    index=0
)

st.sidebar.divider()
st.sidebar.markdown("### Basin Parameters")
st.sidebar.info(
    """
    * **Flood Threshold**: 1,500 ft³/s
    * **Historical Peak**: ~29,700 ft³/s
    * **Max Daily Rain**: 6.96 inches
    * **Models**: Feed-Forward & LSTM
    """
)

models = get_models()
df = get_historical_data()

if not models:
    st.error("No trained models found in `results/`. Please run `python main.py` first.")
    st.stop()


# ============================== TAB 1: SIMULATOR ==============================
if mode == "🎮 Scenario Simulator":
    st.title("🎮 What-If Scenario Flood Simulator")
    st.markdown(
        "Simulate recent weather conditions to predict **tomorrow's river discharge ($Q_t$)** "
        "and test early flood warnings in real time."
    )

    # Preset Quick-Buttons
    st.subheader("1. Quick Presets")
    col_p1, col_p2, col_p3 = st.columns(3)
    preset = None
    if col_p1.button("☀️ Normal Low Flow (Dry Weather)"):
        preset = "dry"
    if col_p2.button("🌧️ Moderate Storm Event"):
        preset = "moderate"
    if col_p3.button("⛈️ Extreme Flood / Hurricane Event"):
        preset = "flood"

    if preset == "dry":
        st.session_state.rain = [0.0, 0.05, 0.0, 0.0, 0.0]
        st.session_state.runoff = [220.0, 240.0, 250.0, 260.0, 270.0]
    elif preset == "moderate":
        st.session_state.rain = [1.2, 0.8, 0.4, 0.1, 0.0]
        st.session_state.runoff = [750.0, 600.0, 450.0, 380.0, 350.0]
    elif preset == "flood":
        st.session_state.rain = [3.8, 2.5, 1.2, 0.5, 0.0]
        st.session_state.runoff = [4200.0, 2800.0, 1500.0, 800.0, 500.0]

    # Initialize session state if empty
    if "rain" not in st.session_state:
        st.session_state.rain = [0.4, 1.2, 0.1, 0.0, 0.0]
    if "runoff" not in st.session_state:
        st.session_state.runoff = [450.0, 600.0, 1100.0, 1600.0, 800.0]

    st.subheader("2. Configure Recent Conditions")
    col_rain, col_flow = st.columns(2)

    with col_rain:
        st.markdown("#### 🌧️ Past Daily Rainfall (inches)")
        r1 = st.slider("Yesterday R(t-1)", 0.0, 7.0, float(st.session_state.rain[0]), 0.1)
        r2 = st.slider("2 Days Ago R(t-2)", 0.0, 7.0, float(st.session_state.rain[1]), 0.1)
        r3 = st.slider("3 Days Ago R(t-3)", 0.0, 7.0, float(st.session_state.rain[2]), 0.1)
        r4 = st.slider("4 Days Ago R(t-4)", 0.0, 7.0, float(st.session_state.rain[3]), 0.1)
        r5 = st.slider("5 Days Ago R(t-5)", 0.0, 7.0, float(st.session_state.rain[4]), 0.1)

    with col_flow:
        st.markdown("#### 🌊 Past River Runoff (ft³/s)")
        q1 = st.slider("Yesterday Q(t-1)", 50.0, 10000.0, float(st.session_state.runoff[0]), 50.0)
        q2 = st.slider("2 Days Ago Q(t-2)", 50.0, 10000.0, float(st.session_state.runoff[1]), 50.0)
        q3 = st.slider("3 Days Ago Q(t-3)", 50.0, 10000.0, float(st.session_state.runoff[2]), 50.0)
        q4 = st.slider("4 Days Ago Q(t-4)", 50.0, 10000.0, float(st.session_state.runoff[3]), 50.0)
        q5 = st.slider("5 Days Ago Q(t-5)", 50.0, 10000.0, float(st.session_state.runoff[4]), 50.0)

    rain_input = [r1, r2, r3, r4, r5]
    runoff_input = [q1, q2, q3, q4, q5]

    st.divider()
    st.subheader("3. Model Forecast Results")

    preds = predict_single(models, rain_input, runoff_input)
    col_res1, col_res2 = st.columns(2)

    for i, (m_name, val) in enumerate(preds.items()):
        col = col_res1 if i == 0 else col_res2
        diff = val - q1
        with col:
            st.metric(
                label=f"**{m_name} Forecast**",
                value=f"{val:,.1f} ft³/s",
                delta=f"{diff:+,.1f} ft³/s vs Yesterday"
            )
            if val >= PEAK_THRESHOLD:
                st.error(f"🚨 **FLOOD WARNING**: Forecast exceeds flood threshold ({PEAK_THRESHOLD:,.0f} ft³/s)!")
            else:
                st.success(f"🟢 **NORMAL STREAMFLOW**: Flow is within safe bank capacity (< {PEAK_THRESHOLD:,.0f} ft³/s).")

    # Bar chart comparison
    chart_data = pd.DataFrame({
        "Discharge (ft³/s)": [q1, preds.get("Feed-Forward ANN", 0), preds.get("LSTM Network", 0)]
    }, index=["Yesterday Observed", "Feed-Forward Prediction", "LSTM Prediction"])
    st.bar_chart(chart_data)


# ============================== TAB 2: HISTORICAL EXPLORER ==============================
elif mode == "📅 Historical Date Explorer":
    st.title("📅 Historical Date Explorer (Validation 2008–2012)")
    st.markdown("Test the models against actual historical measurements recorded at the Collins USGS gauge.")

    if df is None:
        st.warning("`data/leaf_river.csv` not found.")
        st.stop()

    df["date"] = pd.to_datetime(df["date"])
    val_df = df[(df["date"] >= "2008-01-06") & (df["date"] <= "2012-12-31")].copy()
    dates = val_df["date"].dt.strftime("%Y-%m-%d").tolist()

    selected_date_str = st.select_slider(
        "Select a Date to Test",
        options=dates,
        value="2008-03-20"
    )
    sel_date = pd.to_datetime(selected_date_str)
    idx = df[df["date"] == sel_date].index[0]

    rain_hist = df["rainfall"].iloc[idx - 5:idx].values[::-1]
    runoff_hist = df["runoff"].iloc[idx - 5:idx].values[::-1]
    actual_flow = float(df["runoff"].iloc[idx])
    actual_rain = float(df["rainfall"].iloc[idx])

    preds = predict_single(models, list(rain_hist), list(runoff_hist))

    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Actual Gauge Discharge", f"{actual_flow:,.1f} ft³/s")
    if "Feed-Forward ANN" in preds:
        ff_val = preds["Feed-Forward ANN"]
        ff_err = abs(ff_val - actual_flow)
        m_col2.metric("Feed-Forward Prediction", f"{ff_val:,.1f} ft³/s",
                      delta=f"Err: {ff_err:.1f} cfs ({ff_err / actual_flow * 100:.1f}%)", delta_color="inverse")
    if "LSTM Network" in preds:
        lstm_val = preds["LSTM Network"]
        lstm_err = abs(lstm_val - actual_flow)
        m_col3.metric("LSTM Prediction", f"{lstm_val:,.1f} ft³/s",
                      delta=f"Err: {lstm_err:.1f} cfs ({lstm_err / actual_flow * 100:.1f}%)", delta_color="inverse")

    if actual_flow >= PEAK_THRESHOLD:
        st.warning(f"⚠️ This date was a registered **Flood Peak** event (> {PEAK_THRESHOLD:,.0f} ft³/s) in the Leaf River basin.")

    # 30-day window hydrograph around selected date
    st.subheader("30-Day Surrounding Hydrograph")
    w_start = max(0, idx - 15)
    w_end = min(len(df), idx + 15)
    sub = df.iloc[w_start:w_end].copy().set_index("date")

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(sub.index, sub["runoff"], "k-o", ms=3, label="Actual Observed Discharge")
    ax.axvline(sel_date, color="blue", ls="--", lw=1.5, label=f"Selected Date ({selected_date_str})")
    ax.axhline(PEAK_THRESHOLD, color="red", ls=":", lw=1, label="Flood Threshold (1500)")
    ax.set(ylabel="Runoff (ft³/s)", title="Hydrograph Window Around Selected Date")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)


# ============================== TAB 3: BENCHMARKS ==============================
elif mode == "📊 Benchmarks & Hydrographs":
    st.title("📊 Model Benchmarks & Hydrographs")
    st.markdown("Quantitative results and visual validation across the 5-year test period (2008–2012).")

    if os.path.exists("results/metrics.json"):
        with open("results/metrics.json") as f:
            metrics = json.load(f)

        rows = []
        for name, m in metrics.items():
            rows.append({
                "Model": name.capitalize(),
                "Overall Error (%)": f"{m.get('overall_pct', 0):.2f}%",
                "Peak Error (%)": f"{m.get('peak_pct', 0):.2f}%",
                "Non-Peak Error (%)": f"{m.get('nonpeak_pct', 0):.2f}%",
                "Nash-Sutcliffe (NSE)": f"{m.get('nse_overall', float('nan')):.3f}",
                "Kling-Gupta (KGE)": f"{m.get('kge_overall', float('nan')):.3f}",
            })
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.markdown("#### Model Comparison Hydrograph (2008–2012)")
        if os.path.exists("results/pred_comparison.png"):
            st.image("results/pred_comparison.png")
    with col_img2:
        st.markdown("#### Major Storm Event Zoom-In (Peak Flood)")
        if os.path.exists("results/peak_event_zoom.png"):
            st.image("results/peak_event_zoom.png")
