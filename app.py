"""
app.py  —  Wireless Channel Capacity Predictor
Phase 7: Streamlit dashboard

Run from the project root (venv active):
    streamlit run app.py
"""

import math
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ─────────────────────────────────────────────
# 0.  Page config — must be first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Wireless Capacity Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# 1.  Custom CSS  (amber oscilloscope theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
  /* ── background & base ── */
  .stApp { background-color: #0E1117; }
  section[data-testid="stSidebar"] { background-color: #161B27; }

  /* ── metric cards ── */
  div[data-testid="metric-container"] {
    background: #1C2333;
    border-radius: 10px;
    padding: 14px 18px;
    border: 1px solid #2D3748;
  }
  div[data-testid="metric-container"] label {
    color: #8B949E !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #E8A020 !important;
    font-size: 1.9rem !important;
    font-weight: 700;
    font-family: 'Courier New', monospace;
  }
  div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.8rem !important;
  }

  /* ── section headings ── */
  h1 { color: #E8A020 !important; font-family: 'Courier New', monospace; }
  h3 { color: #CDD9E5 !important; border-bottom: 1px solid #2D3748; padding-bottom: 6px; }

  /* ── info / status box ── */
  .rf-badge {
    background: #1C2333;
    border-left: 3px solid #E8A020;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    font-family: 'Courier New', monospace;
    font-size: 0.84rem;
    color: #CDD9E5;
    margin: 6px 0;
  }
  .rf-badge b { color: #E8A020; }

  /* ── sliders amber thumb ── */
  div[data-baseweb="slider"] [role="slider"] { background-color: #E8A020 !important; }
  div[data-baseweb="slider"] [data-testid="stSliderTrack"] > div:first-child {
    background: #E8A020 !important;
  }

  /* ── sidebar labels ── */
  .sidebar-label {
    font-size: 0.72rem;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #8B949E;
    margin-bottom: -8px;
    margin-top: 8px;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 2.  Load artifacts (cached — loads once)
# ─────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent / "models"

@st.cache_resource(show_spinner="Loading model artifacts…")
def load_artifacts():
    return {
        "model":      joblib.load(MODELS_DIR / "best_model.pkl"),
        "x_scaler":   joblib.load(MODELS_DIR / "scaler.pkl"),
        "y_scaler":   joblib.load(MODELS_DIR / "target_scaler.pkl"),
        "features":   joblib.load(MODELS_DIR / "feature_columns.pkl"),
        "model_name": joblib.load(MODELS_DIR / "best_model_name.pkl"),
    }

arts = load_artifacts()

# ─────────────────────────────────────────────
# 3.  Helper functions
# ─────────────────────────────────────────────
SPEED_OF_LIGHT = 3e8
ENV_COLS = ["environment_Rural", "environment_Suburban", "environment_Urban"]

def build_row(distance_m, freq_ghz, bw_mhz, tx_dbm, gain_db, nf_db, env):
    return {
        "distance_m":          distance_m,
        "frequency_ghz":       freq_ghz,
        "bandwidth_mhz":       bw_mhz,
        "tx_power_dbm":        tx_dbm,
        "antenna_gain_db":     gain_db,
        "noise_figure_db":     nf_db,
        "environment_Rural":   1 if env == "Rural"    else 0,
        "environment_Suburban":1 if env == "Suburban" else 0,
        "environment_Urban":   1 if env == "Urban"    else 0,
    }

def ml_predict(df: pd.DataFrame) -> np.ndarray:
    """Scale features → infer → inverse-scale target. Returns Mbps (clipped ≥ 0)."""
    X = arts["x_scaler"].transform(df[arts["features"]])
    p = arts["model"].predict(X)
    cap = arts["y_scaler"].inverse_transform(p.reshape(-1, 1)).ravel()
    return np.maximum(cap, 0.0)

def shannon_capacity(distance_m, freq_ghz, bw_mhz, tx_dbm, gain_db, nf_db):
    """Friis-based FSPL + thermal noise → Shannon C = B log2(1+SNR)."""
    fspl_db = (
        20 * math.log10(max(distance_m, 1))
        + 20 * math.log10(freq_ghz * 1e9)
        + 20 * math.log10(4 * math.pi / SPEED_OF_LIGHT)
    )
    rx_dbm   = tx_dbm + gain_db - fspl_db
    noise_dbm = -174 + 10 * math.log10(bw_mhz * 1e6) + nf_db
    snr_db   = rx_dbm - noise_dbm
    snr_lin  = 10 ** (snr_db / 10)
    cap_mbps = max(bw_mhz * math.log2(1 + snr_lin), 0)
    return fspl_db, rx_dbm, noise_dbm, snr_db, cap_mbps

def quality_label(snr_db):
    if snr_db >= 25: return "🟢 Excellent", "#2ECC71"
    if snr_db >= 15: return "🟡 Good",      "#F1C40F"
    if snr_db >= 5:  return "🟠 Fair",       "#E67E22"
    return                    "🔴 Poor",      "#E74C3C"

# ─────────────────────────────────────────────
# 4.  Sidebar — link parameter inputs
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Link Parameters")
    st.markdown("Adjust any slider — prediction updates instantly.")
    st.markdown("---")

    def label(text): st.markdown(f'<p class="sidebar-label">{text}</p>', unsafe_allow_html=True)

    label("Environment")
    environment = st.selectbox("", ["Rural", "Suburban", "Urban"],
                               label_visibility="collapsed")

    label("Frequency band")
    FREQ_MAP = {
        "0.9 GHz  (GSM / LTE-800)": 0.9,
        "1.8 GHz  (GSM-1800 / LTE)": 1.8,
        "2.1 GHz  (UMTS / LTE)": 2.1,
        "2.4 GHz  (Wi-Fi / LTE)": 2.4,
        "3.5 GHz  (5G Sub-6)": 3.5,
        "5.0 GHz  (Wi-Fi / 5G)": 5.0,
        "28 GHz   (5G mmWave)": 28.0,
    }
    freq_label = st.selectbox("", list(FREQ_MAP.keys()), index=3,
                              label_visibility="collapsed")
    freq_ghz = FREQ_MAP[freq_label]

    label("Distance  (m)")
    distance_m = st.slider("", 10, 5000, 500, step=10,
                            label_visibility="collapsed",
                            format="%d m")

    label("Bandwidth  (MHz)")
    bandwidth_mhz = st.select_slider("", options=[5,10,15,20,40,80,100,200],
                                      value=20, label_visibility="collapsed")

    label("Tx power  (dBm)")
    tx_power_dbm = st.slider("", 20, 46, 30, step=1,
                              label_visibility="collapsed",
                              format="%d dBm")

    label("Antenna gain  (dBi)")
    antenna_gain_db = st.slider("", 0, 20, 8, step=1,
                                 label_visibility="collapsed",
                                 format="%d dBi")

    label("Noise figure  (dB)")
    noise_figure_db = st.slider("", 5.0, 10.0, 7.0, step=0.5,
                                 label_visibility="collapsed",
                                 format="%.1f dB")

    st.markdown("---")
    st.markdown(f"**Model:** {arts['model_name']}")
    st.caption("Phase 7 · Wireless Capacity Predictor")

# ─────────────────────────────────────────────
# 5.  Compute current prediction
# ─────────────────────────────────────────────
row_df = pd.DataFrame([build_row(
    distance_m, freq_ghz, bandwidth_mhz,
    tx_power_dbm, antenna_gain_db, noise_figure_db, environment
)])
ml_cap = float(ml_predict(row_df)[0])

fspl_db, rx_dbm, noise_dbm, snr_db, sh_cap = shannon_capacity(
    distance_m, freq_ghz, bandwidth_mhz,
    tx_power_dbm, antenna_gain_db, noise_figure_db
)
ql, ql_color = quality_label(snr_db)
fading_loss = max(sh_cap - ml_cap, 0)

# ─────────────────────────────────────────────
# 6.  Main layout
# ─────────────────────────────────────────────
st.markdown("# 📡 Wireless Channel Capacity Predictor")
st.markdown(
    f"**{environment}** · **{freq_ghz} GHz** · "
    f"**{distance_m} m** · **{bandwidth_mhz} MHz** BW"
)
st.markdown("---")

# ── 6a. Top metric row ─────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("ML Prediction",      f"{ml_cap:,.1f} Mbps",
          help="Neural Network (MLP) output, trained on 8,000 synthetic link samples.")
c2.metric("Shannon Theory",     f"{sh_cap:,.1f} Mbps",
          delta=f"−{fading_loss:.1f} Mbps fading",
          delta_color="inverse",
          help="Theoretical ceiling: B·log₂(1+SNR) with free-space path loss only.")
c3.metric("Received SNR",       f"{snr_db:.1f} dB")
c4.metric("Link Quality",       ql.split()[-1],  # text only, emoji above
          help=f"SNR {snr_db:.1f} dB → {ql}")

st.markdown("")

# ── 6b. Distance sweep chart ───────────────────────
st.markdown("### Capacity vs. Distance")

distances = np.linspace(10, 5000, 200)
sweep_rows = [
    build_row(d, freq_ghz, bandwidth_mhz,
              tx_power_dbm, antenna_gain_db, noise_figure_db, environment)
    for d in distances
]
sweep_df  = pd.DataFrame(sweep_rows)
ml_sweep  = ml_predict(sweep_df)
sh_sweep  = np.array([
    shannon_capacity(d, freq_ghz, bandwidth_mhz,
                     tx_power_dbm, antenna_gain_db, noise_figure_db)[4]
    for d in distances
])

fig = go.Figure()

# Shannon theoretical
fig.add_trace(go.Scatter(
    x=distances, y=sh_sweep,
    name="Shannon Theory",
    line=dict(color="#4A90D9", width=2, dash="dash"),
    hovertemplate="Distance: %{x:.0f} m<br>Theory: %{y:.1f} Mbps<extra></extra>",
))

# ML prediction band
fig.add_trace(go.Scatter(
    x=distances, y=ml_sweep,
    name="ML Prediction",
    line=dict(color="#E8A020", width=2.5),
    fill="tozeroy",
    fillcolor="rgba(232,160,32,0.10)",
    hovertemplate="Distance: %{x:.0f} m<br>Predicted: %{y:.1f} Mbps<extra></extra>",
))

# Current point marker
fig.add_trace(go.Scatter(
    x=[distance_m], y=[ml_cap],
    name="Current setting",
    mode="markers",
    marker=dict(color="#E8A020", size=12, symbol="diamond",
                line=dict(color="white", width=2)),
    hovertemplate=f"You are here<br>{ml_cap:.1f} Mbps<extra></extra>",
))

fig.update_layout(
    plot_bgcolor="#0E1117",
    paper_bgcolor="#0E1117",
    font_color="#CDD9E5",
    xaxis=dict(title="Distance (m)", gridcolor="#2D3748", showline=True,
               linecolor="#4A5568", range=[0, 5000]),
    yaxis=dict(title="Capacity (Mbps)", gridcolor="#2D3748", showline=True,
               linecolor="#4A5568"),
    legend=dict(bgcolor="#1C2333", bordercolor="#2D3748", borderwidth=1),
    margin=dict(l=10, r=10, t=10, b=10),
    hovermode="x unified",
    height=360,
)
st.plotly_chart(fig, use_container_width=True)

# ── 6c. Link budget + model comparison (two columns) ────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### Link Budget")
    items = [
        ("Tx Power",         f"{tx_power_dbm:+.1f} dBm"),
        ("Antenna Gain",     f"{antenna_gain_db:+.1f} dBi"),
        ("Free-Space Path Loss", f"−{fspl_db:.1f} dB"),
        ("Received Power",   f"{rx_dbm:.1f} dBm"),
        ("Noise Floor",      f"{noise_dbm:.1f} dBm"),
        ("SNR",              f"{snr_db:.1f} dB  ← {ql}"),
        ("Bandwidth",        f"{bandwidth_mhz} MHz"),
        ("Shannon Ceiling",  f"{sh_cap:.1f} Mbps"),
        ("ML Prediction",    f"{ml_cap:.1f} Mbps"),
    ]
    for label_text, val in items:
        st.markdown(
            f'<div class="rf-badge"><b>{label_text}:</b> {val}</div>',
            unsafe_allow_html=True,
        )

with col_right:
    st.markdown("### All 5 Models — Phase 4 Results")
    results_path = Path(__file__).parent / "models" / "model_comparison.csv"
    if results_path.exists():
        df_res = pd.read_csv(results_path).sort_values("R2", ascending=False)
        best_model_name = arts["model_name"]

        fig2 = go.Figure()
        colors = [
            "#E8A020" if m == best_model_name else "#4A5568"
            for m in df_res["Model"]
        ]
        fig2.add_trace(go.Bar(
            x=df_res["R2"],
            y=df_res["Model"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in df_res["R2"]],
            textposition="outside",
            textfont=dict(color="#CDD9E5", size=11),
            hovertemplate="%{y}<br>R² = %{x:.4f}<extra></extra>",
        ))
        fig2.update_layout(
            plot_bgcolor="#0E1117",
            paper_bgcolor="#0E1117",
            font_color="#CDD9E5",
            xaxis=dict(title="R² Score", range=[0, 1.05],
                       gridcolor="#2D3748", showline=True, linecolor="#4A5568"),
            yaxis=dict(gridcolor="#2D3748"),
            margin=dict(l=10, r=40, t=10, b=10),
            height=280,
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.caption(
            f"Best: **{best_model_name}** · "
            f"R²={df_res.iloc[0]['R2']} · "
            f"RMSE={df_res.iloc[0]['RMSE (Mbps)']} Mbps"
        )
    else:
        st.warning("Run `python src/train_models.py` first to generate model_comparison.csv")

# ── 6d. Footer ─────────────────────────────────────
st.markdown("---")
st.caption(
    "Physics pipeline: Friis free-space path loss → log-distance model "
    "→ Shannon-Hartley capacity  |  "
    "10,000 synthetic samples, 5 ML models, best: " + arts["model_name"]
)