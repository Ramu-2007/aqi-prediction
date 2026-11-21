import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Hyperlocal AQI Intelligence Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Modern CSS ---
st.markdown("""
<style>
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 60%, #0284C7 100%);
        padding: 24px 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.15);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-sub {
        font-size: 0.98rem;
        color: #E2E8F0;
        margin-top: 6px;
    }
    .status-pill {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.05rem;
        color: white;
        margin-bottom: 10px;
    }
    .info-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        color: #0F172A;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    .info-card h4 {
        margin: 0;
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .info-card p {
        margin: 6px 0 0 0;
        font-size: 1.15rem;
        font-weight: 700;
    }
    .advisory-box {
        padding: 16px 20px;
        border-radius: 12px;
        background-color: #F8FAFC;
        border-left: 6px solid #1E3A8A;
        color: #1E293B;
        font-size: 0.98rem;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Model & Optional Dataset ---
FEATURES = ['PM2.5', 'PM10', 'NO2', 'NH3', 'CO', 'SO2', 'O3']

@st.cache_resource
def load_model():
    return joblib.load("models/aqi_model.pkl")

@st.cache_data
def load_city_data():
    if os.path.exists("city_day.csv"):
        df = pd.read_csv("city_day.csv")
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return df.dropna(subset=['AQI'])
    return None

model = load_model()
city_df = load_city_data()

# --- Indian CPCB AQI Helper ---
def get_aqi_details(aqi):
    if aqi <= 50:
        return ("Good", "#16A34A", "Minimal impact. Air is clean and fresh.",
                "Safe", "Open", "Not Needed", "Optional")
    elif aqi <= 100:
        return ("Satisfactory", "#65A30D", "Minor breathing discomfort possible for sensitive people.",
                "Safe", "Open", "Not Needed", "Optional")
    elif aqi <= 200:
        return ("Moderate", "#D97706", "Discomfort possible for people with asthma, lung, or heart disease.",
                "Light Only", "Limit", "Optional", "Recommended")
    elif aqi <= 300:
        return ("Poor", "#EA580C", "Breathing discomfort on prolonged exposure. Reduce outdoor time.",
                "Avoid", "Close", "Wear Mask", "Turn On")
    elif aqi <= 400:
        return ("Very Poor", "#DC2626", "Respiratory illness risk on prolonged exposure. Stay indoors.",
                "Stay Indoors", "Keep Closed", "N95 Mask", "High Mode")
    else:
        return ("Severe", "#7F1D1D", "Health emergency! Affects healthy people and seriously impacts those with conditions.",
                "No Outdoor", "Seal Tight", "N95 Mandatory", "Max Mode")

# --- Plotly AQI Gauge ---
def create_aqi_gauge(aqi_val, color, category):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi_val,
        number={'font': {'size': 52, 'color': color, 'family': 'Arial Black'}},
        title={'text': f"<b>{category}</b>", 'font': {'size': 20, 'color': color}},
        gauge={
            'axis': {'range': [0, 500], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
            'bar': {'color': color, 'thickness': 0.28},
            'bgcolor': "white",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': '#DCFCE7'},
                {'range': [50, 100], 'color': '#ECFCCB'},
                {'range': [100, 200], 'color': '#FEF3C7'},
                {'range': [200, 300], 'color': '#FFEDD5'},
                {'range': [300, 400], 'color': '#FEE2E2'},
                {'range': [400, 500], 'color': '#FECACA'},
            ],
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig

# --- Session State for Presets ---
defaults = {"pm25": 65.0, "pm10": 115.0, "no2": 32.0, "nh3": 20.0, "co": 1.1, "so2": 14.0, "o3": 38.0}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def set_preset(pm25, pm10, no2, nh3, co, so2, o3):
    st.session_state.pm25 = pm25
    st.session_state.pm10 = pm10
    st.session_state.no2 = no2
    st.session_state.nh3 = nh3
    st.session_state.co = co
    st.session_state.so2 = so2
    st.session_state.o3 = o3

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 🌍 AQI Control Panel")
    st.caption("Quickly simulate real-world air quality scenarios:")
    st.button("🌿 Clean Morning", use_container_width=True, on_click=set_preset, args=(18.0, 35.0, 12.0, 8.0, 0.4, 6.0, 25.0))
    st.button("🏙️ Moderate City", use_container_width=True, on_click=set_preset, args=(75.0, 140.0, 40.0, 22.0, 1.3, 18.0, 45.0))
    st.button("🏭 Industrial Haze", use_container_width=True, on_click=set_preset, args=(160.0, 260.0, 68.0, 38.0, 2.4, 35.0, 55.0))
    st.button("🌫️ Severe Winter Smog", use_container_width=True, on_click=set_preset, args=(280.0, 420.0, 95.0, 55.0, 3.8, 42.0, 75.0))
    st.divider()
    st.markdown("**Model:** Random Forest Regressor")
    st.markdown("**Standard:** Indian CPCB AQI")
    st.markdown("**Dataset:** India Air Quality (2015–2020)")

# --- Hero Header ---
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🌍 Hyperlocal Air Pollution (AQI) Predictor</div>
    <div class="hero-sub">Interactive Machine Learning dashboard for real-time AQI estimation, health guidance, and Indian city analytics.</div>
</div>
""", unsafe_allow_html=True)

# --- Main Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🎛️ Live AQI Predictor",
    "🏙️ Indian City Explorer",
    "📂 Batch CSV Prediction",
    "📊 Model & CPCB Guide"
])

# ==========================================
# TAB 1: LIVE AQI PREDICTOR
# ==========================================
with tab1:
    left_col, right_col = st.columns([1.05, 1.25], gap="large")

    with left_col:
        st.subheader("Pollutant Sensor Readings")
        c1, c2 = st.columns(2)
        with c1:
            pm25 = st.slider("PM2.5 (µg/m³)", 0.0, 500.0, key="pm25", step=1.0)
            no2  = st.slider("NO₂ (µg/m³)", 0.0, 200.0, key="no2", step=1.0)
            co   = st.slider("CO (mg/m³)", 0.0, 10.0, key="co", step=0.1)
            o3   = st.slider("O₃ (µg/m³)", 0.0, 200.0, key="o3", step=1.0)
        with c2:
            pm10 = st.slider("PM10 (µg/m³)", 0.0, 600.0, key="pm10", step=1.0)
            nh3  = st.slider("NH₃ (µg/m³)", 0.0, 150.0, key="nh3", step=1.0)
            so2  = st.slider("SO₂ (µg/m³)", 0.0, 150.0, key="so2", step=1.0)

        # Dominant Pollutant Calculation (relative to safe limits)
        ratios = {
            "PM2.5": pm25 / 60, "PM10": pm10 / 100, "NO₂": no2 / 80,
            "NH₃": nh3 / 400, "CO": co / 2.0, "SO₂": so2 / 80, "O₃": o3 / 100
        }
        dominant = max(ratios, key=ratios.get)
        st.info(f"🔎 **Dominant Pollutant:** `{dominant}` is currently the biggest contributor to air pollution.")

    # Predict
    input_df = pd.DataFrame([[pm25, pm10, no2, nh3, co, so2, o3]], columns=FEATURES)
    predicted_aqi = max(0.0, round(float(model.predict(input_df)[0]), 1))
    category, color, advisory, outdoor, windows, mask, purifier = get_aqi_details(predicted_aqi)

    with right_col:
        st.plotly_chart(create_aqi_gauge(predicted_aqi, color, category), use_container_width=True)

        # 4 Actionable Safety Cards
        a1, a2, a3, a4 = st.columns(4)
        a1.markdown(f'<div class="info-card"><h4>🏃 Outdoor</h4><p>{outdoor}</p></div>', unsafe_allow_html=True)
        a2.markdown(f'<div class="info-card"><h4>🪟 Windows</h4><p>{windows}</p></div>', unsafe_allow_html=True)
        a3.markdown(f'<div class="info-card"><h4>😷 Mask</h4><p>{mask}</p></div>', unsafe_allow_html=True)
        a4.markdown(f'<div class="info-card"><h4>💨 Purifier</h4><p>{purifier}</p></div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="advisory-box" style="border-left-color: {color};">
            <strong>🩺 Health Advisory:</strong> {advisory}
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.subheader("📈 Current Pollutant Concentration Breakdown")
    bar_df = pd.DataFrame({
        "Pollutant": ["PM2.5", "PM10", "NO2", "NH3", "SO2", "O3"],
        "Concentration (µg/m³)": [pm25, pm10, no2, nh3, so2, o3]
    })
    fig_bar = px.bar(
        bar_df, x="Pollutant", y="Concentration (µg/m³)",
        color="Concentration (µg/m³)", color_continuous_scale="RdYlGn_r",
        text_auto=".0f", height=280
    )
    fig_bar.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# TAB 2: INDIAN CITY EXPLORER
# ==========================================
with tab2:
    st.subheader("🏙️ Historical Air Quality Across Indian Cities (2015–2020)")
    if city_df is None:
        st.warning("`city_day.csv` not found in the project folder. Place it next to `app.py` to unlock City Analytics.")
    else:
        cities = sorted(city_df['City'].unique())
        default_idx = cities.index("Delhi") if "Delhi" in cities else 0
        selected_city = st.selectbox("Select an Indian City:", cities, index=default_idx)

        cdf = city_df[city_df['City'] == selected_city].sort_values("Date")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Average AQI", f"{cdf['AQI'].mean():.1f}")
        m2.metric("Worst Recorded AQI", f"{cdf['AQI'].max():.0f}")
        m3.metric("Cleanest Recorded AQI", f"{cdf['AQI'].min():.0f}")
        m4.metric("Total Days Recorded", f"{len(cdf):,}")

        fig_line = px.line(
            cdf, x="Date", y="AQI",
            title=f"Daily AQI Trend in {selected_city}",
            color_discrete_sequence=["#0284C7"]
        )
        fig_line.add_hline(y=200, line_dash="dash", line_color="orange", annotation_text="Poor Threshold (200)")
        fig_line.update_layout(height=360)
        st.plotly_chart(fig_line, use_container_width=True)

# ==========================================
# TAB 3: BATCH CSV PREDICTION
# ==========================================
with tab3:
    st.subheader("📂 Bulk AQI Prediction from CSV")
    st.write(f"Upload a CSV file containing these columns: `{', '.join(FEATURES)}`")

    uploaded_file = st.file_uploader("Upload Sensor CSV", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        missing_cols = [c for c in FEATURES if c not in batch_df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            clean_batch = batch_df[FEATURES].fillna(batch_df[FEATURES].median())
            preds = np.maximum(0, np.round(model.predict(clean_batch), 1))
            batch_df["Predicted_AQI"] = preds
            batch_df["AQI_Category"] = [get_aqi_details(v)[0] for v in preds]

            st.success(f"✅ Predicted AQI for {len(batch_df)} rows!")
            st.dataframe(batch_df.head(20), use_container_width=True)

            csv_out = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Predictions CSV",
                data=csv_out,
                file_name="aqi_predictions.csv",
                mime="text/csv",
                type="primary"
            )

# ==========================================
# TAB 4: MODEL INSIGHTS & CPCB REFERENCE
# ==========================================
with tab4:
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.subheader("🧠 What Drives the ML Model?")
        if hasattr(model, "feature_importances_"):
            imp_df = pd.DataFrame({
                "Pollutant": FEATURES,
                "Importance (%)": np.round(model.feature_importances_ * 100, 2)
            }).sort_values("Importance (%)", ascending=True)
            fig_imp = px.bar(
                imp_df, x="Importance (%)", y="Pollutant", orientation="h",
                color="Importance (%)", color_continuous_scale="Blues", text_auto=True
            )
            fig_imp.update_layout(showlegend=False, height=340)
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance is available for tree-based models like Random Forest.")

    with col_b:
        st.subheader("📋 Indian CPCB AQI Reference Scale")
        cpcb_table = pd.DataFrame({
            "AQI Range": ["0 – 50", "51 – 100", "101 – 200", "201 – 300", "301 – 400", "401 – 500+"],
            "Category": ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"],
            "Health Impact": [
                "Minimal impact",
                "Minor discomfort to sensitive people",
                "Discomfort to people with lung/heart disease",
                "Discomfort on prolonged exposure",
                "Respiratory illness on prolonged exposure",
                "Serious impact even on healthy people"
            ]
        })
        st.table(cpcb_table)