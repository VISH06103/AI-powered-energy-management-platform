import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as _go
import streamlit as st
from groq import Groq
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Residential Energy AI Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# DATA ENGINE & SYNTHETIC DATA GENERATION
# ==========================================
@st.cache_data
def generate_energy_data(days=60):
    """Generates synthetic hourly smart-meter data with temperature and anomaly spikes."""
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days * 24, freq="h")
    df = pd.DataFrame({"timestamp": dates})

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
    df["month"] = df["timestamp"].dt.month

    # Outdoor temperature simulation (°C)
    df["temperature"] = (
        20 + 8 * np.sin(2 * np.pi * df["hour"] / 24) + np.random.normal(0, 2, len(df))
    )

    # Base load (refrigerator, standby power)
    base_load = 0.5

    # Peak usage patterns (morning 7-9 AM, evening 6-10 PM)
    peak_multiplier = np.where(
        df["hour"].isin([7, 8, 9, 18, 19, 20, 21, 22]), 2.2, 1.0
    )

    # HVAC consumption proportional to extreme temp
    hvac = np.abs(df["temperature"] - 21) * 0.12

    # Calculate kWh usage
    df["kwh"] = (
        base_load
        + (np.random.normal(1.2, 0.3, len(df)) * peak_multiplier)
        + hvac
    )
    df["kwh"] = df["kwh"].clip(lower=0.2)

    # Inject random energy inefficiencies/anomalies (5% of data)
    anomaly_indices = np.random.choice(
        df.index, size=int(len(df) * 0.04), replace=False
    )
    df.loc[anomaly_indices, "kwh"] *= np.random.uniform(2.5, 4.0, size=len(anomaly_indices))

    return df

# ==========================================
# ML PIPELINE (FORECASTING & ANOMALY DETECTION)
# ==========================================
class EnergyAnalyticsML:
    def __init__(self, data):
        self.df = data.copy()
        self.forecaster = RandomForestRegressor(n_estimators=100, random_state=42)
        self.anomaly_detector = IsolationForest(contamination=0.04, random_state=42)

    def prepare_features(self):
        # Create lag features for forecasting
        self.df["kwh_lag_1h"] = self.df["kwh"].shift(1)
        self.df["kwh_lag_24h"] = self.df["kwh"].shift(24)
        self.df["rolling_avg_6h"] = self.df["kwh"].shift(1).rolling(6).mean()
        self.df.dropna(inplace=True)

        self.features = [
            "hour",
            "day_of_week",
            "is_weekend",
            "temperature",
            "kwh_lag_1h",
            "kwh_lag_24h",
            "rolling_avg_6h",
        ]

    def train_forecaster(self):
        X = self.df[self.features]
        y = self.df["kwh"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        self.forecaster.fit(X_train, y_train)
        preds = self.forecaster.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        return mae

    def detect_anomalies(self):
        X = self.df[["hour", "temperature", "kwh"]]
        self.df["is_anomaly"] = self.anomaly_detector.fit_predict(X)
        # -1 represents an anomaly in IsolationForest
        self.df["is_anomaly"] = self.df["is_anomaly"].map({1: 0, -1: 1})
        return self.df

    def forecast_next_24h(self):
        last_row = self.df.iloc[-1].copy()
        future_forecasts = []

        curr_kwh_lag_1h = last_row["kwh"]
        history_kwh = list(self.df["kwh"].tail(24).values)

        current_time = last_row["timestamp"]

        for i in range(1, 25):
            next_time = current_time + pd.Timedelta(hours=i)
            hour = next_time.hour
            day_of_week = next_time.dayofweek
            is_weekend = 1 if day_of_week >= 5 else 0
            temp = 20 + 8 * np.sin(2 * np.pi * hour / 24)
            lag_24h = history_kwh[-24] if len(history_kwh) >= 24 else curr_kwh_lag_1h
            rolling_6h = np.mean(history_kwh[-6:])

            input_features = pd.DataFrame(
                [[hour, day_of_week, is_weekend, temp, curr_kwh_lag_1h, lag_24h, rolling_6h]],
                columns=self.features,
            )

            pred_kwh = float(self.forecaster.predict(input_features)[0])
            future_forecasts.append({"timestamp": next_time, "predicted_kwh": pred_kwh})

            curr_kwh_lag_1h = pred_kwh
            history_kwh.append(pred_kwh)

        return pd.DataFrame(future_forecasts)

# ==========================================
# GROQ LLM AGENT FOR NLP RECOMMENDATIONS
# ==========================================
def get_groq_recommendations(api_key, metrics_summary):
    """Generates personalized energy recommendations using Groq Llama-3."""
    if not api_key:
        return "⚠️ Please provide a valid Groq API Key in the sidebar to generate AI recommendations."

    try:
        client = Groq(api_key=api_key)
        prompt = f"""
        You are an expert Smart Home Energy Management System (HEMS) AI agent.
        Analyze the following household energy consumption data and anomalies summary:

        {metrics_summary}

        Provide concise, structured, actionable energy-saving recommendations:
        1. **Immediate Inefficiency Alerts**: Address detected anomalies.
        2. **Optimization Strategy**: Peaks vs off-peak load shifting suggestions.
        3. **Estimated Savings**: Quantitative estimate of potential energy/cost reduction.

        Keep recommendations direct, highly specific, and formatted in clear markdown.
        """

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful AI Energy Analyst."},
                {"role": "user", "content": prompt},
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=600,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"❌ Groq API Error: {str(e)}"

# ==========================================
# STREAMLIT UI LAYOUT
# ==========================================
st.title("⚡ Residential Energy AI Analytics & Optimization")
st.caption("AI-Powered Real-Time Consumption Monitoring, Forecasting & Anomaly Detection")

# Sidebar Configuration
st.sidebar.header("🔧 Configuration & API")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password", help="Enter your Groq API key (gsk_...)")
electricity_cost = st.sidebar.number_input("Electricity Rate ($/kWh)", value=0.18, step=0.01)
data_days = st.sidebar.slider("Historical Data Range (Days)", min_value=14, max_value=90, value=30)

# Load Data and Run ML Pipeline
with st.spinner("Processing smart meter data & executing ML models..."):
    raw_df = generate_energy_data(days=data_days)
    ml_engine = EnergyAnalyticsML(raw_df)
    ml_engine.prepare_features()
    mae_score = ml_engine.train_forecaster()
    processed_df = ml_engine.detect_anomalies()
    forecast_df = ml_engine.forecast_next_24h()

# Top KPIs Row
latest_24h = processed_df.tail(24)
prev_24h = processed_df.iloc[-48:-24]

today_kwh = latest_24h["kwh"].sum()
yesterday_kwh = prev_24h["kwh"].sum()
pct_change = ((today_kwh - yesterday_kwh) / yesterday_kwh) * 100

total_anomalies = processed_df.tail(24 * 7)["is_anomaly"].sum()
predicted_24h_kwh = forecast_df["predicted_kwh"].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("24h Actual Usage", f"{today_kwh:.1f} kWh", f"{pct_change:+.1f}% vs yesterday")
col2.metric("24h Forecasted Usage", f"{predicted_24h_kwh:.1f} kWh")
col3.metric("Anomalies (Past 7 Days)", f"{total_anomalies} Events", delta_color="inverse")
col4.metric("Est. Daily Cost", f"${today_kwh * electricity_cost:.2f}")

st.markdown("---")

# Visualizations Tab Setup
tab1, tab2, tab3 = st.tabs(["📊 Real-Time Monitoring & Anomalies", "🔮 24-Hour Forecast", "🤖 AI Recommendations"])

with tab1:
    st.subheader("Historical Consumption & Inefficiency Detection")
    
    # Filter for last 7 days chart display
    display_df = processed_df.tail(24 * 7).copy()
    
    fig = _go.Figure()
    
    # Base Consumption Line
    fig.add_trace(_go.Scatter(
        x=display_df["timestamp"], 
        y=display_df["kwh"],
        mode='lines',
        name='Consumption (kWh)',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Overlay Anomalies
    anomalies = display_df[display_df["is_anomaly"] == 1]
    fig.add_trace(_go.Scatter(
        x=anomalies["timestamp"],
        y=anomalies["kwh"],
        mode='markers',
        name='Detected Inefficiency/Spike',
        marker=dict(color='red', size=8, symbol='x')
    ))
    
    fig.update_layout(
        title="7-Day Consumption Profile (Red = Detected Anomaly)",
        xaxis_title="Timestamp",
        yaxis_title="kWh",
        hovermode="x unified",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Next 24-Hour Energy Demand Prediction")
    
    fig_forecast = _go.Figure()
    
    # Historical tail
    fig_forecast.add_trace(_go.Scatter(
        x=latest_24h["timestamp"],
        y=latest_24h["kwh"],
        mode='lines+markers',
        name='Actual (Past 24h)',
        line=dict(color='#2ca02c')
    ))
    
    # Forecast
    fig_forecast.add_trace(_go.Scatter(
        x=forecast_df["timestamp"],
        y=forecast_df["predicted_kwh"],
        mode='lines+markers',
        name='Predicted (Next 24h)',
        line=dict(color='#ff7f0e', dash='dash')
    ))
    
    fig_forecast.update_layout(
        title=f"ML Forecast Model (RandomForest MAE: {mae_score:.3f} kWh)",
        xaxis_title="Time",
        yaxis_title="kWh",
        template="plotly_white"
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

with tab3:
    st.subheader("LLM-Powered Smart Energy Advisory")
    
    # Prepare summary data for LLM
    peak_hours_avg = processed_df[processed_df["hour"].isin([18, 19, 20, 21])]["kwh"].mean()
    off_peak_avg = processed_df[~processed_df["hour"].isin([18, 19, 20, 21])]["kwh"].mean()
    
    summary_text = f"""
    - **Total 24h Consumption**: {today_kwh:.2f} kWh
    - **Expected Next 24h Consumption**: {predicted_24h_kwh:.2f} kWh
    - **Anomalies/Spikes Detected (Last 7 days)**: {total_anomalies} hours flagged
    - **Average Peak Hour Usage (6 PM - 10 PM)**: {peak_hours_avg:.2f} kWh/hr
    - **Average Off-Peak Hour Usage**: {off_peak_avg:.2f} kWh/hr
    - **Current Cost Rate**: ${electricity_cost}/kWh
    """

    if st.button("Generate AI Optimization Plan", type="primary"):
        with st.spinner("Analyzing energy context via Groq LLM..."):
            ai_recommendation = get_groq_recommendations(groq_api_key, summary_text)
            st.markdown(ai_recommendation)
    else:
        st.info("Click the button above to synthesize real-time energy insights using Groq Llama 3.")
