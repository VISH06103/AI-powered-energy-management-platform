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
    page_title="Universal Energy Analytics Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# UNIVERSAL FILE LOADER & AUTO-MAPPING ENGINE
# ==========================================
@st.cache_data
def generate_synthetic_data(days=30):
    """Generates fallback synthetic smart-meter data."""
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days * 24, freq="h")
    df = pd.DataFrame({"timestamp": dates})

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
    df["temperature"] = 20 + 8 * np.sin(2 * np.pi * df["hour"] / 24) + np.random.normal(0, 2, len(df))

    base_load = 0.5
    peak_multiplier = np.where(df["hour"].isin([7, 8, 9, 18, 19, 20, 21, 22]), 2.2, 1.0)
    hvac = np.abs(df["temperature"] - 21) * 0.12

    df["kwh"] = base_load + (np.random.normal(1.2, 0.3, len(df)) * peak_multiplier) + hvac
    df["kwh"] = df["kwh"].clip(lower=0.2)

    # Inefficiency spikes
    anomaly_indices = np.random.choice(df.index, size=int(len(df) * 0.04), replace=False)
    df.loc[anomaly_indices, "kwh"] *= np.random.uniform(2.5, 4.0, size=len(anomaly_indices))

    return df

def load_file_to_dataframe(uploaded_file):
    """Loads CSV, Excel, JSON, or Parquet files into a DataFrame."""
    file_name = uploaded_file.name.lower()
    try:
        if file_name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif file_name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(uploaded_file)
        elif file_name.endswith('.json'):
            return pd.read_json(uploaded_file)
        elif file_name.endswith('.parquet'):
            return pd.read_parquet(uploaded_file)
        else:
            st.error("❌ Unsupported file extension.")
            return None
    except Exception as e:
        st.error(f"❌ Failed to read file: {str(e)}")
        return None

def auto_detect_columns(df):
    """Intelligently detects time, consumption, and temperature columns."""
    time_col, kwh_col, temp_col = None, None, None

    # 1. Detect Timestamp Column
    for col in df.columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in ["time", "date", "datetime", "timestamp", "dt"]):
            time_col = col
            break
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            time_col = col
            break

    # Fallback timestamp search by attempting pd.to_datetime on string columns
    if not time_col:
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].dropna().iloc[:10])
                    time_col = col
                    break
                except (ValueError, TypeError):
                    continue

    # 2. Detect kWh / Energy Consumption Column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in ["kwh", "usage", "energy", "power", "consumption", "load", "demand"]):
            kwh_col = col
            break

    # If no keyword match, select the first numeric non-timestamp column
    if not kwh_col and len(numeric_cols) > 0:
        kwh_col = numeric_cols[0]

    # 3. Detect Temperature Column
    for col in numeric_cols:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in ["temp", "temperature", "weather", "deg"]):
            temp_col = col
            break

    return time_col, kwh_col, temp_col

# ==========================================
# ML PIPELINE (FORECASTING & ANOMALY DETECTION)
# ==========================================
class EnergyAnalyticsML:
    def __init__(self, data):
        self.df = data.copy()
        self.forecaster = RandomForestRegressor(n_estimators=100, random_state=42)
        self.anomaly_detector = IsolationForest(contamination=0.04, random_state=42)

    def prepare_features(self):
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
        return mean_absolute_error(y_test, preds)

    def detect_anomalies(self):
        X = self.df[["hour", "temperature", "kwh"]]
        self.df["is_anomaly"] = self.anomaly_detector.fit_predict(X)
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
# GROQ LLM AGENT FOR RECOMMENDATIONS
# ==========================================
def get_groq_recommendations(api_key, metrics_summary):
    if not api_key:
        return "⚠️ Please enter your Groq API key in the sidebar to generate AI energy recommendations."

    try:
        client = Groq(api_key=api_key)
        prompt = f"""
        You are an expert Smart Home Energy Management System (HEMS) AI agent.
        Analyze the following energy usage metrics and anomaly statistics:

        {metrics_summary}

        Provide concise, structured, actionable recommendations:
        1. **Inefficiency Spikes**: Address detected consumption anomalies.
        2. **Load Shifting Strategy**: Recommendations for peak vs off-peak hours.
        3. **Cost Savings Potential**: Quantitative estimation of possible energy reduction.

        Keep recommendations direct, practical, and in clear markdown format.
        """

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a specialized AI Energy Systems Analyst."},
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
# MAIN APP LAYOUT
# ==========================================
st.title("⚡ Residential Energy AI Analytics & Optimization")
st.caption("Universal File Processing | Anomaly Detection | ML Demand Forecasting | Groq AI Insights")

# Sidebar Configuration
st.sidebar.header("📂 Universal File Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload Any Energy File", 
    type=["csv", "xlsx", "xls", "json", "parquet"], 
    help="Upload energy dataset in CSV, Excel, JSON, or Parquet format."
)

st.sidebar.header("🔧 Settings & API Key")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password", help="Enter API key starting with gsk_...")
electricity_cost = st.sidebar.number_input("Electricity Rate ($/kWh)", value=0.18, step=0.01)

# File Parsing and Column Mapping Logic
data_ready = False
raw_df = None

if uploaded_file is not None:
    df_upload = load_file_to_dataframe(uploaded_file)
    
    if df_upload is not None:
        st.sidebar.subheader("🎯 Column Auto-Detection")
        auto_time, auto_kwh, auto_temp = auto_detect_columns(df_upload)

        # Allow user to confirm or override mapping
        selected_time_col = st.sidebar.selectbox(
            "Timestamp Column", 
            df_upload.columns, 
            index=df_upload.columns.get_loc(auto_time) if auto_time else 0
        )
        selected_kwh_col = st.sidebar.selectbox(
            "Energy Usage Column", 
            df_upload.columns, 
            index=df_upload.columns.get_loc(auto_kwh) if auto_kwh else 0
        )
        selected_temp_col = st.sidebar.selectbox(
            "Temperature Column (Optional)", 
            ["None"] + list(df_upload.columns), 
            index=(list(df_upload.columns).index(auto_temp) + 1) if auto_temp else 0
        )

        try:
            # Process dataframe to unified format
            df_processed = pd.DataFrame()
            df_processed["timestamp"] = pd.to_datetime(df_upload[selected_time_col])
            df_processed["kwh"] = pd.to_numeric(df_upload[selected_kwh_col], errors="coerce")

            if selected_temp_col != "None":
                df_processed["temperature"] = pd.to_numeric(df_upload[selected_temp_col], errors="coerce")
            else:
                df_processed["temperature"] = 20.0  # Default baseline

            df_processed = df_processed.dropna(subset=["timestamp", "kwh"]).sort_values("timestamp").reset_index(drop=True)

            # Feature Engineering
            df_processed["hour"] = df_processed["timestamp"].dt.hour
            df_processed["day_of_week"] = df_processed["timestamp"].dt.dayofweek
            df_processed["is_weekend"] = df_processed["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

            raw_df = df_processed
            data_ready = True
            st.sidebar.success(f"✅ Loaded {len(raw_df)} records!")

        except Exception as e:
            st.error(f"❌ Error parsing selected columns: {str(e)}")

if not data_ready:
    st.info("ℹ️ No file uploaded or parsing failed. Running with simulated energy data.")
    data_days = st.sidebar.slider("Simulation Days", min_value=14, max_value=90, value=30)
    raw_df = generate_synthetic_data(days=data_days)

# ==========================================
# ML PIPELINE EXECUTION & DASHBOARD
# ==========================================
with st.spinner("Executing Machine Learning Pipeline..."):
    ml_engine = EnergyAnalyticsML(raw_df)
    ml_engine.prepare_features()
    mae_score = ml_engine.train_forecaster()
    processed_df = ml_engine.detect_anomalies()
    forecast_df = ml_engine.forecast_next_24h()

# Top KPIs Row
latest_24h = processed_df.tail(24)
prev_24h = processed_df.iloc[-48:-24] if len(processed_df) >= 48 else latest_24h

today_kwh = latest_24h["kwh"].sum()
yesterday_kwh = prev_24h["kwh"].sum()
pct_change = ((today_kwh - yesterday_kwh) / yesterday_kwh * 100) if yesterday_kwh > 0 else 0

total_anomalies = processed_df.tail(24 * 7)["is_anomaly"].sum()
predicted_24h_kwh = forecast_df["predicted_kwh"].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("24h Actual Usage", f"{today_kwh:.1f} kWh", f"{pct_change:+.1f}% vs yesterday")
col2.metric("24h Forecasted Usage", f"{predicted_24h_kwh:.1f} kWh")
col3.metric("Anomalies (Past 7 Days)", f"{int(total_anomalies)} Spikes", delta_color="inverse")
col4.metric("Est. Daily Cost", f"${today_kwh * electricity_cost:.2f}")

st.markdown("---")

# Visualizations Tab Setup
tab1, tab2, tab3 = st.tabs(["📊 Real-Time Monitoring & Anomalies", "🔮 24-Hour Forecast", "🤖 AI Recommendations"])

with tab1:
    st.subheader("Historical Consumption & Inefficiency Spikes")
    display_df = processed_df.tail(24 * 7).copy()
    
    fig = _go.Figure()
    fig.add_trace(_go.Scatter(
        x=display_df["timestamp"], 
        y=display_df["kwh"],
        mode='lines',
        name='Consumption (kWh)',
        line=dict(color='#1f77b4', width=2)
    ))
    
    anomalies = display_df[display_df["is_anomaly"] == 1]
    fig.add_trace(_go.Scatter(
        x=anomalies["timestamp"],
        y=anomalies["kwh"],
        mode='markers',
        name='Detected Inefficiency/Spike',
        marker=dict(color='red', size=8, symbol='x')
    ))
    
    fig.update_layout(
        title="Consumption Profile (Red = Detected Anomaly)",
        xaxis_title="Timestamp",
        yaxis_title="kWh",
        hovermode="x unified",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Next 24-Hour Demand Forecast")
    fig_forecast = _go.Figure()
    
    fig_forecast.add_trace(_go.Scatter(
        x=latest_24h["timestamp"],
        y=latest_24h["kwh"],
        mode='lines+markers',
        name='Actual (Past 24h)',
        line=dict(color='#2ca02c')
    ))
    
    fig_forecast.add_trace(_go.Scatter(
        x=forecast_df["timestamp"],
        y=forecast_df["predicted_kwh"],
        mode='lines+markers',
        name='Predicted (Next 24h)',
        line=dict(color='#ff7f0e', dash='dash')
    ))
    
    fig_forecast.update_layout(
        title=f"ML Random Forest Regressor (MAE: {mae_score:.3f} kWh)",
        xaxis_title="Timestamp",
        yaxis_title="kWh",
        template="plotly_white"
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

with tab3:
    st.subheader("LLM-Powered Energy Optimization Agent")
    peak_hours_avg = processed_df[processed_df["hour"].isin([18, 19, 20, 21])]["kwh"].mean()
    off_peak_avg = processed_df[~processed_df["hour"].isin([18, 19, 20, 21])]["kwh"].mean()

    summary_text = f"""
    - **Total 24h Consumption**: {today_kwh:.2f} kWh
    - **Expected Next 24h Consumption**: {predicted_24h_kwh:.2f} kWh
    - **Anomalies/Spikes Detected (Last 7 days)**: {int(total_anomalies)} hours flagged
    - **Average Peak Hour Usage (6 PM - 10 PM)**: {peak_hours_avg:.2f} kWh/hr
    - **Average Off-Peak Hour Usage**: {off_peak_avg:.2f} kWh/hr
    - **Current Electricity Rate**: ${electricity_cost}/kWh
    """

    if st.button("Generate AI Optimization Plan", type="primary"):
        with st.spinner("Analyzing data through Groq Llama 3..."):
            ai_recommendation = get_groq_recommendations(groq_api_key, summary_text)
            st.markdown(ai_recommendation)
    else:
        st.info("Click the button above to generate customized, real-time energy insights.")
