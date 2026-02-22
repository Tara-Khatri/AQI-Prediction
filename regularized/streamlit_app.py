import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from io import StringIO
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="PM2.5 Future Prediction",
    page_icon="🌫️",
    layout="wide"
)

st.title("🌫️ PM2.5 Future Prediction Dashboard")
st.markdown("*Predict future PM2.5 levels using Random Forest Model*")
st.divider()

# ──────────────────────────────────────────────
# Load Model and Configuration
# ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained model"""
    model_files = [
        os.path.join(BASE_DIR, 'improved_random_forest_pm25_model.pkl')
    ]
    
    for model_file in model_files:
        try:
            model = joblib.load(model_file)
            st.success(f" Model loaded from: {model_file}")
            return model
        except FileNotFoundError:
            continue
        except Exception as e:
            st.warning(f"!! Error loading {model_file}: {e}")
            continue
    
    st.error("❌ Model file not found. Please ensure model file exists.")
    return None

@st.cache_resource
def load_features():
    """Load selected features"""
    features = joblib.load(os.path.join(BASE_DIR,'selected_features.pkl'))
    return features

model = load_model()
if model is None:
    st.stop()

selected_features = load_features()
st.info(f"📊 Using {len(selected_features)} features for prediction")

# ──────────────────────────────────────────────
# Feature Engineering Function
# ──────────────────────────────────────────────
def create_features(df, drop_na=True):
    """
    Create all features needed for prediction
    """
    df = df.copy()

    # Standardize column names to match notebook
    column_mapping = {
        'date': 'datetime',
        'Khumaltar Temperature (°C)': 'temperature',
        'Khumaltar Relative Humidity (%)': 'relative_humidity',
        'PM2.5': 'pm25'
    }
    existing_mappings = {old: new for old, new in column_mapping.items() if old in df.columns}
    df = df.rename(columns=existing_mappings)

    # Ensure required columns exist
    required_cols = ['datetime', 'temperature', 'relative_humidity', 'pm25']
    missing_core = [c for c in required_cols if c not in df.columns]
    if missing_core:
        # If core columns are missing, return empty to let callers handle it
        return pd.DataFrame()

    # Convert datetime and sort (same as notebook)
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    
    # Only drop invalid dates, not full rows yet
    df = df.dropna(subset=['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # Basic time features
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['weekofyear'] = df['datetime'].dt.isocalendar().week

    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    # Weekend indicator
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    # Time of day categories (not used by model but harmless)
    df['time_of_day'] = pd.cut(
        df['hour'],
        bins=[0, 6, 12, 18, 24],
        labels=['Night', 'Morning', 'Afternoon', 'Evening'],
        include_lowest=True
    )

    # Lag features (previous values)
    for lag in [1, 2, 3, 6, 12, 24, 48, 72]:
        df[f'pm25_lag_{lag}'] = df['pm25'].shift(lag)
        df[f'temp_lag_{lag}'] = df['temperature'].shift(lag)
        df[f'humidity_lag_{lag}'] = df['relative_humidity'].shift(lag)

    # Rolling statistics (min_periods=1 EXACT as notebook)
    for window in [3, 6, 12, 24, 48, 72]:
        df[f'pm25_rolling_mean_{window}'] = df['pm25'].rolling(window=window, min_periods=1).mean()
        df[f'pm25_rolling_std_{window}'] = df['pm25'].rolling(window=window, min_periods=1).std()
        df[f'pm25_rolling_max_{window}'] = df['pm25'].rolling(window=window, min_periods=1).max()
        df[f'pm25_rolling_min_{window}'] = df['pm25'].rolling(window=window, min_periods=1).min()
        df[f'temp_rolling_mean_{window}'] = df['temperature'].rolling(window=window, min_periods=1).mean()
        df[f'humidity_rolling_mean_{window}'] = df['relative_humidity'].rolling(window=window, min_periods=1).mean()

    # Rate of change
    df['pm25_change_1hr'] = df['pm25'].diff(1)
    df['pm25_change_3hr'] = df['pm25'].diff(3)
    df['pm25_change_6hr'] = df['pm25'].diff(6)
    df['temp_change_1hr'] = df['temperature'].diff(1)
    df['humidity_change_1hr'] = df['relative_humidity'].diff(1)

    # Acceleration
    df['pm25_acceleration'] = df['pm25_change_1hr'].diff(1)

    # Interaction features
    df['temp_humidity'] = df['temperature'] * df['relative_humidity']
    df['temp_squared'] = df['temperature'] ** 2
    df['humidity_squared'] = df['relative_humidity'] ** 2
    df['temp_humidity_ratio'] = df['temperature'] / (df['relative_humidity'] + 1)
    df['pm25_temp_ratio'] = df['pm25'] / (df['temperature'] + 1)

    # Exponential weighted moving averages
    df['pm25_ewm_12'] = df['pm25'].ewm(span=12, adjust=False).mean()
    df['pm25_ewm_24'] = df['pm25'].ewm(span=24, adjust=False).mean()

    # Drop rows with NaN if requested
    if drop_na:
        df = df.dropna()

    return df


# ──────────────────────────────────────────────
# Load Horizon Models
# ──────────────────────────────────────────────
@st.cache_resource
def load_horizon_models():
    HORIZONS = [1, 2, 3, 6, 12, 24]
    models = {}
    for h in HORIZONS:
        try:
            models[h] = joblib.load(os.path.join(BASE_DIR,f'model_{h}h_ahead.pkl'))
        except FileNotFoundError:
            pass
        except Exception as e:
            st.warning(f"⚠️ Could not load model_{h}h_ahead.pkl: {e}")
    return models

horizon_models = load_horizon_models()

if horizon_models:
    st.success(f"✅ Horizon models loaded for horizons: {sorted(horizon_models.keys())}h")
else:
    st.warning("⚠️ Horizon models not found — place model_Xh_ahead.pkl files next to app.py")


def _find_best_horizon(step, available):
    """Return largest horizon <= step."""
    best = available[0]
    for h in available:
        if h <= step:
            best = h
    return best


# ──────────────────────────────────────────────
# Future Prediction Function (Horizon-Based)
# ──────────────────────────────────────────────
def predict_future(df_history, n_hours, future_temp=None, future_humidity=None):
    """
    Direct multi-step horizon prediction.
    Each forecast step uses the best matching pre-trained horizon model.
    Features are always built from REAL historical data — no iterative rollout.
    Falls back to iterative rollout if horizon models are unavailable.
    """
    available_horizons = sorted(horizon_models.keys()) if horizon_models else []

    # ── FALLBACK: iterative rollout ───────────────────────────────────────────
    if not available_horizons:
        st.info("ℹ️ Using iterative fallback — train horizon models for better forecasts")
        df        = df_history.copy()
        last_temp     = df['temperature'].iloc[-1]
        last_humidity = df['relative_humidity'].iloc[-1]
        last_datetime = df['datetime'].iloc[-1]
        last_pm25     = df['pm25'].iloc[-1]
        predictions, prediction_times = [], []
        progress_bar = st.progress(0)
        status_text  = st.empty()
        try:
            for i in range(1, n_hours + 1):
                progress_bar.progress(i / n_hours)
                status_text.text(f"Predicting hour {i}/{n_hours}...")
                next_datetime = last_datetime + timedelta(hours=i)
                temp_val = future_temp[i-1]     if (future_temp     is not None and i <= len(future_temp))     else last_temp
                hum_val  = future_humidity[i-1] if (future_humidity is not None and i <= len(future_humidity)) else last_humidity
                new_row = pd.DataFrame([{'datetime': next_datetime, 'pm25': last_pm25,
                                          'temperature': temp_val, 'relative_humidity': hum_val}])
                df = pd.concat([df, new_row], ignore_index=True)
                df_feat = create_features(df, drop_na=False)
                if len(df_feat) == 0: break
                last_row = df_feat.iloc[[-1]]
                missing = [f for f in selected_features if f not in last_row.columns]
                for feat in missing: last_row[feat] = 0
                X_pred   = last_row[selected_features].fillna(last_pm25)
                pred     = max(0.0, float(model.predict(X_pred)[0]))
                df.loc[df.index[-1], 'pm25'] = pred
                last_pm25 = pred
                predictions.append(pred); prediction_times.append(next_datetime)
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            import traceback; st.text(traceback.format_exc())
        progress_bar.empty(); status_text.empty()
        if not predictions:
            return pd.DataFrame()
        return pd.DataFrame({'datetime': prediction_times, 'predicted_pm25': predictions})

    # ── MAIN: Horizon-based direct prediction ─────────────────────────────────
    df_hist   = df_history.copy()
    df_feat   = create_features(df_hist, drop_na=False)
    if len(df_feat) == 0:
        st.error("❌ Feature creation failed.")
        return pd.DataFrame()

    base_row      = df_feat.iloc[[-1]].copy()
    last_datetime = df_hist['datetime'].iloc[-1]
    last_temp     = df_hist['temperature'].iloc[-1]
    last_humidity = df_hist['relative_humidity'].iloc[-1]
    predictions, prediction_times = [], []

    progress_bar = st.progress(0)
    status_text  = st.empty()

    try:
        for i in range(1, n_hours + 1):
            progress_bar.progress(i / n_hours)
            status_text.text(f"Predicting hour {i}/{n_hours} (using +{_find_best_horizon(i, available_horizons)}h model)...")

            next_datetime    = last_datetime + timedelta(hours=i)
            temp_val = future_temp[i-1]     if (future_temp     is not None and i <= len(future_temp))     else last_temp
            hum_val  = future_humidity[i-1] if (future_humidity is not None and i <= len(future_humidity)) else last_humidity

            # Select best horizon model for this step
            best_h   = _find_best_horizon(i, available_horizons)
            model_h  = horizon_models[best_h]

            # Build feature row — update time and weather features for target hour
            feat_row         = base_row.copy()
            target_hour      = next_datetime.hour
            target_month     = next_datetime.month
            target_dow       = next_datetime.weekday()

            feat_row['hour_sin']        = np.sin(2 * np.pi * target_hour  / 24)
            feat_row['hour_cos']        = np.cos(2 * np.pi * target_hour  / 24)
            feat_row['month_sin']       = np.sin(2 * np.pi * target_month / 12)
            feat_row['month_cos']       = np.cos(2 * np.pi * target_month / 12)
            feat_row['dayofweek_sin']   = np.sin(2 * np.pi * target_dow   / 7)
            feat_row['dayofweek_cos']   = np.cos(2 * np.pi * target_dow   / 7)
            feat_row['is_weekend']      = int(target_dow >= 5)
            feat_row['temperature']     = temp_val
            feat_row['relative_humidity'] = hum_val
            feat_row['temp_squared']    = temp_val ** 2
            feat_row['humidity_squared']= hum_val ** 2
            feat_row['temp_humidity']   = temp_val * hum_val
            feat_row['temp_humidity_ratio'] = temp_val / (hum_val + 1)

            missing = [f for f in selected_features if f not in feat_row.columns]
            for feat in missing: feat_row[feat] = 0

            X_pred = feat_row[selected_features].fillna(0)
            pred   = max(0.0, float(model_h.predict(X_pred)[0]))

            predictions.append(pred)
            prediction_times.append(next_datetime)

    except Exception as e:
        st.error(f"❌ Horizon prediction error: {e}")
        import traceback; st.text(traceback.format_exc())

    progress_bar.empty()
    status_text.empty()

    if not predictions:
        st.error("❌ No predictions generated.")
        return pd.DataFrame()

    return pd.DataFrame({'datetime': prediction_times, 'predicted_pm25': predictions})



# ──────────────────────────────────────────────
# Sidebar Configuration
# ──────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

prediction_mode = st.sidebar.radio(
    "Prediction Mode",
    ["🌅 Tomorrow's Forecast"]
)

# Hours slider (not shown for Tomorrow's Forecast mode)
if prediction_mode != "🌅 Tomorrow's Forecast":
    n_hours = st.sidebar.slider(
        "Hours to Predict Ahead",
        min_value=1,
        max_value=72,
        value=24,
        step=1
    )
else:
    n_hours = 24  # Default for Tomorrow's Forecast (set internally in that mode)

st.sidebar.divider()
st.sidebar.markdown("### 📈 Model Performance")
st.sidebar.info("""
**Test Metrics:**
- R² Score: 0.9977
- RMSE: 1.28 µg/m³
- MAE: 0.63 µg/m³
- MAPE: 2.99%
""")

# ──────────────────────────────────────────────
# Main Content
# ──────────────────────────────────────────────
# Initialize session state
if 'df_history' not in st.session_state:
    st.session_state.df_history = None

df_history = st.session_state.df_history

if prediction_mode == "🌅 Tomorrow's Forecast":
    st.subheader("🌅 Tomorrow's PM2.5 Forecast")
    st.markdown("""
    **Quick & Easy** - Enter today's values and get tomorrow's PM2.5 predictions!
    
    ⚠️ **Important:** Enter realistic current values. The model uses these to generate historical patterns.
    If your current PM2.5 is unusually high/low, predictions may reflect that trend.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Today's Current Values")
        
        # Get current time
        now = datetime.now()
        current_hour = now.replace(minute=0, second=0, microsecond=0)

        # ── Load real last values from CSV as defaults ────────────────
        try:
            _csv = pd.read_csv(os.path.join(BASE_DIR,'Khumaltar hourly merged 2022-2024.csv'), encoding='latin1')
            _csv = _csv.rename(columns={
                'date': 'datetime',
                'Khumaltar Temperature (°C)': 'temperature',
                'Khumaltar Relative Humidity (%)': 'relative_humidity',
                'PM2.5': 'pm25'
            })
            _csv['pm25']              = pd.to_numeric(_csv.get('pm25',              pd.Series()), errors='coerce')
            _csv['temperature']       = pd.to_numeric(_csv.get('temperature',       pd.Series()), errors='coerce')
            _csv['relative_humidity'] = pd.to_numeric(_csv.get('relative_humidity', pd.Series()), errors='coerce')
            _default_pm25  = float(round(_csv['pm25'].dropna().iloc[-1],              1))
            _default_temp  = float(round(_csv['temperature'].dropna().iloc[-1],       1))
            _default_hum   = float(round(_csv['relative_humidity'].dropna().iloc[-1], 1))
            st.caption(f"📂 Defaults loaded from CSV — last entry: PM2.5={_default_pm25}, Temp={_default_temp}°C, RH={_default_hum}%")
        except Exception:
            _default_pm25, _default_temp, _default_hum = 50.0, 20.0, 70.0

        # Input current values — pre-filled from CSV
        current_pm25 = st.number_input(
            "Current PM2.5 (µg/m³)",
            min_value=0.0,
            value=_default_pm25,
            step=1.0,
            help="Enter the current PM2.5 reading"
        )
        
        current_temp = st.number_input(
            "Current Temperature (°C)",
            value=_default_temp,
            step=1.0,
            help="Enter the current temperature"
        )
        
        current_humidity = st.number_input(
            "Current Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=_default_hum,
            step=1.0,
            help="Enter the current relative humidity"
        )
        
        st.markdown("---")
        st.markdown("### 📈 Recent Hours (Optional)")
        st.info("💡 Enter recent hourly values for better accuracy. Leave blank to use current values.")
        
        num_recent = st.slider("How many recent hours do you have?", min_value=0, max_value=24, value=3, step=1)
        
        recent_data = []
        if num_recent > 0:
            for i in range(num_recent):
                with st.expander(f"Hour -{num_recent-i} ({(current_hour - timedelta(hours=num_recent-i)).strftime('%H:00')})"):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        pm_val = st.number_input(f"PM2.5", min_value=0.0, value=current_pm25, key=f"tom_pm_{i}")
                    with col_b:
                        temp_val = st.number_input(f"Temp (°C)", value=current_temp, key=f"tom_temp_{i}")
                    with col_c:
                        hum_val = st.number_input(f"Humidity (%)", min_value=0.0, max_value=100.0, value=current_humidity, key=f"tom_hum_{i}")
                    recent_data.append({
                        'datetime': current_hour - timedelta(hours=num_recent-i),
                        'pm25': pm_val,
                        'temperature': temp_val,
                        'relative_humidity': hum_val
                    })
    
    with col2:
        st.markdown("### ⚙️ Settings")
        
        forecast_type = st.radio(
            "Forecast Type",
            ["Tomorrow (24h)", "Next 12 hours", "Next 48 hours"],
            help="Choose how far ahead to predict"
        )
        
        if forecast_type == "Tomorrow (24h)":
            forecast_hours = 24
        elif forecast_type == "Next 12 hours":
            forecast_hours = 12
        else:
            forecast_hours = 48
        
        st.markdown("---")
        st.markdown("### 🌤️ Tomorrow's Weather")
        st.info("💡 Optional: Enter forecasted weather for better predictions")
        
        use_weather_forecast = st.checkbox("Use weather forecast", value=False)
        
        if use_weather_forecast:
            tomorrow_temp = st.number_input("Tomorrow's Avg Temp (°C)", value=current_temp, step=1.0)
            tomorrow_humidity = st.number_input("Tomorrow's Avg Humidity (%)", min_value=0.0, max_value=100.0, value=current_humidity, step=1.0)
        else:
            tomorrow_temp = None
            tomorrow_humidity = None
    
    if st.button("🚀 Predict Tomorrow's PM2.5", type="primary", use_container_width=True):
        with st.spinner("Generating historical data and making predictions..."):
            hours_back = 72
            df_history = None  # ← always initialise before try/except

            # ── Try loading real CSV first ──────────────────────────────
            try:
                sample_df = pd.read_csv(os.path.join(BASE_DIR,'Khumaltar hourly merged 2022-2024.csv'), encoding='latin1')
                column_mapping = {
                    'date': 'datetime',
                    'Khumaltar Temperature (°C)': 'temperature',
                    'Khumaltar Relative Humidity (%)': 'relative_humidity',
                    'PM2.5': 'pm25'
                }
                existing_mappings = {old: new for old, new in column_mapping.items() if old in sample_df.columns}
                sample_df = sample_df.rename(columns=existing_mappings)
                sample_df['datetime'] = pd.to_datetime(sample_df['datetime'], errors='coerce')
                sample_df = sample_df.dropna(subset=['datetime']).sort_values('datetime')

                sample_tail = sample_df.tail(hours_back).copy()

                # Scale to current pm25 (multiply preserves spike shape)
                if sample_tail['pm25'].std() > 0:
                    scale = current_pm25 / (sample_tail['pm25'].mean() + 1e-9)
                    scale = float(np.clip(scale, 0.3, 3.0))
                    sample_tail['pm25'] = (sample_tail['pm25'] * scale).clip(lower=0)
                else:
                    sample_tail['pm25'] = current_pm25

                temp_diff = current_temp - sample_tail['temperature'].mean()
                hum_diff  = current_humidity - sample_tail['relative_humidity'].mean()
                sample_tail['temperature']       = sample_tail['temperature'] + temp_diff
                sample_tail['relative_humidity'] = (sample_tail['relative_humidity'] + hum_diff).clip(0, 100)
                sample_tail['datetime'] = [current_hour - timedelta(hours=i) for i in range(hours_back - 1, -1, -1)]

                df_history = sample_tail[['datetime', 'pm25', 'temperature', 'relative_humidity']].reset_index(drop=True)

            except Exception:
                # ── Fallback: realistic synthetic history ───────────────
                try:
                    np.random.seed(int(datetime.now().timestamp()) % 10000)

                    past_hours = np.array([
                        (current_hour - timedelta(hours=hours_back - 1 - i)).hour
                        for i in range(hours_back)
                    ])

                    # Khumaltar diurnal PM2.5 pattern (morning + evening peaks)
                    morning_peak = np.exp(-0.5 * ((past_hours - 8)  / 3) ** 2)
                    evening_peak = np.exp(-0.5 * ((past_hours - 19) / 3) ** 2)
                    night_low    = np.exp(-0.5 * ((past_hours - 3)  / 2) ** 2)
                    diurnal      = 1.0 + 0.4 * morning_peak + 0.35 * evening_peak - 0.2 * night_low

                    # AR(1) correlated noise — spikes persist across hours
                    raw_noise    = np.random.normal(0, 1, hours_back)
                    ar_noise     = np.zeros(hours_back)
                    ar_noise[0]  = raw_noise[0]
                    for k in range(1, hours_back):
                        ar_noise[k] = 0.85 * ar_noise[k - 1] + raw_noise[k]
                    ar_noise = ar_noise / (ar_noise.std() + 1e-9) * (current_pm25 * 0.15)

                    pm25_values = np.clip(current_pm25 * diurnal + ar_noise, 0, None)
                    temp_values = (current_temp + 4 * np.sin(2 * np.pi * (past_hours - 6) / 24)
                                   + np.random.normal(0, 0.5, hours_back))
                    hum_values  = np.clip(
                        current_humidity - 10 * np.sin(2 * np.pi * (past_hours - 6) / 24)
                        + np.random.normal(0, 2, hours_back),
                        0, 100
                    )

                    df_history = pd.DataFrame({
                        'datetime':          [current_hour - timedelta(hours=hours_back - 1 - i) for i in range(hours_back)],
                        'pm25':              pm25_values,
                        'temperature':       temp_values,
                        'relative_humidity': hum_values,
                    })
                except Exception as fallback_err:
                    st.error(f"❌ Failed to build historical data: {fallback_err}")
                    st.stop()

            # ── Safety check ────────────────────────────────────────────
            if df_history is None:
                st.error("❌ Could not build historical data. Please check your inputs.")
                st.stop()

            # ── Inject recent user-provided data ───────────────────────
            if recent_data:
                for i, data in enumerate(reversed(recent_data)):
                    idx = len(df_history) - 1 - i
                    if idx >= 0:
                        df_history.loc[idx, 'pm25']              = data['pm25']
                        df_history.loc[idx, 'temperature']       = data['temperature']
                        df_history.loc[idx, 'relative_humidity'] = data['relative_humidity']

            # Only overwrite pm25 if user has manually entered recent data.
            # Otherwise keep the real CSV last value so lag features stay accurate.
            if recent_data:
                df_history.loc[df_history.index[-1], 'pm25']              = current_pm25
            df_history.loc[df_history.index[-1], 'temperature']       = current_temp
            df_history.loc[df_history.index[-1], 'relative_humidity'] = current_humidity

            # ── Future weather with diurnal variation ───────────────────
            future_temp     = None
            future_humidity = None
            if use_weather_forecast and tomorrow_temp is not None:
                start_hour            = (current_hour + timedelta(hours=1)).hour
                forecast_hour_indices = np.array([(start_hour + i) % 24 for i in range(forecast_hours)])
                future_temp     = (tomorrow_temp + 4 * np.sin(2 * np.pi * (forecast_hour_indices - 6) / 24)).tolist()
                future_humidity = np.clip(
                    tomorrow_humidity - 10 * np.sin(2 * np.pi * (forecast_hour_indices - 6) / 24),
                    0, 100
                ).tolist()

            # ── Make predictions (single call, diurnal weather already set above) ──
            predictions_df = predict_future(df_history, forecast_hours, future_temp, future_humidity)
            
            if len(predictions_df) > 0:
                st.success(f"✅ Generated {forecast_hours}-hour forecast!")
                
                # Display key metrics
                st.markdown("---")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    next_hour = predictions_df['predicted_pm25'].iloc[0]
                    st.metric("Next Hour", f"{next_hour:.1f} µg/m³")
                
                with col2:
                    tomorrow_avg = predictions_df['predicted_pm25'].mean()
                    st.metric("Tomorrow Avg", f"{tomorrow_avg:.1f} µg/m³")
                
                with col3:
                    tomorrow_max = predictions_df['predicted_pm25'].max()
                    max_time = predictions_df.loc[predictions_df['predicted_pm25'].idxmax(), 'datetime']
                    st.metric("Peak Tomorrow", f"{tomorrow_max:.1f} µg/m³", 
                             delta=f"at {max_time.strftime('%H:00')}")
                
                with col4:
                    # AQI category
                    avg_val = predictions_df['predicted_pm25'].mean()
                    if avg_val <= 12:
                        aqi = "Good ✅"
                    elif avg_val <= 35.4:
                        aqi = "Moderate ⚠️"
                    elif avg_val <= 55.4:
                        aqi = "Unhealthy for Sensitive Groups 🚨"
                    elif avg_val <= 150.4:
                        aqi = "Unhealthy 🚨"
                    elif avg_val <= 250.4:
                        aqi = "Very Unhealthy ⛔"
                    else:
                        aqi = "Hazardous ☠️"

                
                # Visualization
                st.markdown("---")
                st.subheader("📈 Tomorrow's PM2.5 Forecast")
                
                fig, ax = plt.subplots(figsize=(14, 6))
                
                # Plot today's last few hours
                today_tail = df_history.tail(12)
                ax.plot(
                    today_tail['datetime'],
                    today_tail['pm25'],
                    label="Today (Historical)",
                    color='blue',
                    linewidth=2,
                    alpha=0.7,
                    marker='o'
                )
                
                # Plot tomorrow's predictions
                ax.plot(
                    predictions_df['datetime'],
                    predictions_df['predicted_pm25'],
                    label=f"Tomorrow's Forecast ({forecast_hours}h)",
                    color='red',
                    linewidth=2.5,
                    marker='s',
                    markersize=6
                )
                
                # Add guidelines
                ax.axhline(y=12, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='Good (12 µg/m³)')
                ax.axhline(y=35.4, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='Moderate (35.4 µg/m³)')
                ax.axhline(y=55.4, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Unhealthy (55.4 µg/m³)')
                
                ax.set_xlabel('Time', fontsize=12, fontweight='bold')
                ax.set_ylabel('PM2.5 (µg/m³)', fontsize=12, fontweight='bold')
                ax.set_title(f"Tomorrow's PM2.5 Forecast - {predictions_df['datetime'].iloc[0].strftime('%B %d, %Y')}", 
                           fontsize=14, fontweight='bold')
                ax.legend(loc='best', fontsize=10)
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                st.pyplot(fig)
                plt.close()
                
                # Hourly breakdown
                st.markdown("---")
                st.subheader("📋 Hourly Breakdown")
                
                display_df = predictions_df.copy()
                display_df['datetime'] = display_df['datetime'].dt.strftime('%Y-%m-%d %H:00')
                display_df['predicted_pm25'] = display_df['predicted_pm25'].round(2)
                
                # Add AQI category
                def get_aqi_category(val):
                    if val <= 12:
                        return "Good ✅"
                    elif val <= 35.4:
                        return "Moderate ⚠️"
                    elif val <= 55.4:
                        return "Unhealthy 🚨"
                    else:
                        return "Very Unhealthy ⛔"
                
                display_df['AQI Category'] = display_df['predicted_pm25'].apply(get_aqi_category)
                display_df = display_df.rename(columns={
                    'datetime': 'Time',
                    'predicted_pm25': 'PM2.5 (µg/m³)'
                })
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Summary statistics
                st.markdown("---")
                st.subheader("📊 Summary Statistics")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Forecast Statistics:**")
                    st.write(f"- Minimum: {predictions_df['predicted_pm25'].min():.2f} µg/m³")
                    st.write(f"- Maximum: {predictions_df['predicted_pm25'].max():.2f} µg/m³")
                    st.write(f"- Average: {predictions_df['predicted_pm25'].mean():.2f} µg/m³")
                    st.write(f"- Standard Deviation: {predictions_df['predicted_pm25'].std():.2f} µg/m³")
                
                with col2:
                    st.markdown("**Air Quality Hours:**")
                    good = (predictions_df['predicted_pm25'] <= 12).sum()
                    moderate = ((predictions_df['predicted_pm25'] > 12) & (predictions_df['predicted_pm25'] <= 35.4)).sum()
                    unhealthy = (predictions_df['predicted_pm25'] > 35.4).sum()
                    
                    st.write(f"- Good (≤12): {good} hours ({good/len(predictions_df)*100:.1f}%)")
                    st.write(f"- Moderate (12-35.4): {moderate} hours ({moderate/len(predictions_df)*100:.1f}%)")
                    st.write(f"- Unhealthy (>35.4): {unhealthy} hours ({unhealthy/len(predictions_df)*100:.1f}%)")
                
                # Download button
                csv = predictions_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Download Tomorrow's Forecast (CSV)",
                    data=csv,
                    file_name=f"tomorrow_pm25_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                st.session_state.df_history = df_history


# ──────────────────────────────────────────────
# Make Predictions
# ──────────────────────────────────────────────
# Update df_history from session state
df_history = st.session_state.df_history

if df_history is not None:
    st.divider()
    
    # Check if we have enough data
    if len(df_history) < 72:
        st.error("❌ Need at least 72 hours of historical data for lag features")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"🔮 Future PM2.5 Predictions ({n_hours} hours ahead)")
        
        with col2:
            use_future_weather = st.checkbox("Use future weather forecast", value=False)
        
        # Future weather inputs
        future_temp = None
        future_humidity = None
        
        if use_future_weather:
            st.subheader("🌤️ Future Weather Forecast")
            col1, col2 = st.columns(2)
            
            with col1:
                temp_input = st.text_input(
                    "Future Temperatures (°C)",
                    value=str(df_history['temperature'].iloc[-1]),
                    help="Comma-separated values for each hour"
                )
                if temp_input:
                    try:
                        future_temp = [float(x.strip()) for x in temp_input.split(',')]
                        if len(future_temp) < n_hours:
                            future_temp = future_temp + [future_temp[-1]] * (n_hours - len(future_temp))
                        future_temp = future_temp[:n_hours]
                    except:
                        st.warning("⚠️ Invalid temperature format")
            
            with col2:
                humidity_input = st.text_input(
                    "Future Humidity (%)",
                    value=str(df_history['relative_humidity'].iloc[-1]),
                    help="Comma-separated values for each hour"
                )
                if humidity_input:
                    try:
                        future_humidity = [float(x.strip()) for x in humidity_input.split(',')]
                        if len(future_humidity) < n_hours:
                            future_humidity = future_humidity + [future_humidity[-1]] * (n_hours - len(future_humidity))
                        future_humidity = future_humidity[:n_hours]
                    except:
                        st.warning("⚠️ Invalid humidity format")
        
        # Make predictions
        if st.button("🚀 Generate Predictions", type="primary"):
            with st.spinner("Computing predictions..."):
                predictions_df = predict_future(df_history, n_hours, future_temp, future_humidity)
            
            if len(predictions_df) > 0:
                st.success(f"✅ Generated {len(predictions_df)} predictions!")
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Next Hour", f"{predictions_df['predicted_pm25'].iloc[0]:.1f} µg/m³")
                with col2:
                    st.metric("Average", f"{predictions_df['predicted_pm25'].mean():.1f} µg/m³")
                with col3:
                    st.metric("Maximum", f"{predictions_df['predicted_pm25'].max():.1f} µg/m³")
                with col4:
                    st.metric("Minimum", f"{predictions_df['predicted_pm25'].min():.1f} µg/m³")
                
                # Visualization
                st.subheader("📈 Prediction Visualization")
                
                fig, ax = plt.subplots(figsize=(14, 6))
                
                # Plot historical data (last 48 hours)
                hist_tail = df_history.tail(48)
                ax.plot(
                    hist_tail['datetime'],
                    hist_tail['pm25'],
                    label='Historical PM2.5',
                    color='blue',
                    linewidth=2,
                    alpha=0.7
                )
                
                # Plot predictions
                ax.plot(
                    predictions_df['datetime'],
                    predictions_df['predicted_pm25'],
                    label='Predicted PM2.5',
                    color='red',
                    linewidth=2,
                    marker='o',
                    markersize=4
                )
                
                # Add WHO guideline
                ax.axhline(y=15, color='green', linestyle='--', alpha=0.5, label='WHO Guideline (15 µg/m³)')
                ax.axhline(y=35, color='orange', linestyle='--', alpha=0.5, label='Moderate (35 µg/m³)')
                ax.axhline(y=55, color='red', linestyle='--', alpha=0.5, label='Unhealthy (55 µg/m³)')
                
                ax.set_xlabel('Datetime', fontsize=12)
                ax.set_ylabel('PM2.5 (µg/m³)', fontsize=12)
                ax.set_title(f'PM2.5 Predictions - Next {n_hours} Hours', fontsize=14, fontweight='bold')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                st.pyplot(fig)
                plt.close()
                
                # Display predictions table
                st.subheader("📋 Detailed Predictions")
                
                display_df = predictions_df.copy()
                display_df['datetime'] = display_df['datetime'].dt.strftime('%Y-%m-%d %H:%M')
                display_df['predicted_pm25'] = display_df['predicted_pm25'].round(2)
                display_df = display_df.rename(columns={
                    'datetime': 'Datetime',
                    'predicted_pm25': 'Predicted PM2.5 (µg/m³)'
                })
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Download button
                csv = predictions_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Download Predictions (CSV)",
                    data=csv,
                    file_name=f"pm25_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Performance analysis (if we have actual values to compare)
                st.subheader("📊 Model Performance Analysis")
                
                # Calculate some statistics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Prediction Statistics:**")
                    st.write(f"- Mean: {predictions_df['predicted_pm25'].mean():.2f} µg/m³")
                    st.write(f"- Std Dev: {predictions_df['predicted_pm25'].std():.2f} µg/m³")
                    st.write(f"- Range: {predictions_df['predicted_pm25'].min():.2f} - {predictions_df['predicted_pm25'].max():.2f} µg/m³")
                
                with col2:
                    st.markdown("**Air Quality Categories:**")
                    good = (predictions_df['predicted_pm25'] <= 12).sum()
                    moderate = ((predictions_df['predicted_pm25'] > 12) & (predictions_df['predicted_pm25'] <= 35.4)).sum()
                    unhealthy = (predictions_df['predicted_pm25'] > 35.4).sum()
                    
                    st.write(f"- Good (≤12): {good} hours ({good/len(predictions_df)*100:.1f}%)")
                    st.write(f"- Moderate (12-35.4): {moderate} hours ({moderate/len(predictions_df)*100:.1f}%)")
                    st.write(f"- Unhealthy (>35.4): {unhealthy} hours ({unhealthy/len(predictions_df)*100:.1f}%)")
            else:
                st.error("❌ Failed to generate predictions")

else:
    st.info("👈 Please select a data input method and provide historical data to make predictions.")

