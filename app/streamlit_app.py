"""
Streamlit Web Dashboard for AQI Predictions
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

from src.data_collector import AQICNDataCollector
from src.feature_engineering import FeatureEngineer
from src.models.model_trainer import AQIModelTrainer
from src.config import MODELS_DIR, CITY_NAME, AQI_BREAKPOINTS, ALERT_THRESHOLD

# Page configuration
st.set_page_config(
    page_title="AQI Predictor - Karachi",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:50px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)


def get_aqi_category(aqi_value):
    """Get AQI category and color"""
    for min_val, max_val, category in AQI_BREAKPOINTS:
        if min_val <= aqi_value <= max_val:
            colors = {
                "Good": "#00E400",
                "Moderate": "#FFFF00",
                "Unhealthy for Sensitive Groups": "#FF7E00",
                "Unhealthy": "#FF0000",
                "Very Unhealthy": "#8F3F97",
                "Hazardous": "#7E0023"
            }
            return category, colors.get(category, "#808080")
    return "Unknown", "#808080"


@st.cache_resource
def load_model(model_type="random_forest"):
    """Load trained model"""
    try:
        # Find latest model directory
        model_dirs = list(MODELS_DIR.glob(f"{model_type}_v*"))
        if not model_dirs:
            return None
        
        latest_model_dir = sorted(model_dirs)[-1]
        
        trainer = AQIModelTrainer(model_type=model_type)
        trainer.load_models(str(latest_model_dir))
        
        return trainer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_current_aqi():
    """Fetch current AQI data"""
    try:
        collector = AQICNDataCollector()
        return collector.fetch_current_data()
    except Exception as e:
        st.error(f"Error fetching AQI data: {e}")
        return None


def create_gauge_chart(aqi_value, title="Current AQI"):
    """Create a gauge chart for AQI"""
    category, color = get_aqi_category(aqi_value)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=aqi_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 500], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#00E400'},
                {'range': [50, 100], 'color': '#FFFF00'},
                {'range': [100, 150], 'color': '#FF7E00'},
                {'range': [150, 200], 'color': '#FF0000'},
                {'range': [200, 300], 'color': '#8F3F97'},
                {'range': [300, 500], 'color': '#7E0023'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': ALERT_THRESHOLD
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig


def create_forecast_chart(predictions):
    """Create forecast line chart"""
    fig = go.Figure()
    
    # Add predicted values
    fig.add_trace(go.Scatter(
        x=predictions['date'],
        y=predictions['aqi'],
        mode='lines+markers',
        name='Predicted AQI',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=10)
    ))
    
    # Add AQI threshold bands
    for min_val, max_val, category in AQI_BREAKPOINTS[:4]:  # Show first 4 categories
        fig.add_hrect(
            y0=min_val, y1=max_val,
            fillcolor=get_aqi_category((min_val + max_val) / 2)[1],
            opacity=0.1,
            layer="below",
            line_width=0,
        )
    
    fig.update_layout(
        title="3-Day AQI Forecast",
        xaxis_title="Date",
        yaxis_title="AQI Value",
        height=400,
        hovermode='x unified',
        showlegend=True
    )
    
    return fig


def create_pollutant_chart(current_data):
    """Create pollutant comparison chart"""
    pollutants = {
        'PM2.5': current_data.get('pm25', 0),
        'PM10': current_data.get('pm10', 0),
        'O₃': current_data.get('o3', 0),
        'NO₂': current_data.get('no2', 0),
        'SO₂': current_data.get('so2', 0),
        'CO': current_data.get('co', 0)
    }
    
    # Filter out None values
    pollutants = {k: v for k, v in pollutants.items() if v is not None}
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(pollutants.keys()),
            y=list(pollutants.values()),
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
        )
    ])
    
    fig.update_layout(
        title="Current Pollutant Levels",
        xaxis_title="Pollutant",
        yaxis_title="AQI Value",
        height=350
    )
    
    return fig


def main():
    # Header
    st.title(f"🌍 Air Quality Index Predictor - {CITY_NAME}")
    st.markdown("### Real-time AQI monitoring and 3-day forecast powered by Machine Learning")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        model_type = st.selectbox(
            "Select Model",
            ["random_forest", "ridge", "neural_network"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        This dashboard provides:
        - ✅ Real-time AQI monitoring
        - ✅ 3-day AQI predictions
        - ✅ Pollutant analysis
        - ✅ Health recommendations
        """)
        
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Load model
    with st.spinner("Loading ML model..."):
        trainer = load_model(model_type)
    
    if trainer is None:
        st.error("⚠️ No trained model found! Please run training pipeline first.")
        st.code("python src/pipelines/training_pipeline.py --model random_forest")
        return
    
    # Fetch current data
    with st.spinner("Fetching latest AQI data..."):
        current_data = fetch_current_aqi()
    
    if current_data is None:
        st.error("⚠️ Failed to fetch current AQI data. Please check your API connection.")
        return
    
    current_aqi = current_data.get('aqi', 0)
    category, color = get_aqi_category(current_aqi)
    
    # Current AQI Section
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.plotly_chart(create_gauge_chart(current_aqi), use_container_width=True)
    
    with col2:
        st.metric("AQI Category", category)
        st.metric("PM2.5", f"{current_data.get('pm25', 'N/A')}")
        st.metric("Temperature", f"{current_data.get('temp', 'N/A')}°C")
    
    with col3:
        st.metric("Humidity", f"{current_data.get('humidity', 'N/A')}%")
        st.metric("Pressure", f"{current_data.get('pressure', 'N/A')} hPa")
        st.metric("Wind Speed", f"{current_data.get('wind_speed', 'N/A')} m/s")
    
    # Alert if unhealthy
    if current_aqi >= ALERT_THRESHOLD:
        st.warning(f"⚠️ **ALERT**: Air quality is {category}! Limit outdoor activities.")
    
    # Generate predictions
    st.markdown("---")
    st.header("📈 3-Day Forecast")
    
    with st.spinner("Generating predictions..."):
        # Prepare features
        df_raw = pd.DataFrame([current_data])
        engineer = FeatureEngineer()
        df_features = engineer.engineer_features(df_raw, create_targets=False)
        
        # Remove excluded columns
        exclude_cols = ["timestamp", "city"]
        feature_cols = [col for col in df_features.columns if col not in exclude_cols]
        X = df_features[feature_cols]
        
        # Scale features
        X_scaled = trainer.scaler.transform(X)
        
        # Make predictions for each horizon
        predictions = []
        for i, horizon in enumerate(['24h', '48h', '72h'], 1):
            model = trainer.models.get(horizon)
            if model:
                pred = model.predict(X_scaled)[0]
                if isinstance(pred, np.ndarray):
                    pred = pred[0]
                
                pred_date = datetime.now() + timedelta(days=i)
                predictions.append({
                    'date': pred_date.strftime('%Y-%m-%d'),
                    'day': f"Day {i}",
                    'aqi': float(pred),
                    'category': get_aqi_category(pred)[0]
                })
    
    if predictions:
        # Forecast chart
        st.plotly_chart(create_forecast_chart(pd.DataFrame(predictions)), use_container_width=True)
        
        # Predictions table
        col1, col2, col3 = st.columns(3)
        
        for i, (col, pred) in enumerate(zip([col1, col2, col3], predictions)):
            with col:
                pred_category, pred_color = get_aqi_category(pred['aqi'])
                st.markdown(f"""
                <div style="background-color: {pred_color}33; padding: 20px; border-radius: 10px; border-left: 5px solid {pred_color}">
                    <h3>{pred['day']}</h3>
                    <p style="font-size: 14px; color: #666;">{pred['date']}</p>
                    <p style="font-size: 36px; font-weight: bold; margin: 10px 0;">{pred['aqi']:.0f}</p>
                    <p style="font-size: 16px; font-weight: bold;">{pred_category}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Pollutant Analysis
    st.markdown("---")
    st.header("🔬 Pollutant Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.plotly_chart(create_pollutant_chart(current_data), use_container_width=True)
    
    with col2:
        st.markdown("### Health Recommendations")
        if current_aqi <= 50:
            st.success("✅ Air quality is good. Enjoy outdoor activities!")
        elif current_aqi <= 100:
            st.info("ℹ️ Air quality is acceptable. Unusually sensitive people should consider limiting prolonged outdoor exertion.")
        elif current_aqi <= 150:
            st.warning("⚠️ Members of sensitive groups may experience health effects. Consider reducing prolonged outdoor exertion.")
        elif current_aqi <= 200:
            st.warning("⚠️ Everyone may begin to experience health effects. Avoid prolonged outdoor exertion.")
        elif current_aqi <= 300:
            st.error("🚨 Health alert! Everyone may experience more serious health effects. Avoid outdoor activities.")
        else:
            st.error("🚨 Health warning! Emergency conditions. Everyone should avoid outdoor activities.")
    
    # Model Performance
    st.markdown("---")
    st.header("📊 Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    metrics_data = trainer.metrics
    for i, (col, horizon) in enumerate(zip([col1, col2, col3], ['24h', '48h', '72h'])):
        with col:
            if horizon in metrics_data:
                metrics = metrics_data[horizon]
                st.metric(f"{horizon} Forecast RMSE", f"{metrics['rmse']:.2f}")
                st.metric(f"MAE", f"{metrics['mae']:.2f}")
                st.metric(f"R² Score", f"{metrics['r2']:.4f}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>Data source: <a href="https://aqicn.org" target="_blank">World Air Quality Index Project</a></p>
        <p>Last updated: {} | Model: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), model_type), unsafe_allow_html=True)


if __name__ == "__main__":
    main()