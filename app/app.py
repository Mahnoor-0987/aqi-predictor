"""Streamlit Dashboard for AQI Predictions (Enhanced & Humanized UI)"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference_pipeline import InferencePipeline
from src.feature_pipeline import FeaturePipeline
from src.utils import calculate_aqi_category, get_health_recommendation
from src.config import config

# --- Page Config ---
st.set_page_config(
    page_title="AQI Predictor 🌍",
    page_icon="🌍",
    layout="wide"
)

# --- CSS for UI ---
st.markdown("""
<style>
/* Page background gradient */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #e0f7fa, #f1f8e9);
}

/* Header */
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    color: #1f77b4;
    margin-bottom: 1rem;
}

/* AQI metric card */
.aqi-card {
    padding: 15px;
    border-radius: 15px;
    margin: 8px 0;
    text-align: center;
    font-weight: bold;
    color: white;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}

/* Standard metric card */
.metric-card {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 10px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}

/* Hazard alert */
.hazard-alert {
    background-color: #ff4d4d;
    color: white;
    padding: 15px;
    border-radius: 12px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 20px;
}

/* Footer */
.footer {
    text-align: center;
    color: #555;
    padding: 15px;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# --- Cached Functions ---
@st.cache_data(ttl=3600)
def get_current_aqi():
    try:
        fp = FeaturePipeline()
        fp.initialize()
        df = fp.fetch_current_data()
        return df.iloc[0]['aqi']
    except:
        return None

@st.cache_data(ttl=3600)
def get_predictions():
    try:
        pipeline = InferencePipeline()
        result = pipeline.run()
        return result
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# --- AQI color helper ---
def get_aqi_color(aqi):
    if aqi <= 50: return '#00e400'
    elif aqi <= 100: return '#ffff00'
    elif aqi <= 150: return '#ff7e00'
    elif aqi <= 200: return '#ff0000'
    elif aqi <= 300: return '#8f3f97'
    else: return '#7e0023'

# --- Charts ---
def plot_forecast(predictions_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=predictions_df['timestamp'],
        y=predictions_df['predicted_aqi'],
        mode='lines+markers',
        name='Predicted AQI',
        line=dict(color='#1f77b4', width=3)
    ))
    # AQI bands
    bands = [(0,50,'#00e400'), (50,100,'#ffff00'), (100,150,'#ff7e00'), (150,200,'#ff0000')]
    for y0, y1, color in bands:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, opacity=0.1, line_width=0)
    fig.update_layout(title="3-Day AQI Forecast", xaxis_title="Time", yaxis_title="AQI", hovermode='x unified', height=400)
    return fig

def plot_daily_avg(daily_avg):
    fig = go.Figure()
    colors = [get_aqi_color(aqi) for aqi in daily_avg['predicted_aqi']]
    fig.add_trace(go.Bar(
        x=daily_avg['date'].astype(str),
        y=daily_avg['predicted_aqi'],
        marker_color=colors,
        text=daily_avg['predicted_aqi'].round(0),
        textposition='auto'
    ))
    fig.update_layout(title="Daily Average AQI", xaxis_title="Date", yaxis_title="Average AQI", height=300)
    return fig

# --- Main App ---
def main():
    st.markdown('<h1 class="main-header">🌍 AQI Predictor Dashboard</h1>', unsafe_allow_html=True)
    st.markdown(f"**Location:** {config.location.city_name} | **Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Settings")
        st.subheader("📍 Location")
        st.write(f"**City:** {config.location.city_name}")
        st.write(f"**Coordinates:** {config.location.latitude:.2f}, {config.location.longitude:.2f}")
        st.divider()
        if st.button("🔄 Refresh Predictions", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.divider()
        st.subheader("ℹ️ About")
        st.info("Real-time AQI predictions using ML models, Hopsworks Feature Store, and AQICN API.")
        st.divider()
        st.subheader("📊 AQI Categories")
        st.markdown("""
        🟢 0-50: Good  <br>
        🟡 51-100: Moderate  <br>
        🟠 101-150: Unhealthy for Sensitive  <br>
        🔴 151-200: Unhealthy  <br>
        🟣 201-300: Very Unhealthy  <br>
        🟤 301+: Hazardous
        """, unsafe_allow_html=True)

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "🎯 Current AQI", "🔍 Analysis", "📋 Data"])
    with st.spinner("Loading predictions..."):
        result = get_predictions()
    if result is None:
        st.error("⚠️ Unable to load predictions")
        return

    predictions_df = result['predictions']
    daily_avg = result['daily_average']
    hazards = result['hazards']

    # --- Forecast Tab ---
    with tab1:
        st.header("📈 3-Day AQI Forecast")
        if hazards:
            st.markdown(f"<div class='hazard-alert'>⚠️ HAZARD ALERT: {len(hazards)} unhealthy periods detected!</div>", unsafe_allow_html=True)
            for hazard in hazards[:3]:
                st.markdown(f"🔴 {hazard['timestamp'].strftime('%Y-%m-%d %H:%M')}: AQI {hazard['aqi']:.0f} - {hazard['category']}")
        st.plotly_chart(plot_forecast(predictions_df), use_container_width=True)
        col1, col2 = st.columns([2,1])
        with col1:
            st.plotly_chart(plot_daily_avg(daily_avg), use_container_width=True)
        with col2:
            st.subheader("📅 Daily Summary")
            for _, row in daily_avg.iterrows():
                color = get_aqi_color(row['predicted_aqi'])
                st.markdown(f"<div class='aqi-card' style='background-color:{color}; color: {'white' if row['predicted_aqi']>100 else 'black'};'>{row['date']}<br>AQI: {row['predicted_aqi']:.0f}<br>{row['category']}</div>", unsafe_allow_html=True)

    # --- Current AQI Tab ---
    with tab2:
        st.header("🎯 Current Air Quality")
        current_aqi = get_current_aqi()
        if current_aqi:
            category = calculate_aqi_category(current_aqi)
            recommendation = get_health_recommendation(current_aqi)
            color = get_aqi_color(current_aqi)
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Current AQI", f"{current_aqi:.0f}")
            with col2: st.markdown(f"<div class='aqi-card' style='background-color:{color}; color: {'white' if current_aqi>100 else 'black'}; font-size:1.5rem;'>{category}</div>", unsafe_allow_html=True)
            with col3:
                next_24h_avg = predictions_df.head(24)['predicted_aqi'].mean()
                st.metric("24h Forecast", f"{next_24h_avg:.0f}", f"{next_24h_avg-current_aqi:+.0f}")
            st.subheader("💡 Health Recommendations")
            st.info(recommendation)
        else:
            st.warning("Unable to fetch current AQI")

    # --- Analysis Tab ---
    with tab3:
        st.header("🔍 Advanced Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Stats")
            stats_df = pd.DataFrame({
                'Metric':['Mean AQI','Max AQI','Min AQI','Std Dev'],
                'Value':[f"{predictions_df['predicted_aqi'].mean():.2f}", f"{predictions_df['predicted_aqi'].max():.2f}", f"{predictions_df['predicted_aqi'].min():.2f}", f"{predictions_df['predicted_aqi'].std():.2f}"]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            st.subheader("📈 Distribution")
            fig = px.histogram(predictions_df, x='predicted_aqi', nbins=30)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("📊 Forecast by Category")
            category_counts = predictions_df['category'].value_counts()
            fig = px.pie(values=category_counts.values, names=category_counts.index)
            st.plotly_chart(fig, use_container_width=True)

    # --- Data Tab ---
    with tab4:
        st.header("📋 Raw Data & Downloads")
        st.subheader("Predictions")
        st.dataframe(predictions_df[['timestamp','predicted_aqi','category','health_recommendation']], use_container_width=True)
        csv = predictions_df.to_csv(index=False)
        st.download_button("📥 Download CSV", data=csv, file_name=f"aqi_predictions_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")

    st.markdown("<div class='footer'>Built with ❤️ using Streamlit, Hopsworks & AQICN API | Data updates hourly</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
