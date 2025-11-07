"""Utility functions for AQI Predictor"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict
from loguru import logger
import hopsworks
from src.config import config


class AQICNClient:
    """Client for AQICN API"""
    
    def __init__(self):
        self.base_url = config.api.aqicn_base_url
        self.token = config.api.aqicn_token
    
    def get_current_aqi(self, city=None):
        """Get current AQI data"""
        city = city or config.location.city_name
        url = f"{self.base_url}/feed/{city}/?token={self.token}"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] != 'ok':
                raise ValueError(f"API error: {data}")
            
            return data['data']
        except Exception as e:
            logger.error(f"Error fetching AQI: {e}")
            raise


class HopsworksClient:
    """Client for Hopsworks - CORRECTED FOR HOPSWORKS 4.x"""
    
    def __init__(self):
        self.project = None
        self.fs = None
    
    def connect(self):
        """Connect to Hopsworks"""
        try:
            # CORRECTED: Hopsworks 4.x uses hopsworks.login()
            self.project = hopsworks.login(
                api_key_value=config.hopsworks.api_key,
                project=config.hopsworks.project_name
            )
            self.fs = self.project.get_feature_store()
            logger.info(f"Connected to Hopsworks: {config.hopsworks.project_name}")
        except Exception as e:
            logger.error(f"Hopsworks connection error: {e}")
            raise


def parse_aqi_data(raw_data: Dict) -> Dict:
    """Parse raw AQI data"""
    parsed = {
        'timestamp': datetime.now(),
        'aqi': raw_data.get('aqi', np.nan),
        'city': raw_data.get('city', {}).get('name', 'Unknown'),
        'latitude': raw_data.get('city', {}).get('geo', [np.nan])[0] if raw_data.get('city', {}).get('geo') else np.nan,
        'longitude': raw_data.get('city', {}).get('geo', [np.nan, np.nan])[1] if len(raw_data.get('city', {}).get('geo', [])) > 1 else np.nan,
        'dominant_pollutant': raw_data.get('dominentpol', 'unknown')
    }
    
    iaqi = raw_data.get('iaqi', {})
    for pollutant in config.features.pollutants:
        parsed[f'{pollutant}_value'] = iaqi.get(pollutant, {}).get('v', np.nan)
    
    parsed['temperature'] = iaqi.get('t', {}).get('v', np.nan)
    parsed['pressure'] = iaqi.get('p', {}).get('v', np.nan)
    parsed['humidity'] = iaqi.get('h', {}).get('v', np.nan)
    parsed['wind_speed'] = iaqi.get('w', {}).get('v', np.nan)
    
    return parsed


def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer time features"""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    return df


def engineer_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer derived features"""
    df = df.copy()
    df = df.sort_values('timestamp')
    
    df['aqi_change'] = df['aqi'].diff()
    df['rolling_mean_24h'] = df['aqi'].rolling(window=24, min_periods=1).mean()
    df['rolling_std_24h'] = df['aqi'].rolling(window=24, min_periods=1).std()
    
    for lag in [1, 3, 6, 12, 24]:
        df[f'aqi_lag_{lag}h'] = df['aqi'].shift(lag)
    
    return df


def calculate_aqi_category(aqi: float) -> str:
    """Calculate AQI category"""
    if pd.isna(aqi):
        return 'Unknown'
    elif aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Moderate'
    elif aqi <= 150:
        return 'Unhealthy for Sensitive Groups'
    elif aqi <= 200:
        return 'Unhealthy'
    elif aqi <= 300:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'


def get_health_recommendation(aqi: float) -> str:
    """Get health recommendation"""
    category = calculate_aqi_category(aqi)
    
    recommendations = {
        'Good': 'Air quality is satisfactory. Enjoy outdoor activities!',
        'Moderate': 'Air quality is acceptable. Unusually sensitive people should consider limiting prolonged outdoor exertion.',
        'Unhealthy for Sensitive Groups': 'Members of sensitive groups may experience health effects.',
        'Unhealthy': 'Everyone may begin to experience health effects. Limit prolonged outdoor exertion.',
        'Very Unhealthy': 'Health alert! Avoid prolonged outdoor exertion.',
        'Hazardous': 'Health warning! Everyone should avoid all outdoor exertion.',
        'Unknown': 'AQI data not available.'
    }
    
    return recommendations.get(category, recommendations['Unknown'])