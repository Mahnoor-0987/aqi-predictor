"""
Unit tests for Feature Pipeline
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_pipeline import FeaturePipeline
from src.utils import (
    AQICNClient, parse_aqi_data, engineer_time_features,
    engineer_derived_features, calculate_aqi_category
)


class TestAQICNClient:
    """Test AQICN API client"""
    
    def test_client_initialization(self):
        """Test client can be initialized"""
        client = AQICNClient()
        assert client.base_url is not None
        assert client.token is not None
    
    def test_get_current_aqi(self):
        """Test fetching current AQI data"""
        client = AQICNClient()
        try:
            data = client.get_current_aqi()
            assert 'aqi' in data
            assert isinstance(data['aqi'], (int, float))
        except Exception as e:
            pytest.skip(f"API call failed: {e}")
    
    def test_get_aqi_by_coords(self):
        """Test fetching AQI by coordinates"""
        client = AQICNClient()
        try:
            data = client.get_aqi_by_coords(24.8607, 67.0011)  # Karachi
            assert 'aqi' in data
        except Exception as e:
            pytest.skip(f"API call failed: {e}")


class TestDataParsing:
    """Test data parsing functions"""
    
    def test_parse_aqi_data(self):
        """Test parsing raw AQI data"""
        raw_data = {
            'aqi': 85,
            'city': {'name': 'Test City', 'geo': [24.8607, 67.0011]},
            'dominentpol': 'pm25',
            'iaqi': {
                'pm25': {'v': 35},
                'pm10': {'v': 50},
                't': {'v': 25},
                'h': {'v': 60}
            }
        }
        
        parsed = parse_aqi_data(raw_data)
        
        assert parsed['aqi'] == 85
        assert parsed['city'] == 'Test City'
        assert parsed['pm25_value'] == 35
        assert parsed['pm10_value'] == 50
        assert parsed['temperature'] == 25
        assert parsed['humidity'] == 60
    
    def test_parse_aqi_data_missing_fields(self):
        """Test parsing with missing fields"""
        raw_data = {
            'aqi': 85,
            'city': {'name': 'Test City'},
            'iaqi': {}
        }
        
        parsed = parse_aqi_data(raw_data)
        
        assert parsed['aqi'] == 85
        assert pd.isna(parsed['pm25_value'])


class TestFeatureEngineering:
    """Test feature engineering functions"""
    
    def test_engineer_time_features(self):
        """Test time-based feature engineering"""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=24, freq='H'),
            'aqi': np.random.randint(50, 150, 24)
        })
        
        result = engineer_time_features(df)
        
        assert 'hour' in result.columns
        assert 'day' in result.columns
        assert 'month' in result.columns
        assert 'day_of_week' in result.columns
        assert 'is_weekend' in result.columns
        assert 'hour_sin' in result.columns
        assert 'hour_cos' in result.columns
        
        # Verify hour range
        assert result['hour'].min() >= 0
        assert result['hour'].max() <= 23
    
    def test_engineer_derived_features(self):
        """Test derived feature engineering"""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=48, freq='H'),
            'aqi': np.random.randint(50, 150, 48)
        })
        
        result = engineer_derived_features(df)
        
        assert 'aqi_change' in result.columns
        assert 'aqi_change_rate' in result.columns
        assert 'rolling_mean_24h' in result.columns
        assert 'rolling_std_24h' in result.columns
        assert 'aqi_lag_1h' in result.columns
        assert 'aqi_lag_24h' in result.columns


class TestAQICategory:
    """Test AQI categorization"""
    
    def test_calculate_aqi_category_good(self):
        """Test Good category"""
        assert calculate_aqi_category(25) == 'Good'
        assert calculate_aqi_category(50) == 'Good'
    
    def test_calculate_aqi_category_moderate(self):
        """Test Moderate category"""
        assert calculate_aqi_category(75) == 'Moderate'
        assert calculate_aqi_category(100) == 'Moderate'
    
    def test_calculate_aqi_category_unhealthy_sensitive(self):
        """Test Unhealthy for Sensitive Groups"""
        assert calculate_aqi_category(125) == 'Unhealthy for Sensitive Groups'
        assert calculate_aqi_category(150) == 'Unhealthy for Sensitive Groups'
    
    def test_calculate_aqi_category_unhealthy(self):
        """Test Unhealthy category"""
        assert calculate_aqi_category(175) == 'Unhealthy'
        assert calculate_aqi_category(200) == 'Unhealthy'
    
    def test_calculate_aqi_category_very_unhealthy(self):
        """Test Very Unhealthy category"""
        assert calculate_aqi_category(250) == 'Very Unhealthy'
        assert calculate_aqi_category(300) == 'Very Unhealthy'
    
    def test_calculate_aqi_category_hazardous(self):
        """Test Hazardous category"""
        assert calculate_aqi_category(350) == 'Hazardous'
        assert calculate_aqi_category(500) == 'Hazardous'
    
    def test_calculate_aqi_category_nan(self):
        """Test NaN handling"""
        assert calculate_aqi_category(np.nan) == 'Unknown'


class TestFeaturePipeline:
    """Test Feature Pipeline"""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance"""
        return FeaturePipeline()
    
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline can be initialized"""
        assert pipeline.aqi_client is not None
        assert pipeline.hops_client is not None
    
    def test_fetch_current_data(self, pipeline):
        """Test fetching current data"""
        try:
            df = pipeline.fetch_current_data()
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert 'aqi' in df.columns
            assert 'timestamp' in df.columns
        except Exception as e:
            pytest.skip(f"API call failed: {e}")
    
    def test_fetch_historical_data(self, pipeline):
        """Test historical data generation"""
        df = pipeline.fetch_historical_data(days=7)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 7 * 24  # 7 days * 24 hours
        assert 'aqi' in df.columns
        assert 'timestamp' in df.columns
    
    def test_engineer_features(self, pipeline):
        """Test feature engineering"""
        # Create sample data
        df = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=48, freq='H'),
            'aqi': np.random.randint(50, 150, 48),
            'pm25_value': np.random.randint(20, 100, 48),
            'temperature': np.random.randint(15, 35, 48)
        })
        
        result = pipeline.engineer_features(df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
        # Check time features
        assert 'hour' in result.columns
        assert 'day' in result.columns
        # Check derived features
        assert 'rolling_mean_24h' in result.columns


if __name__ == "__main__":
    pytest.main([__file__, '-v'])