"""
Unit tests for data collector
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_collector import AQICNDataCollector
from src.features.feature_engineering import FeatureEngineer
import pandas as pd


class TestAQICNDataCollector:
    """Test suite for AQICN data collector"""
    
    def test_collector_initialization(self):
        """Test that collector initializes correctly"""
        collector = AQICNDataCollector()
        assert collector.api_token is not None
        assert collector.base_url == "https://api.waqi.info"
    
    def test_fetch_current_data(self):
        """Test fetching current AQI data"""
        collector = AQICNDataCollector()
        data = collector.fetch_current_data()
        
        # Check that we got data
        assert data is not None
        
        # Check required fields
        assert "timestamp" in data
        assert "aqi" in data
        assert "city" in data
    
    def test_fetch_historical_data(self):
        """Test fetching historical data"""
        collector = AQICNDataCollector()
        df = collector.fetch_historical_data(days=7)
        
        # Check data frame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "timestamp" in df.columns
        assert "aqi" in df.columns
    
    def test_station_info(self):
        """Test getting station information"""
        collector = AQICNDataCollector()
        info = collector.get_station_info()
        
        if info:  # Only test if API returns data
            assert "city" in info or "station_name" in info


class TestFeatureEngineer:
    """Test suite for feature engineering"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        collector = AQICNDataCollector()
        return collector.fetch_historical_data(days=2)
    
    def test_time_features(self, sample_data):
        """Test time feature creation"""
        engineer = FeatureEngineer()
        df = engineer.create_time_features(sample_data)
        
        # Check new columns
        assert "hour" in df.columns
        assert "day_of_week" in df.columns
        assert "month" in df.columns
        assert "is_weekend" in df.columns
        assert "hour_sin" in df.columns
        assert "hour_cos" in df.columns
    
    def test_lag_features(self, sample_data):
        """Test lag feature creation"""
        engineer = FeatureEngineer()
        df = engineer.create_time_features(sample_data)
        df = engineer.create_lag_features(df, columns=["aqi"], lags=[1, 3, 6])
        
        # Check lag columns
        assert "aqi_lag_1h" in df.columns
        assert "aqi_lag_3h" in df.columns
        assert "aqi_lag_6h" in df.columns
    
    def test_full_pipeline(self, sample_data):
        """Test complete feature engineering pipeline"""
        engineer = FeatureEngineer()
        df = engineer.engineer_features(sample_data, create_targets=True)
        
        # Check output
        assert len(df) > 0
        assert "aqi_target_24h" in df.columns
        assert "aqi_target_48h" in df.columns
        assert "aqi_target_72h" in df.columns
        
        # Check feature names are stored
        feature_names = engineer.get_feature_names()
        assert len(feature_names) > 0


def test_environment_variables():
    """Test that required environment variables are set"""
    from src.config import AQICN_API_TOKEN, CITY_NAME
    
    assert AQICN_API_TOKEN is not None, "AQICN_API_TOKEN not set"
    assert CITY_NAME is not None, "CITY_NAME not set"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])