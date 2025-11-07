"""Configuration management for AQI Predictor"""
import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()


@dataclass
class APIConfig:
    aqicn_token: str = os.getenv('AQICN_API_TOKEN', '')
    aqicn_base_url: str = 'https://api.waqi.info'


@dataclass
class LocationConfig:
    city_name: str = os.getenv('CITY_NAME', 'Karachi')
    latitude: float = float(os.getenv('CITY_LAT', '24.8607'))
    longitude: float = float(os.getenv('CITY_LON', '67.0011'))


@dataclass
class HopsworksConfig:
    api_key: str = os.getenv('HOPSWORKS_API_KEY', '')
    project_name: str = os.getenv('HOPSWORKS_PROJECT_NAME', 'aqi_predictor')
    feature_group_name: str = 'aqi_features'
    feature_group_version: int = 1
    feature_view_name: str = 'aqi_feature_view'
    feature_view_version: int = 1
    model_name: str = 'aqi_predictor_model'


@dataclass
class FeatureConfig:
    pollutants: list = None
    
    def __post_init__(self):
        if self.pollutants is None:
            self.pollutants = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']


@dataclass
class ModelConfig:
    test_size: float = 0.2
    random_state: int = 42
    
    random_forest_params: dict = None
    ridge_params: dict = None
    neural_network_params: dict = None
    
    def __post_init__(self):
        if self.random_forest_params is None:
            self.random_forest_params = {
                'n_estimators': 200,
                'max_depth': 20,
                'min_samples_split': 5,
                'random_state': self.random_state,
                'n_jobs': -1
            }
        if self.ridge_params is None:
            self.ridge_params = {
                'alpha': 1.0,
                'random_state': self.random_state
            }
        if self.neural_network_params is None:
            self.neural_network_params = {
                'hidden_layers': [128, 64, 32],
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'epochs': 100,
                'batch_size': 32
            }


@dataclass
class PipelineConfig:
    forecast_days: int = 3


class Config:
    def __init__(self):
        self.api = APIConfig()
        self.location = LocationConfig()
        self.hopsworks = HopsworksConfig()
        self.features = FeatureConfig()
        self.model = ModelConfig()
        self.pipeline = PipelineConfig()


config = Config()