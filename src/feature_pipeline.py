"""
Feature Pipeline: Fetch, process, and store AQI features
"""
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger
from src.config import config
from src.utils import (
    AQICNClient, HopsworksClient, parse_aqi_data,
    engineer_time_features, engineer_derived_features
)


class FeaturePipeline:
    """Feature pipeline for AQI data"""
    
    def __init__(self):
        self.aqi_client = AQICNClient()
        self.hops_client = HopsworksClient()
        self.feature_group = None
    
    def initialize(self):
        """Initialize Hopsworks connection and feature group"""
        logger.info("Initializing feature pipeline...")
        self.hops_client.connect()
        
        try:
            self.feature_group = self.hops_client.fs.get_or_create_feature_group(
                name=config.hopsworks.feature_group_name,
                version=config.hopsworks.feature_group_version,
                description="AQI features for prediction",
                primary_key=['timestamp'],
                event_time='timestamp',
                online_enabled=False
            )
            logger.info("Feature group ready")
        except Exception as e:
            logger.info(f"Feature group exists, retrieving: {e}")
            self.feature_group = self.hops_client.fs.get_feature_group(
                name=config.hopsworks.feature_group_name,
                version=config.hopsworks.feature_group_version
            )
    
    def fetch_current_data(self) -> pd.DataFrame:
        """Fetch current AQI data"""
        logger.info(f"Fetching AQI for {config.location.city_name}...")
        try:
            raw_data = self.aqi_client.get_current_aqi()
            parsed_data = parse_aqi_data(raw_data)
            df = pd.DataFrame([parsed_data])
            logger.info(f"Fetched: AQI={parsed_data['aqi']}")
            return df
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise
    
    def fetch_historical_data(self, days: int = 30) -> pd.DataFrame:
        """Generate historical data (synthetic for free API)"""
        logger.info(f"Generating {days} days of historical data...")
        
        try:
            current_df = self.fetch_current_data()
            base_aqi = current_df['aqi'].values[0]
        except:
            logger.warning("Using default baseline")
            base_aqi = 85
            current_df = pd.DataFrame([{
                'timestamp': datetime.now(),
                'aqi': base_aqi,
                'city': config.location.city_name,
                'latitude': config.location.latitude,
                'longitude': config.location.longitude,
                'dominant_pollutant': 'pm25',
                'pm25_value': 35, 'pm10_value': 50,
                'o3_value': 40, 'no2_value': 30,
                'so2_value': 10, 'co_value': 500,
                'temperature': 25, 'pressure': 1013,
                'humidity': 60, 'wind_speed': 10
            }])
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        timestamps = pd.date_range(start=start_time, end=end_time, freq='H')
        
        historical_data = []
        np.random.seed(42)
        
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            dow = ts.dayofweek
            
            daily = 15 * np.sin(2 * np.pi * (hour - 6) / 24)
            if hour in [7, 8, 9, 17, 18, 19]:
                daily += 10
            weekly = -5 if dow in [5, 6] else 0
            random_var = np.random.normal(0, 8)
            trend = 5 * np.sin(2 * np.pi * i / (24 * 7))
            
            aqi = base_aqi + daily + weekly + random_var + trend
            aqi = max(20, min(300, aqi))
            
            record = current_df.iloc[0].to_dict()
            record['timestamp'] = ts
            record['aqi'] = aqi
            
            aqi_factor = aqi / base_aqi
            for pollutant in config.features.pollutants:
                col = f'{pollutant}_value'
                if col in record and not pd.isna(record[col]):
                    base_val = record[col]
                    record[col] = max(0, base_val * aqi_factor + np.random.normal(0, base_val * 0.15))
            
            historical_data.append(record)
        
        df = pd.DataFrame(historical_data)
        logger.info(f"Generated {len(df)} records. AQI: {df['aqi'].min():.1f}-{df['aqi'].max():.1f}")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer all features"""
        logger.info("Engineering features...")
        df = engineer_time_features(df)
        df = engineer_derived_features(df)
        df = df.ffill().bfill().fillna(0)
        logger.info(f"Engineered {df.shape[1]} features")
        return df
    
    def store_features(self, df: pd.DataFrame):
        """Store features in Hopsworks with type compatibility"""
        logger.info("Storing features...")
        try:
            # Explicitly cast boolean/int columns to int
            int_columns = ['is_weekend']
            for col in int_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(0).astype(int)
    
            # Cast all other numeric columns to float
            float_columns = df.select_dtypes(include=['int64', 'float64']).columns.difference(int_columns)
            for col in float_columns:
                df[col] = df[col].astype(float)
    
            self.feature_group.insert(df)
            logger.info(f"Stored {len(df)} records")
        except Exception as e:
            logger.error(f"Storage error: {e}")
            raise

    
    def run(self, backfill: bool = False, backfill_days: int = 30):
        """Run the full pipeline"""
        try:
            self.initialize()
            
            if backfill:
                logger.info(f"Backfilling {backfill_days} days...")
                df = self.fetch_historical_data(days=backfill_days)
            else:
                logger.info("Fetching current data...")
                df = self.fetch_current_data()
            
            df = self.engineer_features(df)
            self.store_features(df)
            
            logger.info("✅ Feature pipeline completed")
            return df
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='AQI Feature Pipeline')
    parser.add_argument('--backfill', action='store_true', help='Backfill historical data')
    parser.add_argument('--days', type=int, default=30, help='Days to backfill')
    args = parser.parse_args()
    
    import os
    os.makedirs('logs', exist_ok=True)
    logger.add("logs/feature_pipeline_{time}.log", rotation="1 day")
    
    pipeline = FeaturePipeline()
    pipeline.run(backfill=args.backfill, backfill_days=args.days)


if __name__ == "__main__":
    main()
