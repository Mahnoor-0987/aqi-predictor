"""Inference Pipeline: Generate predictions"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
import joblib
import os
from src.config import config
from src.utils import HopsworksClient, calculate_aqi_category, get_health_recommendation


class InferencePipeline:
    """Inference pipeline for predictions"""
    
    def __init__(self):
        self.hops_client = HopsworksClient()
        self.model = None
        self.scaler = None
    
    def initialize(self):
        """Initialize"""
        logger.info("Initializing inference...")
        self.hops_client.connect()
        self.load_model()
    
    def load_model(self):
        """Load trained model"""
        logger.info("Loading model...")
        
        if os.path.exists('models/best_model.pkl'):
            self.model = joblib.load('models/best_model.pkl')
            logger.info("Loaded sklearn model")
        elif os.path.exists('models/best_model.h5'):
            from tensorflow import keras
            self.model = keras.models.load_model('models/best_model.h5')
            self.scaler = joblib.load('models/scaler.pkl')
            logger.info("Loaded TensorFlow model")
        else:
            raise FileNotFoundError("No model found in models/")
    
    def get_latest_features(self):
        """Get latest features"""
        logger.info("Fetching latest features...")
        
        fg = self.hops_client.fs.get_feature_group(
            name=config.hopsworks.feature_group_name,
            version=config.hopsworks.feature_group_version
        )
        
        df = fg.read()
        df = df.sort_values('timestamp', ascending=False)
        latest = df.head(24)
        
        logger.info(f"Retrieved {len(latest)} records")
        return latest
    
    def prepare_forecast_features(self, latest_data, forecast_hours=72):
        """Prepare forecast features"""
        logger.info(f"Preparing {forecast_hours}h forecast...")
        
        base_record = latest_data.iloc[0].to_dict()
        start_time = pd.to_datetime(base_record['timestamp']) + timedelta(hours=1)
        future_times = pd.date_range(start=start_time, periods=forecast_hours, freq='H')
        
        forecast_data = []
        for ts in future_times:
            record = base_record.copy()
            record['timestamp'] = ts
            record['hour'] = ts.hour
            record['day'] = ts.day
            record['month'] = ts.month
            record['day_of_week'] = ts.dayofweek
            record['is_weekend'] = int(ts.dayofweek in [5, 6])
            record['hour_sin'] = np.sin(2 * np.pi * ts.hour / 24)
            record['hour_cos'] = np.cos(2 * np.pi * ts.hour / 24)
            forecast_data.append(record)
        
        forecast_df = pd.DataFrame(forecast_data)
        exclude_cols = ['timestamp', 'city', 'latitude', 'longitude', 'dominant_pollutant', 'aqi']
        feature_cols = [col for col in forecast_df.columns if col not in exclude_cols]
        
        X = forecast_df[feature_cols].fillna(0)
        return X, forecast_df[['timestamp']]
    
    def predict(self, X):
        """Make predictions"""
        logger.info("Generating predictions...")
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled, verbose=0).flatten()
        else:
            predictions = self.model.predict(X)
        
        logger.info(f"Generated {len(predictions)} predictions")
        return predictions
    
    def format_predictions(self, timestamps, predictions):
        """Format predictions"""
        results = pd.DataFrame({
            'timestamp': timestamps['timestamp'],
            'predicted_aqi': predictions,
            'category': [calculate_aqi_category(aqi) for aqi in predictions],
            'health_recommendation': [get_health_recommendation(aqi) for aqi in predictions]
        })
        
        results['date'] = results['timestamp'].dt.date
        daily_avg = results.groupby('date').agg({'predicted_aqi': 'mean'}).reset_index()
        daily_avg['category'] = daily_avg['predicted_aqi'].apply(calculate_aqi_category)
        
        return results, daily_avg
    
    def check_hazards(self, predictions_df):
        """Check for hazards"""
        hazards = []
        for _, row in predictions_df.iterrows():
            if row['predicted_aqi'] > 150:
                hazards.append({
                    'timestamp': row['timestamp'],
                    'aqi': row['predicted_aqi'],
                    'category': row['category'],
                    'severity': 'High' if row['predicted_aqi'] > 200 else 'Moderate'
                })
        
        if hazards:
            logger.warning(f"⚠️  {len(hazards)} hazardous periods detected")
        
        return hazards
    
    def run(self):
        """Run inference"""
        try:
            self.initialize()
            latest_data = self.get_latest_features()
            X, timestamps = self.prepare_forecast_features(latest_data, forecast_hours=config.pipeline.forecast_days * 24)
            predictions = self.predict(X)
            predictions_df, daily_avg = self.format_predictions(timestamps, predictions)
            hazards = self.check_hazards(predictions_df)
            
            result = {
                'predictions': predictions_df,
                'daily_average': daily_avg,
                'hazards': hazards,
                'generated_at': datetime.now()
            }
            
            logger.info("\n" + "="*50)
            logger.info("FORECAST SUMMARY")
            logger.info("="*50)
            for _, row in daily_avg.iterrows():
                logger.info(f"{row['date']}: AQI {row['predicted_aqi']:.0f} - {row['category']}")
            logger.info("="*50)
            
            predictions_df.to_csv('predictions.csv', index=False)
            logger.info("✅ Inference completed")
            
            return result
        except Exception as e:
            logger.error(f"❌ Inference failed: {e}")
            raise


def main():
    logger.add("logs/inference_pipeline_{time}.log", rotation="1 day")
    pipeline = InferencePipeline()
    pipeline.run()


if __name__ == "__main__":
    main()