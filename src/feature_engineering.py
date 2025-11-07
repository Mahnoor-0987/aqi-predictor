"""
Feature Engineering for AQI Prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List
from loguru import logger

# Pollutant features
POLLUTANT_FEATURES = ["pm25", "pm10", "o3", "no2", "so2", "co", "temp", "humidity", "pressure", "wind_speed"]


class FeatureEngineer:
    """Creates engineered features from raw AQI data"""

    def __init__(self):
        self.feature_names: List[str] = []

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features including cyclical encoding"""
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["day_of_month"] = df["timestamp"].dt.day
        df["month"] = df["timestamp"].dt.month
        df["year"] = df["timestamp"].dt.year
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Cyclical encoding
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # Rush hour
        df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

        logger.info("Created time features")
        return df

    def create_lag_features(self, df: pd.DataFrame, columns: List[str], lags: List[int] = [1, 3, 6, 12, 24]) -> pd.DataFrame:
        """Create lag features for given columns"""
        df = df.copy()
        for col in columns:
            if col not in df.columns:
                continue
            for lag in lags:
                df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
        logger.info(f"Created lag features for {columns}")
        return df

    def create_rolling_features(self, df: pd.DataFrame, columns: List[str], windows: List[int] = [3, 6, 12, 24]) -> pd.DataFrame:
        """Create rolling statistics features"""
        df = df.copy()
        for col in columns:
            if col not in df.columns:
                continue
            for window in windows:
                df[f"{col}_rolling_mean_{window}h"] = df[col].rolling(window=window, min_periods=1).mean()
                df[f"{col}_rolling_std_{window}h"] = df[col].rolling(window=window, min_periods=1).std()
                df[f"{col}_rolling_min_{window}h"] = df[col].rolling(window=window, min_periods=1).min()
                df[f"{col}_rolling_max_{window}h"] = df[col].rolling(window=window, min_periods=1).max()
        logger.info(f"Created rolling features for {columns}")
        return df

    def create_rate_of_change_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Compute 1-hour difference for selected columns"""
        df = df.copy()
        for col in columns:
            if col in df.columns:
                df[col] = df[col].replace({None: np.nan}).astype(float)
                df[f"{col}_change_1h"] = df[col].diff(1)
        logger.info(f"Created rate-of-change features for {columns}")
        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interactions between pollutants and weather features"""
        df = df.copy()
        if "pm25" in df.columns and "humidity" in df.columns:
            df["pm25_humidity_interaction"] = df["pm25"] * df["humidity"]
        if "pm25" in df.columns and "temp" in df.columns:
            df["pm25_temp_interaction"] = df["pm25"] * df["temp"]
        if "wind_speed" in df.columns:
            for pollutant in ["pm25", "pm10", "no2"]:
                if pollutant in df.columns:
                    df[f"{pollutant}_wind_ratio"] = df[pollutant] / (df["wind_speed"] + 1)
        logger.info("Created interaction features")
        return df

    def create_target_features(self, df: pd.DataFrame, target_col: str = "aqi", horizons: List[int] = [24, 48, 72]) -> pd.DataFrame:
        """Create future AQI targets for multi-step prediction"""
        df = df.copy()
        for horizon in horizons:
            df[f"{target_col}_target_{horizon}h"] = df[target_col].shift(-horizon)
        logger.info(f"Created target features for horizons: {horizons}")
        return df

    def engineer_features(self, df: pd.DataFrame, create_targets: bool = True) -> pd.DataFrame:
        """Full feature engineering pipeline"""
        logger.info("Starting feature engineering pipeline...")
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Time features
        df = self.create_time_features(df)

        # Lag and rolling features
        columns_to_use = ["aqi"] + [col for col in POLLUTANT_FEATURES if col in df.columns]
        df = self.create_lag_features(df, columns_to_use)
        df = self.create_rolling_features(df, columns_to_use)

        # Rate of change
        change_columns = ["aqi", "pm25", "pm10", "temp"]
        df = self.create_rate_of_change_features(df, change_columns)

        # Interaction features
        df = self.create_interaction_features(df)

        # Target features
        if create_targets:
            df = self.create_target_features(df, target_col="aqi")

        # Fill missing values
        df = df.fillna(method="ffill").fillna(method="bfill").fillna(0)

        # Store feature names (excluding timestamp, city, targets)
        self.feature_names = [col for col in df.columns if col not in ["timestamp", "city", "aqi_target_24h", "aqi_target_48h", "aqi_target_72h"]]

        logger.info(f"Feature engineering complete. Total features: {len(self.feature_names)}")
        return df

    def get_feature_names(self) -> List[str]:
        return self.feature_names


def test_feature_engineering():
    """Test the FeatureEngineer with sample AQI data"""
    from data_collector import AQICNDataCollector

    print("\n" + "="*60)
    print("Testing Feature Engineering")
    print("="*60)

    collector = AQICNDataCollector()
    df = collector.fetch_historical_data(days=7)
    print(f"\nRaw data shape: {df.shape}")

    engineer = FeatureEngineer()
    df_features = engineer.engineer_features(df, create_targets=True)
    print(f"\nEngineered data shape: {df_features.shape}")
    print(df_features.head())

    print(f"\nFeature names ({len(engineer.get_feature_names())}):")
    print(engineer.get_feature_names()[:20], " ...")


if __name__ == "__main__":
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), colorize=True)
    test_feature_engineering()
