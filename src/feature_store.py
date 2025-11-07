"""Hopsworks Feature Store integration"""

import time
from typing import Optional, Tuple
from datetime import datetime
import pandas as pd
import hopsworks
import hsfs
from loguru import logger
from src.config import config


class FeatureStore:
    """Manages feature storage and retrieval in Hopsworks"""

    def __init__(self):
        self.project = None
        self.fs = None
        self.feature_group = None
        self.feature_view = None
        # Load Hopsworks configuration from config
        self.api_key = config.hopsworks.api_key
        self.project_name = config.hopsworks.project_name
        self.feature_group_name = config.hopsworks.feature_group_name
        self.feature_group_version = config.hopsworks.feature_group_version

    def connect(self) -> bool:
        """Connect to Hopsworks feature store"""
        try:
            logger.info("Connecting to Hopsworks...")
            self.project = hopsworks.login(api_key_value=self.api_key, project=self.project_name)
            self.fs = self.project.get_feature_store()
            logger.info(f"✓ Connected to Hopsworks project: {self.project_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Hopsworks: {e}")
            return False

    def create_feature_group(self, df: pd.DataFrame, description: str = "AQI features for prediction"):
        """Create or get feature group"""
        if self.fs is None and not self.connect():
            return None
        try:
            logger.info(f"Creating/getting feature group: {self.feature_group_name}")
            self.feature_group = self.fs.get_or_create_feature_group(
                name=self.feature_group_name,
                version=self.feature_group_version,
                description=description,
                primary_key=["timestamp"],
                event_time="timestamp",
                online_enabled=False
            )
            logger.info(f"✓ Feature group ready: {self.feature_group_name} v{self.feature_group_version}")
            return self.feature_group
        except Exception as e:
            logger.error(f"Error creating feature group: {e}")
            return None

    def insert_features(self, df: pd.DataFrame, overwrite: bool = False) -> bool:
        """Insert features into feature group"""
        try:
            if self.feature_group is None:
                self.create_feature_group(df)
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            self.feature_group.insert(df, overwrite=overwrite)
            logger.info(f"✓ Successfully inserted {len(df)} rows")
            return True
        except Exception as e:
            logger.error(f"Error inserting features: {e}")
            return False

    def create_feature_view(self):
        """Create or get a feature view for inference/training"""
        logger.info("Creating/getting feature view: aqi_feature_view")
        try:
            if self.fs is None and not self.connect():
                return None

            # Retry to handle possible delay in FG creation
            for attempt in range(3):
                try:
                    fg = self.fs.get_feature_group(name=self.feature_group_name, version=self.feature_group_version)
                    if fg:
                        break
                except hsfs.client.exceptions.RestAPIError:
                    logger.warning(f"Retry {attempt+1}/3: Feature group not ready yet...")
                    time.sleep(5)
            else:
                raise Exception("Feature group not found after retries.")

            self.feature_view = self.fs.get_or_create_feature_view(
                name="aqi_feature_view",
                version=1,
                query=fg.select_all()
            )
            logger.info("✓ Feature view successfully created or retrieved.")
            return self.feature_view

        except Exception as e:
            logger.error(f"Error creating feature view: {e}")
            raise

    def get_training_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch training features and targets"""
        try:
            if self.feature_view is None:
                self.create_feature_view()
            df = self.feature_view.get_batch_data()

            if start_date:
                df = df[df["timestamp"] >= start_date]
            if end_date:
                df = df[df["timestamp"] <= end_date]

            target_cols = ["aqi_target_24h", "aqi_target_48h", "aqi_target_72h"]
            feature_cols = [col for col in df.columns if col not in ["timestamp", "city"] + target_cols]

            X = df[feature_cols]
            y = df[target_cols] if all(col in df.columns for col in target_cols) else None

            logger.info(f"✓ Retrieved {len(df)} rows with {len(feature_cols)} features")
            return X, y

        except Exception as e:
            logger.error(f"Error getting training data: {e}")
            return pd.DataFrame(), pd.DataFrame()

    def get_latest_features(self, n_rows: int = 24) -> pd.DataFrame:
        """Fetch the latest rows for inference"""
        try:
            if self.feature_view is None:
                self.create_feature_view()
            df = self.feature_view.get_batch_data()
            df = df.sort_values("timestamp", ascending=False).head(n_rows)
            df = df.sort_values("timestamp").reset_index(drop=True)
            logger.info(f"✓ Retrieved {len(df)} latest rows")
            return df
        except Exception as e:
            logger.error(f"Error getting latest features: {e}")
            return pd.DataFrame()

    def get_feature_statistics(self) -> dict:
        """Get statistics about stored features"""
        try:
            if self.feature_group is None:
                self.feature_group = self.fs.get_feature_group(
                    name=self.feature_group_name,
                    version=self.feature_group_version
                )
            stats = {
                "feature_group_name": self.feature_group.name,
                "version": self.feature_group.version,
                "features": len(self.feature_group.features),
                "primary_keys": self.feature_group.primary_key,
                "event_time": self.feature_group.event_time
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}


def test_feature_store():
    """Test Hopsworks connection and basic operations"""
    print("\n" + "="*60)
    print("Testing Hopsworks Feature Store")
    print("="*60)

    fs = FeatureStore()

    print("\n1. Testing connection...")
    if fs.connect():
        print("✓ Connection successful!")
    else:
        print("✗ Connection failed!")
        return

    from src.data_collector import AQICNDataCollector
    from src.feature_engineering import FeatureEngineer

    collector = AQICNDataCollector()
    df_raw = collector.fetch_historical_data(days=2)

    engineer = FeatureEngineer()
    df_features = engineer.engineer_features(df_raw, create_targets=True)
    print(f"   Sample data shape: {df_features.shape}")

    print("\n2. Creating feature group...")
    fg = fs.create_feature_group(df_features)
    if fg:
        print("✓ Feature group created!")

    print("\n3. Inserting features...")
    if fs.insert_features(df_features, overwrite=True):
        print("✓ Features inserted!")

    print("\n4. Creating feature view...")
    fv = fs.create_feature_view()
    if fv:
        print("✓ Feature view created!")

    print("\n5. Feature statistics:")
    stats = fs.get_feature_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n" + "="*60)
    print("✓ Feature Store test complete!")
    print("="*60)


if __name__ == "__main__":
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), colorize=True)
    test_feature_store()
