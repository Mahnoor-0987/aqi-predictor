"""
Model training pipeline for AQI prediction
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import joblib
import json
from pathlib import Path
from datetime import datetime
from loguru import logger

from src.config import MODELS, TEST_SIZE, VALIDATION_SIZE, RANDOM_STATE, MODELS_DIR


class AQIModelTrainer:
    """Train and evaluate AQI prediction models"""
    
    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize trainer
        
        Args:
            model_type: Type of model ('random_forest', 'ridge', 'neural_network')
        """
        self.model_type = model_type
        self.models = {}  # One model for each horizon (24h, 48h, 72h)
        self.scaler = StandardScaler()
        self.feature_names = []
        self.metrics = {}
        
    def prepare_data(self, X: pd.DataFrame, y: pd.DataFrame, target_horizon: str = "24h"):
        """
        Prepare data for training
        
        Args:
            X: Features DataFrame
            y: Targets DataFrame (with columns: aqi_target_24h, aqi_target_48h, aqi_target_72h)
            target_horizon: Which target to predict ('24h', '48h', '72h')
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Select target column
        target_col = f"aqi_target_{target_horizon}"
        if target_col not in y.columns:
            raise ValueError(f"Target column {target_col} not found in y")
        
        y_target = y[target_col]
        
        # Remove rows with NaN targets
        valid_idx = ~y_target.isna()
        X_clean = X[valid_idx].copy()
        y_clean = y_target[valid_idx].copy()
        
        logger.info(f"Preparing data for {target_horizon} forecast. Valid samples: {len(X_clean)}")
        
        # Store feature names
        self.feature_names = X_clean.columns.tolist()
        
        # Split data: 70% train, 10% validation, 20% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_clean, y_clean, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False
        )
        
        val_size_adjusted = VALIDATION_SIZE / (1 - TEST_SIZE)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=RANDOM_STATE, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return (
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train.values, y_val.values, y_test.values
        )
    
    def build_random_forest(self):
        """Build Random Forest model"""
        params = MODELS["random_forest"]
        model = RandomForestRegressor(**params)
        logger.info(f"Built Random Forest with params: {params}")
        return model
    
    def build_ridge(self):
        """Build Ridge Regression model"""
        params = MODELS["ridge"]
        model = Ridge(**params)
        logger.info(f"Built Ridge Regression with params: {params}")
        return model
    
    def build_neural_network(self, input_dim: int):
        """Build Neural Network model"""
        params = MODELS["neural_network"]
        
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(input_dim,)))
        
        # Hidden layers
        for units in params["hidden_layers"]:
            model.add(keras.layers.Dense(units, activation='relu'))
            model.add(keras.layers.Dropout(0.2))
        
        # Output layer
        model.add(keras.layers.Dense(1))
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"Built Neural Network with architecture: {params['hidden_layers']}")
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, target_horizon: str):
        """
        Train model for specific horizon
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            target_horizon: Forecast horizon ('24h', '48h', '72h')
            
        Returns:
            Trained model
        """
        logger.info(f"Training {self.model_type} model for {target_horizon} forecast...")
        
        if self.model_type == "random_forest":
            model = self.build_random_forest()
            model.fit(X_train, y_train)
            
        elif self.model_type == "ridge":
            model = self.build_ridge()
            model.fit(X_train, y_train)
            
        elif self.model_type == "neural_network":
            model = self.build_neural_network(X_train.shape[1])
            
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            params = MODELS["neural_network"]
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                callbacks=[early_stopping],
                verbose=0
            )
            
            logger.info(f"Training completed. Final val_loss: {history.history['val_loss'][-1]:.4f}")
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.models[target_horizon] = model
        logger.info(f"✓ Model trained for {target_horizon}")
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, target_horizon: str):
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X_test, y_test: Test data
            target_horizon: Forecast horizon
            
        Returns:
            Dictionary with metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        if self.model_type == "neural_network":
            y_pred = y_pred.flatten()
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            "horizon": target_horizon,
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "n_samples": len(y_test)
        }
        
        self.metrics[target_horizon] = metrics
        
        logger.info(f"Evaluation for {target_horizon}:")
        logger.info(f"  RMSE: {rmse:.2f}")
        logger.info(f"  MAE:  {mae:.2f}")
        logger.info(f"  R²:   {r2:.4f}")
        
        return metrics
    
    def train_all_horizons(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Train models for all forecast horizons
        
        Args:
            X: Features DataFrame
            y: Targets DataFrame
        """
        horizons = ["24h", "48h", "72h"]
        
        for horizon in horizons:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training for {horizon} horizon")
            logger.info(f"{'='*60}")
            
            # Prepare data
            X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(X, y, horizon)
            
            # Train model
            model = self.train_model(X_train, y_train, X_val, y_val, horizon)
            
            # Evaluate model
            self.evaluate_model(model, X_test, y_test, horizon)
        
        logger.info(f"\n{'='*60}")
        logger.info("✓ Training complete for all horizons!")
        logger.info(f"{'='*60}")
    
    def save_models(self, version: str = None):
        """
        Save trained models and artifacts
        
        Args:
            version: Model version (default: timestamp)
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_dir = MODELS_DIR / f"{self.model_type}_v{version}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving models to {model_dir}")
        
        # Save each horizon model
        for horizon, model in self.models.items():
            model_path = model_dir / f"model_{horizon}.pkl"
            
            if self.model_type == "neural_network":
                model_path = model_dir / f"model_{horizon}.h5"
                model.save(str(model_path))
            else:
                joblib.dump(model, model_path)
            
            logger.info(f"  Saved model for {horizon}: {model_path.name}")
        
        # Save scaler
        scaler_path = model_dir / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        
        # Save feature names
        features_path = model_dir / "feature_names.json"
        with open(features_path, 'w') as f:
            json.dump(self.feature_names, f)
        
        # Save metrics
        metrics_path = model_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save metadata
        metadata = {
            "model_type": self.model_type,
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "n_features": len(self.feature_names),
            "horizons": list(self.models.keys())
        }
        
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ All artifacts saved to {model_dir}")
        
        return str(model_dir)
    
    def load_models(self, model_dir: str):
        """
        Load trained models and artifacts
        
        Args:
            model_dir: Directory containing saved models
        """
        model_dir = Path(model_dir)
        
        if not model_dir.exists():
            raise ValueError(f"Model directory not found: {model_dir}")
        
        logger.info(f"Loading models from {model_dir}")
        
        # Load metadata
        with open(model_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.model_type = metadata["model_type"]
        
        # Load scaler
        self.scaler = joblib.load(model_dir / "scaler.pkl")
        
        # Load feature names
        with open(model_dir / "feature_names.json", 'r') as f:
            self.feature_names = json.load(f)
        
        # Load models for each horizon
        for horizon in metadata["horizons"]:
            if self.model_type == "neural_network":
                model_path = model_dir / f"model_{horizon}.h5"
                model = keras.models.load_model(str(model_path))
            else:
                model_path = model_dir / f"model_{horizon}.pkl"
                model = joblib.load(model_path)
            
            self.models[horizon] = model
            logger.info(f"  Loaded model for {horizon}")
        
        # Load metrics
        with open(model_dir / "metrics.json", 'r') as f:
            self.metrics = json.load(f)
        
        logger.info(f"✓ Loaded {len(self.models)} models")


if __name__ == "__main__":
    # Configure logger
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), colorize=True)
    
    print("\nThis module should be imported and used in training scripts.")
    print("See: src/pipelines/training_pipeline.py")