"""Training Pipeline: Train and evaluate models"""
import argparse
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import joblib
import os
from src.config import config
from src.utils import HopsworksClient


class TrainingPipeline:
    """Training pipeline for AQI models"""
    
    def __init__(self):
        self.hops_client = HopsworksClient()
        self.models = {}
        self.metrics = {}
        self.best_model = None
        self.best_model_name = None
    
    def initialize(self):
        """Initialize Hopsworks"""
        logger.info("Initializing training pipeline...")
        self.hops_client.connect()
    
    def load_data(self):
        """Load training data"""
        logger.info("Loading data from feature store...")
        
        fg = self.hops_client.fs.get_feature_group(
            name=config.hopsworks.feature_group_name,
            version=config.hopsworks.feature_group_version
        )
        
        df = fg.read()
        
        target_col = 'aqi'
        exclude_cols = ['timestamp', 'city', 'latitude', 'longitude', 'dominant_pollutant', target_col]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].fillna(0)
        y = df[target_col]
        
        logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
        return X, y
    
    def prepare_data(self, X, y):
        """Prepare train/test split"""
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.model.test_size, random_state=config.model.random_state
        )
        
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """Train Random Forest"""
        logger.info("Training Random Forest...")
        
        model = RandomForestRegressor(**config.model.random_forest_params)
        model.fit(X_train, y_train)
        
        y_pred_test = model.predict(X_test)
        
        metrics = {
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'test_r2': r2_score(y_test, y_pred_test)
        }
        
        logger.info(f"RF - RMSE: {metrics['test_rmse']:.2f}, R²: {metrics['test_r2']:.3f}")
        
        self.models['random_forest'] = model
        self.metrics['random_forest'] = metrics
        return metrics
    
    def train_ridge(self, X_train, X_test, y_train, y_test):
        """Train Ridge Regression"""
        logger.info("Training Ridge Regression...")
        
        model = Ridge(**config.model.ridge_params)
        model.fit(X_train, y_train)
        
        y_pred_test = model.predict(X_test)
        
        metrics = {
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'test_r2': r2_score(y_test, y_pred_test)
        }
        
        logger.info(f"Ridge - RMSE: {metrics['test_rmse']:.2f}, R²: {metrics['test_r2']:.3f}")
        
        self.models['ridge'] = model
        self.metrics['ridge'] = metrics
        return metrics
    
    def train_neural_network(self, X_train, X_test, y_train, y_test):
        """Train Neural Network"""
        logger.info("Training Neural Network...")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        nn_params = config.model.neural_network_params
        model = keras.Sequential([
            keras.layers.Dense(nn_params['hidden_layers'][0], activation='relu', input_shape=(X_train.shape[1],)),
            keras.layers.Dropout(nn_params['dropout_rate']),
            keras.layers.Dense(nn_params['hidden_layers'][1], activation='relu'),
            keras.layers.Dropout(nn_params['dropout_rate']),
            keras.layers.Dense(nn_params['hidden_layers'][2], activation='relu'),
            keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=nn_params['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        model.fit(
            X_train_scaled, y_train,
            validation_split=0.2,
            epochs=nn_params['epochs'],
            batch_size=nn_params['batch_size'],
            callbacks=[early_stop],
            verbose=0
        )
        
        y_pred_test = model.predict(X_test_scaled, verbose=0).flatten()
        
        metrics = {
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'test_r2': r2_score(y_test, y_pred_test)
        }
        
        logger.info(f"NN - RMSE: {metrics['test_rmse']:.2f}, R²: {metrics['test_r2']:.3f}")
        
        self.models['neural_network'] = {'model': model, 'scaler': scaler}
        self.metrics['neural_network'] = metrics
        return metrics
    
    def select_best_model(self):
        """Select best model"""
        best_rmse = float('inf')
        for name, metrics in self.metrics.items():
            if metrics['test_rmse'] < best_rmse:
                best_rmse = metrics['test_rmse']
                self.best_model_name = name
                self.best_model = self.models[name]
        
        logger.info(f"🏆 Best model: {self.best_model_name} (RMSE: {best_rmse:.2f})")
    
    def save_models(self):
        """Save models"""
        logger.info("Saving models...")
        os.makedirs('models', exist_ok=True)
        
        if self.best_model_name == 'neural_network':
            self.best_model['model'].save('models/best_model.h5')
            joblib.dump(self.best_model['scaler'], 'models/scaler.pkl')
        else:
            joblib.dump(self.best_model, 'models/best_model.pkl')
        
        logger.info("✅ Models saved")
    import os
    import joblib
    import tensorflow as tf
    import hopsworks
    from hsml.model_registry import ModelRegistry
    
    # Connect to Hopsworks project and model registry
    project = hopsworks.login(
        api_key_value=config.hopsworks.api_key,
        project=config.hopsworks.project_name
    )
    mr: ModelRegistry = project.get_model_registry()
    
    # Example for scikit-learn models (Random Forest & Ridge)
    def register_sklearn_model(model, model_name: str, metrics_dict: dict, version: int = None):
        # Save locally
        local_path = f"models/{model_name}"
        os.makedirs(local_path, exist_ok=True)
        joblib.dump(model, os.path.join(local_path, "model.pkl"))
    
        # Register model
        skl_model = mr.sklearn.create_model(
            name=model_name,
            version=version,
            metrics=metrics_dict,
            description=f"{model_name} for AQI prediction"
        )
        # Save to registry
        skl_model.save(local_path)
        print(f"Registered sklearn model '{model_name}' version {skl_model.version}")
    
    # Example for TensorFlow model
    def register_tf_model(tf_model, model_name: str, metrics_dict: dict, version: int = None):
        # Save model
        local_path = f"models/{model_name}"
        tf_model.save(local_path, include_optimizer=False)
    
        # Register model
        tf_meta = mr.tensorflow.create_model(
            name=model_name,
            version=version,
            metrics=metrics_dict,
            description=f"{model_name} (TensorFlow) for AQI prediction"
        )
        tf_meta.save(local_path)
        print(f"Registered tensorflow model '{model_name}' version {tf_meta.version}")
    
    # After training and selecting best_model_name
    if best_model_name == "random_forest":
        register_sklearn_model(
            model=models["random_forest"],
            model_name="AQI_RF",
            metrics_dict=self.metrics["random_forest"]
        )
    elif best_model_name == "ridge":
        register_sklearn_model(
            model=models["ridge"],
            model_name="AQI_Ridge",
            metrics_dict=self.metrics["ridge"]
        )
    elif best_model_name == "neural_network":
        register_tf_model(
            tf_model=models["neural_network"]["model"],
            model_name="AQI_NN",
            metrics_dict=self.metrics["neural_network"]
        )

    def run(self):
        """Run training pipeline"""
        try:
            self.initialize()
            X, y = self.load_data()
            X_train, X_test, y_train, y_test = self.prepare_data(X, y)
            
            self.train_random_forest(X_train, X_test, y_train, y_test)
            self.train_ridge(X_train, X_test, y_train, y_test)
            self.train_neural_network(X_train, X_test, y_train, y_test)
            
            self.select_best_model()
            self.save_models()
            
            logger.info("\n" + "="*50)
            logger.info("TRAINING SUMMARY")
            logger.info("="*50)
            for name, metrics in self.metrics.items():
                logger.info(f"\n{name.upper()}:")
                logger.info(f"  RMSE: {metrics['test_rmse']:.2f}")
                logger.info(f"  MAE:  {metrics['test_mae']:.2f}")
                logger.info(f"  R²:   {metrics['test_r2']:.3f}")
            logger.info("\n" + "="*50)
            
            logger.info("✅ Training completed")
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise


def main():
    logger.add("logs/training_pipeline_{time}.log", rotation="1 day")
    pipeline = TrainingPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
