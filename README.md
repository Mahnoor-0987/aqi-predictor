# 🌍 AQI Predictor - Air Quality Forecasting System

**Student:** Mahnoor Asim  
**Project:** 10Pearls Shine - Pearls AQI Predictor  
**Domain:** Data Sciences

![Feature Pipeline](https://github.com/Mahnoor-0987/aqi-predictor/actions/workflows/feature_pipeline.yml/badge.svg)
![Training Pipeline](https://github.com/Mahnoor-0987/aqi-predictor/actions/workflows/training_pipeline.yml/badge.svg)


## 📋 Overview

Complete serverless ML pipeline for predicting Air Quality Index (AQI) for the next 3 days using:
- Real-time data from AQICN API
- Hopsworks Feature Store
- Multiple ML models (Random Forest, Ridge, Neural Network)
- Automated CI/CD with GitHub Actions
- Interactive Streamlit dashboard

## ✨ Features

- ⏰ **Hourly Feature Pipeline**: Automated data collection
- 🤖 **Daily Training Pipeline**: Model retraining
- 📊 **3-Day Forecasting**: 72-hour predictions
- 🌐 **Interactive Dashboard**: Real-time visualizations
- ⚠️ **Hazard Alerts**: Warnings for unhealthy air quality
- 🔍 **Model Explainability**: SHAP values
- 🚀 **100% Serverless**: No infrastructure needed

## 🚀 Quick Start

### Prerequisites

1. **AQICN API Token**: `71853b5f-3e68-4be6-acce-306063cef881` ✅
2. **Hopsworks Account**: [Sign up here](https://app.hopsworks.ai/)
3. **Python 3.10+**

### Installation
```bash
# Clone repository
git clone <your-repo-url>
cd aqi-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Hopsworks API key
```

### Run Pipelines
```bash
# 1. Backfill historical data (30 days)
python -m src.feature_pipeline --backfill --days=30

# 2. Train models
python -m src.training_pipeline

# 3. Launch dashboard
cd app
streamlit run app.py
```

Dashboard opens at: http://localhost:8501

## 📊 Architecture
```
AQICN API → Feature Pipeline → Hopsworks Feature Store
                ↓
         Training Pipeline
                ↓
          Model Registry
                ↓
       Inference Pipeline
                ↓
      Streamlit Dashboard
```

## 🛠️ Technologies

- **Python 3.10**
- **Scikit-learn**: Random Forest, Ridge Regression
- **TensorFlow**: Neural Networks
- **Hopsworks**: Feature Store & Model Registry
- **Streamlit**: Web Dashboard
- **GitHub Actions**: CI/CD Automation
- **SHAP**: Model Explainability

## 📁 Project Structure
```
aqi-predictor/
├── src/
│   ├── config.py              # Configuration
│   ├── utils.py               # Utilities
│   ├── feature_pipeline.py    # Data collection
│   ├── training_pipeline.py   # Model training
│   └── inference_pipeline.py  # Predictions
├── app/
│   └── app.py                 # Dashboard
├── .github/workflows/         # CI/CD
├── requirements.txt
└── README.md
```

## 🔄 Automated Pipelines

- **Feature Pipeline**: Runs every hour via GitHub Actions
- **Training Pipeline**: Runs daily at 2 AM UTC

## 📈 Model Performance

Expected metrics:
- **RMSE**: < 15
- **R²**: > 0.85
- **MAE**: < 12
- Actual Performance (latest run):
- Random Forest: RMSE = 3.47, R² = 0.946, MAE = 2.36
- Ridge Regression: RMSE = 0.37, R² = 0.999, MAE = 0.03
- Neural Network: RMSE = 10.69, R² = 0.485, MAE = 8.40


## 🎯 Dashboard Features

1. **Forecast Tab**: 3-day predictions with charts
2. **Current AQI**: Real-time air quality status
3. **Analysis**: Statistics and feature importance
4. **Data**: Raw predictions and CSV export

## 🔧 Configuration

Edit `.env` file:
```bash
AQICN_API_TOKEN=YOUR_AQICN_API_TOKEN_HERE
HOPSWORKS_API_KEY=YOUR_HOPSWORKS_API_KEY_HERE
HOPSWORKS_PROJECT_NAME=pearls_aqi_predictor_M
CITY_NAME=Karachi
CITY_LAT=24.8607
CITY_LON=67.0011
```

## 📝 GitHub Secrets

Add these secrets for CI/CD:
- `AQICN_API_TOKEN`
- `HOPSWORKS_API_KEY`
- `HOPSWORKS_PROJECT_NAME`
- `CITY_NAME`, `CITY_LAT`, `CITY_LON`

## 🧪 Testing
```bash
pytest tests/ -v
```

## 📚 Documentation

- **Feature Pipeline**: Collects and processes AQI data hourly
- **Training Pipeline**: Trains 3 models and selects best performer
- **Inference Pipeline**: Generates 72-hour forecasts
- **Dashboard**: Interactive UI with real-time updates

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Module not found" | `pip install -r requirements.txt` |
| "Hopsworks error" | Check API key in `.env` |
| "No training data" | Run backfill first |
| "Dashboard blank" | Ensure pipelines ran successfully |

## 📞 Support

Check logs in `logs/` directory for detailed error messages.


## 🎉 Success Criteria

- ✅ All pipelines execute successfully
- ✅ Dashboard displays 3-day forecast
- ✅ Models achieve R² > 0.80
- ✅ GitHub Actions automated
- ✅ Complete documentation

## 👥 Author

**Mahnoor Asim** - Data Sciences Project

## 📄 License

MIT License

---

**Built with ❤️ for cleaner air and healthier communities**
