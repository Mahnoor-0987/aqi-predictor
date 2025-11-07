"""
Quick Start Script for AQI Predictor
Automates the entire setup and first run
"""
import os
import sys
import subprocess
from pathlib import Path
from loguru import logger


def print_banner():
    """Print welcome banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║          🌍 AQI PREDICTOR - QUICK START 🌍                ║
    ║                                                           ║
    ║     Air Quality Index Prediction for Karachi             ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_env_file():
    """Check if .env file exists and is configured"""
    env_path = Path(".env")
    
    if not env_path.exists():
        logger.warning(".env file not found!")
        logger.info("Creating .env from .env.example...")
        
        example_path = Path(".env.example")
        if example_path.exists():
            with open(example_path, 'r') as src, open(env_path, 'w') as dst:
                dst.write(src.read())
            
            logger.warning("⚠️  Please edit .env file with your API keys before continuing!")
            logger.info("Required keys:")
            logger.info("  1. AQICN_API_TOKEN (from https://aqicn.org/data-platform/token/)")
            logger.info("  2. HOPSWORKS_API_KEY (from https://app.hopsworks.ai/)")
            
            return False
    
    # Check if keys are configured
    from dotenv import load_dotenv
    load_dotenv()
    
    aqicn_token = os.getenv("AQICN_API_TOKEN")
    hopsworks_key = os.getenv("HOPSWORKS_API_KEY")
    
    if not aqicn_token or aqicn_token == "your_aqicn_token_here":
        logger.error("AQICN_API_TOKEN not configured in .env file!")
        return False
    
    if not hopsworks_key or hopsworks_key == "your_hopsworks_api_key_here":
        logger.warning("HOPSWORKS_API_KEY not configured. Feature store will be disabled.")
    
    return True


def install_dependencies():
    """Install required dependencies"""
    logger.info("Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("✓ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False


def test_api_connection():
    """Test AQICN API connection"""
    logger.info("Testing API connection...")
    
    try:
        from src.data.data_collector import AQICNDataCollector
        
        collector = AQICNDataCollector()
        data = collector.fetch_current_data()
        
        if data:
            logger.info(f"✓ API connection successful!")
            logger.info(f"  City: {data.get('city')}")
            logger.info(f"  Current AQI: {data.get('aqi')}")
            return True
        else:
            logger.error("Failed to fetch data from API")
            return False
            
    except Exception as e:
        logger.error(f"API test failed: {e}")
        return False


def run_initial_setup():
    """Run initial data collection and training"""
    logger.info("\n" + "="*60)
    logger.info("INITIAL SETUP")
    logger.info("="*60)
    
    # Step 1: Collect initial data
    logger.info("\n[1/3] Collecting initial data...")
    try:
        subprocess.check_call([sys.executable, "src/pipelines/feature_pipeline.py"])
        logger.info("✓ Feature pipeline completed!")
    except subprocess.CalledProcessError:
        logger.error("Feature pipeline failed")
        return False
    
    # Step 2: Train models
    logger.info("\n[2/3] Training initial models (this may take a few minutes)...")
    try:
        subprocess.check_call([
            sys.executable, 
            "src/pipelines/training_pipeline.py",
            "--model", "random_forest",
            "--days", "30"
        ])
        logger.info("✓ Training completed!")
    except subprocess.CalledProcessError:
        logger.error("Training failed")
        return False
    
    # Step 3: Test models
    logger.info("\n[3/3] Verifying trained models...")
    models_dir = Path("models")
    model_files = list(models_dir.glob("*/model_*.pkl")) + list(models_dir.glob("*/model_*.h5"))
    
    if model_files:
        logger.info(f"✓ Found {len(model_files)} trained model files!")
        return True
    else:
        logger.error("No trained models found")
        return False


def launch_dashboard():
    """Launch Streamlit dashboard"""
    logger.info("\n" + "="*60)
    logger.info("LAUNCHING DASHBOARD")
    logger.info("="*60)
    logger.info("\nStarting Streamlit app...")
    logger.info("Dashboard will open in your browser at http://localhost:8501")
    logger.info("\nPress Ctrl+C to stop the server.\n")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"])
    except KeyboardInterrupt:
        logger.info("\n\nDashboard stopped.")


def main():
    """Main quick start function"""
    print_banner()
    
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )
    
    # Step 1: Check environment
    logger.info("\n[Step 1/5] Checking environment configuration...")
    if not check_env_file():
        logger.error("\n❌ Setup incomplete. Please configure .env file first.")
        logger.info("\nEdit .env file with:")
        logger.info("  1. Your AQICN API token")
        logger.info("  2. Your Hopsworks API key")
        logger.info("\nThen run this script again: python quick_start.py")
        return
    logger.info("✓ Environment configured!")
    
    # Step 2: Install dependencies
    logger.info("\n[Step 2/5] Installing dependencies...")
    if not install_dependencies():
        logger.error("\n❌ Failed to install dependencies")
        return
    
    # Step 3: Test API
    logger.info("\n[Step 3/5] Testing API connection...")
    if not test_api_connection():
        logger.error("\n❌ API connection failed")
        logger.info("Please check your AQICN_API_TOKEN in .env file")
        return
    
    # Step 4: Initial setup
    logger.info("\n[Step 4/5] Running initial setup...")
    response = input("\nThis will collect data and train models (~5-10 minutes). Continue? (y/n): ")
    
    if response.lower() == 'y':
        if not run_initial_setup():
            logger.error("\n❌ Initial setup failed")
            return
    else:
        logger.info("Skipping initial setup. You can run training later with:")
        logger.info("  python src/pipelines/training_pipeline.py --model random_forest")
    
    # Step 5: Launch dashboard
    logger.info("\n[Step 5/5] Ready to launch dashboard!")
    
    logger.info("\n" + "="*60)
    logger.info("✓ QUICK START COMPLETE!")
    logger.info("="*60)
    logger.info("\nYour AQI Predictor is ready!")
    logger.info("\nNext steps:")
    logger.info("  1. Launch dashboard: streamlit run app/streamlit_app.py")
    logger.info("  2. Run tests: pytest tests/ -v")
    logger.info("  3. View logs: Check logs/ directory")
    logger.info("  4. Setup CI/CD: Add secrets to GitHub Actions")
    
    launch_now = input("\nLaunch dashboard now? (y/n): ")
    if launch_now.lower() == 'y':
        launch_dashboard()
    else:
        logger.info("\nYou can launch the dashboard anytime with:")
        logger.info("  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Setup failed with error: {e}")
        print("Please check the error message and try again.")