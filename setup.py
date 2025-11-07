"""
Setup script for AQI Predictor
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="aqi-predictor",
    version="1.0.0",
    author="Mahnoor Asim",
    author_email="your.email@example.com",
    description="Air Quality Index Prediction System for Karachi",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mahnoor-0987/aqi-predictor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "aqi-feature-pipeline=src.pipelines.feature_pipeline:main",
            "aqi-train=src.pipelines.training_pipeline:main",
            "aqi-dashboard=app.streamlit_app:main",
        ],
    },
)