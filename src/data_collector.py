"""
AQICN Data Collector
Collects current and simulated historical AQI and weather data from AQICN API
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional
from loguru import logger
import random

# Config variables (import from your config or .env)
from src.config import config

CITY_NAME = config.location.city_name
CITY_LAT = config.location.latitude
CITY_LON = config.location.longitude
AQICN_API_TOKEN = config.api.aqicn_token
AQICN_BASE_URL = config.api.aqicn_base_url.rstrip('/')


class AQICNDataCollector:
    """Collects AQI and weather data from AQICN API"""

    def __init__(self):
        self.api_token = AQICN_API_TOKEN
        self.base_url = AQICN_BASE_URL

        if not self.api_token:
            raise ValueError("AQICN_API_TOKEN not found in environment variables")

    def fetch_current_data(self) -> Optional[Dict]:
        """
        Fetch current AQI data for the configured city.

        Returns:
            Dict containing AQI and weather data, or None if fetch fails.
        """
        try:
            url = f"{self.base_url}/feed/{CITY_NAME}/?token={self.api_token}"
            logger.info(f"Fetching data from: {url}")

            response = requests.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()
            if data.get("status") != "ok":
                logger.warning(f"API returned status: {data.get('status')}")
                return self._fetch_by_coordinates()

            return self._parse_response(data)

        except Exception as e:
            logger.error(f"Error fetching current data: {e}")
            return self._fetch_by_coordinates()

    def _fetch_by_coordinates(self) -> Optional[Dict]:
        """Fetch data using city latitude and longitude"""
        try:
            url = f"{self.base_url}/feed/geo:{CITY_LAT};{CITY_LON}/?token={self.api_token}"
            logger.info(f"Fetching by coordinates: {url}")

            response = requests.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()
            if data.get("status") == "ok":
                return self._parse_response(data)
            else:
                logger.error(f"Failed to fetch by coordinates: {data}")
                return None

        except Exception as e:
            logger.error(f"Error fetching by coordinates: {e}")
            return None

    def _parse_response(self, data: Dict) -> Dict:
        """Parse API response into structured dictionary"""
        try:
            raw_data = data.get("data", {})

            timestamp = raw_data.get("time", {}).get("iso", datetime.utcnow().isoformat())
            aqi = raw_data.get("aqi", None)

            iaqi = raw_data.get("iaqi", {})
            pollutants = {
                "pm25": iaqi.get("pm25", {}).get("v"),
                "pm10": iaqi.get("pm10", {}).get("v"),
                "o3": iaqi.get("o3", {}).get("v"),
                "no2": iaqi.get("no2", {}).get("v"),
                "so2": iaqi.get("so2", {}).get("v"),
                "co": iaqi.get("co", {}).get("v"),
            }

            weather = {
                "temp": iaqi.get("t", {}).get("v"),
                "humidity": iaqi.get("h", {}).get("v"),
                "pressure": iaqi.get("p", {}).get("v"),
                "wind_speed": iaqi.get("w", {}).get("v"),
            }

            parsed_data = {
                "timestamp": timestamp,
                "city": raw_data.get("city", {}).get("name", CITY_NAME),
                "aqi": aqi,
                **pollutants,
                **weather
            }

            logger.info(f"Successfully parsed data: AQI={aqi}")
            return parsed_data

        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return None

    def fetch_historical_data(self, days: int = 30) -> pd.DataFrame:
        """
        Generate simulated historical data based on current AQI.

        Args:
            days: Number of days to generate data for

        Returns:
            DataFrame containing simulated historical AQI data
        """
        logger.warning("Using simulated historical data (free AQICN API doesn't provide it).")

        current_data = self.fetch_current_data()
        if not current_data:
            logger.error("Failed to fetch current data")
            return pd.DataFrame()

        historical_data = []
        base_timestamp = datetime.fromisoformat(current_data["timestamp"].replace("Z", "+00:00"))

        for hour_offset in range(days * 24):
            timestamp = base_timestamp - timedelta(hours=hour_offset)
            data_point = current_data.copy()
            data_point["timestamp"] = timestamp.isoformat()

            for key in ["aqi", "pm25", "pm10", "o3", "no2", "so2", "co"]:
                if data_point.get(key) is not None:
                    variation = random.uniform(-0.2, 0.2)
                    data_point[key] = max(0, data_point[key] * (1 + variation))

            historical_data.append(data_point)

        df = pd.DataFrame(historical_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"Created {len(df)} historical data points")
        return df

    def get_station_info(self) -> Optional[Dict]:
        """Retrieve monitoring station information"""
        try:
            url = f"{self.base_url}/feed/{CITY_NAME}/?token={self.api_token}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()
            if data.get("status") == "ok":
                station_data = data.get("data", {}).get("city", {})
                return {
                    "city": station_data.get("name"),
                    "coordinates": station_data.get("geo"),
                    "url": station_data.get("url"),
                    "station_name": station_data.get("name"),
                }
            return None

        except Exception as e:
            logger.error(f"Error getting station info: {e}")
            return None


def test_connection():
    """Test AQICN API connection and print sample data"""
    collector = AQICNDataCollector()

    print("\n" + "="*60)
    print("Testing AQICN API Connection")
    print("="*60)

    # Fetch current data
    print("\n1. Fetching current AQI data...")
    current_data = collector.fetch_current_data()
    if current_data:
        print("✓ Successfully fetched current data!")
        print(f"  - City: {current_data.get('city')}")
        print(f"  - AQI: {current_data.get('aqi')}")
        print(f"  - PM2.5: {current_data.get('pm25')}")
        print(f"  - Temperature: {current_data.get('temp')}")
        print(f"  - Timestamp: {current_data.get('timestamp')}")
    else:
        print("✗ Failed to fetch current data")

    # Fetch station info
    print("\n2. Fetching station information...")
    station_info = collector.get_station_info()
    if station_info:
        print("✓ Successfully fetched station info!")
        print(f"  - Station: {station_info.get('station_name')}")
        print(f"  - Coordinates: {station_info.get('coordinates')}")

    print("\n" + "="*60)


if __name__ == "__main__":
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), colorize=True, format="{message}")
    test_connection()
