"""Historical data collector for backtesting.

Backfills:
1. IEM CLI data (observed highs/lows) - the settlement truth
2. Open-Meteo archived weather data (actual temperatures)
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import httpx

from src.config.cities import CityConfig
from src.data.iem_client import IEMClient

logger = logging.getLogger(__name__)

ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"


class HistoricalCollector:
    """Backfills historical data for backtesting."""

    def __init__(self):
        self.iem = IEMClient()
        self._client = httpx.Client(timeout=60.0)

    def backfill_observations(
        self,
        city: CityConfig,
        start_date: str,
        end_date: str,
        delay: float = 0.2,
    ) -> int:
        """Backfill IEM CLI observations for a city.

        Args:
            city: City configuration
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD' (inclusive)
            delay: Seconds between requests

        Returns:
            Number of new observations stored
        """
        logger.info(f"Backfilling observations for {city.name} from {start_date} to {end_date}")
        reports = self.iem.get_cli_range(city.station, start_date, end_date, delay=delay)
        inserted = self.iem.store_observations(reports, city.name)
        logger.info(f"Stored {inserted} new observations for {city.name}")
        return inserted

    def backfill_archive_temps(
        self,
        city: CityConfig,
        start_date: str,
        end_date: str,
    ) -> dict:
        """Fetch historical actual temperatures from Open-Meteo archive.

        This provides the actual observed temperature data (not forecasts).
        Useful for computing climatological distributions.

        Returns:
            Dict with 'dates' and 'temp_max' lists
        """
        params = {
            "latitude": city.lat,
            "longitude": city.lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_max,temperature_2m_min",
            "temperature_unit": "fahrenheit",
            "timezone": city.timezone,
        }

        try:
            resp = self._client.get(ARCHIVE_API_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"Archive fetch failed for {city.name}: {e}")
            return {"dates": [], "temp_max": [], "temp_min": []}

        daily = data.get("daily", {})
        return {
            "dates": daily.get("time", []),
            "temp_max": daily.get("temperature_2m_max", []),
            "temp_min": daily.get("temperature_2m_min", []),
        }

    def close(self):
        self.iem.close()
        self._client.close()
