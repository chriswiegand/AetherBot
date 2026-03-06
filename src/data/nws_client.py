"""NWS API client for forecasts and observations.

The NWS API provides official NWS forecasts and station observations.
Requires a User-Agent header with contact info.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime

import httpx

from src.config.cities import CityConfig
from src.config.settings import AppSettings, load_settings
from src.data.db import get_session
from src.data.models import NWSForecast
from src.utils.temperature import celsius_to_fahrenheit

logger = logging.getLogger(__name__)


@dataclass
class NWSForecastResult:
    city: str
    date: str
    high_f: float | None
    low_f: float | None


@dataclass
class NWSObservation:
    station: str
    timestamp: str
    temperature_f: float | None
    description: str | None


class NWSClient:
    """Fetches data from the NWS API (api.weather.gov)."""

    def __init__(self, settings: AppSettings | None = None):
        if settings is None:
            settings = load_settings()
        self.settings = settings
        self.ds = settings.data_sources
        self._client = httpx.Client(
            timeout=30.0,
            headers={
                "User-Agent": self.ds.nws_user_agent,
                "Accept": "application/geo+json",
            },
        )
        self._grid_cache: dict[str, tuple[str, int, int]] = {}

    def _get_grid(self, city: CityConfig) -> tuple[str, int, int]:
        """Get NWS grid coordinates for a city (cached).

        Returns:
            (grid_id, grid_x, grid_y) tuple
        """
        cache_key = f"{city.lat},{city.lon}"
        if cache_key in self._grid_cache:
            return self._grid_cache[cache_key]

        url = f"{self.ds.nws_base_url}/points/{city.lat},{city.lon}"
        try:
            resp = self._client.get(url)
            resp.raise_for_status()
            data = resp.json()
            props = data["properties"]
            grid_id = props["gridId"]
            grid_x = props["gridX"]
            grid_y = props["gridY"]
            self._grid_cache[cache_key] = (grid_id, grid_x, grid_y)
            return grid_id, grid_x, grid_y
        except (httpx.HTTPError, KeyError) as e:
            logger.error(f"NWS grid lookup failed for {city.name}: {e}")
            raise

    def get_forecast(self, city: CityConfig) -> list[NWSForecastResult]:
        """Get NWS 7-day forecast (12-hour periods).

        Returns list of daily high/low forecasts.
        """
        grid_id, grid_x, grid_y = self._get_grid(city)
        url = f"{self.ds.nws_base_url}/gridpoints/{grid_id}/{grid_x},{grid_y}/forecast"

        try:
            resp = self._client.get(url)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            logger.error(f"NWS forecast fetch failed for {city.name}: {e}")
            return []

        periods = data.get("properties", {}).get("periods", [])
        if not periods:
            return []

        # Group periods by day (pairs of daytime + nighttime)
        daily = {}
        for period in periods:
            start = period.get("startTime", "")[:10]  # YYYY-MM-DD
            temp = period.get("temperature")
            is_daytime = period.get("isDaytime", False)

            if start not in daily:
                daily[start] = {"high": None, "low": None}
            if is_daytime and temp is not None:
                daily[start]["high"] = float(temp)
            elif not is_daytime and temp is not None:
                daily[start]["low"] = float(temp)

        results = []
        for date_str, temps in sorted(daily.items()):
            results.append(NWSForecastResult(
                city=city.name,
                date=date_str,
                high_f=temps["high"],
                low_f=temps["low"],
            ))

        return results

    def get_latest_observation(self, city: CityConfig) -> NWSObservation | None:
        """Get the latest observation from the station."""
        url = f"{self.ds.nws_base_url}/stations/{city.station}/observations/latest"

        try:
            resp = self._client.get(url)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            logger.error(f"NWS observation fetch failed for {city.station}: {e}")
            return None

        props = data.get("properties", {})
        temp_c = props.get("temperature", {}).get("value")
        temp_f = celsius_to_fahrenheit(temp_c) if temp_c is not None else None

        return NWSObservation(
            station=city.station,
            timestamp=props.get("timestamp", ""),
            temperature_f=temp_f,
            description=props.get("textDescription"),
        )

    def store_forecasts(self, forecasts: list[NWSForecastResult]) -> int:
        """Store NWS forecasts in the database."""
        session = get_session()
        inserted = 0
        now = datetime.utcnow().isoformat()
        try:
            for fc in forecasts:
                row = NWSForecast(
                    city=fc.city,
                    forecast_date=fc.date,
                    high_f=fc.high_f,
                    low_f=fc.low_f,
                    fetched_at=now,
                )
                session.add(row)
                inserted += 1
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
        return inserted

    def fetch_and_store(self, city: CityConfig) -> list[NWSForecastResult]:
        """Fetch NWS forecast and store in database."""
        forecasts = self.get_forecast(city)
        if forecasts:
            count = self.store_forecasts(forecasts)
            logger.info(f"Stored {count} NWS forecast records for {city.name}")
        return forecasts

    def close(self):
        self._client.close()
