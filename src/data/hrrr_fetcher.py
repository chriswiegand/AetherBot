"""Open-Meteo HRRR data fetcher.

HRRR (High-Resolution Rapid Refresh) is a 3km CONUS model updated hourly.
At short lead times (<12hr), HRRR is dramatically better than GFS.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, date

import httpx

from src.config.cities import CityConfig
from src.config.settings import AppSettings, load_settings
from src.data.db import get_session
from src.data.models import HRRRForecast
from src.utils.time_utils import (
    filter_times_in_observation_window,
    parse_iso_datetime,
)

logger = logging.getLogger(__name__)


@dataclass
class HRRRResult:
    city: str
    model_run_time: str
    target_date: str
    daily_max_f: float
    hourly_temps: list[float]
    valid_times: list[str]


class HRRRFetcher:
    """Fetches HRRR deterministic forecasts from Open-Meteo."""

    def __init__(self, settings: AppSettings | None = None):
        if settings is None:
            settings = load_settings()
        self.settings = settings
        self.ds = settings.data_sources
        self._client = httpx.Client(timeout=60.0)

    def fetch_hrrr(self, city: CityConfig, forecast_days: int = 2) -> list[HRRRResult]:
        """Fetch HRRR forecast for a city.

        Args:
            city: City configuration
            forecast_days: Number of days (HRRR only covers ~48hr)

        Returns:
            List of HRRRResult, one per target date
        """
        params = {
            "latitude": city.lat,
            "longitude": city.lon,
            "hourly": "temperature_2m",
            "models": self.ds.hrrr_model,
            "temperature_unit": "fahrenheit",
            "timezone": "UTC",
            "forecast_days": forecast_days,
        }

        try:
            resp = self._client.get(self.ds.forecast_url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            logger.error(f"HRRR fetch failed for {city.name}: {e}")
            return []

        return self._parse_response(data, city)

    def _parse_response(self, data: dict, city: CityConfig) -> list[HRRRResult]:
        """Parse Open-Meteo HRRR response into daily max."""
        hourly = data.get("hourly", {})
        time_strings = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])

        if not time_strings or not temps:
            logger.warning(f"No HRRR data for {city.name}")
            return []

        times = [parse_iso_datetime(t) for t in time_strings]
        model_run_time = times[0].isoformat()

        # Group by target date
        target_dates = sorted(set(t.date() for t in times))

        results = []
        for target_date in target_dates:
            indices = filter_times_in_observation_window(
                times, target_date, city.timezone
            )
            if not indices:
                continue

            window_temps = [
                temps[i] for i in indices
                if i < len(temps) and temps[i] is not None
            ]
            if not window_temps:
                continue

            results.append(HRRRResult(
                city=city.name,
                model_run_time=model_run_time,
                target_date=target_date.isoformat(),
                daily_max_f=max(window_temps),
                hourly_temps=window_temps,
                valid_times=[time_strings[i] for i in indices],
            ))

        return results

    def store_hrrr(self, results: list[HRRRResult], city: CityConfig) -> int:
        """Store HRRR forecast data in the database."""
        session = get_session()
        inserted = 0
        try:
            for result in results:
                existing = (
                    session.query(HRRRForecast)
                    .filter_by(
                        city=result.city,
                        model_run_time=result.model_run_time,
                        valid_time=result.target_date,
                    )
                    .first()
                )
                if existing:
                    continue

                row = HRRRForecast(
                    city=result.city,
                    station=city.station,
                    model_run_time=result.model_run_time,
                    valid_time=result.target_date,
                    temperature_f=result.daily_max_f,
                    fetched_at=datetime.utcnow().isoformat(),
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

    def fetch_and_store(self, city: CityConfig, forecast_days: int = 2) -> list[HRRRResult]:
        """Fetch HRRR data and store in database."""
        results = self.fetch_hrrr(city, forecast_days)
        if results:
            count = self.store_hrrr(results, city)
            logger.info(f"Stored {count} HRRR records for {city.name}")
        return results

    def close(self):
        self._client.close()
