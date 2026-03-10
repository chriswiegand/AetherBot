"""Open-Meteo ECMWF IFS Ensemble (51-member) data fetcher.

Fetches temperature_2m from the ECMWF IFS 0.25-degree ensemble (ecmwf_ifs025).
Response contains 51 members (member00-member50): 1 control + 50 perturbations.

Uses the same Open-Meteo ensemble API as GFS but with a different model parameter.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone

import httpx

from src.config.cities import CityConfig
from src.config.settings import AppSettings, load_settings
from src.data.db import get_session
from src.data.models import ECMWFForecast
from src.utils.time_utils import (
    filter_times_in_observation_window,
    parse_iso_datetime,
    UTC,
)

logger = logging.getLogger(__name__)

# ECMWF IFS runs at 00Z, 06Z, 12Z, 18Z — data available ~6h after run start
ECMWF_MODEL = "ecmwf_ifs025"
ECMWF_MEMBERS = 51


@dataclass
class ECMWFResult:
    city: str
    model_run_time: str
    target_date: str
    member_daily_maxes: list[float]  # 51 daily max temps in F, one per member
    valid_times: list[str]


class ECMWFFetcher:
    """Fetches ECMWF IFS ensemble forecasts from Open-Meteo."""

    def __init__(self, settings: AppSettings | None = None):
        if settings is None:
            settings = load_settings()
        self.settings = settings
        self.ds = settings.data_sources
        self._client = httpx.Client(timeout=60.0)

    def fetch_ensemble(
        self, city: CityConfig, forecast_days: int = 3
    ) -> list[ECMWFResult]:
        """Fetch ECMWF IFS ensemble for a city and extract daily max per member."""
        params = {
            "latitude": city.lat,
            "longitude": city.lon,
            "hourly": "temperature_2m",
            "models": ECMWF_MODEL,
            "temperature_unit": "fahrenheit",
            "timezone": "UTC",
            "forecast_days": forecast_days,
        }

        try:
            resp = self._client.get(self.ds.ensemble_url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            logger.error(f"ECMWF fetch failed for {city.name}: {e}")
            return []

        return self._parse_response(data, city)

    def _parse_response(self, data: dict, city: CityConfig) -> list[ECMWFResult]:
        """Parse Open-Meteo ECMWF response into daily max per member."""
        hourly = data.get("hourly", {})
        time_strings = hourly.get("time", [])
        if not time_strings:
            logger.warning(f"No time data in ECMWF response for {city.name}")
            return []

        times = [parse_iso_datetime(t) for t in time_strings]

        # Extract member data (member00 through member50)
        member_data: dict[int, list[float]] = {}
        for m in range(ECMWF_MEMBERS):
            if m == 0:
                key = "temperature_2m"
                fallback = "temperature_2m_member00"
                if key in hourly:
                    member_data[m] = hourly[key]
                elif fallback in hourly:
                    member_data[m] = hourly[fallback]
            else:
                key = f"temperature_2m_member{m:02d}"
                if key in hourly:
                    member_data[m] = hourly[key]

        if not member_data:
            return []

        model_run_time = times[0].isoformat()

        target_dates = set()
        for t in times:
            target_dates.add(t.date())

        results = []
        for target_date in sorted(target_dates):
            indices = filter_times_in_observation_window(
                times, target_date, city.timezone
            )
            if not indices:
                continue

            member_maxes = []
            for m in sorted(member_data.keys()):
                temps = member_data[m]
                window_temps = [
                    temps[i] for i in indices
                    if i < len(temps) and temps[i] is not None
                ]
                if window_temps:
                    member_maxes.append(max(window_temps))
                else:
                    member_maxes.append(float("nan"))

            valid_count = sum(1 for t in member_maxes if t == t)
            if valid_count < 20:
                logger.warning(
                    f"Only {valid_count} valid ECMWF members for {city.name} on {target_date}"
                )
                continue

            results.append(ECMWFResult(
                city=city.name,
                model_run_time=model_run_time,
                target_date=target_date.isoformat(),
                member_daily_maxes=member_maxes,
                valid_times=[time_strings[i] for i in indices],
            ))

        return results

    def store_ensemble(self, results: list[ECMWFResult], city: CityConfig) -> int:
        """Store ECMWF forecast data in the database."""
        session = get_session()
        inserted = 0
        now = datetime.now(timezone.utc).isoformat()
        try:
            for result in results:
                for member_idx, temp_f in enumerate(result.member_daily_maxes):
                    if temp_f != temp_f:  # Skip NaN
                        continue

                    existing = (
                        session.query(ECMWFForecast)
                        .filter_by(
                            city=result.city,
                            model_run_time=result.model_run_time,
                            valid_time=result.target_date,
                            member=member_idx,
                        )
                        .first()
                    )
                    if existing:
                        if existing.temperature_f != temp_f:
                            existing.temperature_f = temp_f
                        existing.fetched_at = now
                        continue

                    row = ECMWFForecast(
                        city=result.city,
                        station=city.station,
                        model_run_time=result.model_run_time,
                        valid_time=result.target_date,
                        member=member_idx,
                        temperature_f=temp_f,
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

    def fetch_and_store(self, city: CityConfig, forecast_days: int = 3) -> list[ECMWFResult]:
        """Fetch ECMWF data and store in database."""
        results = self.fetch_ensemble(city, forecast_days)
        if results:
            count = self.store_ensemble(results, city)
            logger.info(
                f"Stored {count} new ECMWF records for {city.name} ({len(results)} dates)"
            )
        return results

    def close(self):
        self._client.close()
