"""Open-Meteo GFS Ensemble (31-member) data fetcher.

Fetches temperature_2m from the GFS 0.25-degree ensemble (gfs025 model).
Response contains 31 members (member00-member30): 1 control + 30 perturbations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, date, timedelta

import httpx

from src.config.cities import CityConfig
from src.config.settings import AppSettings, load_settings
from src.data.db import get_session
from src.data.models import EnsembleForecast
from src.utils.time_utils import (
    filter_times_in_observation_window,
    parse_iso_datetime,
    UTC,
)

logger = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    city: str
    model_run_time: str
    target_date: str
    member_daily_maxes: list[float]  # 31 daily max temps in F, one per member
    member_hourly: dict[int, list[float]]  # {member: [hourly temps]}
    valid_times: list[str]


class EnsembleFetcher:
    """Fetches GFS ensemble forecasts from Open-Meteo."""

    def __init__(self, settings: AppSettings | None = None):
        if settings is None:
            settings = load_settings()
        self.settings = settings
        self.ds = settings.data_sources
        self._client = httpx.Client(timeout=60.0)

    def fetch_ensemble(
        self, city: CityConfig, forecast_days: int = 3
    ) -> list[EnsembleResult]:
        """Fetch GFS ensemble for a city and extract daily max per member.

        Args:
            city: City configuration
            forecast_days: Number of days to forecast (1-10 for gfs025)

        Returns:
            List of EnsembleResult, one per target date
        """
        params = {
            "latitude": city.lat,
            "longitude": city.lon,
            "hourly": "temperature_2m",
            "models": self.ds.ensemble_model,
            "temperature_unit": "fahrenheit",
            "timezone": "UTC",
            "forecast_days": forecast_days,
        }

        try:
            resp = self._client.get(self.ds.ensemble_url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            logger.error(f"Ensemble fetch failed for {city.name}: {e}")
            return []

        return self._parse_response(data, city)

    def _parse_response(self, data: dict, city: CityConfig) -> list[EnsembleResult]:
        """Parse Open-Meteo ensemble response into daily max per member."""
        hourly = data.get("hourly", {})
        time_strings = hourly.get("time", [])
        if not time_strings:
            logger.warning(f"No time data in ensemble response for {city.name}")
            return []

        # Parse times as UTC
        times = [parse_iso_datetime(t) for t in time_strings]

        # Extract member data (member00 through member30)
        member_data: dict[int, list[float]] = {}
        for m in range(self.ds.ensemble_members):
            key = f"temperature_2m_member{m:02d}"
            if key in hourly:
                member_data[m] = hourly[key]
            else:
                logger.warning(f"Missing {key} in ensemble response for {city.name}")

        if not member_data:
            return []

        # Determine model run time from response metadata
        # Open-Meteo doesn't explicitly provide this; approximate from first valid time
        model_run_time = times[0].isoformat()

        # Group by target date and compute daily max per member
        target_dates = set()
        for t in times:
            # Convert UTC time to local date for grouping
            local_date = t.date()
            target_dates.add(local_date)

        results = []
        for target_date in sorted(target_dates):
            # Get indices within observation window
            indices = filter_times_in_observation_window(
                times, target_date, city.timezone
            )
            if not indices:
                continue

            member_maxes = []
            for m in sorted(member_data.keys()):
                temps = member_data[m]
                window_temps = [temps[i] for i in indices if i < len(temps) and temps[i] is not None]
                if window_temps:
                    member_maxes.append(max(window_temps))
                else:
                    member_maxes.append(float("nan"))

            # Skip if too many missing members
            valid_count = sum(1 for t in member_maxes if t == t)  # NaN != NaN
            if valid_count < 20:
                logger.warning(
                    f"Only {valid_count} valid members for {city.name} on {target_date}"
                )
                continue

            results.append(EnsembleResult(
                city=city.name,
                model_run_time=model_run_time,
                target_date=target_date.isoformat(),
                member_daily_maxes=member_maxes,
                member_hourly={m: [member_data[m][i] for i in indices] for m in member_data},
                valid_times=[time_strings[i] for i in indices],
            ))

        return results

    def store_ensemble(self, results: list[EnsembleResult], city: CityConfig) -> int:
        """Store ensemble forecast data in the database.

        Returns:
            Number of new records inserted
        """
        session = get_session()
        inserted = 0
        try:
            for result in results:
                for member_idx, temp_f in enumerate(result.member_daily_maxes):
                    if temp_f != temp_f:  # Skip NaN
                        continue

                    existing = (
                        session.query(EnsembleForecast)
                        .filter_by(
                            city=result.city,
                            model_run_time=result.model_run_time,
                            valid_time=result.target_date,
                            member=member_idx,
                        )
                        .first()
                    )
                    if existing:
                        continue

                    row = EnsembleForecast(
                        city=result.city,
                        station=city.station,
                        model_run_time=result.model_run_time,
                        valid_time=result.target_date,
                        member=member_idx,
                        temperature_f=temp_f,
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

    def fetch_and_store(self, city: CityConfig, forecast_days: int = 3) -> list[EnsembleResult]:
        """Fetch ensemble data and store in database."""
        results = self.fetch_ensemble(city, forecast_days)
        if results:
            count = self.store_ensemble(results, city)
            logger.info(f"Stored {count} ensemble records for {city.name}")
        return results

    def close(self):
        self._client.close()
