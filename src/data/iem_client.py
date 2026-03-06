"""Iowa Environmental Mesonet (IEM) client for NWS CLI data.

The IEM CLI API is the best programmatic source for NWS Daily Climate Reports.
CLI data is the SETTLEMENT SOURCE for all Kalshi weather markets.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import httpx

from src.config.cities import CityConfig
from src.config.settings import AppSettings
from src.data.db import get_session
from src.data.models import Observation

logger = logging.getLogger(__name__)

IEM_CLI_URL = "https://mesonet.agron.iastate.edu/json/cli.py"


@dataclass
class CLIReport:
    station: str
    date: str  # YYYY-MM-DD
    high_f: int | None
    low_f: int | None
    high_time: str | None
    low_time: str | None
    precip: float | None
    snow: float | None
    raw: dict


class IEMClient:
    """Fetches NWS CLI (Climate) reports from Iowa Environmental Mesonet."""

    def __init__(self, settings: AppSettings | None = None):
        self.base_url = IEM_CLI_URL
        self._client = httpx.Client(timeout=30.0)

    def get_cli(self, station: str, target_date: str) -> CLIReport | None:
        """Fetch a single CLI report.

        Args:
            station: ICAO station ID (e.g., 'KNYC')
            target_date: Date string 'YYYY-MM-DD'

        Returns:
            CLIReport or None if data not available
        """
        params = {"station": station, "date": target_date}
        try:
            resp = self._client.get(self.base_url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            logger.error(f"IEM CLI fetch failed for {station} on {target_date}: {e}")
            return None

        if not data:
            logger.warning(f"No CLI data for {station} on {target_date}")
            return None

        # IEM returns a list of records; take the first matching
        record = data[0] if isinstance(data, list) else data

        high = record.get("high")
        low = record.get("low")

        # IEM uses "M" for missing data
        if high == "M" or high is None:
            high = None
        else:
            high = int(high)

        if low == "M" or low is None:
            low = None
        else:
            low = int(low)

        return CLIReport(
            station=station,
            date=target_date,
            high_f=high,
            low_f=low,
            high_time=record.get("high_time"),
            low_time=record.get("low_time"),
            precip=_parse_precip(record.get("precip")),
            snow=_parse_precip(record.get("snow")),
            raw=record,
        )

    def get_cli_range(
        self, station: str, start_date: str, end_date: str, delay: float = 0.2
    ) -> list[CLIReport]:
        """Fetch CLI reports for a date range.

        Args:
            station: ICAO station ID
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD' (inclusive)
            delay: Seconds between requests (be nice to IEM)

        Returns:
            List of CLIReport objects
        """
        reports = []
        current = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)

        while current <= end:
            report = self.get_cli(station, current.isoformat())
            if report is not None and report.high_f is not None:
                reports.append(report)
            current += timedelta(days=1)
            time.sleep(delay)

        logger.info(
            f"Fetched {len(reports)} CLI reports for {station} "
            f"from {start_date} to {end_date}"
        )
        return reports

    def store_observations(self, reports: list[CLIReport], city_name: str) -> int:
        """Store CLI reports in the observations table.

        Returns:
            Number of new records inserted
        """
        session = get_session()
        inserted = 0
        try:
            for report in reports:
                # Check for existing record
                existing = (
                    session.query(Observation)
                    .filter_by(city=city_name, date=report.date, source="iem_cli")
                    .first()
                )
                if existing:
                    # Update if high changed
                    if existing.high_f != report.high_f:
                        existing.high_f = report.high_f
                        existing.low_f = report.low_f
                        existing.raw_json = json.dumps(report.raw)
                        existing.fetched_at = datetime.utcnow().isoformat()
                    continue

                obs = Observation(
                    city=city_name,
                    station=report.station,
                    date=report.date,
                    high_f=report.high_f,
                    low_f=report.low_f,
                    source="iem_cli",
                    raw_json=json.dumps(report.raw),
                    fetched_at=datetime.utcnow().isoformat(),
                )
                session.add(obs)
                inserted += 1

            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        return inserted

    def close(self):
        self._client.close()


def _parse_precip(value) -> float | None:
    """Parse precipitation value, handling 'T' (trace) and 'M' (missing)."""
    if value is None or value == "M":
        return None
    if value == "T":
        return 0.001  # Trace amount
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
