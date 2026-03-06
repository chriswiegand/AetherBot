#!/usr/bin/env python3
"""Backfill historical observation data from IEM CLI.

Downloads 2+ years of daily climate reports for all configured cities.
This data is used for:
  - Calibration training (isotonic regression)
  - Backtest replay
  - Station bias analysis

Usage:
    python scripts/backfill_history.py                    # Default: 2 years
    python scripts/backfill_history.py --days 365         # 1 year
    python scripts/backfill_history.py --start 2024-01-01 # From specific date
    python scripts/backfill_history.py --city NYC         # Single city
"""

import argparse
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

from src.config.settings import load_settings
from src.config.cities import load_cities
from src.data.db import init_db
from src.data.historical_collector import backfill_observations, backfill_archive_temps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Backfill historical weather data")
    parser.add_argument("--days", type=int, default=730, help="Days of history to fetch (default: 730)")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD). Overrides --days.")
    parser.add_argument("--city", type=str, default=None, help="Single city to backfill (e.g., NYC)")
    parser.add_argument(
        "--include-archive",
        action="store_true",
        help="Also backfill Open-Meteo archive temps for climatological distributions",
    )
    args = parser.parse_args()

    settings = load_settings()
    init_db(settings)
    cities = load_cities()

    # Filter to single city if specified
    if args.city:
        if args.city not in cities:
            logger.error(f"Unknown city: {args.city}. Available: {', '.join(cities.keys())}")
            sys.exit(1)
        cities = {args.city: cities[args.city]}

    # Determine date range
    end_date = date.today() - timedelta(days=1)  # Yesterday (today may not have CLI yet)
    if args.start:
        start_date = date.fromisoformat(args.start)
    else:
        start_date = end_date - timedelta(days=args.days)

    logger.info("=" * 60)
    logger.info("AetherBot - Historical Data Backfill")
    logger.info("=" * 60)
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Cities: {', '.join(cities.keys())}")

    # Backfill CLI observations
    logger.info("\n--- Backfilling IEM CLI observations ---")
    backfill_observations(cities, start_date, end_date)

    # Optionally backfill archive temps
    if args.include_archive:
        logger.info("\n--- Backfilling Open-Meteo archive temps ---")
        backfill_archive_temps(cities, start_date, end_date)

    logger.info("\nBackfill complete.")


if __name__ == "__main__":
    main()
