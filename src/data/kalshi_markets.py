"""Kalshi KXHIGH market discovery and bracket parsing.

Discovers active weather markets and parses bracket structure
from market titles/subtitles.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from src.config.cities import CityConfig
from src.data.db import get_session
from src.data.kalshi_client import KalshiClient, Market
from src.data.models import KalshiMarket

logger = logging.getLogger(__name__)


@dataclass
class ParsedMarket:
    market_ticker: str
    event_ticker: str
    city: str
    target_date: str
    bracket_low: float | None
    bracket_high: float | None
    is_above_contract: bool
    threshold_f: float | None
    yes_price: float
    no_price: float
    volume: int
    close_time: str | None
    status: str


class KalshiMarketDiscovery:
    """Discovers and parses KXHIGH weather markets."""

    def __init__(self, kalshi_client: KalshiClient):
        self.client = kalshi_client

    def discover_active_markets(
        self, cities: dict[str, CityConfig]
    ) -> list[ParsedMarket]:
        """Discover all active KXHIGH markets across cities.

        Returns:
            List of ParsedMarket objects
        """
        all_markets = []

        for city_name, city_config in cities.items():
            series = city_config.kalshi_series
            try:
                raw_markets = self.client.get_markets(series_ticker=series, status="open")
                parsed = self._parse_markets(raw_markets, city_name)
                all_markets.extend(parsed)
                logger.info(
                    f"Found {len(parsed)} active markets for {city_name} ({series})"
                )
            except Exception as e:
                logger.error(f"Failed to discover markets for {city_name}: {e}")

        return all_markets

    def _parse_markets(
        self, markets: list[Market], city_name: str
    ) -> list[ParsedMarket]:
        """Parse market titles to extract bracket/threshold info."""
        parsed = []
        for m in markets:
            pm = self._parse_single_market(m, city_name)
            if pm is not None:
                parsed.append(pm)
        return parsed

    def _parse_single_market(
        self, market: Market, city_name: str
    ) -> ParsedMarket | None:
        """Parse a single market's title to determine its type and parameters."""
        title = market.title.lower()
        subtitle = market.subtitle.lower() if market.subtitle else ""
        combined = f"{title} {subtitle}"

        # Extract date from event ticker (e.g., KXHIGHNY-26MAR06 -> 2026-03-06)
        target_date = _parse_date_from_ticker(market.event_ticker)

        # Determine if this is an "above" contract or a bracket contract
        is_above = False
        threshold_f = None
        bracket_low = None
        bracket_high = None

        # Pattern: "above X degrees" or "X degrees or above" or "> X"
        above_patterns = [
            r"(?:above|over|greater than|higher than)\s+(\d+)",
            r"(\d+)\s*(?:degrees?\s+)?(?:or\s+)?(?:above|higher|more)",
            r">\s*(\d+)",
        ]
        for pattern in above_patterns:
            match = re.search(pattern, combined)
            if match:
                is_above = True
                threshold_f = float(match.group(1))
                break

        if not is_above:
            # Pattern: "X to Y degrees" or "between X and Y"
            bracket_patterns = [
                r"(\d+)\s*(?:to|through|-)\s*(\d+)",
                r"between\s+(\d+)\s+and\s+(\d+)",
            ]
            for pattern in bracket_patterns:
                match = re.search(pattern, combined)
                if match:
                    bracket_low = float(match.group(1))
                    bracket_high = float(match.group(2))
                    break

            if bracket_low is None:
                # Pattern: "X or below" / "X or lower"
                below_patterns = [
                    r"(\d+)\s*(?:degrees?\s+)?(?:or\s+)?(?:below|lower|less|under)",
                    r"(?:below|under|less than)\s+(\d+)",
                ]
                for pattern in below_patterns:
                    match = re.search(pattern, combined)
                    if match:
                        bracket_high = float(match.group(1))
                        break

        # If we couldn't parse anything, try extracting threshold from ticker
        if not is_above and bracket_low is None and bracket_high is None:
            threshold_from_ticker = _parse_threshold_from_ticker(market.ticker)
            if threshold_from_ticker is not None:
                is_above = True
                threshold_f = threshold_from_ticker

        return ParsedMarket(
            market_ticker=market.ticker,
            event_ticker=market.event_ticker,
            city=city_name,
            target_date=target_date or "",
            bracket_low=bracket_low,
            bracket_high=bracket_high,
            is_above_contract=is_above,
            threshold_f=threshold_f,
            yes_price=market.yes_bid,
            no_price=1.0 - market.yes_ask if market.yes_ask > 0 else 0,
            volume=market.volume,
            close_time=market.close_time,
            status=market.status,
        )

    def refresh_prices(self, markets: list[ParsedMarket]) -> list[ParsedMarket]:
        """Refresh YES/NO prices for a list of markets."""
        updated = []
        for pm in markets:
            try:
                m = self.client.get_market(pm.market_ticker)
                pm.yes_price = m.yes_bid
                pm.no_price = 1.0 - m.yes_ask if m.yes_ask > 0 else 0
                pm.volume = m.volume
                pm.status = m.status
            except Exception as e:
                logger.warning(f"Failed to refresh price for {pm.market_ticker}: {e}")
            updated.append(pm)
        return updated

    def store_markets(self, markets: list[ParsedMarket]) -> int:
        """Store discovered markets in the database."""
        session = get_session()
        inserted = 0
        now = datetime.now(timezone.utc).isoformat()

        try:
            for pm in markets:
                existing = (
                    session.query(KalshiMarket)
                    .filter_by(market_ticker=pm.market_ticker)
                    .first()
                )
                if existing:
                    existing.yes_price = pm.yes_price
                    existing.no_price = pm.no_price
                    existing.volume = pm.volume
                    existing.status = pm.status
                    existing.last_updated = now
                else:
                    row = KalshiMarket(
                        event_ticker=pm.event_ticker,
                        market_ticker=pm.market_ticker,
                        city=pm.city,
                        target_date=pm.target_date,
                        bracket_low=pm.bracket_low,
                        bracket_high=pm.bracket_high,
                        is_above_contract=1 if pm.is_above_contract else 0,
                        threshold_f=pm.threshold_f,
                        yes_price=pm.yes_price,
                        no_price=pm.no_price,
                        volume=pm.volume,
                        close_time=pm.close_time,
                        status=pm.status,
                        first_discovered_at=now,
                        last_updated=now,
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


def _parse_date_from_ticker(event_ticker: str) -> str | None:
    """Parse date from event ticker like KXHIGHNY-26MAR06."""
    match = re.search(r"-(\d{2})([A-Z]{3})(\d{2})$", event_ticker)
    if not match:
        return None

    year = int(match.group(1)) + 2000
    month_str = match.group(2)
    day = int(match.group(3))

    months = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    month = months.get(month_str, 1)

    return f"{year}-{month:02d}-{day:02d}"


def _parse_threshold_from_ticker(market_ticker: str) -> float | None:
    """Try to extract temperature threshold from market ticker suffix.

    E.g., KXHIGHNY-26MAR06-T48 -> 48.0
    """
    match = re.search(r"-T(\d+)$", market_ticker)
    if match:
        return float(match.group(1))
    return None
