"""Synthetic market builder for backtesting.

Since we don't have historical Kalshi prices, we reconstruct
hypothetical market brackets around the climatological mean.
"""

from __future__ import annotations

import logging
from datetime import date

from src.config.cities import CityConfig
from src.data.models import Observation

logger = logging.getLogger(__name__)


class SyntheticMarketBuilder:
    """Builds synthetic market brackets from historical data."""

    def build_brackets(
        self,
        city: CityConfig,
        target_date: str,
        session,
        bracket_width: int = 2,
        n_brackets: int = 6,
    ) -> list[dict]:
        """Build synthetic above-threshold contracts around climatological mean.

        Uses the historical mean temperature for this month to center brackets.

        Args:
            city: City configuration
            target_date: Target date 'YYYY-MM-DD'
            session: SQLAlchemy session
            bracket_width: Width of each bracket in degrees F
            n_brackets: Number of brackets to generate

        Returns:
            List of dicts with threshold, market_price, etc.
        """
        d = date.fromisoformat(target_date)
        month = d.month

        # Get historical temperatures for this month
        obs = (
            session.query(Observation)
            .filter_by(city=city.name, source="iem_cli")
            .filter(Observation.high_f != None)
            .all()
        )

        # Filter to same month
        month_highs = []
        for o in obs:
            try:
                obs_date = date.fromisoformat(o.date)
                if obs_date.month == month:
                    month_highs.append(o.high_f)
            except (ValueError, TypeError):
                continue

        if not month_highs:
            return []

        mean_high = sum(month_highs) / len(month_highs)

        # Build above-threshold contracts centered around the mean
        # These simulate the binary "above X" contracts on Kalshi
        brackets = []
        start_threshold = int(mean_high) - (n_brackets // 2) * bracket_width

        for i in range(n_brackets * 2):
            threshold = start_threshold + i

            # Climatological probability: fraction of historical days above this
            clim_prob = sum(1 for h in month_highs if h > threshold) / len(month_highs)
            clim_prob = max(0.05, min(0.95, clim_prob))

            brackets.append({
                "threshold": threshold,
                "is_above": True,
                "market_price": clim_prob,  # Use climatology as market price
                "city": city.name,
                "target_date": target_date,
            })

        return brackets
