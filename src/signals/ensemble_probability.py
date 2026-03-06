"""GFS Ensemble probability calculator.

Converts 31 ensemble member daily-max forecasts into bracket/threshold
probabilities by counting the fraction of members satisfying the condition.

Critical details:
- Round member daily maxes to integer F before comparison (matches NWS CLI)
- 'Above X' contracts use strictly greater than: high > threshold
- Bracket contracts: bracket_low <= high <= bracket_high
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.utils.temperature import ensemble_daily_max_to_integer

logger = logging.getLogger(__name__)


@dataclass
class ProbabilityResult:
    market_ticker: str
    probability: float
    members_qualifying: int
    total_members: int
    member_maxes_rounded: list[int]


class EnsembleProbabilityCalculator:
    """Computes probabilities from GFS ensemble member counts."""

    def calculate_above_probability(
        self,
        member_maxes: list[float],
        threshold: float,
    ) -> float:
        """Calculate probability that daily high > threshold.

        Kalshi 'above X' contracts settle YES when observed high
        is STRICTLY GREATER THAN the threshold. Since observations
        are integers, 'above 50' means observed high >= 51.

        Args:
            member_maxes: Daily max temperature for each ensemble member (F)
            threshold: Temperature threshold (F)

        Returns:
            Probability (0.0 to 1.0)
        """
        rounded = ensemble_daily_max_to_integer(member_maxes)
        valid = [t for t in rounded if t == t]  # Filter NaN (NaN != NaN)
        if not valid:
            return 0.5  # No data -> return uninformative prior

        qualifying = sum(1 for t in valid if t > threshold)
        return qualifying / len(valid)

    def calculate_bracket_probability(
        self,
        member_maxes: list[float],
        bracket_low: float | None,
        bracket_high: float | None,
    ) -> float:
        """Calculate probability that daily high falls in a bracket.

        For bracket [low, high]: low <= observed_high <= high.
        Open-ended brackets: bracket_low=None means no lower bound,
        bracket_high=None means no upper bound.

        Args:
            member_maxes: Daily max temperature for each ensemble member (F)
            bracket_low: Lower bound (inclusive), or None for open-ended
            bracket_high: Upper bound (inclusive), or None for open-ended

        Returns:
            Probability (0.0 to 1.0)
        """
        rounded = ensemble_daily_max_to_integer(member_maxes)
        valid = [t for t in rounded if t == t]
        if not valid:
            return 0.5

        qualifying = 0
        for t in valid:
            if bracket_low is not None and t < bracket_low:
                continue
            if bracket_high is not None and t > bracket_high:
                continue
            qualifying += 1

        return qualifying / len(valid)

    def get_full_distribution(
        self,
        member_maxes: list[float],
        markets: list,  # list of ParsedMarket
    ) -> dict[str, ProbabilityResult]:
        """Compute probabilities for all markets in an event.

        For bracket markets, probabilities should sum to ~1.0.
        For above markets, they should be monotonically decreasing.

        Returns:
            {market_ticker: ProbabilityResult}
        """
        rounded = ensemble_daily_max_to_integer(member_maxes)
        valid = [t for t in rounded if t == t]
        total = len(valid)

        results = {}
        for market in markets:
            if market.is_above_contract and market.threshold_f is not None:
                qualifying = sum(1 for t in valid if t > market.threshold_f)
                prob = qualifying / total if total > 0 else 0.5
            else:
                qualifying = 0
                for t in valid:
                    if market.bracket_low is not None and t < market.bracket_low:
                        continue
                    if market.bracket_high is not None and t > market.bracket_high:
                        continue
                    qualifying += 1
                prob = qualifying / total if total > 0 else 0.5

            results[market.market_ticker] = ProbabilityResult(
                market_ticker=market.market_ticker,
                probability=prob,
                members_qualifying=qualifying,
                total_members=total,
                member_maxes_rounded=rounded,
            )

        return results

    def get_ensemble_mean_and_spread(
        self, member_maxes: list[float]
    ) -> tuple[float, float]:
        """Calculate ensemble mean and standard deviation.

        Useful for HRRR correction and model blending.
        """
        valid = [t for t in member_maxes if t == t]
        if not valid:
            return 0.0, 0.0

        mean = sum(valid) / len(valid)
        if len(valid) < 2:
            return mean, 0.0

        variance = sum((t - mean) ** 2 for t in valid) / (len(valid) - 1)
        return mean, variance ** 0.5
