"""Edge detection: compare calibrated model probabilities to market prices.

An 'edge' exists when our model probability significantly differs from
the market's implied probability (YES price). We only trade when the
edge exceeds a configurable threshold (default 8%).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.config.settings import StrategyConfig

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    market_ticker: str
    city: str
    target_date: str
    side: str           # 'yes' or 'no'
    model_prob: float   # Our calibrated probability of YES
    market_price: float # Current YES price (0.0-1.0)
    edge: float         # model_prob - market_price (signed)
    abs_edge: float     # |edge|
    lead_hours: float
    confidence: str     # 'low', 'medium', 'high'

    @property
    def expected_value_per_contract(self) -> float:
        """Expected value per contract in dollars."""
        if self.side == "yes":
            win_amount = 1.0 - self.market_price
            lose_amount = self.market_price
            return self.model_prob * win_amount - (1 - self.model_prob) * lose_amount
        else:
            # Buying NO at (1 - yes_price)
            no_price = 1.0 - self.market_price
            no_prob = 1.0 - self.model_prob
            win_amount = 1.0 - no_price
            lose_amount = no_price
            return no_prob * win_amount - (1 - no_prob) * lose_amount


class EdgeDetector:
    """Identifies trading opportunities from model-market divergences."""

    def __init__(self, strategy_config: StrategyConfig):
        self.config = strategy_config

    def scan_for_edges(
        self,
        signals: dict[str, float],   # {market_ticker: calibrated_prob}
        markets: dict[str, dict],     # {market_ticker: {yes_price, city, target_date, ...}}
        lead_hours: float = 24.0,
    ) -> list[TradeSignal]:
        """Scan for edges across all markets.

        Args:
            signals: Calibrated model probabilities per market
            markets: Current market data per market
            lead_hours: Current lead time to target date

        Returns:
            List of TradeSignal objects, sorted by |edge| descending
        """
        if lead_hours > self.config.max_lead_hours:
            return []

        trade_signals = []

        for ticker, model_prob in signals.items():
            market = markets.get(ticker)
            if market is None:
                continue

            yes_price = market.get("yes_price", 0.5)

            # Skip extreme prices (thin markets, likely to be illiquid)
            if yes_price > self.config.max_price or yes_price < self.config.min_price:
                continue

            edge = model_prob - yes_price
            abs_edge = abs(edge)

            # Determine side
            if edge > 0:
                side = "yes"  # Model says more likely than market
            else:
                side = "no"   # Model says less likely than market

            # Apply threshold
            if abs_edge < self.config.edge_threshold:
                continue

            # Determine confidence level
            if abs_edge > 0.20:
                confidence = "high"
            elif abs_edge > 0.12:
                confidence = "medium"
            else:
                confidence = "low"

            signal = TradeSignal(
                market_ticker=ticker,
                city=market.get("city", ""),
                target_date=market.get("target_date", ""),
                side=side,
                model_prob=model_prob,
                market_price=yes_price,
                edge=edge,
                abs_edge=abs_edge,
                lead_hours=lead_hours,
                confidence=confidence,
            )

            # Only trade if expected value is positive
            if signal.expected_value_per_contract > 0:
                trade_signals.append(signal)

        # Sort by |edge| descending (strongest signals first)
        trade_signals.sort(key=lambda s: s.abs_edge, reverse=True)

        return trade_signals

    def filter_with_hrrr_confirmation(
        self,
        signals: list[TradeSignal],
        hrrr_probs: dict[str, float],
    ) -> list[TradeSignal]:
        """Optionally lower the edge threshold when HRRR confirms the signal.

        If HRRR probability agrees with our ensemble signal direction,
        we can be more confident and accept a lower edge threshold.
        """
        filtered = []
        for signal in signals:
            hrrr_prob = hrrr_probs.get(signal.market_ticker)
            if hrrr_prob is not None:
                hrrr_agrees = (
                    (signal.side == "yes" and hrrr_prob > signal.market_price)
                    or (signal.side == "no" and hrrr_prob < signal.market_price)
                )
                if hrrr_agrees:
                    # HRRR confirms - accept lower threshold
                    if signal.abs_edge >= self.config.min_edge_hrrr_confirm:
                        if signal.confidence == "low":
                            signal.confidence = "medium"
                        filtered.append(signal)
                        continue

            # Standard threshold
            if signal.abs_edge >= self.config.edge_threshold:
                filtered.append(signal)

        return filtered
