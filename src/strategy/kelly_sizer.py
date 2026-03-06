"""Fractional Kelly criterion position sizing.

The Kelly criterion determines optimal bet sizing to maximize
long-term growth rate. We use fractional Kelly (default 15%)
to reduce variance at the cost of slightly lower expected growth.

On Kalshi:
- Buying YES at price p: win (100-p) cents, lose p cents
- Buying NO at price (100-p): win p cents, lose (100-p) cents
- Odds (b) = potential_win / potential_loss
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from src.config.settings import StrategyConfig
from src.strategy.edge_detector import TradeSignal

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    contracts: int
    price_cents: int
    total_cost: float       # In dollars
    kelly_fraction: float   # Full Kelly fraction
    sized_fraction: float   # After fractional Kelly
    capped_by: str | None   # 'pct', 'dollar', 'min_contracts', None


class KellySizer:
    """Fractional Kelly criterion position sizing."""

    def __init__(self, strategy_config: StrategyConfig):
        self.config = strategy_config

    def calculate_kelly(
        self,
        model_prob: float,
        market_price: float,
        side: str,
    ) -> float:
        """Calculate full Kelly fraction.

        Args:
            model_prob: Our probability of YES outcome
            market_price: YES price (0.0-1.0)
            side: 'yes' or 'no'

        Returns:
            Full Kelly fraction (negative means don't bet)
        """
        if side == "yes":
            win_prob = model_prob
            lose_prob = 1.0 - model_prob
            price = market_price
        else:
            win_prob = 1.0 - model_prob
            lose_prob = model_prob
            price = 1.0 - market_price

        if price <= 0 or price >= 1:
            return 0.0

        # Odds: potential win / potential loss
        b = (1.0 - price) / price

        if b <= 0:
            return 0.0

        # Kelly formula: (p*b - q) / b
        kelly = (win_prob * b - lose_prob) / b

        return kelly

    def calculate_position_size(
        self,
        signal: TradeSignal,
        bankroll: float,
    ) -> PositionSize:
        """Calculate the number of contracts to trade.

        Applies fractional Kelly, then caps by:
        1. max_position_pct of bankroll
        2. max_position_dollars absolute cap
        3. Minimum 1 contract

        Args:
            signal: Trade signal with model_prob and market_price
            bankroll: Current bankroll in dollars

        Returns:
            PositionSize with contract count and cost
        """
        kelly = self.calculate_kelly(
            signal.model_prob, signal.market_price, signal.side
        )

        if kelly <= 0:
            return PositionSize(
                contracts=0, price_cents=0, total_cost=0.0,
                kelly_fraction=kelly, sized_fraction=0.0, capped_by=None,
            )

        # Apply fractional Kelly
        sized = kelly * self.config.fractional_kelly

        # Dollar amount to risk
        dollar_amount = sized * bankroll

        # Apply caps
        capped_by = None

        pct_cap = self.config.max_position_pct * bankroll
        if dollar_amount > pct_cap:
            dollar_amount = pct_cap
            capped_by = "pct"

        if dollar_amount > self.config.max_position_dollars:
            dollar_amount = self.config.max_position_dollars
            capped_by = "dollar"

        # Calculate contract count
        if signal.side == "yes":
            price = signal.market_price
        else:
            price = 1.0 - signal.market_price

        price_cents = max(1, round(price * 100))
        cost_per_contract = price_cents / 100.0  # dollars

        if cost_per_contract <= 0:
            return PositionSize(
                contracts=0, price_cents=price_cents, total_cost=0.0,
                kelly_fraction=kelly, sized_fraction=sized, capped_by=None,
            )

        contracts = int(dollar_amount / cost_per_contract)

        # Ensure minimum contracts
        if contracts < self.config.min_contracts and kelly > 0:
            contracts = self.config.min_contracts
            capped_by = "min_contracts"

        total_cost = contracts * cost_per_contract

        logger.debug(
            f"Kelly sizing: full={kelly:.3f}, sized={sized:.3f}, "
            f"${dollar_amount:.2f} -> {contracts} contracts @ {price_cents}c "
            f"(capped_by={capped_by})"
        )

        return PositionSize(
            contracts=contracts,
            price_cents=price_cents,
            total_cost=total_cost,
            kelly_fraction=kelly,
            sized_fraction=sized,
            capped_by=capped_by,
        )
