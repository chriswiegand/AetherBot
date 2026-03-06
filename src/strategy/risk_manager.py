"""Risk management: enforces position and loss limits before trading.

All risk checks must pass before any trade is executed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, date

from src.config.settings import StrategyConfig
from src.data.db import get_session
from src.data.models import Trade

logger = logging.getLogger(__name__)


@dataclass
class PortfolioState:
    bankroll: float
    open_positions: int
    daily_realized_pnl: float
    daily_unrealized_pnl: float
    positions_by_city: dict[str, int] = field(default_factory=dict)
    positions_by_date: dict[str, int] = field(default_factory=dict)
    total_exposure: float = 0.0

    @property
    def daily_total_pnl(self) -> float:
        return self.daily_realized_pnl + self.daily_unrealized_pnl


@dataclass
class RiskCheckResult:
    allowed: bool
    reason: str


class RiskManager:
    """Enforces all risk limits before trade execution."""

    def __init__(self, strategy_config: StrategyConfig):
        self.config = strategy_config

    def check_trade_allowed(
        self,
        city: str,
        target_date: str,
        total_cost: float,
        portfolio: PortfolioState,
    ) -> RiskCheckResult:
        """Check if a trade is allowed by all risk limits.

        Returns:
            RiskCheckResult with (allowed, reason)
        """
        # 1. Daily loss limit
        if portfolio.daily_total_pnl <= -self.config.daily_loss_limit:
            return RiskCheckResult(
                False,
                f"Daily loss limit reached: ${portfolio.daily_total_pnl:.2f} "
                f"(limit: -${self.config.daily_loss_limit})",
            )

        # Check if this trade would push us past the loss limit
        worst_case_loss = portfolio.daily_total_pnl - total_cost
        if worst_case_loss < -self.config.daily_loss_limit:
            return RiskCheckResult(
                False,
                f"Trade would exceed daily loss limit: "
                f"current PnL ${portfolio.daily_total_pnl:.2f} - "
                f"cost ${total_cost:.2f} = ${worst_case_loss:.2f}",
            )

        # 2. Max concurrent positions
        if portfolio.open_positions >= self.config.max_concurrent_positions:
            return RiskCheckResult(
                False,
                f"Max positions reached: {portfolio.open_positions} "
                f"(limit: {self.config.max_concurrent_positions})",
            )

        # 3. City concentration
        city_count = portfolio.positions_by_city.get(city, 0)
        if city_count >= self.config.max_positions_per_city:
            return RiskCheckResult(
                False,
                f"Max positions for {city}: {city_count} "
                f"(limit: {self.config.max_positions_per_city})",
            )

        # 4. Date concentration
        date_count = portfolio.positions_by_date.get(target_date, 0)
        if date_count >= self.config.max_positions_per_date:
            return RiskCheckResult(
                False,
                f"Max positions for {target_date}: {date_count} "
                f"(limit: {self.config.max_positions_per_date})",
            )

        # 5. Bankroll check
        if total_cost > portfolio.bankroll:
            return RiskCheckResult(
                False,
                f"Insufficient bankroll: ${portfolio.bankroll:.2f} "
                f"< cost ${total_cost:.2f}",
            )

        return RiskCheckResult(True, "All risk checks passed")

    def get_portfolio_state(self, bankroll: float) -> PortfolioState:
        """Build current portfolio state from the database."""
        session = get_session()
        try:
            today = date.today().isoformat()

            # Open positions
            open_trades = (
                session.query(Trade)
                .filter(Trade.status.in_(["filled", "pending"]))
                .all()
            )

            positions_by_city: dict[str, int] = {}
            positions_by_date: dict[str, int] = {}
            total_exposure = 0.0

            for trade in open_trades:
                positions_by_city[trade.city] = positions_by_city.get(trade.city, 0) + 1
                positions_by_date[trade.target_date] = (
                    positions_by_date.get(trade.target_date, 0) + 1
                )
                total_exposure += trade.total_cost

            # Today's realized PnL
            settled_today = (
                session.query(Trade)
                .filter(Trade.settled_at != None)
                .filter(Trade.settled_at >= today)
                .all()
            )
            daily_realized = sum(t.pnl or 0 for t in settled_today)

            return PortfolioState(
                bankroll=bankroll,
                open_positions=len(open_trades),
                daily_realized_pnl=daily_realized,
                daily_unrealized_pnl=0.0,  # TODO: compute from current prices
                positions_by_city=positions_by_city,
                positions_by_date=positions_by_date,
                total_exposure=total_exposure,
            )

        finally:
            session.close()
