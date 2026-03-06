"""Paper trading execution engine.

Simulates trades without real money. Assumes immediate fill at the
signal price. Tracks a virtual bankroll in the database.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

from src.config.settings import AppSettings, load_settings
from src.data.db import get_session
from src.data.models import Trade
from src.strategy.edge_detector import TradeSignal
from src.strategy.kelly_sizer import PositionSize

logger = logging.getLogger(__name__)


class PaperTrader:
    """Simulated trade execution for paper trading mode."""

    def __init__(self, settings: AppSettings | None = None):
        if settings is None:
            settings = load_settings()
        self.settings = settings
        self._bankroll = settings.paper_trading.initial_bankroll

    def execute_trade(
        self, signal: TradeSignal, size: PositionSize
    ) -> Trade | None:
        """Execute a simulated trade.

        Assumes immediate fill at the market price.
        """
        if size.contracts <= 0:
            return None

        now = datetime.utcnow().isoformat()
        trade_id = str(uuid.uuid4())

        session = get_session()
        try:
            trade = Trade(
                trade_id=trade_id,
                mode="paper",
                market_ticker=signal.market_ticker,
                city=signal.city,
                target_date=signal.target_date,
                side=signal.side,
                direction="buy",
                contracts=size.contracts,
                price=signal.market_price if signal.side == "yes" else (1.0 - signal.market_price),
                total_cost=size.total_cost,
                model_prob=signal.model_prob,
                market_price=signal.market_price,
                edge=signal.edge,
                kelly_fraction=size.kelly_fraction,
                status="filled",
                fill_price=signal.market_price if signal.side == "yes" else (1.0 - signal.market_price),
                created_at=now,
                updated_at=now,
            )
            session.add(trade)
            session.commit()

            # Update bankroll
            self._bankroll -= size.total_cost

            logger.info(
                f"PAPER TRADE: {signal.side.upper()} {size.contracts}x "
                f"{signal.market_ticker} @ {size.price_cents}c "
                f"(edge={signal.edge:+.1%}, cost=${size.total_cost:.2f})"
            )

            return trade

        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def settle_position(
        self, trade_id: str, settlement_value: int
    ) -> float:
        """Settle a paper trade and compute PnL.

        Args:
            trade_id: UUID of the trade
            settlement_value: 100 if YES wins, 0 if NO wins

        Returns:
            PnL in dollars
        """
        session = get_session()
        try:
            trade = session.query(Trade).filter_by(trade_id=trade_id).first()
            if trade is None:
                logger.error(f"Trade not found: {trade_id}")
                return 0.0

            if trade.status == "settled":
                logger.warning(f"Trade already settled: {trade_id}")
                return trade.pnl or 0.0

            # Compute PnL
            if trade.side == "yes":
                if settlement_value == 100:
                    # YES wins: profit = (1.00 - price) * contracts
                    pnl = (1.0 - trade.price) * trade.contracts
                else:
                    # YES loses: loss = -price * contracts
                    pnl = -trade.price * trade.contracts
            else:
                # NO side
                no_price = trade.price  # Already stored as NO price
                if settlement_value == 0:
                    # NO wins: profit = (1.00 - no_price) * contracts
                    pnl = (1.0 - no_price) * trade.contracts
                else:
                    # NO loses: loss = -no_price * contracts
                    pnl = -no_price * trade.contracts

            now = datetime.utcnow().isoformat()
            trade.status = "settled"
            trade.settled_at = now
            trade.settlement_value = settlement_value
            trade.pnl = pnl
            trade.updated_at = now

            session.commit()

            self._bankroll += pnl + trade.total_cost  # Return cost + profit/loss

            logger.info(
                f"SETTLED: {trade.market_ticker} -> "
                f"{'YES' if settlement_value == 100 else 'NO'} "
                f"PnL=${pnl:+.2f}"
            )

            return pnl

        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_open_positions(self) -> list[Trade]:
        """Get all open paper trades."""
        session = get_session()
        try:
            return (
                session.query(Trade)
                .filter_by(mode="paper", status="filled")
                .all()
            )
        finally:
            session.close()

    @property
    def bankroll(self) -> float:
        return self._bankroll
