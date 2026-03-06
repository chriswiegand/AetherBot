"""Live trading execution via Kalshi API.

Submits real orders through the authenticated Kalshi API.
Only used when MODE=live.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

from src.data.db import get_session
from src.data.kalshi_client import KalshiClient
from src.data.models import Trade
from src.strategy.edge_detector import TradeSignal
from src.strategy.kelly_sizer import PositionSize

logger = logging.getLogger(__name__)


class LiveTrader:
    """Real Kalshi order execution."""

    def __init__(self, kalshi_client: KalshiClient):
        self.client = kalshi_client

    def execute_trade(
        self, signal: TradeSignal, size: PositionSize
    ) -> Trade | None:
        """Place a real order on Kalshi.

        Uses limit orders at the current bid price.
        """
        if size.contracts <= 0:
            return None

        now = datetime.utcnow().isoformat()
        trade_id = str(uuid.uuid4())

        try:
            order = self.client.create_order(
                ticker=signal.market_ticker,
                side=signal.side,
                action="buy",
                order_type="limit",
                yes_price=size.price_cents if signal.side == "yes" else None,
                no_price=size.price_cents if signal.side == "no" else None,
                count=size.contracts,
            )

            session = get_session()
            try:
                trade = Trade(
                    trade_id=trade_id,
                    mode="live",
                    market_ticker=signal.market_ticker,
                    city=signal.city,
                    target_date=signal.target_date,
                    side=signal.side,
                    direction="buy",
                    contracts=size.contracts,
                    price=size.price_cents / 100.0,
                    total_cost=size.total_cost,
                    model_prob=signal.model_prob,
                    market_price=signal.market_price,
                    edge=signal.edge,
                    kelly_fraction=size.kelly_fraction,
                    kalshi_order_id=order.order_id,
                    status=order.status or "pending",
                    created_at=now,
                    updated_at=now,
                )
                session.add(trade)
                session.commit()

                logger.info(
                    f"LIVE ORDER: {signal.side.upper()} {size.contracts}x "
                    f"{signal.market_ticker} @ {size.price_cents}c "
                    f"(order_id={order.order_id})"
                )

                return trade

            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

        except Exception as e:
            logger.error(f"Live order failed for {signal.market_ticker}: {e}")
            return None

    def check_order_status(self, trade: Trade) -> str:
        """Check if a pending order has been filled."""
        if not trade.kalshi_order_id:
            return trade.status

        # TODO: Implement order status check via Kalshi API
        # For now, return current status
        return trade.status
