"""Live trading execution via Kalshi API.

Submits real orders through the authenticated Kalshi API.
Only used when MODE=live.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

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
        If the order fills immediately, records actual fill price
        (which may differ from the limit price).
        """
        if size.contracts <= 0:
            return None

        now = datetime.now(timezone.utc).isoformat()
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

            # Map Kalshi status to internal status on initial write
            initial_status = self._map_status(order.status)

            # Default to limit price; overwrite with actual fill if executed
            entry_price = size.price_cents / 100.0
            fill_price = None
            total_cost = size.total_cost
            fill_contracts = size.contracts

            # If order executed immediately, fetch actual fill data
            if initial_status == "filled" and order.order_id:
                try:
                    order_data = self.client.get_order(order.order_id)
                    fill_count = order_data.get("fill_count") or size.contracts
                    taker_cost = order_data.get("taker_fill_cost")
                    maker_cost = order_data.get("maker_fill_cost") or 0
                    if taker_cost is not None:
                        total_fill_cents = (taker_cost or 0) + maker_cost
                        fill_price = total_fill_cents / (fill_count * 100) if fill_count else entry_price
                        total_cost = total_fill_cents / 100
                        fill_contracts = fill_count
                        entry_price = fill_price  # Use actual fill as entry
                        logger.info(
                            f"Immediate fill: {fill_count}x @ {fill_price*100:.1f}c "
                            f"(limit was {size.price_cents}c)"
                        )
                except Exception as e:
                    logger.warning(f"Could not fetch fill data for {order.order_id}: {e}")

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
                    contracts=fill_contracts,
                    price=entry_price,
                    fill_price=fill_price,
                    total_cost=total_cost,
                    model_prob=signal.model_prob,
                    market_price=signal.market_price,
                    edge=signal.edge,
                    kelly_fraction=size.kelly_fraction,
                    kalshi_order_id=order.order_id,
                    status=initial_status,
                    created_at=now,
                    updated_at=now,
                )
                session.add(trade)
                session.commit()

                logger.info(
                    f"LIVE ORDER: {signal.side.upper()} {fill_contracts}x "
                    f"{signal.market_ticker} @ {entry_price*100:.1f}c "
                    f"(order_id={order.order_id}, status={initial_status})"
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

    @staticmethod
    def _map_status(api_status: str) -> str:
        """Map Kalshi API status to internal status."""
        if api_status in ("resting",):
            return "pending"
        elif api_status in ("executed", "filled"):
            return "filled"
        elif api_status in ("canceled", "cancelled"):
            return "cancelled"
        return "pending"

    def check_order_status(self, trade: Trade) -> tuple[str, dict]:
        """Check order status and return (mapped_status, raw_order_data)."""
        if not trade.kalshi_order_id:
            return trade.status, {}

        try:
            order_data = self.client.get_order(trade.kalshi_order_id)
            api_status = order_data.get("status", "")
            return self._map_status(api_status), order_data
        except Exception as e:
            logger.warning(
                f"Failed to check order {trade.kalshi_order_id}: {e}"
            )
            return trade.status, {}

    def sync_open_orders(self) -> int:
        """Sync all non-terminal live orders with Kalshi API.

        Also back-fills fill_price for filled orders that are missing it
        (e.g. orders that executed immediately before the fill-price fix).

        Returns number of orders whose status changed or were updated.
        """
        session = get_session()
        updated = 0
        try:
            # 1. Sync pending/resting orders
            open_trades = (
                session.query(Trade)
                .filter_by(mode="live")
                .filter(Trade.status.notin_(["filled", "settled", "cancelled"]))
                .all()
            )

            for trade in open_trades:
                new_status, order_data = self.check_order_status(trade)
                if new_status != trade.status:
                    now = datetime.now(timezone.utc).isoformat()
                    trade.status = new_status
                    trade.updated_at = now

                    if new_status == "filled" and order_data:
                        self._apply_fill_data(trade, order_data)

                    updated += 1
                    logger.info(
                        f"Order status update: {trade.market_ticker} "
                        f"{trade.kalshi_order_id} -> {new_status}"
                    )

            # 2. Back-fill fill_price for already-filled orders missing it
            missing_fill = (
                session.query(Trade)
                .filter_by(mode="live", status="filled")
                .filter(Trade.kalshi_order_id.isnot(None))
                .filter(Trade.fill_price.is_(None))
                .all()
            )

            for trade in missing_fill:
                _, order_data = self.check_order_status(trade)
                if order_data:
                    self._apply_fill_data(trade, order_data)
                    trade.updated_at = datetime.now(timezone.utc).isoformat()
                    updated += 1
                    logger.info(
                        f"Back-filled price for {trade.market_ticker}: "
                        f"limit={trade.price:.4f} -> fill={trade.fill_price:.4f}"
                    )

            if updated:
                session.commit()
            logger.info(
                f"Order sync: {len(open_trades)} pending checked, "
                f"{len(missing_fill)} fills back-filled, {updated} total updated"
            )
            return updated

        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def _apply_fill_data(trade: Trade, order_data: dict):
        """Extract actual fill price and cost from Kalshi order data."""
        fill_count = order_data.get("fill_count") or trade.contracts
        taker_cost = order_data.get("taker_fill_cost") or 0
        maker_cost = order_data.get("maker_fill_cost") or 0
        total_fill_cents = taker_cost + maker_cost

        trade.contracts = fill_count
        if total_fill_cents > 0 and fill_count > 0:
            trade.fill_price = total_fill_cents / (fill_count * 100)
            trade.total_cost = total_fill_cents / 100
            trade.price = trade.fill_price  # Update entry price to actual fill
        else:
            trade.fill_price = trade.price
