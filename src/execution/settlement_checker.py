"""Settlement checker: verifies outcomes using IEM CLI data.

Runs daily (11 AM ET) to:
1. Fetch yesterday's CLI reports for all cities with open positions
2. Determine if each trade settled YES or NO
3. Compute and record PnL
4. Update Brier scores
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from src.config.cities import CityConfig
from src.data.db import get_session
from src.data.iem_client import IEMClient
from src.data.models import Trade, BrierScore, Observation
from src.execution.paper_trader import PaperTrader
from src.utils.temperature import settles_above, settles_in_bracket

logger = logging.getLogger(__name__)


@dataclass
class SettlementResult:
    trade_id: str
    market_ticker: str
    city: str
    observed_high: int
    settled_yes: bool
    pnl: float


class SettlementChecker:
    """Checks settlements using IEM CLI data."""

    def __init__(
        self,
        cities: dict[str, CityConfig],
        paper_trader: PaperTrader | None = None,
    ):
        self.cities = cities
        self.iem = IEMClient()
        self.paper_trader = paper_trader

    def check_settlements(
        self, target_date: str | None = None
    ) -> list[SettlementResult]:
        """Check and settle all trades for a target date.

        Args:
            target_date: Date to check (default: yesterday)

        Returns:
            List of settlement results
        """
        if target_date is None:
            target_date = (date.today() - timedelta(days=1)).isoformat()

        session = get_session()
        results = []

        try:
            # Get open trades for this target date
            open_trades = (
                session.query(Trade)
                .filter_by(target_date=target_date, status="filled")
                .all()
            )

            if not open_trades:
                logger.info(f"No open trades for {target_date}")
                return []

            # Group by city to minimize API calls
            cities_needed = set(t.city for t in open_trades)

            # Fetch CLI data for each city
            cli_data: dict[str, int | None] = {}
            for city_name in cities_needed:
                city = self.cities.get(city_name)
                if city is None:
                    logger.error(f"Unknown city: {city_name}")
                    continue

                report = self.iem.get_cli(city.station, target_date)
                if report and report.high_f is not None:
                    cli_data[city_name] = report.high_f
                    # Store observation
                    self.iem.store_observations([report], city_name)
                else:
                    logger.warning(
                        f"No CLI data for {city_name} on {target_date}"
                    )

            # Settle each trade
            for trade in open_trades:
                observed_high = cli_data.get(trade.city)
                if observed_high is None:
                    logger.warning(
                        f"Cannot settle {trade.market_ticker}: "
                        f"no CLI data for {trade.city}"
                    )
                    continue

                result = self._settle_trade(trade, observed_high, session)
                if result:
                    results.append(result)

            session.commit()

        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        # Log summary
        if results:
            total_pnl = sum(r.pnl for r in results)
            wins = sum(1 for r in results if r.pnl > 0)
            losses = sum(1 for r in results if r.pnl < 0)
            logger.info(
                f"Settlement for {target_date}: {len(results)} trades, "
                f"{wins}W/{losses}L, PnL=${total_pnl:+.2f}"
            )

        return results

    def _settle_trade(
        self, trade: Trade, observed_high: int, session
    ) -> SettlementResult | None:
        """Settle a single trade based on the observed high."""
        # Determine if the contract settled YES
        # We need to look up the market details
        from src.data.models import KalshiMarket

        market = (
            session.query(KalshiMarket)
            .filter_by(market_ticker=trade.market_ticker)
            .first()
        )

        if market is None:
            logger.warning(f"Market not found: {trade.market_ticker}")
            return None

        if market.is_above_contract and market.threshold_f is not None:
            settled_yes = settles_above(observed_high, market.threshold_f)
        else:
            settled_yes = settles_in_bracket(
                observed_high, market.bracket_low, market.bracket_high
            )

        settlement_value = 100 if settled_yes else 0

        # Use paper trader to settle if available
        if self.paper_trader and trade.mode == "paper":
            pnl = self.paper_trader.settle_position(
                trade.trade_id, settlement_value
            )
        else:
            # Manual settlement
            if trade.side == "yes":
                if settled_yes:
                    pnl = (1.0 - trade.price) * trade.contracts
                else:
                    pnl = -trade.price * trade.contracts
            else:
                if not settled_yes:
                    pnl = (1.0 - trade.price) * trade.contracts
                else:
                    pnl = -trade.price * trade.contracts

            now = datetime.now(timezone.utc).isoformat()
            trade.status = "settled"
            trade.settled_at = now
            trade.settlement_value = settlement_value
            trade.pnl = pnl
            trade.updated_at = now

        # Record Brier score
        if trade.model_prob is not None:
            outcome = 1 if settled_yes else 0
            brier_contribution = (trade.model_prob - outcome) ** 2

            brier = BrierScore(
                city=trade.city,
                target_date=trade.target_date,
                market_ticker=trade.market_ticker,
                forecast_prob=trade.model_prob,
                outcome=outcome,
                brier_contribution=brier_contribution,
                lead_hours=None,
                model_source="calibrated",
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            session.add(brier)

        logger.info(
            f"Settled {trade.market_ticker}: observed={observed_high}F, "
            f"{'YES' if settled_yes else 'NO'}, PnL=${pnl:+.2f}"
        )

        return SettlementResult(
            trade_id=trade.trade_id,
            market_ticker=trade.market_ticker,
            city=trade.city,
            observed_high=observed_high,
            settled_yes=settled_yes,
            pnl=pnl,
        )

    def close(self):
        self.iem.close()
