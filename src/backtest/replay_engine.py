"""Walk-forward backtest engine.

Replays historical data to evaluate strategy performance.
Uses historical observed temperatures and reconstructed
ensemble probabilities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

from src.config.cities import CityConfig
from src.config.settings import AppSettings, load_settings
from src.data.db import get_session
from src.data.models import Observation
from src.signals.calibration import ForecastCalibrator, BrierDecomposition
from src.backtest.synthetic_markets import SyntheticMarketBuilder
from src.backtest.performance_report import PerformanceReport, compute_performance

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    date: str
    city: str
    side: str
    threshold: float
    model_prob: float
    market_price: float
    edge: float
    contracts: int
    price: float
    cost: float
    observed_high: int | None = None
    settled_yes: bool | None = None
    pnl: float = 0.0


@dataclass
class BacktestResult:
    start_date: str
    end_date: str
    cities: list[str]
    trades: list[BacktestTrade] = field(default_factory=list)
    daily_pnl: dict[str, float] = field(default_factory=dict)
    performance: PerformanceReport | None = None
    brier: BrierDecomposition | None = None


class BacktestEngine:
    """Walk-forward backtest using historical observations."""

    def __init__(self, settings: AppSettings | None = None):
        if settings is None:
            settings = load_settings()
        self.settings = settings
        self.market_builder = SyntheticMarketBuilder()

    def run(
        self,
        start_date: str,
        end_date: str,
        cities: dict[str, CityConfig],
        edge_threshold: float | None = None,
    ) -> BacktestResult:
        """Run a walk-forward backtest.

        For each historical date:
        1. Build synthetic market brackets around climatological mean
        2. Use actual observed temperatures from neighboring days
           to estimate what ensemble probabilities would have been
        3. Detect edges and simulate trades
        4. Settle using actual observed high

        Args:
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            cities: City configurations to backtest
            edge_threshold: Override edge threshold (default from settings)
        """
        if edge_threshold is None:
            edge_threshold = self.settings.strategy.edge_threshold

        result = BacktestResult(
            start_date=start_date,
            end_date=end_date,
            cities=list(cities.keys()),
        )

        session = get_session()
        calibrator = ForecastCalibrator()
        all_forecasts = []
        all_outcomes = []

        try:
            current = date.fromisoformat(start_date)
            end = date.fromisoformat(end_date)
            bankroll = self.settings.paper_trading.initial_bankroll

            while current <= end:
                date_str = current.isoformat()

                for city_name, city_config in cities.items():
                    trade = self._simulate_day(
                        date_str, city_name, city_config,
                        session, calibrator, bankroll, edge_threshold
                    )
                    if trade and trade.observed_high is not None:
                        result.trades.append(trade)
                        bankroll += trade.pnl
                        result.daily_pnl[date_str] = (
                            result.daily_pnl.get(date_str, 0) + trade.pnl
                        )

                        # Track for calibration
                        all_forecasts.append(trade.model_prob)
                        outcome = 1 if trade.settled_yes else 0
                        all_outcomes.append(outcome)

                        # Refit calibrator periodically (walk-forward)
                        if len(all_forecasts) >= 100 and len(all_forecasts) % 50 == 0:
                            calibrator.fit(all_forecasts, all_outcomes)

                current += timedelta(days=1)

        finally:
            session.close()

        # Compute performance metrics
        if result.trades:
            result.performance = compute_performance(result.trades, self.settings)
            if all_forecasts:
                result.brier = ForecastCalibrator.compute_brier_decomposition(
                    all_forecasts, all_outcomes
                )

        return result

    def _simulate_day(
        self,
        date_str: str,
        city_name: str,
        city_config: CityConfig,
        session,
        calibrator: ForecastCalibrator,
        bankroll: float,
        edge_threshold: float,
    ) -> BacktestTrade | None:
        """Simulate one day's trading for one city."""
        # Get actual observed high for this date
        obs = (
            session.query(Observation)
            .filter_by(city=city_name, date=date_str)
            .first()
        )
        if obs is None or obs.high_f is None:
            return None

        observed_high = obs.high_f

        # Build synthetic market brackets
        brackets = self.market_builder.build_brackets(
            city_config, date_str, session
        )
        if not brackets:
            return None

        # Use a simple proxy for ensemble probability:
        # Based on historical error distribution around the observed high
        # (In production, we'd use actual archived ensemble forecasts)
        # For backtesting, simulate what the forecast would have been
        # using a Gaussian centered near the observed value with noise
        import random
        forecast_temp = observed_high + random.gauss(0, 3.0)  # ~3F std error

        # Pick the most interesting bracket (closest to forecast)
        best_bracket = None
        best_edge = 0

        for bracket in brackets:
            # Estimate ensemble probability using Gaussian
            from src.signals.model_blender import ModelBlender, _norm_cdf
            if bracket.get("is_above"):
                threshold = bracket["threshold"]
                z = (threshold - forecast_temp) / 3.0
                model_prob = 1.0 - _norm_cdf(z)
            else:
                continue  # Focus on above contracts for simplicity

            market_price = bracket.get("market_price", 0.5)
            model_prob = calibrator.calibrate(model_prob)
            edge = model_prob - market_price

            if abs(edge) > abs(best_edge) and abs(edge) > edge_threshold:
                best_edge = edge
                best_bracket = bracket
                best_bracket["model_prob"] = model_prob
                best_bracket["edge"] = edge

        if best_bracket is None:
            return None

        # Simulate trade
        side = "yes" if best_edge > 0 else "no"
        model_prob = best_bracket["model_prob"]
        market_price = best_bracket.get("market_price", 0.5)
        threshold = best_bracket.get("threshold", 0)

        price = market_price if side == "yes" else (1.0 - market_price)
        contracts = min(10, max(1, int(bankroll * 0.01 / max(price, 0.01))))
        cost = contracts * price

        # Settle
        settled_yes = observed_high > threshold
        if side == "yes":
            pnl = ((1.0 - price) * contracts) if settled_yes else (-price * contracts)
        else:
            pnl = ((1.0 - price) * contracts) if not settled_yes else (-price * contracts)

        return BacktestTrade(
            date=date_str,
            city=city_name,
            side=side,
            threshold=threshold,
            model_prob=model_prob,
            market_price=market_price,
            edge=best_edge,
            contracts=contracts,
            price=price,
            cost=cost,
            observed_high=observed_high,
            settled_yes=settled_yes,
            pnl=pnl,
        )
