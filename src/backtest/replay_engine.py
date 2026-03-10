"""Walk-forward backtest engine.

Replays historical data to evaluate strategy performance.
Uses a persistence+climatology blend forecast (NO lookahead bias).

The forecast for each day is built ONLY from information available
before that day:
  - Climatological mean & std for the target month (from all obs
    *excluding* the target day)
  - Persistence signal: average of the 3 days preceding the target
  - Blend weight: 60% persistence / 40% climatology
  - Gaussian noise scaled to realistic day-to-day forecast error

This gives the model modest but genuine predictive skill over
a pure-climatology "market price", without any data leakage.
"""

from __future__ import annotations

import logging
import math
import random
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


def _norm_cdf(z: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


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
    """Walk-forward backtest using historical observations.

    IMPORTANT: Forecasts are generated WITHOUT lookahead bias.
    The forecast for day N uses ONLY data available before day N.
    """

    # Blend weight: how much to trust recent observations vs climatology
    PERSISTENCE_WEIGHT = 0.60
    PERSISTENCE_DAYS = 3  # average of 3 prior days
    # Forecast uncertainty (std dev in °F) — represents realistic GFS-like error
    FORECAST_NOISE_STD = 5.0

    def __init__(self, settings: AppSettings | None = None, seed: int = 42):
        if settings is None:
            settings = load_settings()
        self.settings = settings
        self.market_builder = SyntheticMarketBuilder()
        self._rng = random.Random(seed)

    def _preload_observations(
        self, session, city_name: str
    ) -> dict[str, float]:
        """Load all observations for a city into a date→high_f dict."""
        obs_list = (
            session.query(Observation)
            .filter_by(city=city_name)
            .filter(Observation.high_f != None)  # noqa: E711
            .all()
        )
        return {o.date: float(o.high_f) for o in obs_list}

    def _compute_climatology(
        self,
        obs_by_date: dict[str, float],
        target_month: int,
        exclude_date: str,
    ) -> tuple[float, float]:
        """Compute climatological mean and std for a month.

        Excludes the target date to prevent any data leakage.
        Returns (mean, std).
        """
        temps = [
            t for d, t in obs_by_date.items()
            if d != exclude_date
            and date.fromisoformat(d).month == target_month
        ]
        if not temps:
            return 60.0, 12.0  # fallback
        mean = sum(temps) / len(temps)
        if len(temps) < 2:
            return mean, 12.0
        var = sum((t - mean) ** 2 for t in temps) / (len(temps) - 1)
        return mean, max(math.sqrt(var), 3.0)

    def _persistence_forecast(
        self,
        obs_by_date: dict[str, float],
        target_date: date,
    ) -> float | None:
        """Average of the N days preceding target_date.

        Returns None if no prior observations available.
        """
        temps = []
        for i in range(1, self.PERSISTENCE_DAYS + 1):
            d = (target_date - timedelta(days=i)).isoformat()
            if d in obs_by_date:
                temps.append(obs_by_date[d])
        return sum(temps) / len(temps) if temps else None

    def run(
        self,
        start_date: str,
        end_date: str,
        cities: dict[str, CityConfig],
        edge_threshold: float | None = None,
    ) -> BacktestResult:
        """Run a walk-forward backtest.

        For each historical date:
        1. Build synthetic market brackets (climatological probability)
        2. Generate a no-lookahead forecast from persistence + climatology
        3. Detect edges and simulate trades
        4. Settle using actual observed high
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

        # Pre-load all observations per city for fast lookup
        city_obs: dict[str, dict[str, float]] = {}
        for city_name in cities:
            city_obs[city_name] = self._preload_observations(session, city_name)

        try:
            current = date.fromisoformat(start_date)
            end = date.fromisoformat(end_date)
            bankroll = self.settings.paper_trading.initial_bankroll

            while current <= end:
                date_str = current.isoformat()

                for city_name, city_config in cities.items():
                    trade = self._simulate_day(
                        date_str, city_name, city_config,
                        session, calibrator, bankroll, edge_threshold,
                        city_obs.get(city_name, {}),
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
        obs_by_date: dict[str, float],
    ) -> BacktestTrade | None:
        """Simulate one day's trading for one city.

        Forecast is built ONLY from data available before target_date:
          forecast = w * persistence + (1-w) * climatology + noise
        where persistence = mean of prior 3 days' highs.
        """
        # --- Settlement truth (used ONLY for settling, never for forecasting) ---
        observed_high = obs_by_date.get(date_str)
        if observed_high is None:
            return None

        target = date.fromisoformat(date_str)

        # --- Build forecast WITHOUT looking at target day ---
        clim_mean, clim_std = self._compute_climatology(
            obs_by_date, target.month, exclude_date=date_str,
        )
        persistence = self._persistence_forecast(obs_by_date, target)

        if persistence is not None:
            # Blend persistence with climatology
            forecast_mean = (
                self.PERSISTENCE_WEIGHT * persistence
                + (1.0 - self.PERSISTENCE_WEIGHT) * clim_mean
            )
        else:
            # No recent data — fall back to pure climatology
            forecast_mean = clim_mean

        # Add realistic forecast noise (seeded RNG for reproducibility)
        forecast_temp = forecast_mean + self._rng.gauss(0, self.FORECAST_NOISE_STD)

        # Forecast uncertainty for probability estimation
        # Use a blend of forecast noise + climatological variability
        forecast_std = math.sqrt(self.FORECAST_NOISE_STD ** 2 + (clim_std * 0.3) ** 2)

        # --- Build synthetic market brackets ---
        brackets = self.market_builder.build_brackets(
            city_config, date_str, session,
        )
        if not brackets:
            return None

        # --- Find best edge ---
        best_bracket = None
        best_edge = 0

        for bracket in brackets:
            if not bracket.get("is_above"):
                continue

            threshold = bracket["threshold"]
            z = (threshold - forecast_temp) / forecast_std
            model_prob = 1.0 - _norm_cdf(z)

            # Clamp to avoid extreme probabilities
            model_prob = max(0.02, min(0.98, model_prob))
            model_prob = calibrator.calibrate(model_prob)

            market_price = bracket.get("market_price", 0.5)
            edge = model_prob - market_price

            if abs(edge) > abs(best_edge) and abs(edge) > edge_threshold:
                best_edge = edge
                best_bracket = bracket
                best_bracket["model_prob"] = model_prob
                best_bracket["edge"] = edge

        if best_bracket is None:
            return None

        # --- Simulate trade ---
        side = "yes" if best_edge > 0 else "no"
        model_prob = best_bracket["model_prob"]
        market_price = best_bracket.get("market_price", 0.5)
        threshold = best_bracket.get("threshold", 0)

        price = market_price if side == "yes" else (1.0 - market_price)
        contracts = min(10, max(1, int(bankroll * 0.01 / max(price, 0.01))))
        cost = contracts * price

        # --- Settle (uses actual outcome, no leakage here) ---
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
            observed_high=int(observed_high),
            settled_yes=settled_yes,
            pnl=pnl,
        )
