"""Core job definitions for the scheduler.

Each job corresponds to a step in the bot's operational cycle.
"""

from __future__ import annotations

import logging
from datetime import datetime, date, timezone
from src.utils.time_utils import compute_lead_hours, parse_iso_datetime

from src.config.cities import CityConfig, load_cities
from src.config.settings import AppSettings, load_settings
from src.data.db import init_db, get_session
from src.data.ensemble_fetcher import EnsembleFetcher
from src.data.hrrr_fetcher import HRRRFetcher
from src.data.nws_client import NWSClient
from src.data.kalshi_client import KalshiClient
from src.data.kalshi_markets import KalshiMarketDiscovery, ParsedMarket
from src.data.models import Signal, KalshiMarket
from src.signals.ensemble_probability import EnsembleProbabilityCalculator
from src.signals.hrrr_correction import HRRRCorrector
from src.signals.model_blender import ModelBlender
from src.signals.calibration import ForecastCalibrator
from src.strategy.edge_detector import EdgeDetector, TradeSignal
from src.strategy.kelly_sizer import KellySizer
from src.strategy.risk_manager import RiskManager
from src.execution.paper_trader import PaperTrader
from src.monitoring.alerting import AlertManager

logger = logging.getLogger(__name__)


class BotContext:
    """Shared context for all bot jobs."""

    def __init__(self, settings: AppSettings | None = None):
        if settings is None:
            settings = load_settings()
        self.settings = settings
        self.cities = load_cities()

        # Data clients
        self.ensemble_fetcher = EnsembleFetcher(settings)
        self.hrrr_fetcher = HRRRFetcher(settings)
        self.nws_client = NWSClient(settings)
        self.kalshi_client = KalshiClient(settings)

        # Market discovery
        self.market_discovery = KalshiMarketDiscovery(self.kalshi_client)

        # Signal engine
        self.ensemble_calc = EnsembleProbabilityCalculator()
        self.hrrr_corrector = HRRRCorrector(settings.model_weights)
        self.model_blender = ModelBlender(settings.model_weights)
        self.calibrator = ForecastCalibrator()

        # Strategy
        self.edge_detector = EdgeDetector(settings.strategy)
        self.kelly_sizer = KellySizer(settings.strategy)
        self.risk_manager = RiskManager(settings.strategy)

        # Execution
        self.paper_trader = PaperTrader(settings)

        # Monitoring
        self.alerts = AlertManager()

        # State
        self._active_markets: list[ParsedMarket] = []

    def shutdown(self):
        """Clean up resources."""
        self.ensemble_fetcher.close()
        self.hrrr_fetcher.close()
        self.nws_client.close()
        self.kalshi_client.close()


def fetch_all_ensembles(ctx: BotContext):
    """Fetch GFS ensemble data for all cities."""
    for city_name, city_config in ctx.cities.items():
        try:
            ctx.ensemble_fetcher.fetch_and_store(city_config)
        except Exception as e:
            logger.error(f"Ensemble fetch failed for {city_name}: {e}")
            ctx.alerts.warning(f"Ensemble fetch failed: {city_name}", "data")


def fetch_all_hrrr(ctx: BotContext):
    """Fetch HRRR data for all cities."""
    for city_name, city_config in ctx.cities.items():
        try:
            ctx.hrrr_fetcher.fetch_and_store(city_config)
        except Exception as e:
            logger.error(f"HRRR fetch failed for {city_name}: {e}")


def fetch_all_nws(ctx: BotContext):
    """Fetch NWS forecasts for all cities."""
    for city_name, city_config in ctx.cities.items():
        try:
            ctx.nws_client.fetch_and_store(city_config)
        except Exception as e:
            logger.error(f"NWS fetch failed for {city_name}: {e}")


def discover_markets(ctx: BotContext):
    """Discover active KXHIGH markets."""
    try:
        markets = ctx.market_discovery.discover_active_markets(ctx.cities)
        ctx.market_discovery.store_markets(markets)
        ctx._active_markets = markets
        logger.info(f"Discovered {len(markets)} active markets")
    except Exception as e:
        logger.error(f"Market discovery failed: {e}")
        ctx.alerts.warning(f"Market discovery failed: {e}", "kalshi")


def _load_active_strategy(ctx: BotContext) -> int | None:
    """Load active strategy from DB and override strategy config.

    Returns the strategy_id if one is active, else None.
    """
    import sqlite3
    db_path = ctx.settings.database.absolute_path
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='strategies'"
        )
        if not cur.fetchone():
            conn.close()
            return None
        cur.execute("SELECT * FROM strategies WHERE is_active = 1 LIMIT 1")
        row = cur.fetchone()
        conn.close()
        if row is None:
            return None
        strat = dict(row)

        field_map = {
            "edge_threshold": float, "min_edge_hrrr_confirm": float,
            "min_model_prob": float, "fractional_kelly": float,
            "max_position_pct": float, "max_position_dollars": float,
            "daily_loss_limit": float, "max_concurrent_positions": int,
            "max_positions_per_city": int, "max_positions_per_date": int,
            "min_price": float, "max_price": float, "max_lead_hours": float,
        }
        for field, cast in field_map.items():
            val = strat.get(field)
            if val is not None:
                setattr(ctx.settings.strategy, field, cast(val))

        ctx.edge_detector = EdgeDetector(ctx.settings.strategy)
        ctx.kelly_sizer = KellySizer(ctx.settings.strategy)
        ctx.risk_manager = RiskManager(ctx.settings.strategy)

        logger.info(f"Using active strategy: {strat.get('name')} (id={strat['id']})")
        return strat["id"]
    except Exception as e:
        logger.warning(f"Could not load active strategy: {e}")
        return None


def scan_and_trade(ctx: BotContext):
    """Core trading cycle: signal -> edge -> size -> trade."""
    if not ctx._active_markets:
        discover_markets(ctx)

    if not ctx._active_markets:
        logger.info("No active markets to trade")
        return

    # Load active strategy (if any) to override config
    active_strategy_id = _load_active_strategy(ctx)

    # Refresh market prices
    ctx._active_markets = ctx.market_discovery.refresh_prices(ctx._active_markets)

    # Group markets by city and target date
    by_city_date: dict[tuple[str, str], list[ParsedMarket]] = {}
    for m in ctx._active_markets:
        key = (m.city, m.target_date)
        by_city_date.setdefault(key, []).append(m)

    # Get portfolio state
    portfolio = ctx.risk_manager.get_portfolio_state(ctx.paper_trader.bankroll)

    for (city_name, target_date), markets in by_city_date.items():
        city_config = ctx.cities.get(city_name)
        if city_config is None:
            continue

        # Get latest ensemble data
        ensemble_results = ctx.ensemble_fetcher.fetch_ensemble(city_config, forecast_days=3)
        target_ensemble = None
        for er in ensemble_results:
            if er.target_date == target_date:
                target_ensemble = er
                break

        if target_ensemble is None:
            continue

        # Compute actual lead hours
        target_date_obj = date.fromisoformat(target_date)
        model_run_dt = parse_iso_datetime(target_ensemble.model_run_time)
        lead_hours = compute_lead_hours(model_run_dt, target_date_obj, city_config.timezone)

        # Calculate ensemble probabilities
        probs = ctx.ensemble_calc.get_full_distribution(
            target_ensemble.member_daily_maxes, markets
        )

        # Build signal dict and market dict for edge detection
        signals: dict[str, float] = {}
        market_data: dict[str, dict] = {}

        for m in markets:
            prob_result = probs.get(m.market_ticker)
            if prob_result is None:
                continue

            calibrated = ctx.calibrator.calibrate(prob_result.probability)
            signals[m.market_ticker] = calibrated

            market_data[m.market_ticker] = {
                "yes_price": m.yes_price,
                "city": m.city,
                "target_date": m.target_date,
            }

        # Detect edges
        trade_signals = ctx.edge_detector.scan_for_edges(signals, market_data, lead_hours=lead_hours)

        # Execute trades
        now = datetime.now(timezone.utc).isoformat()
        for signal in trade_signals:
            # Check risk
            size = ctx.kelly_sizer.calculate_position_size(signal, portfolio.bankroll)
            if size.contracts <= 0:
                continue

            risk_check = ctx.risk_manager.check_trade_allowed(
                signal.city, signal.target_date, size.total_cost, portfolio
            )
            if not risk_check.allowed:
                logger.info(f"Risk blocked: {risk_check.reason}")
                continue

            # Execute
            trade = ctx.paper_trader.execute_trade(signal, size)
            if trade:
                portfolio.open_positions += 1
                portfolio.bankroll -= size.total_cost

                # Store signal
                session = get_session()
                try:
                    sig = Signal(
                        city=signal.city,
                        target_date=signal.target_date,
                        market_ticker=signal.market_ticker,
                        computed_at=now,
                        ensemble_prob=probs[signal.market_ticker].probability,
                        calibrated_prob=signals[signal.market_ticker],
                        market_yes_price=signal.market_price,
                        raw_edge=signal.edge,
                        abs_edge=signal.abs_edge,
                        lead_hours=signal.lead_hours,
                    )
                    session.add(sig)
                    session.commit()
                except Exception:
                    session.rollback()
                finally:
                    session.close()
