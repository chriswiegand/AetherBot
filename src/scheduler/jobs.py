"""Core job definitions for the scheduler.

Each job corresponds to a step in the bot's operational cycle.
"""

from __future__ import annotations

import logging
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

from src.utils.time_utils import compute_lead_hours, parse_iso_datetime

from src.config.cities import CityConfig, load_cities
from src.config.settings import AppSettings, load_settings
from src.data.db import init_db, get_session
from src.data.ensemble_fetcher import EnsembleFetcher, EnsembleResult
from src.data.ecmwf_fetcher import ECMWFFetcher
from src.data.hrrr_fetcher import HRRRFetcher
from src.data.nws_client import NWSClient
from src.data.kalshi_client import KalshiClient
from src.data.kalshi_markets import KalshiMarketDiscovery, ParsedMarket
from src.data.models import (
    Signal, KalshiMarket, EnsembleForecast, ECMWFForecast,
    HRRRForecast, NWSForecast, MarketPriceHistory,
)
from src.data.freshness import DataFreshnessTracker
from src.signals.ensemble_probability import EnsembleProbabilityCalculator
from src.signals.hrrr_correction import HRRRCorrector
from src.signals.model_blender import ModelBlender
from src.signals.calibration import ForecastCalibrator
from src.strategy.edge_detector import EdgeDetector, TradeSignal
from src.strategy.kelly_sizer import KellySizer
from src.strategy.risk_manager import RiskManager
from src.execution.paper_trader import PaperTrader
from src.execution.live_trader import LiveTrader
from src.monitoring.alerting import AlertManager
from src.monitoring.calibrator_trainer import CalibratorTrainer
from src.signals.adaptive_weights import AdaptiveWeightManager

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
        self.ecmwf_fetcher = ECMWFFetcher(settings)
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

        # Execution — select trader based on mode
        if settings.mode == "live":
            self.trader = LiveTrader(self.kalshi_client)
            self.paper_trader = None  # Not used in live mode
        else:
            self.trader = PaperTrader(settings)
            self.paper_trader = self.trader  # Backwards compat

        # Monitoring
        self.alerts = AlertManager()

        # Calibrator training — load from disk if available
        self.calibrator_trainer = CalibratorTrainer(self.calibrator)
        self.calibrator_trainer.load_if_exists()

        # Adaptive blend weights — load from disk if available
        self.adaptive_weight_mgr = AdaptiveWeightManager()
        try:
            adaptive = self.adaptive_weight_mgr.load()
            if adaptive:
                self.model_blender.set_adaptive_weights(adaptive)
        except Exception as e:
            logger.warning(f"Failed to load adaptive weights (non-fatal): {e}")

        # State
        self._active_markets: list[ParsedMarket] = []

    def shutdown(self):
        """Clean up resources."""
        self.ensemble_fetcher.close()
        self.ecmwf_fetcher.close()
        self.hrrr_fetcher.close()
        self.nws_client.close()
        self.kalshi_client.close()


# ---------------------------------------------------------------------------
# Data fetching jobs (kept for use by smart_data_fetch)
# ---------------------------------------------------------------------------

def fetch_all_ensembles(ctx: BotContext):
    """Fetch GFS ensemble data for all cities."""
    for city_name, city_config in ctx.cities.items():
        try:
            ctx.ensemble_fetcher.fetch_and_store(city_config)
        except Exception as e:
            logger.error(f"Ensemble fetch failed for {city_name}: {e}")
            ctx.alerts.warning(f"Ensemble fetch failed: {city_name}", "data")


def fetch_all_ecmwf(ctx: BotContext):
    """Fetch ECMWF IFS ensemble data for all cities."""
    for city_name, city_config in ctx.cities.items():
        try:
            ctx.ecmwf_fetcher.fetch_and_store(city_config)
        except Exception as e:
            logger.error(f"ECMWF fetch failed for {city_name}: {e}")


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


# ---------------------------------------------------------------------------
# Smart data fetch — replaces fixed-interval fetch jobs
# ---------------------------------------------------------------------------

def _check_manual_refresh_signals(ctx: BotContext) -> set[str]:
    """Check for manual refresh signal files from the dashboard.

    Returns set of sources that need manual refresh (e.g. {'gfs', 'hrrr'}).
    """
    signal_dir = Path(ctx.settings.database.absolute_path).parent / "signals"
    triggered: set[str] = set()
    for source in ["gfs", "hrrr", "nws", "ecmwf"]:
        signal_file = signal_dir / f"refresh_{source}.signal"
        if signal_file.exists():
            try:
                signal_file.unlink()
                triggered.add(source)
                logger.info(f"Manual refresh triggered for {source}")
            except Exception as e:
                logger.warning(f"Could not consume signal file for {source}: {e}")
    return triggered


def smart_data_fetch(ctx: BotContext):
    """Check freshness of each data source and fetch only when new data exists.

    When smart_fetch_enabled is False, unconditionally fetches all sources
    every cycle (matching Kalshi poll frequency for fresh model data).
    """
    # Check for manual refresh signals from dashboard
    manual = _check_manual_refresh_signals(ctx)

    # If smart fetch is disabled, unconditionally fetch everything
    if not ctx.settings.scheduler.smart_fetch_enabled:
        logger.info("Smart fetch disabled — fetching all sources unconditionally")
        fetch_all_ensembles(ctx)
        fetch_all_ecmwf(ctx)
        fetch_all_hrrr(ctx)
        fetch_all_nws(ctx)
        return

    tracker = DataFreshnessTracker(
        ctx.settings.database.absolute_path,
        gfs_lag_hours=ctx.settings.scheduler.gfs_availability_lag_hours,
        hrrr_lag_hours=ctx.settings.scheduler.hrrr_availability_lag_hours,
    )

    new_data = False  # Track whether any source got fresh data

    # GFS Ensemble
    if "gfs" in manual or tracker.should_fetch_gfs():
        reason = "manual refresh" if "gfs" in manual else "new GFS run available"
        logger.info(f"Fetching GFS ensemble ({reason})")
        fetch_all_ensembles(ctx)
        new_data = True
    else:
        logger.debug("GFS ensemble data is fresh — skipping fetch")

    # ECMWF IFS Ensemble
    if "ecmwf" in manual or tracker.should_fetch_ecmwf():
        reason = "manual refresh" if "ecmwf" in manual else "new ECMWF run available"
        logger.info(f"Fetching ECMWF IFS ({reason})")
        fetch_all_ecmwf(ctx)
        new_data = True
    else:
        logger.debug("ECMWF IFS data is fresh — skipping fetch")

    # HRRR
    if "hrrr" in manual or tracker.should_fetch_hrrr():
        reason = "manual refresh" if "hrrr" in manual else "new HRRR run available"
        logger.info(f"Fetching HRRR ({reason})")
        fetch_all_hrrr(ctx)
        new_data = True
    else:
        logger.debug("HRRR data is fresh — skipping fetch")

    # NWS
    if "nws" in manual or tracker.should_fetch_nws():
        reason = "manual refresh" if "nws" in manual else "NWS data stale"
        logger.info(f"Fetching NWS ({reason})")
        fetch_all_nws(ctx)
    else:
        logger.debug("NWS data is fresh — skipping fetch")

    # --- Fast-path: if any model got fresh data, trigger immediate scan ---
    if new_data:
        logger.info("⚡ New model data detected — triggering fast-path scan & trade")
        try:
            scan_and_trade(ctx)
        except Exception as e:
            logger.error(f"Fast-path scan_and_trade failed: {e}")


# ---------------------------------------------------------------------------
# Market discovery
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Price discovery scan — fast-poll recently discovered markets
# ---------------------------------------------------------------------------

def price_discovery_scan(ctx: BotContext):
    """Fast-poll markets discovered in the last N hours (price discovery window).

    New contracts have inefficient prices. Poll more frequently during
    the window to capture large edges before the market stabilizes.
    """
    window_hours = ctx.settings.scheduler.price_discovery_window_hours

    session = get_session()
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=window_hours)).isoformat()
        new_markets = (
            session.query(KalshiMarket)
            .filter(
                KalshiMarket.first_discovered_at >= cutoff,
                KalshiMarket.status == "open",
            )
            .all()
        )
    except Exception as e:
        logger.debug(f"Price discovery query failed (table may not have column yet): {e}")
        session.close()
        return
    finally:
        session.close()

    if not new_markets:
        return

    logger.info(f"Price discovery: {len(new_markets)} markets in window")

    # Convert DB rows to ParsedMarket for the trading pipeline
    discovery_markets: list[ParsedMarket] = []
    for km in new_markets:
        discovery_markets.append(ParsedMarket(
            market_ticker=km.market_ticker,
            event_ticker=km.event_ticker,
            city=km.city,
            target_date=km.target_date,
            bracket_low=km.bracket_low,
            bracket_high=km.bracket_high,
            is_above_contract=bool(km.is_above_contract),
            threshold_f=km.threshold_f,
            yes_price=km.yes_price or 0.0,
            no_price=km.no_price or 0.0,
            volume=km.volume or 0,
            close_time=km.close_time,
            status=km.status or "open",
        ))

    # Refresh prices for these markets
    discovery_markets = ctx.market_discovery.refresh_prices(discovery_markets)
    _snapshot_prices(discovery_markets)

    # Run the standard trading logic scoped to just these markets
    _run_trading_cycle(ctx, discovery_markets, tag="price_discovery")


# ---------------------------------------------------------------------------
# Price history snapshots — record prices every refresh cycle
# ---------------------------------------------------------------------------

def _snapshot_prices(markets: list[ParsedMarket]) -> int:
    """Record a price snapshot for each market into market_price_history.

    Called after every refresh_prices() so we build continuous price history
    from discovery through settlement.
    """
    if not markets:
        return 0
    session = get_session()
    now = datetime.now(timezone.utc).isoformat()
    count = 0
    try:
        for m in markets:
            if m.yes_price is None:
                continue
            snapshot = MarketPriceHistory(
                market_ticker=m.market_ticker,
                captured_at=now,
                yes_price=m.yes_price,
                no_price=m.no_price,
                volume=getattr(m, "volume", None),
                status=getattr(m, "status", "open"),
            )
            session.add(snapshot)
            count += 1
        session.commit()
        logger.debug("Snapshotted %d market prices", count)
    except Exception as e:
        session.rollback()
        logger.debug("Price snapshot error: %s", e)
    finally:
        session.close()
    return count


# ---------------------------------------------------------------------------
# DB cache helper — reads ensemble data without HTTP calls
# ---------------------------------------------------------------------------

def _load_ensemble_from_db(city_name: str, target_date: str) -> EnsembleResult | None:
    """Load the latest ensemble data for a city+date from the DB.

    Reads from the ensemble_forecasts table — NO HTTP calls.
    Returns an EnsembleResult or None if no data available.
    """
    session = get_session()
    try:
        # Find the latest model_run_time for this city
        from sqlalchemy import func
        latest_run = (
            session.query(func.max(EnsembleForecast.model_run_time))
            .filter(EnsembleForecast.city == city_name)
            .scalar()
        )
        if not latest_run:
            return None

        # Get all member data for this run + target_date
        rows = (
            session.query(EnsembleForecast)
            .filter(
                EnsembleForecast.city == city_name,
                EnsembleForecast.model_run_time == latest_run,
                EnsembleForecast.valid_time == target_date,
            )
            .order_by(EnsembleForecast.member)
            .all()
        )

        if not rows:
            return None

        # Build member_daily_maxes (31 values, NaN for missing)
        member_maxes = [float("nan")] * 31
        for row in rows:
            if 0 <= row.member < 31:
                member_maxes[row.member] = row.temperature_f

        valid_count = sum(1 for t in member_maxes if t == t)  # NaN != NaN
        if valid_count < 20:
            logger.warning(
                f"Only {valid_count} members in DB for {city_name} on {target_date}"
            )
            return None

        return EnsembleResult(
            city=city_name,
            model_run_time=latest_run,
            target_date=target_date,
            member_daily_maxes=member_maxes,
            member_hourly={},  # Not needed for probability calc
            valid_times=[],
        )
    except Exception as e:
        logger.warning(f"Failed to load ensemble from DB for {city_name}/{target_date}: {e}")
        return None
    finally:
        session.close()


def _load_ecmwf_from_db(city_name: str, target_date: str) -> list[float] | None:
    """Load the latest ECMWF IFS ensemble member daily maxes from the DB.

    Returns list of 51 member maxes, or None if insufficient data.
    """
    session = get_session()
    try:
        from sqlalchemy import func
        latest_run = (
            session.query(func.max(ECMWFForecast.model_run_time))
            .filter(ECMWFForecast.city == city_name)
            .scalar()
        )
        if not latest_run:
            return None

        rows = (
            session.query(ECMWFForecast)
            .filter(
                ECMWFForecast.city == city_name,
                ECMWFForecast.model_run_time == latest_run,
                ECMWFForecast.valid_time == target_date,
            )
            .order_by(ECMWFForecast.member)
            .all()
        )

        if not rows:
            return None

        member_maxes = [float("nan")] * 51
        for row in rows:
            if 0 <= row.member < 51:
                member_maxes[row.member] = row.temperature_f

        valid_count = sum(1 for t in member_maxes if t == t)
        if valid_count < 20:
            return None

        return member_maxes
    except Exception as e:
        logger.debug(f"Failed to load ECMWF from DB for {city_name}/{target_date}: {e}")
        return None
    finally:
        session.close()


def _load_hrrr_daily_max(city_name: str, target_date: str) -> tuple[float | None, str | None]:
    """Load the latest HRRR daily max forecast for a city+date from the DB.

    HRRR stores one row per (city, model_run_time, valid_time) where valid_time
    is a date string. Returns (daily_max_f, model_run_time) or (None, None).
    """
    session = get_session()
    try:
        from sqlalchemy import func

        latest_run = (
            session.query(func.max(HRRRForecast.model_run_time))
            .filter(HRRRForecast.city == city_name)
            .scalar()
        )
        if not latest_run:
            return None, None

        row = (
            session.query(HRRRForecast.temperature_f, HRRRForecast.model_run_time)
            .filter(
                HRRRForecast.city == city_name,
                HRRRForecast.model_run_time == latest_run,
                HRRRForecast.valid_time == target_date,
            )
            .first()
        )
        if row is None:
            return None, None

        return float(row[0]), row[1]

    except Exception as e:
        logger.warning(f"Failed to load HRRR from DB for {city_name}/{target_date}: {e}")
        return None, None
    finally:
        session.close()


def _load_nws_high(city_name: str, target_date: str) -> float | None:
    """Load the latest NWS high temperature forecast for a city+date from the DB.

    NWS stores multiple rows per (city, forecast_date) — one per fetch.
    Returns the most recent high_f or None.
    """
    session = get_session()
    try:
        row = (
            session.query(NWSForecast.high_f)
            .filter(
                NWSForecast.city == city_name,
                NWSForecast.forecast_date == target_date,
            )
            .order_by(NWSForecast.fetched_at.desc())
            .first()
        )
        if row and row[0] is not None:
            return float(row[0])
        return None

    except Exception as e:
        logger.warning(f"Failed to load NWS from DB for {city_name}/{target_date}: {e}")
        return None
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Active strategy loader
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Core trading cycle
# ---------------------------------------------------------------------------

def _store_all_signals(
    signals: dict[str, float],
    market_data: dict[str, dict],
    probs: dict,
    city_name: str,
    target_date: str,
    lead_hours: float,
    now: str,
    *,
    ticker_probs: dict[str, dict] | None = None,
    ensemble_run: str | None = None,
    hrrr_run: str | None = None,
):
    """Store ALL computed signals for edge tracking (not just traded ones).

    Now also stores hrrr_prob, nws_prob, blended_prob, ensemble_run, hrrr_run.
    """
    if ticker_probs is None:
        ticker_probs = {}

    session = get_session()
    try:
        for ticker, calibrated_prob in signals.items():
            market_info = market_data.get(ticker)
            prob_result = probs.get(ticker)
            if not market_info or not prob_result:
                continue

            yes_price = market_info["yes_price"]
            raw_edge = calibrated_prob - yes_price
            extra = ticker_probs.get(ticker, {})

            sig = Signal(
                city=city_name,
                target_date=target_date,
                market_ticker=ticker,
                computed_at=now,
                ensemble_prob=prob_result.probability,
                ecmwf_prob=extra.get("ecmwf_prob"),
                hrrr_prob=extra.get("hrrr_prob"),
                nws_prob=extra.get("nws_prob"),
                blended_prob=extra.get("blended_prob"),
                calibrated_prob=calibrated_prob,
                market_yes_price=yes_price,
                raw_edge=raw_edge,
                abs_edge=abs(raw_edge),
                lead_hours=lead_hours,
                ensemble_run=ensemble_run,
                hrrr_run=hrrr_run,
            )
            session.add(sig)
        session.commit()
    except Exception as e:
        session.rollback()
        logger.debug(f"Signal storage error (possible duplicate): {e}")
    finally:
        session.close()


def _run_trading_cycle(
    ctx: BotContext,
    markets: list[ParsedMarket],
    tag: str = "scan",
):
    """Shared trading logic used by both scan_and_trade and price_discovery_scan.

    1. Group markets by city+date
    2. Load ensemble from DB (NO live HTTP)
    3. Compute probabilities + edges
    4. Store ALL signals for edge tracking
    5. Execute trades where edge is sufficient
    """
    # Load active strategy (if any) to override config
    active_strategy_id = _load_active_strategy(ctx)

    # Group markets by city and target date
    by_city_date: dict[tuple[str, str], list[ParsedMarket]] = {}
    for m in markets:
        key = (m.city, m.target_date)
        by_city_date.setdefault(key, []).append(m)

    # Get portfolio state — use Kalshi balance in live mode, paper bankroll otherwise
    if ctx.settings.mode == "live":
        bankroll = ctx.kalshi_client.get_balance()
    else:
        bankroll = ctx.trader.bankroll
    portfolio = ctx.risk_manager.get_portfolio_state(bankroll, mode=ctx.settings.mode)

    for (city_name, target_date), city_markets in by_city_date.items():
        city_config = ctx.cities.get(city_name)
        if city_config is None:
            continue

        # Load ensemble from DB cache (NO live HTTP call)
        target_ensemble = _load_ensemble_from_db(city_name, target_date)

        if target_ensemble is None:
            logger.debug(f"[{tag}] No ensemble data in DB for {city_name}/{target_date}")
            continue

        # Compute actual lead hours
        target_date_obj = date.fromisoformat(target_date)
        model_run_dt = parse_iso_datetime(target_ensemble.model_run_time)
        lead_hours = compute_lead_hours(model_run_dt, target_date_obj, city_config.timezone)

        # --- HRRR correction of ensemble distribution ---
        hrrr_max, hrrr_run = _load_hrrr_daily_max(city_name, target_date)
        corrected_member_maxes = target_ensemble.member_daily_maxes

        if hrrr_max is not None and lead_hours < 48:
            ens_mean, _ens_std = ctx.ensemble_calc.get_ensemble_mean_and_spread(
                target_ensemble.member_daily_maxes
            )
            correction = ctx.hrrr_corrector.apply_correction(
                target_ensemble.member_daily_maxes,
                hrrr_max,
                ens_mean,
                lead_hours,
            )
            corrected_member_maxes = correction.adjusted_member_maxes
            logger.debug(
                f"[{tag}] HRRR correction for {city_name}/{target_date}: "
                f"shift={correction.shift:.1f}F, weight={correction.correction_weight:.2f}"
            )

        # Calculate ensemble probabilities using HRRR-corrected members
        probs = ctx.ensemble_calc.get_full_distribution(
            corrected_member_maxes, city_markets
        )

        # --- Load ECMWF IFS ensemble from DB ---
        ecmwf_maxes = _load_ecmwf_from_db(city_name, target_date)
        ecmwf_probs = {}
        if ecmwf_maxes is not None:
            ecmwf_probs = ctx.ensemble_calc.get_full_distribution(
                ecmwf_maxes, city_markets
            )
            logger.debug(
                f"[{tag}] ECMWF loaded for {city_name}/{target_date}: "
                f"{sum(1 for t in ecmwf_maxes if t == t)} valid members"
            )

        # --- Load NWS high forecast ---
        nws_high = _load_nws_high(city_name, target_date)

        # Build signal dict and market dict for edge detection
        signals: dict[str, float] = {}
        market_data: dict[str, dict] = {}
        ticker_probs: dict[str, dict] = {}

        for m in city_markets:
            prob_result = probs.get(m.market_ticker)
            if prob_result is None:
                continue

            ensemble_prob = prob_result.probability

            # --- Compute ECMWF probability ---
            ecmwf_prob = None
            ecmwf_result = ecmwf_probs.get(m.market_ticker)
            if ecmwf_result is not None:
                ecmwf_prob = ecmwf_result.probability

            # --- Compute HRRR probability ---
            hrrr_prob = None
            if hrrr_max is not None:
                if m.is_above_contract and m.threshold_f is not None:
                    hrrr_prob = ctx.model_blender.prob_from_deterministic(
                        hrrr_max, m.threshold_f, historical_std=3.0, is_above=True,
                    )
                elif m.bracket_low is not None or m.bracket_high is not None:
                    hrrr_prob = ctx.model_blender.prob_from_deterministic_bracket(
                        hrrr_max, m.bracket_low, m.bracket_high, historical_std=3.0,
                    )

            # --- Compute NWS probability ---
            nws_prob = None
            if nws_high is not None:
                if m.is_above_contract and m.threshold_f is not None:
                    nws_prob = ctx.model_blender.prob_from_deterministic(
                        nws_high, m.threshold_f, historical_std=4.0, is_above=True,
                    )
                elif m.bracket_low is not None or m.bracket_high is not None:
                    nws_prob = ctx.model_blender.prob_from_deterministic_bracket(
                        nws_high, m.bracket_low, m.bracket_high, historical_std=4.0,
                    )

            # --- Blend all sources (now including ECMWF) ---
            blended = ctx.model_blender.blend(
                ensemble_prob, hrrr_prob, nws_prob, lead_hours,
                ecmwf_prob=ecmwf_prob,
            )

            # Calibrate the BLENDED probability (not raw ensemble)
            calibrated = ctx.calibrator.calibrate(blended)
            signals[m.market_ticker] = calibrated

            market_data[m.market_ticker] = {
                "yes_price": m.yes_price,
                "city": m.city,
                "target_date": m.target_date,
            }
            ticker_probs[m.market_ticker] = {
                "hrrr_prob": hrrr_prob,
                "nws_prob": nws_prob,
                "ecmwf_prob": ecmwf_prob,
                "blended_prob": blended,
            }

        now = datetime.now(timezone.utc).isoformat()

        # Store ALL signals for edge tracking
        _store_all_signals(
            signals, market_data, probs, city_name, target_date, lead_hours, now,
            ticker_probs=ticker_probs,
            ensemble_run=target_ensemble.model_run_time,
            hrrr_run=hrrr_run,
        )

        # Detect edges
        trade_signals = ctx.edge_detector.scan_for_edges(signals, market_data, lead_hours=lead_hours)

        # Execute trades
        for signal in trade_signals:
            # Check risk
            size = ctx.kelly_sizer.calculate_position_size(signal, portfolio.bankroll)
            if size.contracts <= 0:
                continue

            risk_check = ctx.risk_manager.check_trade_allowed(
                signal.city, signal.target_date, size.total_cost, portfolio
            )
            if not risk_check.allowed:
                logger.info(f"[{tag}] Risk blocked: {risk_check.reason}")
                continue

            # Execute
            trade = ctx.trader.execute_trade(signal, size)
            if trade:
                portfolio.open_positions += 1
                portfolio.bankroll -= size.total_cost
                logger.info(
                    f"[{tag}] Trade executed: {signal.market_ticker} "
                    f"edge={signal.edge:.3f} contracts={size.contracts}"
                )


def scan_and_trade(ctx: BotContext):
    """Core trading cycle: signal -> edge -> size -> trade.

    Runs every 5 minutes. Reads ensemble data from DB cache
    (NOT live HTTP) so this is cheap to run frequently.
    """
    if not ctx._active_markets:
        discover_markets(ctx)

    if not ctx._active_markets:
        logger.info("No active markets to trade")
        return

    # Refresh market prices (this IS an API call, but lightweight)
    ctx._active_markets = ctx.market_discovery.refresh_prices(ctx._active_markets)
    _snapshot_prices(ctx._active_markets)

    # Run the shared trading cycle
    _run_trading_cycle(ctx, ctx._active_markets, tag="scan")
