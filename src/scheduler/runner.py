"""Main scheduler: orchestrates all periodic jobs."""

from __future__ import annotations

import logging
import signal
import sys

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from src.config.settings import load_settings
from src.data.db import init_db, get_session
from src.scheduler.jobs import (
    BotContext,
    smart_data_fetch,
    discover_markets,
    scan_and_trade,
    price_discovery_scan,
)
from src.execution.settlement_checker import SettlementChecker
from src.monitoring.pnl_tracker import PnLTracker
from src.monitoring.email_reporter import send_daily_email

logger = logging.getLogger(__name__)


def run_bot():
    """Main entry point: initialize everything and start the scheduler."""
    # Setup logging
    from pathlib import Path
    log_dir = Path(__file__).resolve().parent.parent.parent / "data"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "bot.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )

    logger.info("=" * 60)
    logger.info("AetherBot - Kalshi Weather Market Predictor")
    logger.info("=" * 60)

    # Load settings and initialize
    settings = load_settings()
    logger.info(f"Mode: {settings.mode}")

    # Initialize database
    init_db(settings)
    logger.info("Database initialized")

    # Create bot context
    ctx = BotContext(settings)
    logger.info(f"Tracking {len(ctx.cities)} cities: {', '.join(ctx.cities.keys())}")

    # Create settlement checker (paper_trader is None in live mode — that's fine,
    # settlement_checker handles live trades via manual settlement path)
    settlement_checker = SettlementChecker(
        ctx.cities, paper_trader=ctx.paper_trader, settings=settings
    )

    # Bankroll source depends on mode
    if settings.mode == "live":
        pnl_tracker = PnLTracker(get_bankroll=lambda: ctx.kalshi_client.get_balance())
    else:
        pnl_tracker = PnLTracker(get_bankroll=lambda: ctx.trader.bankroll)

    # Create scheduler
    scheduler = BlockingScheduler()

    sched = settings.scheduler

    # ---------------------------------------------------------------
    # Smart data fetch — replaces old fixed-interval ensemble/hrrr/nws
    # Checks every 10min and only fetches when new model runs exist
    # ---------------------------------------------------------------
    scheduler.add_job(
        smart_data_fetch,
        trigger=IntervalTrigger(minutes=sched.smart_fetch_check_minutes),
        args=[ctx],
        id="smart_data_fetch",
        name="Smart Data Fetch",
    )

    # ---------------------------------------------------------------
    # Market discovery (every 30 minutes)
    # ---------------------------------------------------------------
    scheduler.add_job(
        discover_markets,
        trigger=IntervalTrigger(minutes=sched.market_discovery_interval_minutes),
        args=[ctx],
        id="discover_markets",
        name="Discover Markets",
    )

    # ---------------------------------------------------------------
    # Order sync (live mode only — check pending orders every 5 min)
    # ---------------------------------------------------------------
    if settings.mode == "live":
        scheduler.add_job(
            ctx.trader.sync_open_orders,
            trigger=IntervalTrigger(minutes=5),
            id="sync_orders",
            name="Sync Live Orders",
        )

    # ---------------------------------------------------------------
    # Core trading cycle (every 5 minutes, reads from DB — cheap)
    # ---------------------------------------------------------------
    scheduler.add_job(
        scan_and_trade,
        trigger=IntervalTrigger(minutes=sched.market_scan_interval_minutes),
        args=[ctx],
        id="scan_and_trade",
        name="Scan & Trade",
    )

    # ---------------------------------------------------------------
    # Price discovery scan (every 2 minutes for newly listed markets)
    # ---------------------------------------------------------------
    scheduler.add_job(
        price_discovery_scan,
        trigger=IntervalTrigger(minutes=sched.price_discovery_scan_minutes),
        args=[ctx],
        id="price_discovery_scan",
        name="Price Discovery Scan",
    )

    # ---------------------------------------------------------------
    # Helpers for post-settlement model retuning
    # ---------------------------------------------------------------
    def _retrain_calibrator():
        """Retrain isotonic calibrator if new settlement data is available."""
        try:
            if ctx.calibrator_trainer.should_retrain():
                logger.info("Calibrator retraining triggered")
                if ctx.calibrator_trainer.retrain():
                    ctx.calibrator = ctx.calibrator_trainer.calibrator
                    logger.info("Live calibrator updated from retraining")
        except Exception as e:
            logger.error(f"Calibrator retrain check failed: {e}")

    def _update_adaptive_weights():
        """Recompute adaptive blend weights from recent model performance."""
        try:
            weights = ctx.adaptive_weight_mgr.compute_and_save()
            if weights:
                ctx.model_blender.set_adaptive_weights(weights)
                logger.info(f"Adaptive blend weights updated: {weights}")
        except Exception as e:
            logger.error(f"Adaptive weight update failed: {e}")

    def _get_observed_high(city_name: str, target_date: str) -> int | None:
        """Look up observed high from the observations table."""
        from src.data.models import Observation
        session = get_session()
        try:
            obs = (
                session.query(Observation)
                .filter_by(city=city_name, date=target_date)
                .first()
            )
            return obs.high_f if obs else None
        finally:
            session.close()

    # ---------------------------------------------------------------
    # Settlement check + immediate model retuning (daily at 11:15 AM ET)
    # Settles trades → scores models → retrains calibrator → updates weights
    # ---------------------------------------------------------------
    def _settlement_and_retune():
        settlement_checker.check_settlements()
        _retrain_calibrator()
        _update_adaptive_weights()

    scheduler.add_job(
        _settlement_and_retune,
        trigger=CronTrigger(
            hour=sched.settlement_check_hour,
            minute=sched.settlement_check_minute,
            timezone="America/New_York",
        ),
        id="check_settlements",
        name="Settlement + Retune",
    )

    # ---------------------------------------------------------------
    # Post-settlement analysis safety net (11:45 AM ET)
    # Re-runs model scoring + postmortem for yesterday in case the
    # settlement integration missed anything. Also re-checks retune
    # (idempotent — should_retrain() returns False if already ran)
    # ---------------------------------------------------------------
    def _post_settlement_analysis():
        from datetime import date as dt_date, timedelta
        yesterday = (dt_date.today() - timedelta(days=1)).isoformat()
        logger.info(f"Running post-settlement analysis safety net for {yesterday}")
        settlement_checker._run_post_settlement_analysis(
            yesterday,
            {city_name: _get_observed_high(city_name, yesterday)
             for city_name in ctx.cities},
        )
        _retrain_calibrator()
        _update_adaptive_weights()

    scheduler.add_job(
        _post_settlement_analysis,
        trigger=CronTrigger(hour=11, minute=45, timezone="America/New_York"),
        id="post_settlement_analysis",
        name="Post-Settlement Analysis",
    )

    # ---------------------------------------------------------------
    # Daily report + email (noon ET)
    # ---------------------------------------------------------------
    def _daily_report_and_email():
        pnl_tracker.generate_daily_report()
        if settings.mode == "live":
            send_daily_email(settings, get_bankroll=lambda: ctx.kalshi_client.get_balance())
        else:
            send_daily_email(settings, get_bankroll=lambda: ctx.trader.bankroll)

    scheduler.add_job(
        _daily_report_and_email,
        trigger=CronTrigger(hour=12, timezone="America/New_York"),
        id="daily_report",
        name="Daily Report + Email",
    )

    # Graceful shutdown
    def shutdown_handler(signum, frame):
        logger.info("Shutting down...")
        scheduler.shutdown(wait=False)
        ctx.shutdown()
        settlement_checker.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Run initial data fetch and market discovery
    logger.info("Running initial smart data fetch...")
    smart_data_fetch(ctx)

    logger.info("Running initial market discovery...")
    discover_markets(ctx)

    # Start scheduler
    logger.info("Scheduler started. Press Ctrl+C to stop.")
    logger.info(
        f"  Smart fetch: every {sched.smart_fetch_check_minutes}min | "
        f"Discovery: every {sched.market_discovery_interval_minutes}min | "
        f"Scan: every {sched.market_scan_interval_minutes}min | "
        f"Price discovery: every {sched.price_discovery_scan_minutes}min"
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        ctx.shutdown()
        settlement_checker.close()
