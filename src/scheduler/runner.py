"""Main scheduler: orchestrates all periodic jobs."""

from __future__ import annotations

import logging
import signal
import sys

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from src.config.settings import load_settings
from src.data.db import init_db
from src.scheduler.jobs import (
    BotContext,
    fetch_all_ensembles,
    fetch_all_hrrr,
    fetch_all_nws,
    discover_markets,
    scan_and_trade,
)
from src.execution.settlement_checker import SettlementChecker
from src.monitoring.pnl_tracker import PnLTracker

logger = logging.getLogger(__name__)


def run_bot():
    """Main entry point: initialize everything and start the scheduler."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
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

    # Create settlement checker
    settlement_checker = SettlementChecker(
        ctx.cities, paper_trader=ctx.paper_trader
    )
    pnl_tracker = PnLTracker()

    # Create scheduler
    scheduler = BlockingScheduler()

    sched = settings.scheduler

    # Data fetch jobs
    scheduler.add_job(
        fetch_all_ensembles,
        trigger=IntervalTrigger(minutes=sched.ensemble_fetch_interval_minutes),
        args=[ctx],
        id="fetch_ensemble",
        name="Fetch GFS Ensemble",
        next_run_time=None,  # Don't run immediately; let market scan trigger first
    )

    scheduler.add_job(
        fetch_all_hrrr,
        trigger=IntervalTrigger(minutes=sched.hrrr_fetch_interval_minutes),
        args=[ctx],
        id="fetch_hrrr",
        name="Fetch HRRR",
    )

    scheduler.add_job(
        fetch_all_nws,
        trigger=IntervalTrigger(minutes=120),
        args=[ctx],
        id="fetch_nws",
        name="Fetch NWS Forecasts",
    )

    # Market discovery (3x/day)
    scheduler.add_job(
        discover_markets,
        trigger=CronTrigger(hour="6,12,18"),
        args=[ctx],
        id="discover_markets",
        name="Discover Markets",
    )

    # Core trading cycle (every 5 minutes)
    scheduler.add_job(
        scan_and_trade,
        trigger=IntervalTrigger(minutes=sched.market_scan_interval_minutes),
        args=[ctx],
        id="scan_and_trade",
        name="Scan & Trade",
    )

    # Settlement check (daily at 11:15 AM ET)
    scheduler.add_job(
        settlement_checker.check_settlements,
        trigger=CronTrigger(
            hour=sched.settlement_check_hour,
            minute=sched.settlement_check_minute,
            timezone="America/New_York",
        ),
        id="check_settlements",
        name="Check Settlements",
    )

    # Daily report (noon ET)
    scheduler.add_job(
        pnl_tracker.generate_daily_report,
        trigger=CronTrigger(hour=12, timezone="America/New_York"),
        id="daily_report",
        name="Daily Report",
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

    # Run initial discovery
    logger.info("Running initial market discovery...")
    discover_markets(ctx)

    # Start scheduler
    logger.info("Scheduler started. Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        ctx.shutdown()
        settlement_checker.close()
