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
    smart_data_fetch,
    discover_markets,
    scan_and_trade,
    price_discovery_scan,
)
from src.execution.settlement_checker import SettlementChecker
from src.monitoring.pnl_tracker import PnLTracker

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

    # Create settlement checker
    settlement_checker = SettlementChecker(
        ctx.cities, paper_trader=ctx.paper_trader
    )
    pnl_tracker = PnLTracker(get_bankroll=lambda: ctx.paper_trader.bankroll)

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
    # Settlement check (daily at 11:15 AM ET)
    # ---------------------------------------------------------------
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

    # ---------------------------------------------------------------
    # Daily report (noon ET)
    # ---------------------------------------------------------------
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
