#!/usr/bin/env python3
"""Run a walk-forward backtest over historical data.

Uses stored observations + synthetic market brackets to simulate
the full signal-to-trade pipeline and evaluate performance.

Usage:
    python scripts/run_backtest.py                         # Default: 1 year
    python scripts/run_backtest.py --days 180              # 6 months
    python scripts/run_backtest.py --city NYC              # Single city
    python scripts/run_backtest.py --edge-threshold 0.10   # Tighter edge
"""

import argparse
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

from src.config.settings import load_settings
from src.config.cities import load_cities
from src.data.db import init_db
from src.backtest.replay_engine import BacktestEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run backtest on historical data")
    parser.add_argument("--days", type=int, default=365, help="Days to backtest (default: 365)")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--city", type=str, default=None, help="Single city to test")
    parser.add_argument("--edge-threshold", type=float, default=None, help="Override edge threshold")
    parser.add_argument("--kelly-fraction", type=float, default=None, help="Override Kelly fraction")
    parser.add_argument("--bankroll", type=float, default=None, help="Starting bankroll (default: from settings)")
    args = parser.parse_args()

    settings = load_settings()
    init_db(settings)
    cities = load_cities()

    # Apply overrides
    if args.edge_threshold is not None:
        settings.strategy.edge_threshold = args.edge_threshold
    if args.kelly_fraction is not None:
        settings.strategy.fractional_kelly = args.kelly_fraction

    # Filter cities
    if args.city:
        if args.city not in cities:
            logger.error(f"Unknown city: {args.city}. Available: {', '.join(cities.keys())}")
            sys.exit(1)
        cities = {args.city: cities[args.city]}

    # Date range
    if args.end:
        end_date = date.fromisoformat(args.end)
    else:
        end_date = date.today() - timedelta(days=1)

    if args.start:
        start_date = date.fromisoformat(args.start)
    else:
        start_date = end_date - timedelta(days=args.days)

    bankroll = args.bankroll or settings.paper_trading.initial_bankroll

    logger.info("=" * 60)
    logger.info("AetherBot - Backtest Engine")
    logger.info("=" * 60)
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Cities: {', '.join(cities.keys())}")
    logger.info(f"Edge threshold: {settings.strategy.edge_threshold}")
    logger.info(f"Kelly fraction: {settings.strategy.fractional_kelly}")
    logger.info(f"Starting bankroll: ${bankroll:.2f}")

    # Run backtest
    engine = BacktestEngine(settings)
    results = engine.run(
        start_date.isoformat(),
        end_date.isoformat(),
        cities,
    )

    # Report results
    metrics = results.performance
    if metrics is None:
        logger.warning("No trades generated during backtest period.")
        return

    final_bankroll = bankroll + metrics.gross_pnl

    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total trades:    {metrics.total_trades}")
    logger.info(f"Win rate:        {metrics.win_rate:.1%}")
    logger.info(f"Gross PnL:       ${metrics.gross_pnl:+.2f}")
    logger.info(f"Max drawdown:    ${metrics.max_drawdown:.2f}")
    logger.info(f"Sharpe ratio:    {metrics.sharpe_ratio:.2f}")
    logger.info(f"Profit factor:   {metrics.profit_factor:.2f}")
    logger.info(f"Avg edge:        {metrics.avg_edge:.1%}")
    logger.info(f"Final bankroll:  ${final_bankroll:.2f}")
    if results.brier is not None:
        logger.info(f"Brier score:     {results.brier.brier_score:.4f}")
        logger.info(f"  Reliability:   {results.brier.reliability:.4f}")
        logger.info(f"  Resolution:    {results.brier.resolution:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
