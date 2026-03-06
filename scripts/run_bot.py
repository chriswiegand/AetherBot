#!/usr/bin/env python3
"""AetherBot - Kalshi Weather Market Predictor.

Main entry point. Starts the scheduler and all periodic jobs.

Usage:
    python scripts/run_bot.py
    python scripts/run_bot.py --mode paper   # explicit paper mode
    python scripts/run_bot.py --mode live     # live trading (use with caution)
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scheduler.runner import run_bot


def main():
    parser = argparse.ArgumentParser(description="AetherBot - Kalshi Weather Market Predictor")
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default=None,
        help="Trading mode (overrides .env/settings). Default: paper",
    )
    args = parser.parse_args()

    if args.mode:
        os.environ["MODE"] = args.mode

    run_bot()


if __name__ == "__main__":
    main()
