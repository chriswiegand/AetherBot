#!/usr/bin/env python3
"""Inspect live Kalshi weather markets for debugging.

Displays current market state, orderbook depths, and computed signals
for all active KXHIGH markets.

Usage:
    python scripts/inspect_markets.py                  # All cities
    python scripts/inspect_markets.py --city NYC       # Single city
    python scripts/inspect_markets.py --ticker KXHIGHNY-26MAR06-T48  # Single market
    python scripts/inspect_markets.py --show-orderbook # Include depth
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

from src.config.settings import load_settings
from src.config.cities import load_cities
from src.data.kalshi_client import KalshiClient
from src.data.kalshi_markets import KalshiMarketDiscovery

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Inspect Kalshi weather markets")
    parser.add_argument("--city", type=str, default=None, help="Filter by city (e.g., NYC)")
    parser.add_argument("--ticker", type=str, default=None, help="Inspect a specific market ticker")
    parser.add_argument("--show-orderbook", action="store_true", help="Show orderbook depth")
    args = parser.parse_args()

    settings = load_settings()
    cities = load_cities()

    client = KalshiClient(settings)
    discovery = KalshiMarketDiscovery(client)

    if args.ticker:
        # Single market inspection
        market = client.get_market(args.ticker)
        print(f"\n{'=' * 60}")
        print(f"Market: {market.get('ticker', 'N/A')}")
        print(f"Title:  {market.get('title', 'N/A')}")
        print(f"Status: {market.get('status', 'N/A')}")
        print(f"Yes Price: {market.get('yes_ask', 'N/A')}c")
        print(f"No Price:  {market.get('no_ask', 'N/A')}c")
        print(f"Volume:    {market.get('volume', 0)}")
        print(f"Open Int:  {market.get('open_interest', 0)}")

        if args.show_orderbook:
            book = client.get_orderbook(args.ticker)
            print(f"\nOrderbook:")
            print(f"  YES bids: {book.get('yes', [])[:5]}")
            print(f"  NO  bids: {book.get('no', [])[:5]}")

        print(f"{'=' * 60}\n")
        client.close()
        return

    # Discover all active markets
    if args.city:
        if args.city not in cities:
            logger.error(f"Unknown city: {args.city}. Available: {', '.join(cities.keys())}")
            sys.exit(1)
        target_cities = {args.city: cities[args.city]}
    else:
        target_cities = cities

    markets = discovery.discover_active_markets(target_cities)

    if not markets:
        print("\nNo active markets found.")
        client.close()
        return

    # Group by city and date
    by_city: dict[str, list] = {}
    for m in markets:
        by_city.setdefault(m.city, []).append(m)

    for city_name, city_markets in sorted(by_city.items()):
        print(f"\n{'=' * 60}")
        print(f"  {city_name} - {len(city_markets)} active markets")
        print(f"{'=' * 60}")

        # Sort by date then threshold
        city_markets.sort(key=lambda m: (m.target_date, m.threshold_f or 0))

        current_date = None
        for m in city_markets:
            if m.target_date != current_date:
                current_date = m.target_date
                print(f"\n  Date: {current_date}")
                print(f"  {'Ticker':<35} {'Type':<8} {'Thresh':<8} {'YES':>6} {'Vol':>8}")
                print(f"  {'-' * 70}")

            mtype = "Above" if m.is_above_contract else "Bracket"
            thresh = f"{m.threshold_f}F" if m.threshold_f else f"{m.bracket_low}-{m.bracket_high}F"
            yes_price = f"{m.yes_price}c" if m.yes_price else "N/A"
            volume = str(m.volume) if m.volume else "0"

            print(f"  {m.market_ticker:<35} {mtype:<8} {thresh:<8} {yes_price:>6} {volume:>8}")

    print()
    client.close()


if __name__ == "__main__":
    main()
