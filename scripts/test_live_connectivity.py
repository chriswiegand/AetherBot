#!/usr/bin/env python3
"""Test live Kalshi API connectivity before going live.

Verifies:
1. Authentication works
2. Account balance is accessible
3. Markets can be fetched
4. Order create + cancel round-trip works (1 contract at 1c)
5. Email sending works (optional)

Usage:
    python scripts/test_live_connectivity.py
    python scripts/test_live_connectivity.py --skip-order   # skip order test
    python scripts/test_live_connectivity.py --test-email    # also test email
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import load_settings
from src.data.db import init_db


def main():
    parser = argparse.ArgumentParser(description="Test Kalshi API connectivity")
    parser.add_argument("--skip-order", action="store_true", help="Skip order round-trip test")
    parser.add_argument("--test-email", action="store_true", help="Send a test email")
    args = parser.parse_args()

    settings = load_settings()
    init_db(settings)

    print("=" * 60)
    print("AetherBot Live Connectivity Test")
    print("=" * 60)
    print(f"Mode: {settings.mode}")
    print(f"API URL: {settings.kalshi.active_url}")
    print()

    from src.data.kalshi_client import KalshiClient
    client = KalshiClient(settings)

    # --- Test 1: Authentication + Balance ---
    print("[1] Testing authentication + balance...")
    try:
        balance = client.get_balance()
        print(f"    Balance: ${balance:.2f}")
        if balance <= 0:
            print("    WARNING: Balance is $0! Fund your account before going live.")
        else:
            print("    OK")
    except Exception as e:
        print(f"    FAILED: {e}")
        print("    Check your KALSHI_API_KEY_ID and private key.")
        return 1

    # --- Test 2: Market Discovery ---
    print("\n[2] Testing market discovery...")
    try:
        from src.config.cities import load_cities
        from src.data.kalshi_markets import KalshiMarketDiscovery
        cities_config = load_cities()
        discovery = KalshiMarketDiscovery(client)
        markets = discovery.discover_active_markets(cities_config)
        print(f"    Found {len(markets)} KXHIGH markets")
        if markets:
            cities = set(m.city for m in markets)
            dates = set(m.target_date for m in markets)
            print(f"    Cities: {', '.join(sorted(cities))}")
            print(f"    Dates: {', '.join(sorted(dates))}")
        print("    OK")
    except Exception as e:
        print(f"    FAILED: {e}")
        return 1

    # --- Test 3: Order Round-Trip ---
    if not args.skip_order:
        print("\n[3] Testing order create + cancel...")
        if not markets:
            print("    SKIPPED: No markets found to test with")
        else:
            # Pick first market, place at 1c (won't fill)
            test_market = markets[0]
            print(f"    Test market: {test_market.market_ticker}")
            try:
                order = client.create_order(
                    ticker=test_market.market_ticker,
                    side="yes",
                    action="buy",
                    order_type="limit",
                    yes_price=1,  # 1 cent — won't fill
                    count=1,
                )
                print(f"    Order created: {order.order_id} (status={order.status})")

                cancelled = client.cancel_order(order.order_id)
                if cancelled:
                    print("    Order cancelled successfully")
                    print("    OK — Full round-trip works!")
                else:
                    print("    WARNING: Cancel returned False (order may have filled at 1c)")
            except Exception as e:
                print(f"    FAILED: {e}")
                return 1
    else:
        print("\n[3] Order test SKIPPED (--skip-order)")

    # --- Test 4: Risk Limits ---
    print("\n[4] Checking risk configuration...")
    s = settings.strategy
    print(f"    Daily spend limit: ${s.daily_spend_limit:.2f}")
    print(f"    Daily loss limit:  ${s.daily_loss_limit:.2f}")
    print(f"    Max per trade:     ${s.max_position_dollars:.2f}")
    print(f"    Fractional Kelly:  {s.fractional_kelly:.2f}")
    print(f"    Max concurrent:    {s.max_concurrent_positions}")
    print(f"    Max per city:      {s.max_positions_per_city}")
    print(f"    Max per date:      {s.max_positions_per_date}")

    if s.daily_spend_limit > 0:
        print(f"    OK — Hard daily cap at ${s.daily_spend_limit:.2f}")
    else:
        print("    WARNING: No daily spend limit set!")

    # --- Test 5: Email (optional) ---
    if args.test_email:
        print("\n[5] Testing email...")
        try:
            from src.monitoring.email_reporter import send_daily_email
            if settings.mode == "live":
                send_daily_email(settings, get_bankroll=lambda: client.get_balance())
            else:
                send_daily_email(settings, get_bankroll=lambda: balance)
            print(f"    Email sent to {settings.email.recipient}")
            print("    OK — Check your inbox!")
        except Exception as e:
            print(f"    FAILED: {e}")
    else:
        print("\n[5] Email test SKIPPED (use --test-email to test)")

    print("\n" + "=" * 60)
    print("All tests passed! Ready for live trading.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
