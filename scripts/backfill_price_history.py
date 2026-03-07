#!/usr/bin/env python3
"""Backfill market_price_history from existing Signal table snapshots.

Run once to seed historical price data from signal computations.
Subsequent price history is captured live by _snapshot_prices() in jobs.py.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.db import init_db, get_session
from src.data.models import Signal, MarketPriceHistory


def backfill():
    """Insert price history rows from signals that have market_yes_price."""
    init_db()
    session = get_session()

    # Count existing price history
    existing = session.query(MarketPriceHistory).count()
    print(f"Existing price history rows: {existing}")

    # Get all signals with a market price
    signals = (
        session.query(
            Signal.market_ticker,
            Signal.computed_at,
            Signal.market_yes_price,
        )
        .filter(Signal.market_yes_price.isnot(None))
        .order_by(Signal.computed_at.asc())
        .all()
    )
    print(f"Signal rows with market_yes_price: {len(signals)}")

    if not signals:
        print("Nothing to backfill.")
        session.close()
        return

    # Build set of existing (ticker, time) to avoid duplicates
    existing_keys = set()
    if existing > 0:
        rows = session.query(
            MarketPriceHistory.market_ticker,
            MarketPriceHistory.captured_at,
        ).all()
        existing_keys = {(r[0], r[1]) for r in rows}
        print(f"Existing unique (ticker, time) keys: {len(existing_keys)}")

    inserted = 0
    skipped = 0
    for sig in signals:
        key = (sig.market_ticker, sig.computed_at)
        if key in existing_keys:
            skipped += 1
            continue

        snapshot = MarketPriceHistory(
            market_ticker=sig.market_ticker,
            captured_at=sig.computed_at,
            yes_price=sig.market_yes_price,
            no_price=None,   # Not available from signals
            volume=None,
            status=None,
        )
        session.add(snapshot)
        inserted += 1

        # Batch commits
        if inserted % 500 == 0:
            session.commit()
            print(f"  ... inserted {inserted} rows")

    session.commit()
    session.close()

    print(f"\nDone! Inserted {inserted} rows, skipped {skipped} duplicates.")
    total = existing + inserted
    print(f"Total price history rows: {total}")


if __name__ == "__main__":
    backfill()
