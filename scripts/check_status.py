#!/usr/bin/env python3
"""Quick status check of the running bot.

Queries the SQLite database and prints a dashboard summary.

Usage:
    python scripts/check_status.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date
from src.config.settings import load_settings
from src.data.db import init_db, get_session
from src.data.models import Trade, Signal, EnsembleForecast, KalshiMarket

settings = load_settings()
init_db(settings)
session = get_session()

today = date.today().isoformat()

print("=" * 50)
print("AetherBot - Status Dashboard")
print("=" * 50)

# Open positions
open_trades = (
    session.query(Trade)
    .filter_by(status="filled", mode="paper")
    .all()
)
print(f"Open paper positions:  {len(open_trades)}")

# Today's trades
today_trades = (
    session.query(Trade)
    .filter(Trade.created_at >= today)
    .all()
)
print(f"Trades today:          {len(today_trades)}")

# Today's signals
today_signals = (
    session.query(Signal)
    .filter(Signal.computed_at >= today)
    .all()
)
print(f"Signals today:         {len(today_signals)}")

# Latest ensemble
latest_ens = (
    session.query(EnsembleForecast)
    .order_by(EnsembleForecast.fetched_at.desc())
    .first()
)
if latest_ens:
    print(f"Latest ensemble fetch: {latest_ens.fetched_at} ({latest_ens.city})")
else:
    print("Latest ensemble fetch: None")

# Active markets
active = session.query(KalshiMarket).filter_by(status="open").count()
print(f"Active markets in DB:  {active}")

# PnL summary
settled = (
    session.query(Trade)
    .filter_by(status="settled", mode="paper")
    .all()
)
total_pnl = sum(t.pnl or 0 for t in settled)
wins = sum(1 for t in settled if (t.pnl or 0) > 0)
losses = sum(1 for t in settled if (t.pnl or 0) <= 0)
print(f"\nSettled trades:        {len(settled)} ({wins}W / {losses}L)")
print(f"Total settled PnL:     ${total_pnl:+.2f}")

# Bankroll estimate
initial = settings.paper_trading.initial_bankroll
open_cost = sum(t.total_cost or 0 for t in open_trades)
print(f"Estimated bankroll:    ${initial + total_pnl - open_cost:,.2f}")

print("=" * 50)
session.close()
