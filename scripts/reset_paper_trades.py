#!/usr/bin/env python3
"""Reset paper trades: clears trades, signals, and brier_scores tables.

Usage:
    python scripts/reset_paper_trades.py
"""

import os
import sys
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "weather_bot.db"


def main():
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    # Count before
    cur.execute("SELECT COUNT(*) FROM trades")
    trades_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM signals")
    signals_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM brier_scores")
    brier_count = cur.fetchone()[0]

    print(f"Before reset:")
    print(f"  trades:       {trades_count}")
    print(f"  signals:      {signals_count}")
    print(f"  brier_scores: {brier_count}")
    print()

    # Clear tables
    cur.execute("DELETE FROM trades")
    cur.execute("DELETE FROM signals")
    cur.execute("DELETE FROM brier_scores")
    conn.commit()

    print("All trades, signals, and brier_scores cleared.")
    print("Bankroll reset to $10,000 (initial value).")
    conn.close()


if __name__ == "__main__":
    main()
