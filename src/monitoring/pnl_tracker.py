"""PnL tracking and daily performance reporting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

from src.data.db import get_session
from src.data.models import Trade, DailyPnL

logger = logging.getLogger(__name__)


@dataclass
class DailyReport:
    date: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    gross_pnl: float
    bankroll: float
    avg_edge: float


class PnLTracker:
    """Tracks PnL and generates daily performance reports."""

    def get_daily_pnl(self, target_date: str) -> float:
        """Get total PnL for a specific date."""
        session = get_session()
        try:
            trades = (
                session.query(Trade)
                .filter_by(target_date=target_date, status="settled")
                .all()
            )
            return sum(t.pnl or 0 for t in trades)
        finally:
            session.close()

    def get_cumulative_pnl(self, start: str, end: str) -> float:
        """Get cumulative PnL over a date range."""
        session = get_session()
        try:
            trades = (
                session.query(Trade)
                .filter(Trade.target_date >= start)
                .filter(Trade.target_date <= end)
                .filter_by(status="settled")
                .all()
            )
            return sum(t.pnl or 0 for t in trades)
        finally:
            session.close()

    def get_win_rate(self, days: int = 30) -> float:
        """Get win rate over the last N days."""
        session = get_session()
        try:
            cutoff = (date.today() - timedelta(days=days)).isoformat()
            trades = (
                session.query(Trade)
                .filter(Trade.target_date >= cutoff)
                .filter_by(status="settled")
                .all()
            )
            if not trades:
                return 0.0
            wins = sum(1 for t in trades if (t.pnl or 0) > 0)
            return wins / len(trades)
        finally:
            session.close()

    def generate_daily_report(self, target_date: str | None = None) -> DailyReport:
        """Generate a daily performance report."""
        if target_date is None:
            target_date = (date.today() - timedelta(days=1)).isoformat()

        session = get_session()
        try:
            trades = (
                session.query(Trade)
                .filter_by(target_date=target_date, status="settled")
                .all()
            )

            wins = [t for t in trades if (t.pnl or 0) > 0]
            losses = [t for t in trades if (t.pnl or 0) <= 0]
            gross_pnl = sum(t.pnl or 0 for t in trades)
            avg_edge = (
                sum(abs(t.edge or 0) for t in trades) / len(trades)
                if trades else 0
            )

            report = DailyReport(
                date=target_date,
                total_trades=len(trades),
                wins=len(wins),
                losses=len(losses),
                win_rate=len(wins) / len(trades) if trades else 0,
                gross_pnl=gross_pnl,
                bankroll=0,  # TODO: compute from cumulative
                avg_edge=avg_edge,
            )

            logger.info(
                f"Daily Report [{target_date}]: "
                f"{report.total_trades} trades, "
                f"{report.wins}W/{report.losses}L, "
                f"PnL=${report.gross_pnl:+.2f}"
            )

            return report
        finally:
            session.close()
