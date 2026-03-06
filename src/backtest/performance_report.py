"""Backtest performance metrics computation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.config.settings import AppSettings


@dataclass
class PerformanceReport:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    gross_pnl: float
    avg_pnl_per_trade: float
    avg_edge: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    avg_contracts: float


def compute_performance(trades: list, settings: AppSettings) -> PerformanceReport:
    """Compute performance metrics from backtest trades."""
    if not trades:
        return PerformanceReport(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        )

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    gross_pnl = sum(t.pnl for t in trades)
    avg_pnl = gross_pnl / len(trades)
    avg_edge = sum(abs(t.edge) for t in trades) / len(trades)
    avg_contracts = sum(t.contracts for t in trades) / len(trades)

    # Max drawdown
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        cumulative += t.pnl
        peak = max(peak, cumulative)
        dd = peak - cumulative
        max_dd = max(max_dd, dd)

    # Sharpe ratio (annualized, assuming ~250 trading days)
    if len(trades) > 1:
        pnls = [t.pnl for t in trades]
        mean_pnl = sum(pnls) / len(pnls)
        std_pnl = (sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)) ** 0.5
        sharpe = (mean_pnl / std_pnl * math.sqrt(250)) if std_pnl > 0 else 0.0
    else:
        sharpe = 0.0

    # Profit factor
    total_wins = sum(t.pnl for t in wins) if wins else 0
    total_losses = abs(sum(t.pnl for t in losses)) if losses else 0.001
    profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

    return PerformanceReport(
        total_trades=len(trades),
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate=len(wins) / len(trades) if trades else 0,
        gross_pnl=gross_pnl,
        avg_pnl_per_trade=avg_pnl,
        avg_edge=avg_edge,
        max_drawdown=max_dd,
        sharpe_ratio=sharpe,
        profit_factor=profit_factor,
        avg_contracts=avg_contracts,
    )
