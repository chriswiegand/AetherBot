"""Run a backtest using strategy parameters from the DB.

Wraps the existing BacktestEngine so it can accept a strategy dict
(from the strategies table) and return results suitable for the API.
"""

from __future__ import annotations

import logging
from dataclasses import asdict

from src.config.settings import load_settings
from src.config.cities import load_cities
from src.backtest.replay_engine import BacktestEngine

logger = logging.getLogger(__name__)

# Fields in the strategy dict that map to StrategyConfig attributes
_STRATEGY_TO_CONFIG = {
    "edge_threshold": float,
    "min_edge_hrrr_confirm": float,
    "min_model_prob": float,
    "fractional_kelly": float,
    "max_position_pct": float,
    "max_position_dollars": float,
    "daily_loss_limit": float,
    "max_concurrent_positions": int,
    "max_positions_per_city": int,
    "max_positions_per_date": int,
    "min_price": float,
    "max_price": float,
    "max_lead_hours": float,
}


def run_strategy_backtest(
    strategy: dict,
    start_date: str,
    end_date: str,
    city_names: list[str] | None = None,
) -> dict:
    """Run a backtest using strategy parameters.

    Args:
        strategy: Strategy dict from DB row
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'
        city_names: Optional list of city names to test (default: all)

    Returns:
        dict with keys: performance, trades, daily_pnl, cities, brier_score
    """
    settings = load_settings()

    # Override StrategyConfig from strategy dict
    for field, cast in _STRATEGY_TO_CONFIG.items():
        val = strategy.get(field)
        if val is not None:
            setattr(settings.strategy, field, cast(val))

    # Load cities
    all_cities = load_cities()
    if city_names:
        cities = {k: v for k, v in all_cities.items() if k in city_names}
    else:
        cities = all_cities

    # Run backtest
    engine = BacktestEngine(settings)
    result = engine.run(
        start_date=start_date,
        end_date=end_date,
        cities=cities,
        edge_threshold=settings.strategy.edge_threshold,
    )

    # Serialize
    trades_list = []
    for t in result.trades:
        trades_list.append({
            "date": t.date,
            "city": t.city,
            "side": t.side,
            "threshold": t.threshold,
            "model_prob": round(t.model_prob, 4),
            "market_price": round(t.market_price, 4),
            "edge": round(t.edge, 4),
            "contracts": t.contracts,
            "price": round(t.price, 4),
            "cost": round(t.cost, 2),
            "observed_high": t.observed_high,
            "settled_yes": t.settled_yes,
            "pnl": round(t.pnl, 2),
        })

    perf = {}
    if result.performance:
        perf = {
            "total_trades": result.performance.total_trades,
            "winning_trades": result.performance.winning_trades,
            "losing_trades": result.performance.losing_trades,
            "win_rate": round(result.performance.win_rate, 4),
            "gross_pnl": round(result.performance.gross_pnl, 2),
            "avg_pnl_per_trade": round(result.performance.avg_pnl_per_trade, 2),
            "avg_edge": round(result.performance.avg_edge, 4),
            "max_drawdown": round(result.performance.max_drawdown, 2),
            "sharpe_ratio": round(result.performance.sharpe_ratio, 4),
            "profit_factor": round(result.performance.profit_factor, 4),
            "avg_contracts": round(result.performance.avg_contracts, 1),
        }

    brier = None
    if result.brier:
        brier = round(result.brier.brier_score, 4)

    return {
        "performance": perf,
        "trades": trades_list,
        "daily_pnl": {k: round(v, 2) for k, v in result.daily_pnl.items()},
        "cities": result.cities,
        "brier_score": brier,
    }
