"""Parameter sweep optimizer for strategy backtesting.

Runs a grid search across parameter combinations and returns
performance metrics for each, enabling co-optimization of
entry filters and position sizing.
"""

from __future__ import annotations

import itertools
import logging
from copy import deepcopy

from src.backtest.strategy_runner import run_strategy_backtest

logger = logging.getLogger(__name__)

# Default base strategy (matches current production defaults)
_BASE_STRATEGY = {
    "edge_threshold": 0.08,
    "min_edge_hrrr_confirm": 0.06,
    "min_model_prob": 0.55,
    "fractional_kelly": 0.15,
    "max_position_pct": 0.10,
    "max_position_dollars": 1000,
    "daily_loss_limit": 300,
    "max_concurrent_positions": 20,
    "max_positions_per_city": 6,
    "max_positions_per_date": 4,
    "min_price": 0.08,
    "max_price": 0.92,
    "max_lead_hours": 72,
}


class BacktestOptimizer:
    """Grid search over strategy parameters."""

    def grid_search(
        self,
        param_ranges: dict[str, list[float]],
        start_date: str,
        end_date: str,
        target_metric: str = "sharpe_ratio",
        city_names: list[str] | None = None,
    ) -> list[dict]:
        """Run backtest for every combination of parameter values.

        Args:
            param_ranges: Dict of {param_name: [values_to_try]}.
                e.g. {"edge_threshold": [0.06, 0.08, 0.10],
                       "fractional_kelly": [0.10, 0.15, 0.20]}
            start_date: Backtest start date
            end_date: Backtest end date
            target_metric: Metric to optimize (sharpe_ratio, win_rate,
                           gross_pnl, profit_factor)
            city_names: Optional city filter

        Returns:
            List of dicts: [{params: {...}, metrics: {...}}, ...]
        """
        # Build all combinations
        param_names = list(param_ranges.keys())
        param_values = [param_ranges[k] for k in param_names]
        combinations = list(itertools.product(*param_values))

        logger.info(
            f"Optimization: {len(combinations)} combinations across "
            f"{len(param_names)} parameters"
        )

        results = []
        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))

            # Build strategy dict from base + overrides
            strategy = deepcopy(_BASE_STRATEGY)
            strategy.update(params)

            try:
                bt_result = run_strategy_backtest(
                    strategy, start_date, end_date, city_names
                )
                perf = bt_result.get("performance", {})

                result_entry = {
                    "params": params,
                    "metrics": {
                        "total_trades": perf.get("total_trades", 0),
                        "win_rate": perf.get("win_rate", 0),
                        "gross_pnl": perf.get("gross_pnl", 0),
                        "sharpe_ratio": perf.get("sharpe_ratio", 0),
                        "max_drawdown": perf.get("max_drawdown", 0),
                        "profit_factor": perf.get("profit_factor", 0),
                        "avg_edge": perf.get("avg_edge", 0),
                    },
                    "brier_score": bt_result.get("brier_score"),
                }
                results.append(result_entry)

                logger.info(
                    f"  [{i+1}/{len(combinations)}] {params} -> "
                    f"{target_metric}={perf.get(target_metric, 0):.4f}"
                )

            except Exception as e:
                logger.warning(f"  [{i+1}/{len(combinations)}] {params} -> ERROR: {e}")
                results.append({
                    "params": params,
                    "metrics": {"error": str(e)},
                })

        return results
