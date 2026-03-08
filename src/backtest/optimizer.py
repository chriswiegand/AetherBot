"""Parameter sweep optimizer for strategy backtesting.

Runs a grid search across parameter combinations and returns
performance metrics for each (aggregate + per-city), enabling
co-optimization of entry filters and position sizing.
"""

from __future__ import annotations

import itertools
import logging
import math
from collections import defaultdict
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


def _compute_city_metrics(trades: list[dict]) -> dict[str, dict]:
    """Compute per-city performance from a list of trade dicts.

    Returns {city_name: {total_trades, win_rate, gross_pnl, sharpe_ratio, profit_factor}}.
    """
    by_city: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        by_city[t["city"]].append(t)

    city_metrics = {}
    for city, city_trades in by_city.items():
        n = len(city_trades)
        wins = [t for t in city_trades if t["pnl"] > 0]
        losses = [t for t in city_trades if t["pnl"] <= 0]

        gross_pnl = sum(t["pnl"] for t in city_trades)
        win_rate = len(wins) / n if n else 0

        # Sharpe (annualized)
        if n > 1:
            pnls = [t["pnl"] for t in city_trades]
            mean_pnl = sum(pnls) / len(pnls)
            std_pnl = (sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)) ** 0.5
            sharpe = (mean_pnl / std_pnl * math.sqrt(250)) if std_pnl > 0 else 0.0
        else:
            sharpe = 0.0

        # Profit factor
        total_wins = sum(t["pnl"] for t in wins) if wins else 0
        total_losses = abs(sum(t["pnl"] for t in losses)) if losses else 0.001
        pf = total_wins / total_losses if total_losses > 0 else 0.0

        city_metrics[city] = {
            "total_trades": n,
            "win_rate": round(win_rate, 4),
            "gross_pnl": round(gross_pnl, 2),
            "sharpe_ratio": round(sharpe, 4),
            "profit_factor": round(pf, 4),
        }

    return city_metrics


class BacktestOptimizer:
    """Grid search over strategy parameters."""

    def grid_search(
        self,
        param_ranges: dict[str, list[float]],
        start_date: str,
        end_date: str,
        target_metric: str = "sharpe_ratio",
        city_names: list[str] | None = None,
    ) -> dict:
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
            Dict with:
              "results": [{params, metrics, city_metrics}, ...]
              "best_by_city": {city: {params, metrics}}
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
        all_cities_seen: set[str] = set()

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
                trades = bt_result.get("trades", [])

                # Per-city breakdown
                city_metrics = _compute_city_metrics(trades)
                all_cities_seen.update(city_metrics.keys())

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
                    "city_metrics": city_metrics,
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
                    "city_metrics": {},
                })

        # Compute best parameter set per city
        best_by_city: dict[str, dict] = {}
        for city in sorted(all_cities_seen):
            best_score = -float("inf")
            best_entry = None
            for r in results:
                cm = r.get("city_metrics", {}).get(city)
                if cm and cm.get(target_metric, 0) > best_score:
                    best_score = cm[target_metric]
                    best_entry = {"params": r["params"], "metrics": cm}
            if best_entry:
                best_by_city[city] = best_entry

        return {
            "results": results,
            "best_by_city": best_by_city,
        }
