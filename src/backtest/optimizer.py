"""Parameter sweep optimizer for strategy backtesting.

Runs grid search or adaptive two-phase search across parameter
combinations and returns performance metrics for each (aggregate +
per-city), enabling co-optimization of entry filters and position sizing.

Strategies:
  - grid_search():     Exhaustive grid (original)
  - adaptive_search(): MC→Refine or Coarse→Fine (two-phase)
"""

from __future__ import annotations

import itertools
import logging
import math
import random as rng
from collections import defaultdict
from copy import deepcopy
from typing import Callable

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


class OptimizationAborted(Exception):
    """Raised when the user aborts a running optimization."""


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
    """Grid search and adaptive two-phase optimizer."""

    def __init__(self):
        self._abort_requested = False

    def request_abort(self):
        """Signal the optimizer to stop after the current combo."""
        self._abort_requested = True

    # ------------------------------------------------------------------
    #  Shared helper: evaluate a single parameter combination
    # ------------------------------------------------------------------
    def _run_single_combo(
        self,
        params: dict,
        start_date: str,
        end_date: str,
        city_names: list[str] | None,
        target_metric: str,
    ) -> dict:
        """Run one backtest for a parameter set, return result entry."""
        strategy = deepcopy(_BASE_STRATEGY)
        strategy.update(params)

        try:
            bt_result = run_strategy_backtest(
                strategy, start_date, end_date, city_names
            )
            perf = bt_result.get("performance", {})
            trades = bt_result.get("trades", [])
            city_metrics = _compute_city_metrics(trades)

            return {
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
        except Exception as e:
            logger.warning(f"  {params} -> ERROR: {e}")
            return {
                "params": params,
                "metrics": {"error": str(e)},
                "city_metrics": {},
            }

    # ------------------------------------------------------------------
    #  Compute best-by-city from results
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_best_by_city(
        results: list[dict], target_metric: str
    ) -> dict[str, dict]:
        all_cities: set[str] = set()
        for r in results:
            all_cities.update(r.get("city_metrics", {}).keys())

        best_by_city: dict[str, dict] = {}
        for city in sorted(all_cities):
            best_score = -float("inf")
            best_entry = None
            for r in results:
                cm = r.get("city_metrics", {}).get(city)
                if cm and cm.get(target_metric, 0) > best_score:
                    best_score = cm[target_metric]
                    best_entry = {"params": r["params"], "metrics": cm}
            if best_entry:
                best_by_city[city] = best_entry

        return best_by_city

    # ==================================================================
    #  EXHAUSTIVE GRID SEARCH (original, unchanged interface)
    # ==================================================================
    def grid_search(
        self,
        param_ranges: dict[str, list[float]],
        start_date: str,
        end_date: str,
        target_metric: str = "sharpe_ratio",
        city_names: list[str] | None = None,
    ) -> dict:
        """Run backtest for every combination of parameter values.

        Returns:
            Dict with:
              "results": [{params, metrics, city_metrics}, ...]
              "best_by_city": {city: {params, metrics}}
        """
        param_names = list(param_ranges.keys())
        param_values = [param_ranges[k] for k in param_names]
        combinations = list(itertools.product(*param_values))

        logger.info(
            f"Optimization: {len(combinations)} combinations across "
            f"{len(param_names)} parameters"
        )

        results = []
        for i, combo in enumerate(combinations):
            if self._abort_requested:
                raise OptimizationAborted("Aborted by user")

            params = dict(zip(param_names, combo))
            entry = self._run_single_combo(
                params, start_date, end_date, city_names, target_metric
            )
            results.append(entry)

            metric_val = entry.get("metrics", {}).get(target_metric, 0)
            logger.info(
                f"  [{i+1}/{len(combinations)}] {params} -> "
                f"{target_metric}={metric_val:.4f}"
            )

        return {
            "results": results,
            "best_by_city": self._compute_best_by_city(results, target_metric),
        }

    # ==================================================================
    #  ADAPTIVE TWO-PHASE SEARCH
    # ==================================================================
    def adaptive_search(
        self,
        param_ranges: dict[str, list[float]],
        start_date: str,
        end_date: str,
        target_metric: str = "sharpe_ratio",
        city_names: list[str] | None = None,
        strategy: str = "mc_refine",
        progress_cb: Callable | None = None,
        phase1_budget: int = 200,
    ) -> dict:
        """Two-phase optimization: explore broadly, then refine.

        Args:
            param_ranges: {param_name: [values_to_try]}
            strategy: "mc_refine" (Latin Hypercube → fine grid) or
                      "coarse_fine" (coarse grid → fine grid)
            progress_cb: Called after each combo with
                (phase, current, total, best_so_far, entry)
            phase1_budget: Max combos for Phase 1 (default 200)

        Returns:
            Same format as grid_search() with added "phase" key per entry.
        """
        self._abort_requested = False

        if strategy == "mc_refine":
            results = self._mc_refine_search(
                param_ranges, start_date, end_date,
                target_metric, city_names, progress_cb, phase1_budget,
            )
        elif strategy == "coarse_fine":
            results = self._coarse_fine_search(
                param_ranges, start_date, end_date,
                target_metric, city_names, progress_cb, phase1_budget,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return {
            "results": results,
            "best_by_city": self._compute_best_by_city(results, target_metric),
        }

    # ------------------------------------------------------------------
    #  Monte Carlo → Refine
    # ------------------------------------------------------------------
    def _mc_refine_search(
        self,
        param_ranges: dict[str, list[float]],
        start_date: str,
        end_date: str,
        target_metric: str,
        city_names: list[str] | None,
        progress_cb: Callable | None,
        phase1_budget: int,
    ) -> list[dict]:
        # Phase 1: Latin Hypercube Sampling
        total_combos = 1
        for v in param_ranges.values():
            total_combos *= len(v)
        n_samples = min(phase1_budget, total_combos)

        logger.info(
            f"MC→Refine: Phase 1 sampling {n_samples} of {total_combos} "
            f"total combinations"
        )

        samples = self._latin_hypercube_sample(param_ranges, n_samples)
        phase1_results = self._run_phase(
            samples, 1, start_date, end_date, city_names,
            target_metric, progress_cb,
        )

        # Phase 2: Fine grid around best region
        narrowed = self._determine_phase2_ranges(
            phase1_results, param_ranges, target_metric,
        )

        phase2_combos = self._grid_from_ranges(narrowed)
        # Deduplicate combos already tested in Phase 1
        phase1_keys = {self._params_key(r["params"]) for r in phase1_results}
        phase2_combos = [c for c in phase2_combos if self._params_key(c) not in phase1_keys]

        logger.info(
            f"MC→Refine: Phase 2 fine grid has {len(phase2_combos)} "
            f"new combinations (narrowed to {len(narrowed)} params)"
        )

        phase2_results = self._run_phase(
            phase2_combos, 2, start_date, end_date, city_names,
            target_metric, progress_cb,
            best_so_far_init=self._best_metric(phase1_results, target_metric),
        )

        return phase1_results + phase2_results

    # ------------------------------------------------------------------
    #  Coarse Grid → Fine Grid
    # ------------------------------------------------------------------
    def _coarse_fine_search(
        self,
        param_ranges: dict[str, list[float]],
        start_date: str,
        end_date: str,
        target_metric: str,
        city_names: list[str] | None,
        progress_cb: Callable | None,
        phase1_budget: int,
    ) -> list[dict]:
        # Compute coarse step multiplier
        n_params = len(param_ranges)
        total_combos = 1
        for v in param_ranges.values():
            total_combos *= len(v)

        if total_combos <= phase1_budget:
            mult = 1
        else:
            mult = max(2, round((total_combos / phase1_budget) ** (1.0 / n_params)))

        # Build coarsened ranges: take every mult-th value
        coarse_ranges = {}
        for param, values in param_ranges.items():
            coarse_ranges[param] = values[::mult] if mult > 1 else values
            # Always include the last value
            if values[-1] not in coarse_ranges[param]:
                coarse_ranges[param].append(values[-1])

        phase1_combos = self._grid_from_ranges(coarse_ranges)
        logger.info(
            f"Coarse→Fine: Phase 1 with {mult}x step → "
            f"{len(phase1_combos)} combinations"
        )

        phase1_results = self._run_phase(
            phase1_combos, 1, start_date, end_date, city_names,
            target_metric, progress_cb,
        )

        # Phase 2: Narrow and refine
        narrowed = self._determine_phase2_ranges(
            phase1_results, param_ranges, target_metric,
        )

        phase2_combos = self._grid_from_ranges(narrowed)
        phase1_keys = {self._params_key(r["params"]) for r in phase1_results}
        phase2_combos = [c for c in phase2_combos if self._params_key(c) not in phase1_keys]

        logger.info(
            f"Coarse→Fine: Phase 2 fine grid has {len(phase2_combos)} "
            f"new combinations"
        )

        phase2_results = self._run_phase(
            phase2_combos, 2, start_date, end_date, city_names,
            target_metric, progress_cb,
            best_so_far_init=self._best_metric(phase1_results, target_metric),
        )

        return phase1_results + phase2_results

    # ------------------------------------------------------------------
    #  Phase runner (shared by both strategies)
    # ------------------------------------------------------------------
    def _run_phase(
        self,
        combos: list[dict],
        phase: int,
        start_date: str,
        end_date: str,
        city_names: list[str] | None,
        target_metric: str,
        progress_cb: Callable | None,
        best_so_far_init: float = -float("inf"),
    ) -> list[dict]:
        """Run a list of param combos, tagging each with its phase."""
        results = []
        best_so_far = best_so_far_init
        total = len(combos)

        for i, params in enumerate(combos):
            if self._abort_requested:
                raise OptimizationAborted("Aborted by user")

            entry = self._run_single_combo(
                params, start_date, end_date, city_names, target_metric,
            )
            entry["phase"] = phase
            results.append(entry)

            metric_val = entry.get("metrics", {}).get(target_metric, 0)
            if "error" not in entry.get("metrics", {}):
                best_so_far = max(best_so_far, metric_val)

            logger.info(
                f"  P{phase} [{i+1}/{total}] {params} -> "
                f"{target_metric}={metric_val:.4f}  best={best_so_far:.4f}"
            )

            if progress_cb:
                progress_cb(phase, i + 1, total, best_so_far, entry)

        return results

    # ------------------------------------------------------------------
    #  Phase 2 range narrowing
    # ------------------------------------------------------------------
    def _determine_phase2_ranges(
        self,
        phase1_results: list[dict],
        original_ranges: dict[str, list[float]],
        target_metric: str,
        top_fraction: float = 0.10,
        min_top: int = 5,
    ) -> dict[str, list[float]]:
        """Pick top results from Phase 1, compute narrowed ranges."""
        valid = [
            r for r in phase1_results
            if "error" not in r.get("metrics", {})
        ]
        if not valid:
            return original_ranges

        valid.sort(
            key=lambda r: r["metrics"].get(target_metric, 0),
            reverse=True,
        )
        top_n = max(min_top, int(len(valid) * top_fraction))
        top_results = valid[:top_n]

        narrowed: dict[str, list[float]] = {}
        for param, orig_values in original_ranges.items():
            orig_sorted = sorted(orig_values)
            top_vals = [r["params"][param] for r in top_results if param in r["params"]]

            if not top_vals:
                narrowed[param] = orig_sorted
                continue

            lo = min(top_vals)
            hi = max(top_vals)

            # Pad by one step in each direction
            step = (orig_sorted[1] - orig_sorted[0]) if len(orig_sorted) > 1 else 0
            lo = max(orig_sorted[0], lo - step)
            hi = min(orig_sorted[-1], hi + step)

            # Filter original values to narrowed window
            vals = [v for v in orig_sorted if lo - 1e-9 <= v <= hi + 1e-9]
            narrowed[param] = vals if vals else orig_sorted

        return narrowed

    # ------------------------------------------------------------------
    #  Latin Hypercube Sampling (pure Python, no scipy)
    # ------------------------------------------------------------------
    def _latin_hypercube_sample(
        self,
        param_ranges: dict[str, list[float]],
        n_samples: int,
    ) -> list[dict[str, float]]:
        """Generate LHS samples snapped to nearest valid parameter values.

        For each parameter, divides the value space into n_samples strata
        and picks one value from each stratum, then shuffles assignments
        across parameters for even coverage.
        """
        param_names = list(param_ranges.keys())
        strata: dict[str, list[int]] = {}

        for pname in param_names:
            values = param_ranges[pname]
            n_vals = len(values)
            indices = []
            for i in range(n_samples):
                # Map stratum i to an index in [0, n_vals-1]
                pos = (i + rng.random()) * n_vals / n_samples
                idx = min(int(pos), n_vals - 1)
                indices.append(idx)
            rng.shuffle(indices)
            strata[pname] = indices

        samples = []
        for i in range(n_samples):
            sample = {}
            for pname in param_names:
                sample[pname] = param_ranges[pname][strata[pname][i]]
            samples.append(sample)

        return samples

    # ------------------------------------------------------------------
    #  Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _grid_from_ranges(param_ranges: dict[str, list[float]]) -> list[dict]:
        """Build full grid of param dicts from ranges."""
        names = list(param_ranges.keys())
        values = [param_ranges[k] for k in names]
        return [dict(zip(names, combo)) for combo in itertools.product(*values)]

    @staticmethod
    def _params_key(params: dict) -> str:
        """Hashable key for a params dict (for dedup)."""
        return "|".join(f"{k}={v}" for k, v in sorted(params.items()))

    @staticmethod
    def _best_metric(results: list[dict], target_metric: str) -> float:
        """Return the best target_metric value across results."""
        vals = [
            r["metrics"].get(target_metric, -float("inf"))
            for r in results
            if "error" not in r.get("metrics", {})
        ]
        return max(vals) if vals else -float("inf")
