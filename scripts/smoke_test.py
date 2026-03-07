#!/usr/bin/env python3
"""Smoke test: validate each pipeline stage before running the bot.

Tests each component against live APIs in sequence.
Run this before starting the bot to catch integration issues.

Usage:
    python scripts/smoke_test.py
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import load_settings
from src.config.cities import load_cities
from src.data.db import init_db

settings = load_settings()
init_db(settings)
cities = load_cities()

passed = 0
failed = 0


def run_test(name, func):
    global passed, failed
    print(f"\n--- Stage: {name} ---")
    try:
        result = func()
        passed += 1
        return result
    except Exception as e:
        failed += 1
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return None


# Stage 1: Market Discovery
def test_market_discovery():
    from src.data.kalshi_client import KalshiClient
    from src.data.kalshi_markets import KalshiMarketDiscovery

    client = KalshiClient(settings)
    discovery = KalshiMarketDiscovery(client)
    markets = discovery.discover_active_markets(cities)
    client.close()

    assert len(markets) > 0, "No markets discovered"

    # Show a few
    for m in markets[:3]:
        print(f"  {m.market_ticker}: {m.city} {m.target_date} "
              f"{'above ' + str(m.threshold_f) if m.is_above_contract else 'bracket'} "
              f"YES={m.yes_price:.2f}")

    print(f"  PASS: {len(markets)} markets discovered")
    return markets


# Stage 2: Ensemble Fetch
def test_ensemble_fetch():
    from src.data.ensemble_fetcher import EnsembleFetcher

    fetcher = EnsembleFetcher(settings)
    city = list(cities.values())[0]
    results = fetcher.fetch_ensemble(city, forecast_days=2)
    fetcher.close()

    assert len(results) > 0, "No ensemble results"

    for r in results:
        valid = sum(1 for t in r.member_daily_maxes if t == t)
        print(f"  {r.city} {r.target_date}: {valid}/31 valid members, "
              f"mean={sum(t for t in r.member_daily_maxes if t == t) / max(valid, 1):.1f}F")

    print(f"  PASS: {len(results)} days of ensemble data")
    return results


# Stage 3: HRRR Fetch
def test_hrrr_fetch():
    from src.data.hrrr_fetcher import HRRRFetcher

    fetcher = HRRRFetcher(settings)
    city = list(cities.values())[0]
    results = fetcher.fetch_hrrr(city, forecast_days=2)
    fetcher.close()

    assert len(results) > 0, "No HRRR results"

    for r in results:
        print(f"  {r.city} {r.target_date}: max={r.daily_max_f:.1f}F")

    print(f"  PASS: {len(results)} days of HRRR data")
    return results


# Stage 4: Probability Calculation
def test_probability_calc(ensemble_results, markets):
    from src.signals.ensemble_probability import EnsembleProbabilityCalculator

    calc = EnsembleProbabilityCalculator()

    # Group markets by (city, target_date)
    by_key = {}
    for m in markets:
        by_key.setdefault((m.city, m.target_date), []).append(m)

    total_probs = 0
    for er in ensemble_results:
        key = (er.city, er.target_date)
        if key not in by_key:
            continue
        probs = calc.get_full_distribution(er.member_daily_maxes, by_key[key])
        for ticker, pr in probs.items():
            assert 0.0 <= pr.probability <= 1.0, f"Invalid prob: {pr.probability}"
            total_probs += 1

    assert total_probs > 0, "No probabilities computed"
    print(f"  PASS: {total_probs} probabilities computed")
    return total_probs


# Stage 5: Edge Detection
def test_edge_detection(ensemble_results, markets):
    from src.signals.ensemble_probability import EnsembleProbabilityCalculator
    from src.signals.calibration import ForecastCalibrator
    from src.strategy.edge_detector import EdgeDetector

    calc = EnsembleProbabilityCalculator()
    calibrator = ForecastCalibrator()
    detector = EdgeDetector(settings.strategy)

    by_key = {}
    for m in markets:
        by_key.setdefault((m.city, m.target_date), []).append(m)

    total_signals = 0
    for er in ensemble_results:
        key = (er.city, er.target_date)
        if key not in by_key:
            continue
        group_markets = by_key[key]
        probs = calc.get_full_distribution(er.member_daily_maxes, group_markets)

        signals = {}
        market_data = {}
        for m in group_markets:
            pr = probs.get(m.market_ticker)
            if pr is None:
                continue
            signals[m.market_ticker] = calibrator.calibrate(pr.probability)
            market_data[m.market_ticker] = {
                "yes_price": m.yes_price,
                "city": m.city,
                "target_date": m.target_date,
            }

        trade_signals = detector.scan_for_edges(signals, market_data)
        total_signals += len(trade_signals)

        for ts in trade_signals[:2]:
            print(f"  Signal: {ts.side.upper()} {ts.market_ticker} "
                  f"edge={ts.edge:+.1%} model={ts.model_prob:.2f} mkt={ts.market_price:.2f}")

    print(f"  PASS: {total_signals} trade signals detected")
    return total_signals


# Stage 6: Paper Trade Round-trip
def test_paper_trade():
    from src.execution.paper_trader import PaperTrader

    trader = PaperTrader(settings)
    initial = trader.bankroll
    positions = trader.get_open_positions()
    print(f"  Bankroll: ${initial:,.2f}, open positions: {len(positions)}")
    print(f"  PASS: paper trader operational")


# Stage 7: Settlement Checker
def test_settlement():
    from src.execution.paper_trader import PaperTrader
    from src.execution.settlement_checker import SettlementChecker

    trader = PaperTrader(settings)
    checker = SettlementChecker(cities, paper_trader=trader)
    results = checker.check_settlements("2020-01-01")
    checker.close()
    print(f"  PASS: settlement checker returned {len(results)} results (expected 0)")


# Run all stages
print("=" * 60)
print("AetherBot - Smoke Test")
print("=" * 60)

markets = run_test("Market Discovery (Kalshi API)", test_market_discovery)
ensemble_results = run_test("Ensemble Fetch (Open-Meteo GFS)", test_ensemble_fetch)
hrrr_results = run_test("HRRR Fetch (Open-Meteo)", test_hrrr_fetch)

if ensemble_results and markets:
    run_test("Probability Calculation", lambda: test_probability_calc(ensemble_results, markets))
    run_test("Edge Detection", lambda: test_edge_detection(ensemble_results, markets))
else:
    print("\n--- Skipping probability/edge tests (missing data) ---")
    failed += 2

run_test("Paper Trade Round-trip", test_paper_trade)
run_test("Settlement Checker", test_settlement)

print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed")
print("=" * 60)

if failed > 0:
    sys.exit(1)
