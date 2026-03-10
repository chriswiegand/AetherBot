"""AetherBot Dashboard - Flask web server.

Serves the dashboard UI and provides JSON API endpoints for
trade monitoring, PnL tracking, signal inspection, market data,
strategy management, backtesting, and parameter optimization.

Usage:
    python dashboard/app.py
"""

import json
import logging
import sqlite3
import sys
import threading
import time
from datetime import datetime, date, timezone
from pathlib import Path

import queue
import uuid

from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

# Add project root to path so we can import src modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import load_settings
from src.config.cities import load_cities
from src.data.ensemble_fetcher import EnsembleFetcher
from src.data.hrrr_fetcher import HRRRFetcher
from src.data.nws_client import NWSClient
from src.data.freshness import DataFreshnessTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DB_PATH = PROJECT_ROOT / "data" / "weather_bot.db"
STATIC_DIR = Path(__file__).resolve().parent  # dashboard/

INITIAL_BANKROLL = 10_000.0

# ---------------------------------------------------------------------------
# Kalshi client (lazy-loaded, only in live mode)
# ---------------------------------------------------------------------------
_kalshi_client = None
_kalshi_lock = threading.Lock()


def _get_kalshi_balance() -> float | None:
    """Fetch real Kalshi balance. Returns None if not in live mode or on error."""
    global _kalshi_client
    try:
        settings = load_settings()
        if settings.mode != "live":
            return None
        with _kalshi_lock:
            if _kalshi_client is None:
                from src.data.kalshi_client import KalshiClient
                _kalshi_client = KalshiClient(settings)
            return _kalshi_client.get_balance()
    except Exception as e:
        logger.warning("Failed to fetch Kalshi balance: %s", e)
        return None

# ---------------------------------------------------------------------------
# Direct-fetch infrastructure (dashboard calls fetchers without the bot)
# ---------------------------------------------------------------------------

_refresh_locks = {s: threading.Lock() for s in ("gfs", "hrrr", "nws", "markets")}
_refresh_status: dict[str, dict] = {
    s: {"running": False, "last_result": None} for s in ("gfs", "hrrr", "nws", "markets")
}

# Optimization job state for async (SSE) optimization runs
_opt_jobs: dict[str, dict] = {}
_opt_lock = threading.Lock()


def _do_direct_fetch(source: str):
    """Run a data fetch in a background thread.

    Each fetcher is standalone — no BotContext needed.
    Creates its own fetcher instance, iterates all 5 cities,
    and closes the HTTP client when done.
    """
    lock = _refresh_locks[source]
    if not lock.acquire(blocking=False):
        logger.info("Refresh for %s already in progress — skipping", source)
        return

    _refresh_status[source]["running"] = True
    try:
        settings = load_settings()
        cities = load_cities()
        total_records = 0
        errors = []

        if source == "gfs":
            fetcher = EnsembleFetcher(settings)
            try:
                for city_name, city_config in cities.items():
                    try:
                        results = fetcher.fetch_and_store(city_config)
                        total_records += len(results) if results else 0
                    except Exception as e:
                        errors.append(f"{city_name}: {e}")
                        logger.error("GFS fetch failed for %s: %s", city_name, e)
            finally:
                fetcher.close()

        elif source == "hrrr":
            fetcher = HRRRFetcher(settings)
            try:
                for city_name, city_config in cities.items():
                    try:
                        results = fetcher.fetch_and_store(city_config)
                        total_records += len(results) if results else 0
                    except Exception as e:
                        errors.append(f"{city_name}: {e}")
                        logger.error("HRRR fetch failed for %s: %s", city_name, e)
            finally:
                fetcher.close()

        elif source == "nws":
            client = NWSClient(settings)
            try:
                for city_name, city_config in cities.items():
                    try:
                        results = client.fetch_and_store(city_config)
                        total_records += len(results) if results else 0
                    except Exception as e:
                        errors.append(f"{city_name}: {e}")
                        logger.error("NWS fetch failed for %s: %s", city_name, e)
            finally:
                client.close()

        elif source == "markets":
            try:
                from src.data.kalshi_client import KalshiClient
                from src.data.kalshi_markets import KalshiMarketDiscovery
                kalshi = KalshiClient(settings)
                discovery = KalshiMarketDiscovery(kalshi)
                markets = discovery.discover_active_markets(cities)
                discovery.store_markets(markets)
                markets = discovery.refresh_prices(markets)
                total_records = len(markets)
                kalshi.close()
            except Exception as e:
                errors.append(f"markets: {e}")
                logger.error("Market refresh failed: %s", e)

        _refresh_status[source]["last_result"] = {
            "records": total_records,
            "errors": errors,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("Direct refresh %s complete: %d records, %d errors",
                     source, total_records, len(errors))

    except Exception as e:
        _refresh_status[source]["last_result"] = {"error": str(e)}
        logger.error("Direct refresh %s failed: %s", source, e)
    finally:
        _refresh_status[source]["running"] = False
        lock.release()


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder=None)


# ---------------------------------------------------------------------------
# CORS (simple approach - add header to every response)
# ---------------------------------------------------------------------------

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    return response


# ---------------------------------------------------------------------------
# Database helper
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    """Open a read-only SQLite connection with Row factory."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def get_db_rw() -> sqlite3.Connection:
    """Open a read-write SQLite connection with Row factory."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict]:
    """Convert sqlite3.Row objects to plain dicts for JSON serialization."""
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Static file serving
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the dashboard index.html."""
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/whitepaper")
def whitepaper():
    """Serve the condensed whitepaper page."""
    return send_from_directory(STATIC_DIR, "whitepaper.html")


@app.route("/strategy-lab")
def strategy_lab():
    """Serve the Strategy Lab page."""
    return send_from_directory(STATIC_DIR, "strategy-lab.html")


@app.route("/math")
def math_page():
    """Serve the Math (pipeline visualization) page."""
    return send_from_directory(STATIC_DIR, "math.html")


@app.route("/tracker")
def tracker_page():
    """Serve the Position Tracker page."""
    return send_from_directory(STATIC_DIR, "tracker.html")


@app.route("/convergence")
def convergence_page():
    """Serve the Temperature + Probability Convergence page."""
    return send_from_directory(STATIC_DIR, "convergence.html")


@app.route("/<path:filename>")
def static_files(filename):
    """Serve other static assets (CSS, JS, images) from dashboard/."""
    return send_from_directory(STATIC_DIR, filename)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


# ---------------------------------------------------------------------------
# API: Overall status
# ---------------------------------------------------------------------------

@app.route("/api/status")
def api_status():
    """Overall dashboard stats: bankroll, open positions, today's trades,
    win rate, total PnL, and Brier score."""
    try:
        conn = get_db()
        cur = conn.cursor()
        today_str = date.today().isoformat()

        # Settled PnL
        cur.execute(
            "SELECT COALESCE(SUM(pnl), 0) AS total_pnl FROM trades WHERE status = 'settled'"
        )
        total_pnl = cur.fetchone()["total_pnl"]

        # Open position cost
        cur.execute(
            "SELECT COALESCE(SUM(total_cost), 0) AS open_cost "
            "FROM trades WHERE status = 'filled'"
        )
        open_cost = cur.fetchone()["open_cost"]

        # Bankroll: real Kalshi balance in live mode, computed in paper mode
        live_balance = _get_kalshi_balance()
        if live_balance is not None:
            bankroll = live_balance
        else:
            bankroll = INITIAL_BANKROLL + total_pnl - open_cost

        # Open positions count
        cur.execute("SELECT COUNT(*) AS cnt FROM trades WHERE status = 'filled'")
        open_positions = cur.fetchone()["cnt"]

        # Today's trade count
        cur.execute(
            "SELECT COUNT(*) AS cnt FROM trades WHERE created_at >= ?",
            (today_str,),
        )
        today_trades = cur.fetchone()["cnt"]

        # Win rate (settled trades with pnl > 0)
        cur.execute(
            "SELECT COUNT(*) AS total FROM trades WHERE status = 'settled'"
        )
        settled_total = cur.fetchone()["total"]

        cur.execute(
            "SELECT COUNT(*) AS wins FROM trades WHERE status = 'settled' AND pnl > 0"
        )
        wins = cur.fetchone()["wins"]

        win_rate = (wins / settled_total) if settled_total > 0 else 0.0

        # Brier score (average of brier_contribution across all records)
        cur.execute(
            "SELECT AVG(brier_contribution) AS avg_brier FROM brier_scores"
        )
        row = cur.fetchone()
        brier_score = row["avg_brier"] if row["avg_brier"] is not None else None

        conn.close()

        return jsonify({
            "bankroll": round(bankroll, 2),
            "open_positions": open_positions,
            "today_trades": today_trades,
            "win_rate": round(win_rate, 4),
            "total_pnl": round(total_pnl, 2),
            "brier_score": round(brier_score, 4) if brier_score is not None else None,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# API: Open positions
# ---------------------------------------------------------------------------

@app.route("/api/positions")
def api_positions():
    """List all open (filled) positions with mark-to-market data."""
    try:
        conn = get_db()
        cur = conn.cursor()

        # 1a. Main query: trades JOIN kalshi_markets for current prices
        #     Use COALESCE(fill_price, price) so actual fill price takes priority
        cur.execute(
            """
            SELECT
                t.market_ticker, t.side, t.contracts,
                COALESCE(t.fill_price, t.price) AS entry_price,
                t.total_cost, t.city,
                t.target_date, t.edge AS edge_at_entry,
                t.model_prob, t.created_at,
                km.yes_price AS current_yes_price,
                km.no_price  AS current_no_price,
                km.close_time, km.status AS market_status
            FROM trades t
            LEFT JOIN kalshi_markets km
                ON km.market_ticker = t.market_ticker
            WHERE t.status = 'filled'
            ORDER BY t.target_date ASC, t.city ASC
            """
        )
        rows = cur.fetchall()
        if not rows:
            conn.close()
            return jsonify([])

        tickers = list({r["market_ticker"] for r in rows})
        placeholders = ",".join("?" * len(tickers))

        # 1b. Batch-fetch latest signal per open ticker
        cur.execute(
            f"""
            SELECT s.market_ticker, s.calibrated_prob, s.lead_hours
            FROM signals s
            INNER JOIN (
                SELECT market_ticker, MAX(computed_at) AS max_ct
                FROM signals
                WHERE market_ticker IN ({placeholders})
                GROUP BY market_ticker
            ) latest
                ON s.market_ticker = latest.market_ticker
               AND s.computed_at   = latest.max_ct
            """,
            tickers,
        )
        sig_rows = cur.fetchall()
        signals_by_ticker = {
            sr["market_ticker"]: {
                "calibrated_prob": sr["calibrated_prob"],
                "lead_hours": sr["lead_hours"],
            }
            for sr in sig_rows
        }

        # 1c. Batch-fetch sparkline price history
        cur.execute(
            f"""
            SELECT market_ticker, yes_price
            FROM market_price_history
            WHERE market_ticker IN ({placeholders})
            ORDER BY market_ticker, captured_at ASC
            """,
            tickers,
        )
        raw_spark = {}
        for sr in cur.fetchall():
            raw_spark.setdefault(sr["market_ticker"], []).append(sr["yes_price"])

        # Downsample to ≤12 points per ticker
        sparkline_data = {}
        for tk, vals in raw_spark.items():
            if len(vals) <= 12:
                sparkline_data[tk] = vals
            else:
                step = len(vals) / 12.0
                sparkline_data[tk] = [vals[int(i * step)] for i in range(12)]

        conn.close()

        # 1d. Compute mark-to-market per position
        positions = []
        for r in rows:
            entry_price = r["entry_price"] or 0
            side = r["side"] or "yes"
            contracts = r["contracts"] or 0
            current_yes = r["current_yes_price"]
            edge_at_entry = r["edge_at_entry"] or 0

            # Defaults when no current market price available
            current_price = None
            price_move = None
            unrealized_pnl = None
            unrealized_pnl_pct = None
            current_edge = None
            edge_trend = None
            recommendation = None

            if current_yes is not None:
                # Side-adjusted pricing
                if side == "yes":
                    current_price = current_yes
                    cost = entry_price * contracts
                    current_value = current_yes * contracts
                    price_move = current_yes - entry_price
                else:
                    current_price = 1.0 - current_yes
                    cost = entry_price * contracts  # entry_price IS the NO price paid
                    current_value = (1.0 - current_yes) * contracts
                    price_move = (1.0 - current_yes) - entry_price  # NO price change

                unrealized_pnl = round(current_value - cost, 2)
                unrealized_pnl_pct = (
                    round((current_value - cost) / cost * 100, 1)
                    if cost > 0
                    else 0
                )

                # Current edge from latest signal
                sig = signals_by_ticker.get(r["market_ticker"])
                model_prob = sig["calibrated_prob"] if sig else None
                if model_prob is not None:
                    if side == "yes":
                        current_edge = round(model_prob - current_yes, 4)
                    else:
                        current_edge = round(
                            (1.0 - model_prob) - (1.0 - current_yes), 4
                        )

                    # Edge trend vs entry
                    if edge_at_entry:
                        if abs(current_edge) > abs(edge_at_entry) + 0.01:
                            edge_trend = "improving"
                        elif abs(current_edge) < abs(edge_at_entry) - 0.01:
                            edge_trend = "worsening"
                        else:
                            edge_trend = "stable"

                    # Simplified recommendation
                    if side == "yes" and current_yes > model_prob:
                        recommendation = "SELL"
                    elif side == "no" and (1.0 - current_yes) > (
                        1.0 - model_prob
                    ):
                        recommendation = "SELL"
                    elif (
                        unrealized_pnl < 0
                        and current_edge is not None
                        and abs(current_edge) < 0.03
                    ):
                        recommendation = "SELL"
                    else:
                        recommendation = "HOLD"

                current_price = round(current_price, 4) if current_price else None
                price_move = round(price_move, 4) if price_move is not None else None

            # entry_price is already in the traded side's denomination
            # (YES price for YES trades, NO price for NO trades)
            display_entry = round(entry_price, 4)

            positions.append(
                {
                    "market_ticker": r["market_ticker"],
                    "side": side,
                    "contracts": contracts,
                    "entry_price": display_entry,
                    "current_price": current_price,
                    "cost": round(r["total_cost"] or 0, 2),
                    "city": r["city"],
                    "target_date": r["target_date"],
                    "edge_at_entry": edge_at_entry,
                    "current_edge": current_edge,
                    "edge_trend": edge_trend,
                    "price_move": price_move,
                    "unrealized_pnl": unrealized_pnl,
                    "unrealized_pnl_pct": unrealized_pnl_pct,
                    "recommendation": recommendation,
                    "model_prob": r["model_prob"],
                    "sparkline": sparkline_data.get(r["market_ticker"], []),
                }
            )

        return jsonify(positions)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# API: Recent trades
# ---------------------------------------------------------------------------

@app.route("/api/trades/recent")
def api_recent_trades():
    """Last 50 trades across all statuses."""
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                created_at,
                market_ticker,
                side,
                contracts,
                COALESCE(fill_price, price) AS price,
                pnl,
                status,
                edge,
                city
            FROM trades
            ORDER BY created_at DESC
            LIMIT 50
            """
        )
        rows = cur.fetchall()
        conn.close()

        trades = []
        for r in rows:
            trades.append({
                "timestamp": r["created_at"],
                "market_ticker": r["market_ticker"],
                "side": r["side"],
                "contracts": r["contracts"],
                "price": r["price"],
                "pnl": r["pnl"],
                "status": r["status"],
                "edge": r["edge"],
                "city": r["city"],
            })

        return jsonify(trades)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# API: Reset paper trades
# ---------------------------------------------------------------------------

@app.route("/api/trades/reset", methods=["POST"])
def api_reset_trades():
    """Reset paper trades. Accepts optional scope=today (default: all)."""
    try:
        scope = request.args.get("scope", "all")
        conn = get_db_rw()
        cur = conn.cursor()

        if scope == "today":
            today_str = date.today().isoformat()
            cur.execute("SELECT COUNT(*) FROM trades WHERE created_at >= ?", (today_str,))
            trades_count = cur.fetchone()[0]
            cur.execute("DELETE FROM trades WHERE created_at >= ?", (today_str,))
            cur.execute("DELETE FROM signals WHERE computed_at >= ?", (today_str,))
            conn.commit()
            conn.close()
            return jsonify({
                "status": "ok",
                "scope": "today",
                "trades_deleted": trades_count,
            })
        else:
            # Count before
            cur.execute("SELECT COUNT(*) FROM trades")
            trades_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM signals")
            signals_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM brier_scores")
            brier_count = cur.fetchone()[0]

            # Clear tables
            cur.execute("DELETE FROM trades")
            cur.execute("DELETE FROM signals")
            cur.execute("DELETE FROM brier_scores")
            conn.commit()
            conn.close()

            return jsonify({
                "status": "ok",
                "scope": "all",
                "trades_deleted": trades_count,
                "signals_deleted": signals_count,
                "brier_deleted": brier_count,
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# API: Daily PnL aggregation (for charting)
# ---------------------------------------------------------------------------

@app.route("/api/pnl/daily")
def api_daily_pnl():
    """Daily PnL aggregation computed from settled trades, with running
    cumulative PnL for equity-curve charting."""
    try:
        conn = get_db()
        cur = conn.cursor()

        # Aggregate settled trades by target_date
        cur.execute(
            """
            SELECT
                target_date AS date,
                COALESCE(SUM(pnl), 0) AS pnl,
                COUNT(*) AS trade_count,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS win_count
            FROM trades
            WHERE status = 'settled'
            GROUP BY target_date
            ORDER BY target_date ASC
            """
        )
        rows = cur.fetchall()
        conn.close()

        cumulative = 0.0
        result = []
        for r in rows:
            cumulative += r["pnl"]
            result.append({
                "date": r["date"],
                "pnl": round(r["pnl"], 2),
                "cumulative_pnl": round(cumulative, 2),
                "trade_count": r["trade_count"],
                "win_count": r["win_count"],
            })

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# API: Latest signals
# ---------------------------------------------------------------------------

@app.route("/api/signals/latest")
def api_latest_signals():
    """Most recent signal per market_ticker with Kelly-sized recommendation."""
    try:
        conn = get_db()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT
                s.market_ticker,
                s.city,
                s.target_date,
                s.ensemble_prob,
                s.blended_prob,
                s.calibrated_prob,
                s.raw_edge AS edge,
                s.market_yes_price,
                s.lead_hours,
                s.computed_at,
                km.yes_price AS yes_bid,
                km.no_price AS no_bid
            FROM signals s
            INNER JOIN (
                SELECT market_ticker, MAX(computed_at) AS max_computed
                FROM signals
                GROUP BY market_ticker
            ) latest
                ON s.market_ticker = latest.market_ticker
               AND s.computed_at = latest.max_computed
            LEFT JOIN kalshi_markets km
                ON km.market_ticker = s.market_ticker
            ORDER BY ABS(s.raw_edge) DESC, s.target_date ASC, s.city ASC
            """
        )
        rows = cur.fetchall()
        conn.close()

        # Compute Kelly sizing for each signal
        settings = load_settings()
        from src.strategy.kelly_sizer import KellySizer
        from src.strategy.edge_detector import TradeSignal
        sizer = KellySizer(settings.strategy)
        live_balance = _get_kalshi_balance()
        bankroll = live_balance if live_balance is not None else INITIAL_BANKROLL

        signals = []
        for r in rows:
            model_prob = r["calibrated_prob"] or r["blended_prob"] or r["ensemble_prob"]
            market_price = r["market_yes_price"]
            edge = r["edge"]

            # Determine side
            if model_prob is not None and market_price is not None:
                side = "yes" if (model_prob - market_price) > 0 else "no"
            else:
                side = "yes"

            # Build a TradeSignal for the sizer
            rec = {"side": side, "contracts": 0, "price_cents": 0,
                   "total_cost": 0, "kelly": 0, "tradeable": False}
            if model_prob and market_price and edge:
                try:
                    sig = TradeSignal(
                        market_ticker=r["market_ticker"],
                        city=r["city"] or "",
                        target_date=r["target_date"] or "",
                        side=side,
                        model_prob=model_prob,
                        market_price=market_price,
                        edge=edge,
                        abs_edge=abs(edge),
                        lead_hours=r["lead_hours"] or 0,
                        confidence="medium",
                    )
                    pos = sizer.calculate_position_size(sig, bankroll)
                    rec = {
                        "side": side,
                        "contracts": pos.contracts,
                        "price_cents": pos.price_cents,
                        "total_cost": round(pos.total_cost, 2),
                        "kelly": round(pos.kelly_fraction, 4),
                        "tradeable": pos.contracts > 0 and abs(edge) >= settings.strategy.edge_threshold,
                    }
                except Exception:
                    pass

            # Bid/ask: yes_bid is what you get buying YES, no_bid for NO
            yes_bid = r["yes_bid"]
            no_bid = r["no_bid"]
            # yes_ask = complement of no_bid (what you pay to buy YES)
            yes_ask = round(1.0 - no_bid, 2) if no_bid and no_bid > 0 else None
            no_ask = round(1.0 - yes_bid, 2) if yes_bid and yes_bid > 0 else None

            signals.append({
                "market_ticker": r["market_ticker"],
                "city": r["city"],
                "target_date": r["target_date"],
                "ensemble_prob": r["ensemble_prob"],
                "blended_prob": r["blended_prob"],
                "calibrated_prob": r["calibrated_prob"],
                "edge": edge,
                "market_price": market_price,
                "lead_hours": r["lead_hours"],
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "no_bid": no_bid,
                "no_ask": no_ask,
                "rec": rec,
            })

        return jsonify(signals)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# API: Active markets
# ---------------------------------------------------------------------------

@app.route("/api/markets")
def api_markets():
    """Active (open) Kalshi markets."""
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                market_ticker AS ticker,
                city,
                target_date,
                is_above_contract,
                threshold_f AS threshold,
                bracket_low,
                bracket_high,
                yes_price,
                no_price
            FROM kalshi_markets
            WHERE status IN ('open', 'active')
            ORDER BY target_date ASC, city ASC, threshold_f ASC
            """
        )
        rows = cur.fetchall()
        conn.close()

        return jsonify(rows_to_dicts(rows))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===========================================================================
# STRATEGY MANAGEMENT API
# ===========================================================================

STRATEGY_FIELDS = [
    "name", "description", "edge_threshold", "min_edge_hrrr_confirm",
    "min_model_prob", "fractional_kelly", "max_position_pct",
    "max_position_dollars", "daily_loss_limit", "max_concurrent_positions",
    "max_positions_per_city", "max_positions_per_date",
    "min_price", "max_price", "max_lead_hours",
]

STRATEGY_FLOAT_FIELDS = {
    "edge_threshold", "min_edge_hrrr_confirm", "min_model_prob",
    "fractional_kelly", "max_position_pct", "max_position_dollars",
    "daily_loss_limit", "min_price", "max_price", "max_lead_hours",
}

STRATEGY_INT_FIELDS = {
    "max_concurrent_positions", "max_positions_per_city", "max_positions_per_date",
}


def _ensure_hourly_observations_table(conn):
    """Create hourly_observations table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hourly_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT NOT NULL,
            station TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            temperature_f REAL,
            description TEXT,
            fetched_at TEXT NOT NULL,
            UNIQUE(station, observed_at)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_hourly_obs_city_time
        ON hourly_observations(city, observed_at)
    """)


def _ensure_strategies_table(conn):
    """Create strategies table if it doesn't exist (lightweight migration)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT DEFAULT '',
            edge_threshold REAL DEFAULT 0.08,
            min_edge_hrrr_confirm REAL DEFAULT 0.06,
            min_model_prob REAL DEFAULT 0.55,
            fractional_kelly REAL DEFAULT 0.15,
            max_position_pct REAL DEFAULT 0.10,
            max_position_dollars REAL DEFAULT 1000,
            daily_loss_limit REAL DEFAULT 300,
            max_concurrent_positions INTEGER DEFAULT 20,
            max_positions_per_city INTEGER DEFAULT 6,
            max_positions_per_date INTEGER DEFAULT 4,
            min_price REAL DEFAULT 0.08,
            max_price REAL DEFAULT 0.92,
            max_lead_hours REAL DEFAULT 72,
            is_active INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)


def _ensure_backtest_table(conn):
    """Create backtest_runs table if needed."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS backtest_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id INTEGER NOT NULL REFERENCES strategies(id),
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            cities TEXT,
            total_trades INTEGER,
            win_rate REAL,
            gross_pnl REAL,
            sharpe_ratio REAL,
            max_drawdown REAL,
            profit_factor REAL,
            brier_score REAL,
            avg_edge REAL,
            trades_json TEXT,
            daily_pnl_json TEXT,
            created_at TEXT NOT NULL,
            duration_seconds REAL
        )
    """)


def _ensure_optimization_table(conn):
    """Create optimization_runs table if needed."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS optimization_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            param_ranges_json TEXT,
            target_metric TEXT,
            total_combinations INTEGER,
            results_json TEXT,
            best_params_json TEXT,
            best_by_city_json TEXT,
            created_at TEXT NOT NULL,
            duration_seconds REAL
        )
    """)
    # Add columns if missing (migrate existing tables)
    cur = conn.execute("PRAGMA table_info(optimization_runs)")
    cols = {row[1] for row in cur.fetchall()}
    if "best_by_city_json" not in cols:
        conn.execute("ALTER TABLE optimization_runs ADD COLUMN best_by_city_json TEXT")
    if "strategy_type" not in cols:
        conn.execute("ALTER TABLE optimization_runs ADD COLUMN strategy_type TEXT DEFAULT 'full_grid'")


def _ensure_strategy_id_column(conn):
    """Add strategy_id column to trades table if missing."""
    cur = conn.execute("PRAGMA table_info(trades)")
    cols = {row[1] for row in cur.fetchall()}
    if "strategy_id" not in cols:
        conn.execute("ALTER TABLE trades ADD COLUMN strategy_id INTEGER REFERENCES strategies(id)")


@app.route("/api/strategies", methods=["GET"])
def api_list_strategies():
    """List all strategies."""
    try:
        conn = get_db_rw()
        _ensure_strategies_table(conn)
        cur = conn.cursor()
        cur.execute("SELECT * FROM strategies ORDER BY created_at DESC")
        rows = cur.fetchall()
        conn.close()
        return jsonify(rows_to_dicts(rows))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/strategies", methods=["POST"])
def api_create_strategy():
    """Create a new strategy."""
    try:
        data = request.get_json(force=True)
        name = data.get("name", "").strip()
        if not name:
            return jsonify({"error": "Strategy name is required"}), 400

        conn = get_db_rw()
        _ensure_strategies_table(conn)
        cur = conn.cursor()
        now = datetime.utcnow().isoformat() + "Z"

        cols = ["name", "created_at", "updated_at"]
        vals = [name, now, now]

        for field in STRATEGY_FIELDS:
            if field == "name":
                continue
            if field in data:
                cols.append(field)
                if field in STRATEGY_FLOAT_FIELDS:
                    vals.append(float(data[field]))
                elif field in STRATEGY_INT_FIELDS:
                    vals.append(int(data[field]))
                else:
                    vals.append(str(data[field]))

        placeholders = ",".join(["?"] * len(vals))
        col_names = ",".join(cols)
        cur.execute(f"INSERT INTO strategies ({col_names}) VALUES ({placeholders})", vals)
        strategy_id = cur.lastrowid
        conn.commit()

        # Fetch the created strategy
        cur.execute("SELECT * FROM strategies WHERE id = ?", (strategy_id,))
        row = cur.fetchone()
        conn.close()
        return jsonify(dict(row)), 201

    except sqlite3.IntegrityError:
        return jsonify({"error": f"Strategy name '{name}' already exists"}), 409
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/strategies/<int:sid>", methods=["GET"])
def api_get_strategy(sid):
    """Get a single strategy."""
    try:
        conn = get_db_rw()
        _ensure_strategies_table(conn)
        cur = conn.cursor()
        cur.execute("SELECT * FROM strategies WHERE id = ?", (sid,))
        row = cur.fetchone()
        conn.close()
        if row is None:
            return jsonify({"error": "Strategy not found"}), 404
        return jsonify(dict(row))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/strategies/<int:sid>", methods=["PUT"])
def api_update_strategy(sid):
    """Update strategy parameters."""
    try:
        data = request.get_json(force=True)
        conn = get_db_rw()
        _ensure_strategies_table(conn)
        cur = conn.cursor()

        # Check exists
        cur.execute("SELECT id FROM strategies WHERE id = ?", (sid,))
        if cur.fetchone() is None:
            conn.close()
            return jsonify({"error": "Strategy not found"}), 404

        now = datetime.utcnow().isoformat() + "Z"
        sets = ["updated_at = ?"]
        vals = [now]

        for field in STRATEGY_FIELDS:
            if field in data:
                sets.append(f"{field} = ?")
                if field in STRATEGY_FLOAT_FIELDS:
                    vals.append(float(data[field]))
                elif field in STRATEGY_INT_FIELDS:
                    vals.append(int(data[field]))
                else:
                    vals.append(str(data[field]))

        vals.append(sid)
        cur.execute(f"UPDATE strategies SET {', '.join(sets)} WHERE id = ?", vals)
        conn.commit()

        cur.execute("SELECT * FROM strategies WHERE id = ?", (sid,))
        row = cur.fetchone()
        conn.close()
        return jsonify(dict(row))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/strategies/<int:sid>", methods=["DELETE"])
def api_delete_strategy(sid):
    """Delete a strategy."""
    try:
        conn = get_db_rw()
        _ensure_strategies_table(conn)
        cur = conn.cursor()
        cur.execute("DELETE FROM strategies WHERE id = ?", (sid,))
        deleted = cur.rowcount
        conn.commit()
        conn.close()
        if deleted == 0:
            return jsonify({"error": "Strategy not found"}), 404
        return jsonify({"status": "ok", "deleted_id": sid})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/strategies/<int:sid>/activate", methods=["POST"])
def api_activate_strategy(sid):
    """Set a strategy as the active one for the bot."""
    try:
        conn = get_db_rw()
        _ensure_strategies_table(conn)
        cur = conn.cursor()
        cur.execute("SELECT id FROM strategies WHERE id = ?", (sid,))
        if cur.fetchone() is None:
            conn.close()
            return jsonify({"error": "Strategy not found"}), 404

        # Deactivate all, then activate this one
        cur.execute("UPDATE strategies SET is_active = 0")
        cur.execute("UPDATE strategies SET is_active = 1 WHERE id = ?", (sid,))
        conn.commit()
        conn.close()
        return jsonify({"status": "ok", "active_id": sid})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/strategies/<int:sid>/trades", methods=["GET"])
def api_strategy_trades(sid):
    """Get trades for a specific strategy."""
    try:
        conn = get_db_rw()
        _ensure_strategies_table(conn)
        _ensure_strategy_id_column(conn)
        cur = conn.cursor()
        cur.execute(
            """SELECT created_at, market_ticker, side, contracts, price,
                      pnl, status, edge, city
               FROM trades WHERE strategy_id = ?
               ORDER BY created_at DESC LIMIT 100""",
            (sid,),
        )
        rows = cur.fetchall()
        conn.close()
        return jsonify(rows_to_dicts(rows))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/strategies/<int:sid>/status", methods=["GET"])
def api_strategy_status(sid):
    """Get performance metrics for a strategy's trades."""
    try:
        conn = get_db_rw()
        _ensure_strategies_table(conn)
        _ensure_strategy_id_column(conn)
        cur = conn.cursor()

        cur.execute(
            "SELECT COUNT(*) AS total FROM trades WHERE strategy_id = ?", (sid,)
        )
        total = cur.fetchone()["total"]

        cur.execute(
            "SELECT COUNT(*) AS wins FROM trades WHERE strategy_id = ? AND status = 'settled' AND pnl > 0",
            (sid,),
        )
        wins = cur.fetchone()["wins"]

        cur.execute(
            "SELECT COUNT(*) AS settled FROM trades WHERE strategy_id = ? AND status = 'settled'",
            (sid,),
        )
        settled = cur.fetchone()["settled"]

        cur.execute(
            "SELECT COALESCE(SUM(pnl), 0) AS pnl FROM trades WHERE strategy_id = ? AND status = 'settled'",
            (sid,),
        )
        pnl = cur.fetchone()["pnl"]

        conn.close()
        return jsonify({
            "total_trades": total,
            "settled_trades": settled,
            "winning_trades": wins,
            "win_rate": round(wins / settled, 4) if settled > 0 else 0,
            "total_pnl": round(pnl, 2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===========================================================================
# BACKTEST API
# ===========================================================================

@app.route("/api/backtest/run", methods=["POST"])
def api_run_backtest():
    """Run a backtest for a strategy. Synchronous (typically 5-30s)."""
    try:
        data = request.get_json(force=True)
        strategy_id = data.get("strategy_id")
        start_date = data.get("start_date")
        end_date = data.get("end_date")

        if not strategy_id or not start_date or not end_date:
            return jsonify({"error": "strategy_id, start_date, end_date required"}), 400

        conn = get_db_rw()
        _ensure_strategies_table(conn)
        _ensure_backtest_table(conn)
        cur = conn.cursor()

        cur.execute("SELECT * FROM strategies WHERE id = ?", (strategy_id,))
        strat_row = cur.fetchone()
        if strat_row is None:
            conn.close()
            return jsonify({"error": "Strategy not found"}), 404

        strat = dict(strat_row)
        conn.close()

        # Import and run backtest
        from src.backtest.strategy_runner import run_strategy_backtest
        t0 = time.time()
        result = run_strategy_backtest(strat, start_date, end_date)
        duration = time.time() - t0

        # Store results
        conn = get_db_rw()
        _ensure_backtest_table(conn)
        cur = conn.cursor()
        now = datetime.utcnow().isoformat() + "Z"

        perf = result.get("performance", {})
        cur.execute(
            """INSERT INTO backtest_runs
               (strategy_id, start_date, end_date, cities,
                total_trades, win_rate, gross_pnl, sharpe_ratio,
                max_drawdown, profit_factor, brier_score, avg_edge,
                trades_json, daily_pnl_json, created_at, duration_seconds)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                strategy_id, start_date, end_date,
                json.dumps(result.get("cities", [])),
                perf.get("total_trades", 0),
                perf.get("win_rate", 0),
                perf.get("gross_pnl", 0),
                perf.get("sharpe_ratio", 0),
                perf.get("max_drawdown", 0),
                perf.get("profit_factor", 0),
                result.get("brier_score"),
                perf.get("avg_edge", 0),
                json.dumps(result.get("trades", [])),
                json.dumps(result.get("daily_pnl", {})),
                now, duration,
            ),
        )
        run_id = cur.lastrowid
        conn.commit()

        cur.execute("SELECT * FROM backtest_runs WHERE id = ?", (run_id,))
        row = cur.fetchone()
        conn.close()

        return jsonify(dict(row)), 201

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtest/runs", methods=["GET"])
def api_list_backtest_runs():
    """List all backtest runs."""
    try:
        conn = get_db_rw()
        _ensure_backtest_table(conn)
        cur = conn.cursor()
        strategy_id = request.args.get("strategy_id")
        if strategy_id:
            cur.execute(
                """SELECT id, strategy_id, start_date, end_date, cities,
                          total_trades, win_rate, gross_pnl, sharpe_ratio,
                          max_drawdown, profit_factor, brier_score, avg_edge,
                          created_at, duration_seconds
                   FROM backtest_runs WHERE strategy_id = ?
                   ORDER BY created_at DESC""",
                (strategy_id,),
            )
        else:
            cur.execute(
                """SELECT id, strategy_id, start_date, end_date, cities,
                          total_trades, win_rate, gross_pnl, sharpe_ratio,
                          max_drawdown, profit_factor, brier_score, avg_edge,
                          created_at, duration_seconds
                   FROM backtest_runs ORDER BY created_at DESC"""
            )
        rows = cur.fetchall()
        conn.close()
        return jsonify(rows_to_dicts(rows))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtest/runs/<int:rid>", methods=["GET"])
def api_get_backtest_run(rid):
    """Get full backtest run including trade list and daily PnL."""
    try:
        conn = get_db_rw()
        _ensure_backtest_table(conn)
        cur = conn.cursor()
        cur.execute("SELECT * FROM backtest_runs WHERE id = ?", (rid,))
        row = cur.fetchone()
        conn.close()
        if row is None:
            return jsonify({"error": "Backtest run not found"}), 404
        return jsonify(dict(row))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtest/runs/<int:rid>", methods=["DELETE"])
def api_delete_backtest_run(rid):
    """Delete a backtest run."""
    try:
        conn = get_db_rw()
        _ensure_backtest_table(conn)
        cur = conn.cursor()
        cur.execute("DELETE FROM backtest_runs WHERE id = ?", (rid,))
        deleted = cur.rowcount
        conn.commit()
        conn.close()
        if deleted == 0:
            return jsonify({"error": "Run not found"}), 404
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===========================================================================
# OPTIMIZATION API
# ===========================================================================

@app.route("/api/optimize/run", methods=["POST"])
def api_run_optimization():
    """Run a parameter grid search. Synchronous."""
    try:
        data = request.get_json(force=True)
        param_ranges = data.get("param_ranges", {})
        start_date = data.get("start_date")
        end_date = data.get("end_date")
        target_metric = data.get("target_metric", "sharpe_ratio")
        name = data.get("name", "Optimization " + datetime.utcnow().strftime("%Y-%m-%d %H:%M"))

        if not param_ranges or not start_date or not end_date:
            return jsonify({"error": "param_ranges, start_date, end_date required"}), 400

        # Convert {min, max, step} range specs to value arrays
        for key, spec in list(param_ranges.items()):
            if isinstance(spec, dict) and "min" in spec and "max" in spec and "step" in spec:
                values = []
                v = float(spec["min"])
                step = float(spec["step"])
                mx = float(spec["max"])
                while v <= mx + 1e-9:
                    values.append(round(v, 6))
                    v += step
                param_ranges[key] = values
            # else: already a list (backward compatible)

        from src.backtest.optimizer import BacktestOptimizer

        t0 = time.time()
        optimizer = BacktestOptimizer()
        opt_result = optimizer.grid_search(
            param_ranges=param_ranges,
            start_date=start_date,
            end_date=end_date,
            target_metric=target_metric,
        )
        duration = time.time() - t0

        results = opt_result["results"]
        best_by_city = opt_result.get("best_by_city", {})

        # Sort by target metric descending
        results.sort(key=lambda r: r.get("metrics", {}).get(target_metric, 0), reverse=True)
        best = results[0] if results else {}

        conn = get_db_rw()
        _ensure_optimization_table(conn)
        cur = conn.cursor()
        now = datetime.utcnow().isoformat() + "Z"

        cur.execute(
            """INSERT INTO optimization_runs
               (name, param_ranges_json, target_metric, total_combinations,
                results_json, best_params_json, best_by_city_json,
                created_at, duration_seconds)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                name,
                json.dumps(data.get("param_ranges", {})),
                target_metric,
                len(results),
                json.dumps(results),
                json.dumps(best.get("params", {})),
                json.dumps(best_by_city),
                now, duration,
            ),
        )
        run_id = cur.lastrowid
        conn.commit()

        cur.execute("SELECT * FROM optimization_runs WHERE id = ?", (run_id,))
        row = cur.fetchone()
        conn.close()

        return jsonify(dict(row)), 201

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# ADAPTIVE OPTIMIZATION (async with SSE progress)
# ---------------------------------------------------------------------------

def _convert_range_specs(param_ranges: dict) -> dict:
    """Convert {min, max, step} specs to value arrays in-place, return result."""
    for key, spec in list(param_ranges.items()):
        if isinstance(spec, dict) and "min" in spec and "max" in spec and "step" in spec:
            values = []
            v = float(spec["min"])
            step = float(spec["step"])
            mx = float(spec["max"])
            while v <= mx + 1e-9:
                values.append(round(v, 6))
                v += step
            param_ranges[key] = values
    return param_ranges


@app.route("/api/optimize/start", methods=["POST"])
def api_start_optimization():
    """Launch an adaptive optimization in a background thread.

    Returns a job_id immediately; use /api/optimize/stream/<job_id>
    for SSE progress events.
    """
    try:
        data = request.get_json(force=True)
        param_ranges = data.get("param_ranges", {})
        start_date = data.get("start_date")
        end_date = data.get("end_date")
        target_metric = data.get("target_metric", "sharpe_ratio")
        name = data.get("name", "Optimization " + datetime.utcnow().strftime("%Y-%m-%d %H:%M"))
        strategy = data.get("strategy", "mc_refine")
        phase1_budget = int(data.get("phase1_budget", 200))

        if not param_ranges or not start_date or not end_date:
            return jsonify({"error": "param_ranges, start_date, end_date required"}), 400

        # Convert range specs to value arrays
        raw_ranges = json.dumps(data.get("param_ranges", {}))  # save original
        param_ranges = _convert_range_specs(param_ranges)

        job_id = str(uuid.uuid4())
        job = {
            "status": "running",
            "phase": 0,
            "current": 0,
            "total": 0,
            "queue": queue.Queue(),
            "optimizer": None,
            "name": name,
            "raw_ranges": raw_ranges,
            "target_metric": target_metric,
            "strategy_type": strategy,
        }

        with _opt_lock:
            _opt_jobs[job_id] = job

        def run():
            from src.backtest.optimizer import BacktestOptimizer, OptimizationAborted

            optimizer = BacktestOptimizer()
            job["optimizer"] = optimizer
            t0 = time.time()

            _prev_best = [None]  # mutable ref for closure

            def progress_cb(phase, current, total, best_so_far, entry):
                job["phase"] = phase
                job["current"] = current
                job["total"] = total
                has_error = not entry or "error" in entry.get("metrics", {})
                metrics = entry.get("metrics", {}) if not has_error else {}
                metric_val = metrics.get(target_metric)

                # Detect if this combo set a new best
                is_new_best = (
                    not has_error
                    and metric_val is not None
                    and best_so_far > -1e10
                    and abs(metric_val - best_so_far) < 1e-9
                    and _prev_best[0] != best_so_far
                )
                if is_new_best:
                    _prev_best[0] = best_so_far

                # Find best city for this combo
                city_metrics = entry.get("city_metrics", {}) if not has_error else {}
                best_city = None
                best_city_val = -float("inf")
                for city, cm in city_metrics.items():
                    cv = cm.get(target_metric, 0)
                    if cv > best_city_val:
                        best_city_val = cv
                        best_city = city

                evt = {
                    "event": "progress",
                    "phase": phase,
                    "current": current,
                    "total": total,
                    "best_so_far": round(best_so_far, 6) if best_so_far > -1e10 else None,
                    "metric_val": round(metric_val, 6) if metric_val is not None else None,
                    "is_new_best": is_new_best,
                }
                # Include full entry data (params + key metrics + best city)
                if not has_error:
                    evt["params"] = entry.get("params", {})
                    evt["metrics"] = {
                        k: round(v, 6) if isinstance(v, float) else v
                        for k, v in metrics.items()
                    }
                    evt["best_city"] = best_city
                    evt["best_city_metric"] = (
                        round(best_city_val, 6)
                        if best_city_val > -float("inf") else None
                    )

                job["queue"].put(evt)

            try:
                opt_result = optimizer.adaptive_search(
                    param_ranges=param_ranges,
                    start_date=start_date,
                    end_date=end_date,
                    target_metric=target_metric,
                    strategy=strategy,
                    progress_cb=progress_cb,
                    phase1_budget=phase1_budget,
                )
                duration = time.time() - t0

                results = opt_result["results"]
                best_by_city = opt_result.get("best_by_city", {})

                # Sort by target metric
                results.sort(
                    key=lambda r: r.get("metrics", {}).get(target_metric, 0),
                    reverse=True,
                )
                best = results[0] if results else {}

                # Save to DB
                conn = get_db_rw()
                _ensure_optimization_table(conn)
                cur = conn.cursor()
                now = datetime.utcnow().isoformat() + "Z"
                cur.execute(
                    """INSERT INTO optimization_runs
                       (name, param_ranges_json, target_metric, total_combinations,
                        results_json, best_params_json, best_by_city_json,
                        created_at, duration_seconds, strategy_type)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (
                        name,
                        raw_ranges,
                        target_metric,
                        len(results),
                        json.dumps(results),
                        json.dumps(best.get("params", {})),
                        json.dumps(best_by_city),
                        now, duration, strategy,
                    ),
                )
                run_id = cur.lastrowid
                conn.commit()
                conn.close()

                job["status"] = "done"
                job["queue"].put({
                    "event": "done",
                    "run_id": run_id,
                    "duration": round(duration, 1),
                    "total_evaluated": len(results),
                })

            except OptimizationAborted:
                job["status"] = "aborted"
                job["queue"].put({"event": "error", "message": "Aborted by user"})

            except Exception as e:
                import traceback
                traceback.print_exc()
                job["status"] = "error"
                job["queue"].put({"event": "error", "message": str(e)})

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        return jsonify({"job_id": job_id}), 202

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/optimize/stream/<job_id>")
def api_stream_optimization(job_id):
    """SSE endpoint for real-time optimization progress."""
    with _opt_lock:
        job = _opt_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404

    def generate():
        while True:
            try:
                msg = job["queue"].get(timeout=30)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("event") in ("done", "error"):
                    break
            except queue.Empty:
                # Heartbeat to keep connection alive
                yield f"data: {json.dumps({'event': 'heartbeat'})}\n\n"

        # Clean up job after terminal event
        with _opt_lock:
            _opt_jobs.pop(job_id, None)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/optimize/abort/<job_id>", methods=["POST"])
def api_abort_optimization(job_id):
    """Signal a running optimization to stop."""
    with _opt_lock:
        job = _opt_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404

    optimizer = job.get("optimizer")
    if optimizer:
        optimizer.request_abort()
        return jsonify({"status": "abort_requested"})
    return jsonify({"error": "Optimizer not yet started"}), 409


@app.route("/api/optimize/runs", methods=["GET"])
def api_list_optimization_runs():
    """List all optimization runs."""
    try:
        conn = get_db_rw()
        _ensure_optimization_table(conn)
        cur = conn.cursor()
        cur.execute(
            """SELECT id, name, target_metric, total_combinations,
                      best_params_json, created_at, duration_seconds
               FROM optimization_runs ORDER BY created_at DESC"""
        )
        rows = cur.fetchall()
        conn.close()
        return jsonify(rows_to_dicts(rows))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/optimize/runs/<int:rid>", methods=["GET"])
def api_get_optimization_run(rid):
    """Get full optimization run details."""
    try:
        conn = get_db_rw()
        _ensure_optimization_table(conn)
        cur = conn.cursor()
        cur.execute("SELECT * FROM optimization_runs WHERE id = ?", (rid,))
        row = cur.fetchone()
        conn.close()
        if row is None:
            return jsonify({"error": "Run not found"}), 404
        return jsonify(dict(row))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/optimize/runs/<int:rid>/create-strategy", methods=["POST"])
def api_create_strategy_from_optimization(rid):
    """Create a strategy from the best parameters of an optimization run."""
    try:
        data = request.get_json(force=True) if request.data else {}
        name = data.get("name", "").strip()

        conn = get_db_rw()
        _ensure_optimization_table(conn)
        _ensure_strategies_table(conn)
        cur = conn.cursor()

        cur.execute("SELECT best_params_json FROM optimization_runs WHERE id = ?", (rid,))
        row = cur.fetchone()
        if row is None:
            conn.close()
            return jsonify({"error": "Optimization run not found"}), 404

        best_params = json.loads(row["best_params_json"]) if row["best_params_json"] else {}
        if not name:
            name = f"Optimized #{rid}"

        now = datetime.utcnow().isoformat() + "Z"
        best_params["name"] = name
        best_params["created_at"] = now
        best_params["updated_at"] = now

        cols = ["name", "created_at", "updated_at"]
        vals = [name, now, now]
        for field in STRATEGY_FIELDS:
            if field == "name":
                continue
            if field in best_params:
                cols.append(field)
                if field in STRATEGY_FLOAT_FIELDS:
                    vals.append(float(best_params[field]))
                elif field in STRATEGY_INT_FIELDS:
                    vals.append(int(best_params[field]))
                else:
                    vals.append(str(best_params[field]))

        placeholders = ",".join(["?"] * len(vals))
        col_names = ",".join(cols)
        cur.execute(f"INSERT INTO strategies ({col_names}) VALUES ({placeholders})", vals)
        strategy_id = cur.lastrowid
        conn.commit()

        cur.execute("SELECT * FROM strategies WHERE id = ?", (strategy_id,))
        strat_row = cur.fetchone()
        conn.close()

        return jsonify(dict(strat_row)), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": f"Strategy name '{name}' already exists"}), 409
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===========================================================================
# DATA FRESHNESS & CONTROL PANEL API
# ===========================================================================

def _compute_staleness_minutes(latest_time):
    """Compute minutes since a timestamp string. Returns 9999 if None."""
    if not latest_time:
        return 9999.0
    try:
        from datetime import timezone
        ts = latest_time.replace(" ", "T")
        if not ts.endswith("Z") and "+" not in ts and "-" not in ts[10:]:
            ts += "Z"
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - dt
        return max(0.0, delta.total_seconds() / 60.0)
    except Exception:
        return 9999.0


def _staleness_status(source, staleness_min):
    """Return 'green', 'yellow', or 'red' based on staleness."""
    thresholds = {
        "gfs_ensemble": {"green": 420, "yellow": 780},
        "hrrr":         {"green": 120, "yellow": 240},
        "nws":          {"green": 360, "yellow": 720},
    }
    t = thresholds.get(source, {"green": 360, "yellow": 720})
    if staleness_min <= t["green"]:
        return "green"
    elif staleness_min <= t["yellow"]:
        return "yellow"
    return "red"


@app.route("/api/data-freshness")
def api_data_freshness():
    """Return freshness status for GFS, HRRR, NWS data sources.
    Also returns price discovery market count and total active markets."""
    try:
        conn = get_db()
        cur = conn.cursor()

        sources = []

        # GFS Ensemble — use fetched_at for "last refreshed" display,
        # model_run_time for freshness logic (is there a newer run available?)
        cur.execute(
            "SELECT MAX(model_run_time) AS latest_run, MAX(fetched_at) AS latest_fetch "
            "FROM ensemble_forecasts"
        )
        row = cur.fetchone()
        gfs_run = row["latest_run"] if row else None
        gfs_fetch = row["latest_fetch"] if row else None
        gfs_staleness = _compute_staleness_minutes(gfs_fetch)
        gfs_run_staleness = _compute_staleness_minutes(gfs_run)
        sources.append({
            "source": "gfs_ensemble",
            "label": "GFS Ensemble",
            "latest_model_run": gfs_run,
            "latest_fetch": gfs_fetch,
            "staleness_minutes": round(gfs_staleness, 1),
            "status": _staleness_status("gfs_ensemble", gfs_run_staleness),
            "has_data": gfs_fetch is not None,
        })

        # HRRR
        cur.execute(
            "SELECT MAX(model_run_time) AS latest_run, MAX(fetched_at) AS latest_fetch "
            "FROM hrrr_forecasts"
        )
        row = cur.fetchone()
        hrrr_run = row["latest_run"] if row else None
        hrrr_fetch = row["latest_fetch"] if row else None
        hrrr_staleness = _compute_staleness_minutes(hrrr_fetch)
        hrrr_run_staleness = _compute_staleness_minutes(hrrr_run)
        sources.append({
            "source": "hrrr",
            "label": "HRRR",
            "latest_model_run": hrrr_run,
            "latest_fetch": hrrr_fetch,
            "staleness_minutes": round(hrrr_staleness, 1),
            "status": _staleness_status("hrrr", hrrr_run_staleness),
            "has_data": hrrr_fetch is not None,
        })

        # NWS (no model_run_time concept, just fetched_at)
        cur.execute("SELECT MAX(fetched_at) AS latest FROM nws_forecasts")
        row = cur.fetchone()
        nws_latest = row["latest"] if row else None
        nws_staleness = _compute_staleness_minutes(nws_latest)
        sources.append({
            "source": "nws",
            "label": "NWS",
            "latest_model_run": nws_latest,
            "latest_fetch": nws_latest,
            "staleness_minutes": round(nws_staleness, 1),
            "status": _staleness_status("nws", nws_staleness),
            "has_data": nws_latest is not None,
        })

        # Active markets count
        cur.execute("SELECT COUNT(*) AS cnt FROM kalshi_markets WHERE status IN ('open', 'active')")
        total_markets = cur.fetchone()["cnt"]

        # Price discovery count (markets discovered in last 2 hours)
        price_discovery_count = 0
        try:
            from datetime import timezone
            cutoff = (datetime.now(timezone.utc) - __import__("datetime").timedelta(hours=2)).isoformat()
            cur.execute(
                "SELECT COUNT(*) AS cnt FROM kalshi_markets WHERE first_discovered_at >= ? AND status IN ('open', 'active')",
                (cutoff,)
            )
            price_discovery_count = cur.fetchone()["cnt"]
        except Exception:
            pass  # Column may not exist yet

        # Last signal computed_at
        cur.execute("SELECT MAX(computed_at) AS latest FROM signals")
        row = cur.fetchone()
        last_scan = row["latest"] if row else None
        last_scan_minutes = _compute_staleness_minutes(last_scan)

        conn.close()

        return jsonify({
            "sources": sources,
            "total_markets": total_markets,
            "price_discovery_count": price_discovery_count,
            "last_scan": last_scan,
            "last_scan_minutes_ago": round(last_scan_minutes, 1),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/refresh/<source>", methods=["POST"])
def api_manual_refresh(source):
    """Trigger direct data refresh in a background thread.

    Calls the fetchers for all 5 cities — no bot process needed.
    Returns immediately; poll /api/refresh/<source>/status for progress.
    """
    valid_sources = {"gfs", "hrrr", "nws", "markets"}
    if source not in valid_sources:
        return jsonify({"error": f"Invalid source. Use one of: {valid_sources}"}), 400

    if _refresh_status[source]["running"]:
        return jsonify({"status": "already_running", "source": source})

    thread = threading.Thread(
        target=_do_direct_fetch,
        args=(source,),
        daemon=True,
        name=f"refresh-{source}",
    )
    thread.start()

    return jsonify({"status": "started", "source": source})


@app.route("/api/refresh/all", methods=["POST"])
def api_refresh_all():
    """Trigger refresh of ALL data sources + markets in parallel.

    Starts background threads for each source. Returns immediately.
    Poll /api/refresh/all/status for aggregate progress.
    """
    started = []
    already_running = []
    for source in ("gfs", "hrrr", "nws", "markets"):
        if _refresh_status[source]["running"]:
            already_running.append(source)
            continue
        thread = threading.Thread(
            target=_do_direct_fetch,
            args=(source,),
            daemon=True,
            name=f"refresh-{source}",
        )
        thread.start()
        started.append(source)

    return jsonify({
        "status": "started",
        "started": started,
        "already_running": already_running,
    })


@app.route("/api/refresh/all/status")
def api_refresh_all_status():
    """Aggregate status for all refresh sources."""
    sources = {}
    any_running = False
    for source in ("gfs", "hrrr", "nws", "markets"):
        running = _refresh_status[source]["running"]
        if running:
            any_running = True
        sources[source] = {
            "running": running,
            "last_result": _refresh_status[source]["last_result"],
        }
    return jsonify({"running": any_running, "sources": sources})


@app.route("/api/refresh/<source>/status")
def api_refresh_status(source):
    """Check status of a running or completed refresh."""
    valid_sources = {"gfs", "hrrr", "nws", "markets"}
    if source not in valid_sources:
        return jsonify({"error": f"Invalid source"}), 400

    return jsonify({
        "source": source,
        "running": _refresh_status[source]["running"],
        "last_result": _refresh_status[source]["last_result"],
    })


# ===========================================================================
# EDGE HISTORY API (for edge trajectory chart)
# ===========================================================================

@app.route("/api/signals/edge-history")
def api_edge_history():
    """Return edge trajectory over time for given market(s).

    Query params:
        market_ticker - single ticker (e.g., KXHIGHNY-26MAR06-T55)
        city - filter by city
        target_date - filter by target_date
        hours - how many hours of history (default: 24)
    """
    try:
        conn = get_db()
        cur = conn.cursor()

        market_ticker = request.args.get("market_ticker")
        city = request.args.get("city")
        target_date = request.args.get("target_date")
        hours = int(request.args.get("hours", 24))

        from datetime import timezone
        cutoff = (datetime.now(timezone.utc) - __import__("datetime").timedelta(hours=hours)).isoformat()

        conditions = ["s.computed_at >= ?"]
        params = [cutoff]

        if market_ticker:
            conditions.append("s.market_ticker = ?")
            params.append(market_ticker)
        if city:
            conditions.append("s.city = ?")
            params.append(city)
        if target_date:
            conditions.append("s.target_date = ?")
            params.append(target_date)

        where_clause = " AND ".join(conditions)

        cur.execute(
            f"""
            SELECT
                s.computed_at,
                s.market_ticker,
                s.city,
                s.target_date,
                s.raw_edge,
                s.abs_edge,
                s.calibrated_prob,
                s.market_yes_price,
                s.lead_hours
            FROM signals s
            WHERE {where_clause}
            ORDER BY s.computed_at ASC
            LIMIT 5000
            """,
            params,
        )
        rows = cur.fetchall()
        conn.close()

        # Group by market_ticker for easy charting
        by_ticker: dict[str, list] = {}
        for r in rows:
            ticker = r["market_ticker"]
            by_ticker.setdefault(ticker, []).append({
                "t": r["computed_at"],
                "edge": r["raw_edge"],
                "abs_edge": r["abs_edge"],
                "prob": r["calibrated_prob"],
                "price": r["market_yes_price"],
                "lead_hours": r["lead_hours"],
            })

        return jsonify({
            "tickers": list(by_ticker.keys()),
            "series": by_ticker,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===========================================================================
# MATH / PIPELINE VISUALIZATION API
# ===========================================================================

def _interpolate_hrrr_weight(lead_hours):
    """Interpolate HRRR weight based on lead hours (mirrors ModelWeightsConfig)."""
    breakpoints = {0: 0.45, 6: 0.35, 12: 0.25, 24: 0.15, 48: 0.05}
    keys = sorted(breakpoints)
    if lead_hours <= keys[0]:
        return breakpoints[keys[0]]
    if lead_hours >= keys[-1]:
        return breakpoints[keys[-1]]
    for i in range(len(keys) - 1):
        lo, hi = keys[i], keys[i + 1]
        if lo <= lead_hours <= hi:
            frac = (lead_hours - lo) / (hi - lo)
            return breakpoints[lo] + frac * (breakpoints[hi] - breakpoints[lo])
    return 0.25


@app.route("/api/math/available-dates")
def api_math_available_dates():
    """Return distinct target dates that have data for the city/date selectors."""
    try:
        city = request.args.get("city")
        conn = get_db()
        cur = conn.cursor()

        if city:
            cur.execute(
                "SELECT DISTINCT target_date FROM signals WHERE city = ? "
                "UNION SELECT DISTINCT target_date FROM kalshi_markets "
                "WHERE status IN ('open','active','closed','settled') AND city = ? "
                "ORDER BY target_date DESC LIMIT 30",
                (city, city),
            )
        else:
            cur.execute(
                "SELECT DISTINCT target_date FROM signals "
                "UNION SELECT DISTINCT target_date FROM kalshi_markets "
                "WHERE status IN ('open','active','closed','settled') "
                "ORDER BY target_date DESC LIMIT 30"
            )

        dates = [r["target_date"] for r in cur.fetchall() if r["target_date"]]
        cities_row = cur.execute(
            "SELECT DISTINCT city FROM kalshi_markets ORDER BY city"
        ).fetchall()
        cities = [r["city"] for r in cities_row]
        conn.close()
        return jsonify({"dates": dates, "cities": cities})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/math/raw-data")
def api_math_raw_data():
    """Return raw forecast data (ensemble members, HRRR, NWS) for a city+date."""
    city = request.args.get("city")
    target_date = request.args.get("target_date")
    if not city or not target_date:
        return jsonify({"error": "city and target_date required"}), 400

    try:
        conn = get_db()
        cur = conn.cursor()

        # Ensemble: get member daily maxes for the latest model run
        ens_data = None
        cur.execute(
            "SELECT model_run_time, member, temperature_f FROM ensemble_forecasts "
            "WHERE city = ? AND DATE(valid_time) = ? "
            "ORDER BY model_run_time DESC, member ASC",
            (city, target_date),
        )
        ens_rows = cur.fetchall()
        if ens_rows:
            # Group by model_run_time, take latest
            latest_run = ens_rows[0]["model_run_time"]
            member_temps = [
                r["temperature_f"] for r in ens_rows
                if r["model_run_time"] == latest_run
            ]
            if member_temps:
                import statistics
                ens_data = {
                    "model_run_time": latest_run,
                    "member_maxes": member_temps,
                    "member_count": len(member_temps),
                    "mean": round(statistics.mean(member_temps), 1),
                    "std": round(statistics.stdev(member_temps), 1) if len(member_temps) > 1 else 0,
                    "min": round(min(member_temps), 1),
                    "max": round(max(member_temps), 1),
                    "median": round(statistics.median(member_temps), 1),
                }

        # HRRR
        hrrr_data = None
        cur.execute(
            "SELECT model_run_time, temperature_f, fetched_at FROM hrrr_forecasts "
            "WHERE city = ? AND DATE(valid_time) = ? "
            "ORDER BY model_run_time DESC LIMIT 1",
            (city, target_date),
        )
        hr = cur.fetchone()
        if hr:
            hrrr_data = {
                "daily_max_f": hr["temperature_f"],
                "model_run_time": hr["model_run_time"],
                "fetched_at": hr["fetched_at"],
            }

        # NWS
        nws_data = None
        cur.execute(
            "SELECT high_f, low_f, fetched_at FROM nws_forecasts "
            "WHERE city = ? AND forecast_date = ? "
            "ORDER BY fetched_at DESC LIMIT 1",
            (city, target_date),
        )
        nw = cur.fetchone()
        if nw:
            nws_data = {
                "high_f": nw["high_f"],
                "low_f": nw["low_f"],
                "fetched_at": nw["fetched_at"],
            }

        conn.close()
        return jsonify({
            "city": city,
            "target_date": target_date,
            "ensemble": ens_data,
            "hrrr": hrrr_data,
            "nws": nws_data,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/math/probability-pipeline")
def api_math_probability_pipeline():
    """Return full probability pipeline data for each market in a city+date."""
    city = request.args.get("city")
    target_date = request.args.get("target_date")
    if not city or not target_date:
        return jsonify({"error": "city and target_date required"}), 400

    try:
        conn = get_db()
        cur = conn.cursor()

        # Get latest signals per market
        cur.execute(
            "SELECT s.*, m.threshold_f, m.bracket_low, m.bracket_high, "
            "m.is_above_contract, m.yes_price AS mkt_price, m.volume "
            "FROM signals s "
            "JOIN kalshi_markets m ON s.market_ticker = m.market_ticker "
            "WHERE s.city = ? AND s.target_date = ? "
            "ORDER BY s.computed_at DESC",
            (city, target_date),
        )
        rows = cur.fetchall()

        # Deduplicate: keep latest computed_at per market_ticker
        seen = set()
        markets = []
        lead_hours = None
        for r in rows:
            ticker = r["market_ticker"]
            if ticker in seen:
                continue
            seen.add(ticker)
            lh = r["lead_hours"] or 0
            if lead_hours is None:
                lead_hours = lh
            abs_edge = r["abs_edge"] or abs(r["raw_edge"] or 0)
            edge = r["raw_edge"] or 0
            is_tradeable = abs_edge >= 0.08  # edge_threshold
            side = "yes" if edge > 0 else "no"
            confidence = "high" if abs_edge > 0.20 else ("medium" if abs_edge > 0.12 else "low")

            # Short label for chart
            is_above = r["is_above_contract"]
            thresh = r["threshold_f"]
            blo = r["bracket_low"]
            bhi = r["bracket_high"]
            if is_above and thresh is not None:
                short_label = f"T{int(thresh)}"
            elif blo is not None and bhi is not None:
                short_label = f"B{blo}-{bhi}"
            elif bhi is not None:
                short_label = f"≤{int(bhi)}"
            else:
                short_label = ticker.split("-")[-1] if "-" in ticker else ticker

            markets.append({
                "market_ticker": ticker,
                "short_label": short_label,
                "threshold_f": thresh,
                "bracket_low": blo,
                "bracket_high": bhi,
                "is_above": bool(is_above),
                "ensemble_prob": r["ensemble_prob"],
                "hrrr_prob": r["hrrr_prob"],
                "nws_prob": r["nws_prob"],
                "blended_prob": r["blended_prob"],
                "calibrated_prob": r["calibrated_prob"],
                "market_yes_price": r["market_yes_price"],
                "raw_edge": edge,
                "abs_edge": abs_edge,
                "lead_hours": lh,
                "is_tradeable": is_tradeable,
                "side": side,
                "confidence": confidence,
                "volume": r["volume"],
            })

        # Compute blending weights based on lead hours
        lh = lead_hours or 20
        hrrr_adj = _interpolate_hrrr_weight(lh)
        weights = {
            "gfs_ensemble": {"base": 0.60, "adjusted": 0.60},
            "hrrr": {"base": 0.25, "adjusted": round(hrrr_adj, 3)},
            "nws": {"base": 0.15, "adjusted": 0.15},
        }

        conn.close()
        return jsonify({
            "city": city,
            "target_date": target_date,
            "lead_hours": lh,
            "weights": weights,
            "markets": sorted(markets, key=lambda m: m["threshold_f"] or m["bracket_low"] or 0),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/math/kelly-sizing")
def api_math_kelly_sizing():
    """Return Kelly sizing breakdown and risk check status for a city+date."""
    city = request.args.get("city")
    target_date = request.args.get("target_date")
    if not city or not target_date:
        return jsonify({"error": "city and target_date required"}), 400

    try:
        conn = get_db()
        cur = conn.cursor()

        # Get trades for this city+date
        cur.execute(
            "SELECT * FROM trades WHERE city = ? AND target_date = ? ORDER BY market_ticker",
            (city, target_date),
        )
        trade_rows = cur.fetchall()

        trades = []
        for t in trade_rows:
            price = t["price"] or 0
            side = t["side"] or "yes"
            model_prob = t["model_prob"] or 0
            p_win = model_prob if side == "yes" else (1 - model_prob)
            q_lose = 1 - p_win

            if side == "yes":
                odds_b = ((1 - price) / price) if price > 0 else 0
            else:
                no_price = 1 - price if price < 1 else 0
                odds_b = (price / no_price) if no_price > 0 else 0

            full_kelly = ((p_win * odds_b - q_lose) / odds_b) if odds_b > 0 else 0
            full_kelly = max(0, full_kelly)
            fractional = full_kelly * 0.15

            contracts = t["contracts"] or 0
            total_cost = t["total_cost"] or 0

            # EV per contract
            if side == "yes":
                ev = p_win * (1 - price) - q_lose * price
            else:
                no_price = 1 - price
                ev = p_win * price - q_lose * no_price

            trades.append({
                "market_ticker": t["market_ticker"],
                "side": side,
                "direction": t["direction"] or "buy",
                "model_prob": round(model_prob, 4),
                "market_price": round(price, 4),
                "edge": round(t["edge"] or 0, 4),
                "p_win": round(p_win, 4),
                "odds_b": round(odds_b, 3),
                "full_kelly": round(full_kelly, 4),
                "fractional_kelly": round(fractional, 4),
                "contracts": contracts,
                "price": round(price, 4),
                "total_cost": round(total_cost, 2),
                "ev_per_contract": round(ev, 4),
                "status": t["status"],
            })

        # Portfolio risk state
        cur.execute(
            "SELECT COUNT(*) AS cnt FROM trades WHERE status = 'filled' AND pnl IS NULL"
        )
        open_positions = cur.fetchone()["cnt"]

        cur.execute(
            "SELECT COUNT(*) AS cnt FROM trades WHERE status = 'filled' "
            "AND pnl IS NULL AND city = ?", (city,)
        )
        city_positions = cur.fetchone()["cnt"]

        cur.execute(
            "SELECT COUNT(*) AS cnt FROM trades WHERE status = 'filled' "
            "AND pnl IS NULL AND target_date = ?", (target_date,)
        )
        date_positions = cur.fetchone()["cnt"]

        cur.execute(
            "SELECT COALESCE(SUM(pnl), 0) AS total FROM trades "
            "WHERE DATE(created_at) = DATE('now') AND pnl IS NOT NULL"
        )
        daily_pnl = cur.fetchone()["total"]

        # Current bankroll: real Kalshi balance in live mode
        live_balance = _get_kalshi_balance()
        if live_balance is not None:
            bankroll = round(live_balance, 2)
        else:
            cur.execute("SELECT COALESCE(SUM(total_cost), 0) AS spent FROM trades WHERE status='filled' AND pnl IS NULL")
            open_cost = cur.fetchone()["spent"]
            cur.execute("SELECT COALESCE(SUM(pnl), 0) AS realized FROM trades WHERE pnl IS NOT NULL")
            realized = cur.fetchone()["realized"]
            bankroll = round(INITIAL_BANKROLL + realized - open_cost, 2)

        risk_checks = {
            "daily_loss_limit": {"current": round(daily_pnl, 2), "limit": 300, "ok": daily_pnl > -300},
            "concurrent_positions": {"current": open_positions, "limit": 20, "ok": open_positions < 20},
            "city_positions": {"current": city_positions, "limit": 6, "ok": city_positions < 6},
            "date_positions": {"current": date_positions, "limit": 4, "ok": date_positions < 4},
            "bankroll": {"available": bankroll, "required": sum(t["total_cost"] for t in trades), "ok": True},
        }

        conn.close()
        return jsonify({
            "city": city,
            "target_date": target_date,
            "bankroll": bankroll,
            "trades": trades,
            "risk_checks": risk_checks,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/math/calibration-curve")
def api_math_calibration_curve():
    """Return reliability diagram data from backtest trades."""
    try:
        conn = get_db()
        cur = conn.cursor()

        # Try to get calibration data from the latest backtest run
        cur.execute(
            "SELECT trades_json FROM backtest_runs ORDER BY created_at DESC LIMIT 1"
        )
        row = cur.fetchone()
        conn.close()

        if not row or not row["trades_json"]:
            return jsonify({"bins": [], "total_samples": 0, "source": "none"})

        bt_trades = json.loads(row["trades_json"])

        # Extract (forecast_prob, outcome) pairs
        pairs = []
        for t in bt_trades:
            prob = t.get("model_prob") or t.get("forecast_prob")
            pnl = t.get("pnl")
            side = t.get("side", "yes")
            if prob is None or pnl is None:
                continue
            # Outcome: did the YES side win?
            if side == "yes":
                outcome = 1 if pnl > 0 else 0
            else:
                outcome = 0 if pnl > 0 else 1
            pairs.append((prob, outcome))

        if not pairs:
            return jsonify({"bins": [], "total_samples": 0, "source": "backtest"})

        # Bin into deciles
        bins = []
        for i in range(10):
            lo = i * 0.1
            hi = (i + 1) * 0.1
            in_bin = [(p, o) for p, o in pairs if lo <= p < hi]
            if i == 9:
                in_bin = [(p, o) for p, o in pairs if lo <= p <= hi]
            if in_bin:
                avg_f = sum(p for p, _ in in_bin) / len(in_bin)
                obs_f = sum(o for _, o in in_bin) / len(in_bin)
                bins.append({
                    "bin_start": round(lo, 1),
                    "bin_end": round(hi, 1),
                    "count": len(in_bin),
                    "avg_forecast": round(avg_f, 4),
                    "observed_freq": round(obs_f, 4),
                })
            else:
                bins.append({
                    "bin_start": round(lo, 1),
                    "bin_end": round(hi, 1),
                    "count": 0,
                    "avg_forecast": round(lo + 0.05, 2),
                    "observed_freq": None,
                })

        # Brier score decomposition
        n = len(pairs)
        base_rate = sum(o for _, o in pairs) / n
        brier = sum((p - o) ** 2 for p, o in pairs) / n
        uncertainty = base_rate * (1 - base_rate)

        # Reliability and resolution from bins
        reliability = 0
        resolution = 0
        for b in bins:
            if b["count"] == 0 or b["observed_freq"] is None:
                continue
            nk = b["count"]
            fk = b["avg_forecast"]
            ok = b["observed_freq"]
            reliability += nk * (fk - ok) ** 2
            resolution += nk * (ok - base_rate) ** 2
        reliability /= n
        resolution /= n

        return jsonify({
            "bins": bins,
            "total_samples": n,
            "brier_score": round(brier, 4),
            "reliability": round(reliability, 4),
            "resolution": round(resolution, 4),
            "uncertainty": round(uncertainty, 4),
            "base_rate": round(base_rate, 4),
            "source": "backtest",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/math/observation-histogram")
def api_math_observation_histogram():
    """Return historical temperature distribution for a city+month."""
    city = request.args.get("city")
    month = request.args.get("month", type=int)
    if not city:
        return jsonify({"error": "city required"}), 400

    try:
        conn = get_db()
        cur = conn.cursor()

        if month:
            cur.execute(
                "SELECT high_f FROM observations WHERE city = ? "
                "AND CAST(SUBSTR(date, 6, 2) AS INTEGER) = ? "
                "AND high_f IS NOT NULL ORDER BY date",
                (city, month),
            )
        else:
            cur.execute(
                "SELECT high_f FROM observations WHERE city = ? "
                "AND high_f IS NOT NULL ORDER BY date",
                (city,),
            )

        rows = cur.fetchall()
        conn.close()

        if not rows:
            return jsonify({"city": city, "month": month, "bins": [], "sample_count": 0})

        temps = [r["high_f"] for r in rows]
        import statistics

        t_min = int(min(temps))
        t_max = int(max(temps)) + 1
        # Create 2-degree bins
        bin_width = 2
        bin_start = (t_min // bin_width) * bin_width
        bin_end = ((t_max // bin_width) + 1) * bin_width

        bins = []
        for lo in range(bin_start, bin_end, bin_width):
            hi = lo + bin_width
            count = sum(1 for t in temps if lo <= t < hi)
            if lo + bin_width >= bin_end:
                count = sum(1 for t in temps if lo <= t <= hi)
            bins.append({"temp_low": lo, "temp_high": hi, "count": count})

        return jsonify({
            "city": city,
            "month": month,
            "sample_count": len(temps),
            "bins": bins,
            "mean": round(statistics.mean(temps), 1),
            "std": round(statistics.stdev(temps), 1) if len(temps) > 1 else 0,
            "min": int(min(temps)),
            "max": int(max(temps)),
            "median": round(statistics.median(temps), 1),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Position Tracker API
# ---------------------------------------------------------------------------

def _format_contract_desc(row):
    """Build human-readable contract description from DB row."""
    if row["is_above_contract"]:
        return f"Above {int(row['threshold_f'])}°F"
    elif row["bracket_low"] is not None and row["bracket_high"] is not None:
        return f"{int(row['bracket_low'])}–{int(row['bracket_high'])}°F"
    elif row["bracket_high"] is not None:
        return f"≤ {int(row['bracket_high'])}°F"
    elif row["bracket_low"] is not None:
        return f"≥ {int(row['bracket_low'])}°F"
    return "Unknown"


def _compute_exit_analysis(trade, current_market, latest_signal):
    """Compute hold-vs-sell exit recommendation."""
    if not trade or not current_market:
        return None

    entry_price = trade.get("fill_price") or trade.get("price", 0)
    contracts = trade.get("contracts", 0)
    side = trade.get("side", "yes")
    current_yes = current_market.get("yes_price") or 0
    model_prob = latest_signal.get("calibrated_prob") if latest_signal else None
    lead_hours = latest_signal.get("lead_hours") if latest_signal else None
    edge_at_entry = trade.get("edge") or 0

    if side == "yes":
        cost = entry_price * contracts
        current_value = current_yes * contracts
        unrealized_pnl = current_value - cost
    else:
        entry_no = 1.0 - entry_price
        cost = entry_no * contracts
        current_no = 1.0 - current_yes
        current_value = current_no * contracts
        unrealized_pnl = current_value - cost

    # EV calculations
    ev_hold = None
    ev_sell = unrealized_pnl
    breakeven_prob = None

    if model_prob is not None:
        if side == "yes":
            ev_hold = model_prob * contracts - cost
            breakeven_prob = current_yes
        else:
            ev_hold = (1.0 - model_prob) * contracts - cost
            breakeven_prob = current_yes

    # Current edge
    current_edge = None
    if model_prob is not None:
        if side == "yes":
            current_edge = model_prob - current_yes
        else:
            current_edge = (1.0 - model_prob) - (1.0 - current_yes)

    # Recommendation
    if ev_hold is not None:
        if side == "yes":
            overvalued = current_yes > model_prob
        else:
            overvalued = (1.0 - current_yes) > (1.0 - model_prob)

        if overvalued:
            recommendation = "SELL"
            reason = "Market price exceeds model fair value — lock in profit"
        elif unrealized_pnl > 0 and lead_hours is not None and lead_hours < 6:
            recommendation = "HOLD (take profit?)"
            reason = "Profitable with settlement approaching — edge window closing"
        elif unrealized_pnl < 0 and current_edge is not None and abs(current_edge) < 0.03:
            recommendation = "SELL (cut loss)"
            reason = "Underwater with minimal remaining edge"
        else:
            recommendation = "HOLD"
            reason = "Positive expected value — edge persists"
    else:
        recommendation = "HOLD"
        reason = "No recent model signal for comparison"

    # Time to settlement
    time_to_settlement = None
    close_time = current_market.get("close_time")
    if close_time:
        try:
            from datetime import timezone
            ct = close_time.replace("Z", "+00:00")
            if "+" not in ct and "-" not in ct[10:]:
                ct += "+00:00"
            close_dt = datetime.fromisoformat(ct)
            now = datetime.now(timezone.utc)
            delta = close_dt - now
            secs = max(0, delta.total_seconds())
            hrs = int(secs // 3600)
            mins = int((secs % 3600) // 60)
            time_to_settlement = {
                "hours": round(secs / 3600, 1),
                "human": f"{hrs}h {mins}m",
            }
        except Exception:
            pass

    return {
        "entry_price": entry_price,
        "current_price": current_yes,
        "contracts": contracts,
        "side": side,
        "cost": round(cost, 2),
        "current_value": round(current_value, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "unrealized_pnl_pct": round(unrealized_pnl / cost * 100, 1) if cost > 0 else 0,
        "model_prob": model_prob,
        "ev_hold": round(ev_hold, 2) if ev_hold is not None else None,
        "ev_sell": round(ev_sell, 2),
        "breakeven_prob": breakeven_prob,
        "current_edge": round(current_edge, 4) if current_edge is not None else None,
        "edge_at_entry": edge_at_entry,
        "recommendation": recommendation,
        "reason": reason,
        "lead_hours": lead_hours,
        "time_to_settlement": time_to_settlement,
    }


@app.route("/api/tracker/positions")
def api_tracker_positions():
    """List positions available for tracking: open + recently settled."""
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT
                t.trade_id, t.market_ticker, t.city, t.target_date,
                t.side, t.contracts, t.price as entry_price,
                t.fill_price, t.total_cost, t.model_prob, t.edge,
                t.kelly_fraction, t.status, t.pnl, t.settled_at,
                t.settlement_value, t.created_at,
                km.close_time, km.threshold_f, km.is_above_contract,
                km.bracket_low, km.bracket_high, km.yes_price as current_yes
            FROM trades t
            LEFT JOIN kalshi_markets km ON km.market_ticker = t.market_ticker
            WHERE t.status IN ('filled', 'settled')
            ORDER BY
                CASE t.status WHEN 'filled' THEN 0 ELSE 1 END,
                t.created_at DESC
        """)
        rows = cur.fetchall()

        positions = []
        for r in rows:
            # Skip settled positions older than 7 days
            if r["status"] == "settled" and r["settled_at"]:
                try:
                    settled_dt = datetime.fromisoformat(r["settled_at"].replace("Z", "+00:00"))
                    from datetime import timezone
                    if (datetime.now(timezone.utc) - settled_dt).days > 7:
                        continue
                except Exception:
                    pass

            positions.append({
                "trade_id": r["trade_id"],
                "market_ticker": r["market_ticker"],
                "city": r["city"],
                "target_date": r["target_date"],
                "side": r["side"],
                "contracts": r["contracts"],
                "entry_price": r["fill_price"] or r["entry_price"],
                "total_cost": r["total_cost"],
                "model_prob": r["model_prob"],
                "edge_at_entry": r["edge"],
                "status": r["status"],
                "pnl": r["pnl"],
                "settlement_value": r["settlement_value"],
                "entered_at": r["created_at"],
                "current_yes": r["current_yes"],
                "contract_desc": _format_contract_desc(r) if r["is_above_contract"] is not None else r["market_ticker"].split("-")[-1],
            })

        return jsonify(positions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


@app.route("/api/tracker/<market_ticker>")
def api_tracker_detail(market_ticker):
    """Full position data: price history, signal trajectory, exit analysis."""
    conn = get_db()
    cur = conn.cursor()
    try:
        # 1. Trade info (most recent active trade for this ticker)
        cur.execute("""
            SELECT * FROM trades
            WHERE market_ticker = ? AND status IN ('filled', 'settled')
            ORDER BY created_at DESC LIMIT 1
        """, (market_ticker,))
        trade_row = cur.fetchone()
        trade = dict(trade_row) if trade_row else {}

        # 2. Price history from market_price_history
        cur.execute("""
            SELECT captured_at, yes_price, no_price, volume, status
            FROM market_price_history
            WHERE market_ticker = ?
            ORDER BY captured_at ASC
        """, (market_ticker,))
        price_history = [dict(r) for r in cur.fetchall()]

        # 3. Signal history (model probability trajectory)
        cur.execute("""
            SELECT computed_at, calibrated_prob, ensemble_prob,
                   hrrr_prob, nws_prob, blended_prob,
                   market_yes_price, raw_edge, abs_edge, lead_hours,
                   ensemble_run, hrrr_run
            FROM signals
            WHERE market_ticker = ?
            ORDER BY computed_at ASC
        """, (market_ticker,))
        signal_history = [dict(r) for r in cur.fetchall()]

        # 4. Current market data
        cur.execute("""
            SELECT * FROM kalshi_markets WHERE market_ticker = ?
        """, (market_ticker,))
        mkt_row = cur.fetchone()
        current_market = dict(mkt_row) if mkt_row else {}

        # 5. Latest signal for exit analysis
        latest_signal = signal_history[-1] if signal_history else {}

        # 6. Compute exit analysis
        exit_analysis = _compute_exit_analysis(trade, current_market, latest_signal)

        return jsonify({
            "trade": trade,
            "price_history": price_history,
            "signal_history": signal_history,
            "current_market": current_market,
            "exit_analysis": exit_analysis,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Manual trade placement from dashboard
# ---------------------------------------------------------------------------

@app.route("/api/trade/place", methods=["POST"])
def api_place_trade():
    """Place a live trade on Kalshi from the dashboard.

    Expects JSON: {market_ticker, city, target_date, side, contracts, price_cents,
                   model_prob, market_price, edge}
    """
    try:
        data = request.get_json(force=True)
        required = ["market_ticker", "side", "contracts", "price_cents"]
        missing = [k for k in required if k not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        settings = load_settings()
        if settings.mode != "live":
            return jsonify({"error": "Bot is in paper mode — switch to live first"}), 400

        contracts = int(data["contracts"])
        price_cents = int(data["price_cents"])
        side = data["side"]

        if contracts <= 0:
            return jsonify({"error": "Contracts must be > 0"}), 400
        if price_cents < 1 or price_cents > 99:
            return jsonify({"error": "Price must be 1-99 cents"}), 400
        if side not in ("yes", "no"):
            return jsonify({"error": "Side must be 'yes' or 'no'"}), 400

        total_cost = contracts * price_cents / 100.0

        # Risk check
        from src.strategy.risk_manager import RiskManager, PortfolioState
        rm = RiskManager(settings.strategy)
        live_balance = _get_kalshi_balance()
        if live_balance is None:
            return jsonify({"error": "Cannot fetch Kalshi balance"}), 500
        portfolio = rm.get_portfolio_state(live_balance, mode="live")
        check = rm.check_trade_allowed(
            city=data.get("city", ""),
            target_date=data.get("target_date", ""),
            total_cost=total_cost,
            portfolio=portfolio,
        )
        if not check.allowed:
            return jsonify({"error": f"Risk check failed: {check.reason}"}), 400

        # Place order
        from src.data.kalshi_client import KalshiClient
        from src.data.db import get_session, init_db
        from src.data.models import Trade
        import uuid as _uuid

        init_db(settings)

        with _kalshi_lock:
            global _kalshi_client
            if _kalshi_client is None:
                _kalshi_client = KalshiClient(settings)
            client = _kalshi_client

        order = client.create_order(
            ticker=data["market_ticker"],
            side=side,
            action="buy",
            order_type="limit",
            yes_price=price_cents if side == "yes" else None,
            no_price=price_cents if side == "no" else None,
            count=contracts,
        )

        # Record in DB
        now = datetime.now(timezone.utc).isoformat()
        session = get_session()
        try:
            trade = Trade(
                trade_id=str(_uuid.uuid4()),
                mode="live",
                market_ticker=data["market_ticker"],
                city=data.get("city", ""),
                target_date=data.get("target_date", ""),
                side=side,
                direction="buy",
                contracts=contracts,
                price=price_cents / 100.0,
                total_cost=total_cost,
                model_prob=data.get("model_prob"),
                market_price=data.get("market_price"),
                edge=data.get("edge"),
                kelly_fraction=data.get("kelly"),
                kalshi_order_id=order.order_id,
                status=order.status or "pending",
                created_at=now,
                updated_at=now,
            )
            session.add(trade)
            session.commit()
            logger.info(
                "DASHBOARD TRADE: %s %dx %s @ %dc (order=%s)",
                side.upper(), contracts, data["market_ticker"],
                price_cents, order.order_id
            )
            return jsonify({
                "ok": True,
                "order_id": order.order_id,
                "status": order.status,
                "side": side,
                "contracts": contracts,
                "price_cents": price_cents,
                "total_cost": total_cost,
            })
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    except Exception as e:
        logger.error("Dashboard trade failed: %s", e)
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Test Email API
# ---------------------------------------------------------------------------

@app.route("/api/email/test", methods=["POST"])
def api_test_email():
    """Send a test daily email report."""
    try:
        from src.monitoring.email_reporter import send_daily_email

        settings = load_settings()
        if not settings.email.enabled or not settings.email.app_password:
            return jsonify({"error": "Email not configured. Set GMAIL_APP_PASSWORD in .env and email.enabled in settings.yaml"}), 400

        live_balance = _get_kalshi_balance()
        if live_balance is not None:
            get_bankroll = lambda: live_balance
        else:
            get_bankroll = lambda: INITIAL_BANKROLL

        send_daily_email(settings, get_bankroll=get_bankroll)
        return jsonify({"ok": True, "recipient": settings.email.recipient})
    except Exception as e:
        logger.error("Test email failed: %s", e)
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Bracket Arbitrage API
# ---------------------------------------------------------------------------

@app.route("/api/arbitrage/scan")
def api_arbitrage_scan():
    """Scan for bracket arbitrage opportunities across all active events."""
    try:
        from src.strategy.arbitrage_scanner import scan_arbitrage
        opportunities = scan_arbitrage()
        return jsonify({
            "opportunities": [
                {
                    "city": a.city,
                    "target_date": a.target_date,
                    "event_ticker": a.event_ticker,
                    "n_brackets": a.n_brackets,
                    "sum_yes_ask": round(a.sum_yes_ask, 4),
                    "total_fees": round(a.total_fees, 4),
                    "guaranteed_profit": round(a.guaranteed_profit, 4),
                    "profitable": a.guaranteed_profit > 0,
                    "brackets": a.brackets,
                }
                for a in opportunities
            ],
            "count": len(opportunities),
            "profitable_count": sum(1 for a in opportunities if a.guaranteed_profit > 0),
        })
    except Exception as e:
        logger.error("Arbitrage scan failed: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/arbitrage/sweep", methods=["POST"])
def api_arbitrage_sweep():
    """Execute an arbitrage sweep: buy NO on every bracket in an event.

    Body: { "city": "NYC", "target_date": "2026-03-10", "contracts": 1 }
    """
    try:
        settings = load_settings()
        if settings.mode != "live":
            return jsonify({"error": "Arbitrage sweep only available in live mode"}), 400

        body = request.get_json() or {}
        city = body.get("city")
        target_date = body.get("target_date")
        contracts = body.get("contracts", 1)

        if not city or not target_date:
            return jsonify({"error": "city and target_date required"}), 400
        if contracts < 1 or contracts > 10:
            return jsonify({"error": "contracts must be 1-10"}), 400

        from src.strategy.arbitrage_scanner import scan_arbitrage, execute_sweep

        # Find the specific opportunity
        opportunities = scan_arbitrage()
        arb = None
        for a in opportunities:
            if a.city == city and a.target_date == target_date:
                arb = a
                break

        if arb is None:
            return jsonify({"error": f"No bracket set found for {city} {target_date}"}), 404

        if arb.guaranteed_profit <= 0:
            return jsonify({
                "error": f"No arbitrage opportunity — sum(YES)={arb.sum_yes_ask:.2%}, "
                         f"need >{1.0 + arb.total_fees:.2%} for profit"
            }), 400

        # Execute the sweep
        global _kalshi_client
        with _kalshi_lock:
            if _kalshi_client is None:
                from src.data.kalshi_client import KalshiClient
                _kalshi_client = KalshiClient(settings)
            client = _kalshi_client

        results = execute_sweep(client, arb, contracts=contracts)

        return jsonify({
            "ok": True,
            "city": city,
            "target_date": target_date,
            "n_brackets": arb.n_brackets,
            "contracts_per_bracket": contracts,
            "guaranteed_profit_per_contract": round(arb.guaranteed_profit, 4),
            "total_guaranteed_profit": round(arb.guaranteed_profit * contracts, 4),
            "orders": results,
        })

    except Exception as e:
        logger.error("Arbitrage sweep failed: %s", e)
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Convergence Dashboard API
# ---------------------------------------------------------------------------

@app.route("/api/convergence/observations")
def api_convergence_observations():
    """Hourly temperature observations for the convergence chart."""
    city = request.args.get("city")
    date_str = request.args.get("date")
    if not city or not date_str:
        return jsonify({"error": "city and date required"}), 400

    try:
        from datetime import date as date_type
        from src.utils.time_utils import get_observation_window_utc
        cities = load_cities()
        city_cfg = cities.get(city)
        if not city_cfg:
            return jsonify({"error": f"Unknown city: {city}"}), 400

        target = date_type.fromisoformat(date_str)
        start_utc, end_utc = get_observation_window_utc(target, city_cfg.timezone)

        conn = get_db()
        rows = conn.execute(
            """SELECT observed_at, temperature_f, description
               FROM hourly_observations
               WHERE city = ? AND observed_at >= ? AND observed_at < ?
               ORDER BY observed_at ASC""",
            (city, start_utc.isoformat(), end_utc.isoformat())
        ).fetchall()
        conn.close()

        observations = [
            {"t": r[0], "temp_f": r[1], "description": r[2]}
            for r in rows if r[1] is not None
        ]

        running_max = None
        running_max_time = None
        for o in observations:
            if running_max is None or o["temp_f"] > running_max:
                running_max = o["temp_f"]
                running_max_time = o["t"]

        return jsonify({
            "observations": observations,
            "running_max": running_max,
            "running_max_time": running_max_time,
            "city": city,
            "date": date_str,
            "timezone": city_cfg.timezone,
        })
    except Exception as e:
        logger.error("Convergence observations failed: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/convergence/forecasts")
def api_convergence_forecasts():
    """Ensemble spread and HRRR forecast for the convergence chart."""
    city = request.args.get("city")
    date_str = request.args.get("date")
    if not city or not date_str:
        return jsonify({"error": "city and date required"}), 400

    try:
        conn = get_db()

        # Latest ensemble run — get all 31 member daily maxes
        ens_rows = conn.execute(
            """SELECT member, temperature_f, model_run_time
               FROM ensemble_forecasts
               WHERE city = ? AND valid_time LIKE ?
                 AND model_run_time = (
                   SELECT MAX(model_run_time) FROM ensemble_forecasts
                   WHERE city = ? AND valid_time LIKE ?
                 )
               ORDER BY member ASC""",
            (city, date_str + "%", city, date_str + "%")
        ).fetchall()

        ensemble_data = None
        if ens_rows:
            temps = [r[1] for r in ens_rows]
            temps.sort()
            n = len(temps)
            p10_idx = max(0, int(n * 0.10))
            p90_idx = min(n - 1, int(n * 0.90))
            ensemble_data = {
                "min": min(temps),
                "max": max(temps),
                "p10": temps[p10_idx],
                "p90": temps[p90_idx],
                "median": temps[n // 2],
                "mean": round(sum(temps) / n, 1),
                "members": temps,
                "run_time": ens_rows[0][2],
                "n_members": n,
            }

        # Latest HRRR daily max
        hrrr_row = conn.execute(
            """SELECT temperature_f, model_run_time
               FROM hrrr_forecasts
               WHERE city = ? AND valid_time LIKE ?
               ORDER BY model_run_time DESC LIMIT 1""",
            (city, date_str + "%")
        ).fetchone()
        conn.close()

        hrrr_data = None
        if hrrr_row:
            hrrr_data = {
                "max_f": hrrr_row[0],
                "run_time": hrrr_row[1],
            }

        return jsonify({
            "ensemble": ensemble_data,
            "hrrr": hrrr_data,
            "city": city,
            "date": date_str,
        })
    except Exception as e:
        logger.error("Convergence forecasts failed: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/convergence/distribution")
def api_convergence_distribution():
    """Probability distribution per bracket over time for playback."""
    city = request.args.get("city")
    date_str = request.args.get("date")
    if not city or not date_str:
        return jsonify({"error": "city and date required"}), 400

    try:
        conn = get_db()

        # Bracket definitions
        bracket_rows = conn.execute(
            """SELECT market_ticker, bracket_low, bracket_high,
                      is_above_contract, threshold_f, yes_price, no_price
               FROM kalshi_markets
               WHERE city = ? AND target_date = ?
                 AND status IN ('open', 'active')
               ORDER BY COALESCE(bracket_low, threshold_f - 1, -999) ASC""",
            (city, date_str)
        ).fetchall()

        # Identify T (threshold) contracts to determine low-end vs high-end
        t_thresholds = []
        for r in bracket_rows:
            ticker, bl, bh, is_above, thresh, yp, np_ = r
            if is_above and bl is None and bh is None and thresh is not None:
                t_thresholds.append(thresh)
        t_thresholds.sort()
        low_t = t_thresholds[0] if t_thresholds else None
        high_t = t_thresholds[-1] if len(t_thresholds) > 1 else None

        brackets = []
        for r in bracket_rows:
            ticker, bl, bh, is_above, thresh, yp, np_ = r

            if is_above and bl is None and bh is None and thresh is not None:
                if thresh == low_t:
                    # Low-end bracket: "≤(threshold-1)°F"
                    label = f"{int(thresh) - 1}\u00b0 or below"
                    sort_temp = float(thresh) - 1.0
                elif thresh == high_t:
                    # High-end bracket: "≥(threshold+1)°F"
                    label = f"{int(thresh) + 1}\u00b0 or above"
                    sort_temp = float(thresh) + 1.0
                else:
                    label = f"T{int(thresh)}"
                    sort_temp = float(thresh)
            elif bl is not None and bh is not None:
                label = f"{int(bl)}-{int(bh)}\u00b0F"
                sort_temp = (bl + bh) / 2.0
            elif bl is None and bh is not None:
                label = f"\u2264{int(bh)}\u00b0F"
                sort_temp = float(bh)
            elif bl is not None and bh is None:
                label = f"\u2265{int(bl)}\u00b0F"
                sort_temp = float(bl)
            else:
                label = ticker
                sort_temp = 0.0

            brackets.append({
                "ticker": ticker,
                "label": label,
                "sort_temp": sort_temp,
                "bracket_prob": round(yp, 4) if yp else 0,
                "bracket_low": bl,
                "bracket_high": bh,
                "is_above": bool(is_above),
                "threshold": thresh,
                "current_yes": yp,
                "current_no": np_,
            })

        brackets.sort(key=lambda b: b["sort_temp"])

        # Signal history — all snapshots for these markets
        tickers = [b["ticker"] for b in brackets]
        if not tickers:
            conn.close()
            return jsonify({"brackets": [], "snapshots": []})

        placeholders = ",".join("?" for _ in tickers)
        sig_rows = conn.execute(
            f"""SELECT computed_at, market_ticker,
                       COALESCE(calibrated_prob, blended_prob, ensemble_prob) as model_prob,
                       market_yes_price
                FROM signals
                WHERE target_date = ? AND city = ?
                  AND market_ticker IN ({placeholders})
                ORDER BY computed_at ASC""",
            (date_str, city, *tickers)
        ).fetchall()
        conn.close()

        # Group by computed_at to create snapshots
        from collections import OrderedDict
        snapshot_map = OrderedDict()
        for computed_at, ticker, model_prob, mkt_price in sig_rows:
            if computed_at not in snapshot_map:
                snapshot_map[computed_at] = {}
            snapshot_map[computed_at][ticker] = {
                "model_prob": round(model_prob, 4) if model_prob else None,
                "market_price": round(mkt_price, 4) if mkt_price else None,
            }

        # Normalize model_prob per snapshot so they sum to 1.0
        ticker_set = {b["ticker"] for b in brackets}
        snapshots = []
        for t, probs in snapshot_map.items():
            # Sum model probs for normalization
            total = sum(
                p["model_prob"] for p in probs.values()
                if p.get("model_prob") and p["model_prob"] > 0
            )
            normalized = {}
            for tk in ticker_set:
                if tk in probs:
                    mp = probs[tk].get("model_prob")
                    mkp = probs[tk].get("market_price")
                    normalized[tk] = {
                        "model_prob": round(mp / total, 4) if mp and total > 0 else 0,
                        "market_price": round(mkp, 4) if mkp else 0,
                    }
                else:
                    normalized[tk] = {"model_prob": 0, "market_price": 0}
            snapshots.append({"t": t, "probabilities": normalized})

        return jsonify({
            "brackets": brackets,
            "snapshots": snapshots,
            "city": city,
            "date": date_str,
        })
    except Exception as e:
        logger.error("Convergence distribution failed: %s", e)
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Risk Limits API — read/write strategy limits in settings.yaml
# ---------------------------------------------------------------------------

SETTINGS_YAML_PATH = PROJECT_ROOT / "config" / "settings.yaml"

# Fields the dashboard is allowed to edit (whitelist)
_EDITABLE_LIMITS = {
    "daily_spend_limit",
    "max_position_dollars",
    "max_concurrent_positions",
    "max_positions_per_city",
    "max_positions_per_date",
    "daily_loss_limit",
}

# Validation rules: (min, max, type)
_LIMIT_RULES = {
    "daily_spend_limit":       (0, 500, float),
    "max_position_dollars":    (0.5, 200, float),
    "max_concurrent_positions": (1, 100, int),
    "max_positions_per_city":  (1, 50, int),
    "max_positions_per_date":  (1, 50, int),
    "daily_loss_limit":        (1, 1000, float),
}


@app.route("/api/risk-limits", methods=["GET"])
def get_risk_limits():
    """Return the current risk limit values from settings.yaml."""
    try:
        import yaml
        with open(SETTINGS_YAML_PATH) as f:
            raw = yaml.safe_load(f)
        strategy = raw.get("strategy", {})
        limits = {k: strategy.get(k) for k in _EDITABLE_LIMITS if k in strategy}
        return jsonify(limits)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/risk-limits", methods=["PUT"])
def update_risk_limits():
    """Update risk limits in settings.yaml.

    Accepts a JSON body with any subset of the editable limit fields.
    Validates ranges, writes back to YAML, preserving comments and structure.
    """
    try:
        import yaml
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Validate each field
        errors = []
        cleaned = {}
        for key, value in data.items():
            if key not in _EDITABLE_LIMITS:
                errors.append(f"Unknown field: {key}")
                continue
            rule = _LIMIT_RULES.get(key)
            if rule:
                lo, hi, typ = rule
                try:
                    value = typ(value)
                except (ValueError, TypeError):
                    errors.append(f"{key}: must be {typ.__name__}")
                    continue
                if value < lo or value > hi:
                    errors.append(f"{key}: must be between {lo} and {hi}")
                    continue
            cleaned[key] = value

        if errors:
            return jsonify({"error": "Validation failed", "details": errors}), 400

        if not cleaned:
            return jsonify({"error": "No valid fields to update"}), 400

        # Read current YAML (preserving structure)
        with open(SETTINGS_YAML_PATH) as f:
            raw = yaml.safe_load(f)

        strategy = raw.setdefault("strategy", {})
        for key, value in cleaned.items():
            strategy[key] = value

        # Write back
        with open(SETTINGS_YAML_PATH, "w") as f:
            yaml.dump(raw, f, default_flow_style=False, sort_keys=False)

        logger.info("Risk limits updated via dashboard: %s", cleaned)
        return jsonify({"ok": True, "updated": cleaned})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Scorecard page route
# ---------------------------------------------------------------------------

@app.route("/scorecard")
def scorecard_page():
    return send_from_directory(STATIC_DIR, "scorecard.html")


# ---------------------------------------------------------------------------
# Scorecard API endpoints — Model performance comparison
# ---------------------------------------------------------------------------

@app.route("/api/model-scores")
def api_model_scores():
    """Aggregated model scorecard data.

    Query params:
        days: Number of days to look back (default 30)
        city: Optional city filter
    """
    try:
        days = int(request.args.get("days", 30))
        city = request.args.get("city", "")

        conn = get_db()
        query = """
            SELECT model_source,
                   COUNT(*) as n,
                   AVG(brier_contribution) as avg_brier,
                   SUM(was_best_model) as wins,
                   SUM(was_worst_model) as losses,
                   AVG(distance_from_outcome) as avg_distance,
                   AVG(max_prob_swing) as avg_swing
            FROM model_scorecards
            WHERE created_at >= datetime('now', ?)
        """
        params = [f"-{days} days"]

        if city:
            query += " AND city = ?"
            params.append(city)

        query += " GROUP BY model_source ORDER BY avg_brier ASC"

        rows = conn.execute(query, params).fetchall()
        conn.close()

        results = []
        for r in rows:
            results.append({
                "model_source": r[0],
                "n": r[1],
                "avg_brier": round(r[2], 4) if r[2] else None,
                "wins": r[3] or 0,
                "losses": r[4] or 0,
                "win_rate": round((r[3] or 0) / r[1], 3) if r[1] > 0 else 0,
                "avg_distance": round(r[5], 4) if r[5] else None,
                "avg_swing": round(r[6], 4) if r[6] else None,
            })

        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/model-comparison")
def api_model_comparison():
    """Rolling Brier per model source for charting.

    Returns daily average Brier by model source.
    Query params:
        days: Number of days (default 30)
    """
    try:
        days = int(request.args.get("days", 30))
        conn = get_db()

        rows = conn.execute("""
            SELECT target_date, model_source,
                   AVG(brier_contribution) as avg_brier,
                   COUNT(*) as n
            FROM model_scorecards
            WHERE created_at >= datetime('now', ?)
            GROUP BY target_date, model_source
            ORDER BY target_date
        """, [f"-{days} days"]).fetchall()
        conn.close()

        # Structure as {date: {source: brier}}
        by_date = {}
        for r in rows:
            d = r[0]
            if d not in by_date:
                by_date[d] = {}
            by_date[d][r[1]] = round(r[2], 4) if r[2] else None

        return jsonify({
            "dates": sorted(by_date.keys()),
            "data": by_date,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/postmortem/<target_date>")
def api_postmortem(target_date):
    """Get postmortem report for a target date.

    Query params:
        city: Optional city filter (returns all cities if omitted)
    """
    try:
        city = request.args.get("city", "")
        settings = load_settings()
        archive_dir = settings.archival.absolute_dir / "postmortem"

        reports = []
        if city:
            path = archive_dir / f"{target_date}_{city}.json"
            if path.exists():
                with open(path) as f:
                    reports.append(json.load(f))
        else:
            # Load all city reports for this date
            if archive_dir.exists():
                for path in sorted(archive_dir.glob(f"{target_date}_*.json")):
                    with open(path) as f:
                        reports.append(json.load(f))

        return jsonify(reports)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/scorecard/details")
def api_scorecard_details():
    """Per-market scorecard details for a specific date and city.

    Query params:
        date: Target date (required)
        city: City name (required)
    """
    try:
        target_date = request.args.get("date", "")
        city = request.args.get("city", "")
        if not target_date or not city:
            return jsonify({"error": "date and city required"}), 400

        conn = get_db()
        rows = conn.execute("""
            SELECT market_ticker, model_source, final_prob, outcome,
                   observed_high_f, brier_contribution, first_prob,
                   prob_at_24h, prob_at_12h, prob_at_6h,
                   max_prob_swing, final_lead_hours,
                   distance_from_outcome, was_best_model, was_worst_model
            FROM model_scorecards
            WHERE city = ? AND target_date = ?
            ORDER BY market_ticker, model_source
        """, [city, target_date]).fetchall()
        conn.close()

        results = []
        for r in rows:
            results.append({
                "market_ticker": r[0],
                "model_source": r[1],
                "final_prob": r[2],
                "outcome": r[3],
                "observed_high_f": r[4],
                "brier_contribution": round(r[5], 4) if r[5] else None,
                "first_prob": r[6],
                "prob_at_24h": r[7],
                "prob_at_12h": r[8],
                "prob_at_6h": r[9],
                "max_prob_swing": r[10],
                "final_lead_hours": r[11],
                "distance_from_outcome": r[12],
                "was_best_model": bool(r[13]),
                "was_worst_model": bool(r[14]),
            })

        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Evolution Dashboard — model forecast evolution vs actual temperature
# ---------------------------------------------------------------------------

@app.route("/evolution")
def evolution_page():
    return send_from_directory(STATIC_DIR, "evolution.html")


def _market_contracts_to_percentiles(contracts):
    """Convert a snapshot of market contract prices into implied P10/P50/P90 temps.

    Uses bracket + above contracts to build a piecewise-uniform CDF,
    then linearly interpolates for the requested percentiles.

    Returns dict {p10, p50, p90} or None if data is insufficient.
    """
    brackets = sorted(
        [c for c in contracts if not c["above"] and c["yp"] is not None],
        key=lambda c: c["blo"],
    )
    above_list = sorted(
        [c for c in contracts if c["above"] and c["yp"] is not None],
        key=lambda c: c["thresh"],
    )

    if not brackets and len(above_list) < 2:
        return None

    # --- Build probability bins: (lower_edge, upper_edge, probability) ---
    bins = []

    # Each bracket covers integer temps [low, high], so upper edge = high + 1
    for bc in brackets:
        bins.append((bc["blo"], bc["bhi"] + 1, bc["yp"]))

    # Use the HIGHEST above contract for the upper tail
    if above_list:
        above_high = above_list[-1]
        # Tail extension: half the total bracket span, minimum 3°F
        if brackets:
            span = brackets[-1]["bhi"] - brackets[0]["blo"]
            tail = max(3.0, span / 2.0)
        else:
            tail = 5.0
        bins.append((above_high["thresh"] + 1, above_high["thresh"] + 1 + tail,
                      above_high["yp"]))

    # Below-low bin: residual probability
    total_assigned = sum(b[2] for b in bins)
    below_prob = max(0.0, 1.0 - total_assigned)

    if brackets:
        lo_edge = brackets[0]["blo"]
        if above_list:
            span = brackets[-1]["bhi"] - brackets[0]["blo"]
            tail = max(3.0, span / 2.0)
        else:
            tail = 5.0
        bins.insert(0, (lo_edge - tail, lo_edge, below_prob))
    elif above_list:
        lo_thresh = above_list[0]["thresh"]
        bins.insert(0, (lo_thresh - 5, lo_thresh + 1, below_prob))

    bins.sort(key=lambda x: x[0])

    # Normalise
    total = sum(b[2] for b in bins)
    if total < 0.02:
        return None
    bins = [(lo, hi, p / total) for lo, hi, p in bins]

    # --- Build CDF (piecewise-uniform within each bin) ---
    cdf = []
    cum = 0.0
    for lo, hi, p in bins:
        cdf.append((lo, cum))
        cum += p
        cdf.append((hi, cum))

    def _interp(target):
        for i in range(1, len(cdf)):
            if cdf[i][1] >= target - 1e-9:
                t0, p0 = cdf[i - 1]
                t1, p1 = cdf[i]
                if abs(p1 - p0) < 1e-12:
                    return (t0 + t1) / 2
                return t0 + (target - p0) / (p1 - p0) * (t1 - t0)
        return cdf[-1][0]

    return {
        "p10": round(_interp(0.10), 1),
        "p50": round(_interp(0.50), 1),
        "p90": round(_interp(0.90), 1),
    }


@app.route("/api/evolution/temperature")
def api_evolution_temperature():
    """Temperature forecast evolution from all model sources + observations.

    Returns time-aligned series showing how each model's forecast evolved
    as new runs came in, alongside actual observed temperatures.
    """
    city = request.args.get("city", "NYC")
    date_str = request.args.get("date", date.today().isoformat())

    try:
        conn = get_db()

        # 1. All GFS ensemble runs for this city+date
        #    EnsembleForecast stores one row per (city, model_run_time, valid_time, member)
        #    valid_time matches the target date. Group by model_run_time to get per-run stats.
        ens_rows = conn.execute(
            """SELECT model_run_time, member, temperature_f
               FROM ensemble_forecasts
               WHERE city = ? AND date(valid_time) = ?
               ORDER BY model_run_time ASC, member ASC""",
            (city, date_str),
        ).fetchall()

        # Group by model_run_time → compute median, P10, P90
        from collections import defaultdict
        ens_runs = defaultdict(list)
        for run_time, _member, temp_f in ens_rows:
            ens_runs[run_time].append(temp_f)

        ensemble_series = []
        for run_time, temps in sorted(ens_runs.items()):
            temps.sort()
            n = len(temps)
            ensemble_series.append({
                "t": run_time,
                "median": temps[n // 2],
                "mean": round(sum(temps) / n, 1),
                "p10": temps[max(0, int(n * 0.10))],
                "p90": temps[min(n - 1, int(n * 0.90))],
                "min": min(temps),
                "max": max(temps),
                "n_members": n,
            })

        # 2. All ECMWF IFS ensemble runs for this city+date
        ecmwf_rows = conn.execute(
            """SELECT model_run_time, member, temperature_f
               FROM ecmwf_forecasts
               WHERE city = ? AND date(valid_time) = ?
               ORDER BY model_run_time ASC, member ASC""",
            (city, date_str),
        ).fetchall()

        ecmwf_runs = defaultdict(list)
        for run_time, _member, temp_f in ecmwf_rows:
            ecmwf_runs[run_time].append(temp_f)

        ecmwf_series = []
        for run_time, temps in sorted(ecmwf_runs.items()):
            temps.sort()
            n = len(temps)
            ecmwf_series.append({
                "t": run_time,
                "median": temps[n // 2],
                "mean": round(sum(temps) / n, 1),
                "p10": temps[max(0, int(n * 0.10))],
                "p90": temps[min(n - 1, int(n * 0.90))],
                "n_members": n,
            })

        # 3. All HRRR runs for this city+date
        hrrr_rows = conn.execute(
            """SELECT model_run_time, temperature_f
               FROM hrrr_forecasts
               WHERE city = ? AND date(valid_time) = ?
               ORDER BY model_run_time ASC""",
            (city, date_str),
        ).fetchall()

        hrrr_series = [{"t": r[0], "max_f": r[1]} for r in hrrr_rows]

        # 4. All NWS forecasts for this city+date
        nws_rows = conn.execute(
            """SELECT fetched_at, high_f
               FROM nws_forecasts
               WHERE city = ? AND forecast_date = ?
               ORDER BY fetched_at ASC""",
            (city, date_str),
        ).fetchall()

        nws_series = [{"t": r[0], "high_f": r[1]} for r in nws_rows if r[1]]

        # 4. Hourly observations throughout the target date
        obs_rows = conn.execute(
            """SELECT observed_at, temperature_f, description
               FROM hourly_observations
               WHERE city = ? AND date(observed_at) = ?
               ORDER BY observed_at ASC""",
            (city, date_str),
        ).fetchall()

        observations = [
            {"t": r[0], "temp_f": r[1], "desc": r[2]} for r in obs_rows
        ]

        # Compute running max
        running_max = []
        current_max = None
        for obs in observations:
            if current_max is None or obs["temp_f"] > current_max:
                current_max = obs["temp_f"]
            running_max.append({"t": obs["t"], "max_f": current_max})

        # 5. Settlement temperature
        obs_row = conn.execute(
            """SELECT high_f FROM observations
               WHERE city = ? AND date = ?
               ORDER BY fetched_at DESC LIMIT 1""",
            (city, date_str),
        ).fetchone()

        settlement_temp = obs_row[0] if obs_row else None

        # 6. Market-implied temperature distribution from price history
        #    Join price snapshots with market metadata to get strike-level CDF
        #    over time, then interpolate P10 / P50 / P90.
        mkt_rows = conn.execute(
            """SELECT mph.captured_at, mph.yes_price,
                      km.is_above_contract, km.threshold_f,
                      km.bracket_low, km.bracket_high
               FROM market_price_history mph
               JOIN kalshi_markets km ON km.market_ticker = mph.market_ticker
               WHERE km.city = ? AND km.target_date = ?
               ORDER BY mph.captured_at ASC""",
            (city, date_str),
        ).fetchall()

        # Group by snapshot time (truncate to minute for alignment)
        snap_map = defaultdict(list)
        for cap_at, yes_p, is_above, thresh, b_lo, b_hi in mkt_rows:
            t_key = cap_at[:16]  # "2026-03-10T01:55"
            snap_map[t_key].append({
                "yp": yes_p if yes_p is not None else 0.0,
                "above": bool(is_above),
                "thresh": thresh,
                "blo": b_lo,
                "bhi": b_hi,
            })

        market_implied = []
        for t_key, contracts in sorted(snap_map.items()):
            result = _market_contracts_to_percentiles(contracts)
            if result:
                result["t"] = t_key
                market_implied.append(result)

        conn.close()

        # Compute expected observation peak (approx 2-3 PM local time on target date)
        # and the city's timezone for correct display
        city_tz_map = {
            "NYC": "America/New_York", "Chicago": "America/Chicago",
            "Miami": "America/New_York", "LA": "America/Los_Angeles",
            "Denver": "America/Denver",
        }
        city_tz = city_tz_map.get(city, "America/New_York")

        # Expected high temp occurs ~2-3 PM local time
        # Convert to UTC ISO string for the chart annotation
        try:
            from zoneinfo import ZoneInfo
            local_tz = ZoneInfo(city_tz)
            target = datetime.strptime(date_str, "%Y-%m-%d")
            # Peak temp around 2:30 PM local
            peak_local = target.replace(hour=14, minute=30, tzinfo=local_tz)
            expected_peak_utc = peak_local.astimezone(ZoneInfo("UTC")).isoformat()
        except Exception:
            expected_peak_utc = None

        return jsonify({
            "ensemble": ensemble_series,
            "ecmwf": ecmwf_series,
            "hrrr": hrrr_series,
            "nws": nws_series,
            "observations": observations,
            "running_max": running_max,
            "settlement_temp": settlement_temp,
            "market_implied": market_implied,
            "city": city,
            "date": date_str,
            "city_timezone": city_tz,
            "expected_peak_utc": expected_peak_utc,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/evolution/probabilities")
def api_evolution_probabilities():
    """Per-model probability trajectories from the Signal table."""
    city = request.args.get("city", "NYC")
    date_str = request.args.get("date", date.today().isoformat())
    ticker = request.args.get("ticker", "")

    try:
        conn = get_db()

        if ticker:
            rows = conn.execute(
                """SELECT computed_at, ensemble_prob, hrrr_prob, nws_prob,
                          blended_prob, calibrated_prob, market_yes_price,
                          lead_hours, market_ticker, ecmwf_prob
                   FROM signals
                   WHERE city = ? AND target_date = ? AND market_ticker = ?
                   ORDER BY computed_at ASC""",
                (city, date_str, ticker),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT computed_at, ensemble_prob, hrrr_prob, nws_prob,
                          blended_prob, calibrated_prob, market_yes_price,
                          lead_hours, market_ticker, ecmwf_prob
                   FROM signals
                   WHERE city = ? AND target_date = ?
                   ORDER BY computed_at ASC""",
                (city, date_str),
            ).fetchall()

        # Get list of available tickers
        tickers = sorted(set(r[8] for r in rows))

        series = [{
            "t": r[0], "ensemble": r[1], "hrrr": r[2], "nws": r[3],
            "blended": r[4], "calibrated": r[5], "market": r[6],
            "lead_hours": r[7], "ticker": r[8], "ecmwf": r[9],
        } for r in rows]

        conn.close()

        return jsonify({
            "series": series,
            "tickers": tickers,
            "city": city,
            "date": date_str,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/evolution/weights")
def api_evolution_weights():
    """Adaptive blend weight history from JSON file."""
    days = int(request.args.get("days", 30))

    weights_path = PROJECT_ROOT / "data" / "adaptive_weights.json"
    if not weights_path.exists():
        return jsonify({"history": [], "current": None})

    try:
        with open(weights_path) as f:
            data = json.load(f)

        history = data.get("history", [])

        # Filter to last N days
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        history = [h for h in history if h.get("at", "") >= cutoff]

        return jsonify({
            "current": data.get("adaptive_weights"),
            "brier_scores": data.get("brier_scores"),
            "n_markets": data.get("n_markets"),
            "updated_at": data.get("updated_at"),
            "history": history,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Auto-refresh background thread
# ---------------------------------------------------------------------------

def _auto_refresh_loop():
    """Periodically check data freshness and trigger fetches when stale.

    Uses the existing DataFreshnessTracker to decide when new model data
    should be available.  Runs every 10 minutes in a daemon thread.
    """
    import time as _time
    CHECK_INTERVAL = 600  # 10 minutes

    # Wait 30 seconds after startup before first check
    _time.sleep(30)

    while True:
        try:
            settings = load_settings()
            tracker = DataFreshnessTracker(
                DB_PATH,
                gfs_lag_hours=getattr(
                    settings.scheduler, "gfs_availability_lag_hours", 4.5
                ),
                hrrr_lag_hours=getattr(
                    settings.scheduler, "hrrr_availability_lag_hours", 1.0
                ),
            )

            if tracker.should_fetch_gfs():
                logger.info("[auto-refresh] GFS data stale — triggering fetch")
                _do_direct_fetch("gfs")

            if tracker.should_fetch_hrrr():
                logger.info("[auto-refresh] HRRR data stale — triggering fetch")
                _do_direct_fetch("hrrr")

            if tracker.should_fetch_nws():
                logger.info("[auto-refresh] NWS data stale — triggering fetch")
                _do_direct_fetch("nws")

        except Exception as e:
            logger.error("[auto-refresh] Error: %s", e)

        _time.sleep(CHECK_INTERVAL)


# Start auto-refresh thread (daemon=True so it dies with the process)
_auto_refresh_thread = threading.Thread(
    target=_auto_refresh_loop,
    daemon=True,
    name="auto-refresh",
)
_auto_refresh_thread.start()


# ---------------------------------------------------------------------------
# Observation polling thread — collects hourly temps from NWS for convergence
# ---------------------------------------------------------------------------

def _observation_polling_loop():
    """Poll NWS for current temperature observations every 5 minutes."""
    import time as _time
    POLL_INTERVAL = 300  # 5 minutes

    _time.sleep(10)  # Brief startup delay

    while True:
        try:
            settings = load_settings()
            cities = load_cities()
            nws = NWSClient(settings)
            conn = get_db_rw()
            _ensure_hourly_observations_table(conn)

            now_utc = datetime.now(timezone.utc).isoformat()
            for city_name, city in cities.items():
                try:
                    obs = nws.get_latest_observation(city)
                    if obs and obs.temperature_f is not None:
                        conn.execute(
                            """INSERT OR IGNORE INTO hourly_observations
                               (city, station, observed_at, temperature_f,
                                description, fetched_at)
                               VALUES (?, ?, ?, ?, ?, ?)""",
                            (city_name, city.station, obs.timestamp,
                             round(obs.temperature_f, 1), obs.description,
                             now_utc)
                        )
                except Exception as e:
                    logger.warning("[obs-poll] Failed for %s: %s", city_name, e)

            conn.commit()
            conn.close()
            nws.close()
        except Exception as e:
            logger.error("[obs-poll] Error: %s", e)

        _time.sleep(POLL_INTERVAL)


_obs_poll_thread = threading.Thread(
    target=_observation_polling_loop,
    daemon=True,
    name="obs-poll",
)
_obs_poll_thread.start()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Ensure new tables exist on startup
    try:
        conn = get_db_rw()
        _ensure_hourly_observations_table(conn)
        _ensure_strategies_table(conn)
        _ensure_backtest_table(conn)
        _ensure_optimization_table(conn)
        _ensure_strategy_id_column(conn)
        conn.commit()
        conn.close()
    except Exception:
        pass  # DB might not exist yet
    app.run(host="0.0.0.0", port=5050, debug=True)
