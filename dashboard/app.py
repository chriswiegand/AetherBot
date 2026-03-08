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
# Direct-fetch infrastructure (dashboard calls fetchers without the bot)
# ---------------------------------------------------------------------------

_refresh_locks = {s: threading.Lock() for s in ("gfs", "hrrr", "nws")}
_refresh_status: dict[str, dict] = {
    s: {"running": False, "last_result": None} for s in ("gfs", "hrrr", "nws")
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

        # Bankroll = initial + settled PnL - open trade costs
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
    """List all open (filled) positions."""
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                market_ticker,
                side,
                contracts,
                price,
                total_cost,
                city,
                target_date,
                edge,
                model_prob
            FROM trades
            WHERE status = 'filled'
            ORDER BY target_date ASC, city ASC
            """
        )
        rows = cur.fetchall()
        conn.close()

        positions = []
        for r in rows:
            positions.append({
                "market_ticker": r["market_ticker"],
                "side": r["side"],
                "contracts": r["contracts"],
                "price": r["price"],
                "cost": r["total_cost"],
                "city": r["city"],
                "target_date": r["target_date"],
                "edge": r["edge"],
                "model_prob": r["model_prob"],
            })

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
                price,
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
    """Most recent signal per market_ticker (latest computed_at)."""
    try:
        conn = get_db()
        cur = conn.cursor()

        # Use a window function to pick the latest signal per market_ticker
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
                s.computed_at
            FROM signals s
            INNER JOIN (
                SELECT market_ticker, MAX(computed_at) AS max_computed
                FROM signals
                GROUP BY market_ticker
            ) latest
                ON s.market_ticker = latest.market_ticker
               AND s.computed_at = latest.max_computed
            ORDER BY s.target_date ASC, s.city ASC
            """
        )
        rows = cur.fetchall()
        conn.close()

        signals = []
        for r in rows:
            signals.append({
                "market_ticker": r["market_ticker"],
                "city": r["city"],
                "target_date": r["target_date"],
                "ensemble_prob": r["ensemble_prob"],
                "blended_prob": r["blended_prob"],
                "calibrated_prob": r["calibrated_prob"],
                "edge": r["edge"],
                "market_price": r["market_yes_price"],
                "lead_hours": r["lead_hours"],
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
    valid_sources = {"gfs", "hrrr", "nws"}
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


@app.route("/api/refresh/<source>/status")
def api_refresh_status(source):
    """Check status of a running or completed refresh."""
    valid_sources = {"gfs", "hrrr", "nws"}
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

        # Current bankroll
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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Ensure new tables exist on startup
    try:
        conn = get_db_rw()
        _ensure_strategies_table(conn)
        _ensure_backtest_table(conn)
        _ensure_optimization_table(conn)
        _ensure_strategy_id_column(conn)
        conn.commit()
        conn.close()
    except Exception:
        pass  # DB might not exist yet
    app.run(host="0.0.0.0", port=5050, debug=True)
