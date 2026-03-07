"""AetherBot Dashboard - Flask web server.

Serves the dashboard UI and provides JSON API endpoints for
trade monitoring, PnL tracking, signal inspection, market data,
strategy management, backtesting, and parameter optimization.

Usage:
    python dashboard/app.py
"""

import json
import sqlite3
import sys
import time
from datetime import datetime, date
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

# Add project root to path so we can import src modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DB_PATH = PROJECT_ROOT / "data" / "weather_bot.db"
STATIC_DIR = Path(__file__).resolve().parent  # dashboard/

INITIAL_BANKROLL = 10_000.0

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
            created_at TEXT NOT NULL,
            duration_seconds REAL
        )
    """)


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

        from src.backtest.optimizer import BacktestOptimizer

        t0 = time.time()
        optimizer = BacktestOptimizer()
        results = optimizer.grid_search(
            param_ranges=param_ranges,
            start_date=start_date,
            end_date=end_date,
            target_metric=target_metric,
        )
        duration = time.time() - t0

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
                results_json, best_params_json, created_at, duration_seconds)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                name,
                json.dumps(param_ranges),
                target_metric,
                len(results),
                json.dumps(results),
                json.dumps(best.get("params", {})),
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
