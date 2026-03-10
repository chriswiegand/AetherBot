"""Matplotlib-based temperature evolution chart renderer.

Produces PNG images for email embedding, replicating the same
data and visual style as the dashboard evolution page.
Uses raw sqlite3 (WAL read-only) to avoid SQLAlchemy threading issues.
"""

from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")  # Headless backend — must be before pyplot import
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from src.config.settings import PROJECT_ROOT

logger = logging.getLogger(__name__)

DB_PATH = PROJECT_ROOT / "data" / "weather_bot.db"

# Dashboard-matching color palette
COLOR_GFS_BAND = "#f6ad55"       # Orange
COLOR_GFS_LINE = "#dd6b20"       # Dark orange
COLOR_ECMWF_BAND = "#f778ba"     # Pink
COLOR_ECMWF_LINE = "#d53f8c"     # Dark pink
COLOR_ICON_BAND = "#38b2ac"      # Teal
COLOR_ICON_LINE = "#2c9a8f"      # Dark teal
COLOR_GEM_BAND = "#9ae6b4"       # Yellow-green
COLOR_GEM_LINE = "#68d391"       # Dark yellow-green
COLOR_HRRR = "#63b3ed"           # Blue
COLOR_NWS = "#4fd1c5"            # Cyan
COLOR_MARKET_BAND = "#b794f4"    # Purple
COLOR_MARKET_LINE = "#805ad5"    # Dark purple
COLOR_OBS = "#68d391"            # Green
COLOR_SETTLEMENT = "#fc8181"     # Red
COLOR_PEAK = "#d29922"           # Yellow
COLOR_BG = "#1a202c"             # Dark background
COLOR_GRID = "#2d3748"           # Grid lines
COLOR_TEXT = "#e2e8f0"           # Light text

CITY_TZ_MAP = {
    "NYC": "America/New_York",
    "Chicago": "America/Chicago",
    "Miami": "America/New_York",
    "LA": "America/Los_Angeles",
    "Denver": "America/Denver",
}


def _get_db(db_path: Path | None = None) -> sqlite3.Connection:
    """Open a read-only WAL connection."""
    path = db_path or DB_PATH
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _parse_ts(ts_str: str) -> datetime | None:
    """Parse an ISO timestamp string to a UTC-aware datetime."""
    if not ts_str:
        return None
    try:
        s = ts_str.replace(" ", "T")
        # Handle timezone-aware strings
        if "+" in s and s.index("+") > 10:
            return datetime.fromisoformat(s)
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        # Bare ISO → treat as UTC
        return datetime.fromisoformat(s).replace(tzinfo=ZoneInfo("UTC"))
    except Exception:
        return None


def _market_contracts_to_percentiles(contracts: list[dict]) -> dict | None:
    """Convert market contract prices into implied P10/P50/P90 temps.

    Replicates dashboard/app.py logic exactly.
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

    bins = []
    for bc in brackets:
        bins.append((bc["blo"], bc["bhi"] + 1, bc["yp"]))

    if above_list:
        above_high = above_list[-1]
        if brackets:
            span = brackets[-1]["bhi"] - brackets[0]["blo"]
            tail = max(3.0, span / 2.0)
        else:
            tail = 5.0
        bins.append((
            above_high["thresh"] + 1,
            above_high["thresh"] + 1 + tail,
            above_high["yp"],
        ))

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

    total = sum(b[2] for b in bins)
    if total < 0.02:
        return None
    bins = [(lo, hi, p / total) for lo, hi, p in bins]

    # Build CDF
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


def _query_chart_data(
    conn: sqlite3.Connection, city: str, date_str: str
) -> dict:
    """Query all evolution data for one city+date (mirrors dashboard API)."""

    # 1. GFS ensemble
    ens_rows = conn.execute(
        """SELECT model_run_time, member, temperature_f
           FROM ensemble_forecasts
           WHERE city = ? AND date(valid_time) = ?
           ORDER BY model_run_time ASC, member ASC""",
        (city, date_str),
    ).fetchall()

    ens_runs = defaultdict(list)
    for r in ens_rows:
        ens_runs[r["model_run_time"]].append(r["temperature_f"])

    ensemble_series = []
    for run_time, temps in sorted(ens_runs.items()):
        temps.sort()
        n = len(temps)
        ensemble_series.append({
            "t": _parse_ts(run_time),
            "median": temps[n // 2],
            "p10": temps[max(0, int(n * 0.10))],
            "p90": temps[min(n - 1, int(n * 0.90))],
        })

    # 2. ECMWF IFS ensemble
    ecmwf_rows = conn.execute(
        """SELECT model_run_time, member, temperature_f
           FROM ecmwf_forecasts
           WHERE city = ? AND date(valid_time) = ?
           ORDER BY model_run_time ASC, member ASC""",
        (city, date_str),
    ).fetchall()

    ecmwf_runs = defaultdict(list)
    for r in ecmwf_rows:
        ecmwf_runs[r["model_run_time"]].append(r["temperature_f"])

    ecmwf_series = []
    for run_time, temps in sorted(ecmwf_runs.items()):
        temps.sort()
        n = len(temps)
        ecmwf_series.append({
            "t": _parse_ts(run_time),
            "median": temps[n // 2],
            "p10": temps[max(0, int(n * 0.10))],
            "p90": temps[min(n - 1, int(n * 0.90))],
        })

    # 2b. ICON-EPS ensemble
    icon_eps_rows = conn.execute(
        """SELECT model_run_time, member, temperature_f
           FROM icon_eps_forecasts
           WHERE city = ? AND date(valid_time) = ?
           ORDER BY model_run_time ASC, member ASC""",
        (city, date_str),
    ).fetchall()

    icon_eps_runs = defaultdict(list)
    for r in icon_eps_rows:
        icon_eps_runs[r["model_run_time"]].append(r["temperature_f"])

    icon_eps_series = []
    for run_time, temps in sorted(icon_eps_runs.items()):
        temps.sort()
        n = len(temps)
        icon_eps_series.append({
            "t": _parse_ts(run_time),
            "median": temps[n // 2],
            "p10": temps[max(0, int(n * 0.10))],
            "p90": temps[min(n - 1, int(n * 0.90))],
        })

    # 2c. GEM/GEPS ensemble
    gem_rows = conn.execute(
        """SELECT model_run_time, member, temperature_f
           FROM gem_forecasts
           WHERE city = ? AND date(valid_time) = ?
           ORDER BY model_run_time ASC, member ASC""",
        (city, date_str),
    ).fetchall()

    gem_runs = defaultdict(list)
    for r in gem_rows:
        gem_runs[r["model_run_time"]].append(r["temperature_f"])

    gem_series = []
    for run_time, temps in sorted(gem_runs.items()):
        temps.sort()
        n = len(temps)
        gem_series.append({
            "t": _parse_ts(run_time),
            "median": temps[n // 2],
            "p10": temps[max(0, int(n * 0.10))],
            "p90": temps[min(n - 1, int(n * 0.90))],
        })

    # 3. HRRR deterministic
    hrrr_rows = conn.execute(
        """SELECT model_run_time, temperature_f
           FROM hrrr_forecasts
           WHERE city = ? AND date(valid_time) = ?
           ORDER BY model_run_time ASC""",
        (city, date_str),
    ).fetchall()
    hrrr_series = [
        {"t": _parse_ts(r["model_run_time"]), "max_f": r["temperature_f"]}
        for r in hrrr_rows
    ]

    # 4. NWS forecasts
    nws_rows = conn.execute(
        """SELECT fetched_at, high_f
           FROM nws_forecasts
           WHERE city = ? AND forecast_date = ?
           ORDER BY fetched_at ASC""",
        (city, date_str),
    ).fetchall()
    nws_series = [
        {"t": _parse_ts(r["fetched_at"]), "high_f": r["high_f"]}
        for r in nws_rows if r["high_f"]
    ]

    # 5. Hourly observations
    obs_rows = conn.execute(
        """SELECT observed_at, temperature_f
           FROM hourly_observations
           WHERE city = ? AND date(observed_at) = ?
           ORDER BY observed_at ASC""",
        (city, date_str),
    ).fetchall()
    observations = [
        {"t": _parse_ts(r["observed_at"]), "temp_f": r["temperature_f"]}
        for r in obs_rows
    ]

    # Running max
    running_max = []
    current_max = None
    for obs in observations:
        if current_max is None or obs["temp_f"] > current_max:
            current_max = obs["temp_f"]
        running_max.append({"t": obs["t"], "max_f": current_max})

    # 6. Settlement temperature
    settle_row = conn.execute(
        """SELECT high_f FROM observations
           WHERE city = ? AND date = ?
           ORDER BY fetched_at DESC LIMIT 1""",
        (city, date_str),
    ).fetchone()
    settlement_temp = settle_row["high_f"] if settle_row else None

    # 7. Market-implied distribution
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

    snap_map = defaultdict(list)
    for r in mkt_rows:
        t_key = r["captured_at"][:16]
        snap_map[t_key].append({
            "yp": r["yes_price"] if r["yes_price"] is not None else 0.0,
            "above": bool(r["is_above_contract"]),
            "thresh": r["threshold_f"],
            "blo": r["bracket_low"],
            "bhi": r["bracket_high"],
        })

    market_implied = []
    for t_key, contracts in sorted(snap_map.items()):
        result = _market_contracts_to_percentiles(contracts)
        if result:
            result["t"] = _parse_ts(t_key)
            market_implied.append(result)

    return {
        "ensemble": ensemble_series,
        "ecmwf": ecmwf_series,
        "icon_eps": icon_eps_series,
        "gem": gem_series,
        "hrrr": hrrr_series,
        "nws": nws_series,
        "observations": observations,
        "running_max": running_max,
        "settlement_temp": settlement_temp,
        "market_implied": market_implied,
    }


def render_evolution_chart(
    city: str,
    target_date: str,
    db_path: Path | None = None,
) -> bytes | None:
    """Render a dark-themed temperature evolution chart as PNG bytes.

    Args:
        city: City code (e.g. "NYC")
        target_date: ISO date string (e.g. "2026-03-10")
        db_path: Optional override for the database path

    Returns:
        PNG image bytes, or None if no data available.
    """
    conn = _get_db(db_path)
    try:
        data = _query_chart_data(conn, city, target_date)
    finally:
        conn.close()

    # Check if we have ANY data to plot
    has_data = any([
        data["ensemble"], data["ecmwf"], data["icon_eps"], data["gem"],
        data["hrrr"], data["nws"], data["observations"], data["market_implied"],
    ])
    if not has_data:
        logger.info(f"No chart data for {city} {target_date}")
        return None

    # --- Build the figure ---
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    # Style axes
    ax.tick_params(colors=COLOR_TEXT, which="both")
    ax.spines["bottom"].set_color(COLOR_GRID)
    ax.spines["left"].set_color(COLOR_GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.label.set_color(COLOR_TEXT)
    ax.yaxis.label.set_color(COLOR_TEXT)
    ax.title.set_color(COLOR_TEXT)
    ax.grid(True, color=COLOR_GRID, alpha=0.5, linewidth=0.5)

    plotted_anything = False

    # 1. GFS P10-P90 band + median
    if data["ensemble"]:
        ts = [d["t"] for d in data["ensemble"] if d["t"]]
        p10 = [d["p10"] for d in data["ensemble"] if d["t"]]
        p90 = [d["p90"] for d in data["ensemble"] if d["t"]]
        med = [d["median"] for d in data["ensemble"] if d["t"]]
        if ts:
            ax.fill_between(ts, p10, p90, alpha=0.15, color=COLOR_GFS_BAND,
                            label="GFS P10-P90")
            ax.plot(ts, med, color=COLOR_GFS_LINE, linewidth=1.5,
                    marker="o", markersize=3, label="GFS Median")
            plotted_anything = True

    # 2. ECMWF P10-P90 band + median
    if data["ecmwf"]:
        ts = [d["t"] for d in data["ecmwf"] if d["t"]]
        p10 = [d["p10"] for d in data["ecmwf"] if d["t"]]
        p90 = [d["p90"] for d in data["ecmwf"] if d["t"]]
        med = [d["median"] for d in data["ecmwf"] if d["t"]]
        if ts:
            ax.fill_between(ts, p10, p90, alpha=0.10, color=COLOR_ECMWF_BAND,
                            label="ECMWF P10-P90")
            ax.plot(ts, med, color=COLOR_ECMWF_LINE, linewidth=1.5,
                    marker="s", markersize=3, label="ECMWF Median")
            plotted_anything = True

    # 2b. ICON-EPS P10-P90 band + median
    if data["icon_eps"]:
        ts = [d["t"] for d in data["icon_eps"] if d["t"]]
        p10 = [d["p10"] for d in data["icon_eps"] if d["t"]]
        p90 = [d["p90"] for d in data["icon_eps"] if d["t"]]
        med = [d["median"] for d in data["icon_eps"] if d["t"]]
        if ts:
            ax.fill_between(ts, p10, p90, alpha=0.10, color=COLOR_ICON_BAND,
                            label="ICON-EPS P10-P90")
            ax.plot(ts, med, color=COLOR_ICON_LINE, linewidth=1.5,
                    marker="D", markersize=3, label="ICON-EPS Median")
            plotted_anything = True

    # 2c. GEM P10-P90 band + median
    if data["gem"]:
        ts = [d["t"] for d in data["gem"] if d["t"]]
        p10 = [d["p10"] for d in data["gem"] if d["t"]]
        p90 = [d["p90"] for d in data["gem"] if d["t"]]
        med = [d["median"] for d in data["gem"] if d["t"]]
        if ts:
            ax.fill_between(ts, p10, p90, alpha=0.10, color=COLOR_GEM_BAND,
                            label="GEM P10-P90")
            ax.plot(ts, med, color=COLOR_GEM_LINE, linewidth=1.5,
                    marker="v", markersize=3, label="GEM Median")
            plotted_anything = True

    # 3. HRRR points
    if data["hrrr"]:
        ts = [d["t"] for d in data["hrrr"] if d["t"]]
        vals = [d["max_f"] for d in data["hrrr"] if d["t"]]
        if ts:
            ax.plot(ts, vals, color=COLOR_HRRR, linewidth=1.2,
                    marker="^", markersize=4, label="HRRR")
            plotted_anything = True

    # 4. NWS stepped line
    if data["nws"]:
        ts = [d["t"] for d in data["nws"] if d["t"]]
        vals = [d["high_f"] for d in data["nws"] if d["t"]]
        if ts:
            ax.step(ts, vals, where="post", color=COLOR_NWS,
                    linewidth=1.2, label="NWS")
            plotted_anything = True

    # 5. Market-implied P10-P90 band + P50
    if data["market_implied"]:
        ts = [d["t"] for d in data["market_implied"] if d["t"]]
        p10 = [d["p10"] for d in data["market_implied"] if d["t"]]
        p50 = [d["p50"] for d in data["market_implied"] if d["t"]]
        p90 = [d["p90"] for d in data["market_implied"] if d["t"]]
        if ts:
            ax.fill_between(ts, p10, p90, alpha=0.10, color=COLOR_MARKET_BAND,
                            label="Market P10-P90")
            ax.plot(ts, p50, color=COLOR_MARKET_LINE, linewidth=1.2,
                    linestyle="--", label="Market P50")
            plotted_anything = True

    # 6. Observations + running max
    if data["observations"]:
        ts = [d["t"] for d in data["observations"] if d["t"]]
        vals = [d["temp_f"] for d in data["observations"] if d["t"]]
        if ts:
            ax.plot(ts, vals, color=COLOR_OBS, linewidth=1.0,
                    alpha=0.6, label="Observed")
            plotted_anything = True

    if data["running_max"]:
        ts = [d["t"] for d in data["running_max"] if d["t"]]
        vals = [d["max_f"] for d in data["running_max"] if d["t"]]
        if ts:
            ax.plot(ts, vals, color=COLOR_OBS, linewidth=1.5,
                    linestyle="--", label="Running Max")

    # 7. Settlement horizontal line
    if data["settlement_temp"] is not None:
        ax.axhline(y=data["settlement_temp"], color=COLOR_SETTLEMENT,
                   linewidth=1.5, linestyle="-", alpha=0.8,
                   label=f"Settlement: {data['settlement_temp']}°F")

    # 8. Expected peak annotation
    city_tz_name = CITY_TZ_MAP.get(city, "America/New_York")
    try:
        local_tz = ZoneInfo(city_tz_name)
        target = datetime.strptime(target_date, "%Y-%m-%d")
        peak_local = target.replace(hour=14, minute=30, tzinfo=local_tz)
        peak_utc = peak_local.astimezone(ZoneInfo("UTC"))
        ax.axvline(x=peak_utc, color=COLOR_PEAK, linewidth=1.0,
                   linestyle=":", alpha=0.7, label="Expected High")
    except Exception:
        pass

    if not plotted_anything:
        plt.close(fig)
        return None

    # Format axes
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M", tz=ZoneInfo("UTC")))
    fig.autofmt_xdate(rotation=30, ha="right")
    ax.set_ylabel("Temperature (°F)", fontsize=10)
    ax.set_title(f"{city} — {target_date}", fontsize=13, fontweight="bold", pad=10)

    # Legend
    legend = ax.legend(
        loc="upper left", fontsize=7, framealpha=0.7,
        facecolor=COLOR_BG, edgecolor=COLOR_GRID,
        labelcolor=COLOR_TEXT, ncol=2,
    )

    plt.tight_layout()

    # Export to PNG bytes
    buf = BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    buf.seek(0)
    png_bytes = buf.read()

    logger.info(
        f"Rendered evolution chart for {city} {target_date}: "
        f"{len(png_bytes) / 1024:.0f} KB"
    )
    return png_bytes


def render_all_active_charts(
    db_path: Path | None = None,
) -> list[dict]:
    """Render evolution charts for all active city+date combinations.

    Returns:
        List of {"city": str, "date": str, "png": bytes}
    """
    path = db_path or DB_PATH
    conn = _get_db(path)
    try:
        today_str = date.today().isoformat()
        rows = conn.execute(
            """SELECT DISTINCT city, target_date
               FROM kalshi_markets
               WHERE target_date >= ? AND status IN ('open', 'active')
               ORDER BY target_date, city""",
            (today_str,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        logger.info("No active markets found for chart rendering")
        return []

    charts = []
    for r in rows:
        city = r["city"]
        tdate = r["target_date"]
        try:
            png = render_evolution_chart(city, tdate, path)
            if png:
                charts.append({"city": city, "date": tdate, "png": png})
        except Exception as e:
            logger.error(f"Failed to render chart for {city} {tdate}: {e}")

    logger.info(f"Rendered {len(charts)} evolution charts")
    return charts
