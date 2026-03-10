"""Data freshness tracking for smart fetch scheduling.

Tracks when each data source was last updated and determines whether
new model runs should be available for fetching.

GFS Ensemble: Runs at 00Z, 06Z, 12Z, 18Z — data available ~4.5h after run start
ECMWF IFS: Runs at 00Z, 06Z, 12Z, 18Z — data available ~6h after run start
HRRR: Runs every hour — data available ~55min after run start
NWS: Updates ~2x/day — no fixed schedule, check staleness
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# GFS runs at these UTC hours
GFS_RUN_HOURS = [0, 6, 12, 18]

# Staleness thresholds (minutes)
THRESHOLDS = {
    "gfs_ensemble": {"green": 420, "yellow": 780},   # 7h / 13h
    "ecmwf":        {"green": 480, "yellow": 840},   # 8h / 14h (slightly laggier than GFS)
    "hrrr":         {"green": 120, "yellow": 240},    # 2h / 4h
    "nws":          {"green": 360, "yellow": 720},    # 6h / 12h
}


@dataclass
class FreshnessStatus:
    """Status of a single data source."""
    source: str            # "gfs_ensemble", "hrrr", "nws"
    latest_model_run: str | None   # ISO8601 UTC
    latest_fetch: str | None       # ISO8601 UTC
    staleness_minutes: float
    is_fresh: bool
    status: str            # "green", "yellow", "red"
    next_expected: str | None      # ISO8601 UTC — when next model run should be available


class DataFreshnessTracker:
    """Checks data freshness and decides whether to fetch new data."""

    def __init__(self, db_path: Path, gfs_lag_hours: float = 4.5, hrrr_lag_hours: float = 1.0,
                 ecmwf_lag_hours: float = 6.0):
        self.db_path = db_path
        self.gfs_lag_hours = gfs_lag_hours
        self.hrrr_lag_hours = hrrr_lag_hours
        self.ecmwf_lag_hours = ecmwf_lag_hours

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Core freshness queries
    # ------------------------------------------------------------------

    def _get_latest_model_run(self, table: str, run_col: str = "model_run_time") -> str | None:
        """Get the latest model_run_time from a table."""
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute(f"SELECT MAX({run_col}) as latest FROM {table}")
            row = cur.fetchone()
            conn.close()
            return row["latest"] if row and row["latest"] else None
        except Exception as e:
            logger.warning(f"Could not query {table}: {e}")
            return None

    def _get_latest_fetch(self, table: str) -> str | None:
        """Get the latest fetched_at from a table."""
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute(f"SELECT MAX(fetched_at) as latest FROM {table}")
            row = cur.fetchone()
            conn.close()
            return row["latest"] if row and row["latest"] else None
        except Exception as e:
            logger.warning(f"Could not query {table} fetched_at: {e}")
            return None

    def _compute_staleness(self, latest_time: str | None) -> float:
        """Compute minutes since latest_time. Returns 9999 if None."""
        if not latest_time:
            return 9999.0
        try:
            # Handle both "2026-03-06T..." and "2026-03-06 ..." formats
            ts = latest_time.replace(" ", "T")
            if not ts.endswith("Z") and "+" not in ts and "-" not in ts[10:]:
                ts += "Z"
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            delta = datetime.now(timezone.utc) - dt
            return max(0.0, delta.total_seconds() / 60.0)
        except Exception:
            return 9999.0

    def _status_from_staleness(self, source: str, staleness_min: float) -> str:
        """Return 'green', 'yellow', or 'red' based on staleness thresholds."""
        thresholds = THRESHOLDS.get(source, {"green": 360, "yellow": 720})
        if staleness_min <= thresholds["green"]:
            return "green"
        elif staleness_min <= thresholds["yellow"]:
            return "yellow"
        return "red"

    # ------------------------------------------------------------------
    # GFS schedule awareness
    # ------------------------------------------------------------------

    def _latest_expected_gfs_run(self) -> datetime:
        """Compute the latest GFS run whose data should be available now.

        GFS runs at 00Z, 06Z, 12Z, 18Z. Data is available ~gfs_lag_hours after.
        """
        now = datetime.now(timezone.utc)
        available_cutoff = now - timedelta(hours=self.gfs_lag_hours)

        # Walk backwards through today's and yesterday's run hours
        for day_offset in [0, -1]:
            check_day = now.date() + timedelta(days=day_offset)
            for run_hour in reversed(GFS_RUN_HOURS):
                run_time = datetime(
                    check_day.year, check_day.month, check_day.day,
                    run_hour, 0, 0, tzinfo=timezone.utc
                )
                if run_time <= available_cutoff:
                    return run_time

        # Fallback: oldest possible
        yesterday = now.date() - timedelta(days=1)
        return datetime(yesterday.year, yesterday.month, yesterday.day, 0, 0, 0, tzinfo=timezone.utc)

    def _next_expected_gfs_run(self) -> datetime:
        """When the next GFS data should become available."""
        now = datetime.now(timezone.utc)
        for day_offset in [0, 1]:
            check_day = now.date() + timedelta(days=day_offset)
            for run_hour in GFS_RUN_HOURS:
                run_time = datetime(
                    check_day.year, check_day.month, check_day.day,
                    run_hour, 0, 0, tzinfo=timezone.utc
                )
                available_time = run_time + timedelta(hours=self.gfs_lag_hours)
                if available_time > now:
                    return available_time
        # Fallback
        return now + timedelta(hours=6)

    def _latest_expected_hrrr_run(self) -> datetime:
        """Compute the latest HRRR run whose data should be available now.

        HRRR runs every hour. Data available ~hrrr_lag_hours after.
        """
        now = datetime.now(timezone.utc)
        available_cutoff = now - timedelta(hours=self.hrrr_lag_hours)
        # Latest run hour is the floor of available_cutoff
        run_time = available_cutoff.replace(minute=0, second=0, microsecond=0)
        return run_time

    # ------------------------------------------------------------------
    # Should-fetch decisions
    # ------------------------------------------------------------------

    def should_fetch_gfs(self) -> bool:
        """True if a newer GFS run should be available than what's in the DB."""
        latest_in_db = self._get_latest_model_run("ensemble_forecasts")
        expected = self._latest_expected_gfs_run()

        if not latest_in_db:
            logger.info("No GFS data in DB — should fetch")
            return True

        try:
            ts = latest_in_db.replace(" ", "T")
            if not ts.endswith("Z") and "+" not in ts and "-" not in ts[10:]:
                ts += "Z"
            db_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return True

        if expected > db_time:
            logger.info(
                f"Newer GFS run available: expected={expected.isoformat()}, "
                f"db_latest={latest_in_db}"
            )
            return True

        return False

    def _latest_expected_ecmwf_run(self) -> datetime:
        """Compute the latest ECMWF run whose data should be available now.

        ECMWF IFS runs at 00Z, 06Z, 12Z, 18Z. Data available ~ecmwf_lag_hours after.
        """
        now = datetime.now(timezone.utc)
        available_cutoff = now - timedelta(hours=self.ecmwf_lag_hours)
        for day_offset in [0, -1]:
            check_day = now.date() + timedelta(days=day_offset)
            for run_hour in reversed(GFS_RUN_HOURS):  # Same 4x/day schedule
                run_time = datetime(
                    check_day.year, check_day.month, check_day.day,
                    run_hour, 0, 0, tzinfo=timezone.utc
                )
                if run_time <= available_cutoff:
                    return run_time
        yesterday = now.date() - timedelta(days=1)
        return datetime(yesterday.year, yesterday.month, yesterday.day, 0, 0, 0, tzinfo=timezone.utc)

    def should_fetch_ecmwf(self) -> bool:
        """True if a newer ECMWF run should be available than what's in the DB."""
        latest_in_db = self._get_latest_model_run("ecmwf_forecasts")
        expected = self._latest_expected_ecmwf_run()

        if not latest_in_db:
            logger.info("No ECMWF data in DB — should fetch")
            return True

        try:
            ts = latest_in_db.replace(" ", "T")
            if not ts.endswith("Z") and "+" not in ts and "-" not in ts[10:]:
                ts += "Z"
            db_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return True

        if expected > db_time:
            logger.info(
                f"Newer ECMWF run available: expected={expected.isoformat()}, "
                f"db_latest={latest_in_db}"
            )
            return True

        return False

    def should_fetch_hrrr(self) -> bool:
        """True if a newer HRRR run should be available than what's in the DB."""
        latest_in_db = self._get_latest_model_run("hrrr_forecasts")
        expected = self._latest_expected_hrrr_run()

        if not latest_in_db:
            logger.info("No HRRR data in DB — should fetch")
            return True

        try:
            ts = latest_in_db.replace(" ", "T")
            if not ts.endswith("Z") and "+" not in ts and "-" not in ts[10:]:
                ts += "Z"
            db_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return True

        if expected > db_time:
            logger.info(
                f"Newer HRRR run available: expected={expected.isoformat()}, "
                f"db_latest={latest_in_db}"
            )
            return True

        return False

    def should_fetch_nws(self) -> bool:
        """True if NWS data is stale (older than 4 hours)."""
        latest_fetch = self._get_latest_fetch("nws_forecasts")
        staleness = self._compute_staleness(latest_fetch)
        if staleness > 240:  # 4 hours
            logger.info(f"NWS data stale ({staleness:.0f}min) — should fetch")
            return True
        return False

    # ------------------------------------------------------------------
    # Full freshness report
    # ------------------------------------------------------------------

    def get_freshness(self, source: str) -> FreshnessStatus:
        """Get freshness status for a single source."""
        if source == "gfs_ensemble":
            latest_run = self._get_latest_model_run("ensemble_forecasts")
            latest_fetch = self._get_latest_fetch("ensemble_forecasts")
            staleness = self._compute_staleness(latest_run)
            next_exp = self._next_expected_gfs_run().isoformat()
        elif source == "ecmwf":
            latest_run = self._get_latest_model_run("ecmwf_forecasts")
            latest_fetch = self._get_latest_fetch("ecmwf_forecasts")
            staleness = self._compute_staleness(latest_run)
            next_exp = None  # Similar to GFS schedule
        elif source == "hrrr":
            latest_run = self._get_latest_model_run("hrrr_forecasts")
            latest_fetch = self._get_latest_fetch("hrrr_forecasts")
            staleness = self._compute_staleness(latest_run)
            now = datetime.now(timezone.utc)
            next_exp = (now.replace(minute=0, second=0, microsecond=0)
                        + timedelta(hours=1 + self.hrrr_lag_hours)).isoformat()
        elif source == "nws":
            latest_run = None
            latest_fetch = self._get_latest_fetch("nws_forecasts")
            staleness = self._compute_staleness(latest_fetch)
            next_exp = None  # NWS has no fixed schedule
        else:
            return FreshnessStatus(
                source=source, latest_model_run=None, latest_fetch=None,
                staleness_minutes=9999, is_fresh=False, status="red",
                next_expected=None,
            )

        status = self._status_from_staleness(source, staleness)
        is_fresh = status == "green"

        return FreshnessStatus(
            source=source,
            latest_model_run=latest_run,
            latest_fetch=latest_fetch,
            staleness_minutes=round(staleness, 1),
            is_fresh=is_fresh,
            status=status,
            next_expected=next_exp,
        )

    def get_all_freshness(self) -> list[FreshnessStatus]:
        """Return freshness for all 4 sources."""
        return [
            self.get_freshness("gfs_ensemble"),
            self.get_freshness("ecmwf"),
            self.get_freshness("hrrr"),
            self.get_freshness("nws"),
        ]
