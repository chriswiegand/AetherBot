"""Parquet-based data archiver for high-resolution weather model data.

Preserves hourly GFS ensemble member data and HRRR hourly profiles that
are currently discarded after daily-max extraction. This data is essential
for model development, postmortem analysis, and calibration research.

Storage layout uses Hive-style partitioning so pandas can filter efficiently:
    data/archives/ensemble_hourly/city=NYC/2026-03-09_run_2026-03-09T00.parquet
    data/archives/hrrr_hourly/city=NYC/2026-03-09_run_2026-03-09T12.parquet
    data/archives/convergence/city=NYC/2026-03-09.parquet

Estimated storage: ~270 KB/day, ~194 MB over 2 years.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config.settings import ArchivalConfig, PROJECT_ROOT

logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> None:
    """Create directory tree if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def _run_tag(model_run_time: str) -> str:
    """Convert ISO model run time to a filesystem-safe tag.

    '2026-03-09T00:00:00+00:00' -> '2026-03-09T00'
    """
    # Take up to hour precision, strip colons for safety
    return model_run_time[:13].replace(":", "")


def archive_ensemble_hourly(
    config: ArchivalConfig,
    city: str,
    target_date: str,
    model_run_time: str,
    member_hourly: dict[int, list[float]],
    valid_times: list[str],
) -> Path | None:
    """Archive GFS ensemble hourly member data to Parquet.

    Preserves the full 31-member × N-hour temperature grid that is
    currently reduced to just 31 daily-max values.

    Args:
        config: Archival configuration
        city: City name (e.g., "NYC")
        target_date: Target date ISO string (e.g., "2026-03-09")
        model_run_time: Model run time ISO string
        member_hourly: {member_index: [hourly_temps_f]} — 31 members
        valid_times: ISO time strings for each hourly step

    Returns:
        Path to written Parquet file, or None on failure
    """
    if not config.enabled:
        return None

    try:
        rows = []
        fetched_at = datetime.now(timezone.utc).isoformat()

        for member_idx, temps in member_hourly.items():
            for i, temp_f in enumerate(temps):
                if i >= len(valid_times):
                    break
                if temp_f is None or temp_f != temp_f:  # Skip None/NaN
                    continue
                rows.append({
                    "valid_time_utc": valid_times[i],
                    "member": int(member_idx),
                    "temperature_f": float(temp_f),
                    "city": city,
                    "target_date": target_date,
                    "model_run_time": model_run_time,
                    "fetched_at": fetched_at,
                })

        if not rows:
            logger.debug(f"No ensemble hourly data to archive for {city} {target_date}")
            return None

        # Build Parquet with explicit schema for compact storage
        schema = pa.schema([
            ("valid_time_utc", pa.string()),
            ("member", pa.int8()),
            ("temperature_f", pa.float32()),
            ("city", pa.string()),
            ("target_date", pa.string()),
            ("model_run_time", pa.string()),
            ("fetched_at", pa.string()),
        ])

        table = pa.Table.from_pylist(rows, schema=schema)

        # Hive-style partition: city=NYC/
        run_tag = _run_tag(model_run_time)
        out_dir = config.absolute_dir / "ensemble_hourly" / f"city={city}"
        _ensure_dir(out_dir)
        out_path = out_dir / f"{target_date}_run_{run_tag}.parquet"

        pq.write_table(table, out_path, compression="snappy")

        logger.info(
            f"Archived {len(rows)} ensemble hourly rows for {city} "
            f"{target_date} ({out_path.stat().st_size / 1024:.1f} KB)"
        )
        return out_path

    except Exception as e:
        # Never let archive failure block the operational pipeline
        logger.error(f"Failed to archive ensemble hourly for {city} {target_date}: {e}")
        return None


def archive_hrrr_hourly(
    config: ArchivalConfig,
    city: str,
    target_date: str,
    model_run_time: str,
    hourly_temps: list[float],
    valid_times: list[str],
) -> Path | None:
    """Archive HRRR hourly temperature profile to Parquet.

    Preserves the full hourly profile that is currently reduced to
    just the daily max.

    Args:
        config: Archival configuration
        city: City name
        target_date: Target date ISO string
        model_run_time: Model run time ISO string
        hourly_temps: Hourly temperatures (F) within the observation window
        valid_times: ISO time strings for each hourly step

    Returns:
        Path to written Parquet file, or None on failure
    """
    if not config.enabled:
        return None

    try:
        rows = []
        fetched_at = datetime.now(timezone.utc).isoformat()

        for i, temp_f in enumerate(hourly_temps):
            if i >= len(valid_times):
                break
            if temp_f is None or temp_f != temp_f:
                continue
            rows.append({
                "valid_time_utc": valid_times[i],
                "temperature_f": float(temp_f),
                "city": city,
                "target_date": target_date,
                "model_run_time": model_run_time,
                "fetched_at": fetched_at,
            })

        if not rows:
            logger.debug(f"No HRRR hourly data to archive for {city} {target_date}")
            return None

        schema = pa.schema([
            ("valid_time_utc", pa.string()),
            ("temperature_f", pa.float32()),
            ("city", pa.string()),
            ("target_date", pa.string()),
            ("model_run_time", pa.string()),
            ("fetched_at", pa.string()),
        ])

        table = pa.Table.from_pylist(rows, schema=schema)

        run_tag = _run_tag(model_run_time)
        out_dir = config.absolute_dir / "hrrr_hourly" / f"city={city}"
        _ensure_dir(out_dir)
        out_path = out_dir / f"{target_date}_run_{run_tag}.parquet"

        pq.write_table(table, out_path, compression="snappy")

        logger.info(
            f"Archived {len(rows)} HRRR hourly rows for {city} "
            f"{target_date} ({out_path.stat().st_size / 1024:.1f} KB)"
        )
        return out_path

    except Exception as e:
        logger.error(f"Failed to archive HRRR hourly for {city} {target_date}: {e}")
        return None


def archive_convergence_trajectory(
    config: ArchivalConfig,
    city: str,
    target_date: str,
    observed_high: float,
    session,
) -> Path | None:
    """Archive all Signal rows for a city+date with the observed outcome.

    Called once after settlement to capture the full probability convergence
    trajectory from all model sources. This is the primary dataset for
    postmortem analysis and calibrator training.

    Args:
        config: Archival configuration
        city: City name
        target_date: Target date ISO string
        observed_high: Actual observed high temperature (F)
        session: SQLAlchemy session

    Returns:
        Path to written Parquet file, or None on failure
    """
    if not config.enabled:
        return None

    try:
        from src.data.models import Signal

        signals = (
            session.query(Signal)
            .filter_by(city=city, target_date=target_date)
            .order_by(Signal.fetched_at)
            .all()
        )

        if not signals:
            logger.debug(f"No signals to archive for convergence: {city} {target_date}")
            return None

        rows = []
        for s in signals:
            row = {
                "city": s.city,
                "target_date": s.target_date,
                "market_ticker": s.market_ticker,
                "fetched_at": s.fetched_at,
                "lead_hours": float(s.lead_hours) if s.lead_hours else None,
                "ensemble_prob": float(s.ensemble_prob) if s.ensemble_prob else None,
                "hrrr_prob": float(s.hrrr_prob) if s.hrrr_prob else None,
                "nws_prob": float(s.nws_prob) if s.nws_prob else None,
                "blended_prob": float(s.blended_prob) if s.blended_prob else None,
                "calibrated_prob": float(s.calibrated_prob) if s.calibrated_prob else None,
                "market_yes_price": float(s.market_yes_price) if s.market_yes_price else None,
                "edge": float(s.edge) if s.edge else None,
                "observed_high_f": float(observed_high),
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        out_dir = config.absolute_dir / "convergence" / f"city={city}"
        _ensure_dir(out_dir)
        out_path = out_dir / f"{target_date}.parquet"

        df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)

        logger.info(
            f"Archived {len(rows)} convergence signals for {city} "
            f"{target_date} (observed={observed_high}°F, "
            f"{out_path.stat().st_size / 1024:.1f} KB)"
        )
        return out_path

    except Exception as e:
        logger.error(f"Failed to archive convergence for {city} {target_date}: {e}")
        return None
