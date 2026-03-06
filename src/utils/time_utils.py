"""Timezone and observation window utilities.

NWS CLI uses Local Standard Time (LST) year-round for the observation day.
During DST, the window is 1:00 AM to 12:59 AM civil time (next day).
Outside DST, the window is midnight to midnight civil time.
"""

from __future__ import annotations

from datetime import datetime, date, timedelta, time
from zoneinfo import ZoneInfo

UTC = ZoneInfo("UTC")


def get_observation_window_utc(target_date: date, timezone_name: str) -> tuple[datetime, datetime]:
    """Get the UTC bounds for a station's observation day.

    The NWS CLI observation window runs midnight-to-midnight in
    Local Standard Time (LST), regardless of whether DST is active.

    Args:
        target_date: The calendar date for the observation
        timezone_name: IANA timezone (e.g., 'America/New_York')

    Returns:
        (start_utc, end_utc) tuple of datetime objects in UTC
    """
    tz = ZoneInfo(timezone_name)

    # Determine the standard UTC offset (non-DST) for this timezone
    # We use January 1 of the year to get the standard offset
    jan1 = datetime(target_date.year, 1, 15, 12, tzinfo=tz)
    std_offset = jan1.utcoffset()

    # Observation window: midnight LST to midnight LST (next day)
    # In UTC: midnight_utc = midnight_local - std_offset
    start_lst = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0)
    end_lst = start_lst + timedelta(days=1)

    start_utc = (start_lst - std_offset).replace(tzinfo=UTC)
    end_utc = (end_lst - std_offset).replace(tzinfo=UTC)

    return start_utc, end_utc


def get_observation_window_civil(target_date: date, timezone_name: str) -> tuple[datetime, datetime]:
    """Get the civil (wall clock) time bounds for the observation window.

    Useful for understanding when the observation window starts/ends
    in terms of the local clock (which includes DST).

    During DST: window is 1:00 AM to 12:59 AM next day (civil time)
    Outside DST: window is 12:00 AM to 11:59 PM (civil time)
    """
    start_utc, end_utc = get_observation_window_utc(target_date, timezone_name)
    tz = ZoneInfo(timezone_name)
    return start_utc.astimezone(tz), end_utc.astimezone(tz)


def is_dst(dt: datetime, timezone_name: str) -> bool:
    """Check if DST is active at the given datetime."""
    tz = ZoneInfo(timezone_name)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    else:
        dt = dt.astimezone(tz)
    # Compare UTC offset to standard offset
    jan1 = datetime(dt.year, 1, 15, 12, tzinfo=tz)
    return dt.utcoffset() != jan1.utcoffset()


def filter_times_in_observation_window(
    times: list[datetime],
    target_date: date,
    timezone_name: str,
) -> list[int]:
    """Return indices of timestamps that fall within the observation window.

    Args:
        times: List of UTC-aware or naive-UTC datetime objects
        target_date: Calendar date for the observation
        timezone_name: Station timezone

    Returns:
        List of indices into `times` that are within the window
    """
    start_utc, end_utc = get_observation_window_utc(target_date, timezone_name)

    indices = []
    for i, t in enumerate(times):
        if t.tzinfo is None:
            t = t.replace(tzinfo=UTC)
        if start_utc <= t < end_utc:
            indices.append(i)
    return indices


def compute_lead_hours(model_run_time: datetime, target_date: date, timezone_name: str) -> float:
    """Compute lead time in hours from model run to target date noon local.

    Uses noon local time as the representative time for the target date
    (since daily highs typically occur in the afternoon).
    """
    tz = ZoneInfo(timezone_name)
    target_noon = datetime(target_date.year, target_date.month, target_date.day, 14, 0, tzinfo=tz)
    target_noon_utc = target_noon.astimezone(UTC)

    if model_run_time.tzinfo is None:
        model_run_time = model_run_time.replace(tzinfo=UTC)

    delta = target_noon_utc - model_run_time
    return delta.total_seconds() / 3600.0


def parse_iso_datetime(s: str) -> datetime:
    """Parse an ISO 8601 datetime string to a UTC datetime."""
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)
