"""Temperature conversion and rounding utilities.

NWS CLI reports temperatures as whole-degree Fahrenheit integers.
The ASOS uses 2-minute averaged readings from a platinum RTD sensor.
The official daily max/min are integer F, NOT rounded through Celsius.
"""

from __future__ import annotations

import math


def celsius_to_fahrenheit(temp_c: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return temp_c * 9.0 / 5.0 + 32.0


def fahrenheit_to_celsius(temp_f: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return (temp_f - 32.0) * 5.0 / 9.0


def round_to_integer_f(temp_f: float) -> int:
    """Round temperature to nearest integer Fahrenheit.

    This matches NWS CLI rounding behavior. The official high/low
    is recorded as a whole-degree Fahrenheit integer derived from
    2-minute averages of the RTD sensor.

    Uses banker's rounding (round half to even) to match standard
    meteorological practice.
    """
    return round(temp_f)


def settles_above(observed_high: int, threshold: float) -> bool:
    """Determine if an 'above X' contract settles YES.

    Kalshi uses 'strictly greater than': if threshold is 50,
    observed must be 51+ (since observations are integers).
    """
    return observed_high > threshold


def settles_in_bracket(observed_high: int, bracket_low: float | None, bracket_high: float | None) -> bool:
    """Determine if a bracket contract settles YES.

    bracket_low=None means open-ended low (e.g., '48 or below').
    bracket_high=None means open-ended high (e.g., '91 or above').
    For middle brackets like '83-84', both are set.
    """
    if bracket_low is not None and observed_high < bracket_low:
        return False
    if bracket_high is not None and observed_high > bracket_high:
        return False
    return True


def ensemble_daily_max_to_integer(member_maxes: list[float]) -> list[int]:
    """Round all ensemble member daily maxes to integer F.

    This preprocessing step aligns ensemble forecasts with
    the integer rounding that occurs in official NWS reporting.
    """
    return [round_to_integer_f(t) for t in member_maxes]
