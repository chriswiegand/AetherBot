"""HRRR-based correction of ensemble probabilities.

At short lead times (<12hr), HRRR's 3km resolution captures mesoscale
features (sea-breeze fronts, lake effects, mountain waves) that the
25km GFS ensemble misses. We shift the ensemble distribution toward
the HRRR forecast, weighted by lead time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.config.settings import ModelWeightsConfig

logger = logging.getLogger(__name__)


@dataclass
class HRRRCorrectionResult:
    original_mean: float
    hrrr_forecast: float
    shift: float
    correction_weight: float
    adjusted_member_maxes: list[float]


class HRRRCorrector:
    """Adjusts ensemble distribution using HRRR deterministic forecast."""

    def __init__(self, weights_config: ModelWeightsConfig):
        self.weights = weights_config

    def compute_shift(self, hrrr_max: float, ensemble_mean: float) -> float:
        """Compute the HRRR shift (difference from ensemble mean)."""
        return hrrr_max - ensemble_mean

    def get_correction_weight(self, lead_hours: float) -> float:
        """Get lead-time-dependent correction weight for HRRR.

        HRRR gets maximum weight at 0hr lead time and diminishes
        as lead time increases. Beyond ~48hr, HRRR is unavailable
        or unreliable.
        """
        return self.weights.get_hrrr_weight(lead_hours)

    def apply_correction(
        self,
        member_maxes: list[float],
        hrrr_max: float,
        ensemble_mean: float,
        lead_hours: float,
    ) -> HRRRCorrectionResult:
        """Apply HRRR correction to ensemble member maxes.

        Strategy: shift the entire ensemble distribution by a fraction
        of the HRRR-ensemble difference, weighted by lead time.

        This is analogous to a systematic error correction in analytical
        chemistry - the HRRR provides a "reference standard" measurement
        at short lead times.

        Args:
            member_maxes: Original 31 ensemble member daily maxes (F)
            hrrr_max: HRRR deterministic daily max forecast (F)
            ensemble_mean: Mean of ensemble member maxes
            lead_hours: Hours from model run to target date

        Returns:
            HRRRCorrectionResult with adjusted member maxes
        """
        shift = self.compute_shift(hrrr_max, ensemble_mean)
        weight = self.get_correction_weight(lead_hours)

        # Weighted shift: at 0hr lead, apply full HRRR correction
        # At 48hr lead, apply minimal correction
        effective_shift = shift * weight

        adjusted = [t + effective_shift for t in member_maxes]

        logger.debug(
            f"HRRR correction: shift={shift:.1f}F, weight={weight:.2f}, "
            f"effective_shift={effective_shift:.1f}F, lead={lead_hours:.0f}hr"
        )

        return HRRRCorrectionResult(
            original_mean=ensemble_mean,
            hrrr_forecast=hrrr_max,
            shift=shift,
            correction_weight=weight,
            adjusted_member_maxes=adjusted,
        )
