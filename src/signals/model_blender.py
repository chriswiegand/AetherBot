"""Multi-model probability blender.

Combines probabilities from GFS ensemble, HRRR, and NWS using
inverse-variance-inspired weighting with lead-time adjustments.

This is analogous to combining analytical measurements from
different instruments - more precise instruments (HRRR at short
lead times) get higher weight.
"""

from __future__ import annotations

import logging
import math

from src.config.settings import ModelWeightsConfig

logger = logging.getLogger(__name__)


class ModelBlender:
    """Blends probabilities from multiple weather model sources."""

    def __init__(self, weights_config: ModelWeightsConfig):
        self.weights = weights_config

    def blend(
        self,
        ensemble_prob: float,
        hrrr_prob: float | None = None,
        nws_prob: float | None = None,
        lead_hours: float = 24.0,
    ) -> float:
        """Compute weighted blend of model probabilities.

        Args:
            ensemble_prob: GFS ensemble probability (always available)
            hrrr_prob: HRRR-derived probability (None if unavailable)
            nws_prob: NWS-derived probability (None if unavailable)
            lead_hours: Hours from now to target date afternoon

        Returns:
            Blended probability (0.0 to 1.0)
        """
        # Start with base weights from config
        w_ens = self.weights.gfs_ensemble
        w_hrrr = self.weights.get_hrrr_weight(lead_hours) if hrrr_prob is not None else 0.0
        w_nws = self.weights.nws if nws_prob is not None else 0.0

        # Build weighted sum
        total_weight = w_ens + w_hrrr + w_nws
        if total_weight == 0:
            return ensemble_prob  # Fallback

        blended = w_ens * ensemble_prob
        if hrrr_prob is not None:
            blended += w_hrrr * hrrr_prob
        if nws_prob is not None:
            blended += w_nws * nws_prob

        blended /= total_weight

        # Clamp to valid probability range
        return max(0.01, min(0.99, blended))

    def prob_from_deterministic(
        self,
        forecast_temp: float,
        threshold: float,
        historical_std: float = 3.0,
        is_above: bool = True,
    ) -> float:
        """Convert a deterministic forecast to a probability.

        Uses a Gaussian error model: the actual temperature is
        assumed to be N(forecast, std^2). The probability of
        exceeding the threshold is computed from the CDF.

        Args:
            forecast_temp: Deterministic forecast (F)
            threshold: Temperature threshold (F)
            historical_std: Historical forecast error std dev (F)
            is_above: If True, compute P(actual > threshold)

        Returns:
            Probability (0.0 to 1.0)
        """
        if historical_std <= 0:
            # Deterministic: either 0 or 1
            if is_above:
                return 1.0 if forecast_temp > threshold else 0.0
            else:
                return 1.0 if forecast_temp <= threshold else 0.0

        z = (threshold - forecast_temp) / historical_std

        # Standard normal CDF approximation
        prob_below = _norm_cdf(z)

        if is_above:
            return 1.0 - prob_below
        return prob_below

    def prob_from_deterministic_bracket(
        self,
        forecast_temp: float,
        bracket_low: float | None,
        bracket_high: float | None,
        historical_std: float = 3.0,
    ) -> float:
        """Convert deterministic forecast to bracket probability."""
        if bracket_low is None and bracket_high is None:
            return 1.0

        if bracket_low is None:
            # Open-ended low: P(actual <= bracket_high)
            z = (bracket_high + 0.5 - forecast_temp) / historical_std
            return _norm_cdf(z)

        if bracket_high is None:
            # Open-ended high: P(actual >= bracket_low)
            z = (bracket_low - 0.5 - forecast_temp) / historical_std
            return 1.0 - _norm_cdf(z)

        # Closed bracket: P(bracket_low <= actual <= bracket_high)
        z_low = (bracket_low - 0.5 - forecast_temp) / historical_std
        z_high = (bracket_high + 0.5 - forecast_temp) / historical_std
        return _norm_cdf(z_high) - _norm_cdf(z_low)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
