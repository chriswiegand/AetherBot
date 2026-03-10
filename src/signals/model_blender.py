"""Multi-model probability blender.

Combines probabilities from up to 4 global ensembles (GFS, ECMWF IFS,
ICON-EPS, GEM/GEPS) plus HRRR and NWS using inverse-variance-inspired
weighting with lead-time adjustments.

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
        self._adaptive_weights: dict[str, float] | None = None

    def set_adaptive_weights(self, weights: dict[str, float]) -> None:
        """Set performance-based adaptive weights.

        These modulate (not replace) the fixed lead-time curves.
        Pass None to disable adaptive weighting.
        """
        self._adaptive_weights = weights
        if weights:
            logger.info(
                "Adaptive weights active: "
                + ", ".join(f"{k}={v:.3f}" for k, v in weights.items())
            )

    def blend(
        self,
        ensemble_prob: float,
        hrrr_prob: float | None = None,
        nws_prob: float | None = None,
        lead_hours: float = 24.0,
        ecmwf_prob: float | None = None,
        icon_eps_prob: float | None = None,
        gem_prob: float | None = None,
    ) -> float:
        """Compute weighted blend of model probabilities.

        Args:
            ensemble_prob: GFS ensemble probability (always available)
            hrrr_prob: HRRR-derived probability (None if unavailable)
            nws_prob: NWS-derived probability (None if unavailable)
            lead_hours: Hours from now to target date afternoon
            ecmwf_prob: ECMWF IFS ensemble probability (None if unavailable)
            icon_eps_prob: ICON-EPS ensemble probability (None if unavailable)
            gem_prob: GEM/GEPS ensemble probability (None if unavailable)

        Returns:
            Blended probability (0.0 to 1.0)
        """
        # Lead-time-adaptive weights: GFS decays as HRRR becomes more reliable
        w_global = self.weights.get_gfs_weight(lead_hours)  # Total global-ensemble share
        w_hrrr = self.weights.get_hrrr_weight(lead_hours) if hrrr_prob is not None else 0.0
        w_nws = self.weights.nws if nws_prob is not None else 0.0

        # --- Split global-ensemble weight among available ensembles ---
        # Base shares (proportional to ensemble quality & resolution):
        #   GFS: 30  (31 members, hourly, best CONUS resolution)
        #   ECMWF: 30  (51 members, best global skill)
        #   ICON-EPS: 22  (40 members, independent physics)
        #   GEM: 18  (21 members, smallest ensemble)
        shares = {"gfs": 30.0}
        if ecmwf_prob is not None:
            shares["ecmwf"] = 30.0
        if icon_eps_prob is not None:
            shares["icon_eps"] = 22.0
        if gem_prob is not None:
            shares["gem"] = 18.0

        total_shares = sum(shares.values())
        w_ens = w_global * (shares["gfs"] / total_shares)
        w_ecmwf = w_global * (shares.get("ecmwf", 0) / total_shares)
        w_icon_eps = w_global * (shares.get("icon_eps", 0) / total_shares)
        w_gem = w_global * (shares.get("gem", 0) / total_shares)

        # Modulate with adaptive performance-based weights (30% adaptive, 70% fixed)
        if self._adaptive_weights:
            alpha = 0.3
            if "ensemble" in self._adaptive_weights:
                w_ens = alpha * self._adaptive_weights["ensemble"] + (1 - alpha) * w_ens
            if ecmwf_prob is not None and "ecmwf" in self._adaptive_weights:
                w_ecmwf = alpha * self._adaptive_weights["ecmwf"] + (1 - alpha) * w_ecmwf
            if icon_eps_prob is not None and "icon_eps" in self._adaptive_weights:
                w_icon_eps = alpha * self._adaptive_weights["icon_eps"] + (1 - alpha) * w_icon_eps
            if gem_prob is not None and "gem" in self._adaptive_weights:
                w_gem = alpha * self._adaptive_weights["gem"] + (1 - alpha) * w_gem
            if hrrr_prob is not None and "hrrr" in self._adaptive_weights:
                w_hrrr = alpha * self._adaptive_weights["hrrr"] + (1 - alpha) * w_hrrr
            if nws_prob is not None and "nws" in self._adaptive_weights:
                w_nws = alpha * self._adaptive_weights["nws"] + (1 - alpha) * w_nws

        # Build weighted sum
        total_weight = w_ens + w_ecmwf + w_icon_eps + w_gem + w_hrrr + w_nws
        if total_weight == 0:
            return ensemble_prob  # Fallback

        blended = w_ens * ensemble_prob
        if ecmwf_prob is not None:
            blended += w_ecmwf * ecmwf_prob
        if icon_eps_prob is not None:
            blended += w_icon_eps * icon_eps_prob
        if gem_prob is not None:
            blended += w_gem * gem_prob
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
