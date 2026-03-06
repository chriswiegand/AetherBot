"""Forecast calibration using isotonic regression and Brier score tracking.

Isotonic regression maps raw model probabilities to calibrated probabilities
using historical forecast-vs-outcome pairs. This corrects systematic biases
like overconfidence or underconfidence in the ensemble.

Brier score decomposition provides separate metrics for:
- Reliability: how well calibrated are the probabilities?
- Resolution: how much do forecasts vary from climatology?
- Uncertainty: inherent unpredictability of the outcomes

This mirrors the precision vs. accuracy decomposition in analytical chemistry.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)

MIN_SAMPLES_FOR_CALIBRATION = 100


@dataclass
class BrierDecomposition:
    brier_score: float
    reliability: float
    resolution: float
    uncertainty: float
    n_samples: int


@dataclass
class ReliabilityBin:
    bin_center: float
    forecast_mean: float
    observed_freq: float
    count: int


class ForecastCalibrator:
    """Isotonic regression calibrator for forecast probabilities."""

    def __init__(self):
        self._model = IsotonicRegression(
            y_min=0.01, y_max=0.99, out_of_bounds="clip"
        )
        self._is_fitted = False
        self._n_samples = 0

    def fit(self, forecasts: list[float], outcomes: list[int]) -> None:
        """Train isotonic regression on historical (forecast, outcome) pairs.

        Args:
            forecasts: Model probabilities (0.0-1.0)
            outcomes: Binary outcomes (0 or 1)
        """
        if len(forecasts) < MIN_SAMPLES_FOR_CALIBRATION:
            logger.warning(
                f"Only {len(forecasts)} samples - need {MIN_SAMPLES_FOR_CALIBRATION} "
                f"for meaningful calibration. Passing through raw probabilities."
            )
            self._is_fitted = False
            return

        X = np.array(forecasts)
        y = np.array(outcomes)
        self._model.fit(X, y)
        self._is_fitted = True
        self._n_samples = len(forecasts)

        logger.info(f"Calibrator fitted on {self._n_samples} samples")

    def calibrate(self, raw_prob: float) -> float:
        """Apply calibration to a raw probability.

        If not enough historical data, returns raw probability.
        """
        if not self._is_fitted:
            return max(0.01, min(0.99, raw_prob))

        calibrated = float(self._model.predict([raw_prob])[0])
        return max(0.01, min(0.99, calibrated))

    def calibrate_batch(self, raw_probs: list[float]) -> list[float]:
        """Calibrate a batch of probabilities."""
        return [self.calibrate(p) for p in raw_probs]

    @staticmethod
    def compute_brier_score(forecasts: list[float], outcomes: list[int]) -> float:
        """Compute Brier score: mean((forecast - outcome)^2).

        Perfect = 0.0, climatology ~0.25, random = 0.5.
        """
        if not forecasts:
            return 1.0
        return sum((f - o) ** 2 for f, o in zip(forecasts, outcomes)) / len(forecasts)

    @staticmethod
    def compute_brier_decomposition(
        forecasts: list[float], outcomes: list[int], n_bins: int = 10
    ) -> BrierDecomposition:
        """Decompose Brier score into reliability, resolution, uncertainty.

        Reliability: lower is better (measures calibration)
        Resolution: higher is better (measures sharpness/discrimination)
        Uncertainty: fixed property of the sample
        Brier = Reliability - Resolution + Uncertainty
        """
        n = len(forecasts)
        if n == 0:
            return BrierDecomposition(1.0, 0.0, 0.0, 0.25, 0)

        f = np.array(forecasts)
        o = np.array(outcomes)
        base_rate = o.mean()
        uncertainty = base_rate * (1 - base_rate)

        # Bin forecasts
        bin_edges = np.linspace(0, 1, n_bins + 1)
        reliability = 0.0
        resolution = 0.0

        for i in range(n_bins):
            mask = (f >= bin_edges[i]) & (f < bin_edges[i + 1])
            if i == n_bins - 1:  # Last bin includes right edge
                mask = (f >= bin_edges[i]) & (f <= bin_edges[i + 1])

            n_k = mask.sum()
            if n_k == 0:
                continue

            f_k = f[mask].mean()  # Average forecast in bin
            o_k = o[mask].mean()  # Observed frequency in bin

            reliability += n_k * (f_k - o_k) ** 2
            resolution += n_k * (o_k - base_rate) ** 2

        reliability /= n
        resolution /= n

        brier = reliability - resolution + uncertainty

        return BrierDecomposition(
            brier_score=brier,
            reliability=reliability,
            resolution=resolution,
            uncertainty=uncertainty,
            n_samples=n,
        )

    @staticmethod
    def compute_reliability_diagram(
        forecasts: list[float], outcomes: list[int], n_bins: int = 10
    ) -> list[ReliabilityBin]:
        """Compute reliability diagram data.

        Perfect calibration: each bin's observed frequency equals
        its average forecast probability (points on the diagonal).
        """
        f = np.array(forecasts)
        o = np.array(outcomes)
        bin_edges = np.linspace(0, 1, n_bins + 1)

        bins = []
        for i in range(n_bins):
            mask = (f >= bin_edges[i]) & (f < bin_edges[i + 1])
            if i == n_bins - 1:
                mask = (f >= bin_edges[i]) & (f <= bin_edges[i + 1])

            n_k = mask.sum()
            if n_k == 0:
                continue

            bins.append(ReliabilityBin(
                bin_center=(bin_edges[i] + bin_edges[i + 1]) / 2,
                forecast_mean=float(f[mask].mean()),
                observed_freq=float(o[mask].mean()),
                count=int(n_k),
            ))

        return bins

    def save(self, path: str) -> None:
        """Save calibrator state to file."""
        state = {
            "is_fitted": self._is_fitted,
            "n_samples": self._n_samples,
        }
        if self._is_fitted:
            state["model"] = pickle.dumps(self._model)
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        """Load calibrator state from file."""
        if not Path(path).exists():
            logger.warning(f"Calibrator file not found: {path}")
            return
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._is_fitted = state["is_fitted"]
        self._n_samples = state["n_samples"]
        if self._is_fitted:
            self._model = pickle.loads(state["model"])
