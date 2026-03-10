"""Calibrator trainer: fits isotonic regression from settled data.

The ForecastCalibrator in src/signals/calibration.py is NEVER fitted in
live mode — it just clips raw probabilities to [0.01, 0.99]. This module
collects (blended_prob, outcome) pairs from the ModelScorecard table and
periodically retrains the calibrator.

Training protocol:
1. Collect all (blended_prob, outcome) pairs from scored settlements
2. 80/20 holdout split (chronologically, not random)
3. Fit isotonic regression on training set
4. Validate on holdout — only deploy if Brier improves
5. Save to data/calibrator.pkl
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from src.config.settings import PROJECT_ROOT
from src.data.db import get_session
from src.data.models import ModelScorecard
from src.signals.calibration import ForecastCalibrator

logger = logging.getLogger(__name__)

CALIBRATOR_PATH = PROJECT_ROOT / "data" / "calibrator.pkl"
MIN_SAMPLES = 20
RETRAIN_INTERVAL_DAYS = 1


class CalibratorTrainer:
    """Manages calibrator training lifecycle from settled data."""

    def __init__(self, calibrator: ForecastCalibrator | None = None):
        self.calibrator = calibrator or ForecastCalibrator()
        self._last_retrain_at: str | None = None
        self._last_n_samples: int = 0

    def collect_training_data(
        self, session=None
    ) -> tuple[list[float], list[int]]:
        """Gather (blended_prob, outcome) pairs from ModelScorecard table.

        Uses the 'blended' model source as the input to the calibrator,
        since that's what the live pipeline feeds into calibrate().

        Returns:
            (forecasts, outcomes) — parallel lists, chronologically ordered
        """
        own_session = session is None
        if own_session:
            session = get_session()

        try:
            rows = (
                session.query(
                    ModelScorecard.final_prob,
                    ModelScorecard.outcome,
                    ModelScorecard.target_date,
                )
                .filter_by(model_source="blended")
                .filter(ModelScorecard.final_prob.isnot(None))
                .filter(ModelScorecard.outcome.isnot(None))
                .order_by(ModelScorecard.target_date)
                .all()
            )

            forecasts = [float(r[0]) for r in rows]
            outcomes = [int(r[1]) for r in rows]
            return forecasts, outcomes

        finally:
            if own_session:
                session.close()

    def should_retrain(self, session=None) -> bool:
        """Check if calibrator should be retrained.

        Conditions:
        - At least MIN_SAMPLES settled markets
        - At least RETRAIN_INTERVAL_DAYS since last retrain
        - Either: not fitted, or new data since last retrain
        """
        forecasts, outcomes = self.collect_training_data(session)
        n = len(forecasts)

        if n < MIN_SAMPLES:
            logger.debug(
                f"Calibrator: only {n} samples, need {MIN_SAMPLES} for training"
            )
            return False

        # Check if we have new data since last retrain
        if n == self._last_n_samples and self.calibrator._is_fitted:
            return False

        # Check time since last retrain
        if self._last_retrain_at:
            from datetime import timedelta
            last = datetime.fromisoformat(self._last_retrain_at)
            elapsed = (datetime.now(timezone.utc) - last).days
            if elapsed < RETRAIN_INTERVAL_DAYS:
                logger.debug(
                    f"Calibrator: only {elapsed} days since last retrain, "
                    f"need {RETRAIN_INTERVAL_DAYS}"
                )
                return False

        return True

    def retrain(self, session=None) -> bool:
        """Retrain calibrator with validation.

        Protocol:
        1. Collect all data
        2. Split 80/20 chronologically
        3. Fit on training set
        4. Compute Brier on holdout
        5. Only deploy if new Brier < old Brier (or no old model)

        Returns:
            True if a new calibrator was deployed, False otherwise
        """
        forecasts, outcomes = self.collect_training_data(session)
        n = len(forecasts)

        if n < MIN_SAMPLES:
            logger.warning(f"Cannot retrain: only {n} samples (need {MIN_SAMPLES})")
            return False

        # Chronological split: 80% train, 20% holdout
        split_idx = int(n * 0.8)
        train_f, train_o = forecasts[:split_idx], outcomes[:split_idx]
        hold_f, hold_o = forecasts[split_idx:], outcomes[split_idx:]

        if len(hold_f) < 4:
            logger.warning(f"Holdout set too small ({len(hold_f)}), skipping retrain")
            return False

        # Compute baseline Brier on holdout (raw blended, no calibration)
        baseline_brier = ForecastCalibrator.compute_brier_score(hold_f, hold_o)

        # Fit new calibrator
        new_calibrator = ForecastCalibrator()
        new_calibrator.fit(train_f, train_o)

        if not new_calibrator._is_fitted:
            logger.warning("New calibrator failed to fit")
            return False

        # Compute new Brier on holdout
        calibrated_holdout = new_calibrator.calibrate_batch(hold_f)
        new_brier = ForecastCalibrator.compute_brier_score(calibrated_holdout, hold_o)

        logger.info(
            f"Calibrator retrain: baseline Brier={baseline_brier:.4f}, "
            f"calibrated Brier={new_brier:.4f} "
            f"(train={len(train_f)}, holdout={len(hold_f)})"
        )

        # Only deploy if it improves
        if new_brier >= baseline_brier:
            logger.warning(
                f"New calibrator is NOT better (new={new_brier:.4f} >= "
                f"baseline={baseline_brier:.4f}). Keeping raw probabilities."
            )
            return False

        improvement = baseline_brier - new_brier
        logger.info(
            f"Deploying new calibrator: Brier improved by {improvement:.4f} "
            f"({improvement / baseline_brier * 100:.1f}%)"
        )

        # Deploy
        self.calibrator = new_calibrator
        self._last_retrain_at = datetime.now(timezone.utc).isoformat()
        self._last_n_samples = n

        # Save to disk
        try:
            self.calibrator.save(str(CALIBRATOR_PATH))
            logger.info(f"Calibrator saved to {CALIBRATOR_PATH}")
        except Exception as e:
            logger.error(f"Failed to save calibrator: {e}")

        return True

    def load_if_exists(self) -> bool:
        """Load a previously saved calibrator from disk.

        Returns:
            True if loaded successfully, False otherwise
        """
        if not CALIBRATOR_PATH.exists():
            logger.info("No saved calibrator found — using raw probabilities")
            return False

        try:
            self.calibrator.load(str(CALIBRATOR_PATH))
            logger.info(
                f"Loaded calibrator from {CALIBRATOR_PATH} "
                f"(fitted={self.calibrator._is_fitted}, "
                f"n_samples={self.calibrator._n_samples})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load calibrator: {e}")
            return False
