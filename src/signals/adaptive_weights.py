"""Adaptive model blend weights based on rolling settlement performance.

After each settlement, computes per-model Brier scores from the ModelScorecard
table and converts them to inverse-Brier weights. These adaptive weights
modulate (not replace) the fixed lead-time curves in ModelBlender, ensuring
HRRR still gets higher weight at short lead times (physics) while rewarding
whichever model is empirically performing best (data).

Design:
    final_weight = SMOOTHING_ALPHA * adaptive + (1 - SMOOTHING_ALPHA) * fixed

    With SMOOTHING_ALPHA=0.3, the adaptive component can shift weights by up
    to ±30% but never fully override the physics-based lead-time curves.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from src.config.settings import PROJECT_ROOT
from src.data.db import get_session
from src.data.models import ModelScorecard

logger = logging.getLogger(__name__)

WEIGHT_PATH = PROJECT_ROOT / "data" / "adaptive_weights.json"
MIN_SCORED_MARKETS = 10      # Per-model minimum before adapting
ROLLING_WINDOW_DAYS = 60     # Look-back window for Brier computation
SMOOTHING_ALPHA = 0.3        # 30% adaptive + 70% fixed curves
MODEL_SOURCES = ("ensemble", "ecmwf", "hrrr", "nws")


class AdaptiveWeightManager:
    """Computes and stores adaptive model blend weights from settlement data."""

    def __init__(self, weight_path: Path | None = None):
        self.weight_path = weight_path or WEIGHT_PATH
        self._current_weights: dict[str, float] | None = None

    def compute_model_brier_scores(
        self, session=None
    ) -> dict[str, float]:
        """Query ModelScorecard for average Brier per model source.

        Only considers the last ROLLING_WINDOW_DAYS of settled data.

        Returns:
            Dict mapping model_source → avg Brier contribution.
            Only includes sources with ≥ MIN_SCORED_MARKETS samples.
        """
        own_session = session is None
        if own_session:
            session = get_session()

        try:
            cutoff = (date.today() - timedelta(days=ROLLING_WINDOW_DAYS)).isoformat()

            scores: dict[str, list[float]] = {}
            counts: dict[str, int] = {}

            for source in MODEL_SOURCES:
                rows = (
                    session.query(ModelScorecard.brier_contribution)
                    .filter(
                        ModelScorecard.model_source == source,
                        ModelScorecard.target_date >= cutoff,
                        ModelScorecard.brier_contribution.isnot(None),
                    )
                    .all()
                )

                brier_values = [float(r[0]) for r in rows]
                counts[source] = len(brier_values)

                if len(brier_values) >= MIN_SCORED_MARKETS:
                    scores[source] = brier_values

            result = {}
            for source, values in scores.items():
                avg = sum(values) / len(values)
                result[source] = avg
                logger.info(
                    f"Adaptive weights: {source} avg Brier={avg:.4f} "
                    f"(n={counts[source]})"
                )

            return result

        finally:
            if own_session:
                session.close()

    @staticmethod
    def brier_to_weights(brier_scores: dict[str, float]) -> dict[str, float]:
        """Convert Brier scores to inverse-Brier weights.

        Lower Brier → higher weight. Normalized to sum to 1.0.

        If a model has Brier=0 (perfect), it gets a very high but
        finite weight (capped at 1/0.001).
        """
        if not brier_scores:
            return {}

        # Inverse-Brier: better models get more weight
        inv = {}
        for source, brier in brier_scores.items():
            # Floor at 0.001 to avoid division by zero
            inv[source] = 1.0 / max(brier, 0.001)

        total = sum(inv.values())
        if total == 0:
            return {}

        return {source: v / total for source, v in inv.items()}

    def compute_and_save(self, session=None) -> dict[str, float] | None:
        """Full pipeline: query DB → compute weights → save to JSON.

        Returns:
            Adaptive weights dict, or None if insufficient data.
        """
        brier_scores = self.compute_model_brier_scores(session)

        if len(brier_scores) < 2:
            logger.info(
                f"Adaptive weights: only {len(brier_scores)} model(s) with "
                f"sufficient data (need ≥2). Skipping."
            )
            return None

        weights = self.brier_to_weights(brier_scores)
        self._current_weights = weights

        now = datetime.now(timezone.utc).isoformat()

        # Load existing data (to preserve history)
        existing = self._load_raw()

        # Build counts dict
        own_session = session is None
        if own_session:
            session = get_session()
        try:
            cutoff = (date.today() - timedelta(days=ROLLING_WINDOW_DAYS)).isoformat()
            n_markets = {}
            for source in MODEL_SOURCES:
                count = (
                    session.query(ModelScorecard)
                    .filter(
                        ModelScorecard.model_source == source,
                        ModelScorecard.target_date >= cutoff,
                        ModelScorecard.brier_contribution.isnot(None),
                    )
                    .count()
                )
                n_markets[source] = count
        finally:
            if own_session:
                session.close()

        # Append to history
        history = existing.get("history", []) if existing else []
        history.append({"at": now, **weights})

        # Keep at most 730 entries (2 years of daily)
        if len(history) > 730:
            history = history[-730:]

        data = {
            "updated_at": now,
            "n_markets": n_markets,
            "brier_scores": {k: round(v, 6) for k, v in brier_scores.items()},
            "adaptive_weights": {k: round(v, 4) for k, v in weights.items()},
            "history": history,
        }

        # Save
        try:
            self.weight_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.weight_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(
                f"Adaptive weights saved: "
                + ", ".join(f"{k}={v:.3f}" for k, v in weights.items())
            )
        except Exception as e:
            logger.error(f"Failed to save adaptive weights: {e}")

        return weights

    def load(self) -> dict[str, float] | None:
        """Load last-saved adaptive weights from JSON.

        Returns:
            Adaptive weights dict, or None if file missing/invalid.
        """
        raw = self._load_raw()
        if raw is None:
            return None

        weights = raw.get("adaptive_weights")
        if weights:
            self._current_weights = weights
            logger.info(
                f"Loaded adaptive weights: "
                + ", ".join(f"{k}={v:.3f}" for k, v in weights.items())
            )
        return weights

    def _load_raw(self) -> dict | None:
        """Load raw JSON data from disk."""
        if not self.weight_path.exists():
            return None
        try:
            with open(self.weight_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load adaptive weights: {e}")
            return None

    @property
    def current_weights(self) -> dict[str, float] | None:
        """Currently active adaptive weights."""
        return self._current_weights

    @staticmethod
    def get_blended_weight(
        adaptive_weight: float,
        fixed_weight: float,
        alpha: float = SMOOTHING_ALPHA,
    ) -> float:
        """Blend adaptive weight with fixed-curve weight.

        Args:
            adaptive_weight: Performance-based weight (0-1)
            fixed_weight: Physics-based lead-time weight
            alpha: Smoothing factor (0=pure fixed, 1=pure adaptive)

        Returns:
            Blended weight
        """
        return alpha * adaptive_weight + (1 - alpha) * fixed_weight
