"""Running Brier score tracker segmented by city, lead time, and model source."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

from src.data.db import get_session
from src.data.models import BrierScore

logger = logging.getLogger(__name__)


@dataclass
class BrierSummary:
    city: str
    brier_score: float
    n_samples: int
    period_days: int


class BrierTracker:
    """Tracks running Brier scores across multiple dimensions."""

    def get_brier_by_city(self, days: int = 30) -> list[BrierSummary]:
        """Get Brier scores per city over the last N days."""
        session = get_session()
        try:
            cutoff = (date.today() - timedelta(days=days)).isoformat()
            scores = (
                session.query(BrierScore)
                .filter(BrierScore.target_date >= cutoff)
                .all()
            )

            by_city: dict[str, list[float]] = {}
            for s in scores:
                by_city.setdefault(s.city, []).append(s.brier_contribution)

            results = []
            for city, contribs in sorted(by_city.items()):
                brier = sum(contribs) / len(contribs) if contribs else 1.0
                results.append(BrierSummary(city, brier, len(contribs), days))

            return results
        finally:
            session.close()

    def get_overall_brier(self, days: int = 30) -> float:
        """Get overall Brier score over the last N days."""
        summaries = self.get_brier_by_city(days)
        if not summaries:
            return 1.0
        total = sum(s.brier_score * s.n_samples for s in summaries)
        n = sum(s.n_samples for s in summaries)
        return total / n if n > 0 else 1.0

    def is_calibration_degrading(self, threshold: float = 0.03) -> bool:
        """Check if recent calibration is worse than longer-term average."""
        recent = self.get_overall_brier(days=7)
        baseline = self.get_overall_brier(days=30)
        return recent > baseline + threshold
