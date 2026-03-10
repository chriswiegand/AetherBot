"""Model scorer: grades every model source against settlement outcomes.

After settlement, scores each model (ensemble, HRRR, NWS, blended,
calibrated, market) on every market — recording final probability,
Brier contribution, convergence trajectory, and ranking.

This is the "report card" system: which model was closest to the truth?
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.data.db import get_session
from src.data.models import Signal, KalshiMarket, ModelScorecard
from src.utils.temperature import settles_above, settles_in_bracket

logger = logging.getLogger(__name__)

# Maps model_source name -> Signal column name
PROB_COLUMN_MAP = {
    "ensemble": "ensemble_prob",
    "hrrr": "hrrr_prob",
    "nws": "nws_prob",
    "blended": "blended_prob",
    "calibrated": "calibrated_prob",
    "market": "market_yes_price",
}


class ModelScorer:
    """Scores all model sources against settlement outcomes."""

    def score_settlement(
        self,
        city: str,
        target_date: str,
        observed_high: float,
        session=None,
    ) -> list[ModelScorecard]:
        """Score all models for all markets in a city on a target date.

        For each market_ticker with signals:
        1. Determine binary outcome (YES/NO)
        2. Extract probability trajectory for each model source
        3. Compute Brier, convergence metrics, ranking
        4. INSERT into model_scorecards table

        Args:
            city: City name (e.g., "NYC")
            target_date: Settlement date "YYYY-MM-DD"
            observed_high: Actual observed high temperature (F)
            session: Optional SQLAlchemy session (creates one if None)

        Returns:
            List of created ModelScorecard records
        """
        own_session = session is None
        if own_session:
            session = get_session()

        created = []
        now = datetime.now(timezone.utc).isoformat()

        try:
            # Find all market tickers with signals for this city+date
            signal_tickers = (
                session.query(Signal.market_ticker)
                .filter_by(city=city, target_date=target_date)
                .distinct()
                .all()
            )
            tickers = [row[0] for row in signal_tickers]

            if not tickers:
                logger.debug(f"No signals to score for {city} {target_date}")
                return []

            for ticker in tickers:
                # Get the market to determine outcome
                market = (
                    session.query(KalshiMarket)
                    .filter_by(market_ticker=ticker)
                    .first()
                )
                if market is None:
                    logger.warning(f"Market not found for scoring: {ticker}")
                    continue

                # Determine binary outcome
                if market.is_above_contract and market.threshold_f is not None:
                    settled_yes = settles_above(observed_high, market.threshold_f)
                else:
                    settled_yes = settles_in_bracket(
                        observed_high, market.bracket_low, market.bracket_high
                    )
                outcome = 1 if settled_yes else 0

                # Get all signals for this ticker, ordered by time
                signals = (
                    session.query(Signal)
                    .filter_by(city=city, target_date=target_date, market_ticker=ticker)
                    .order_by(Signal.computed_at)
                    .all()
                )

                if not signals:
                    continue

                # Score each model source
                scorecards = self._score_all_models(
                    signals, ticker, city, target_date,
                    observed_high, outcome, now,
                )

                # Rank: find best and worst by distance_from_outcome
                if scorecards:
                    distances = [
                        (sc, sc.distance_from_outcome)
                        for sc in scorecards
                        if sc.distance_from_outcome is not None
                    ]
                    if distances:
                        best_dist = min(d for _, d in distances)
                        worst_dist = max(d for _, d in distances)
                        for sc, dist in distances:
                            if dist == best_dist:
                                sc.was_best_model = 1
                            if dist == worst_dist:
                                sc.was_worst_model = 1

                # Upsert: check for existing records first
                for sc in scorecards:
                    existing = (
                        session.query(ModelScorecard)
                        .filter_by(
                            city=sc.city,
                            target_date=sc.target_date,
                            market_ticker=sc.market_ticker,
                            model_source=sc.model_source,
                        )
                        .first()
                    )
                    if existing:
                        # Update existing record
                        for attr in [
                            "final_prob", "outcome", "observed_high_f",
                            "brier_contribution", "first_prob", "prob_at_24h",
                            "prob_at_12h", "prob_at_6h", "max_prob_swing",
                            "final_lead_hours", "distance_from_outcome",
                            "was_best_model", "was_worst_model", "created_at",
                        ]:
                            setattr(existing, attr, getattr(sc, attr))
                    else:
                        session.add(sc)
                    created.append(sc)

            if own_session:
                session.commit()

        except Exception as e:
            logger.error(f"Model scoring failed for {city} {target_date}: {e}")
            if own_session:
                session.rollback()
            raise
        finally:
            if own_session:
                session.close()

        logger.info(
            f"Scored {len(created)} model×market entries for {city} {target_date} "
            f"(observed={observed_high}°F)"
        )
        return created

    def _score_all_models(
        self,
        signals: list[Signal],
        ticker: str,
        city: str,
        target_date: str,
        observed_high: float,
        outcome: int,
        now: str,
    ) -> list[ModelScorecard]:
        """Score each of the 6 model sources from the signal trajectory."""
        scorecards = []

        for model_source, col_name in PROB_COLUMN_MAP.items():
            # Extract probability trajectory for this model
            trajectory = []
            for s in signals:
                prob = getattr(s, col_name, None)
                if prob is not None:
                    lead = float(s.lead_hours) if s.lead_hours else None
                    trajectory.append((s.computed_at, prob, lead))

            if not trajectory:
                continue

            # Final probability (most recent signal)
            final_prob = trajectory[-1][1]
            final_lead = trajectory[-1][2]

            # First probability (earliest signal)
            first_prob = trajectory[0][1]

            # Brier contribution
            brier = (final_prob - outcome) ** 2
            distance = abs(final_prob - outcome)

            # Probabilities at specific lead times
            prob_at_24h = self._prob_at_lead(trajectory, 24.0)
            prob_at_12h = self._prob_at_lead(trajectory, 12.0)
            prob_at_6h = self._prob_at_lead(trajectory, 6.0)

            # Max probability swing (largest consecutive change)
            max_swing = 0.0
            for i in range(1, len(trajectory)):
                swing = abs(trajectory[i][1] - trajectory[i - 1][1])
                max_swing = max(max_swing, swing)

            scorecards.append(ModelScorecard(
                city=city,
                target_date=target_date,
                market_ticker=ticker,
                model_source=model_source,
                final_prob=final_prob,
                outcome=outcome,
                observed_high_f=observed_high,
                brier_contribution=brier,
                first_prob=first_prob,
                prob_at_24h=prob_at_24h,
                prob_at_12h=prob_at_12h,
                prob_at_6h=prob_at_6h,
                max_prob_swing=max_swing,
                final_lead_hours=final_lead,
                distance_from_outcome=distance,
                was_best_model=0,
                was_worst_model=0,
                created_at=now,
            ))

        return scorecards

    @staticmethod
    def _prob_at_lead(
        trajectory: list[tuple[str, float, float | None]],
        target_lead: float,
    ) -> float | None:
        """Find the probability closest to a specific lead time.

        Finds the signal with lead_hours closest to target_lead.
        Returns None if no signals have lead_hours data.
        """
        best = None
        best_diff = float("inf")

        for _, prob, lead in trajectory:
            if lead is None:
                continue
            diff = abs(lead - target_lead)
            if diff < best_diff:
                best_diff = diff
                best = prob

        # Only return if we found something within 3 hours of target
        if best is not None and best_diff <= 3.0:
            return best
        return None
