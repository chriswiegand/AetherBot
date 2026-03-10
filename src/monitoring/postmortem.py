"""Automated postmortem generator for failed and succeeded forecasts.

Produces a structured report for each settlement answering:
- Which model was closest? Which was worst?
- At what lead time did forecasts go wrong?
- Root cause classification
- Actionable recommendations

Reports saved as JSON: data/archives/postmortem/{target_date}.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from src.config.settings import ArchivalConfig, PROJECT_ROOT
from src.data.db import get_session
from src.data.models import ModelScorecard, Signal

logger = logging.getLogger(__name__)


@dataclass
class ModelReport:
    """Report for a single model source on a single market."""
    model_source: str
    final_prob: float | None
    brier_contribution: float | None
    distance_from_outcome: float | None
    was_best: bool = False
    was_worst: bool = False
    trajectory_summary: str = ""


@dataclass
class MarketPostmortem:
    """Postmortem for a single market."""
    market_ticker: str
    outcome: int  # 0 or 1
    observed_high_f: float
    best_model: str | None = None
    worst_model: str | None = None
    best_brier: float | None = None
    worst_brier: float | None = None
    root_cause: str | None = None
    root_cause_detail: str = ""
    recommendations: list[str] = field(default_factory=list)
    model_reports: list[ModelReport] = field(default_factory=list)
    crossover_lead_hours: float | None = None  # When forecast crossed 50%


@dataclass
class PostmortemReport:
    """Full postmortem for a city+date."""
    city: str
    target_date: str
    observed_high_f: float
    generated_at: str
    markets: list[MarketPostmortem] = field(default_factory=list)
    summary: str = ""
    overall_best_model: str | None = None
    overall_worst_model: str | None = None


# Root cause classifications
ROOT_CAUSES = {
    "bad_gfs_run": "GFS ensemble was tight and wrong — bad model initialization",
    "hrrr_init_error": "HRRR diverged >5°F from observations at short lead time",
    "nws_bias": "NWS high forecast was >4°F from observed",
    "market_mispricing": "Market was >15% from all model probabilities",
    "calibration_drift": "Calibrated probability was worse than raw blended",
    "black_swan": "ALL models wrong — observed high was extreme outlier",
    "all_correct": "All models within acceptable range",
}


class PostmortemGenerator:
    """Generates automated postmortem reports after settlement."""

    def generate(
        self,
        city: str,
        target_date: str,
        observed_high: float,
        archival_config: ArchivalConfig | None = None,
        session=None,
    ) -> PostmortemReport:
        """Generate a postmortem report for a city+date settlement.

        Args:
            city: City name
            target_date: Settlement date
            observed_high: Actual observed high (F)
            archival_config: Config for saving JSON report
            session: Optional SQLAlchemy session

        Returns:
            PostmortemReport with analysis
        """
        own_session = session is None
        if own_session:
            session = get_session()

        now = datetime.now(timezone.utc).isoformat()
        report = PostmortemReport(
            city=city,
            target_date=target_date,
            observed_high_f=observed_high,
            generated_at=now,
        )

        try:
            # Get all scorecards for this city+date
            scorecards = (
                session.query(ModelScorecard)
                .filter_by(city=city, target_date=target_date)
                .all()
            )

            if not scorecards:
                report.summary = "No scorecards available — scoring may not have run yet."
                return report

            # Group by market_ticker
            by_market: dict[str, list[ModelScorecard]] = {}
            for sc in scorecards:
                by_market.setdefault(sc.market_ticker, []).append(sc)

            # Track model wins/losses across all markets
            model_wins: dict[str, int] = {}
            model_brier_sums: dict[str, float] = {}
            model_counts: dict[str, int] = {}

            for ticker, market_scorecards in by_market.items():
                mkt_pm = self._analyze_market(
                    ticker, market_scorecards, observed_high, session
                )
                report.markets.append(mkt_pm)

                # Aggregate model performance
                for mr in mkt_pm.model_reports:
                    src = mr.model_source
                    if mr.brier_contribution is not None:
                        model_brier_sums[src] = (
                            model_brier_sums.get(src, 0) + mr.brier_contribution
                        )
                        model_counts[src] = model_counts.get(src, 0) + 1
                    if mr.was_best:
                        model_wins[src] = model_wins.get(src, 0) + 1

            # Overall best/worst model
            avg_briers = {
                src: model_brier_sums[src] / model_counts[src]
                for src in model_brier_sums
                if model_counts.get(src, 0) > 0
            }
            if avg_briers:
                report.overall_best_model = min(avg_briers, key=avg_briers.get)
                report.overall_worst_model = max(avg_briers, key=avg_briers.get)

            # Build summary
            n_markets = len(report.markets)
            root_causes = [m.root_cause for m in report.markets if m.root_cause]
            report.summary = (
                f"{city} {target_date}: observed {observed_high}°F. "
                f"Scored {n_markets} markets. "
                f"Best model: {report.overall_best_model or 'N/A'} "
                f"(avg Brier={avg_briers.get(report.overall_best_model, 0):.4f}). "
                f"Root causes: {', '.join(set(root_causes)) or 'none identified'}."
            )

        except Exception as e:
            logger.error(f"Postmortem generation failed for {city} {target_date}: {e}")
            report.summary = f"Error generating postmortem: {e}"
        finally:
            if own_session:
                session.close()

        # Save to JSON
        if archival_config and archival_config.enabled:
            self._save_report(report, archival_config)

        logger.info(f"Postmortem: {report.summary}")
        return report

    def _analyze_market(
        self,
        ticker: str,
        scorecards: list[ModelScorecard],
        observed_high: float,
        session,
    ) -> MarketPostmortem:
        """Analyze a single market's model performance."""
        outcome = scorecards[0].outcome if scorecards else 0
        pm = MarketPostmortem(
            market_ticker=ticker,
            outcome=outcome,
            observed_high_f=observed_high,
        )

        # Build model reports
        for sc in scorecards:
            mr = ModelReport(
                model_source=sc.model_source,
                final_prob=sc.final_prob,
                brier_contribution=sc.brier_contribution,
                distance_from_outcome=sc.distance_from_outcome,
                was_best=bool(sc.was_best_model),
                was_worst=bool(sc.was_worst_model),
            )

            # Trajectory summary
            parts = []
            if sc.first_prob is not None:
                parts.append(f"start={sc.first_prob:.2f}")
            if sc.prob_at_24h is not None:
                parts.append(f"24h={sc.prob_at_24h:.2f}")
            if sc.prob_at_12h is not None:
                parts.append(f"12h={sc.prob_at_12h:.2f}")
            if sc.prob_at_6h is not None:
                parts.append(f"6h={sc.prob_at_6h:.2f}")
            if sc.final_prob is not None:
                parts.append(f"final={sc.final_prob:.2f}")
            mr.trajectory_summary = " → ".join(parts)

            pm.model_reports.append(mr)

            if mr.was_best:
                pm.best_model = sc.model_source
                pm.best_brier = sc.brier_contribution
            if mr.was_worst:
                pm.worst_model = sc.model_source
                pm.worst_brier = sc.brier_contribution

        # Root cause classification
        pm.root_cause, pm.root_cause_detail = self._classify_root_cause(
            scorecards, observed_high
        )

        # Recommendations
        pm.recommendations = self._generate_recommendations(scorecards, pm.root_cause)

        # Find crossover lead time (when forecast crossed 50%)
        pm.crossover_lead_hours = self._find_crossover(
            scorecards, outcome, session, ticker
        )

        return pm

    def _classify_root_cause(
        self,
        scorecards: list[ModelScorecard],
        observed_high: float,
    ) -> tuple[str, str]:
        """Classify the root cause of forecast error."""
        sc_by_model = {sc.model_source: sc for sc in scorecards}

        # Check: ALL models wrong → black swan
        all_distances = [
            sc.distance_from_outcome
            for sc in scorecards
            if sc.distance_from_outcome is not None
        ]
        if all_distances and min(all_distances) > 0.40:
            return "black_swan", (
                f"All models had distance >0.40 from outcome. "
                f"Observed {observed_high}°F was likely an extreme outlier."
            )

        # Check: NWS bias
        nws = sc_by_model.get("nws")
        if nws and nws.distance_from_outcome and nws.distance_from_outcome > 0.40:
            return "nws_bias", (
                f"NWS probability was {nws.final_prob:.2f} vs outcome {nws.outcome}. "
                f"NWS high forecast likely missed by >4°F."
            )

        # Check: HRRR init error (HRRR worst at short lead and diverged)
        hrrr = sc_by_model.get("hrrr")
        if hrrr and hrrr.was_worst_model and hrrr.prob_at_6h is not None:
            if hrrr.distance_from_outcome and hrrr.distance_from_outcome > 0.35:
                return "hrrr_init_error", (
                    f"HRRR was worst model (distance={hrrr.distance_from_outcome:.2f}). "
                    f"6h prob={hrrr.prob_at_6h:.2f} suggests initialization error."
                )

        # Check: Bad GFS run (ensemble was worst and tight)
        ens = sc_by_model.get("ensemble")
        if ens and ens.was_worst_model:
            if ens.max_prob_swing is not None and ens.max_prob_swing < 0.05:
                return "bad_gfs_run", (
                    f"Ensemble was worst model with low swing ({ens.max_prob_swing:.3f}). "
                    f"Ensemble was confidently wrong — likely bad model initialization."
                )

        # Check: Market mispricing
        market = sc_by_model.get("market")
        model_probs = [
            sc.final_prob for sc in scorecards
            if sc.model_source != "market" and sc.final_prob is not None
        ]
        if market and market.final_prob is not None and model_probs:
            avg_model = sum(model_probs) / len(model_probs)
            if abs(market.final_prob - avg_model) > 0.15:
                return "market_mispricing", (
                    f"Market price {market.final_prob:.2f} diverged from "
                    f"model avg {avg_model:.2f} by {abs(market.final_prob - avg_model):.2f}."
                )

        # Check: Calibration drift
        blended = sc_by_model.get("blended")
        calibrated = sc_by_model.get("calibrated")
        if (blended and calibrated
            and blended.brier_contribution is not None
            and calibrated.brier_contribution is not None
            and calibrated.brier_contribution > blended.brier_contribution + 0.01):
            return "calibration_drift", (
                f"Calibrated Brier ({calibrated.brier_contribution:.4f}) was worse than "
                f"blended ({blended.brier_contribution:.4f}). Calibrator may be stale."
            )

        # All models reasonably close
        if all_distances and max(all_distances) < 0.25:
            return "all_correct", "All models within acceptable range."

        return "unclassified", "No clear root cause identified."

    def _generate_recommendations(
        self,
        scorecards: list[ModelScorecard],
        root_cause: str | None,
    ) -> list[str]:
        """Generate actionable recommendations based on root cause."""
        recs = []

        if root_cause == "bad_gfs_run":
            recs.append(
                "Consider adding ensemble spread as a confidence filter — "
                "skip trades when spread is unusually tight."
            )
        elif root_cause == "hrrr_init_error":
            recs.append(
                "HRRR added no value for this market. Consider reducing "
                "HRRR weight when it diverges strongly from ensemble."
            )
        elif root_cause == "nws_bias":
            recs.append(
                "NWS had a significant bias. Monitor NWS reliability "
                "per city — some stations have persistent biases."
            )
        elif root_cause == "market_mispricing":
            recs.append(
                "Market was significantly mispriced vs all models. "
                "This suggests a real edge opportunity was present."
            )
        elif root_cause == "calibration_drift":
            recs.append(
                "Calibrator is performing worse than raw blended. "
                "Consider retraining with recent data or disabling."
            )
        elif root_cause == "black_swan":
            recs.append(
                "All models failed — extreme weather event. "
                "Review if climatological bounds need widening."
            )

        # General recommendation: check if any model consistently wins
        best_models = [
            sc.model_source for sc in scorecards if sc.was_best_model
        ]
        if best_models:
            recs.append(
                f"Best model for this market: {best_models[0]}. "
                f"Track if this pattern persists across similar markets."
            )

        return recs

    def _find_crossover(
        self,
        scorecards: list[ModelScorecard],
        outcome: int,
        session,
        ticker: str,
    ) -> float | None:
        """Find the lead time where the blended forecast crossed 50%.

        Walking backward from the final signal, find where the model
        first crossed from the correct side to the wrong side of 50%.
        """
        try:
            signals = (
                session.query(Signal)
                .filter_by(market_ticker=ticker)
                .order_by(Signal.computed_at.desc())
                .all()
            )

            for s in signals:
                if s.blended_prob is None or s.lead_hours is None:
                    continue
                # Was the forecast on the correct side?
                correct_side = (
                    (s.blended_prob >= 0.5 and outcome == 1) or
                    (s.blended_prob < 0.5 and outcome == 0)
                )
                if not correct_side:
                    return float(s.lead_hours)

        except Exception:
            pass

        return None

    def _save_report(
        self, report: PostmortemReport, config: ArchivalConfig
    ) -> None:
        """Save postmortem report as JSON."""
        try:
            out_dir = config.absolute_dir / "postmortem"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{report.target_date}_{report.city}.json"

            data = asdict(report)
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved postmortem to {out_path}")
        except Exception as e:
            logger.error(f"Failed to save postmortem JSON: {e}")
