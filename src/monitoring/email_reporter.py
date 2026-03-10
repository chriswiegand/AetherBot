"""Daily email report via Gmail SMTP.

Sends an HTML summary of open positions, recent settlements,
balance, PnL, model quality stats, and upcoming event timetable.
"""

from __future__ import annotations

import logging
import smtplib
from collections import defaultdict
from datetime import date, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from sqlalchemy import func

from src.config.settings import AppSettings
from src.data.db import get_session
from src.data.models import KalshiMarket, Signal, Trade

logger = logging.getLogger(__name__)


def send_daily_email(settings: AppSettings, get_bankroll=None):
    """Build and send the daily report email."""
    email_cfg = settings.email
    if not email_cfg.enabled or not email_cfg.app_password:
        logger.info("Email not configured — skipping daily email report")
        return

    session = get_session()
    try:
        today = date.today().isoformat()
        yesterday = (date.today() - timedelta(days=1)).isoformat()

        # Open positions (include resting for legacy orders)
        open_trades = (
            session.query(Trade)
            .filter(Trade.status.in_(["filled", "pending", "resting"]))
            .order_by(Trade.target_date, Trade.city)
            .all()
        )

        # Recently settled (last 2 days)
        cutoff = (date.today() - timedelta(days=2)).isoformat()
        settled_trades = (
            session.query(Trade)
            .filter_by(status="settled")
            .filter(Trade.settled_at >= cutoff)
            .order_by(Trade.settled_at.desc())
            .all()
        )

        # All trades ever (for cumulative stats)
        all_settled = (
            session.query(Trade)
            .filter_by(status="settled")
            .all()
        )

        # Today's trades
        trades_today = (
            session.query(Trade)
            .filter(Trade.created_at >= today)
            .filter(Trade.status != "cancelled")
            .all()
        )

        # Upcoming events: future markets grouped by city+date with model probs
        upcoming_events = _query_upcoming_events(session, today)

    finally:
        session.close()

    # Compute stats
    bankroll = get_bankroll() if get_bankroll else 0
    daily_spend = sum(t.total_cost for t in trades_today)
    cum_pnl = sum(t.pnl or 0 for t in all_settled)
    recent_pnl = sum(t.pnl or 0 for t in settled_trades)
    total_settled = len(all_settled)
    wins = sum(1 for t in all_settled if (t.pnl or 0) > 0)
    win_rate = (wins / total_settled * 100) if total_settled else 0
    avg_edge = (
        sum(abs(t.edge or 0) for t in all_settled) / total_settled
        if total_settled else 0
    )

    # Check calibrator health
    warnings = []
    try:
        from src.monitoring.brier_tracker import BrierTracker
        tracker = BrierTracker()
        if tracker.is_calibration_degrading():
            warnings.append(
                "⚠️ Calibration Degrading: 7-day Brier is worse than "
                "30-day baseline. Review model performance on /scorecard."
            )
        if not tracker.is_calibrator_helping():
            warnings.append(
                "⚠️ Calibrator Not Helping: calibrated Brier is worse than "
                "raw blended. Consider retraining or disabling."
            )
    except Exception:
        pass

    # Build HTML
    html = _build_html(
        open_trades=open_trades,
        settled_trades=settled_trades,
        upcoming_events=upcoming_events,
        bankroll=bankroll,
        daily_spend=daily_spend,
        cum_pnl=cum_pnl,
        recent_pnl=recent_pnl,
        total_settled=total_settled,
        win_rate=win_rate,
        avg_edge=avg_edge,
        mode=settings.mode,
        today=today,
        warnings=warnings,
    )

    # Send
    _send_email(
        to=email_cfg.recipient,
        subject=f"AetherBot Daily Report — {today}",
        html=html,
        smtp_host=email_cfg.smtp_host,
        smtp_port=email_cfg.smtp_port,
        app_password=email_cfg.app_password,
        from_addr=email_cfg.recipient,  # Send from self
    )


def _query_upcoming_events(session, today: str) -> list[dict]:
    """Get upcoming settlement dates with lead model probabilities.

    Returns a list of dicts grouped by city+date, each containing:
      city, target_date, markets: [{ticker, threshold, model_prob, market_price, edge, side}]
    """
    # Future markets (Kalshi uses 'active' and 'open' statuses)
    markets = (
        session.query(KalshiMarket)
        .filter(KalshiMarket.target_date >= today)
        .filter(KalshiMarket.status.in_(["open", "active"]))
        .order_by(KalshiMarket.target_date, KalshiMarket.city, KalshiMarket.threshold_f)
        .all()
    )

    if not markets:
        return []

    # Get latest signal per market_ticker via subquery
    latest_sq = (
        session.query(
            Signal.market_ticker,
            func.max(Signal.computed_at).label("max_computed"),
        )
        .group_by(Signal.market_ticker)
        .subquery()
    )
    latest_signals = (
        session.query(Signal)
        .join(
            latest_sq,
            (Signal.market_ticker == latest_sq.c.market_ticker)
            & (Signal.computed_at == latest_sq.c.max_computed),
        )
        .all()
    )
    sig_by_ticker = {s.market_ticker: s for s in latest_signals}

    # Group by city+date
    grouped = defaultdict(list)
    for m in markets:
        sig = sig_by_ticker.get(m.market_ticker)
        model_prob = None
        edge = None
        side = None
        if sig:
            model_prob = sig.calibrated_prob or sig.blended_prob or sig.ensemble_prob
            if model_prob and sig.market_yes_price:
                edge = model_prob - sig.market_yes_price
                side = "YES" if edge > 0 else "NO"

        label = m.market_ticker
        if m.is_above_contract and m.threshold_f is not None:
            label = f">{int(m.threshold_f)}\u00b0F"
        elif m.bracket_low is not None and m.bracket_high is not None:
            label = f"{int(m.bracket_low)}-{int(m.bracket_high)}\u00b0F"

        grouped[(m.city, m.target_date)].append({
            "ticker": m.market_ticker,
            "label": label,
            "threshold": m.threshold_f,
            "model_prob": model_prob,
            "market_price": m.yes_price,
            "edge": edge,
            "side": side,
        })

    # Build result sorted by date then city
    result = []
    for (city, target_date), mkts in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][0])):
        # Only include markets with model data (skip stale ones with no signal)
        with_signal = [m for m in mkts if m["model_prob"] is not None]
        # Pick top 5 by absolute edge
        top = sorted(with_signal, key=lambda m: abs(m["edge"] or 0), reverse=True)[:5]
        result.append({
            "city": city,
            "target_date": target_date,
            "total_markets": len(mkts),
            "top_signals": top,
        })

    return result


def _build_html(
    *,
    open_trades: list[Trade],
    settled_trades: list[Trade],
    upcoming_events: list[dict],
    bankroll: float,
    daily_spend: float,
    cum_pnl: float,
    recent_pnl: float,
    total_settled: int,
    win_rate: float,
    avg_edge: float,
    mode: str,
    today: str,
    warnings: list[str] | None = None,
) -> str:
    style = """
    <style>
        body { font-family: -apple-system, sans-serif; color: #333; max-width: 700px; margin: auto; }
        h2 { color: #1a73e8; border-bottom: 2px solid #1a73e8; padding-bottom: 4px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: right; font-size: 13px; }
        th { background: #f0f4ff; text-align: center; }
        td:first-child { text-align: left; }
        .positive { color: #0a7c42; font-weight: bold; }
        .negative { color: #c53030; font-weight: bold; }
        .footer { margin-top: 30px; padding: 15px; background: #f7f8fa; border-radius: 8px; font-size: 12px; color: #666; }
        .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 20px; }
        .stat-box { background: #f0f4ff; border-radius: 8px; padding: 12px; text-align: center; }
        .stat-box .value { font-size: 22px; font-weight: bold; color: #1a73e8; }
        .stat-box .label { font-size: 11px; color: #666; text-transform: uppercase; }
        .mode-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }
        .mode-live { background: #c53030; color: white; }
        .mode-paper { background: #718096; color: white; }
    </style>
    """

    mode_class = "mode-live" if mode == "live" else "mode-paper"
    pnl_class = "positive" if cum_pnl >= 0 else "negative"

    # Warning banner HTML
    warning_html = ""
    if warnings:
        warning_items = "".join(f"<div style='margin: 4px 0;'>{w}</div>" for w in warnings)
        warning_html = f"""
        <div style="background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px;
                    padding: 12px 16px; margin-bottom: 16px; font-size: 13px; color: #856404;">
            <strong>Model Health Alerts</strong>
            {warning_items}
        </div>
        """

    html = f"""<html><head>{style}</head><body>
    <h1>AetherBot Daily Report
        <span class="mode-badge {mode_class}">{mode.upper()}</span>
    </h1>

    {warning_html}

    <div class="stat-grid">
        <div class="stat-box">
            <div class="value">${bankroll:.2f}</div>
            <div class="label">Balance</div>
        </div>
        <div class="stat-box">
            <div class="value {pnl_class}">${cum_pnl:+.2f}</div>
            <div class="label">Cumulative P&L</div>
        </div>
        <div class="stat-box">
            <div class="value">${daily_spend:.2f}</div>
            <div class="label">Spent Today</div>
        </div>
        <div class="stat-box">
            <div class="value">{win_rate:.0f}%</div>
            <div class="label">Win Rate ({total_settled} trades)</div>
        </div>
    </div>
    """

    # Open Positions
    html += "<h2>Open Positions</h2>"
    if open_trades:
        html += """<table>
        <tr><th>Ticker</th><th>City</th><th>Date</th><th>Side</th><th>Qty</th><th>Price</th><th>Cost</th><th>Edge</th><th>Status</th></tr>"""
        for t in open_trades:
            html += f"""<tr>
                <td>{t.market_ticker}</td><td>{t.city}</td><td>{t.target_date}</td>
                <td>{t.side.upper()}</td><td>{t.contracts}</td>
                <td>{t.price:.2f}</td><td>${t.total_cost:.2f}</td>
                <td>{t.edge:+.1%}</td><td>{t.status}</td>
            </tr>"""
        html += "</table>"
    else:
        html += "<p>No open positions.</p>"

    # Recently Settled
    html += "<h2>Recently Settled</h2>"
    if settled_trades:
        html += """<table>
        <tr><th>Ticker</th><th>City</th><th>Side</th><th>Qty</th><th>Price</th><th>Result</th><th>P&L</th></tr>"""
        for t in settled_trades:
            result_text = "YES" if t.settlement_value == 100 else "NO"
            pnl_cls = "positive" if (t.pnl or 0) > 0 else "negative"
            html += f"""<tr>
                <td>{t.market_ticker}</td><td>{t.city}</td>
                <td>{t.side.upper()}</td><td>{t.contracts}</td>
                <td>{t.price:.2f}</td><td>{result_text}</td>
                <td class="{pnl_cls}">${t.pnl or 0:+.2f}</td>
            </tr>"""
        html += "</table>"
    else:
        html += "<p>No recent settlements.</p>"

    # Upcoming Events Timetable
    html += "<h2>Upcoming Events</h2>"
    if upcoming_events:
        for event in upcoming_events:
            city = event["city"]
            tdate = event["target_date"]
            total = event["total_markets"]
            top = event["top_signals"]
            html += f'<h3 style="margin:12px 0 4px 0;color:#444;">{city} — {tdate} ({total} markets)</h3>'
            if top:
                html += """<table>
                <tr><th>Contract</th><th>Model Prob</th><th>Mkt Price</th><th>Edge</th><th>Side</th></tr>"""
                for m in top:
                    prob_str = f'{m["model_prob"]:.0%}' if m["model_prob"] else "--"
                    price_str = f'{m["market_price"]:.0%}' if m["market_price"] else "--"
                    if m["edge"] is not None:
                        edge_cls = "positive" if m["edge"] > 0 else "negative"
                        edge_str = f'{m["edge"]:+.1%}'
                    else:
                        edge_cls = ""
                        edge_str = "--"
                    side_str = m["side"] or "--"
                    html += f"""<tr>
                        <td>{m["label"]}</td><td>{prob_str}</td><td>{price_str}</td>
                        <td class="{edge_cls}">{edge_str}</td><td>{side_str}</td>
                    </tr>"""
                html += "</table>"
            else:
                html += "<p style='color:#888;font-size:12px;'>No model signals yet.</p>"
    else:
        html += "<p>No upcoming events found.</p>"

    # Stats
    html += f"""<h2>Model Stats</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Average Edge</td><td>{avg_edge:.1%}</td></tr>
        <tr><td>Recent P&L (2d)</td><td class="{'positive' if recent_pnl >= 0 else 'negative'}">${recent_pnl:+.2f}</td></tr>
    </table>
    """

    # Footer — thesis overview
    html += """
    <div class="footer">
        <strong>AetherBot Engine Overview</strong><br>
        AetherBot trades Kalshi KXHIGH temperature markets across 5 US cities
        (NYC, Chicago, Miami, LA, Denver). The engine blends a 31-member GFS
        ensemble (60%), HRRR deterministic corrections (25%), and NWS forecasts
        (15%) with lead-time-adaptive weighting. Probabilities are calibrated
        against historical accuracy, then compared to market prices to find
        edges. Positions are sized via fractional Kelly criterion with a daily
        spend cap, city concentration limits, and a hard loss stop. Settlement
        uses NWS CLI observed highs from the Iowa Environmental Mesonet.
    </div>
    </body></html>
    """

    return html


def _send_email(
    *,
    to: str,
    subject: str,
    html: str,
    smtp_host: str,
    smtp_port: int,
    app_password: str,
    from_addr: str,
):
    """Send an HTML email via Gmail SMTP."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"AetherBot <{from_addr}>"
    msg["To"] = to
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(from_addr, app_password)
            server.sendmail(from_addr, to, msg.as_string())
        logger.info(f"Daily email sent to {to}")
    except Exception as e:
        logger.error(f"Failed to send daily email: {e}")
