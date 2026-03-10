"""Bracket arbitrage scanner for Kalshi KXHIGH markets.

For a complete set of mutually exclusive bracket contracts on the same
event (city + date), exactly ONE bracket settles YES. If the sum of all
YES ask prices exceeds $1.00 + total fees, buying NO on every bracket
locks in a guaranteed profit regardless of outcome.

Guaranteed profit = sum(YES_ask) - $1.00 - (N_contracts × fee_per_contract)

Kalshi KXHIGH bracket structure per event:
  - "T" threshold contracts with strike_type="less" (open-ended low: "≤X°F")
  - "B" bracket contracts with strike_type="between" (closed: "X-Y°F")
  - "T" threshold contracts with strike_type="greater" (open-ended high: "≥X°F")
All contracts in a single event are mutually exclusive and exhaustive.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from src.data.db import get_session
from src.data.models import KalshiMarket

logger = logging.getLogger(__name__)

# Kalshi charges ~$0.02 per contract (taker fee)
FEE_PER_CONTRACT = 0.02


@dataclass
class BracketArb:
    """A detected bracket arbitrage opportunity."""
    city: str
    target_date: str
    event_ticker: str
    n_brackets: int
    sum_yes_ask: float
    total_fees: float
    guaranteed_profit: float  # per 1-contract sweep
    brackets: list[dict]  # [{ticker, label, yes_price, no_price}]


def scan_arbitrage(fee_per_contract: float = FEE_PER_CONTRACT) -> list[BracketArb]:
    """Scan all active events for bracket arbitrage opportunities.

    Includes ALL market types in each event (bracket + threshold contracts)
    since they form a complete mutually exclusive set.

    Returns a list of BracketArb objects sorted by profit descending.
    """
    session = get_session()
    try:
        today = date.today().isoformat()

        # Get ALL active markets for future dates (brackets AND thresholds)
        markets = (
            session.query(KalshiMarket)
            .filter(KalshiMarket.target_date >= today)
            .filter(KalshiMarket.status.in_(["open", "active"]))
            .order_by(KalshiMarket.target_date, KalshiMarket.city)
            .all()
        )

        # Group by event_ticker (which groups all contracts for same city+date)
        grouped: dict[str, list[KalshiMarket]] = defaultdict(list)
        for m in markets:
            grouped[m.event_ticker].append(m)

        opportunities: list[BracketArb] = []

        for event_ticker, event_markets in grouped.items():
            # Need at least 3 markets to form a useful bracket set
            # (1 low-end + 1+ middle + 1 high-end)
            if len(event_markets) < 3:
                continue

            # Separate into bracket (B) and threshold (T) contracts
            brackets = [m for m in event_markets if not m.is_above_contract]
            thresholds = [m for m in event_markets if m.is_above_contract]

            # Identify the open-ended threshold contracts
            # "less" type = low-end cap (e.g., "62° or below")
            # "greater" type = high-end floor (e.g., "71° or above")
            low_end = None
            high_end = None
            for t in thresholds:
                # Low-end: has threshold_f, brackets have bracket_low >= threshold
                # High-end: has threshold_f, brackets have bracket_high <= threshold
                if brackets:
                    min_bracket_low = min(
                        (b.bracket_low for b in brackets if b.bracket_low is not None),
                        default=999
                    )
                    max_bracket_high = max(
                        (b.bracket_high for b in brackets if b.bracket_high is not None),
                        default=-999
                    )
                    if t.threshold_f is not None:
                        if t.threshold_f <= min_bracket_low:
                            low_end = t
                        elif t.threshold_f >= max_bracket_high:
                            high_end = t

            if not low_end or not high_end:
                continue  # Incomplete — missing open-ended brackets

            # Build the complete bracket set: low_end + sorted brackets + high_end
            all_contracts = [low_end] + sorted(
                brackets, key=lambda b: b.bracket_low or 0
            ) + [high_end]

            # All must have prices
            if not all(m.yes_price is not None and m.yes_price > 0 for m in all_contracts):
                continue

            city = event_markets[0].city
            target_date = event_markets[0].target_date
            n = len(all_contracts)
            sum_yes = sum(m.yes_price for m in all_contracts)
            total_fees = n * fee_per_contract
            profit = sum_yes - 1.0 - total_fees

            # Build bracket details
            bracket_details = []
            for m in all_contracts:
                if m == low_end:
                    label = f"\u2264{int(m.threshold_f - 1)}\u00b0F"
                elif m == high_end:
                    label = f"\u2265{int(m.threshold_f + 1)}\u00b0F"
                else:
                    label = f"{int(m.bracket_low)}-{int(m.bracket_high)}\u00b0F"

                bracket_details.append({
                    "ticker": m.market_ticker,
                    "label": label,
                    "yes_price": m.yes_price,
                    "no_price": m.no_price,
                })

            arb = BracketArb(
                city=city,
                target_date=target_date,
                event_ticker=event_ticker,
                n_brackets=n,
                sum_yes_ask=sum_yes,
                total_fees=total_fees,
                guaranteed_profit=profit,
                brackets=bracket_details,
            )

            opportunities.append(arb)

        # Sort by profit descending
        opportunities.sort(key=lambda a: a.guaranteed_profit, reverse=True)
        return opportunities

    finally:
        session.close()


def execute_sweep(
    kalshi_client,
    arb: BracketArb,
    contracts: int = 1,
) -> list[dict]:
    """Execute an arbitrage sweep: buy NO on every bracket in the set.

    Uses limit orders at the current YES ask price (which means
    NO price = 100 - yes_ask_cents).

    Args:
        kalshi_client: Authenticated KalshiClient instance
        arb: The arbitrage opportunity to sweep
        contracts: Number of contracts per bracket

    Returns:
        List of order results [{ticker, order_id, status, no_price}]
    """
    results = []

    for bracket in arb.brackets:
        ticker = bracket["ticker"]
        yes_price = bracket["yes_price"]
        # NO price in cents = 100 - YES_price_cents
        no_price_cents = 100 - int(round(yes_price * 100))

        try:
            order = kalshi_client.create_order(
                ticker=ticker,
                side="no",
                action="buy",
                order_type="limit",
                no_price=no_price_cents,
                count=contracts,
            )
            results.append({
                "ticker": ticker,
                "label": bracket["label"],
                "order_id": order.order_id,
                "status": order.status,
                "no_price": no_price_cents / 100,
            })
            logger.info(
                f"ARB SWEEP: NO {contracts}x {ticker} @ {no_price_cents}c "
                f"(order_id={order.order_id})"
            )
        except Exception as e:
            logger.error(f"Arb sweep failed for {ticker}: {e}")
            results.append({
                "ticker": ticker,
                "label": bracket["label"],
                "order_id": None,
                "status": "error",
                "error": str(e),
            })

    return results
