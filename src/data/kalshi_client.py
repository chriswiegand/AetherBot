"""Kalshi API v2 client with RSA-PSS authentication.

Public endpoints (market data) require no authentication.
Authenticated endpoints (trading) use RSA-PSS signatures.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from src.config.settings import AppSettings, load_settings

logger = logging.getLogger(__name__)


@dataclass
class Market:
    ticker: str
    event_ticker: str
    title: str
    subtitle: str
    yes_bid: float
    yes_ask: float
    last_price: float
    volume: int
    open_interest: int
    status: str
    result: str | None
    close_time: str | None
    raw: dict


@dataclass
class OrderbookLevel:
    price: float
    quantity: float


@dataclass
class Orderbook:
    ticker: str
    yes_bids: list[OrderbookLevel]
    no_bids: list[OrderbookLevel]


@dataclass
class Order:
    order_id: str
    ticker: str
    side: str
    type: str
    status: str
    price: float
    count: int
    remaining: int


class KalshiClient:
    """Kalshi API v2 wrapper."""

    def __init__(self, settings: AppSettings | None = None):
        if settings is None:
            settings = load_settings()
        self.settings = settings
        self.kalshi_config = settings.kalshi
        self._client = httpx.Client(timeout=30.0)
        self._private_key = None

    @property
    def base_url(self) -> str:
        return self.kalshi_config.active_url

    def _load_private_key(self):
        """Load RSA private key from file for authentication."""
        if self._private_key is not None:
            return
        key_path = self.kalshi_config.private_key_path
        if not key_path or not Path(key_path).exists():
            logger.warning("Kalshi private key not found - authenticated endpoints unavailable")
            return
        with open(key_path, "rb") as f:
            self._private_key = serialization.load_pem_private_key(f.read(), password=None)

    def _sign_request(self, method: str, path: str, timestamp_ms: int) -> str:
        """Create RSA-PSS signature for authenticated requests.

        Message format: {timestamp_ms}{METHOD}{path_without_query}
        """
        self._load_private_key()
        if self._private_key is None:
            raise RuntimeError("Private key not loaded - cannot sign requests")

        path_without_query = path.split("?")[0]
        message = f"{timestamp_ms}{method}{path_without_query}".encode("utf-8")

        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        """Generate authentication headers."""
        timestamp_ms = str(int(time.time() * 1000))
        signature = self._sign_request(method, path, int(timestamp_ms))
        return {
            "KALSHI-ACCESS-KEY": self.kalshi_config.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": signature,
        }

    def _request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        body: dict | None = None,
        auth: bool = False,
    ) -> dict:
        """Make an API request."""
        url = f"{self.base_url}{path}"
        headers = {}

        if auth:
            headers.update(self._auth_headers(method.upper(), f"/trade-api/v2{path}"))

        kwargs = {"headers": headers}
        if params:
            kwargs["params"] = params
        if body:
            kwargs["json"] = body
            headers["Content-Type"] = "application/json"

        resp = self._client.request(method, url, **kwargs)
        resp.raise_for_status()
        return resp.json()

    # --- Public endpoints (no auth) ---

    def get_markets(
        self,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        status: str = "open",
        limit: int = 200,
        cursor: str | None = None,
    ) -> list[Market]:
        """Get markets, optionally filtered by series or event."""
        params = {"limit": limit, "status": status}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if cursor:
            params["cursor"] = cursor

        data = self._request("GET", "/markets", params=params)
        markets = []
        for m in data.get("markets", []):
            markets.append(_parse_market(m))

        return markets

    def get_market(self, ticker: str) -> Market:
        """Get a single market by ticker."""
        data = self._request("GET", f"/markets/{ticker}")
        return _parse_market(data.get("market", data))

    def get_orderbook(self, ticker: str, depth: int = 10) -> Orderbook:
        """Get orderbook for a market."""
        data = self._request("GET", f"/markets/{ticker}/orderbook", params={"depth": depth})
        ob = data.get("orderbook", data)

        yes_bids = []
        no_bids = []

        for price, qty in ob.get("yes", []):
            yes_bids.append(OrderbookLevel(float(price) / 100, float(qty)))
        for price, qty in ob.get("no", []):
            no_bids.append(OrderbookLevel(float(price) / 100, float(qty)))

        return Orderbook(ticker=ticker, yes_bids=yes_bids, no_bids=no_bids)

    def get_events(self, series_ticker: str, status: str = "open") -> list[dict]:
        """Get events for a series."""
        params = {"series_ticker": series_ticker, "status": status}
        data = self._request("GET", "/events", params=params)
        return data.get("events", [])

    def get_trades(self, ticker: str, limit: int = 100) -> list[dict]:
        """Get recent trades for a market."""
        data = self._request("GET", "/markets/trades", params={"ticker": ticker, "limit": limit})
        return data.get("trades", [])

    # --- Authenticated endpoints ---

    def get_balance(self) -> float:
        """Get account balance in dollars."""
        data = self._request("GET", "/portfolio/balance", auth=True)
        return float(data.get("balance", 0)) / 100  # cents to dollars

    def get_positions(self) -> list[dict]:
        """Get all open positions."""
        data = self._request("GET", "/portfolio/positions", auth=True)
        return data.get("market_positions", [])

    def create_order(
        self,
        ticker: str,
        side: str,
        action: str = "buy",
        order_type: str = "limit",
        yes_price: int | None = None,
        no_price: int | None = None,
        count: int = 1,
    ) -> Order:
        """Place an order.

        Args:
            ticker: Market ticker
            side: 'yes' or 'no'
            action: 'buy' or 'sell'
            order_type: 'limit' or 'market'
            yes_price: Price in cents (1-99) for yes side
            no_price: Price in cents for no side
            count: Number of contracts
        """
        body = {
            "ticker": ticker,
            "action": action,
            "side": side,
            "type": order_type,
            "count": count,
        }
        if yes_price is not None:
            body["yes_price"] = yes_price
        if no_price is not None:
            body["no_price"] = no_price

        data = self._request("POST", "/portfolio/orders", body=body, auth=True)
        order = data.get("order", data)
        return Order(
            order_id=order.get("order_id", ""),
            ticker=ticker,
            side=side,
            type=order_type,
            status=order.get("status", ""),
            price=float(order.get("yes_price", order.get("no_price", 0))) / 100,
            count=count,
            remaining=order.get("remaining_count", count),
        )

    def get_order(self, order_id: str) -> dict:
        """Get order details by ID."""
        data = self._request("GET", f"/portfolio/orders/{order_id}", auth=True)
        return data.get("order", data)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            self._request("DELETE", f"/portfolio/orders/{order_id}", auth=True)
            return True
        except httpx.HTTPStatusError:
            return False

    def close(self):
        self._client.close()


def _parse_market(m: dict) -> Market:
    """Parse a market dict from the API response."""
    return Market(
        ticker=m.get("ticker", ""),
        event_ticker=m.get("event_ticker", ""),
        title=m.get("title", ""),
        subtitle=m.get("subtitle", ""),
        yes_bid=float(m.get("yes_bid", 0)) / 100,
        yes_ask=float(m.get("yes_ask", 0)) / 100,
        last_price=float(m.get("last_price", 0)) / 100,
        volume=int(m.get("volume", 0)),
        open_interest=int(m.get("open_interest", 0)),
        status=m.get("status", ""),
        result=m.get("result"),
        close_time=m.get("close_time"),
        raw=m,
    )
