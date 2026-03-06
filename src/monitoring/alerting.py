"""Alert system for risk limits, errors, and notable events."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    level: AlertLevel
    message: str
    source: str


class AlertManager:
    """Manages and dispatches alerts."""

    def __init__(self):
        self._handlers: list = []
        self._recent_alerts: list[Alert] = []

    def alert(self, level: AlertLevel, message: str, source: str = "system"):
        """Send an alert."""
        alert = Alert(level=level, message=message, source=source)
        self._recent_alerts.append(alert)

        # Always log
        log_method = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.CRITICAL: logger.critical,
        }.get(level, logger.info)

        log_method(f"[{source}] {message}")

        # Dispatch to handlers
        for handler in self._handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

    def info(self, message: str, source: str = "system"):
        self.alert(AlertLevel.INFO, message, source)

    def warning(self, message: str, source: str = "system"):
        self.alert(AlertLevel.WARNING, message, source)

    def critical(self, message: str, source: str = "system"):
        self.alert(AlertLevel.CRITICAL, message, source)

    def add_handler(self, handler):
        """Add a custom alert handler (e.g., webhook, email)."""
        self._handlers.append(handler)

    def get_recent(self, n: int = 20) -> list[Alert]:
        """Get recent alerts."""
        return self._recent_alerts[-n:]
