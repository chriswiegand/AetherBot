"""Application settings loaded from config/settings.yaml and .env."""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class KalshiConfig:
    base_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    demo_url: str = "https://demo-api.kalshi.co/trade-api/v2"
    api_key_id: str = ""
    private_key_path: str = ""

    def __post_init__(self):
        self.api_key_id = os.getenv("KALSHI_API_KEY_ID", self.api_key_id)
        self.private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", self.private_key_path)
        if self.private_key_path and not Path(self.private_key_path).is_absolute():
            self.private_key_path = str(PROJECT_ROOT / self.private_key_path)

    @property
    def active_url(self) -> str:
        mode = os.getenv("MODE", "paper")
        return self.demo_url if mode == "paper" else self.base_url


@dataclass
class DataSourceConfig:
    ensemble_url: str = "https://ensemble-api.open-meteo.com/v1/ensemble"
    ensemble_model: str = "gfs025"
    ensemble_members: int = 31
    forecast_url: str = "https://api.open-meteo.com/v1/forecast"
    hrrr_model: str = "ncep_hrrr_conus"
    nws_base_url: str = "https://api.weather.gov"
    nws_user_agent: str = "KalshiWeatherBot/1.0 (weather-bot@example.com)"
    iem_cli_url: str = "https://mesonet.agron.iastate.edu/json/cli.py"
    historical_archive_url: str = "https://archive-api.open-meteo.com/v1/archive"


@dataclass
class StrategyConfig:
    edge_threshold: float = 0.08
    min_edge_hrrr_confirm: float = 0.06
    min_model_prob: float = 0.55
    fractional_kelly: float = 0.15
    max_position_pct: float = 0.10
    max_position_dollars: float = 1000
    daily_loss_limit: float = 300
    daily_spend_limit: float = 0  # 0 = unlimited; >0 = hard cap on daily wagers
    max_concurrent_positions: int = 20
    max_positions_per_city: int = 6
    max_positions_per_date: int = 4
    min_contracts: int = 1
    max_price: float = 0.92
    min_price: float = 0.08
    max_lead_hours: float = 72


@dataclass
class ModelWeightsConfig:
    gfs_ensemble: float = 0.60
    hrrr: float = 0.25
    nws: float = 0.15
    hrrr_weight_by_lead_hours: dict[int, float] = field(default_factory=lambda: {
        0: 0.45, 6: 0.35, 12: 0.25, 24: 0.15, 48: 0.05
    })
    gfs_weight_by_lead_hours: dict[int, float] = field(default_factory=lambda: {
        0: 0.15, 6: 0.25, 12: 0.35, 18: 0.40, 24: 0.50, 48: 0.60
    })

    def _interpolate(self, schedule: dict[int, float], lead_hours: float, fallback: float) -> float:
        """Interpolate a weight from a lead-hour schedule."""
        breakpoints = sorted(schedule.keys())
        if lead_hours <= breakpoints[0]:
            return schedule[breakpoints[0]]
        if lead_hours >= breakpoints[-1]:
            return schedule[breakpoints[-1]]
        for i in range(len(breakpoints) - 1):
            lo, hi = breakpoints[i], breakpoints[i + 1]
            if lo <= lead_hours <= hi:
                frac = (lead_hours - lo) / (hi - lo)
                return schedule[lo] + frac * (schedule[hi] - schedule[lo])
        return fallback

    def get_hrrr_weight(self, lead_hours: float) -> float:
        """Interpolate HRRR weight based on lead time."""
        return self._interpolate(self.hrrr_weight_by_lead_hours, lead_hours, self.hrrr)

    def get_gfs_weight(self, lead_hours: float) -> float:
        """Interpolate GFS ensemble weight based on lead time.

        GFS weight decays as lead time decreases because HRRR
        (hourly updates, 3km resolution) becomes far more accurate
        than GFS (6-hourly, 25km) near settlement.
        """
        return self._interpolate(self.gfs_weight_by_lead_hours, lead_hours, self.gfs_ensemble)


@dataclass
class PaperTradingConfig:
    initial_bankroll: float = 10000


@dataclass
class EmailConfig:
    enabled: bool = False
    recipient: str = ""
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587

    def __post_init__(self):
        self.app_password = os.getenv("GMAIL_APP_PASSWORD", "")


@dataclass
class SchedulerConfig:
    # Smart data fetching
    smart_fetch_enabled: bool = True
    smart_fetch_check_minutes: int = 10
    gfs_availability_lag_hours: float = 4.5
    hrrr_availability_lag_hours: float = 1.0

    # Market scanning
    market_scan_interval_minutes: int = 5
    market_discovery_interval_minutes: int = 30

    # Price discovery
    price_discovery_window_hours: float = 2.0
    price_discovery_scan_minutes: int = 2

    # Settlement
    settlement_check_hour: int = 11
    settlement_check_minute: int = 15


@dataclass
class DatabaseConfig:
    path: str = "data/weather_bot.db"

    @property
    def absolute_path(self) -> Path:
        p = Path(self.path)
        if not p.is_absolute():
            return PROJECT_ROOT / p
        return p

    @property
    def url(self) -> str:
        return f"sqlite:///{self.absolute_path}"


@dataclass
class ArchivalConfig:
    enabled: bool = True
    archive_dir: str = "data/archives"
    retention_days: int = 730

    @property
    def absolute_dir(self) -> Path:
        p = Path(self.archive_dir)
        if not p.is_absolute():
            return PROJECT_ROOT / p
        return p


@dataclass
class AppSettings:
    mode: str = "paper"
    kalshi: KalshiConfig = field(default_factory=KalshiConfig)
    data_sources: DataSourceConfig = field(default_factory=DataSourceConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    model_weights: ModelWeightsConfig = field(default_factory=ModelWeightsConfig)
    paper_trading: PaperTradingConfig = field(default_factory=PaperTradingConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    archival: ArchivalConfig = field(default_factory=ArchivalConfig)


def _build_dataclass(cls, data: dict):
    """Build a dataclass from a dict, ignoring unknown keys."""
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered)


def load_settings(config_path: Path | None = None) -> AppSettings:
    """Load settings from YAML config file and environment variables."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "settings.yaml"

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    mode = os.getenv("MODE", raw.get("mode", "paper"))

    settings = AppSettings(
        mode=mode,
        kalshi=_build_dataclass(KalshiConfig, raw.get("kalshi", {})),
        data_sources=_build_dataclass(DataSourceConfig, raw.get("data_sources", {})),
        strategy=_build_dataclass(StrategyConfig, raw.get("strategy", {})),
        model_weights=_build_dataclass(ModelWeightsConfig, raw.get("model_weights", {})),
        paper_trading=_build_dataclass(PaperTradingConfig, raw.get("paper_trading", {})),
        scheduler=_build_dataclass(SchedulerConfig, raw.get("scheduler", {})),
        email=_build_dataclass(EmailConfig, raw.get("email", {})),
        database=_build_dataclass(DatabaseConfig, raw.get("database", {})),
        archival=_build_dataclass(ArchivalConfig, raw.get("archival", {})),
    )
    return settings
