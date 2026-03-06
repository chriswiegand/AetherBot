"""City/station configuration loaded from config/cities.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class CityConfig:
    name: str
    station: str
    lat: float
    lon: float
    nws_office: str
    timezone: str
    kalshi_series: str
    cli_site: str


def load_cities(config_path: Path | None = None) -> dict[str, CityConfig]:
    """Load city configurations from YAML."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "cities.yaml"

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    cities = {}
    for name, data in raw["cities"].items():
        cities[name] = CityConfig(name=name, **data)
    return cities
