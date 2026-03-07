"""SQLAlchemy ORM models for all database tables."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean, Column, ForeignKey, Integer, Float, Text, DateTime,
    Index, UniqueConstraint, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship


class Base(DeclarativeBase):
    pass


class EnsembleForecast(Base):
    """Raw GFS ensemble data: one row per (run, city, valid_time, member)."""
    __tablename__ = "ensemble_forecasts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    city = Column(Text, nullable=False)
    station = Column(Text, nullable=False)
    model_run_time = Column(Text, nullable=False)  # ISO8601 UTC
    valid_time = Column(Text, nullable=False)        # ISO8601 UTC
    member = Column(Integer, nullable=False)          # 0-30
    temperature_f = Column(Float, nullable=False)
    fetched_at = Column(Text, nullable=False)

    __table_args__ = (
        UniqueConstraint("city", "model_run_time", "valid_time", "member"),
        Index("idx_ensemble_city_date", "city", "valid_time"),
    )


class HRRRForecast(Base):
    """HRRR deterministic forecasts."""
    __tablename__ = "hrrr_forecasts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    city = Column(Text, nullable=False)
    station = Column(Text, nullable=False)
    model_run_time = Column(Text, nullable=False)
    valid_time = Column(Text, nullable=False)
    temperature_f = Column(Float, nullable=False)
    fetched_at = Column(Text, nullable=False)

    __table_args__ = (
        UniqueConstraint("city", "model_run_time", "valid_time"),
        Index("idx_hrrr_city_date", "city", "valid_time"),
    )


class NWSForecast(Base):
    """NWS gridpoint forecasts."""
    __tablename__ = "nws_forecasts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    city = Column(Text, nullable=False)
    forecast_date = Column(Text, nullable=False)  # YYYY-MM-DD local
    high_f = Column(Float)
    low_f = Column(Float)
    fetched_at = Column(Text, nullable=False)

    __table_args__ = (
        UniqueConstraint("city", "forecast_date", "fetched_at"),
    )


class Observation(Base):
    """Actual observed temperatures from NWS CLI (settlement truth)."""
    __tablename__ = "observations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    city = Column(Text, nullable=False)
    station = Column(Text, nullable=False)
    date = Column(Text, nullable=False)        # YYYY-MM-DD local
    high_f = Column(Integer)                    # NWS CLI integer F
    low_f = Column(Integer)
    source = Column(Text, nullable=False)       # 'iem_cli' or 'nws_obs'
    raw_json = Column(Text)                     # Full API response
    fetched_at = Column(Text, nullable=False)

    __table_args__ = (
        UniqueConstraint("city", "date", "source"),
        Index("idx_obs_city_date", "city", "date"),
    )


class KalshiMarket(Base):
    """Discovered KXHIGH markets and their brackets."""
    __tablename__ = "kalshi_markets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_ticker = Column(Text, nullable=False)
    market_ticker = Column(Text, nullable=False, unique=True)
    city = Column(Text, nullable=False)
    target_date = Column(Text, nullable=False)  # YYYY-MM-DD
    bracket_low = Column(Float)                  # None for open-ended low
    bracket_high = Column(Float)                 # None for open-ended high
    is_above_contract = Column(Integer, default=0)
    threshold_f = Column(Float)                  # For 'above X' contracts
    yes_price = Column(Float)                    # 0.0-1.0 scale
    no_price = Column(Float)
    volume = Column(Integer)
    close_time = Column(Text)
    status = Column(Text)                        # open, closed, settled
    result = Column(Text)                        # yes, no (after settlement)
    last_updated = Column(Text, nullable=False)

    __table_args__ = (
        Index("idx_km_city_date", "city", "target_date"),
    )


class Signal(Base):
    """Computed model probabilities and edges per market."""
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    city = Column(Text, nullable=False)
    target_date = Column(Text, nullable=False)
    market_ticker = Column(Text, nullable=False)
    computed_at = Column(Text, nullable=False)

    # Raw probabilities
    ensemble_prob = Column(Float)
    hrrr_prob = Column(Float)
    nws_prob = Column(Float)
    blended_prob = Column(Float)
    calibrated_prob = Column(Float)

    # Market data at signal time
    market_yes_price = Column(Float)

    # Edge
    raw_edge = Column(Float)
    abs_edge = Column(Float)

    # Model metadata
    lead_hours = Column(Float)
    ensemble_run = Column(Text)
    hrrr_run = Column(Text)

    __table_args__ = (
        UniqueConstraint("market_ticker", "computed_at"),
        Index("idx_signals_date", "target_date", "city"),
    )


class Trade(Base):
    """All paper and live trades."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(Text, unique=True)         # UUID
    mode = Column(Text, nullable=False)           # 'paper' or 'live'
    market_ticker = Column(Text, nullable=False)
    city = Column(Text, nullable=False)
    target_date = Column(Text, nullable=False)
    side = Column(Text, nullable=False)           # 'yes' or 'no'
    direction = Column(Text, nullable=False)      # 'buy' or 'sell'
    contracts = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)         # 0.0-1.0 scale
    total_cost = Column(Float, nullable=False)    # dollars

    # Strategy
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=True)

    # Signal at time of trade
    signal_id = Column(Integer)
    model_prob = Column(Float)
    market_price = Column(Float)
    edge = Column(Float)
    kelly_fraction = Column(Float)

    # Execution
    kalshi_order_id = Column(Text)
    status = Column(Text, nullable=False)         # pending, filled, settled, cancelled
    fill_price = Column(Float)

    # Settlement
    settled_at = Column(Text)
    settlement_value = Column(Integer)            # 0 or 100
    pnl = Column(Float)

    created_at = Column(Text, nullable=False)
    updated_at = Column(Text, nullable=False)

    __table_args__ = (
        Index("idx_trades_date", "target_date", "city"),
        Index("idx_trades_status", "status"),
    )


class BrierScore(Base):
    """Forecast quality tracking."""
    __tablename__ = "brier_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    city = Column(Text, nullable=False)
    target_date = Column(Text, nullable=False)
    market_ticker = Column(Text, nullable=False)
    forecast_prob = Column(Float, nullable=False)
    outcome = Column(Integer, nullable=False)     # 0 or 1
    brier_contribution = Column(Float, nullable=False)  # (forecast - outcome)^2
    lead_hours = Column(Float)
    model_source = Column(Text)                   # 'ensemble', 'blended', 'calibrated'
    created_at = Column(Text, nullable=False)

    __table_args__ = (
        Index("idx_brier_city", "city", "target_date"),
    )


class DailyPnL(Base):
    """Aggregated daily performance."""
    __tablename__ = "daily_pnl"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Text, nullable=False, unique=True)
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    gross_pnl = Column(Float)
    fees = Column(Float, default=0)
    net_pnl = Column(Float)
    bankroll_start = Column(Float)
    bankroll_end = Column(Float)
    max_drawdown = Column(Float)
    avg_edge = Column(Float)
    brier_score = Column(Float)


class Strategy(Base):
    """Named strategy configurations for the bot."""
    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, unique=True, nullable=False)
    description = Column(Text, default="")

    # Entry filters
    edge_threshold = Column(Float, default=0.08)
    min_edge_hrrr_confirm = Column(Float, default=0.06)
    min_model_prob = Column(Float, default=0.55)

    # Position sizing
    fractional_kelly = Column(Float, default=0.15)
    max_position_pct = Column(Float, default=0.10)
    max_position_dollars = Column(Float, default=1000)

    # Risk management
    daily_loss_limit = Column(Float, default=300)
    max_concurrent_positions = Column(Integer, default=20)
    max_positions_per_city = Column(Integer, default=6)
    max_positions_per_date = Column(Integer, default=4)

    # Market filters
    min_price = Column(Float, default=0.08)
    max_price = Column(Float, default=0.92)
    max_lead_hours = Column(Float, default=72)

    # Metadata
    is_active = Column(Integer, default=0)   # 1 = bot uses this strategy
    created_at = Column(Text, nullable=False)
    updated_at = Column(Text, nullable=False)

    # Relationships
    trades = relationship("Trade", backref="strategy", lazy="dynamic")
    backtest_runs = relationship("BacktestRun", backref="strategy", lazy="dynamic")


class BacktestRun(Base):
    """Stored results from a backtest run."""
    __tablename__ = "backtest_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)

    # Run config
    start_date = Column(Text, nullable=False)
    end_date = Column(Text, nullable=False)
    cities = Column(Text)          # JSON list

    # Results summary
    total_trades = Column(Integer)
    win_rate = Column(Float)
    gross_pnl = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    profit_factor = Column(Float)
    brier_score = Column(Float)
    avg_edge = Column(Float)

    # Full trade list and daily PnL for charting (JSON)
    trades_json = Column(Text)
    daily_pnl_json = Column(Text)

    created_at = Column(Text, nullable=False)
    duration_seconds = Column(Float)


class OptimizationRun(Base):
    """Results from a parameter grid search."""
    __tablename__ = "optimization_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text)
    param_ranges_json = Column(Text)     # JSON of parameter ranges
    target_metric = Column(Text)
    total_combinations = Column(Integer)
    results_json = Column(Text)          # JSON array of {params, metrics}
    best_params_json = Column(Text)      # JSON of best parameter set
    created_at = Column(Text, nullable=False)
    duration_seconds = Column(Float)
