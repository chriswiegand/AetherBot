"""Database engine and session management."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from src.data.models import Base
from src.config.settings import AppSettings, load_settings


_engine = None
_SessionFactory = None


def get_engine(settings: AppSettings | None = None):
    """Get or create the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        if settings is None:
            settings = load_settings()
        db_path = settings.database.absolute_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(
            settings.database.url,
            echo=False,
            pool_pre_ping=True,
        )
        # Enable WAL mode for better concurrent read performance
        @event.listens_for(_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return _engine


def get_session(settings: AppSettings | None = None) -> Session:
    """Get a new database session."""
    global _SessionFactory
    if _SessionFactory is None:
        engine = get_engine(settings)
        _SessionFactory = sessionmaker(bind=engine)
    return _SessionFactory()


def init_db(settings: AppSettings | None = None):
    """Create all tables if they don't exist."""
    engine = get_engine(settings)
    Base.metadata.create_all(engine)


def reset_engine():
    """Reset the engine and session factory (for testing)."""
    global _engine, _SessionFactory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionFactory = None
