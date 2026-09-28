"""SQLAlchemy engine, session factory and schema bootstrap."""

from __future__ import annotations

from collections.abc import Generator, Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings

settings = get_settings()

_is_sqlite = settings.sqlalchemy_url.startswith("sqlite")

engine: Engine = create_engine(
    settings.sqlalchemy_url,
    echo=False,
    future=True,
    connect_args={"check_same_thread": False, "timeout": 30} if _is_sqlite else {},
)

if _is_sqlite:

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_connection, _connection_record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        # WAL lets the API read while a background scrape job writes.
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    """Create all tables. Safe to call repeatedly."""
    from app import models  # noqa: F401  (registers mappers)

    Base.metadata.create_all(bind=engine)
    _ensure_sqlite_columns()


def _ensure_sqlite_columns() -> None:
    """Add columns introduced after the DB was first created (SQLite has no migrations)."""
    if not _is_sqlite:
        return
    from sqlalchemy import text

    with engine.begin() as conn:
        cols = {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(recommendation_feedback)")).all()
        }
        if cols and "phone_ratings" not in cols:
            conn.execute(
                text("ALTER TABLE recommendation_feedback ADD COLUMN phone_ratings JSON")
            )
        if cols and "ranking_method" not in cols:
            conn.execute(
                text(
                    "ALTER TABLE recommendation_feedback "
                    "ADD COLUMN ranking_method VARCHAR(32) DEFAULT 'weighted'"
                )
            )


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for background jobs and CLI commands."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
