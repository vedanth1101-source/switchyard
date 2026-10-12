"""PostgreSQL connectivity (Neon pooled URL in production)."""

from functools import lru_cache

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

TIMEOUT_SECONDS = 2


@lru_cache(maxsize=4)
def get_engine(url: str) -> Engine:
    """Build (but do not connect) an engine; cached per URL.

    NullPool closes every connection after use, which suits serverless functions and Neon's
    own connection pooler (TRD: NullPool + pool_pre_ping).
    """
    connect_args = {"connect_timeout": TIMEOUT_SECONDS} if url.startswith("postgresql") else {}
    return create_engine(url, poolclass=NullPool, pool_pre_ping=True, connect_args=connect_args)


def check_database(engine: Engine) -> None:
    """Run SELECT 1; raises the driver's error (wrapped by SQLAlchemy) on failure."""
    with engine.connect() as connection:
        if connection.dialect.name == "postgresql":
            # Transaction-scoped, so it also works through Neon's PgBouncer pooler.
            connection.execute(text(f"SET LOCAL statement_timeout = {TIMEOUT_SECONDS * 1000}"))
        connection.execute(text("SELECT 1")).scalar_one()
