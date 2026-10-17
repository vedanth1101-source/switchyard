"""Alembic environment (Task 0.6).

The database URL comes from DATABASE_URL via app settings (process environment only), never from
alembic.ini, so no URL is written to a file. Migrations run from a laptop or CI, never during the
Vercel build (TRD §10).
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel

import app.db.models  # noqa: F401  Registers the request tables on SQLModel.metadata
from app.core.settings import get_settings

config = context.config

# Tests set configure_logger=False so alembic.ini does not reconfigure their logging.
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# The request tables (app/db/models.py, Task S2.3); autogenerate compares against this.
target_metadata = SQLModel.metadata


def database_url() -> str:
    url = get_settings().database_url
    if url is None:
        raise RuntimeError("Alembic needs a database; set DATABASE_URL in the environment")
    return url


def run_migrations_offline() -> None:
    """Render SQL for the URL's dialect without connecting (`alembic upgrade head --sql`)."""
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    # A separate engine from app.infra.db: migrations must not inherit the 2 s health-check timeout.
    engine = create_engine(database_url(), poolclass=NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
