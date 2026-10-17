"""Alembic round trip against a real, local PostgreSQL (Task 0.6). Skipped by the default `pytest` run.

Runs in CI against the Postgres service container. It downgrades to base, so it refuses any
non-local host (for example Neon): migrate Neon with `alembic upgrade head` instead.
"""

import os
from collections.abc import Mapping
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.pool import NullPool

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parent.parent
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def require_local_postgres_url(environ: Mapping[str, str]) -> str:
    """Fail unless DATABASE_URL is PostgreSQL on a local host. Messages never include the URL."""
    # Hide this frame from tracebacks: pytest would otherwise print the `environ` argument, including
    # DATABASE_URL's password and every other variable.
    __tracebackhide__ = True
    database_url = environ.get("DATABASE_URL", "").strip()
    if not database_url:
        pytest.fail("Migration test needs a local PostgreSQL service; set DATABASE_URL")
    try:
        url = make_url(database_url)
    except ArgumentError:
        url = None
    if url is None or url.get_backend_name() != "postgresql":
        pytest.fail("DATABASE_URL must be a PostgreSQL URL (postgresql:// or postgresql+psycopg://)")
    if url.host not in LOCAL_HOSTS:
        pytest.fail("Migration round trip downgrades the database; it only runs against a local host")
    return database_url


def applied_revisions(database_url: str) -> list[str]:
    engine = create_engine(database_url, poolclass=NullPool)
    with engine.connect() as connection:
        return list(connection.execute(text("SELECT version_num FROM alembic_version")).scalars())


def test_migrations_upgrade_downgrade_round_trip():
    database_url = require_local_postgres_url(os.environ)
    config = Config(str(ROOT / "alembic.ini"))
    config.attributes["configure_logger"] = False
    head = ScriptDirectory.from_config(config).get_current_head()

    command.upgrade(config, "head")
    assert applied_revisions(database_url) == [head]

    command.downgrade(config, "base")
    assert applied_revisions(database_url) == []

    command.upgrade(config, "head")
    assert applied_revisions(database_url) == [head]
