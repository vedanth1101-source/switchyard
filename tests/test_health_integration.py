"""Real-service health check (Task 0.5). Skipped by the default `pytest` run.

Run with real PostgreSQL and Redis, e.g. CI service containers or `docker compose up -d`:

    DATABASE_URL=postgresql+psycopg://<user>:<password>@localhost:5432/<db> \
    REDIS_URL=redis://localhost:6379/0 \
    pytest -m integration
"""

import os
from collections.abc import Mapping
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

from app.core.settings import get_settings
from app.main import create_app

pytestmark = pytest.mark.integration


def require_real_service_urls(environ: Mapping[str, str]) -> tuple[str, str]:
    """Fail unless DATABASE_URL is PostgreSQL and REDIS_URL is redis:// or rediss://.

    SQLite or any other database must not satisfy the PostgreSQL requirement. Failure messages
    name the variable, never its value.
    """
    # Hide this frame from tracebacks: pytest would otherwise print the `environ` argument, including
    # both URLs' passwords and every other variable.
    __tracebackhide__ = True
    database_url = environ.get("DATABASE_URL", "").strip()
    redis_url = environ.get("REDIS_URL", "").strip()
    if not database_url:
        pytest.fail("Integration test needs a real PostgreSQL service; set DATABASE_URL")
    if not redis_url:
        pytest.fail("Integration test needs a real Redis service; set REDIS_URL")
    try:
        backend = make_url(database_url).get_backend_name()
    except ArgumentError:
        backend = None
    if backend != "postgresql":
        pytest.fail("DATABASE_URL must be a PostgreSQL URL (postgresql:// or postgresql+psycopg://)")
    if urlsplit(redis_url).scheme not in {"redis", "rediss"}:
        pytest.fail("REDIS_URL must use redis:// or rediss://")
    return database_url, redis_url


def test_health_reports_db_and_redis():
    require_real_service_urls(os.environ)
    get_settings.cache_clear()

    response = TestClient(create_app()).get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "db": "ok", "redis": "ok"}
