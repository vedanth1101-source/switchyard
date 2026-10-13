"""POST /api/route persisting to a real, local PostgreSQL (Task S2.5). Skipped by the default `pytest` run.

Runs in CI against the Postgres service container. Jev and the model gateway are fakes; the recorder,
mapping, engine and database are real. Reuses the migration guard, so it refuses any non-local host
(for example Neon), and deletes exactly the requests each test created (model_calls cascade).
"""

import hashlib
import logging
import os
import socket
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy.pool import NullPool

from app.core.router import RouterDeps
from app.core.settings import get_settings
from app.db.models import ModelCallRecord, RequestRecord
from app.infra.db import get_engine
from app.main import create_app
from app.routes import route as route_api
from tests.fakes import FakeBudget, FakeClock, FakeJev, FakeLLM, RecordingSleep
from tests.test_migrations_integration import require_local_postgres_url
from tests.test_repo_mapping import CHEAP, CONFIG, EASY_JEV, PROMPT, client_error, ok

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parent.parent
REQUESTS = RequestRecord.__table__
MODEL_CALLS = ModelCallRecord.__table__


@pytest.fixture
def engine():
    database_url = require_local_postgres_url(os.environ)
    config = Config(str(ROOT / "alembic.ini"))
    config.attributes["configure_logger"] = False
    command.upgrade(config, "head")
    engine = sa.create_engine(database_url, poolclass=NullPool)
    yield engine
    engine.dispose()


@pytest.fixture
def created(engine):
    """Request ids a test's requests produced; deleted afterwards (model_calls cascade)."""
    ids: list = []
    yield ids
    if ids:
        with engine.begin() as connection:
            connection.execute(REQUESTS.delete().where(REQUESTS.c.id.in_(ids)))


def client_with_fake_services(script) -> TestClient:
    """The real app and recorder; only Jev, the strong budget and the model gateway are fakes."""
    clock = FakeClock()
    deps = RouterDeps(
        config=CONFIG,
        jev=FakeJev(EASY_JEV, clock),
        budget=FakeBudget(),
        llm=FakeLLM(list(script), clock),
        request_budget_seconds=50.0,
        clock=clock,
        sleep=RecordingSleep(clock),
    )

    @asynccontextmanager
    async def factory(settings, *, dry_run):
        yield deps

    app = create_app()
    app.dependency_overrides[route_api.get_deps_factory] = lambda: factory
    return TestClient(app)


def post(script, created: list) -> tuple:
    response = client_with_fake_services(script).post("/api/route", json={"prompt": PROMPT})
    request_id = uuid.UUID(response.headers["X-Request-ID"])
    created.append(request_id)
    return response, request_id


def stored_request(engine, request_id):
    with engine.connect() as connection:
        return connection.execute(sa.select(REQUESTS).where(REQUESTS.c.id == request_id)).mappings().one_or_none()


def stored_calls(engine, request_id) -> list:
    with engine.connect() as connection:
        query = sa.select(MODEL_CALLS).where(MODEL_CALLS.c.request_id == request_id).order_by(MODEL_CALLS.c.seq)
        return list(connection.execute(query).mappings())


def test_routed_request_is_saved_under_its_response_request_id(engine, created):
    response, request_id = post([ok(CHEAP[0])], created)

    assert response.status_code == 200
    stored = stored_request(engine, request_id)
    assert stored is not None
    assert str(stored["id"]) == response.json()["request_id"] == response.headers["X-Request-ID"]
    assert stored["prompt_hash"] == hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
    assert stored["prompt_chars"] == len(PROMPT)
    assert (stored["status"], stored["final_model"], stored["reason"]) == ("success", CHEAP[0], "EASY")
    assert [(call["seq"], call["status"]) for call in stored_calls(engine, request_id)] == [(0, "success")]


def test_failed_route_is_saved_with_its_attempts_in_order(engine, created):
    response, request_id = post([client_error(), client_error()], created)

    assert response.status_code == 502
    stored = stored_request(engine, request_id)
    assert (stored["status"], stored["failure_reason"], stored["final_model"]) == ("failed", "all_failed", None)
    assert [(c["seq"], c["model"], c["status"], c["http_status"]) for c in stored_calls(engine, request_id)] == [
        (0, CHEAP[0], "client_error", 400),
        (1, CHEAP[1], "client_error", 400),
    ]


def closed_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_unreachable_database_still_returns_the_routing_response(monkeypatch, caplog):
    secret = "SECRET-DB-PASSWORD"
    monkeypatch.setenv("DATABASE_URL", f"postgresql+psycopg://user:{secret}@127.0.0.1:{closed_local_port()}/db")
    get_settings.cache_clear()
    get_engine.cache_clear()

    with caplog.at_level(logging.INFO):
        response = client_with_fake_services([ok(CHEAP[0])]).post("/api/route", json={"prompt": PROMPT})

    assert response.status_code == 200
    assert response.json()["response"]["model"] == CHEAP[0]
    failures = [r for r in caplog.records if r.getMessage().startswith("route persistence failed")]
    assert [(r.request_id, r.error_type) for r in failures] == [(response.headers["X-Request-ID"], "OperationalError")]
    assert secret not in caplog.text
    assert "127.0.0.1" not in caplog.text
    get_engine.cache_clear()
