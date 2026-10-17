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

import fakeredis
import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy.pool import NullPool

from app.core.router import RouterDeps
from app.core.settings import get_settings
from app.db.models import ModelCallRecord, RequestRecord
from app.infra.cache import CacheConfig, ResponseCache
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


def test_cache_hit_is_saved_under_its_response_request_id(engine, created):
    # S2.14 step 4: the real app, recorders and database; the cache is fakeredis, Jev and the gateway fakes.
    server = fakeredis.FakeServer()
    config = CacheConfig(
        redis_url="redis://localhost:6379/0",
        namespace="test",
        key_secret=SecretStr("fake-cache-key-secret-for-tests-0123456789"),
        ttl_seconds=60,
    )
    client = client_with_fake_services([ok(CHEAP[0])])
    client.app.dependency_overrides[route_api.get_cache_factory] = lambda: (
        lambda settings: ResponseCache(fakeredis.FakeAsyncRedis(server=server), config)
    )
    body = {"prompt": PROMPT, "use_cache": True}

    routed, hit = client.post("/api/route", json=body), client.post("/api/route", json=body)
    routed_id, hit_id = uuid.UUID(routed.headers["X-Request-ID"]), uuid.UUID(hit.headers["X-Request-ID"])
    created.extend([routed_id, hit_id])

    assert (routed.json()["cache_hit"], hit.json()["cache_hit"]) == (False, True)
    stored_routed, stored_hit = stored_request(engine, routed_id), stored_request(engine, hit_id)
    assert stored_routed["cache_hit"] is False
    assert (stored_hit["cache_hit"], stored_hit["status"], stored_hit["final_model"]) == (True, "success", CHEAP[0])
    assert stored_hit["prompt_hash"] == stored_routed["prompt_hash"]
    assert (stored_hit["strong_reserved"], stored_hit["est_cost_usd"], stored_hit["jev_latency_ms"]) == (
        False,
        None,
        None,
    )
    assert stored_calls(engine, hit_id) == []
