"""GET /api/health contract (Task 0.5): API, PostgreSQL and Redis status."""

import asyncio
import logging

import pytest
from fastapi.testclient import TestClient

from app.core.settings import Settings, get_settings
from app.main import app, create_app
from app.routes import health

SECRET_DB_URL = "postgresql+psycopg://admin:SuperSecret123@db.internal.example:5432/switchyard"
SECRET_REDIS_URL = "rediss://default:RedisSecret456@cache.internal.example:6379"
LEAK_MARKERS = ["SuperSecret123", "RedisSecret456", "admin", "db.internal.example", "cache.internal.example"]


def client_with_statuses(db: str, redis: str) -> TestClient:
    test_app = create_app()
    test_app.dependency_overrides[health.database_status] = lambda: db
    test_app.dependency_overrides[health.redis_status] = lambda: redis
    return TestClient(test_app)


def client_with_urls(database_url: str | None, redis_url: str | None) -> TestClient:
    test_app = create_app()
    test_app.dependency_overrides[get_settings] = lambda: Settings(database_url=database_url, redis_url=redis_url)
    return TestClient(test_app)


def test_health_ok():
    response = client_with_statuses("ok", "ok").get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "db": "ok", "redis": "ok"}


@pytest.mark.parametrize(
    ("db", "redis", "expected"),
    [
        ("error", "ok", {"status": "degraded", "db": "error", "redis": "ok"}),
        ("ok", "error", {"status": "degraded", "db": "ok", "redis": "error"}),
        ("error", "error", {"status": "degraded", "db": "error", "redis": "error"}),
        ("not_configured", "ok", {"status": "degraded", "db": "not_configured", "redis": "ok"}),
        ("ok", "not_configured", {"status": "degraded", "db": "ok", "redis": "not_configured"}),
    ],
)
def test_degraded_dependency_returns_503_with_both_statuses(db, redis, expected):
    response = client_with_statuses(db, redis).get("/api/health")

    assert response.status_code == 503
    assert response.json() == expected


def test_health_has_no_config_version_until_task_1_1():
    assert "config_version" not in client_with_statuses("ok", "ok").get("/api/health").json()


def test_missing_urls_are_not_configured_without_crashing():
    response = client_with_urls(None, None).get("/api/health")

    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "db": "not_configured", "redis": "not_configured"}


def test_blank_env_values_count_as_not_configured(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("REDIS_URL", "   ")

    settings = Settings()

    assert settings.database_url is None
    assert settings.redis_url is None


def test_settings_read_urls_from_environment(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite://")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    get_settings.cache_clear()

    assert get_settings().database_url == "sqlite://"
    assert get_settings().redis_url == "redis://localhost:6379/0"


def test_real_check_paths_report_ok(monkeypatch):
    import fakeredis
    from sqlalchemy import create_engine
    from sqlalchemy.pool import NullPool

    monkeypatch.setattr(health, "get_engine", lambda url: create_engine("sqlite://", poolclass=NullPool))
    monkeypatch.setattr(health, "create_redis_client", lambda url: fakeredis.FakeRedis())

    response = client_with_urls("postgresql+psycopg://example/db", "rediss://example:6379").get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "db": "ok", "redis": "ok"}


def test_database_failure_still_checks_redis(monkeypatch, tmp_path):
    import fakeredis
    from sqlalchemy import create_engine
    from sqlalchemy.pool import NullPool

    unreachable = create_engine(f"sqlite:///{tmp_path / 'missing-dir' / 'x.db'}", poolclass=NullPool)
    monkeypatch.setattr(health, "get_engine", lambda url: unreachable)
    redis_calls = []

    def fake_client(url):
        redis_calls.append(url)
        return fakeredis.FakeRedis()

    monkeypatch.setattr(health, "create_redis_client", fake_client)

    response = client_with_urls("postgresql+psycopg://example/db", "rediss://example:6379").get("/api/health")

    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "db": "error", "redis": "ok"}
    assert len(redis_calls) == 1


def test_redis_outage_and_timeout_are_reported_as_error(monkeypatch):
    import fakeredis
    import redis as redis_lib

    server = fakeredis.FakeServer()
    server.connected = False
    monkeypatch.setattr(health, "create_redis_client", lambda url: fakeredis.FakeRedis(server=server))
    assert health.redis_status(Settings(redis_url="rediss://example:6379")) == "error"

    class TimingOutClient:
        closed = False

        def ping(self):
            raise redis_lib.exceptions.TimeoutError("Timeout reading from socket")

        def close(self):
            TimingOutClient.closed = True

    monkeypatch.setattr(health, "create_redis_client", lambda url: TimingOutClient())
    assert health.redis_status(Settings(redis_url="rediss://example:6379")) == "error"
    assert TimingOutClient.closed  # the client is closed even when PING fails


def test_database_timeout_is_reported_as_error(monkeypatch):
    from sqlalchemy.exc import OperationalError

    def timing_out(engine):
        raise OperationalError("SELECT 1", None, TimeoutError("timeout expired"))

    monkeypatch.setattr(health, "get_engine", lambda url: object())
    monkeypatch.setattr(health, "check_database", timing_out)

    assert health.database_status(Settings(database_url="postgresql+psycopg://example/db")) == "error"


def test_malformed_urls_are_reported_as_error_not_crash():
    settings = Settings(database_url="notadialect://x", redis_url="http://not-redis")

    assert health.database_status(settings) == "error"
    assert health.redis_status(settings) == "error"


def test_no_credentials_or_exception_details_leak(monkeypatch, tmp_path, caplog):
    import redis as redis_lib
    from sqlalchemy import create_engine
    from sqlalchemy.pool import NullPool

    class ConnectionRefusedErrorLike(redis_lib.exceptions.ConnectionError):
        pass

    class RefusingClient:
        def ping(self):
            raise ConnectionRefusedErrorLike(f"Error connecting to {SECRET_REDIS_URL}")

        def close(self):
            pass

    monkeypatch.setattr(
        health,
        "get_engine",
        lambda url: create_engine(f"sqlite:///{tmp_path / 'missing' / 'x.db'}", poolclass=NullPool),
    )
    monkeypatch.setattr(health, "create_redis_client", lambda url: RefusingClient())

    with caplog.at_level(logging.WARNING, logger="app.routes.health"):
        response = client_with_urls(SECRET_DB_URL, SECRET_REDIS_URL).get("/api/health")

    assert response.status_code == 503
    for marker in LEAK_MARKERS + ["unable to open", "Error connecting"]:
        assert marker not in response.text
        assert marker not in caplog.text
    assert {r.component for r in caplog.records} == {"db", "redis"}
    assert {r.error_type for r in caplog.records} == {"OperationalError", "ConnectionRefusedErrorLike"}
    # Default logging renders only the message, so it must name the component and error type itself.
    assert {r.getMessage() for r in caplog.records} == {
        "health check failed component=db error_type=OperationalError",
        "health check failed component=redis error_type=ConnectionRefusedErrorLike",
    }


def test_openapi_documents_degraded_503_response():
    responses = TestClient(app).get("/api/openapi.json").json()["paths"]["/api/health"]["get"]["responses"]

    assert set(responses) == {"200", "503"}
    for code in ("200", "503"):
        schema = responses[code]["content"]["application/json"]["schema"]
        assert schema == {"$ref": "#/components/schemas/HealthResponse"}


def test_unexpected_programming_errors_are_not_swallowed(monkeypatch):
    def broken(engine):
        raise RuntimeError("bug")

    monkeypatch.setattr(health, "get_engine", lambda url: object())
    monkeypatch.setattr(health, "check_database", broken)

    with pytest.raises(RuntimeError, match="bug"):
        health.database_status(Settings(database_url="postgresql+psycopg://example/db"))


def test_blocking_checks_run_off_the_event_loop():
    threads = []

    def probe() -> str:
        try:
            asyncio.get_running_loop()
            threads.append("event-loop")
        except RuntimeError:
            threads.append("worker")
        return "ok"

    test_app = create_app()
    test_app.dependency_overrides[health.database_status] = probe
    test_app.dependency_overrides[health.redis_status] = probe

    assert TestClient(test_app).get("/api/health").status_code == 200
    assert threads == ["worker", "worker"]


def test_openapi_and_docs_served_under_api_prefix():
    client = TestClient(app)
    openapi = client.get("/api/openapi.json")
    docs = client.get("/api/docs")

    assert openapi.status_code == 200
    assert "/api/health" in openapi.json()["paths"]
    assert docs.status_code == 200
    assert "text/html" in docs.headers["content-type"]
