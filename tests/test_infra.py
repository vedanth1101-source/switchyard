"""Database and Redis connectivity helpers (Task 0.5). Offline: SQLite and fakeredis only."""

import fakeredis
import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool

from app.infra import db, redis


def test_check_database_runs_select_1_on_a_real_engine():
    db.check_database(create_engine("sqlite://", poolclass=NullPool))


def test_check_database_raises_operational_error_when_unreachable(tmp_path):
    unreachable = create_engine(f"sqlite:///{tmp_path / 'missing-dir' / 'x.db'}", poolclass=NullPool)

    with pytest.raises(OperationalError):
        db.check_database(unreachable)


def test_get_engine_uses_null_pool_pre_ping_and_connect_timeout(monkeypatch):
    db.get_engine.cache_clear()
    calls = []
    real_create_engine = db.create_engine

    def recording_create_engine(url, **kwargs):
        calls.append(kwargs)
        return real_create_engine(url, **kwargs)

    monkeypatch.setattr(db, "create_engine", recording_create_engine)

    engine = db.get_engine("postgresql+psycopg://user:pw@localhost:5432/switchyard")  # builds only; no connection

    assert isinstance(engine.pool, NullPool)
    assert engine.pool._pre_ping is True
    assert calls == [
        {
            "poolclass": NullPool,
            "pool_pre_ping": True,
            "connect_args": {"connect_timeout": db.TIMEOUT_SECONDS},
            "hide_parameters": True,  # S2.5 (approved D3)
        }
    ]
    db.get_engine.cache_clear()


def test_get_engine_keeps_sql_parameters_out_of_error_messages():
    # Defense in depth for S2.5: a rendered SQLAlchemy error never carries the row values.
    from sqlalchemy import text
    from sqlalchemy.exc import IntegrityError

    db.get_engine.cache_clear()
    engine = db.get_engine("sqlite://")
    assert engine.hide_parameters is True

    with engine.connect() as connection:
        connection.execute(text("CREATE TABLE t (x TEXT NOT NULL, y INTEGER NOT NULL)"))
        with pytest.raises(IntegrityError) as error:
            connection.execute(text("INSERT INTO t (x, y) VALUES (:x, :y)"), {"x": "SECRET-ROW-VALUE", "y": None})

    assert "SECRET-ROW-VALUE" not in str(error.value)
    db.get_engine.cache_clear()


def test_get_engine_is_cached_per_url_and_does_not_hold_connections():
    from sqlalchemy import event

    db.get_engine.cache_clear()
    first = db.get_engine("sqlite://")
    opened, closed = [], []
    event.listen(first, "connect", lambda *args: opened.append(1))
    event.listen(first, "close", lambda *args: closed.append(1))

    assert db.get_engine("sqlite://") is first
    for _ in range(3):
        db.check_database(first)

    assert len(opened) == 3
    assert len(closed) == 3  # NullPool closes every connection after use; none are kept open
    db.get_engine.cache_clear()


def test_timeout_is_about_two_seconds():
    assert db.TIMEOUT_SECONDS == 2
    assert redis.TIMEOUT_SECONDS == 2


def test_create_redis_client_sets_connect_and_operation_timeouts():
    client = redis.create_redis_client("rediss://default:pw@example.upstash.io:6379")  # no connection yet
    try:
        kwargs = client.connection_pool.connection_kwargs
        assert kwargs["socket_connect_timeout"] == redis.TIMEOUT_SECONDS
        assert kwargs["socket_timeout"] == redis.TIMEOUT_SECONDS
        assert client.connection_pool.connection_class.__name__ == "SSLConnection"  # rediss:// means TLS
    finally:
        client.close()


def test_check_redis_pings_fakeredis():
    redis.check_redis(fakeredis.FakeRedis())


def test_check_redis_raises_when_server_unavailable():
    import redis as redis_lib

    server = fakeredis.FakeServer()
    server.connected = False

    with pytest.raises(redis_lib.exceptions.ConnectionError):
        redis.check_redis(fakeredis.FakeRedis(server=server))
