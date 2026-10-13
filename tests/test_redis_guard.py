"""Offline checks for the budget integration guard: it must refuse any Redis that is not local."""

import pytest

from tests.test_budget_integration import require_local_redis_url


@pytest.mark.parametrize(
    "url",
    [
        "redis://localhost:6379/0",
        "redis://127.0.0.1/15",
        "redis://[::1]:6379",
        "redis://user:pw@localhost:6379/0",
        "  redis://localhost:6379/0  ",
    ],
)
def test_accepts_local_redis(url):
    assert require_local_redis_url({"REDIS_URL": url}) == url.strip()


@pytest.mark.parametrize(
    ("environ", "message"),
    [
        ({}, "set REDIS_URL"),
        ({"REDIS_URL": "  "}, "set REDIS_URL"),
        ({"REDIS_URL": "rediss://default:secret@localhost:6379"}, "must use redis://"),
        ({"REDIS_URL": "rediss://default:secret@eu1-x.upstash.io:6379"}, "must use redis://"),
        ({"REDIS_URL": "unix:///tmp/redis-secret.sock"}, "must use redis://"),
        ({"REDIS_URL": "http://localhost:6379/secret"}, "must use redis://"),
        ({"REDIS_URL": "not a url secret"}, "must use redis://"),
        ({"REDIS_URL": "redis://default:secret@redis.internal:6379"}, "only runs against a local host"),
        ({"REDIS_URL": "redis://default:secret@localhost.attacker.example:6379"}, "only runs against a local host"),
        ({"REDIS_URL": "redis://default:secret@10.0.0.5:6379"}, "only runs against a local host"),
        ({"REDIS_URL": "redis:///secret-no-host"}, "only runs against a local host"),
    ],
    ids=[
        "unset",
        "blank",
        "tls-local",
        "upstash",
        "unix-socket",
        "http",
        "garbage",
        "remote-name",
        "lookalike-name",
        "remote-ip",
        "no-host",
    ],
)
def test_rejects_other_urls_without_leaking_them(environ, message):
    with pytest.raises(pytest.fail.Exception, match=message) as failure:
        require_local_redis_url(environ)

    assert "secret" not in str(failure.value)
    assert "upstash" not in str(failure.value)
