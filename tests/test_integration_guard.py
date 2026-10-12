"""Offline checks for the integration test's URL guard. No services are contacted."""

import pytest

from tests.test_health_integration import require_real_service_urls

POSTGRES = "postgresql+psycopg://user:pw@localhost:5432/switchyard"
REDIS = "redis://localhost:6379/0"


@pytest.mark.parametrize(
    ("database_url", "redis_url"),
    [
        (POSTGRES, REDIS),
        ("postgresql://user:pw@localhost:5432/switchyard", "rediss://default:pw@example.upstash.io:6379"),
    ],
)
def test_accepts_postgresql_and_redis_urls(database_url, redis_url):
    environ = {"DATABASE_URL": database_url, "REDIS_URL": redis_url}

    assert require_real_service_urls(environ) == (database_url, redis_url)


@pytest.mark.parametrize(
    ("environ", "reason"),
    [
        ({}, "DATABASE_URL"),
        ({"DATABASE_URL": POSTGRES}, "REDIS_URL"),
        ({"DATABASE_URL": "  ", "REDIS_URL": REDIS}, "DATABASE_URL"),
        ({"DATABASE_URL": "sqlite://", "REDIS_URL": REDIS}, "PostgreSQL"),
        ({"DATABASE_URL": "sqlite:///tmp/x.db", "REDIS_URL": REDIS}, "PostgreSQL"),
        ({"DATABASE_URL": "mysql+pymysql://u:p@localhost/db", "REDIS_URL": REDIS}, "PostgreSQL"),
        ({"DATABASE_URL": "not a url", "REDIS_URL": REDIS}, "PostgreSQL"),
        ({"DATABASE_URL": POSTGRES, "REDIS_URL": "http://localhost:6379"}, "redis://"),
        ({"DATABASE_URL": POSTGRES, "REDIS_URL": "unix:///tmp/redis.sock"}, "redis://"),
    ],
)
def test_rejects_missing_or_non_real_service_urls(environ, reason):
    with pytest.raises(pytest.fail.Exception, match=reason):
        require_real_service_urls(environ)


def test_failure_message_does_not_echo_the_url():
    secret = "postgres-password-should-not-appear"

    with pytest.raises(pytest.fail.Exception) as excinfo:
        require_real_service_urls({"DATABASE_URL": f"sqlite:///{secret}", "REDIS_URL": REDIS})

    assert secret not in str(excinfo.value)
