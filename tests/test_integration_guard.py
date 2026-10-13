"""Offline checks for the integration test's URL guard. No services are contacted."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_health_integration import require_real_service_urls

POSTGRES = "postgresql+psycopg://user:pw@localhost:5432/switchyard"
REDIS = "redis://localhost:6379/0"
SENTINEL = "sentinel-secret-must-never-print"


# Every case is refused by the guard before any connection is attempted.
@pytest.mark.parametrize(
    ("database_url", "redis_url", "message"),
    [
        (None, REDIS, "set DATABASE_URL"),
        (f"mysql+pymysql://user:{SENTINEL}@db.example/db", REDIS, "must be a PostgreSQL URL"),
        (POSTGRES, f"http://default:{SENTINEL}@cache.example:6379", "must use redis:// or rediss://"),
    ],
    ids=["unset", "database", "redis"],
)
def test_guard_failure_output_never_prints_the_environment(database_url, redis_url, message):
    # pytest's long traceback prints each frame's arguments; the guard receives os.environ, so without a
    # hidden frame every environment variable (and both URLs' passwords) appears in the failure.
    root = Path(__file__).resolve().parents[1]
    env = {name: value for name, value in os.environ.items() if name not in {"DATABASE_URL", "REDIS_URL"}}
    if database_url is not None:
        env["DATABASE_URL"] = database_url
    env["REDIS_URL"] = redis_url
    env["UNRELATED_SECRET_FOR_GUARD_TEST"] = SENTINEL
    target = "tests/test_health_integration.py::test_health_reports_db_and_redis"

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-m", "integration", "-p", "no:cacheprovider", "--no-cov", target],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 1
    assert message in output  # The guard still refused it, before connecting
    assert SENTINEL not in output
    assert "environ(" not in output


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
