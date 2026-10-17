"""Offline checks for the migration round-trip guard: it must refuse non-local or non-PostgreSQL URLs."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_migrations_integration import require_local_postgres_url

SENTINEL = "sentinel-secret-must-never-print"


@pytest.mark.parametrize(
    "target",
    [
        "tests/test_migrations_integration.py::test_migrations_upgrade_downgrade_round_trip",  # Direct call
        "tests/test_repo_integration.py::test_every_outcome_shape_route_returns_is_accepted",  # Via a fixture
    ],
    ids=["direct", "fixture"],
)
def test_guard_failure_output_never_prints_the_environment(target):
    # pytest's long traceback prints each frame's arguments; the guard receives os.environ, so without a
    # hidden frame every environment variable (and DATABASE_URL's password) appears in the failure.
    root = Path(__file__).resolve().parents[1]
    env = {name: value for name, value in os.environ.items() if name != "DATABASE_URL"}
    env["DATABASE_URL"] = f"postgresql+psycopg://user:{SENTINEL}@ep-example.neon.tech/db"  # Refused: not local
    env["UNRELATED_SECRET_FOR_GUARD_TEST"] = SENTINEL

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
    assert "only runs against a local host" in output  # The guard still refused it, before connecting
    assert SENTINEL not in output
    assert "environ(" not in output


@pytest.mark.parametrize(
    "url",
    [
        "postgresql+psycopg://user:pw@localhost:5432/db",
        "postgresql://user:pw@127.0.0.1/db",
        "  postgresql+psycopg://user@localhost/db  ",
    ],
)
def test_accepts_local_postgres(url):
    assert require_local_postgres_url({"DATABASE_URL": url}) == url.strip()


@pytest.mark.parametrize(
    ("environ", "message"),
    [
        ({}, "set DATABASE_URL"),
        ({"DATABASE_URL": "  "}, "set DATABASE_URL"),
        ({"DATABASE_URL": "sqlite:///db.sqlite3"}, "must be a PostgreSQL URL"),
        ({"DATABASE_URL": "not a url"}, "must be a PostgreSQL URL"),
        ({"DATABASE_URL": "postgresql://user:secret@ep-x.neon.tech/db"}, "only runs against a local host"),
        ({"DATABASE_URL": "postgresql://user:secret@db.internal/db"}, "only runs against a local host"),
    ],
)
def test_rejects_other_urls_without_leaking_them(environ, message):
    with pytest.raises(pytest.fail.Exception, match=message) as failure:
        require_local_postgres_url(environ)

    assert "secret" not in str(failure.value)
