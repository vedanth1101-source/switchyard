"""Offline checks for the migration round-trip guard: it must refuse non-local or non-PostgreSQL URLs."""

import pytest

from tests.test_migrations_integration import require_local_postgres_url


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
