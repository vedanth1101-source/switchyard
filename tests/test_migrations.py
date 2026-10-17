"""Alembic setup (Task 0.6), offline: no database connection is made."""

import io
import re
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateColumn, CreateTable

from app.db.models import ModelCallRecord, RequestRecord

ROOT = Path(__file__).resolve().parent.parent
# Offline mode only renders SQL for this URL; nothing connects to it.
OFFLINE_URL = "postgresql+psycopg://user@localhost:5432/switchyard"
INITIAL_REVISION = "f969e942a969"
REQUEST_TABLES_REVISION = "18723e702952"
CACHE_HIT_REVISION = "7b67e41902f4"


def alembic_config(output: io.StringIO | None = None) -> Config:
    config = Config(str(ROOT / "alembic.ini"), output_buffer=output)
    # Leave the test process's logging alone (env.py would otherwise apply alembic.ini's config).
    config.attributes["configure_logger"] = False
    return config


def test_revisions_form_one_chain_from_the_empty_initial_revision():
    # S2.3 (approved rewrite): the Sprint 0 revision stays the base; the request tables follow it.
    # S2.14 step 4 (approved): the cache_hit column follows the request tables and is the head.
    script = ScriptDirectory.from_config(alembic_config())
    revisions = list(script.walk_revisions())  # Head first

    assert [revision.revision for revision in revisions] == [
        CACHE_HIT_REVISION,
        REQUEST_TABLES_REVISION,
        INITIAL_REVISION,
    ]
    assert [revision.down_revision for revision in revisions] == [REQUEST_TABLES_REVISION, INITIAL_REVISION, None]
    assert script.get_heads() == [CACHE_HIT_REVISION]


def upgrade_sql(monkeypatch, target: str = "head") -> str:
    monkeypatch.setenv("DATABASE_URL", OFFLINE_URL)
    output = io.StringIO()
    command.upgrade(alembic_config(output), target, sql=True)
    return output.getvalue()


def test_offline_upgrade_creates_only_the_version_and_request_tables(monkeypatch):
    sql = upgrade_sql(monkeypatch)

    head = ScriptDirectory.from_config(alembic_config()).get_current_head()
    assert "CREATE TABLE alembic_version" in sql
    assert f"UPDATE alembic_version SET version_num='{head}'" in sql
    assert sorted(re.findall(r"CREATE TABLE (\w+)", sql)) == ["alembic_version", "model_calls", "requests"]
    assert "CREATE INDEX" not in sql


def test_request_tables_migration_is_additive_only(monkeypatch):
    # S2.14 step 4: scoped to its own revision (it was the head until then).
    sql = upgrade_sql(monkeypatch, f"{INITIAL_REVISION}:{REQUEST_TABLES_REVISION}")

    # Leading keyword of every statement (column and constraint lines are indented, so they don't match).
    statements = re.findall(r"^([A-Z]+)\b", sql, re.MULTILINE)
    assert set(statements) == {"BEGIN", "CREATE", "UPDATE", "COMMIT"}
    assert re.findall(r"^UPDATE (\w+)", sql, re.MULTILINE) == ["alembic_version"]


def test_migration_ddl_matches_the_models(monkeypatch):
    # compare_metadata needs a live database (CI); the DDL text, including every CHECK, can be compared
    # offline because both come from SQLAlchemy's PostgreSQL compiler. S2.14 step 4 (approved rewrite): the
    # models' CREATE TABLE requests equals the request-tables revision's plus the cache_hit revision's
    # ADD COLUMN and ADD CONSTRAINT.
    sql = normalize(upgrade_sql(monkeypatch, f"{INITIAL_REVISION}:head"))
    column, check = cache_hit_ddl()

    requests = normalize(str(CreateTable(RequestRecord.__table__).compile(dialect=postgresql.dialect())).strip())
    assert f"{column}, " in requests and check in requests  # Columns come before constraints, so never last
    before_cache_hit = requests.replace(f"{column}, ", "", 1)
    if f"{check}, " in before_cache_hit:
        before_cache_hit = before_cache_hit.replace(f"{check}, ", "", 1)
    else:  # The last constraint
        before_cache_hit = before_cache_hit.replace(f", {check}", "", 1)
    assert check not in before_cache_hit
    assert before_cache_hit in sql
    model_calls = str(CreateTable(ModelCallRecord.__table__).compile(dialect=postgresql.dialect())).strip()
    assert normalize(model_calls) in sql
    assert f"ALTER TABLE requests ADD COLUMN {column};" in sql
    assert f"ALTER TABLE requests ADD {check};" in sql


def cache_hit_ddl() -> tuple[str, str]:
    """The models' cache_hit column and CHECK, as PostgreSQL DDL fragments."""
    table = RequestRecord.__table__
    column = normalize(str(CreateColumn(table.c.cache_hit).compile(dialect=postgresql.dialect())).strip())
    (constraint,) = [c for c in table.constraints if c.name == "ck_requests_cache_hit_shape"]
    # Not AddConstraint(constraint): constructing it disables the constraint in later CREATE TABLE output.
    return column, normalize(f"CONSTRAINT {constraint.name} CHECK ({constraint.sqltext})")


def test_cache_hit_migration_only_adds_the_column_and_its_check(monkeypatch):
    sql = upgrade_sql(monkeypatch, f"{REQUEST_TABLES_REVISION}:{CACHE_HIT_REVISION}")

    statements = re.findall(r"^([A-Z]+)\b", sql, re.MULTILINE)
    assert set(statements) == {"BEGIN", "ALTER", "UPDATE", "COMMIT"}
    assert re.findall(r"^ALTER TABLE (\w+) ADD (\w+)", sql, re.MULTILINE) == [
        ("requests", "COLUMN"),
        ("requests", "CONSTRAINT"),
    ]
    assert f"UPDATE alembic_version SET version_num='{CACHE_HIT_REVISION}'" in sql


def test_offline_downgrade_of_the_cache_hit_revision_drops_only_the_column_and_its_check(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", OFFLINE_URL)
    output = io.StringIO()

    command.downgrade(alembic_config(output), f"{CACHE_HIT_REVISION}:{REQUEST_TABLES_REVISION}", sql=True)

    sql = output.getvalue()
    assert re.findall(r"^ALTER TABLE (\w+) (DROP \w+ \w+)", sql, re.MULTILINE) == [
        ("requests", "DROP CONSTRAINT ck_requests_cache_hit_shape"),
        ("requests", "DROP COLUMN cache_hit"),
    ]
    assert "DROP TABLE" not in sql
    assert f"UPDATE alembic_version SET version_num='{REQUEST_TABLES_REVISION}'" in sql


def test_offline_downgrade_drops_only_the_request_tables(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", OFFLINE_URL)
    output = io.StringIO()
    head = ScriptDirectory.from_config(alembic_config()).get_current_head()

    command.downgrade(alembic_config(output), f"{head}:{INITIAL_REVISION}", sql=True)

    sql = output.getvalue()
    assert re.findall(r"DROP TABLE (\w+)", sql) == ["model_calls", "requests"]
    assert f"UPDATE alembic_version SET version_num='{INITIAL_REVISION}'" in sql


def normalize(sql: str) -> str:
    return re.sub(r"\s+", " ", sql)


@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_database_url_fails_with_variable_name(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("DATABASE_URL", raising=False)
    else:
        monkeypatch.setenv("DATABASE_URL", value)

    with pytest.raises(RuntimeError, match="set DATABASE_URL"):
        command.upgrade(alembic_config(io.StringIO()), "head", sql=True)
