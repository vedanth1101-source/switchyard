"""Alembic setup (Task 0.6), offline: no database connection is made."""

import io
import re
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from app.db.models import ModelCallRecord, RequestRecord

ROOT = Path(__file__).resolve().parent.parent
# Offline mode only renders SQL for this URL; nothing connects to it.
OFFLINE_URL = "postgresql+psycopg://user@localhost:5432/switchyard"
INITIAL_REVISION = "f969e942a969"


def alembic_config(output: io.StringIO | None = None) -> Config:
    config = Config(str(ROOT / "alembic.ini"), output_buffer=output)
    # Leave the test process's logging alone (env.py would otherwise apply alembic.ini's config).
    config.attributes["configure_logger"] = False
    return config


def test_revisions_form_one_chain_from_the_empty_initial_revision():
    # S2.3 (approved rewrite): the Sprint 0 revision stays the base; the request tables follow it.
    script = ScriptDirectory.from_config(alembic_config())
    revisions = list(script.walk_revisions())  # Head first

    assert [revision.revision for revision in revisions][-1] == INITIAL_REVISION
    assert revisions[-1].down_revision is None
    assert script.get_heads() == [revisions[0].revision]
    assert len(revisions) == 2
    assert revisions[0].down_revision == INITIAL_REVISION


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
    sql = upgrade_sql(monkeypatch, f"{INITIAL_REVISION}:head")

    # Leading keyword of every statement (column and constraint lines are indented, so they don't match).
    statements = re.findall(r"^([A-Z]+)\b", sql, re.MULTILINE)
    assert set(statements) == {"BEGIN", "CREATE", "UPDATE", "COMMIT"}
    assert re.findall(r"^UPDATE (\w+)", sql, re.MULTILINE) == ["alembic_version"]


def test_migration_ddl_matches_the_models(monkeypatch):
    # compare_metadata needs a live database (CI); the CREATE TABLE text, including every CHECK, can be
    # compared offline because both come from SQLAlchemy's PostgreSQL compiler.
    sql = upgrade_sql(monkeypatch, f"{INITIAL_REVISION}:head")

    for table in (RequestRecord.__table__, ModelCallRecord.__table__):
        expected = str(CreateTable(table).compile(dialect=postgresql.dialect())).strip()
        assert normalize(expected) in normalize(sql), table.name


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
