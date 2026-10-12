"""Alembic setup (Task 0.6), offline: no database connection is made."""

import io
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

ROOT = Path(__file__).resolve().parent.parent
# Offline mode only renders SQL for this URL; nothing connects to it.
OFFLINE_URL = "postgresql+psycopg://user@localhost:5432/switchyard"


def alembic_config(output: io.StringIO | None = None) -> Config:
    config = Config(str(ROOT / "alembic.ini"), output_buffer=output)
    # Leave the test process's logging alone (env.py would otherwise apply alembic.ini's config).
    config.attributes["configure_logger"] = False
    return config


def test_single_initial_revision_is_the_head():
    script = ScriptDirectory.from_config(alembic_config())
    revisions = list(script.walk_revisions())

    assert len(revisions) == 1
    assert script.get_heads() == [revisions[0].revision]
    assert revisions[0].down_revision is None


def test_offline_upgrade_only_creates_the_version_table(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", OFFLINE_URL)
    output = io.StringIO()

    command.upgrade(alembic_config(output), "head", sql=True)

    sql = output.getvalue()
    head = ScriptDirectory.from_config(alembic_config()).get_current_head()
    assert "CREATE TABLE alembic_version" in sql
    assert f"INSERT INTO alembic_version (version_num) VALUES ('{head}')" in sql
    assert sql.count("CREATE TABLE") == 1


@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_database_url_fails_with_variable_name(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("DATABASE_URL", raising=False)
    else:
        monkeypatch.setenv("DATABASE_URL", value)

    with pytest.raises(RuntimeError, match="set DATABASE_URL"):
        command.upgrade(alembic_config(io.StringIO()), "head", sql=True)
