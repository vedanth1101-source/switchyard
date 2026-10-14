"""save_outcome against a real, local PostgreSQL (Task S2.4). Skipped by the default `pytest` run.

Runs in CI against the Postgres service container; reuses the migration guard, so it refuses any
non-local host (for example Neon). save_outcome commits, so each test deletes exactly the requests it
created (their model_calls cascade).
"""

import os
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy.pool import NullPool

from app.db.models import ModelCallRecord, RequestRecord
from app.db.repo import OutcomeRows, outcome_to_rows, save_outcome
from tests.test_migrations_integration import require_local_postgres_url
from tests.test_repo_mapping import CHEAP, EASY_JEV, HARD_JEV, PROMPT, all_outcomes, client_error, ok, run_route

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parent.parent
REQUESTS = RequestRecord.__table__
MODEL_CALLS = ModelCallRecord.__table__
CENT_MILLIONTHS = Decimal("1e-8")  # NUMERIC(12, 8)


@pytest.fixture
def engine():
    database_url = require_local_postgres_url(os.environ)
    config = Config(str(ROOT / "alembic.ini"))
    config.attributes["configure_logger"] = False
    command.upgrade(config, "head")
    engine = sa.create_engine(database_url, poolclass=NullPool)
    yield engine
    engine.dispose()


@pytest.fixture
def created(engine):
    """Request ids a test saved; deleted afterwards (model_calls cascade)."""
    ids: list = []
    yield ids
    if ids:
        with engine.begin() as connection:
            connection.execute(REQUESTS.delete().where(REQUESTS.c.id.in_(ids)))


def stored_request(engine, request_id):
    with engine.connect() as connection:
        return connection.execute(sa.select(REQUESTS).where(REQUESTS.c.id == request_id)).mappings().one_or_none()


def stored_calls(engine, request_id) -> list[dict]:
    with engine.connect() as connection:
        query = sa.select(MODEL_CALLS).where(MODEL_CALLS.c.request_id == request_id).order_by(MODEL_CALLS.c.seq)
        return [dict(row) for row in connection.execute(query).mappings()]


def assert_stored_exactly(engine, rows: OutcomeRows) -> None:
    """The saved request and calls read back equal to the mapped rows (costs at NUMERIC(12, 8))."""
    stored = dict(stored_request(engine, rows.request["id"]))
    assert stored.pop("created_at").tzinfo is not None
    expected = dict(rows.request)
    for name in ("est_cost_usd", "baseline_cost_usd"):
        if expected[name] is not None:
            expected[name] = expected[name].quantize(CENT_MILLIONTHS)
    assert stored == expected
    expected_calls = [dict(call) for call in rows.model_calls]
    for call in expected_calls:
        if call["est_cost_usd"] is not None:
            call["est_cost_usd"] = call["est_cost_usd"].quantize(CENT_MILLIONTHS)
    assert stored_calls(engine, rows.request["id"]) == expected_calls


def test_save_outcome_persists_the_request_and_its_attempts_in_order(engine, created):
    rows = outcome_to_rows(run_route(HARD_JEV, [client_error(), client_error(), ok(CHEAP[0])]), prompt=PROMPT)
    created.append(rows.request["id"])

    save_outcome(engine, rows)

    assert_stored_exactly(engine, rows)
    # S2.11: unpriced attempts are SQL NULL; the success is priced at NUMERIC(12, 8), equal to the request cost
    calls = stored_calls(engine, rows.request["id"])
    assert [(call["seq"], call["status"]) for call in calls] == [
        (0, "client_error"),
        (1, "client_error"),
        (2, "success"),
    ]
    assert [call["est_cost_usd"] for call in calls[:2]] == [None, None]
    assert str(calls[2]["est_cost_usd"]) == "0.00004665"  # 13 x 0.05 + 115 x 0.40 per million, float noise rounded
    assert calls[2]["est_cost_usd"] == stored_request(engine, rows.request["id"])["est_cost_usd"]


def test_every_outcome_shape_route_returns_is_accepted(engine, created):
    for outcome in all_outcomes():
        rows = outcome_to_rows(outcome, prompt=PROMPT)
        created.append(rows.request["id"])

        save_outcome(engine, rows)

        assert_stored_exactly(engine, rows)
        assert len(rows.model_calls) == len(outcome.attempts)


def test_jev_failure_maps_absent_jev_data_to_sql_null(engine, created):
    rows = outcome_to_rows(all_outcomes()[3], prompt=PROMPT)  # JevTimeoutError outcome
    created.append(rows.request["id"])
    save_outcome(engine, rows)

    with engine.connect() as connection:
        nulls = connection.execute(
            sa.select(REQUESTS.c.task_type_probs.is_(None), REQUESTS.c.difficulty_dist.is_(None)).where(
                REQUESTS.c.id == rows.request["id"]
            )
        ).one()

    assert tuple(nulls) == (True, True)  # SQL NULL, not the JSON value null


def test_a_rejected_attempt_rolls_back_the_whole_outcome(engine, created):
    rows = outcome_to_rows(run_route(EASY_JEV, [client_error(), ok(CHEAP[1])]), prompt=PROMPT)
    broken = OutcomeRows(
        request=rows.request,
        model_calls=(rows.model_calls[0], MappingProxyType({**rows.model_calls[1], "status": "exploded"})),
    )
    created.append(rows.request["id"])

    with pytest.raises(sa.exc.IntegrityError) as error:
        save_outcome(engine, broken)

    assert error.value.orig.diag.constraint_name == "ck_model_calls_status"  # The bad attempt, nothing else
    assert stored_request(engine, rows.request["id"]) is None
    assert stored_calls(engine, rows.request["id"]) == []


def test_saving_the_same_outcome_twice_fails_and_keeps_exactly_one_copy(engine, created):
    rows = outcome_to_rows(run_route(EASY_JEV, [client_error(), ok(CHEAP[1])]), prompt=PROMPT)
    created.append(rows.request["id"])
    save_outcome(engine, rows)

    with pytest.raises(sa.exc.IntegrityError) as error:
        save_outcome(engine, rows)

    assert error.value.orig.diag.constraint_name == "pk_requests"  # Refused at the request row
    assert stored_request(engine, rows.request["id"]) is not None
    assert len(stored_calls(engine, rows.request["id"])) == 2
