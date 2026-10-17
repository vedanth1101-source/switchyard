"""load_request and GET /api/requests/{request_id} against a real, local PostgreSQL (Task S2.13).

Skipped by the default `pytest` run; runs in CI against the Postgres service container. Reuses the
migration guard, so it refuses any non-local host (for example Neon), and deletes exactly the requests
each test created (model_calls cascade).
"""

import hashlib
import os
import uuid
from decimal import Decimal
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy.pool import NullPool

from app.core.llm_client import LLMMalformedResponseError
from app.db.models import RequestRecord
from app.db.repo import OutcomeRows, load_request, outcome_to_rows, save_outcome
from tests.test_migrations_integration import require_local_postgres_url
from tests.test_repo_mapping import CHEAP, EASY_JEV, HARD_JEV, PROMPT, STRONG, client_error, ok, run_route
from tests.test_route_persistence_integration import client_with_fake_services

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parent.parent
REQUESTS = RequestRecord.__table__
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


def saved(engine, created: list, outcome) -> OutcomeRows:
    rows = outcome_to_rows(outcome, prompt=PROMPT)
    created.append(rows.request["id"])
    save_outcome(engine, rows)
    return rows


def at_numeric_scale(row) -> dict:
    """A mapped row as PostgreSQL returns it: USD values rounded to NUMERIC(12, 8)."""
    values = dict(row)
    for name in ("est_cost_usd", "baseline_cost_usd"):
        if values.get(name) is not None:
            values[name] = values[name].quantize(CENT_MILLIONTHS)
    return values


def test_load_request_reads_back_a_success_with_its_calls_in_order(engine, created):
    rows = saved(engine, created, run_route(HARD_JEV, [client_error(), client_error(), ok(CHEAP[0])]))

    loaded = load_request(engine, rows.request["id"])

    request = dict(loaded.request)
    assert request.pop("created_at").tzinfo is not None
    assert request == at_numeric_scale(rows.request)
    assert [dict(call) for call in loaded.model_calls] == [at_numeric_scale(call) for call in rows.model_calls]
    assert [call["seq"] for call in loaded.model_calls] == [0, 1, 2]
    assert str(loaded.model_calls[2]["est_cost_usd"]) == "0.00004665"
    assert loaded.model_calls[2]["est_cost_usd"] == loaded.request["est_cost_usd"]


def test_load_request_reads_a_dry_run_without_calls(engine, created):
    rows = saved(engine, created, run_route(HARD_JEV, dry_run=True))

    loaded = load_request(engine, rows.request["id"])

    assert (loaded.request["status"], loaded.request["dry_run"], loaded.request["est_cost_usd"]) == (
        "dry_run",
        True,
        None,
    )
    assert loaded.model_calls == ()


def test_load_request_keeps_a_failed_requests_priced_attempt(engine, created):
    malformed = LLMMalformedResponseError("m: empty", status_code=200, input_tokens=13, output_tokens=1024)
    rows = saved(engine, created, run_route(EASY_JEV, [malformed, client_error()]))

    loaded = load_request(engine, rows.request["id"])

    assert (loaded.request["status"], loaded.request["est_cost_usd"], loaded.request["baseline_cost_usd"]) == (
        "failed",
        None,
        None,
    )
    assert [(c["status"], c["est_cost_usd"]) for c in loaded.model_calls] == [
        ("malformed", Decimal("0.00041025")),
        ("client_error", None),
    ]


def test_load_request_orders_calls_by_seq_not_insertion_order(engine, created):
    rows = outcome_to_rows(run_route(HARD_JEV, [client_error(), client_error(), ok(CHEAP[0])]), prompt=PROMPT)
    created.append(rows.request["id"])
    save_outcome(engine, OutcomeRows(request=rows.request, model_calls=tuple(reversed(rows.model_calls))))

    loaded = load_request(engine, rows.request["id"])

    assert [(c["seq"], c["model"]) for c in loaded.model_calls] == [(0, STRONG[0]), (1, STRONG[1]), (2, CHEAP[0])]


def test_load_request_returns_none_for_an_unknown_id(engine):
    assert load_request(engine, uuid.uuid4()) is None


def test_get_after_post_returns_the_saved_request_end_to_end(engine, created):
    client = client_with_fake_services([client_error(), ok(CHEAP[1])])  # Real recorder, reader and database
    posted = client.post("/api/route", json={"prompt": PROMPT})
    request_id = posted.json()["request_id"]
    created.append(uuid.UUID(request_id))

    response = client.get(f"/api/requests/{request_id}")

    body = response.json()
    assert (posted.status_code, response.status_code) == (200, 200)
    assert response.headers["X-Request-ID"] != request_id  # The GET is its own request
    assert body["request"]["id"] == request_id
    assert (body["request"]["status"], body["request"]["final_model"]) == ("success", CHEAP[1])
    assert body["request"]["prompt_chars"] == len(PROMPT)
    assert body["request"]["est_cost_usd"] == pytest.approx(posted.json()["cost"]["est_cost_usd"])
    assert body["decision"]["model_chain"] == posted.json()["decision"]["model_chain"]
    assert [(a["seq"], a["model"], a["status"]) for a in body["attempts"]] == [
        (0, CHEAP[0], "client_error"),
        (1, CHEAP[1], "success"),
    ]
    assert body["attempts"][0]["est_cost_usd"] is None
    assert body["attempts"][1]["est_cost_usd"] == body["request"]["est_cost_usd"]
    assert "prompt_hash" not in response.text
    assert hashlib.sha256(PROMPT.encode("utf-8")).hexdigest() not in response.text
    assert "PROMPT-MARKER" not in response.text

    missing = client.get(f"/api/requests/{uuid.uuid4()}")
    assert (missing.status_code, missing.json()["error"]) == (404, "not_found")
