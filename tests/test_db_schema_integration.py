"""Request tables against a real, local PostgreSQL (Task S2.3). Skipped by the default `pytest` run.

Runs in CI against the Postgres service container. It reuses the migration guard, so it refuses any
non-local host (for example Neon). Every row it writes is inside a transaction that is rolled back.
"""

import hashlib
import os
import uuid
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel

from app.db.models import ModelCallRecord, RequestRecord
from tests.test_migrations_integration import require_local_postgres_url

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parent.parent
REQUESTS = RequestRecord.__table__
MODEL_CALLS = ModelCallRecord.__table__


@pytest.fixture
def engine():
    database_url = require_local_postgres_url(os.environ)
    config = Config(str(ROOT / "alembic.ini"))
    config.attributes["configure_logger"] = False
    command.upgrade(config, "head")
    engine = sa.create_engine(database_url, poolclass=NullPool)
    yield engine
    engine.dispose()


def success_request(**overrides) -> dict:
    row = {
        "id": uuid.uuid4(),
        "prompt_hash": hashlib.sha256(b"prompt").hexdigest(),
        "prompt_chars": 6,
        "dry_run": False,
        "status": "success",
        "failure_reason": None,
        "final_model": "openai/gpt-5-nano",
        "final_tier": "cheap",
        "fell_back_to_cheap": False,
        "strong_reserved": False,
        "total_latency_ms": 1200,
        "config_version": "c" * 64,
        "est_cost_usd": 0.00004665,
        "baseline_cost_usd": 0.0021,
        "task_type": "code",
        "tier": "cheap",
        "reason": "EASY",
        "model_chain": ["openai/gpt-5-nano", "google/gemini-2.5-flash-lite"],
        "confidence": 0.99,
        "difficulty": 1.8,
        "needs_reasoning_prob": 0.12,
        "task_type_probs": {"code": 0.93, "general": 0.07},
        "difficulty_dist": {"0": 0.5, "1": 0.5},
        "jev_model": "jev",
        "jev_latency_ms": 142,
        "jev_input_tokens": 30,
        "jev_credits_charged": 1,
        "jev_error": None,
        "jev_http_status": None,
        "jev_error_cause": None,
    }
    return {**row, **overrides}


def failed_jev_error_request(**overrides) -> dict:
    row = success_request(
        status="failed",
        failure_reason="deadline",
        final_model=None,
        final_tier=None,
        fell_back_to_cheap=None,
        est_cost_usd=None,
        baseline_cost_usd=None,
        task_type="general",
        tier="strong",
        reason="JEV_ERROR",
        confidence=None,
        difficulty=None,
        needs_reasoning_prob=None,
        task_type_probs=None,
        difficulty_dist=None,
        jev_model=None,
        jev_latency_ms=None,
        jev_input_tokens=None,
        jev_credits_charged=None,
        jev_error="JevTimeoutError",
        jev_error_cause="ReadTimeout",
        strong_reserved=True,
    )
    return {**row, **overrides}


def cache_hit_request(**overrides) -> dict:
    row = success_request(
        cache_hit=True,
        est_cost_usd=None,
        baseline_cost_usd=None,
        jev_model=None,
        jev_latency_ms=None,
        jev_input_tokens=None,
        jev_credits_charged=None,
    )
    return {**row, **overrides}


def model_call(request_id, seq: int, **overrides) -> dict:
    row = {
        "request_id": request_id,
        "seq": seq,
        "tier": "cheap",
        "model": "openai/gpt-5-nano",
        "attempt_no": seq + 1,
        "status": "success",
        "latency_ms": 900,
        "http_status": 200,
        "input_tokens": 13,
        "output_tokens": 115,
        "reasoning_tokens": 64,
        "est_cost_usd": 0.00004665,
    }
    return {**row, **overrides}


def test_upgraded_schema_matches_the_models(engine):
    with engine.connect() as connection:
        diff = compare_metadata(MigrationContext.configure(connection), SQLModel.metadata)

    assert diff == []


def test_success_dry_run_and_failed_outcomes_insert_and_round_trip(engine):
    success = success_request()
    dry_run = success_request(
        status="dry_run",
        dry_run=True,
        final_model=None,
        final_tier=None,
        fell_back_to_cheap=None,
        est_cost_usd=None,
        baseline_cost_usd=None,
    )
    failed = failed_jev_error_request(task_type="chat")  # Jev's choice is stored as-is
    with engine.connect() as connection, connection.begin() as transaction:
        connection.execute(REQUESTS.insert(), [success, dry_run, failed])
        connection.execute(
            MODEL_CALLS.insert(),
            [
                model_call(
                    failed["id"],
                    0,
                    tier="strong",
                    status="timeout",
                    http_status=None,
                    input_tokens=None,
                    output_tokens=None,
                    reasoning_tokens=None,
                    est_cost_usd=None,
                ),
                model_call(
                    failed["id"],
                    1,
                    tier="strong",
                    status="deadline",
                    latency_ms=0,
                    http_status=None,
                    input_tokens=None,
                    output_tokens=None,
                    reasoning_tokens=None,
                    est_cost_usd=None,
                ),
                model_call(success["id"], 0),
            ],
        )
        stored = connection.execute(sa.select(REQUESTS).where(REQUESTS.c.id == success["id"])).mappings().one()
        statuses = (
            connection.execute(
                sa.select(MODEL_CALLS.c.status)
                .where(MODEL_CALLS.c.request_id == failed["id"])
                .order_by(MODEL_CALLS.c.seq)
            )
            .scalars()
            .all()
        )
        created_at = stored["created_at"]
        transaction.rollback()

    assert stored["model_chain"] == success["model_chain"]
    assert stored["task_type_probs"] == success["task_type_probs"]
    assert str(stored["est_cost_usd"]) == "0.00004665"  # NUMERIC(12,8) keeps the estimate exactly
    assert created_at is not None and created_at.tzinfo is not None
    assert statuses == ["timeout", "deadline"]


def test_rows_inserted_without_cache_hit_default_to_false(engine):
    # S2.14 step 4: rows written before the column existed, or by code that omits it, are not cache hits.
    routed, hit = success_request(), cache_hit_request()
    with engine.connect() as connection, connection.begin() as transaction:
        connection.execute(REQUESTS.insert(), [routed])
        connection.execute(REQUESTS.insert(), [hit])
        stored = dict(
            connection.execute(
                sa.select(REQUESTS.c.id, REQUESTS.c.cache_hit).where(REQUESTS.c.id.in_([routed["id"], hit["id"]]))
            ).all()
        )
        transaction.rollback()

    assert "cache_hit" not in routed
    assert stored == {routed["id"]: False, hit["id"]: True}


def test_deleting_a_request_deletes_its_model_calls(engine):
    request = success_request()
    with engine.connect() as connection, connection.begin() as transaction:
        connection.execute(REQUESTS.insert(), [request])
        connection.execute(MODEL_CALLS.insert(), [model_call(request["id"], 0), model_call(request["id"], 1)])
        connection.execute(REQUESTS.delete().where(REQUESTS.c.id == request["id"]))
        remaining = connection.execute(
            sa.select(sa.func.count()).select_from(MODEL_CALLS).where(MODEL_CALLS.c.request_id == request["id"])
        ).scalar_one()
        transaction.rollback()

    assert remaining == 0


def test_invalid_rows_are_rejected_by_constraints(engine):
    request_id = uuid.uuid4()
    request_cases = {
        "unknown status": success_request(status="partial"),
        "failure_reason without failure": success_request(failure_reason="deadline"),
        "failure without failure_reason": failed_jev_error_request(failure_reason=None),
        "unknown failure_reason": failed_jev_error_request(failure_reason="crashed"),
        "success without final model": success_request(final_model=None),
        "failed with final model": failed_jev_error_request(
            final_model="m", final_tier="cheap", fell_back_to_cheap=False
        ),
        "unknown final_tier": success_request(final_tier="premium"),
        "raw prompt instead of hash": success_request(prompt_hash="Write a function to check primes"[:64].ljust(64)),
        "uppercase hash": success_request(prompt_hash="A" * 64),
        "zero prompt_chars": success_request(prompt_chars=0),
        "negative latency": success_request(total_latency_ms=-1),
        "cost without baseline": success_request(baseline_cost_usd=None),
        "negative cost": success_request(est_cost_usd=-0.1),
        "cost on a failed request": failed_jev_error_request(est_cost_usd=0.1, baseline_cost_usd=0.2),
        "unknown tier": success_request(tier="premium"),
        "CACHE reason before the cache exists": success_request(reason="CACHE"),
        "missing decision": success_request(model_chain=None),
        # S2.14 step 4: a cache hit is a success with no dry run, reservation, cost or fallback.
        "cache hit with a cost": cache_hit_request(est_cost_usd=0.1, baseline_cost_usd=0.2),
        "cache hit on a dry run": cache_hit_request(
            status="dry_run", dry_run=True, final_model=None, final_tier=None, fell_back_to_cheap=None
        ),
        "cache hit on a failure": failed_jev_error_request(cache_hit=True, strong_reserved=False),
        "cache hit that reserved budget": cache_hit_request(strong_reserved=True),
        "cache hit that fell back": cache_hit_request(fell_back_to_cheap=True),
        "null cache_hit": success_request(cache_hit=None),
    }
    call_cases = {
        "unknown attempt status": model_call(request_id, 0, status="exploded"),
        "unknown attempt tier": model_call(request_id, 0, tier="premium"),
        "negative seq": model_call(request_id, -1),
        "attempt_no 0": model_call(request_id, 0, attempt_no=0),
        "negative attempt latency": model_call(request_id, 0, latency_ms=-5),
        "http_status out of range": model_call(request_id, 0, http_status=700),
        "negative tokens": model_call(request_id, 0, output_tokens=-1),
        "negative attempt cost": model_call(request_id, 0, est_cost_usd=-0.1),
        "unknown request": model_call(uuid.uuid4(), 0),
    }
    accepted = []
    with engine.connect() as connection:
        for name, row in request_cases.items():
            if inserts(connection, [(REQUESTS, row)]):
                accepted.append(name)
        for name, row in call_cases.items():
            if inserts(connection, [(REQUESTS, success_request(id=request_id)), (MODEL_CALLS, row)]):
                accepted.append(name)

    assert accepted == []


def test_duplicate_request_and_attempt_keys_are_rejected(engine):
    request = success_request()
    with engine.connect() as connection:
        duplicate_request = inserts(connection, [(REQUESTS, request), (REQUESTS, request)])
        duplicate_seq = inserts(
            connection,
            [
                (REQUESTS, request),
                (MODEL_CALLS, model_call(request["id"], 0)),
                (MODEL_CALLS, model_call(request["id"], 0)),
            ],
        )

    assert (duplicate_request, duplicate_seq) == (False, False)


def inserts(connection, rows) -> bool:
    """Inserts rows in one transaction that is always rolled back; False if the database rejects them."""
    transaction = connection.begin()
    try:
        for table, row in rows:
            connection.execute(table.insert(), [row])
    except sa.exc.IntegrityError:
        return False
    finally:
        transaction.rollback()
    return True
