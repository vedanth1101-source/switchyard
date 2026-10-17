"""SQLModel tables for routed requests (Task S2.3), checked offline: no database connection is made."""

import re
from typing import get_args

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import psycopg
from sqlalchemy.schema import CreateTable
from sqlmodel import SQLModel

from app.core.executor import AttemptStatus, ExhaustionReason
from app.core.router import RouteStatus
from app.db import models
from app.db.models import ModelCallRecord, RequestRecord

REQUESTS = RequestRecord.__table__
MODEL_CALLS = ModelCallRecord.__table__

REQUEST_COLUMNS = {
    # name: (type, nullable)
    "id": (sa.Uuid, False),
    "created_at": (sa.DateTime, False),
    "prompt_hash": (sa.CHAR, False),
    "prompt_chars": (sa.Integer, False),
    "dry_run": (sa.Boolean, False),
    "status": (sa.Text, False),
    "failure_reason": (sa.Text, True),
    "final_model": (sa.Text, True),
    "final_tier": (sa.Text, True),
    "fell_back_to_cheap": (sa.Boolean, True),
    "strong_reserved": (sa.Boolean, False),
    "total_latency_ms": (sa.Integer, False),
    "config_version": (sa.Text, False),
    "est_cost_usd": (sa.Numeric, True),
    "baseline_cost_usd": (sa.Numeric, True),
    "task_type": (sa.Text, False),
    "tier": (sa.Text, False),
    "reason": (sa.Text, False),
    "model_chain": (postgresql.JSONB, False),
    "confidence": (sa.Double, True),
    "difficulty": (sa.Double, True),
    "needs_reasoning_prob": (sa.Double, True),
    "task_type_probs": (postgresql.JSONB, True),
    "difficulty_dist": (postgresql.JSONB, True),
    "jev_model": (sa.Text, True),
    "jev_latency_ms": (sa.Integer, True),
    "jev_input_tokens": (sa.Integer, True),
    "jev_credits_charged": (sa.Integer, True),
    "jev_error": (sa.Text, True),
    "jev_http_status": (sa.Integer, True),
    "jev_error_cause": (sa.Text, True),
    "cache_hit": (sa.Boolean, False),  # S2.14 step 4
}

MODEL_CALL_COLUMNS = {
    "request_id": (sa.Uuid, False),
    "seq": (sa.Integer, False),
    "tier": (sa.Text, False),
    "model": (sa.Text, False),
    "attempt_no": (sa.Integer, False),
    "status": (sa.Text, False),
    "latency_ms": (sa.Integer, False),
    "http_status": (sa.Integer, True),
    "input_tokens": (sa.Integer, True),
    "output_tokens": (sa.Integer, True),
    "reasoning_tokens": (sa.Integer, True),
    "est_cost_usd": (sa.Numeric, True),
}


def columns(table: sa.Table) -> dict:
    return {column.name: column for column in table.columns}


def ddl(table: sa.Table) -> str:
    return str(CreateTable(table).compile(dialect=postgresql.dialect()))


def check_names(table: sa.Table) -> set[str]:
    return {constraint.name for constraint in table.constraints if isinstance(constraint, sa.CheckConstraint)}


def test_metadata_holds_exactly_the_two_request_tables():
    assert set(SQLModel.metadata.tables) == {"requests", "model_calls"}


def test_requests_columns_types_and_nullability():
    actual = columns(REQUESTS)

    assert set(actual) == set(REQUEST_COLUMNS)
    for name, (type_, nullable) in REQUEST_COLUMNS.items():
        assert isinstance(actual[name].type, type_), name
        assert actual[name].nullable is nullable, name


def test_model_calls_columns_types_and_nullability():
    actual = columns(MODEL_CALLS)

    assert set(actual) == set(MODEL_CALL_COLUMNS)
    for name, (type_, nullable) in MODEL_CALL_COLUMNS.items():
        assert isinstance(actual[name].type, type_), name
        assert actual[name].nullable is nullable, name


def test_exact_type_parameters():
    request_columns, call_columns = columns(REQUESTS), columns(MODEL_CALLS)

    assert request_columns["created_at"].type.timezone is True
    assert request_columns["prompt_hash"].type.length == 64
    for column in (request_columns["est_cost_usd"], request_columns["baseline_cost_usd"], call_columns["est_cost_usd"]):
        assert (column.type.precision, column.type.scale) == (12, 8)


JSONB_COLUMNS = ("model_chain", "task_type_probs", "difficulty_dist")


def test_jsonb_columns_store_python_none_as_sql_null():
    # SQLAlchemy's default (none_as_null=False) writes None as the JSON value `null`, which satisfies
    # NOT NULL and is not found by IS NULL (CI run 38345880897).
    for name in JSONB_COLUMNS:
        assert columns(REQUESTS)[name].type.none_as_null is True, name


def test_jsonb_none_reaches_the_driver_as_null_not_as_a_json_value():
    # The psycopg dialect's bind processor is what turns None into Jsonb(None) (JSON null) by default.
    dialect = psycopg.dialect()
    for name in JSONB_COLUMNS:
        column_type = columns(REQUESTS)[name].type
        assert column_type.should_evaluate_none is False, name
        assert column_type.dialect_impl(dialect).bind_processor(dialect)(None) is None, name


def test_created_at_defaults_to_the_database_clock():
    created_at = columns(REQUESTS)["created_at"]

    assert created_at.server_default is not None
    assert "now()" in ddl(REQUESTS)
    assert created_at.default is None  # The database sets it, not the application


def test_no_prompt_response_or_client_ip_columns_exist():
    # D1/D2: only a hash and a length of the prompt are stored; no answer text, no client IP.
    names = set(columns(REQUESTS)) | set(columns(MODEL_CALLS))

    assert not {name for name in names if re.search(r"prompt_text|response|answer|_text$|(^|_)ip(_|$)", name)}


def test_primary_and_foreign_keys():
    assert [column.name for column in REQUESTS.primary_key.columns] == ["id"]
    assert [column.name for column in MODEL_CALLS.primary_key.columns] == ["request_id", "seq"]
    (foreign_key,) = MODEL_CALLS.foreign_keys
    assert foreign_key.parent.name == "request_id"
    assert foreign_key.target_fullname == "requests.id"
    assert foreign_key.ondelete == "CASCADE"
    assert not REQUESTS.foreign_keys


def test_no_secondary_indexes_or_unique_constraints():
    for table in (REQUESTS, MODEL_CALLS):
        assert not table.indexes
        assert not [constraint for constraint in table.constraints if isinstance(constraint, sa.UniqueConstraint)]


def test_constraint_names_follow_the_naming_convention():
    assert REQUESTS.primary_key.name == "pk_requests"
    assert MODEL_CALLS.primary_key.name == "pk_model_calls"
    (foreign_key,) = MODEL_CALLS.foreign_key_constraints
    assert foreign_key.name == "fk_model_calls_request_id_requests"
    assert all(name.startswith("ck_requests_") for name in check_names(REQUESTS))
    assert all(name.startswith("ck_model_calls_") for name in check_names(MODEL_CALLS))


def test_check_constraints_present():
    assert check_names(REQUESTS) == {
        "ck_requests_prompt_hash_hex",
        "ck_requests_prompt_chars_positive",
        "ck_requests_status",
        "ck_requests_failure_reason",
        "ck_requests_failure_reason_iff_failed",
        "ck_requests_final_tier",
        "ck_requests_final_fields_iff_success",
        "ck_requests_total_latency_ms_non_negative",
        "ck_requests_costs_together",
        "ck_requests_costs_only_on_success",
        "ck_requests_costs_non_negative",
        "ck_requests_tier",
        "ck_requests_reason",
        "ck_requests_cache_hit_shape",  # S2.14 step 4
    }
    assert check_names(MODEL_CALLS) == {
        "ck_model_calls_seq_non_negative",
        "ck_model_calls_tier",
        "ck_model_calls_attempt_no_positive",
        "ck_model_calls_status",
        "ck_model_calls_latency_ms_non_negative",
        "ck_model_calls_http_status_range",
        "ck_model_calls_tokens_non_negative",
        "ck_model_calls_est_cost_non_negative",
    }


def test_value_sets_match_the_routing_code():
    # A drift here means a routed outcome could be rejected by the database.
    assert set(models.ROUTE_STATUSES) == set(get_args(RouteStatus))
    assert set(models.FAILURE_REASONS) == set(get_args(ExhaustionReason))
    assert set(models.ATTEMPT_STATUSES) == set(get_args(AttemptStatus))
    assert set(models.TIERS) == {"cheap", "strong"}
    # CACHE is documented in policy.RouteDecision but not produced until the response cache exists.
    assert set(models.ROUTE_REASONS) == {"EASY", "HARD", "LOW_CONFIDENCE", "JEV_ERROR", "BUDGET"}


def test_value_set_checks_list_every_allowed_value():
    text = ddl(REQUESTS) + ddl(MODEL_CALLS)

    for values in (
        models.ROUTE_STATUSES,
        models.FAILURE_REASONS,
        models.TIERS,
        models.ROUTE_REASONS,
        models.ATTEMPT_STATUSES,
    ):
        assert f"IN ({', '.join(repr(value) for value in values)})" in text


def test_task_type_is_not_constrained():
    # The classifier passes through whatever task_type Jev chooses; the database must accept it.
    assert not [name for name in check_names(REQUESTS) if "task_type" in name]


def test_cache_hit_defaults_to_false_in_the_database():
    cache_hit = columns(REQUESTS)["cache_hit"]

    assert cache_hit.server_default is not None
    assert "cache_hit BOOLEAN DEFAULT false NOT NULL" in ddl(REQUESTS)
    assert list(columns(REQUESTS))[-1] == "cache_hit"  # Last, as ALTER TABLE ... ADD COLUMN places it


def test_cache_hit_rows_are_constrained_to_the_hit_shape():
    assert (
        "CONSTRAINT ck_requests_cache_hit_shape CHECK (NOT cache_hit OR (status = 'success' AND NOT dry_run"
        " AND NOT strong_reserved AND est_cost_usd IS NULL AND NOT fell_back_to_cheap))"
    ) in re.sub(r"\s+", " ", ddl(REQUESTS))
