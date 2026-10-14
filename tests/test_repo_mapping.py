"""Outcome -> row mapping (Task S2.4), offline. Outcomes come from the real route() with fakes."""

import asyncio
import copy
import dataclasses
import hashlib
import json
import uuid
from decimal import Decimal
from pathlib import Path

import pytest

from app.core.jev_client import JevResponseError, JevTimeoutError
from app.core.llm_client import (
    LLMAuthError,
    LLMClientError,
    LLMMalformedResponseError,
    LLMRateLimitError,
    ModelResult,
)
from app.core.model_config import load_model_config
from app.core.router import RouteOutcome, RouterDeps, route
from app.db import models
from app.db.models import ModelCallRecord, RequestRecord
from app.db.repo import OutcomeRows, outcome_to_rows
from tests.fakes import FakeBudget, FakeClock, FakeJev, FakeLLM, RecordingSleep

FIXTURES = Path(__file__).parent / "fixtures"
PROMPT = "Write a function to check primes PROMPT-MARKER-7f3a"
SECRET = "SECRET-MARKER-https://user:pw@internal.example"
CONFIG = load_model_config()
CHEAP = CONFIG.routing.get_chain("general", "cheap")
STRONG = CONFIG.routing.get_chain("general", "strong")
EASY_JEV = json.loads((FIXTURES / "jev_response_success.json").read_text(encoding="utf-8"))
HARD_JEV = copy.deepcopy(EASY_JEV)
HARD_JEV["data"]["answers"]["difficulty"]["score"] = 3.0
HARD_JEV["data"]["answers"]["difficulty"]["probabilities"] = {"0": 0.0, "1": 0.0, "2": 0.0, "3": 1.0, "4": 0.0}

REQUEST_COLUMNS = {column.name for column in RequestRecord.__table__.columns} - {"created_at"}
MODEL_CALL_COLUMNS = {column.name for column in ModelCallRecord.__table__.columns}


def ok(model: str, text: str = "ANSWER-MARKER-9c1e") -> ModelResult:
    return ModelResult(
        text=text,
        model=model,
        input_tokens=13,
        output_tokens=115,
        latency_ms=900,
        finish_reason="stop",
        http_status=200,
        reasoning_tokens=64,
    )


def client_error() -> LLMClientError:
    return LLMClientError(f"m: HTTP 400: {SECRET}", status_code=400)


def malformed_with_usage() -> LLMMalformedResponseError:
    return LLMMalformedResponseError(f"m: empty {SECRET}", status_code=200, input_tokens=13, output_tokens=1024)


def run_route(jev=EASY_JEV, script=(), *, budget=None, dry_run=False, jev_seconds=0.0, prompt=PROMPT) -> RouteOutcome:
    clock = FakeClock()
    deps = RouterDeps(
        config=CONFIG,
        jev=FakeJev(jev, clock, jev_seconds),
        budget=budget or FakeBudget(),
        llm=FakeLLM(list(script), clock),
        request_budget_seconds=50.0,
        clock=clock,
        sleep=RecordingSleep(clock),
        new_request_id=lambda: str(uuid.uuid4()),
    )
    return asyncio.run(route(prompt, dry_run=dry_run, deps=deps))


def all_values(rows: OutcomeRows) -> str:
    return repr([dict(rows.request), *(dict(call) for call in rows.model_calls)])


# --- Outcome shapes -------------------------------------------------------------------------


def test_success_maps_outcome_decision_jev_and_cost():
    outcome = run_route(EASY_JEV, [ok(CHEAP[0])])

    rows = outcome_to_rows(outcome, prompt=PROMPT)

    request = rows.request
    assert request["id"] == uuid.UUID(outcome.request_id)
    assert request["prompt_hash"] == hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
    assert request["prompt_chars"] == len(PROMPT)
    assert (request["dry_run"], request["status"], request["failure_reason"]) == (False, "success", None)
    assert (request["final_model"], request["final_tier"], request["fell_back_to_cheap"]) == (CHEAP[0], "cheap", False)
    assert request["strong_reserved"] is False
    assert request["total_latency_ms"] == outcome.total_latency_ms
    assert request["config_version"] == CONFIG.config_version
    assert request["est_cost_usd"] == Decimal(str(outcome.cost.est_cost_usd))
    assert request["baseline_cost_usd"] == Decimal(str(outcome.cost.baseline_cost_usd))
    assert (request["task_type"], request["tier"], request["reason"]) == ("code", "cheap", "EASY")
    assert request["model_chain"] == list(CHEAP)
    assert (request["confidence"], request["difficulty"], request["needs_reasoning_prob"]) == (0.99, 1.01, 0.19)
    assert request["task_type_probs"] == outcome.jev.task_type_probs
    assert request["difficulty_dist"] == outcome.jev.difficulty_dist
    assert (request["jev_model"], request["jev_latency_ms"], request["jev_input_tokens"]) == ("jev-1.13.0", 1434, 446)
    assert request["jev_credits_charged"] == 1
    assert (request["jev_error"], request["jev_http_status"], request["jev_error_cause"]) == (None, None, None)
    assert [dict(call) for call in rows.model_calls] == [
        {
            "request_id": uuid.UUID(outcome.request_id),
            "seq": 0,
            "tier": "cheap",
            "model": CHEAP[0],
            "attempt_no": 1,
            "status": "success",
            "latency_ms": 0,
            "http_status": 200,
            "input_tokens": 13,
            "output_tokens": 115,
            "reasoning_tokens": 64,
            "est_cost_usd": Decimal(str(outcome.attempt_costs[0])),  # S2.11: the same call as the request cost
        }
    ]
    assert rows.model_calls[0]["est_cost_usd"] == request["est_cost_usd"]


def test_dry_run_has_no_calls_no_final_model_and_no_cost():
    outcome = run_route(EASY_JEV, dry_run=True)

    rows = outcome_to_rows(outcome, prompt=PROMPT)

    assert (rows.request["dry_run"], rows.request["status"], rows.request["failure_reason"]) == (True, "dry_run", None)
    assert (rows.request["final_model"], rows.request["final_tier"], rows.request["fell_back_to_cheap"]) == (
        None,
        None,
        None,
    )
    assert (rows.request["est_cost_usd"], rows.request["baseline_cost_usd"]) == (None, None)
    assert rows.request["strong_reserved"] is False
    assert rows.model_calls == ()


def test_budget_denial_is_a_cheap_budget_decision_without_reservation():
    outcome = run_route(HARD_JEV, [ok(CHEAP[0])], budget=FakeBudget(available=False))

    rows = outcome_to_rows(outcome, prompt=PROMPT)

    assert (rows.request["tier"], rows.request["reason"], rows.request["strong_reserved"]) == ("cheap", "BUDGET", False)
    assert rows.request["status"] == "success"
    assert rows.request["model_chain"] == list(CHEAP)  # The cheap chain, not the strong one
    assert (rows.request["final_tier"], rows.request["fell_back_to_cheap"]) == ("cheap", False)


def test_strong_dry_run_keeps_the_strong_decision_but_reserves_nothing():
    outcome = run_route(HARD_JEV, dry_run=True)

    rows = outcome_to_rows(outcome, prompt=PROMPT)

    assert (rows.request["tier"], rows.request["reason"], rows.request["model_chain"]) == (
        "strong",
        "HARD",
        list(STRONG),
    )
    assert (rows.request["dry_run"], rows.request["status"], rows.request["strong_reserved"]) == (
        True,
        "dry_run",
        False,
    )
    assert rows.model_calls == ()


def test_jev_http_error_keeps_its_status_and_class_but_not_its_message():
    outcome = run_route(
        JevResponseError(f"JevStation server error (HTTP 500) {SECRET}", status_code=500), [ok(STRONG[0])]
    )

    rows = outcome_to_rows(outcome, prompt=PROMPT)

    assert (rows.request["jev_error"], rows.request["jev_http_status"], rows.request["jev_error_cause"]) == (
        "JevResponseError",
        500,
        None,
    )
    assert rows.request["reason"] == "JEV_ERROR"
    assert SECRET not in all_values(rows)


def test_prompt_chars_counts_characters_and_the_hash_covers_utf8_bytes():
    prompt = "Résumé ✓ 日本語 PROMPT-MARKER"
    outcome = run_route(EASY_JEV, dry_run=True, prompt=prompt)

    rows = outcome_to_rows(outcome, prompt=prompt)

    assert rows.request["prompt_chars"] == len(prompt) == 26  # Characters, as the API's 8,000 limit counts
    assert len(prompt.encode("utf-8")) > len(prompt)
    assert rows.request["prompt_hash"] == hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    assert "PROMPT-MARKER" not in all_values(rows)


def test_jev_failure_keeps_only_the_error_class_and_nulls_jev_fields():
    error = JevTimeoutError(f"timed out: {SECRET}", cause_type="ReadTimeout")
    outcome = run_route(error, [ok(STRONG[0])])

    rows = outcome_to_rows(outcome, prompt=PROMPT)

    request = rows.request
    assert (request["task_type"], request["tier"], request["reason"]) == ("general", "strong", "JEV_ERROR")
    assert (request["jev_error"], request["jev_http_status"], request["jev_error_cause"]) == (
        "JevTimeoutError",
        None,
        "ReadTimeout",
    )
    for name in ("confidence", "difficulty", "needs_reasoning_prob", "task_type_probs", "difficulty_dist"):
        assert request[name] is None, name
    for name in ("jev_model", "jev_latency_ms", "jev_input_tokens", "jev_credits_charged"):
        assert request[name] is None, name
    assert request["strong_reserved"] is True
    assert SECRET not in all_values(rows)


def test_strong_failure_falls_back_to_cheap_with_attempts_in_order():
    outcome = run_route(HARD_JEV, [client_error(), client_error(), ok(CHEAP[0])])

    rows = outcome_to_rows(outcome, prompt=PROMPT)

    assert (rows.request["final_tier"], rows.request["fell_back_to_cheap"]) == ("cheap", True)
    assert (rows.request["tier"], rows.request["reason"], rows.request["strong_reserved"]) == ("strong", "HARD", True)
    assert [(c["seq"], c["tier"], c["model"], c["attempt_no"], c["status"]) for c in rows.model_calls] == [
        (0, "strong", STRONG[0], 1, "client_error"),
        (1, "strong", STRONG[1], 1, "client_error"),
        (2, "cheap", CHEAP[0], 1, "success"),
    ]
    assert {call["request_id"] for call in rows.model_calls} == {rows.request["id"]}
    failed = rows.model_calls[0]
    assert (failed["http_status"], failed["input_tokens"], failed["output_tokens"]) == (400, None, None)


def test_retries_keep_their_attempt_numbers_and_sequence():
    rate_limited = LLMRateLimitError("m: HTTP 429", status_code=429)
    outcome = run_route(EASY_JEV, [rate_limited, ok(CHEAP[0])])

    rows = outcome_to_rows(outcome, prompt=PROMPT)

    assert [(c["seq"], c["model"], c["attempt_no"], c["status"]) for c in rows.model_calls] == [
        (0, CHEAP[0], 1, "rate_limited"),
        (1, CHEAP[0], 2, "success"),
    ]


@pytest.mark.parametrize(
    ("script", "jev_seconds", "reason", "statuses"),
    [
        ([client_error(), client_error()], 0.0, "all_failed", ["client_error", "client_error"]),
        ([ok(CHEAP[0])], 50.0, "deadline", ["deadline"]),
        ([LLMAuthError("m: HTTP 401", status_code=401)], 0.0, "fatal", ["auth_error"]),
    ],
    ids=["all-failed", "deadline", "fatal"],
)
def test_failed_outcomes_keep_reason_and_attempts_without_final_model_or_cost(script, jev_seconds, reason, statuses):
    outcome = run_route(EASY_JEV, script, jev_seconds=jev_seconds)

    rows = outcome_to_rows(outcome, prompt=PROMPT)

    assert (rows.request["status"], rows.request["failure_reason"]) == ("failed", reason)
    assert (rows.request["final_model"], rows.request["final_tier"], rows.request["fell_back_to_cheap"]) == (
        None,
        None,
        None,
    )
    assert (rows.request["est_cost_usd"], rows.request["baseline_cost_usd"]) == (None, None)
    assert [call["status"] for call in rows.model_calls] == statuses


def test_deadline_entry_has_zero_latency_and_no_http_status():
    rows = outcome_to_rows(run_route(EASY_JEV, [ok(CHEAP[0])], jev_seconds=50.0), prompt=PROMPT)

    (deadline,) = rows.model_calls
    assert (deadline["latency_ms"], deadline["http_status"], deadline["input_tokens"]) == (0, None, None)


def test_malformed_attempt_keeps_its_billed_usage_but_no_reasoning_tokens():
    malformed = LLMMalformedResponseError("m: empty", status_code=200, input_tokens=13, output_tokens=1024)
    outcome = run_route(EASY_JEV, [malformed, ok(CHEAP[1])])

    rows = outcome_to_rows(outcome, prompt=PROMPT)

    first = rows.model_calls[0]
    assert (first["status"], first["input_tokens"], first["output_tokens"], first["reasoning_tokens"]) == (
        "malformed",
        13,
        1024,
        None,
    )
    assert first["est_cost_usd"] == Decimal(str(outcome.attempt_costs[0]))  # Priced: it reported usage (S2.11)
    assert first["est_cost_usd"] > 0


def test_unknown_success_cost_stays_null_not_zero():
    outcome = dataclasses.replace(run_route(EASY_JEV, [ok(CHEAP[0])]), cost=None)  # e.g. after a CostError

    rows = outcome_to_rows(outcome, prompt=PROMPT)

    assert (rows.request["est_cost_usd"], rows.request["baseline_cost_usd"]) == (None, None)


# --- Per-attempt cost (S2.11) -----------------------------------------------------------------


def test_attempt_costs_map_in_attempt_order_and_unknown_costs_stay_null():
    outcome = run_route(HARD_JEV, [malformed_with_usage(), client_error(), ok(CHEAP[0])])

    rows = outcome_to_rows(outcome, prompt=PROMPT)

    assert [(c["seq"], c["model"], c["status"]) for c in rows.model_calls] == [
        (0, STRONG[0], "malformed"),
        (1, STRONG[1], "client_error"),
        (2, CHEAP[0], "success"),
    ]
    assert [c["est_cost_usd"] for c in rows.model_calls] == [
        Decimal(str(outcome.attempt_costs[0])),
        None,
        Decimal(str(outcome.attempt_costs[2])),
    ]


def test_attempt_costs_are_taken_from_the_outcome_by_position():
    # The mapping is pure: it copies outcome.attempt_costs by position and never prices anything itself.
    outcome = run_route(HARD_JEV, [malformed_with_usage(), client_error(), ok(CHEAP[0])])
    outcome = dataclasses.replace(outcome, attempt_costs=(0.25, None, 0.125))

    rows = outcome_to_rows(outcome, prompt=PROMPT)

    assert [c["est_cost_usd"] for c in rows.model_calls] == [Decimal("0.25"), None, Decimal("0.125")]


def test_failed_request_has_no_cost_but_its_priced_attempt_keeps_one():
    outcome = run_route(EASY_JEV, [malformed_with_usage(), client_error()])

    rows = outcome_to_rows(outcome, prompt=PROMPT)

    assert (rows.request["status"], rows.request["failure_reason"]) == ("failed", "all_failed")
    assert (rows.request["est_cost_usd"], rows.request["baseline_cost_usd"]) == (None, None)
    priced, unpriced = rows.model_calls
    assert priced["est_cost_usd"] == Decimal(str(outcome.attempt_costs[0])) > 0
    assert unpriced["est_cost_usd"] is None


def test_request_cost_is_the_final_call_only_and_never_added_to_attempt_costs():
    outcome = run_route(HARD_JEV, [malformed_with_usage(), client_error(), ok(CHEAP[0])])

    rows = outcome_to_rows(outcome, prompt=PROMPT)

    success = rows.model_calls[-1]
    assert success["est_cost_usd"] == rows.request["est_cost_usd"]  # Same call, same estimate
    attempt_total = sum(c["est_cost_usd"] for c in rows.model_calls if c["est_cost_usd"] is not None)
    assert attempt_total == rows.model_calls[0]["est_cost_usd"] + success["est_cost_usd"]
    assert attempt_total > rows.request["est_cost_usd"]  # Earlier priced attempts are spend the request omits


def test_deadline_and_dry_run_outcomes_have_no_attempt_cost():
    deadline = outcome_to_rows(run_route(EASY_JEV, [ok(CHEAP[0])], jev_seconds=50.0), prompt=PROMPT)
    dry_run = outcome_to_rows(run_route(EASY_JEV, dry_run=True), prompt=PROMPT)

    assert [c["est_cost_usd"] for c in deadline.model_calls] == [None]
    assert dry_run.model_calls == ()


# --- Contract ---------------------------------------------------------------------------------


def all_outcomes() -> list[RouteOutcome]:
    return [
        run_route(EASY_JEV, [ok(CHEAP[0])]),
        run_route(EASY_JEV, dry_run=True),
        run_route(HARD_JEV, [ok(CHEAP[0])], budget=FakeBudget(available=False)),
        run_route(JevTimeoutError("t", cause_type="ReadTimeout"), [ok(STRONG[0])]),
        run_route(HARD_JEV, [client_error(), client_error(), ok(CHEAP[0])]),
        run_route(EASY_JEV, [client_error(), client_error()]),
        run_route(EASY_JEV, [ok(CHEAP[0])], jev_seconds=50.0),
        run_route(EASY_JEV, [LLMAuthError("m", status_code=401)]),
        run_route(EASY_JEV, [malformed_with_usage(), client_error()]),  # Failed, with one priced attempt
    ]


def test_rows_have_exactly_the_table_columns():
    for outcome in all_outcomes():
        rows = outcome_to_rows(outcome, prompt=PROMPT)
        assert set(rows.request) == REQUEST_COLUMNS  # created_at is set by the database
        for call in rows.model_calls:
            assert set(call) == MODEL_CALL_COLUMNS


def test_rows_use_only_values_the_schema_accepts():
    for outcome in all_outcomes():
        request = outcome_to_rows(outcome, prompt=PROMPT).request
        calls = outcome_to_rows(outcome, prompt=PROMPT).model_calls
        assert request["status"] in models.ROUTE_STATUSES
        assert request["failure_reason"] in (None, *models.FAILURE_REASONS)
        assert request["final_tier"] in (None, *models.TIERS)
        assert request["tier"] in models.TIERS
        assert request["reason"] in models.ROUTE_REASONS
        assert (request["status"] == "failed") == (request["failure_reason"] is not None)
        assert (request["status"] == "success") == (request["final_model"] is not None)
        assert (request["est_cost_usd"] is None) == (request["baseline_cost_usd"] is None)
        assert request["est_cost_usd"] is None or request["status"] == "success"
        for call in calls:
            assert call["status"] in models.ATTEMPT_STATUSES
            assert call["tier"] in models.TIERS
            assert call["attempt_no"] >= 1 and call["latency_ms"] >= 0
            assert call["est_cost_usd"] is None or call["est_cost_usd"] >= 0
            has_usage = call["input_tokens"] is not None and call["output_tokens"] is not None
            assert (call["est_cost_usd"] is not None) == has_usage  # Unknown stays NULL, never 0


def test_no_row_holds_the_prompt_the_answer_or_an_exception_message():
    for outcome in all_outcomes():
        text = all_values(outcome_to_rows(outcome, prompt=PROMPT))
        assert "PROMPT-MARKER" not in text
        assert "ANSWER-MARKER" not in text
        assert SECRET not in text


def test_mapping_is_deterministic_and_leaves_the_outcome_unchanged():
    outcome = run_route(HARD_JEV, [client_error(), client_error(), ok(CHEAP[0])])
    before = copy.deepcopy(outcome)

    first, second = outcome_to_rows(outcome, prompt=PROMPT), outcome_to_rows(outcome, prompt=PROMPT)

    assert first == second
    assert outcome == before


def test_rows_are_read_only():
    rows = outcome_to_rows(run_route(EASY_JEV, [ok(CHEAP[0])]), prompt=PROMPT)

    with pytest.raises(TypeError):
        rows.request["status"] = "failed"
    with pytest.raises(TypeError):
        rows.model_calls[0]["seq"] = 5
    with pytest.raises(dataclasses.FrozenInstanceError):
        rows.request = {}


def test_nested_row_values_are_copies_independent_of_the_outcome():
    # Read-only is top-level only: model_chain and the Jev maps stay a list and dicts so they serialize
    # to JSONB. They must be copies, so changing them can never alter the outcome or a later mapping.
    outcome = run_route(EASY_JEV, [ok(CHEAP[0])])
    rows = outcome_to_rows(outcome, prompt=PROMPT)

    rows.request["model_chain"].append("tampered")
    rows.request["task_type_probs"]["code"] = -1.0
    rows.request["difficulty_dist"].clear()

    assert "tampered" not in outcome.decision.model_chain
    assert outcome.jev.task_type_probs["code"] == 1.0
    assert outcome.jev.difficulty_dist
    fresh = outcome_to_rows(outcome, prompt=PROMPT)
    assert fresh.request["model_chain"] == list(CHEAP)
    assert fresh.request["task_type_probs"] == outcome.jev.task_type_probs
    assert fresh.request["difficulty_dist"] == outcome.jev.difficulty_dist


# --- Refusals -------------------------------------------------------------------------------


def test_non_uuid_request_id_is_refused_without_echoing_it():
    outcome = dataclasses.replace(run_route(EASY_JEV, dry_run=True), request_id="req-SECRET-1")

    with pytest.raises(ValueError, match="request_id") as error:
        outcome_to_rows(outcome, prompt=PROMPT)

    assert "SECRET" not in str(error.value)


def test_empty_prompt_is_refused():
    with pytest.raises(ValueError, match="prompt"):
        outcome_to_rows(run_route(EASY_JEV, dry_run=True), prompt="")


@pytest.mark.parametrize("field", ["jev_error", "jev_error_cause"])
def test_jev_error_fields_that_are_not_class_names_are_refused_without_echoing_them(field):
    outcome = run_route(JevTimeoutError("t", cause_type="ReadTimeout"), [ok(STRONG[0])])
    message = f"Timeout reading {SECRET}"
    if field == "jev_error":
        outcome = dataclasses.replace(outcome, decision=dataclasses.replace(outcome.decision, jev_error=message))
    else:
        outcome = dataclasses.replace(outcome, jev_error_cause=message)

    with pytest.raises(ValueError, match=field) as error:
        outcome_to_rows(outcome, prompt=PROMPT)

    assert SECRET not in str(error.value)
