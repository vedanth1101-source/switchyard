"""GET /api/requests/{request_id} contract (Task S2.13). The reader is faked; no database or network access."""

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from types import MappingProxyType

import httpx
import pytest
import sqlalchemy
from fastapi.testclient import TestClient

from app.core.jev_client import JevTimeoutError
from app.core.llm_client import LLMMalformedResponseError
from app.core.settings import get_settings
from app.db.repo import OutcomeRows, outcome_to_rows
from app.main import create_app
from app.routes import requests as requests_api
from tests.test_repo_mapping import CHEAP, EASY_JEV, HARD_JEV, PROMPT, STRONG, client_error, hit_rows, ok, run_route

CREATED_AT = datetime(2026, 10, 14, 8, 30, 15, 123000, tzinfo=timezone.utc)
SECRET = "SECRET-MARKER-postgresql://user:pw@db.internal"


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    async def no_network(*args, **kwargs):
        raise AssertionError("API tests must not reach the network")

    monkeypatch.setattr(httpx.AsyncClient, "send", no_network)


def stored(outcome) -> OutcomeRows:
    """What load_request returns for a saved outcome: the mapped rows plus the database's created_at."""
    rows = outcome_to_rows(outcome, prompt=PROMPT)
    return OutcomeRows(
        request=MappingProxyType({**rows.request, "created_at": CREATED_AT}), model_calls=rows.model_calls
    )


class FakeReader:
    def __init__(self, rows: OutcomeRows | None = None, error: Exception | None = None):
        self.rows = rows
        self.error = error
        self.calls: list = []

    async def __call__(self, request_id):
        self.calls.append(request_id)
        if self.error is not None:
            raise self.error
        if self.rows is not None and self.rows.request["id"] == request_id:
            return self.rows
        return None


def client_with(reader) -> TestClient:
    app = create_app()
    app.dependency_overrides[requests_api.get_request_reader] = lambda: reader
    return TestClient(app, raise_server_exceptions=False)


def get(rows: OutcomeRows):
    reader = FakeReader(rows)
    response = client_with(reader).get(f"/api/requests/{rows.request['id']}")
    return response, reader


# --- Found -------------------------------------------------------------------------------------


def test_success_returns_the_request_decision_and_attempts():
    rows = stored(run_route(EASY_JEV, [ok(CHEAP[0])]))

    response, reader = get(rows)

    assert response.status_code == 200
    assert reader.calls == [rows.request["id"]]  # A uuid.UUID, as stored
    request = rows.request
    call = rows.model_calls[0]
    assert response.json() == {
        "request": {
            "id": str(request["id"]),
            "created_at": response.json()["request"]["created_at"],
            "status": "success",
            "dry_run": False,
            "failure_reason": None,
            "final_model": CHEAP[0],
            "final_tier": "cheap",
            "fell_back_to_cheap": False,
            "strong_reserved": False,
            "total_latency_ms": request["total_latency_ms"],
            "config_version": request["config_version"],
            "prompt_chars": len(PROMPT),
            "est_cost_usd": float(request["est_cost_usd"]),
            "baseline_cost_usd": float(request["baseline_cost_usd"]),
            "cache_hit": False,  # S2.14 step 4: added deliberately (approved read-endpoint field)
        },
        "decision": {
            "task_type": "code",
            "tier": "cheap",
            "reason": "EASY",
            "model_chain": list(CHEAP),
            "confidence": request["confidence"],
            "difficulty": request["difficulty"],
            "needs_reasoning_prob": request["needs_reasoning_prob"],
            "task_type_probs": request["task_type_probs"],
            "difficulty_dist": request["difficulty_dist"],
            "jev_model": "jev-1.13.0",
            "jev_latency_ms": 1434,
            "jev_input_tokens": 446,
            "jev_credits_charged": 1,
            "jev_error": None,
            "jev_http_status": None,
            "jev_error_cause": None,
        },
        "attempts": [
            {
                "seq": 0,
                "tier": "cheap",
                "model": CHEAP[0],
                "attempt_no": 1,
                "status": "success",
                "latency_ms": call["latency_ms"],
                "http_status": 200,
                "input_tokens": 13,
                "output_tokens": 115,
                "reasoning_tokens": 64,
                "est_cost_usd": float(call["est_cost_usd"]),
            }
        ],
    }
    assert datetime.fromisoformat(response.json()["request"]["created_at"]) == CREATED_AT


def test_usd_values_are_json_numbers():
    response, _ = get(stored(run_route(EASY_JEV, [ok(CHEAP[0])])))

    body = response.json()
    for value in (body["request"]["est_cost_usd"], body["request"]["baseline_cost_usd"]):
        assert isinstance(value, float)
    assert isinstance(body["attempts"][0]["est_cost_usd"], float)
    assert body["request"]["est_cost_usd"] == body["attempts"][0]["est_cost_usd"]  # The same call


def test_response_never_contains_the_prompt_or_its_hash():
    rows = stored(run_route(EASY_JEV, [ok(CHEAP[0])]))

    response, _ = get(rows)

    assert response.status_code == 200
    assert "prompt_hash" not in response.text
    assert hashlib.sha256(PROMPT.encode("utf-8")).hexdigest() not in response.text
    assert "PROMPT-MARKER" not in response.text
    assert "ANSWER-MARKER" not in response.text


def test_failed_request_keeps_null_costs_and_its_priced_attempt():
    malformed = LLMMalformedResponseError("m: empty", status_code=200, input_tokens=13, output_tokens=1024)
    rows = stored(run_route(EASY_JEV, [malformed, client_error()]))

    response, _ = get(rows)

    body = response.json()
    assert response.status_code == 200
    assert (body["request"]["status"], body["request"]["failure_reason"]) == ("failed", "all_failed")
    assert (body["request"]["final_model"], body["request"]["final_tier"], body["request"]["fell_back_to_cheap"]) == (
        None,
        None,
        None,
    )
    assert (body["request"]["est_cost_usd"], body["request"]["baseline_cost_usd"]) == (None, None)
    priced, unpriced = body["attempts"]
    assert priced["est_cost_usd"] == float(rows.model_calls[0]["est_cost_usd"]) > 0
    assert (priced["status"], priced["reasoning_tokens"]) == ("malformed", None)
    assert (unpriced["status"], unpriced["est_cost_usd"], unpriced["input_tokens"]) == ("client_error", None, None)


def test_dry_run_has_no_attempts_and_no_cost():
    response, _ = get(stored(run_route(HARD_JEV, dry_run=True)))

    body = response.json()
    assert (body["request"]["status"], body["request"]["dry_run"], body["request"]["strong_reserved"]) == (
        "dry_run",
        True,
        False,
    )
    assert (body["request"]["est_cost_usd"], body["request"]["final_model"]) == (None, None)
    assert (body["decision"]["tier"], body["decision"]["model_chain"]) == ("strong", list(STRONG))
    assert body["attempts"] == []


def test_jev_failure_keeps_absent_jev_fields_null():
    rows = stored(run_route(JevTimeoutError("t", cause_type="ReadTimeout"), [ok(STRONG[0])]))

    decision = get(rows)[0].json()["decision"]

    assert (decision["reason"], decision["jev_error"], decision["jev_error_cause"]) == (
        "JEV_ERROR",
        "JevTimeoutError",
        "ReadTimeout",
    )
    for name in (
        "confidence",
        "difficulty",
        "needs_reasoning_prob",
        "task_type_probs",
        "difficulty_dist",
        "jev_model",
        "jev_latency_ms",
        "jev_input_tokens",
        "jev_credits_charged",
        "jev_http_status",
    ):
        assert decision[name] is None, name


def test_attempts_keep_the_stored_sequence():
    rows = stored(run_route(HARD_JEV, [client_error(), client_error(), ok(CHEAP[0])]))

    attempts = get(rows)[0].json()["attempts"]

    assert [(a["seq"], a["tier"], a["model"], a["status"]) for a in attempts] == [
        (0, "strong", STRONG[0], "client_error"),
        (1, "strong", STRONG[1], "client_error"),
        (2, "cheap", CHEAP[0], "success"),
    ]


def test_response_carries_this_requests_own_id_header():
    rows = stored(run_route(EASY_JEV, [ok(CHEAP[0])]))

    response, _ = get(rows)

    assert response.status_code == 200
    header = uuid.UUID(response.headers["X-Request-ID"])
    assert header.version == 4
    assert header != rows.request["id"]  # The GET is its own request; the body describes the stored one


# --- Not found, invalid and unavailable ------------------------------------------------------------


def test_unknown_id_is_a_404_envelope_with_this_requests_id():
    reader = FakeReader()
    unknown = uuid.uuid4()

    response = client_with(reader).get(f"/api/requests/{unknown}")

    assert response.status_code == 404
    assert response.json() == {"request_id": response.headers["X-Request-ID"], "error": "not_found"}
    assert reader.calls == [unknown]


def test_non_uuid_id_is_a_422_without_echoing_it_or_reading():
    reader = FakeReader()

    response = client_with(reader).get("/api/requests/not-a-uuid-SECRETVALUE")

    body = response.json()
    assert response.status_code == 422
    assert (body["request_id"], body["error"]) == (response.headers["X-Request-ID"], "invalid_request")
    assert [item["loc"] for item in body["detail"]] == [["path", "request_id"]]
    assert "SECRETVALUE" not in response.text
    assert reader.calls == []


def test_database_error_is_a_503_and_logs_only_the_error_type(caplog):
    error = sqlalchemy.exc.OperationalError(f"SELECT * FROM requests {SECRET}", {"id": SECRET}, Exception(SECRET))

    with caplog.at_level(logging.DEBUG):
        response = client_with(FakeReader(error=error)).get(f"/api/requests/{uuid.uuid4()}")

    assert response.status_code == 503
    assert response.json() == {"request_id": response.headers["X-Request-ID"], "error": "database_unavailable"}
    (record,) = [r for r in caplog.records if r.getMessage().startswith("request read failed")]
    assert record.levelno == logging.WARNING
    assert (record.request_id, record.error_type) == (response.headers["X-Request-ID"], "OperationalError")
    assert SECRET not in caplog.text
    assert SECRET not in response.text


def test_unexpected_reader_errors_reach_the_generic_500():
    response = client_with(FakeReader(error=RuntimeError(SECRET))).get(f"/api/requests/{uuid.uuid4()}")

    assert response.status_code == 500
    assert response.json() == {"request_id": response.headers["X-Request-ID"], "error": "internal_error"}
    assert SECRET not in response.text


def test_without_database_url_the_endpoint_is_a_503_not_configured(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    get_settings.cache_clear()
    try:
        response = TestClient(create_app()).get(f"/api/requests/{uuid.uuid4()}")
    finally:
        get_settings.cache_clear()

    assert response.status_code == 503
    assert response.json() == {"request_id": response.headers["X-Request-ID"], "error": "database_not_configured"}


def test_real_reader_loads_through_the_cached_engine_in_a_worker_thread(monkeypatch):
    rows = stored(run_route(EASY_JEV, [ok(CHEAP[0])]))
    seen = []

    def fake_load(engine, request_id):
        seen.append((engine, request_id))
        return rows

    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:pw@localhost:5999/db")
    monkeypatch.setattr(requests_api, "load_request", fake_load)
    get_settings.cache_clear()
    try:
        response = TestClient(create_app()).get(f"/api/requests/{rows.request['id']}")
    finally:
        get_settings.cache_clear()

    assert response.status_code == 200
    ((engine, request_id),) = seen
    assert request_id == rows.request["id"]
    assert engine is requests_api.get_engine("postgresql+psycopg://user:pw@localhost:5999/db")  # Cached per URL


# --- Routing and OpenAPI -------------------------------------------------------------------------


def test_wrong_method_is_405_allowing_get():
    response = client_with(FakeReader()).post(f"/api/requests/{uuid.uuid4()}")

    assert response.status_code == 405
    assert response.headers["allow"] == "GET"


def test_openapi_documents_get_only_without_prompt_hash():
    spec = TestClient(create_app()).get("/api/openapi.json").json()

    operations = spec["paths"]["/api/requests/{request_id}"]
    assert set(operations) == {"get"}
    assert {"200", "404", "422", "503"} <= set(operations["get"]["responses"])
    assert "prompt_hash" not in str(spec)


def test_decimal_costs_from_the_database_are_converted_not_dropped():
    rows = stored(run_route(EASY_JEV, [ok(CHEAP[0])]))
    request = {**rows.request, "est_cost_usd": Decimal("0.00004665"), "baseline_cost_usd": Decimal("0.00023325")}
    call = {**rows.model_calls[0], "est_cost_usd": Decimal("0.00004665")}

    response, _ = get(OutcomeRows(request=MappingProxyType(request), model_calls=(MappingProxyType(call),)))

    body = response.json()
    assert (body["request"]["est_cost_usd"], body["request"]["baseline_cost_usd"]) == (0.00004665, 0.00023325)
    assert body["attempts"][0]["est_cost_usd"] == 0.00004665


def test_cache_hit_is_reported_for_routed_requests_and_cache_hits():
    routed = stored(run_route(EASY_JEV, [ok(CHEAP[0])]))
    hit = hit_rows()
    hit = OutcomeRows(request=MappingProxyType({**hit.request, "created_at": CREATED_AT}), model_calls=())

    routed_body, hit_body = get(routed)[0].json(), get(hit)[0].json()

    assert routed_body["request"]["cache_hit"] is False
    assert (hit_body["request"]["cache_hit"], hit_body["request"]["est_cost_usd"], hit_body["attempts"]) == (
        True,
        None,
        [],
    )
