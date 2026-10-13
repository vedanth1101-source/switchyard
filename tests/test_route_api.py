"""POST /api/route contract (Task S1.7). All external services are faked; no network access."""

import copy
import dataclasses
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.core.jev_client import JevAuthError, JevClientError, JevTimeoutError
from app.core.llm_client import (
    LLMAuthError,
    LLMBudgetExhaustedError,
    LLMClientError,
    LLMMalformedResponseError,
    LLMServerError,
    LLMTimeoutError,
    ModelResult,
)
from app.core.model_config import ModelConfigError, load_model_config
from app.core.policy import RoutingConfig
from app.core.router import CliConfigurationError, RouterDeps
from app.main import create_app
from app.routes import route as route_api
from tests.fakes import FakeBudget, FakeClock, FakeJev, FakeLLM, RecordingSleep

FIXTURES = Path(__file__).parent / "fixtures"
PROMPT = "Write a function to check primes PROMPT-MARKER"
FAKE_KEY = "fake-secret-key-0123456789"
CONFIG = load_model_config()
CHEAP = CONFIG.routing.get_chain("general", "cheap")
STRONG = CONFIG.routing.get_chain("general", "strong")
EASY_JEV = json.loads((FIXTURES / "jev_response_success.json").read_text(encoding="utf-8"))
HARD_JEV = copy.deepcopy(EASY_JEV)
HARD_JEV["data"]["answers"]["difficulty"]["score"] = 3.0
HARD_JEV["data"]["answers"]["difficulty"]["probabilities"] = {"0": 0.0, "1": 0.0, "2": 0.0, "3": 1.0, "4": 0.0}


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    async def no_network(*args, **kwargs):
        raise AssertionError("API tests must not reach the network")

    monkeypatch.setattr(httpx.AsyncClient, "send", no_network)


def ok(model: str) -> ModelResult:
    return ModelResult(
        text=f"answer from {model}",
        model=model,
        input_tokens=5,
        output_tokens=7,
        latency_ms=900,
        finish_reason="stop",
        http_status=200,
    )


def client_error() -> LLMClientError:
    return LLMClientError("m: HTTP 400: request rejected", status_code=400)


class ApiHarness:
    def __init__(self, jev_outcome=EASY_JEV, script=(), budget=None, jev_seconds=0.0, config=CONFIG):
        self.clock = FakeClock()
        self.jev = FakeJev(jev_outcome, self.clock, jev_seconds)
        self.budget = budget or FakeBudget()
        self.llm = FakeLLM(list(script), self.clock)
        self.deps = RouterDeps(
            config=config,
            jev=self.jev,
            budget=self.budget,
            llm=self.llm,
            request_budget_seconds=50.0,
            clock=self.clock,
            sleep=RecordingSleep(self.clock),
            new_request_id=lambda: "req-api-1",
        )
        self.factory_calls: list[bool] = []
        app = create_app()
        app.dependency_overrides[route_api.get_deps_factory] = lambda: self.factory
        self.client = TestClient(app)

    @asynccontextmanager
    async def factory(self, settings, *, dry_run):
        self.factory_calls.append(dry_run)
        yield self.deps

    def post(self, body: dict) -> httpx.Response:
        return self.client.post("/api/route", json=body)


def client_with_failing_factory(error: Exception) -> TestClient:
    @asynccontextmanager
    async def factory(settings, *, dry_run):
        raise error
        yield  # pragma: no cover

    app = create_app()
    app.dependency_overrides[route_api.get_deps_factory] = lambda: factory
    return TestClient(app)


def assert_request_id_header(response: httpx.Response) -> str:
    request_id = response.headers["X-Request-ID"]
    assert response.json()["request_id"] == request_id
    return request_id


# --- 200 ------------------------------------------------------------------------------------


def test_route_200_schema():
    harness = ApiHarness(EASY_JEV, [ok(CHEAP[0])])

    response = harness.post({"prompt": PROMPT})

    assert response.status_code == 200
    body = response.json()
    assert uuid.UUID(assert_request_id_header(response)).version == 4  # S2.1: the middleware assigns it
    assert set(body) == {
        "request_id",
        "dry_run",
        "decision",
        "response",
        "strong_reserved",
        "total_latency_ms",
        "config_version",
        "cost",  # S1.8: added deliberately (docs/04 response contract)
    }
    assert body["dry_run"] is False
    assert body["strong_reserved"] is False
    assert body["config_version"] == CONFIG.config_version
    decision = body["decision"]
    assert (decision["task_type"], decision["tier"], decision["reason"]) == ("code", "cheap", "EASY")
    assert decision["model_chain"] == CHEAP
    assert decision["confidence"] == 0.99
    assert decision["task_type_probs"]["code"] == 1.0
    assert decision["jev_latency_ms"] == 1434
    assert decision["jev_error"] is None
    assert decision["jev_http_status"] is None
    assert decision["jev_error_cause"] is None
    assert body["response"] == {
        "model": CHEAP[0],
        "tier": "cheap",
        "text": f"answer from {CHEAP[0]}",
        "latency_ms": 900,
        "fell_back_to_cheap": False,
        "attempts": [
            {
                "tier": "cheap",
                "model": CHEAP[0],
                "attempt_no": 1,
                "status": "success",
                "latency_ms": 0,
                "input_tokens": 5,
                "output_tokens": 7,
                "reasoning_tokens": None,
                "http_status": 200,
            }
        ],
    }
    assert harness.jev.calls[0][0] == PROMPT
    assert harness.factory_calls == [False]


def test_route_dry_run_returns_decision_without_calling_a_model():
    harness = ApiHarness(HARD_JEV)

    response = harness.post({"prompt": PROMPT, "dry_run": True})

    assert response.status_code == 200
    body = response.json()
    assert body["dry_run"] is True
    assert body["response"] is None
    assert (body["decision"]["tier"], body["decision"]["reason"]) == ("strong", "HARD")
    assert body["strong_reserved"] is False
    assert harness.llm.calls == []
    assert harness.budget.reserve_calls == 0
    assert harness.factory_calls == [True]


def test_strong_fallback_to_cheap_is_reported():
    harness = ApiHarness(HARD_JEV, [client_error(), client_error(), ok(CHEAP[0])])

    body = harness.post({"prompt": PROMPT}).json()

    assert body["decision"]["tier"] == "strong"
    assert body["strong_reserved"] is True
    assert (body["response"]["tier"], body["response"]["fell_back_to_cheap"]) == ("cheap", True)
    assert [a["status"] for a in body["response"]["attempts"]] == ["client_error", "client_error", "success"]


def test_jev_error_is_reported_by_class_name_only():
    harness = ApiHarness(JevTimeoutError(f"timed out {PROMPT} {FAKE_KEY}"), [ok(STRONG[0])])

    response = harness.post({"prompt": PROMPT})

    assert response.status_code == 200
    decision = response.json()["decision"]
    assert (decision["reason"], decision["jev_error"]) == ("JEV_ERROR", "JevTimeoutError")
    assert decision["task_type_probs"] is None
    assert decision["confidence"] is None
    assert FAKE_KEY not in response.text
    assert "timed out" not in response.text


def test_prompt_of_exactly_8000_characters_is_accepted():
    harness = ApiHarness(EASY_JEV, [ok(CHEAP[0])])

    assert harness.post({"prompt": "x" * 8000}).status_code == 200


def test_unused_documented_use_cache_field_is_accepted():
    harness = ApiHarness(EASY_JEV, [ok(CHEAP[0])])

    assert harness.post({"prompt": PROMPT, "use_cache": True}).status_code == 200


# --- 422 ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"prompt": ""},
        {"prompt": "   \n\t"},
        {"prompt": "x" * 8001},
        {"prompt": 42},
        {"prompt": PROMPT, "dry_run": "maybe"},
    ],
    ids=["missing", "empty", "blank", "too-long", "not-a-string", "bad-dry-run"],
)
def test_route_422_rejects_invalid_requests_before_any_call(body):
    harness = ApiHarness()

    response = harness.post(body)

    assert response.status_code == 422
    assert harness.factory_calls == []
    assert harness.jev.calls == []


# --- 502 / 503 from execution ---------------------------------------------------------------


def test_route_502_with_attempts_when_all_models_fail():
    harness = ApiHarness(EASY_JEV, [client_error(), client_error()])

    response = harness.post({"prompt": PROMPT})

    assert response.status_code == 502
    body = response.json()
    assert uuid.UUID(assert_request_id_header(response)).version == 4  # S2.1: the middleware assigns it
    assert body["error"] == "all_models_failed"
    assert [(a["model"], a["status"]) for a in body["attempts"]] == [
        (CHEAP[0], "client_error"),
        (CHEAP[1], "client_error"),
    ]
    assert body["decision"]["tier"] == "cheap"
    assert body["config_version"] == CONFIG.config_version
    assert "request rejected" not in response.text


def test_route_502_when_the_deadline_is_hit():
    harness = ApiHarness(EASY_JEV, [ok(CHEAP[0])], jev_seconds=50.0)

    response = harness.post({"prompt": PROMPT})

    assert response.status_code == 502
    assert response.json()["error"] == "deadline_exceeded"
    assert [a["status"] for a in response.json()["attempts"]] == ["deadline"]
    assert harness.llm.calls == []


@pytest.mark.parametrize(
    "error",
    [LLMAuthError("m: HTTP 401", status_code=401), LLMBudgetExhaustedError("m: HTTP 402", status_code=402)],
    ids=["auth", "budget"],
)
def test_route_503_when_the_gateway_rejects_every_model(error):
    harness = ApiHarness(HARD_JEV, [error])

    response = harness.post({"prompt": PROMPT})

    assert response.status_code == 503
    assert response.json()["error"] == "model_gateway_unavailable"
    assert len(response.json()["attempts"]) == 1
    assert_request_id_header(response)


# --- Dependency and configuration failures --------------------------------------------------


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (LLMAuthError("AI gateway API key is not configured"), 503, "model_gateway_not_configured"),
        (JevAuthError("JEVSTATION_API_KEY is missing"), 503, "classifier_not_configured"),
        (ModelConfigError("C:/secret/path/models.yaml: catalog.x: missing"), 500, "configuration_error"),
        (CliConfigurationError("exactly one model provider is supported"), 500, "configuration_error"),
    ],
    ids=["gateway-key", "jev-key", "model-config", "provider-setup"],
)
def test_dependency_failures_map_to_generic_errors(error, status, code):
    response = client_with_failing_factory(error).post("/api/route", json={"prompt": PROMPT})

    assert response.status_code == status
    body = response.json()
    assert body["error"] == code
    assert body["attempts"] == []
    assert body["decision"] is None
    assert uuid.UUID(assert_request_id_header(response)).version == 4
    assert str(error) not in response.text
    assert "secret/path" not in response.text


def test_routing_config_error_is_a_500_without_details():
    broken = dataclasses.replace(CONFIG, routing=RoutingConfig(models={}, general_strong_chain=list(STRONG)))
    harness = ApiHarness(EASY_JEV, config=broken)

    response = harness.post({"prompt": PROMPT})

    assert response.status_code == 500
    assert response.json()["error"] == "configuration_error"
    assert "RoutingConfigError" not in response.text
    assert harness.llm.calls == []


def test_real_dependencies_without_gateway_key_return_503_before_any_call(monkeypatch):
    monkeypatch.setenv("JEVSTATION_API_KEY", FAKE_KEY)  # env wins, so the real .env is never read
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)

    response = TestClient(create_app()).post("/api/route", json={"prompt": PROMPT})

    assert response.status_code == 503
    assert response.json()["error"] == "model_gateway_not_configured"
    assert FAKE_KEY not in response.text


def test_real_dependencies_without_jev_key_return_503(monkeypatch):
    monkeypatch.delenv("JEVSTATION_API_KEY", raising=False)
    monkeypatch.setattr("app.core.jev_client.load_api_key_from_env_or_file", lambda: None)  # never read .env

    response = TestClient(create_app()).post("/api/route", json={"prompt": PROMPT, "dry_run": True})

    assert response.status_code == 503
    assert response.json()["error"] == "classifier_not_configured"


def test_real_dependencies_with_unreadable_model_config_return_500(monkeypatch, tmp_path):
    missing = tmp_path / "missing-models.yaml"
    monkeypatch.setenv("MODELS_CONFIG_PATH", str(missing))

    response = TestClient(create_app()).post("/api/route", json={"prompt": PROMPT, "dry_run": True})

    assert response.status_code == 500
    assert response.json()["error"] == "configuration_error"
    assert "missing-models.yaml" not in response.text


# --- S2.1: one request ID per HTTP request -------------------------------------------------


def route_log_request_ids(caplog) -> list[str]:
    return [record.request_id for record in caplog.records if record.name == "app.core.router"]


def test_success_header_body_and_route_log_share_the_middleware_request_id(caplog):
    harness = ApiHarness(EASY_JEV, [ok(CHEAP[0])])

    with caplog.at_level(logging.INFO, logger="app.core.router"):
        response = harness.post({"prompt": PROMPT})

    request_id = assert_request_id_header(response)
    assert route_log_request_ids(caplog) == [request_id]
    assert request_id != "req-api-1"  # the deps' own generator is replaced on the HTTP path


def test_failure_header_body_and_route_log_share_one_request_id(caplog):
    harness = ApiHarness(EASY_JEV, [client_error(), client_error()])

    with caplog.at_level(logging.INFO, logger="app.core.router"):
        response = harness.post({"prompt": PROMPT})

    assert response.status_code == 502
    assert route_log_request_ids(caplog) == [assert_request_id_header(response)]


def test_setup_error_uses_the_middleware_request_id_in_body_header_and_log(caplog):
    client = client_with_failing_factory(JevAuthError("JEVSTATION_API_KEY is missing"))

    with caplog.at_level(logging.INFO, logger="app.routes.route"):
        response = client.post("/api/route", json={"prompt": PROMPT})

    assert response.status_code == 503
    request_id = assert_request_id_header(response)
    assert [record.request_id for record in caplog.records if record.name == "app.routes.route"] == [request_id]


def test_inbound_request_id_is_not_used_by_the_route():
    harness = ApiHarness(EASY_JEV, [ok(CHEAP[0])])

    response = harness.client.post("/api/route", json={"prompt": PROMPT}, headers={"X-Request-ID": "client-id"})

    assert assert_request_id_header(response) != "client-id"


def test_unexpected_exception_is_a_generic_500_with_a_sanitized_log(caplog):
    client = client_with_failing_factory(RuntimeError(f"boom {PROMPT} {FAKE_KEY} C:/secret/path"))

    with caplog.at_level(logging.INFO):
        response = client.post("/api/route", json={"prompt": PROMPT})

    assert response.status_code == 500
    request_id = assert_request_id_header(response)
    assert response.json() == {"request_id": request_id, "error": "internal_error"}
    errors = [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert [(record.request_id, record.error_type) for record in errors] == [(request_id, "RuntimeError")]
    for secret in ("PROMPT-MARKER", FAKE_KEY, "secret/path", "Traceback"):
        assert secret not in caplog.text
        assert secret not in response.text


def test_unexpected_jev_exception_is_a_generic_500_not_a_jev_error():
    harness = ApiHarness(RuntimeError("Jev SDK bug PROMPT-MARKER"), [ok(CHEAP[0])])

    response = harness.post({"prompt": PROMPT})

    assert response.status_code == 500
    assert response.json()["error"] == "internal_error"
    assert harness.llm.calls == []


# --- S2.1: the documented 500 schema matches the real 500 bodies ----------------------------

JSON_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def matches_schema(spec: dict, schema: dict, value) -> bool:
    """Minimal JSON Schema check for the keywords FastAPI emits here (no jsonschema dependency)."""
    if "$ref" in schema:
        return matches_schema(spec, spec["components"]["schemas"][schema["$ref"].rsplit("/", 1)[-1]], value)
    if "anyOf" in schema:
        return any(matches_schema(spec, option, value) for option in schema["anyOf"])
    if "oneOf" in schema:
        return sum(matches_schema(spec, option, value) for option in schema["oneOf"]) == 1
    if "const" in schema and value != schema["const"]:
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if "type" in schema:
        if isinstance(value, bool) and schema["type"] in ("integer", "number"):
            return False
        if not isinstance(value, JSON_TYPES[schema["type"]]):
            return False
    if isinstance(value, dict):
        if not set(schema.get("required", [])) <= set(value):
            return False
        properties = schema.get("properties", {})
        return all(matches_schema(spec, properties[key], item) for key, item in value.items() if key in properties)
    if isinstance(value, list) and "items" in schema:
        return all(matches_schema(spec, schema["items"], item) for item in value)
    return True


def documented_route_500() -> tuple[dict, dict]:
    spec = create_app().openapi()
    return spec, spec["paths"]["/api/route"]["post"]["responses"]["500"]["content"]["application/json"]["schema"]


def test_generic_unhandled_500_body_matches_the_documented_500_schema():
    response = client_with_failing_factory(RuntimeError("bug")).post("/api/route", json={"prompt": PROMPT})
    spec, schema = documented_route_500()

    assert response.status_code == 500
    assert response.json()["error"] == "internal_error"
    assert matches_schema(spec, schema, response.json())


def test_configuration_error_500_body_still_matches_the_documented_500_schema():
    response = client_with_failing_factory(ModelConfigError("bad config")).post("/api/route", json={"prompt": PROMPT})
    spec, schema = documented_route_500()

    assert response.status_code == 500
    assert response.json()["error"] == "configuration_error"
    assert matches_schema(spec, schema, response.json())


def test_documented_500_schema_rejects_bodies_outside_both_contracts():
    spec, schema = documented_route_500()

    assert not matches_schema(spec, schema, {"request_id": "r", "error": "something_else"})
    assert not matches_schema(spec, schema, {"error": "internal_error"})
    assert not matches_schema(spec, schema, {"request_id": "r", "error": "configuration_error"})  # route fields missing


# --- Routing table --------------------------------------------------------------------------


def test_unknown_api_path_is_still_a_json_404():
    response = TestClient(create_app()).get("/api/does-not-exist")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")


def test_openapi_documents_route_errors():
    operation = create_app().openapi()["paths"]["/api/route"]["post"]

    assert {"200", "422", "500", "502", "503"} <= set(operation["responses"])


# --- S1.8: cost metadata --------------------------------------------------------------------


def test_route_200_includes_labelled_cost_estimate():
    from app.core.costs import estimate_cost

    harness = ApiHarness(EASY_JEV, [ok(CHEAP[0])])

    cost = harness.post({"prompt": PROMPT}).json()["cost"]

    expected = estimate_cost(CONFIG, CHEAP[0], 5, 7)
    assert cost == {
        "est_cost_usd": pytest.approx(expected.est_cost_usd),
        "baseline_cost_usd": pytest.approx(expected.baseline_cost_usd),
        "saved_usd": pytest.approx(expected.saved_usd),
        "baseline_model": CONFIG.baseline_model,
        "basis": "list_price_estimate",
        "includes_jev": False,
    }


def test_dry_run_cost_is_null():
    assert ApiHarness(HARD_JEV).post({"prompt": PROMPT, "dry_run": True}).json()["cost"] is None


def test_unknown_cost_is_null_not_zero(monkeypatch):
    from app.core.costs import CostError

    def failing_estimate(*args):
        raise CostError("cannot estimate")

    monkeypatch.setattr("app.core.router.estimate_cost", failing_estimate)

    response = ApiHarness(EASY_JEV, [ok(CHEAP[0])]).post({"prompt": PROMPT})

    assert response.status_code == 200
    assert response.json()["cost"] is None


# --- HTTP status diagnostics ----------------------------------------------------------------


def test_502_attempts_carry_http_status_and_none_for_timeouts_and_network_errors():
    harness = ApiHarness(
        EASY_JEV,
        [
            LLMServerError("m: HTTP 503: server error", status_code=503),
            LLMTimeoutError("m: request timed out"),
            LLMServerError("m: network error (ConnectError)"),
            client_error(),
        ],
    )

    response = harness.post({"prompt": PROMPT})

    assert response.status_code == 502
    assert [(a["status"], a["http_status"]) for a in response.json()["attempts"]] == [
        ("server_error", 503),
        ("timeout", None),
        ("server_error", None),
        ("client_error", 400),
    ]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (LLMAuthError("m: HTTP 401", status_code=401), 401),
        (LLMAuthError("m: HTTP 403", status_code=403), 403),
        (LLMBudgetExhaustedError("m: HTTP 402", status_code=402), 402),
    ],
    ids=["401", "403", "402"],
)
def test_503_attempt_distinguishes_401_403_and_402(error, expected):
    response = ApiHarness(EASY_JEV, [error]).post({"prompt": PROMPT})

    assert response.status_code == 503
    assert [a["http_status"] for a in response.json()["attempts"]] == [expected]


def test_jev_http_status_is_reported_without_the_error_message():
    error = JevClientError(f"JevStation client error (HTTP 403): {FAKE_KEY} {PROMPT}", status_code=403)
    harness = ApiHarness(error, [ok(STRONG[0])])

    response = harness.post({"prompt": PROMPT})

    assert response.status_code == 200
    decision = response.json()["decision"]
    assert (decision["jev_error"], decision["jev_http_status"]) == ("JevClientError", 403)
    assert FAKE_KEY not in response.text
    assert "client error" not in response.text


def test_jev_http_status_is_in_failure_bodies():
    error = JevClientError("JevStation client error (HTTP 401)", status_code=401)
    harness = ApiHarness(error, [LLMAuthError("m: HTTP 401", status_code=401)])

    response = harness.post({"prompt": PROMPT})

    assert response.status_code == 503
    assert response.json()["decision"]["jev_http_status"] == 401


def test_jev_error_cause_is_reported_by_class_name_only():
    error = JevClientError(
        f"Network error communicating with JevStation: {FAKE_KEY} {PROMPT}", cause_type="ConnectError"
    )

    response = ApiHarness(error, [ok(STRONG[0])]).post({"prompt": PROMPT})

    assert response.status_code == 200
    decision = response.json()["decision"]
    assert (decision["jev_error"], decision["jev_error_cause"]) == ("JevClientError", "ConnectError")
    assert FAKE_KEY not in response.text
    assert "Network error" not in response.text


def test_jev_error_cause_is_in_failure_bodies():
    error = JevTimeoutError("timed out", cause_type="ConnectTimeout")
    harness = ApiHarness(error, [LLMAuthError("m: HTTP 401", status_code=401)])

    response = harness.post({"prompt": PROMPT})

    assert response.status_code == 503
    assert response.json()["decision"]["jev_error_cause"] == "ConnectTimeout"


def test_jev_timeout_has_null_http_status():
    response = ApiHarness(JevTimeoutError("timed out"), [ok(STRONG[0])]).post({"prompt": PROMPT})

    assert response.json()["decision"]["jev_http_status"] is None


def test_attempts_expose_reasoning_tokens():
    result = dataclasses.replace(ok(CHEAP[0]), reasoning_tokens=4)
    harness = ApiHarness(EASY_JEV, [result])

    response = harness.post({"prompt": PROMPT})

    assert response.status_code == 200
    assert response.json()["response"]["attempts"][0]["reasoning_tokens"] == 4


def test_502_malformed_attempts_carry_usage_tokens():
    empty = LLMMalformedResponseError("m: malformed response", status_code=200, input_tokens=9, output_tokens=64)
    harness = ApiHarness(EASY_JEV, [empty, client_error()])

    response = harness.post({"prompt": PROMPT})

    assert response.status_code == 502
    attempts = response.json()["attempts"]
    assert [(a["status"], a["input_tokens"], a["output_tokens"], a["reasoning_tokens"]) for a in attempts] == [
        ("malformed", 9, 64, None),
        ("client_error", None, None, None),
    ]
