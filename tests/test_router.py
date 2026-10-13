import asyncio
import copy
import dataclasses
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
import yaml

from app.core.classifier import get_default_questions
from app.core.executor import CALL_MARGIN_SECONDS, MIN_CALL_SECONDS
from app.core.jev_client import (
    JevAuthError,
    JevClient,
    JevClientError,
    JevCreditError,
    JevRateLimitError,
    JevResponseError,
    JevTimeoutError,
)
from app.core.llm_client import (
    GatewayClient,
    LLMAuthError,
    LLMClientError,
    LLMMalformedResponseError,
    LLMServerError,
    LLMTimeoutError,
    ModelResult,
)
from app.core.model_config import DEFAULT_CONFIG_PATH, load_model_config
from app.core.policy import RoutingConfig, RoutingConfigError
from app.core.router import GATEWAY_KEY_ENV, RouteOutcome, RouterDeps, build_cli_deps, main, route
from app.core.settings import Settings
from app.db.repo import outcome_to_rows
from app.infra.budget import RedisStrongBudget, UnavailableStrongBudget
from tests.fakes import FakeClock, FakeLLM, RecordingSleep

FIXTURES = Path(__file__).parent / "fixtures"
PROMPT = "Write a function to check primes PROMPT-MARKER"
FAKE_KEY = "fake-secret-key-0123456789"
CONFIG = load_model_config()
CHEAP = CONFIG.routing.get_chain("general", "cheap")
STRONG = CONFIG.routing.get_chain("general", "strong")


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


EASY_JEV = fixture("jev_response_success.json")  # code, trivial -> cheap EASY
LOW_CONFIDENCE_JEV = fixture("jev_response_low_confidence.synthetic.json")  # strong general
HARD_JEV = copy.deepcopy(EASY_JEV)
HARD_JEV["data"]["answers"]["difficulty"]["score"] = 3.0
# Difficulty is the expected level of the distribution: level 3 -> 4.0 >= 3.5 -> strong HARD
HARD_JEV["data"]["answers"]["difficulty"]["probabilities"] = {"0": 0.0, "1": 0.0, "2": 0.0, "3": 1.0, "4": 0.0}


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    async def no_network(*args, **kwargs):
        raise AssertionError("router tests must not reach the network")

    monkeypatch.setattr(httpx.AsyncClient, "send", no_network)


class FakeJev:
    def __init__(self, outcome, clock: FakeClock | None = None, seconds: float = 0.0):
        self.outcome = outcome
        self.clock = clock
        self.seconds = seconds
        self.calls: list[tuple[str, dict]] = []

    async def evaluate(self, state, questions):
        self.calls.append((state, questions))
        if self.clock is not None:
            self.clock.advance(self.seconds)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return copy.deepcopy(self.outcome)


class FakeBudget:
    def __init__(self, available: bool = True, reserve: bool = True):
        self._available = available
        self._reserve = reserve
        self.available_calls = 0
        self.reserve_calls = 0

    async def available(self) -> bool:
        self.available_calls += 1
        return self._available

    async def reserve(self) -> bool:
        self.reserve_calls += 1
        return self._reserve


def ok(model: str) -> ModelResult:
    return ModelResult(
        text=f"answer from {model}",
        model=model,
        input_tokens=5,
        output_tokens=7,
        latency_ms=1,
        finish_reason="stop",
        http_status=200,
    )


def client_error() -> LLMClientError:
    return LLMClientError("m: HTTP 400: request rejected", status_code=400)


class Harness:
    def __init__(self, jev_outcome, script=(), budget: FakeBudget | None = None, jev_seconds=0.0, config=CONFIG):
        self.clock = FakeClock()
        self.jev = FakeJev(jev_outcome, self.clock, jev_seconds)
        self.budget = budget or FakeBudget()
        self.llm = FakeLLM(list(script), self.clock)
        self.sleep = RecordingSleep(self.clock)
        self.deps = RouterDeps(
            config=config,
            jev=self.jev,
            budget=self.budget,
            llm=self.llm,
            request_budget_seconds=50.0,
            clock=self.clock,
            sleep=self.sleep,
            new_request_id=lambda: "req-1",
        )

    def run(self, dry_run: bool = False) -> RouteOutcome:
        return asyncio.run(route(PROMPT, dry_run=dry_run, deps=self.deps))


# --- Success and dry run --------------------------------------------------------------------


def test_route_returns_answer_and_decision():
    harness = Harness(EASY_JEV, [ok(CHEAP[0])])

    outcome = harness.run()

    assert outcome.status == "success"
    assert outcome.request_id == "req-1"
    assert outcome.config_version == CONFIG.config_version
    assert (outcome.decision.tier, outcome.decision.reason, outcome.decision.task_type) == ("cheap", "EASY", "code")
    assert outcome.jev is not None and outcome.jev.task_type == "code"
    assert outcome.execution is not None
    assert outcome.execution.final_model == CHEAP[0]
    assert outcome.execution.result.text == f"answer from {CHEAP[0]}"
    assert outcome.execution.route is outcome.decision
    assert outcome.attempts == outcome.execution.attempts
    assert outcome.failure_reason is None
    assert outcome.strong_reserved is False
    assert harness.budget.reserve_calls == 0
    assert harness.jev.calls == [(PROMPT, get_default_questions())]
    assert harness.llm.calls[0].prompt == PROMPT


def test_dry_run_skips_model_and_does_not_reserve():
    harness = Harness(HARD_JEV, [])

    outcome = harness.run(dry_run=True)

    assert outcome.status == "dry_run"
    assert (outcome.decision.tier, outcome.decision.reason) == ("strong", "HARD")
    assert outcome.execution is None
    assert outcome.attempts == ()
    assert harness.llm.calls == []
    assert harness.budget.reserve_calls == 0
    assert outcome.strong_reserved is False
    assert len(harness.jev.calls) == 1  # dry run still classifies with Jev


def test_dry_run_does_not_need_an_llm_client():
    harness = Harness(EASY_JEV)
    deps = dataclasses.replace(harness.deps, llm=None)

    outcome = asyncio.run(route(PROMPT, dry_run=True, deps=deps))

    assert outcome.status == "dry_run"


def test_missing_llm_client_fails_before_calling_jev():
    harness = Harness(EASY_JEV)
    deps = dataclasses.replace(harness.deps, llm=None)

    with pytest.raises(ValueError, match="llm"):
        asyncio.run(route(PROMPT, dry_run=False, deps=deps))

    assert harness.jev.calls == []


def test_total_latency_includes_jev_and_model_time():
    harness = Harness(EASY_JEV, [(ok(CHEAP[0]), 2.0)], jev_seconds=1.5)

    assert harness.run().total_latency_ms == 3500


def test_deadline_starts_before_jev_and_is_shared_with_execution():
    # Jev takes 45 s of the 50 s budget; the first model call gets 50 - 45 - 2 = 3 s.
    harness = Harness(EASY_JEV, [ok(CHEAP[0])], jev_seconds=45.0)

    harness.run()

    assert harness.llm.calls[0].timeout == 3.0


# --- Jev failures ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("jev_outcome", "error_name"),
    [
        (JevTimeoutError(f"timed out for {PROMPT} with {FAKE_KEY}"), "JevTimeoutError"),
        (JevRateLimitError("rate limited"), "JevRateLimitError"),
        (ValueError(f"State length exceeds maximum: {PROMPT}"), "ValueError"),
        ({"data": {}}, "JevMalformedResponseError"),
    ],
    ids=["timeout", "rate-limit", "prompt-too-long", "malformed"],
)
def test_jev_failure_routes_strong_general_with_sanitized_error(jev_outcome, error_name):
    harness = Harness(jev_outcome, [ok(STRONG[0])])

    outcome = harness.run()

    assert (outcome.decision.tier, outcome.decision.reason, outcome.decision.task_type) == (
        "strong",
        "JEV_ERROR",
        "general",
    )
    assert outcome.decision.jev_error == error_name
    assert outcome.jev is None
    assert outcome.status == "success"
    assert harness.budget.reserve_calls == 1


def test_unexpected_jev_exception_is_not_swallowed():
    harness = Harness(RuntimeError("bug"))

    with pytest.raises(RuntimeError, match="bug"):
        harness.run()


# --- Budget ---------------------------------------------------------------------------------


def test_strong_route_reserves_exactly_once():
    harness = Harness(HARD_JEV, [ok(STRONG[0])], budget=FakeBudget(available=True, reserve=True))

    outcome = harness.run()

    assert (outcome.decision.tier, outcome.decision.reason) == ("strong", "HARD")
    assert outcome.strong_reserved is True
    assert harness.budget.reserve_calls == 1
    assert harness.llm.models_called == [STRONG[0]]


def test_budget_unavailable_downgrades_without_reserving():
    harness = Harness(HARD_JEV, [ok(CHEAP[0])], budget=FakeBudget(available=False))

    outcome = harness.run()

    assert (outcome.decision.tier, outcome.decision.reason) == ("cheap", "BUDGET")
    assert harness.budget.reserve_calls == 0
    assert outcome.strong_reserved is False
    assert not set(harness.llm.models_called) & set(STRONG)


def test_reservation_race_recomputes_with_budget_unavailable():
    harness = Harness(LOW_CONFIDENCE_JEV, [ok(CHEAP[0])], budget=FakeBudget(available=True, reserve=False))

    outcome = harness.run()

    assert (outcome.decision.tier, outcome.decision.reason) == ("cheap", "BUDGET")
    assert outcome.decision.model_chain == CHEAP
    assert harness.budget.reserve_calls == 1
    assert outcome.strong_reserved is False
    assert harness.llm.models_called == [CHEAP[0]]


def test_cheap_route_never_reserves():
    harness = Harness(EASY_JEV, [ok(CHEAP[0])])

    harness.run()

    assert harness.budget.reserve_calls == 0


def test_cheap_fallback_after_strong_failure_does_not_reserve_again():
    harness = Harness(HARD_JEV, [client_error(), client_error(), ok(CHEAP[0])])

    outcome = harness.run()

    assert harness.budget.reserve_calls == 1
    assert outcome.execution is not None
    assert outcome.execution.fell_back_to_cheap is True
    assert outcome.decision.tier == "strong"


# --- Failures -------------------------------------------------------------------------------


def test_chain_exhausted_returns_failed_outcome_with_attempts():
    harness = Harness(EASY_JEV, [client_error(), client_error()])

    outcome = harness.run()

    assert outcome.status == "failed"
    assert outcome.failure_reason == "all_failed"
    assert outcome.execution is None
    assert [(a.model, a.status) for a in outcome.attempts] == [(CHEAP[0], "client_error"), (CHEAP[1], "client_error")]


def test_fatal_gateway_error_is_a_failed_outcome():
    harness = Harness(EASY_JEV, [LLMAuthError("m: HTTP 401", status_code=401)])

    outcome = harness.run()

    assert (outcome.status, outcome.failure_reason) == ("failed", "fatal")


def test_routing_config_error_is_raised_and_logged(caplog):
    broken = dataclasses.replace(CONFIG, routing=RoutingConfig(models={}, general_strong_chain=list(STRONG)))
    harness = Harness(EASY_JEV, config=broken)

    with caplog.at_level(logging.INFO, logger="app.core.router"):
        with pytest.raises(RoutingConfigError):
            harness.run()

    (record,) = caplog.records
    assert "status=config_error" in record.getMessage()


# --- Logging --------------------------------------------------------------------------------


def test_one_structured_log_line_per_request(caplog):
    harness = Harness(EASY_JEV, [ok(CHEAP[0])])

    with caplog.at_level(logging.INFO, logger="app.core.router"):
        harness.run()

    (record,) = caplog.records
    message = record.getMessage()
    for field in (
        "request_id=req-1",
        "status=success",
        "tier=cheap",
        "reason=EASY",
        "task_type=code",
        f"final_model={CHEAP[0]}",
        "attempts=1",
        "latency_ms=",
    ):
        assert field in message
    assert record.request_id == "req-1"
    assert record.final_model == CHEAP[0]


def test_logs_never_contain_prompt_key_url_or_raw_exception_text(caplog):
    error = JevTimeoutError(f"timed out https://jevstation.example/api {PROMPT} {FAKE_KEY}")
    harness = Harness(error, [client_error(), client_error(), client_error(), client_error()])

    with caplog.at_level(logging.DEBUG):
        outcome = harness.run()

    assert outcome.status == "failed"
    assert "jev_error=JevTimeoutError" in caplog.text
    for forbidden in (PROMPT, "PROMPT-MARKER", FAKE_KEY, "https://", "timed out", "request rejected"):
        assert forbidden not in caplog.text


# --- Outcome and request ids ----------------------------------------------------------------


def test_route_outcome_is_frozen():
    outcome = Harness(EASY_JEV, [ok(CHEAP[0])]).run()

    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.status = "failed"  # type: ignore[misc]


def test_default_request_ids_are_unique_uuid4():
    harness = Harness(EASY_JEV)
    deps = dataclasses.replace(harness.deps, new_request_id=RouterDeps.__dataclass_fields__["new_request_id"].default)

    first = asyncio.run(route(PROMPT, dry_run=True, deps=deps)).request_id
    second = asyncio.run(route(PROMPT, dry_run=True, deps=deps)).request_id

    assert first != second
    assert uuid.UUID(first).version == 4


# --- CLI ------------------------------------------------------------------------------------


def fake_deps_factory(harness: Harness):
    @asynccontextmanager
    async def factory(settings, *, dry_run):
        yield harness.deps

    return factory


def test_cli_dry_run_prints_the_decision(capsys):
    harness = Harness(HARD_JEV)

    exit_code = main([PROMPT, "--dry-run"], deps_factory=fake_deps_factory(harness))

    printed = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert printed["status"] == "dry_run"
    assert printed["decision"]["tier"] == "strong"
    assert printed["answer"] is None
    assert harness.llm.calls == []


def test_cli_success_prints_the_answer(capsys):
    harness = Harness(EASY_JEV, [ok(CHEAP[0])])

    exit_code = main([PROMPT], deps_factory=fake_deps_factory(harness))

    printed = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert printed["answer"] == f"answer from {CHEAP[0]}"
    assert printed["final_model"] == CHEAP[0]
    assert printed["attempts"][0]["status"] == "success"


def test_cli_failed_route_exits_1(capsys):
    harness = Harness(EASY_JEV, [client_error(), client_error()])

    exit_code = main([PROMPT], deps_factory=fake_deps_factory(harness))

    assert exit_code == 1
    assert json.loads(capsys.readouterr().out)["failure_reason"] == "all_failed"


def test_build_cli_deps_constructs_real_clients_without_network(monkeypatch):
    monkeypatch.setenv("JEVSTATION_API_KEY", FAKE_KEY)  # Reaches the client through Settings
    settings = Settings(ai_gateway_api_key=FAKE_KEY, redis_url=None, request_budget_seconds=40, jev_timeout_seconds=6.5)

    async def build(dry_run):
        async with build_cli_deps(settings, dry_run=dry_run) as deps:
            return deps

    full = asyncio.run(build(dry_run=False))
    dry = asyncio.run(build(dry_run=True))

    assert isinstance(full.jev, JevClient)
    assert (full.jev.timeout, dry.jev.timeout) == (6.5, 6.5)
    assert isinstance(full.llm, GatewayClient)
    assert isinstance(full.budget, UnavailableStrongBudget)
    assert full.request_budget_seconds == 40
    assert (
        full.config.config_version
        == load_model_config(confidence_threshold=settings.confidence_threshold).config_version
    )
    assert dry.llm is None


def test_cli_missing_gateway_key_exits_2_without_calling_anything(monkeypatch, capsys):
    monkeypatch.setenv("JEVSTATION_API_KEY", FAKE_KEY)
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)

    exit_code = main([PROMPT])

    err = capsys.readouterr().err
    assert exit_code == 2
    assert GATEWAY_KEY_ENV in err
    assert FAKE_KEY not in err


def test_cli_missing_jev_key_exits_2(monkeypatch, capsys):
    monkeypatch.delenv("JEVSTATION_API_KEY", raising=False)  # The client never falls back to a .env file

    exit_code = main([PROMPT, "--dry-run"])

    assert exit_code == 2
    assert "JEVSTATION_API_KEY" in capsys.readouterr().err


def test_build_cli_deps_passes_the_jev_key_from_settings(monkeypatch):
    monkeypatch.delenv("JEVSTATION_API_KEY", raising=False)  # Only the Settings value can reach the client
    settings = Settings(jevstation_api_key=FAKE_KEY, ai_gateway_api_key=FAKE_KEY, redis_url=None)

    async def build():
        async with build_cli_deps(settings, dry_run=True) as deps:
            return deps.jev

    jev = asyncio.run(build())

    assert jev._api_key.get_secret_value() == FAKE_KEY


def test_build_cli_deps_without_a_jev_key_in_settings_raises_auth_error(monkeypatch):
    monkeypatch.setenv("JEVSTATION_API_KEY", FAKE_KEY)  # Read by Settings(); this instance overrides it
    settings = Settings(jevstation_api_key=None, ai_gateway_api_key=FAKE_KEY, redis_url=None)

    async def build():
        async with build_cli_deps(settings, dry_run=True):
            pass

    with pytest.raises(JevAuthError) as excinfo:
        asyncio.run(build())

    assert FAKE_KEY not in str(excinfo.value)


def test_gateway_key_env_matches_the_settings_field():
    assert GATEWAY_KEY_ENV.lower() in Settings.model_fields
    assert {provider.api_key_env for provider in CONFIG.providers.values()} == {GATEWAY_KEY_ENV}


def write_models_yaml(tmp_path: Path, change) -> Path:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    change(raw)
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def rename_key_env(raw: dict) -> None:
    raw["providers"]["ai_gateway"]["api_key_env"] = "OTHER_GATEWAY_KEY"


def add_second_provider(raw: dict) -> None:
    raw["providers"]["other"] = dict(raw["providers"]["ai_gateway"])
    raw["catalog"][CHEAP[0]]["provider"] = "other"


@pytest.mark.parametrize(
    ("change", "message"),
    [(rename_key_env, "api_key_env must be AI_GATEWAY_API_KEY"), (add_second_provider, "exactly one model provider")],
    ids=["key-env-mismatch", "two-providers"],
)
def test_cli_rejects_provider_setups_it_cannot_serve(monkeypatch, capsys, tmp_path, change, message):
    monkeypatch.setenv("JEVSTATION_API_KEY", FAKE_KEY)
    monkeypatch.setenv("AI_GATEWAY_API_KEY", FAKE_KEY)
    monkeypatch.setenv("MODELS_CONFIG_PATH", str(write_models_yaml(tmp_path, change)))

    exit_code = main([PROMPT])

    err = capsys.readouterr().err
    assert exit_code == 2
    assert message in err
    assert FAKE_KEY not in err


def test_cli_invalid_settings_exit_2(monkeypatch, capsys):
    monkeypatch.setenv("STRONG_DAILY_CAP", "-1")

    exit_code = main([PROMPT, "--dry-run"])

    err = capsys.readouterr().err
    assert exit_code == 2
    assert "invalid settings" in err
    assert "strong_daily_cap" in err


def test_cli_unreadable_model_config_exit_2(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("MODELS_CONFIG_PATH", str(tmp_path / "missing.yaml"))

    exit_code = main([PROMPT, "--dry-run"])

    assert exit_code == 2
    assert "cannot read model config" in capsys.readouterr().err


def test_cli_routing_config_error_exit_2(capsys):
    broken = dataclasses.replace(CONFIG, routing=RoutingConfig(models={}, general_strong_chain=list(STRONG)))
    harness = Harness(EASY_JEV, config=broken)

    exit_code = main([PROMPT, "--dry-run"], deps_factory=fake_deps_factory(harness))

    assert exit_code == 2
    assert "no usable model chain" in capsys.readouterr().err


def test_build_cli_deps_uses_and_closes_the_redis_budget(monkeypatch):
    monkeypatch.setenv("JEVSTATION_API_KEY", FAKE_KEY)
    settings = Settings(redis_url="redis://localhost:6399/0", strong_daily_cap=9)

    async def build():
        async with build_cli_deps(settings, dry_run=True) as deps:
            return deps

    deps = asyncio.run(build())

    assert isinstance(deps.budget, RedisStrongBudget)
    assert deps.budget.daily_cap == 9


# --- Pre-S1.7 regressions: malformed Jev values and Jev using the whole deadline ---------------


def malformed_probability_jev() -> dict:
    payload = copy.deepcopy(EASY_JEV)
    payload["data"]["answers"]["task_type"]["probabilities"] = {"code": None, "math": 0.1}
    payload["data"]["echo"] = f"{PROMPT} {FAKE_KEY}"  # response body content must never reach logs
    return payload


def test_null_jev_probability_routes_jev_error_without_leaking(caplog):
    harness = Harness(malformed_probability_jev(), [ok(STRONG[0])])

    with caplog.at_level(logging.DEBUG):
        outcome = harness.run()

    assert (outcome.decision.tier, outcome.decision.reason, outcome.decision.task_type) == (
        "strong",
        "JEV_ERROR",
        "general",
    )
    assert outcome.decision.jev_error == "JevMalformedResponseError"
    assert outcome.jev is None
    assert outcome.status == "success"
    assert harness.budget.reserve_calls == 1
    assert "jev_error=JevMalformedResponseError" in caplog.text
    for forbidden in (PROMPT, "PROMPT-MARKER", FAKE_KEY, "probabilities"):
        assert forbidden not in caplog.text


def malformed_difficulty_jev() -> dict:
    payload = copy.deepcopy(EASY_JEV)
    payload["data"]["answers"]["difficulty"]["probabilities"] = {"0": 0.5, "1": 0.5}  # keys 2-4 missing
    return payload


def test_malformed_difficulty_distribution_routes_jev_error_strong_general_and_reserves_once():
    harness = Harness(malformed_difficulty_jev(), [ok(STRONG[0])], budget=FakeBudget(available=True))

    outcome = harness.run()

    assert outcome.jev is None
    assert (outcome.decision.tier, outcome.decision.reason) == ("strong", "JEV_ERROR")
    assert outcome.decision.jev_error == "JevMalformedResponseError"
    assert outcome.decision.model_chain == STRONG
    assert (outcome.strong_reserved, harness.budget.reserve_calls) == (True, 1)
    assert (outcome.jev_http_status, outcome.jev_error_cause) == (None, None)


def test_malformed_difficulty_distribution_without_budget_downgrades_to_cheap_budget():
    harness = Harness(malformed_difficulty_jev(), [ok(CHEAP[0])], budget=FakeBudget(available=False))

    outcome = harness.run()

    assert (outcome.decision.tier, outcome.decision.reason) == ("cheap", "BUDGET")
    assert outcome.decision.jev_error == "JevMalformedResponseError"
    assert (outcome.strong_reserved, harness.budget.reserve_calls) == (False, 0)


def test_malformed_difficulty_distribution_keeps_the_shared_deadline():
    harness = Harness(malformed_difficulty_jev(), [ok(STRONG[0])], jev_seconds=8.0)

    outcome = harness.run()

    assert outcome.status == "success"
    assert harness.llm.calls[0].timeout == 20.0  # min(20, 50 - 8 - 2): Jev time still counts
    assert outcome.total_latency_ms == 8000


@pytest.mark.parametrize("error", [TypeError("parser bug"), ValueError("parser bug")], ids=["type", "value"])
def test_parser_bugs_propagate_instead_of_becoming_jev_error(monkeypatch, error):
    def broken_parser(raw):
        raise error

    monkeypatch.setattr("app.core.router.parse_jev_response", broken_parser)
    harness = Harness(EASY_JEV, [ok(CHEAP[0])])

    with pytest.raises(type(error), match="parser bug"):
        harness.run()

    assert harness.llm.calls == []


@pytest.mark.parametrize("jev_seconds", [48.5, 50.0, 65.0], ids=["leaves-1.5s", "exactly-all", "overruns"])
@pytest.mark.parametrize("jev_outcome", [EASY_JEV, HARD_JEV], ids=["cheap-route", "strong-route"])
def test_jev_using_the_whole_deadline_prevents_any_model_call(jev_outcome, jev_seconds):
    harness = Harness(jev_outcome, [ok(CHEAP[0])], jev_seconds=jev_seconds)

    outcome = harness.run()

    assert harness.llm.calls == []
    assert harness.sleep.delays == []
    assert (outcome.status, outcome.failure_reason) == ("failed", "deadline")
    assert [attempt.status for attempt in outcome.attempts] == ["deadline"]
    assert outcome.attempts[0].model == outcome.decision.model_chain[0]


def test_a_jev_timeout_at_the_default_8s_is_charged_to_the_shared_50s_deadline():
    # Jev uses its full default timeout, then every model call times out at its own limit.
    jev_timeout = Settings().jev_timeout_seconds
    script = [(LLMTimeoutError("m: request timed out"), 20.0) for _ in range(12)]
    harness = Harness(JevTimeoutError("timed out"), script, jev_seconds=jev_timeout)

    outcome = harness.run()

    assert jev_timeout == 8.0
    assert harness.deps.request_budget_seconds == 50.0
    assert (outcome.status, outcome.failure_reason) == ("failed", "deadline")
    # The first call gets the full 20 s cap: 50 - 8 - 2 leaves room for it.
    assert harness.llm.calls[0].timeout == 20.0
    # The deadline is not reset or extended after Jev: elapsed time never exceeds the 50 s budget.
    assert outcome.total_latency_ms <= 50_000
    assert sum(call.timeout for call in harness.llm.calls) + jev_timeout + sum(harness.sleep.delays) <= 50.0
    assert outcome.attempts[-1].status == "deadline"


def test_slow_jev_shrinks_every_later_model_timeout():
    harness = Harness(EASY_JEV, [ok(CHEAP[0])], jev_seconds=8.0)
    harness.deps = dataclasses.replace(harness.deps, request_budget_seconds=20.0)

    harness.run()

    # 20 s budget - 8 s of Jev - 2 s margin
    assert harness.llm.calls[0].timeout == 10.0


# --- S1.8: cost metadata --------------------------------------------------------------------


def test_successful_outcome_has_a_cost_estimate_for_the_final_model():
    from app.core.costs import estimate_cost

    harness = Harness(HARD_JEV, [client_error(), client_error(), ok(CHEAP[0])])

    outcome = harness.run()

    assert outcome.cost == estimate_cost(CONFIG, CHEAP[0], 5, 7)


def test_dry_run_and_failed_outcomes_have_no_cost():
    assert Harness(EASY_JEV).run(dry_run=True).cost is None
    assert Harness(EASY_JEV, [client_error(), client_error()]).run().cost is None


def test_cost_error_leaves_cost_unknown_and_is_logged(monkeypatch, caplog):
    from app.core.costs import CostError

    def failing_estimate(*args):
        raise CostError(f"cannot estimate {PROMPT}")

    monkeypatch.setattr("app.core.router.estimate_cost", failing_estimate)
    harness = Harness(EASY_JEV, [ok(CHEAP[0])])

    with caplog.at_level(logging.WARNING, logger="app.core.router"):
        outcome = harness.run()

    assert outcome.status == "success"
    assert outcome.cost is None
    assert "error_type=CostError" in caplog.text
    assert PROMPT not in caplog.text


def test_cli_prints_the_cost(capsys):
    harness = Harness(EASY_JEV, [ok(CHEAP[0])])

    main([PROMPT], deps_factory=fake_deps_factory(harness))

    cost = json.loads(capsys.readouterr().out)["cost"]
    assert cost["model"] == CHEAP[0]
    assert cost["basis"] == "list_price_estimate"
    assert cost["includes_jev"] is False


# --- HTTP status diagnostics ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("jev_outcome", "expected"),
    [
        (JevAuthError(f"Unauthorized {FAKE_KEY}", status_code=401), 401),
        (JevClientError("JevStation client error (HTTP 403): body", status_code=403), 403),
        (JevCreditError("Payment required", status_code=402), 402),
        (JevRateLimitError("rate limited"), 429),
        (JevResponseError("JevStation server error (HTTP 502)", status_code=502), 502),
        (JevClientError("JevStation client error (HTTP 404): body", status_code=404), 404),
        (JevTimeoutError(f"timed out {PROMPT}"), None),
        (JevClientError(f"Network error communicating with JevStation: {FAKE_KEY}"), None),
        (ValueError(f"State length exceeds maximum: {PROMPT}"), None),
        ({"data": {}}, None),
    ],
    ids=["401", "403", "402", "429", "502", "404", "timeout", "network", "prompt-too-long", "malformed"],
)
def test_jev_failure_records_its_http_status(jev_outcome, expected):
    outcome = Harness(jev_outcome, [ok(STRONG[0])]).run()

    assert outcome.decision.reason == "JEV_ERROR"
    assert outcome.jev_http_status == expected


def test_jev_network_failure_records_the_cause_class_only(caplog):
    error = JevTimeoutError(f"timed out https://jevstation.example {PROMPT} {FAKE_KEY}", cause_type="ReadTimeout")

    with caplog.at_level(logging.DEBUG):
        outcome = Harness(error, [ok(STRONG[0])]).run()

    assert (outcome.decision.jev_error, outcome.jev_error_cause) == ("JevTimeoutError", "ReadTimeout")
    assert "jev_error_cause=ReadTimeout" in caplog.text
    (record,) = [r for r in caplog.records if r.name == "app.core.router"]
    assert record.jev_error_cause == "ReadTimeout"
    for forbidden in (PROMPT, "PROMPT-MARKER", FAKE_KEY, "https://", "timed out"):
        assert forbidden not in caplog.text


@pytest.mark.parametrize(
    "jev_outcome",
    [
        EASY_JEV,
        JevAuthError("Unauthorized", status_code=401),
        JevTimeoutError("timed out"),  # raised without a recorded cause
        ValueError("State length exceeds maximum"),
        {"data": {}},
    ],
    ids=["success", "http-401", "no-cause", "prompt-too-long", "malformed"],
)
def test_jev_error_cause_is_none_without_a_network_cause(jev_outcome):
    assert Harness(jev_outcome, [ok(STRONG[0])]).run().jev_error_cause is None


def test_route_outcome_jev_error_cause_defaults_to_none():
    assert RouteOutcome.__dataclass_fields__["jev_error_cause"].default is None


def test_cli_summary_reports_jev_error_cause(capsys):
    error = JevClientError(f"Network error communicating with JevStation: {FAKE_KEY}", cause_type="ConnectError")
    harness = Harness(error, [ok(STRONG[0])])

    main([PROMPT], deps_factory=fake_deps_factory(harness))

    out = capsys.readouterr().out
    assert json.loads(out)["jev_error_cause"] == "ConnectError"
    assert FAKE_KEY not in out
    assert "Network error" not in out


def test_successful_jev_call_has_no_http_status():
    assert Harness(EASY_JEV, [ok(CHEAP[0])]).run().jev_http_status is None


def test_route_outcome_jev_http_status_defaults_to_none():
    assert RouteOutcome.__dataclass_fields__["jev_http_status"].default is None


def test_failed_outcome_attempts_carry_http_statuses():
    harness = Harness(
        EASY_JEV,
        [
            LLMServerError("m: HTTP 500: server error", status_code=500),
            LLMTimeoutError("m: request timed out"),
            LLMServerError("m: network error (ConnectError)"),
            LLMAuthError("m: HTTP 401: authentication or permission failure", status_code=401),
        ],
    )

    outcome = harness.run()

    assert (outcome.status, outcome.failure_reason) == ("failed", "fatal")
    assert [(a.status, a.http_status) for a in outcome.attempts] == [
        ("server_error", 500),
        ("timeout", None),
        ("server_error", None),
        ("auth_error", 401),
    ]


def test_cli_summary_reports_http_statuses_without_secrets(capsys):
    jev_error = JevClientError(f"JevStation client error (HTTP 403): {FAKE_KEY} {PROMPT}", status_code=403)
    harness = Harness(jev_error, [LLMAuthError("m: HTTP 401: authentication or permission failure", status_code=401)])

    exit_code = main([PROMPT], deps_factory=fake_deps_factory(harness))

    out = capsys.readouterr().out
    printed = json.loads(out)
    assert exit_code == 1
    assert printed["jev_http_status"] == 403
    assert printed["decision"]["jev_error"] == "JevClientError"
    assert [(a["status"], a["http_status"]) for a in printed["attempts"]] == [("auth_error", 401)]
    for forbidden in (FAKE_KEY, "PROMPT-MARKER", "authentication or permission failure", "client error"):
        assert forbidden not in out


def test_cli_summary_jev_http_status_is_null_on_success(capsys):
    harness = Harness(EASY_JEV, [ok(CHEAP[0])])

    main([PROMPT], deps_factory=fake_deps_factory(harness))

    printed = json.loads(capsys.readouterr().out)
    assert printed["jev_http_status"] is None
    assert printed["attempts"][0]["http_status"] == 200


def test_cli_summary_reports_reasoning_and_malformed_usage_tokens(capsys):
    empty = LLMMalformedResponseError("m: malformed response", status_code=200, input_tokens=9, output_tokens=64)
    answer = dataclasses.replace(ok(CHEAP[1]), reasoning_tokens=3)
    harness = Harness(EASY_JEV, [empty, answer])

    main([PROMPT], deps_factory=fake_deps_factory(harness))

    printed = json.loads(capsys.readouterr().out)
    assert [
        (a["status"], a["input_tokens"], a["output_tokens"], a["reasoning_tokens"]) for a in printed["attempts"]
    ] == [
        ("malformed", 9, 64, None),
        ("success", ok(CHEAP[1]).input_tokens, ok(CHEAP[1]).output_tokens, 3),
    ]


# --- S2.9: no strong reservation when no model call can start (D14: keep the decision) ------

CALL_START_SECONDS = MIN_CALL_SECONDS + CALL_MARGIN_SECONDS  # The executor's minimum remaining time


@pytest.mark.parametrize(
    ("jev_outcome", "reason"),
    [(HARD_JEV, "HARD"), (LOW_CONFIDENCE_JEV, "LOW_CONFIDENCE"), (JevTimeoutError("timed out"), "JEV_ERROR")],
    ids=["hard", "low-confidence", "jev-error"],
)
def test_strong_route_does_not_reserve_when_no_model_call_can_start(jev_outcome, reason):
    # The plan's case: Jev takes 48.5 s of the 50 s budget, leaving 1.5 s (< 4 s needed to start a call).
    harness = Harness(jev_outcome, [ok(STRONG[0])], jev_seconds=48.5)

    outcome = harness.run()

    assert (outcome.decision.tier, outcome.decision.reason) == ("strong", reason)  # Kept, not recomputed (D14)
    assert outcome.strong_reserved is False
    assert harness.budget.reserve_calls == 0
    assert harness.budget.available_calls == 1  # Unchanged: availability is still read once to decide
    assert len(harness.jev.calls) == 1
    assert harness.llm.calls == []
    assert harness.sleep.delays == []
    assert (outcome.status, outcome.failure_reason, outcome.execution) == ("failed", "deadline", None)
    assert [(a.tier, a.model, a.status) for a in outcome.attempts] == [
        ("strong", outcome.decision.model_chain[0], "deadline")
    ]
    # The harness's request ID is not a UUID; the mapping requires one.
    rows = outcome_to_rows(dataclasses.replace(outcome, request_id=str(uuid.uuid4())), prompt=PROMPT)
    assert {key: rows.request[key] for key in ("status", "failure_reason", "strong_reserved", "tier", "reason")} == {
        "status": "failed",
        "failure_reason": "deadline",
        "strong_reserved": False,
        "tier": "strong",
        "reason": reason,
    }


@pytest.mark.parametrize(
    ("remaining", "can_start"),
    [(CALL_START_SECONDS - 0.001, False), (CALL_START_SECONDS, True), (CALL_START_SECONDS + 0.001, True)],
    ids=["just-below", "exactly", "just-above"],
)
def test_strong_reservation_boundary_matches_the_executor(remaining, can_start):
    harness = Harness(HARD_JEV, [ok(STRONG[0])], jev_seconds=50.0 - remaining)

    outcome = harness.run()

    assert harness.budget.reserve_calls == (1 if can_start else 0)
    assert outcome.strong_reserved is can_start
    assert harness.llm.models_called == ([STRONG[0]] if can_start else [])  # The executor agrees
    assert outcome.status == ("success" if can_start else "failed")


def test_strong_route_with_time_left_reserves_once_and_runs_as_before():
    harness = Harness(HARD_JEV, [ok(STRONG[0])], jev_seconds=40.0)  # 10 s left

    outcome = harness.run()

    assert (outcome.decision.tier, outcome.decision.reason) == ("strong", "HARD")
    assert (outcome.status, outcome.strong_reserved) == ("success", True)
    assert harness.budget.reserve_calls == 1
    assert harness.llm.models_called == [STRONG[0]]


def test_no_time_left_does_not_turn_a_reservation_race_into_a_reservation():
    # Budget lost the race (reserve() would say False) but no call can start anyway: reserve() is not called.
    harness = Harness(HARD_JEV, [ok(CHEAP[0])], budget=FakeBudget(available=True, reserve=False), jev_seconds=48.5)

    outcome = harness.run()

    assert harness.budget.reserve_calls == 0
    assert (outcome.decision.tier, outcome.decision.reason, outcome.strong_reserved) == ("strong", "HARD", False)
    assert outcome.failure_reason == "deadline"


def test_cheap_route_without_time_left_still_never_reserves_or_escalates():
    harness = Harness(EASY_JEV, [ok(CHEAP[0])], jev_seconds=48.5)

    outcome = harness.run()

    assert harness.budget.reserve_calls == 0
    assert (outcome.decision.tier, outcome.strong_reserved, outcome.failure_reason) == ("cheap", False, "deadline")
    assert [a.tier for a in outcome.attempts] == ["cheap"]


class SlowReserveBudget(FakeBudget):
    """FakeBudget whose successful reserve() takes `reserve_seconds` of the shared fake clock."""

    def __init__(self, reserve_seconds: float):
        super().__init__(available=True, reserve=True)
        self.reserve_seconds = reserve_seconds
        self.clock: FakeClock | None = None  # Set to the harness clock after construction

    async def reserve(self) -> bool:
        self.clock.advance(self.reserve_seconds)
        return await super().reserve()


def test_residual_race_a_slow_successful_reservation_can_leave_no_time_for_a_call():
    # Documents a known, accepted limitation of S2.9: the check runs before reserve(), so time spent inside
    # a slow reserve() is not covered. 5 s remain at the check (enough), reserve() takes 1.5 s, the
    # executor then sees 3.5 s (< 4 s) and makes no call. The slot is spent and reported as such.
    budget = SlowReserveBudget(reserve_seconds=1.5)
    harness = Harness(HARD_JEV, [ok(STRONG[0])], budget=budget, jev_seconds=50.0 - (CALL_START_SECONDS + 1.0))
    budget.clock = harness.clock

    outcome = harness.run()

    assert harness.budget.reserve_calls == 1
    assert harness.llm.calls == []
    assert harness.sleep.delays == []
    assert (outcome.decision.tier, outcome.decision.reason) == ("strong", "HARD")
    assert outcome.strong_reserved is True  # The reservation succeeded, so the outcome must say so
    assert (outcome.status, outcome.failure_reason, outcome.execution) == ("failed", "deadline", None)
    assert [(a.tier, a.model, a.status) for a in outcome.attempts] == [("strong", STRONG[0], "deadline")]
