import asyncio
import copy
import dataclasses

import httpx
import pytest

from app.core.deadline import Deadline
from app.core.executor import (
    CALL_MARGIN_SECONDS,
    MIN_CALL_SECONDS,
    Attempt,
    ChainExhausted,
    ExecutionResult,
    can_start_call,
    execute,
)
from app.core.llm_client import (
    LLMAuthError,
    LLMBudgetExhaustedError,
    LLMClientError,
    LLMError,
    LLMMalformedResponseError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
    ModelResult,
)
from app.core.model_config import Limits, load_model_config
from app.core.policy import RouteDecision
from tests.fakes import FakeClock, FakeLLM, RecordingSleep

PROMPT = "Reverse a string in Python"
CONFIG = load_model_config()
CHEAP = CONFIG.routing.get_chain("general", "cheap")
STRONG = CONFIG.routing.get_chain("general", "strong")


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    async def no_network(*args, **kwargs):
        raise AssertionError("executor tests must not reach the network")

    monkeypatch.setattr(httpx.AsyncClient, "send", no_network)


def make_route(tier: str) -> RouteDecision:
    chain = STRONG if tier == "strong" else CHEAP
    return RouteDecision(
        tier=tier, task_type="general", reason="HARD" if tier == "strong" else "EASY", model_chain=chain
    )


def ok(model: str, input_tokens: int = 11, output_tokens: int = 22, http_status: int = 200) -> ModelResult:
    return ModelResult(
        text=f"answer from {model}",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=1,
        finish_reason="stop",
        http_status=http_status,
    )


def rate_limited() -> LLMRateLimitError:
    return LLMRateLimitError("m: HTTP 429: rate limited", retry_after=30.0)


def server_error() -> LLMServerError:
    return LLMServerError("m: HTTP 500: server error", status_code=500)


def timed_out() -> LLMTimeoutError:
    return LLMTimeoutError("m: request timed out")


def client_error() -> LLMClientError:
    return LLMClientError("m: HTTP 400: request rejected", status_code=400)


def malformed() -> LLMMalformedResponseError:
    return LLMMalformedResponseError("m: malformed response (empty completion)", status_code=200)


class Harness:
    def __init__(self, script, budget: float = 50.0, config=CONFIG):
        self.clock = FakeClock()
        self.deadline = Deadline(budget, clock=self.clock)
        self.sleep = RecordingSleep(self.clock)
        self.llm = FakeLLM(script, self.clock)
        self.config = config

    def run(self, route: RouteDecision) -> ExecutionResult:
        return asyncio.run(
            execute(PROMPT, route, self.config, self.llm, self.deadline, sleep=self.sleep, clock=self.clock)
        )

    def run_failing(self, route: RouteDecision) -> ChainExhausted:
        with pytest.raises(ChainExhausted) as excinfo:
            self.run(route)
        return excinfo.value


def summary(attempts) -> list:
    return [(a.tier, a.model, a.attempt_no, a.status) for a in attempts]


# --- Success and retries --------------------------------------------------------------------


def test_first_success_one_attempt():
    harness = Harness([ok(CHEAP[0])])

    result = harness.run(make_route("cheap"))

    assert result.final_model == CHEAP[0]
    assert result.final_tier == "cheap"
    assert result.fell_back_to_cheap is False
    assert result.result.text == f"answer from {CHEAP[0]}"
    assert summary(result.attempts) == [("cheap", CHEAP[0], 1, "success")]
    assert harness.sleep.delays == []
    (call,) = harness.llm.calls
    assert call.prompt == PROMPT
    assert call.max_tokens == CONFIG.catalog[CHEAP[0]].max_output_tokens


def test_retries_rate_limit_then_succeeds():
    harness = Harness([rate_limited(), ok(CHEAP[0])])

    result = harness.run(make_route("cheap"))

    assert summary(result.attempts) == [
        ("cheap", CHEAP[0], 1, "rate_limited"),
        ("cheap", CHEAP[0], 2, "success"),
    ]
    assert harness.sleep.delays == [1.0]


def test_server_error_and_timeout_retry_with_1s_then_2s_backoff():
    harness = Harness([server_error(), timed_out(), ok(CHEAP[0])])

    result = harness.run(make_route("cheap"))

    assert summary(result.attempts) == [
        ("cheap", CHEAP[0], 1, "server_error"),
        ("cheap", CHEAP[0], 2, "timeout"),
        ("cheap", CHEAP[0], 3, "success"),
    ]
    assert harness.sleep.delays == [1.0, 2.0]


def test_falls_back_to_next_model_after_retries_without_extra_sleep():
    harness = Harness([server_error(), server_error(), server_error(), ok(CHEAP[1])])

    result = harness.run(make_route("cheap"))

    assert harness.llm.models_called == [CHEAP[0]] * 3 + [CHEAP[1]]
    assert result.final_model == CHEAP[1]
    assert harness.sleep.delays == [1.0, 2.0]


def test_backoff_beyond_schedule_repeats_last_delay():
    config = dataclasses.replace(CONFIG, limits=Limits(max_chain_length=2, max_attempts_per_model=4))
    harness = Harness([server_error()] * 3 + [ok(CHEAP[0])], config=config)

    harness.run(make_route("cheap"))

    assert harness.sleep.delays == [1.0, 2.0, 2.0]


@pytest.mark.parametrize(
    ("error", "status"), [(client_error, "client_error"), (malformed, "malformed")], ids=["client", "malformed"]
)
def test_permanent_errors_skip_retries(error, status):
    harness = Harness([error(), ok(CHEAP[1])])

    result = harness.run(make_route("cheap"))

    assert summary(result.attempts) == [("cheap", CHEAP[0], 1, status), ("cheap", CHEAP[1], 1, "success")]
    assert harness.sleep.delays == []


# --- Fallback between tiers ------------------------------------------------------------------


def test_strong_chain_exhausted_falls_back_to_cheap():
    route = make_route("strong")
    harness = Harness([client_error(), malformed(), ok(CHEAP[0])])

    result = harness.run(route)

    assert summary(result.attempts) == [
        ("strong", STRONG[0], 1, "client_error"),
        ("strong", STRONG[1], 1, "malformed"),
        ("cheap", CHEAP[0], 1, "success"),
    ]
    assert result.final_tier == "cheap"
    assert result.final_model == CHEAP[0]
    assert result.fell_back_to_cheap is True
    assert result.route.tier == "strong"


def test_strong_success_does_not_touch_cheap_chain():
    result = Harness([ok(STRONG[0])]).run(make_route("strong"))

    assert (result.final_tier, result.fell_back_to_cheap) == ("strong", False)


def test_cheap_chain_exhausted_never_tries_strong():
    harness = Harness([client_error(), client_error()])

    exhausted = harness.run_failing(make_route("cheap"))

    assert harness.llm.models_called == list(CHEAP)
    assert not set(harness.llm.models_called) & set(STRONG)
    assert exhausted.reason == "all_failed"
    assert summary(exhausted.attempts) == [
        ("cheap", CHEAP[0], 1, "client_error"),
        ("cheap", CHEAP[1], 1, "client_error"),
    ]


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (LLMAuthError("m: HTTP 401", status_code=401), "auth_error"),
        (LLMBudgetExhaustedError("m: HTTP 402", status_code=402), "budget_exhausted"),
    ],
    ids=["auth", "budget"],
)
def test_fatal_errors_abort_the_whole_chain_including_fallback(error, status):
    harness = Harness([error])

    exhausted = harness.run_failing(make_route("strong"))

    assert exhausted.reason == "fatal"
    assert summary(exhausted.attempts) == [("strong", STRONG[0], 1, status)]
    assert len(harness.llm.calls) == 1
    assert harness.sleep.delays == []


# --- Deadline --------------------------------------------------------------------------------


def test_deadline_too_short_prevents_any_call():
    harness = Harness([], budget=2 + MIN_CALL_SECONDS - 0.5)

    exhausted = harness.run_failing(make_route("strong"))

    assert harness.llm.calls == []
    assert exhausted.reason == "deadline"
    assert summary(exhausted.attempts) == [("strong", STRONG[0], 1, "deadline")]


def test_no_backoff_when_insufficient_time_remains_moves_to_next_model():
    # Budget 6 s; the first call takes 2 s -> 4 s left. Backoff 1 s + 2 s margin leaves 1 s,
    # below MIN_CALL_SECONDS, so no sleep. The next model can still be called with 4 - 2 = 2 s.
    harness = Harness([(server_error(), 2.0), ok(CHEAP[1])], budget=6.0)

    result = harness.run(make_route("cheap"))

    assert harness.sleep.delays == []
    assert harness.llm.models_called == [CHEAP[0], CHEAP[1]]
    assert harness.llm.calls[1].timeout == 2.0
    assert result.final_model == CHEAP[1]


def test_deadline_reached_mid_chain_stops_without_further_calls():
    harness = Harness([(client_error(), 47.0)], budget=50.0)

    exhausted = harness.run_failing(make_route("strong"))

    assert exhausted.reason == "deadline"
    assert summary(exhausted.attempts) == [
        ("strong", STRONG[0], 1, "client_error"),
        ("strong", STRONG[1], 1, "deadline"),
    ]
    assert len(harness.llm.calls) == 1


def test_call_timeout_capped_at_20s_and_by_remaining_budget():
    # 50 s budget: first timeout is 20 s. That call takes 35 s, backoff 1 s -> 14 s left -> 12 s.
    harness = Harness([(server_error(), 35.0), ok(CHEAP[0])], budget=50.0)

    harness.run(make_route("cheap"))

    assert [call.timeout for call in harness.llm.calls] == [20.0, 12.0]


def test_deadline_is_shared_across_fallback_and_never_reset():
    # Two strong failures take 10 s each from a 30 s budget; the cheap call gets 30 - 20 - 2 = 8 s.
    harness = Harness([(client_error(), 10.0), (client_error(), 10.0), ok(CHEAP[0])], budget=30.0)

    harness.run(make_route("strong"))

    assert [call.timeout for call in harness.llm.calls] == [20.0, 18.0, 8.0]
    assert harness.deadline.remaining() == 10.0


# --- Result integrity ------------------------------------------------------------------------


def test_original_route_decision_is_unchanged_and_returned():
    route = make_route("strong")
    before = copy.deepcopy(route)
    harness = Harness([client_error(), client_error(), ok(CHEAP[0])])

    result = harness.run(route)

    assert result.route is route
    assert route == before


def test_attempt_history_is_complete_and_ordered():
    harness = Harness(
        [
            (rate_limited(), 0.5),
            (server_error(), 0.25),
            (timed_out(), 3.0),
            (client_error(), 0.1),
            (ok(CHEAP[0], input_tokens=7, output_tokens=9), 1.2),
        ]
    )

    result = harness.run(make_route("strong"))

    assert result.attempts == (
        Attempt("strong", STRONG[0], 1, "rate_limited", 500, None, None, http_status=429),
        Attempt("strong", STRONG[0], 2, "server_error", 250, None, None, http_status=500),
        Attempt("strong", STRONG[0], 3, "timeout", 3000, None, None),
        Attempt("strong", STRONG[1], 1, "client_error", 100, None, None, http_status=400),
        Attempt("cheap", CHEAP[0], 1, "success", 1200, 7, 9, http_status=200),
    )


def test_maximum_attempt_count_is_enforced():
    max_calls = (len(STRONG) + len(CHEAP)) * CONFIG.limits.max_attempts_per_model
    assert (len(STRONG), len(CHEAP), CONFIG.limits.max_attempts_per_model, max_calls) == (2, 2, 3, 12)
    harness = Harness([server_error()] * max_calls)

    exhausted = harness.run_failing(make_route("strong"))

    assert len(harness.llm.calls) == max_calls
    assert len(exhausted.attempts) == max_calls
    assert exhausted.reason == "all_failed"
    for model in STRONG + CHEAP:
        assert harness.llm.models_called.count(model) == 3
    assert harness.sleep.delays == [1.0, 2.0] * 4


def test_unexpected_exception_is_not_swallowed():
    harness = Harness([RuntimeError("bug in client")])

    with pytest.raises(RuntimeError, match="bug in client"):
        harness.run(make_route("cheap"))


def test_unclassified_llm_error_fails_loudly():
    harness = Harness([LLMError("m: new failure kind")])

    with pytest.raises(TypeError, match="Unclassified LLMError"):
        harness.run(make_route("cheap"))


def test_chain_exhausted_message_has_no_prompt():
    exhausted = Harness([client_error(), client_error()]).run_failing(make_route("cheap"))

    assert PROMPT not in str(exhausted)
    assert "all_failed" in str(exhausted)


# --- HTTP status recording ------------------------------------------------------------------


def first_attempt(script) -> Attempt:
    harness = Harness(script)
    try:
        attempts = harness.run(make_route("cheap")).attempts
    except ChainExhausted as exc:
        attempts = exc.attempts
    return attempts[0]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (LLMAuthError("m: HTTP 401", status_code=401), 401),
        (LLMAuthError("m: HTTP 403", status_code=403), 403),
        (LLMBudgetExhaustedError("m: HTTP 402", status_code=402), 402),
        (LLMRateLimitError("m: HTTP 429: rate limited"), 429),
        (LLMServerError("m: HTTP 500: server error", status_code=500), 500),
        (LLMServerError("m: HTTP 503: server error", status_code=503), 503),
        (LLMClientError("m: HTTP 404: request rejected", status_code=404), 404),
        (LLMClientError("m: HTTP 422: request rejected", status_code=422), 422),
        (LLMMalformedResponseError("m: malformed response (missing usage)", status_code=200), 200),
        (LLMTimeoutError("m: request timed out after 20s"), None),
        (LLMServerError("m: network error (ConnectError)"), None),
    ],
    ids=["401", "403", "402", "429", "500", "503", "404", "422", "malformed-200", "timeout", "network"],
)
def test_failed_attempt_records_the_http_status(error, expected):
    attempt = first_attempt([error, ok(CHEAP[0])])

    assert attempt.status != "success"
    assert attempt.http_status == expected


@pytest.mark.parametrize("status", [200, 201])
def test_successful_attempt_records_the_actual_http_status(status):
    attempt = first_attempt([ok(CHEAP[0], http_status=status)])

    assert (attempt.status, attempt.http_status) == ("success", status)


def test_deadline_attempt_has_no_http_status():
    harness = Harness([ok(CHEAP[0])], budget=MIN_CALL_SECONDS)

    exhausted = harness.run_failing(make_route("cheap"))

    assert [(a.status, a.http_status) for a in exhausted.attempts] == [("deadline", None)]


def test_attempt_http_status_is_optional_and_last():
    attempt = Attempt("cheap", CHEAP[0], 1, "success", 10, 1, 2)

    assert attempt.http_status is None
    assert [field.name for field in dataclasses.fields(Attempt)][-1] == "http_status"


def test_successful_attempt_records_reasoning_tokens():
    result = dataclasses.replace(ok(CHEAP[0], output_tokens=200), reasoning_tokens=180)

    attempt = first_attempt([result])

    assert (attempt.status, attempt.output_tokens, attempt.reasoning_tokens) == ("success", 200, 180)


def test_successful_attempt_reasoning_tokens_are_none_when_unknown():
    attempt = first_attempt([ok(CHEAP[0])])

    assert attempt.reasoning_tokens is None


def test_malformed_attempt_records_usage_tokens_and_moves_on_without_retrying():
    error = LLMMalformedResponseError(
        "m: malformed response (empty completion)", status_code=200, input_tokens=12, output_tokens=256
    )
    harness = Harness([error, ok(CHEAP[1])])

    result = harness.run(make_route("cheap"))

    first, second = result.attempts
    assert (first.model, first.status, first.input_tokens, first.output_tokens) == (CHEAP[0], "malformed", 12, 256)
    assert first.reasoning_tokens is None
    assert (second.model, second.status) == (CHEAP[1], "success")
    assert harness.llm.models_called == [CHEAP[0], CHEAP[1]]
    assert harness.sleep.delays == []


def test_malformed_attempt_without_usage_has_null_tokens():
    attempt = first_attempt([malformed(), ok(CHEAP[1])])

    assert (attempt.status, attempt.input_tokens, attempt.output_tokens) == ("malformed", None, None)


def test_non_malformed_failures_never_record_tokens():
    attempt = first_attempt([client_error(), ok(CHEAP[1])])

    assert (attempt.input_tokens, attempt.output_tokens, attempt.reasoning_tokens) == (None, None, None)


# --- S2.9: can_start_call, the executor's own start rule ------------------------------------

CALL_START_SECONDS = MIN_CALL_SECONDS + CALL_MARGIN_SECONDS


@pytest.mark.parametrize(
    ("budget", "expected"),
    [
        (CALL_START_SECONDS - 0.001, False),
        (CALL_START_SECONDS, True),
        (CALL_START_SECONDS + 0.001, True),
        (0.5, False),
        (60.0, True),
    ],
    ids=["just-below", "exactly", "just-above", "far-below", "beyond-max-call"],
)
def test_can_start_call_needs_the_minimum_call_plus_the_margin(budget, expected):
    assert can_start_call(Deadline(budget, clock=FakeClock())) is expected


def test_can_start_call_follows_the_shared_clock_and_is_false_once_expired():
    clock = FakeClock()
    deadline = Deadline(10.0, clock=clock)

    clock.advance(10.0 - CALL_START_SECONDS)
    assert can_start_call(deadline) is True
    clock.advance(0.001)
    assert can_start_call(deadline) is False
    clock.advance(100.0)
    assert can_start_call(deadline) is False


@pytest.mark.parametrize(
    "budget",
    [CALL_START_SECONDS - 0.001, CALL_START_SECONDS, CALL_START_SECONDS + 0.001],
    ids=["just-below", "exactly", "just-above"],
)
@pytest.mark.parametrize("tier", ["cheap", "strong"])
def test_can_start_call_agrees_with_whether_the_executor_starts_a_call(budget, tier):
    harness = Harness([ok(STRONG[0] if tier == "strong" else CHEAP[0])], budget=budget)
    predicted = can_start_call(harness.deadline)

    if predicted:
        harness.run(make_route(tier))
    else:
        assert harness.run_failing(make_route(tier)).reason == "deadline"

    assert (len(harness.llm.calls) == 1) is predicted
