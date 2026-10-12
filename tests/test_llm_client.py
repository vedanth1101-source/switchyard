import asyncio
import json
import math
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from app.core.llm_client import (
    FATAL_ERRORS,
    RETRYABLE_ERRORS,
    GatewayClient,
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

BASE_URL = "https://gateway.test/v1"
FAKE_KEY = "fake-gateway-key-0123456789"
PROMPT = "reverse the string PROMPT-MARKER in Python"
MODEL = "vendor/model-a"
BODY_MARKER = "BODY-MARKER"
FIXTURES = Path(__file__).parent / "fixtures"


def success_body(content="def reverse(s): return s[::-1]", finish_reason="stop", usage=None) -> dict:
    return {
        "id": "chatcmpl-1",
        "model": "provider-echoed-name",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
        "usage": usage if usage is not None else {"prompt_tokens": 12, "completion_tokens": 34, "total_tokens": 46},
    }


class SequenceClock:
    def __init__(self, *values: float):
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class Recorder:
    """MockTransport handler that records requests and returns a fixed response."""

    def __init__(self, response: httpx.Response | Exception):
        self.response = response
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def make_client(handler, clock=None, base_url: str = BASE_URL) -> GatewayClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    if clock is None:
        return GatewayClient(base_url, SecretStr(FAKE_KEY), client=http)
    return GatewayClient(base_url, SecretStr(FAKE_KEY), client=http, clock=clock)


def complete(client: GatewayClient, *, max_tokens=256, timeout=5.0) -> ModelResult:
    async def run() -> ModelResult:
        async with client:
            return await client.complete(MODEL, PROMPT, max_tokens, timeout)

    return asyncio.run(run())


def failing_call(response: httpx.Response | Exception) -> LLMError:
    recorder = Recorder(response)
    with pytest.raises(LLMError) as excinfo:
        complete(make_client(recorder))
    assert len(recorder.requests) == 1  # never retried inside the client
    return excinfo.value


def error_response(status: int, headers: dict | None = None) -> httpx.Response:
    body = {"error": {"message": f"echo {PROMPT} {FAKE_KEY} {BODY_MARKER}", "type": "some_error"}}
    return httpx.Response(status, json=body, headers=headers)


# --- Success ---------------------------------------------------------------------------------


def test_returns_text_token_usage_latency_and_finish_reason():
    recorder = Recorder(httpx.Response(200, json=success_body()))

    result = complete(make_client(recorder, clock=SequenceClock(10.0, 10.25)))

    assert result == ModelResult(
        text="def reverse(s): return s[::-1]",
        model=MODEL,
        input_tokens=12,
        output_tokens=34,
        latency_ms=250,
        finish_reason="stop",
        http_status=200,
    )


def test_parses_the_real_gateway_response_fixture():
    # Captured from the live gateway (sanitized): extra fields such as reasoning_details,
    # provider_metadata, generationId and the extended usage object must not break parsing.
    body = json.loads((FIXTURES / "gateway_response_success.json").read_text(encoding="utf-8"))
    recorder = Recorder(httpx.Response(200, json=body))

    result = complete(make_client(recorder))

    assert result.text == "<redacted>"
    assert (result.input_tokens, result.output_tokens, result.reasoning_tokens) == (13, 115, 64)
    assert (result.finish_reason, result.http_status) == ("stop", 200)
    assert result.model == MODEL  # the requested config id, not the provider's echoed name


def test_success_records_the_actual_2xx_status():
    recorder = Recorder(httpx.Response(201, json=success_body()))

    assert complete(make_client(recorder)).http_status == 201


def test_sends_openai_compatible_chat_completion_request():
    recorder = Recorder(httpx.Response(200, json=success_body()))

    complete(make_client(recorder), max_tokens=321)

    (request,) = recorder.requests
    assert request.method == "POST"
    assert str(request.url) == f"{BASE_URL}/chat/completions"
    assert request.headers["Authorization"] == f"Bearer {FAKE_KEY}"
    assert json.loads(request.content) == {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 321,
    }


def test_trailing_slash_in_base_url_is_not_doubled():
    recorder = Recorder(httpx.Response(200, json=success_body()))

    complete(make_client(recorder, base_url=f"{BASE_URL}/"))

    assert str(recorder.requests[0].url) == f"{BASE_URL}/chat/completions"


def test_null_finish_reason_is_allowed():
    result = complete(make_client(Recorder(httpx.Response(200, json=success_body(finish_reason=None)))))

    assert result.finish_reason is None


# --- Timeouts --------------------------------------------------------------------------------


def test_passes_timeout():
    recorder = Recorder(httpx.Response(200, json=success_body()))

    complete(make_client(recorder), timeout=7.5)

    assert recorder.requests[0].extensions["timeout"] == {"connect": 7.5, "read": 7.5, "write": 7.5, "pool": 7.5}


def test_httpx_timeout_raises_timeout_error():
    error = failing_call(httpx.ReadTimeout("read timed out"))

    assert isinstance(error, LLMTimeoutError)
    assert error.status_code is None


def test_slow_response_is_cut_off_at_the_timeout():
    async def slow_handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200, json=success_body())

    with pytest.raises(LLMTimeoutError):
        complete(make_client(slow_handler), timeout=0.05)


# --- HTTP error mapping ----------------------------------------------------------------------


def test_maps_429_to_rate_limited_with_retry_after():
    error = failing_call(error_response(429, headers={"Retry-After": "3"}))

    assert isinstance(error, LLMRateLimitError)
    assert error.status_code == 429
    assert error.retry_after == 3.0


@pytest.mark.parametrize("header", [None, "soon", "-1", "Wed, 21 Oct 2026 07:28:00 GMT"])
def test_unusable_retry_after_is_none(header):
    headers = {"Retry-After": header} if header is not None else None

    error = failing_call(error_response(429, headers=headers))

    assert isinstance(error, LLMRateLimitError)
    assert error.retry_after is None


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_maps_5xx_to_server_error(status):
    error = failing_call(error_response(status))

    assert type(error) is LLMServerError
    assert error.status_code == status


def test_network_error_is_a_server_error_without_status():
    error = failing_call(httpx.ConnectError("connection refused"))

    assert type(error) is LLMServerError
    assert error.status_code is None


@pytest.mark.parametrize("status", [400, 404, 409, 413, 422])
def test_other_4xx_is_a_client_error(status):
    error = failing_call(error_response(status))

    assert type(error) is LLMClientError
    assert error.status_code == status


@pytest.mark.parametrize("status", [401, 403])
def test_401_and_403_are_fatal_auth_errors(status):
    error = failing_call(error_response(status))

    assert type(error) is LLMAuthError
    assert error.status_code == status


@pytest.mark.parametrize(
    "body",
    [{"error": {"type": "quota_for_entity_exceeded"}}, {"error": {"type": "insufficient_funds"}}, {}],
    ids=["entity-quota", "no-credits", "no-body"],
)
def test_402_is_fatal_budget_exhaustion(body):
    error = failing_call(httpx.Response(402, json=body))

    assert type(error) is LLMBudgetExhaustedError
    assert error.status_code == 402


def test_quota_wording_in_a_non_402_body_is_not_budget_exhaustion():
    error = failing_call(httpx.Response(400, json={"error": {"type": "quota_for_entity_exceeded"}}))

    assert type(error) is LLMClientError


def test_error_classification_tuples():
    assert set(RETRYABLE_ERRORS) == {LLMRateLimitError, LLMServerError, LLMTimeoutError}
    assert set(FATAL_ERRORS) == {LLMAuthError, LLMBudgetExhaustedError}
    for permanent in (LLMClientError, LLMMalformedResponseError):
        assert not issubclass(permanent, RETRYABLE_ERRORS + FATAL_ERRORS)


# --- Malformed responses ---------------------------------------------------------------------


def with_message(content):
    return success_body(content=content)


def with_usage(usage):
    return success_body(usage=usage)


MALFORMED = [
    pytest.param(httpx.Response(200, content=b"not json PROMPT-MARKER"), id="invalid-json"),
    pytest.param(httpx.Response(200, json=[1, 2]), id="not-an-object"),
    pytest.param(httpx.Response(200, json={"usage": success_body()["usage"]}), id="choices-missing"),
    pytest.param(httpx.Response(200, json={**success_body(), "choices": []}), id="choices-empty"),
    pytest.param(httpx.Response(200, json={**success_body(), "choices": ["x"]}), id="choice-not-object"),
    pytest.param(httpx.Response(200, json={**success_body(), "choices": [{"finish_reason": "stop"}]}), id="no-message"),
    pytest.param(httpx.Response(200, json=with_message(None)), id="content-null"),
    pytest.param(httpx.Response(200, json=with_message("")), id="content-empty"),
    pytest.param(httpx.Response(200, json=with_message("   \n")), id="content-whitespace"),
    pytest.param(httpx.Response(200, json=with_message(["part"])), id="content-not-string"),
    pytest.param(httpx.Response(200, json=success_body(finish_reason=5)), id="finish-reason-not-string"),
    pytest.param(
        httpx.Response(200, json={k: v for k, v in success_body().items() if k != "usage"}), id="usage-missing"
    ),
    pytest.param(httpx.Response(200, json=with_usage({"completion_tokens": 3})), id="prompt-tokens-missing"),
    pytest.param(httpx.Response(200, json=with_usage({"prompt_tokens": 3})), id="completion-tokens-missing"),
    pytest.param(
        httpx.Response(200, json=with_usage({"prompt_tokens": "12", "completion_tokens": 3})), id="tokens-str"
    ),
    pytest.param(httpx.Response(200, json=with_usage({"prompt_tokens": -1, "completion_tokens": 3})), id="tokens-neg"),
    pytest.param(
        httpx.Response(200, json=with_usage({"prompt_tokens": True, "completion_tokens": 3})), id="tokens-bool"
    ),
]


@pytest.mark.parametrize("response", MALFORMED)
def test_malformed_response(response):
    error = failing_call(response)

    assert type(error) is LLMMalformedResponseError


def test_empty_completion_reports_finish_reason():
    # e.g. a reasoning model that spent its whole max_tokens on reasoning.
    error = failing_call(httpx.Response(200, json=success_body(content="", finish_reason="length")))

    assert type(error) is LLMMalformedResponseError
    assert "finish_reason=length" in str(error)


def test_unexpected_non_error_status_is_malformed():
    # httpx does not follow redirects by default.
    error = failing_call(httpx.Response(302, headers={"Location": "https://elsewhere.test/"}))

    assert type(error) is LLMMalformedResponseError
    assert error.status_code == 302


# --- Redaction -------------------------------------------------------------------------------

REDACTION_CASES = [
    pytest.param(error_response(400), id="400"),
    pytest.param(error_response(401), id="401"),
    pytest.param(error_response(402), id="402"),
    pytest.param(error_response(429, headers={"Retry-After": "1"}), id="429"),
    pytest.param(error_response(500), id="500"),
    pytest.param(httpx.Response(200, content=f"{PROMPT} {FAKE_KEY} {BODY_MARKER}".encode()), id="malformed"),
    pytest.param(httpx.ConnectError(f"cannot reach {BASE_URL} with {FAKE_KEY}"), id="network"),
    pytest.param(httpx.ReadTimeout(f"timeout {FAKE_KEY} {PROMPT}"), id="timeout"),
]


@pytest.mark.parametrize("response", REDACTION_CASES)
def test_errors_never_contain_key_prompt_body_or_url(response):
    error = failing_call(response)

    for rendered in (str(error), repr(error), error.message):
        for secret in (FAKE_KEY, PROMPT, "PROMPT-MARKER", BODY_MARKER, BASE_URL):
            assert secret not in rendered
    assert MODEL in str(error)


def test_errors_do_not_chain_the_original_exception():
    # httpx exceptions carry the request (URL, headers incl. Authorization) and the response body.
    error = failing_call(httpx.ConnectError(f"cannot reach {BASE_URL}"))

    assert error.__cause__ is None
    assert error.__suppress_context__ is True


# --- Construction and arguments --------------------------------------------------------------


@pytest.mark.parametrize("api_key", [None, SecretStr(""), SecretStr("   ")], ids=["none", "empty", "blank"])
def test_missing_api_key_raises_auth_error(api_key):
    with pytest.raises(LLMAuthError, match="not configured"):
        GatewayClient(BASE_URL, api_key)


@pytest.mark.parametrize(
    ("max_tokens", "timeout"),
    [(0, 5.0), (-1, 5.0), (True, 5.0), (1.5, 5.0), (256, 0), (256, -1), (256, math.nan), (256, math.inf)],
)
def test_invalid_arguments_are_rejected_before_any_request(max_tokens, timeout):
    recorder = Recorder(httpx.Response(200, json=success_body()))

    with pytest.raises(ValueError):
        complete(make_client(recorder), max_tokens=max_tokens, timeout=timeout)

    assert recorder.requests == []


# --- Reasoning tokens and usage on empty completions -----------------------------------------


def usage_with_details(details) -> dict:
    return {"prompt_tokens": 12, "completion_tokens": 200, "total_tokens": 212, "completion_tokens_details": details}


def test_parses_reasoning_tokens_from_completion_tokens_details():
    recorder = Recorder(httpx.Response(200, json=with_usage(usage_with_details({"reasoning_tokens": 192}))))

    result = complete(make_client(recorder))

    assert (result.output_tokens, result.reasoning_tokens) == (200, 192)


def test_zero_reasoning_tokens_are_kept_as_zero():
    recorder = Recorder(httpx.Response(200, json=with_usage(usage_with_details({"reasoning_tokens": 0}))))

    assert complete(make_client(recorder)).reasoning_tokens == 0


def test_reasoning_tokens_are_none_when_details_are_absent():
    recorder = Recorder(httpx.Response(200, json=success_body()))

    assert complete(make_client(recorder)).reasoning_tokens is None


@pytest.mark.parametrize(
    "details",
    [
        None,
        "x",
        [192],
        {},
        {"reasoning_tokens": None},
        {"reasoning_tokens": -1},
        {"reasoning_tokens": "192"},
        {"reasoning_tokens": True},
        {"reasoning_tokens": 1.5},
    ],
    ids=["null", "str", "list", "empty", "value-null", "neg", "value-str", "bool", "float"],
)
def test_invalid_reasoning_tokens_are_none_and_do_not_make_the_response_malformed(details):
    recorder = Recorder(httpx.Response(200, json=with_usage(usage_with_details(details))))

    result = complete(make_client(recorder))

    assert result.reasoning_tokens is None
    assert (result.input_tokens, result.output_tokens) == (12, 200)


def test_empty_completion_carries_valid_usage_token_counts():
    body = success_body(content="", finish_reason="length", usage=usage_with_details({"reasoning_tokens": 200}))

    error = failing_call(httpx.Response(200, json=body))

    assert type(error) is LLMMalformedResponseError
    assert (error.input_tokens, error.output_tokens) == (12, 200)
    assert error.status_code == 200
    assert "finish_reason=length" in str(error)


@pytest.mark.parametrize(
    "usage",
    [None, "x", {"completion_tokens": 3}, {"prompt_tokens": 3}, {"prompt_tokens": -1, "completion_tokens": 3}],
    ids=["missing", "not-object", "prompt-missing", "completion-missing", "negative"],
)
def test_empty_completion_without_valid_usage_has_no_token_counts(usage):
    body = success_body(content="  ", finish_reason="length")
    if usage is None:
        del body["usage"]
    else:
        body["usage"] = usage

    error = failing_call(httpx.Response(200, json=body))

    assert type(error) is LLMMalformedResponseError
    assert (error.input_tokens, error.output_tokens) == (None, None)


def test_other_malformed_responses_have_no_token_counts():
    error = failing_call(httpx.Response(200, json=with_usage({"prompt_tokens": 3})))

    assert (error.input_tokens, error.output_tokens) == (None, None)
