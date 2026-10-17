import asyncio
import gc
import json
import logging
import time
import traceback
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import SecretStr

from app.core import jev_client
from app.core.jev_client import (
    JevAuthError,
    JevClient,
    JevClientError,
    JevCreditError,
    JevRateLimitError,
    JevResponseError,
    JevTimeoutError,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    with open(FIXTURES_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


class TestJevClient(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.success_data = load_fixture("jev_response_success.json")
        self.get_auth_data = load_fixture("jev_response_get_auth.json")

    async def test_client_initialization_with_explicit_key(self):
        client = JevClient(api_key="test_key_123")
        self.assertEqual(client._api_key.get_secret_value(), "test_key_123")  # Held privately as a SecretStr
        self.assertEqual(client.timeout, 8.0)

    async def test_missing_api_key_raises_auth_error(self):
        with self.assertRaises(JevAuthError) as ctx:
            JevClient(api_key=None)
        self.assertIn("JEVSTATION_API_KEY is missing", str(ctx.exception))

    async def test_successful_evaluate_returns_data(self):
        client = JevClient(api_key="test_key_123")
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = self.success_data

        with patch.object(client, "_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            result = await client.evaluate(
                state="A user asks to reverse a string in Python.",
                questions={"q": {"type": "choice"}},
            )

        self.assertEqual(result["code"], 0)
        self.assertEqual(result["data"]["model"], "jev-1.13.0")
        self.assertEqual(result["data"]["credits"]["charged"], 1)

    async def test_timeout_raises_jev_timeout_error(self):
        client = JevClient(api_key="test_key_123", timeout=2.0)
        with patch.object(client, "_client") as mock_http:
            mock_http.post = AsyncMock(side_effect=httpx.TimeoutException("Request timed out"))
            with self.assertRaises(JevTimeoutError) as ctx:
                await client.evaluate(state="test state", questions={})
            self.assertIn("timed out", str(ctx.exception).lower())

    async def test_http_401_raises_auth_error(self):
        client = JevClient(api_key="invalid_key")
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 401
        mock_response.text = '{"code": -1, "message": "bad key"}'

        with patch.object(client, "_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            with self.assertRaises(JevAuthError) as ctx:
                await client.evaluate(state="test state", questions={})
            self.assertEqual(ctx.exception.status_code, 401)

    async def test_http_402_raises_credit_error(self):
        client = JevClient(api_key="test_key_123")
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 402
        mock_response.text = '{"code": -1, "message": "not enough credits"}'

        with patch.object(client, "_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            with self.assertRaises(JevCreditError) as ctx:
                await client.evaluate(state="test state", questions={})
            self.assertEqual(ctx.exception.status_code, 402)

    async def test_http_429_raises_rate_limit_error_with_retry_after(self):
        client = JevClient(api_key="test_key_123")
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "45"}
        mock_response.text = '{"code": -1, "message": "rate limited"}'

        with patch.object(client, "_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            with self.assertRaises(JevRateLimitError) as ctx:
                await client.evaluate(state="test state", questions={})
            self.assertEqual(ctx.exception.status_code, 429)
            self.assertEqual(ctx.exception.retry_after, 45)

    async def test_http_502_raises_response_error(self):
        client = JevClient(api_key="test_key_123")
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 502
        mock_response.text = '{"code": -1, "message": "evaluation failed"}'

        with patch.object(client, "_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            with self.assertRaises(JevResponseError) as ctx:
                await client.evaluate(state="test state", questions={})
            self.assertEqual(ctx.exception.status_code, 502)

    async def test_non_zero_envelope_code_raises_response_error(self):
        client = JevClient(api_key="test_key_123")
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"code": -1, "message": "internal error"}

        with patch.object(client, "_client") as mock_http:
            mock_http.post = AsyncMock(return_value=mock_response)
            with self.assertRaises(JevResponseError) as ctx:
                await client.evaluate(state="test state", questions={})
            # S2.7: the code is kept; Jev's message is not (it can echo input)
            self.assertIn("-1", str(ctx.exception))
            self.assertNotIn("internal error", str(ctx.exception))

    async def test_check_auth_get_request(self):
        client = JevClient(api_key="test_key_123")
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = self.get_auth_data

        with patch.object(client, "_client") as mock_http:
            mock_http.get = AsyncMock(return_value=mock_response)
            result = await client.check_auth()

        self.assertEqual(result["data"]["plan"], "Free")
        self.assertEqual(result["data"]["credits_remaining"], 200)

    async def test_state_character_limit_guardrail(self):
        client = JevClient(api_key="test_key_123")
        oversized_state = "x" * 24001
        with self.assertRaises(ValueError) as ctx:
            await client.evaluate(state=oversized_state, questions={})
        self.assertIn("exceeds maximum allowed length", str(ctx.exception))


def raising_client(error: Exception) -> JevClient:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    return JevClient(api_key="test_key_123", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


@pytest.mark.parametrize(
    ("error", "expected_type"),
    [
        (httpx.ReadTimeout("read timed out"), JevTimeoutError),
        (httpx.ConnectTimeout("connect timed out"), JevTimeoutError),
        (httpx.ConnectError("certificate verify failed"), JevClientError),
        (httpx.RemoteProtocolError("server disconnected"), JevClientError),
    ],
    ids=["ReadTimeout", "ConnectTimeout", "ConnectError", "RemoteProtocolError"],
)
@pytest.mark.parametrize("call", ["evaluate", "check_auth"])
def test_network_errors_record_the_httpx_exception_class_name(error, expected_type, call):
    client = raising_client(error)

    async def run():
        async with client:
            if call == "evaluate":
                await client.evaluate(state="test state", questions={})
            else:
                await client.check_auth()

    with pytest.raises(expected_type) as excinfo:
        asyncio.run(run())

    assert excinfo.value.cause_type == type(error).__name__
    assert excinfo.value.status_code is None


def test_default_timeout_is_8_seconds_and_reaches_the_http_client():
    client = JevClient(api_key="test_key_123")

    assert JevClient.DEFAULT_TIMEOUT_SECONDS == 8.0
    assert client.timeout == 8.0
    assert client._client.timeout.read == 8.0
    assert client._client.timeout.connect == 8.0


def test_custom_timeout_reaches_the_http_client():
    client = JevClient(api_key="test_key_123", timeout=5.5)

    assert client.timeout == 5.5
    assert client._client.timeout.read == 5.5


@pytest.mark.parametrize("timeout", [0, -1, 0.0, float("nan"), float("inf"), True, "8"])
def test_invalid_timeout_is_rejected_before_any_request(timeout):
    with pytest.raises(ValueError, match="timeout"):
        JevClient(api_key="test_key_123", timeout=timeout)


@pytest.mark.parametrize("call", ["evaluate", "check_auth"])
def test_timeout_bounds_the_whole_call_not_each_phase(call):
    # A server that accepts the request but never finishes the response: the whole call must end
    # at the configured timeout, even though the HTTP client's own per-phase limit is much longer.
    requests = []

    async def stalled(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        await asyncio.sleep(5)
        return httpx.Response(200, json={"code": 0})

    async def run():
        transport = httpx.MockTransport(stalled)
        http = httpx.AsyncClient(transport=transport, timeout=30.0)
        async with JevClient(api_key="test_key_123", timeout=0.05, client=http) as client:
            if call == "evaluate":
                await client.evaluate(state="test state", questions={})
            else:
                await client.check_auth()

    started = time.monotonic()
    with pytest.raises(JevTimeoutError) as excinfo:
        asyncio.run(run())
    elapsed = time.monotonic() - started

    assert len(requests) == 1  # never retried
    assert elapsed < 1.0
    assert excinfo.value.cause_type == "TimeoutError"
    assert excinfo.value.status_code is None
    assert "test_key_123" not in str(excinfo.value)


def test_http_errors_have_no_cause_type():
    async def run():
        transport = httpx.MockTransport(lambda request: httpx.Response(401))
        async with JevClient(api_key="test_key_123", client=httpx.AsyncClient(transport=transport)) as client:
            await client.evaluate(state="test state", questions={})

    with pytest.raises(JevAuthError) as excinfo:
        asyncio.run(run())

    assert excinfo.value.cause_type is None


def test_cause_type_defaults_to_none():
    assert JevClientError("x").cause_type is None
    assert JevTimeoutError("x").cause_type is None


# --- S2.7: key from Settings only; sanitized errors -----------------------------------------

CANARY_KEY = "jev-canary-key-S27-abcdef0123456789"
CANARY_URL = "https://canary-user:canary-pass@canary.example.test/secret-path"
CANARY_BODY = "CANARY-UPSTREAM-BODY-S27"
CANARIES = (CANARY_KEY, CANARY_URL, "canary-pass", "canary.example.test", CANARY_BODY)


def transport_client(handler, key=CANARY_KEY, **kwargs) -> JevClient:
    return JevClient(api_key=key, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), **kwargs)


def run_call(client: JevClient, call: str):
    async def run():
        async with client:
            if call == "evaluate":
                return await client.evaluate(state="test state", questions={})
            return await client.check_auth()

    return asyncio.run(run())


def assert_sanitized(error: BaseException) -> None:
    rendered = "".join(traceback.format_exception(error))
    for canary in CANARIES:
        assert canary not in str(error)
        assert canary not in repr(error)
        assert canary not in rendered
    assert error.__cause__ is None
    assert error.__context__ is None  # The upstream exception (holding the request) is not kept at all
    assert error.__suppress_context__ is True
    for frame, _ in traceback.walk_tb(error.__traceback__):
        if frame.f_code.co_filename == jev_client.__file__:  # The client's own frames, not this test's
            for value in frame.f_locals.values():
                assert CANARY_KEY not in repr(value)  # e.g. no headers dict left in a frame


def assert_body_absent(error: BaseException) -> None:
    rendered = "".join(traceback.format_exception(error))
    for canary in CANARIES:
        assert canary not in str(error)
        assert canary not in rendered


def test_missing_key_raises_auth_error_without_reading_the_environment_or_a_dotenv_file(monkeypatch, tmp_path):
    monkeypatch.setenv("JEVSTATION_API_KEY", CANARY_KEY)  # Ignored: the key comes only through Settings
    (tmp_path / ".env").write_text(f"JEVSTATION_API_KEY={CANARY_KEY}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    opened = []
    real_open = open

    def recording_open(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", recording_open)

    for missing in (None, "", "   ", SecretStr("  ")):
        with pytest.raises(JevAuthError) as excinfo:
            JevClient(api_key=missing)
        assert "JEVSTATION_API_KEY" in str(excinfo.value)
        assert CANARY_KEY not in str(excinfo.value)

    assert not any(path.endswith(".env") for path in opened)
    assert not hasattr(jev_client, "load_api_key_from_env_or_file")


@pytest.mark.parametrize("key", [CANARY_KEY, SecretStr(CANARY_KEY)], ids=["str", "SecretStr"])
@pytest.mark.parametrize("call", ["evaluate", "check_auth"])
def test_both_key_types_send_the_bearer_header(key, call):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["Authorization"])
        return httpx.Response(200, json={"code": 0, "data": {}})

    run_call(transport_client(handler, key=key), call)

    assert seen == [f"Bearer {CANARY_KEY}"]


@pytest.mark.parametrize("key", [CANARY_KEY, SecretStr(CANARY_KEY)], ids=["str", "SecretStr"])
def test_key_is_not_exposed_through_repr_or_attributes(key):
    client = JevClient(api_key=key)

    assert CANARY_KEY not in repr(client)
    assert CANARY_KEY not in repr(vars(client))
    assert all(CANARY_KEY not in str(value) for value in vars(client).values())
    assert not hasattr(client, "api_key")
    asyncio.run(client.aclose())


NETWORK_FAILURES = [
    (httpx.ReadTimeout(f"read timed out {CANARY_URL} {CANARY_KEY}"), JevTimeoutError, "ReadTimeout"),
    (httpx.ConnectTimeout(f"connect timed out {CANARY_URL}"), JevTimeoutError, "ConnectTimeout"),
    (
        httpx.ConnectError(f"cannot reach {CANARY_URL} Authorization: Bearer {CANARY_KEY}"),
        JevClientError,
        "ConnectError",
    ),
    (httpx.RemoteProtocolError(f"{CANARY_BODY} {CANARY_URL}"), JevClientError, "RemoteProtocolError"),
]


@pytest.mark.parametrize(
    ("error", "expected_type", "cause_type"), NETWORK_FAILURES, ids=lambda v: getattr(v, "__name__", None)
)
@pytest.mark.parametrize("call", ["evaluate", "check_auth"])
def test_network_failures_are_sanitized_and_keep_their_classification(error, expected_type, cause_type, call):
    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    with pytest.raises(JevClientError) as excinfo:
        run_call(transport_client(handler), call)

    assert type(excinfo.value) is expected_type
    assert (excinfo.value.cause_type, excinfo.value.status_code) == (cause_type, None)
    assert_sanitized(excinfo.value)


@pytest.mark.parametrize("call", ["evaluate", "check_auth"])
def test_whole_call_timeout_is_sanitized(call):
    async def stalled(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200, json={"code": 0})

    with pytest.raises(JevTimeoutError) as excinfo:
        run_call(transport_client(stalled, timeout=0.05), call)

    assert excinfo.value.cause_type == "TimeoutError"
    assert_sanitized(excinfo.value)


@pytest.mark.parametrize(
    ("status", "expected_type", "retry_after"),
    [
        (400, JevClientError, None),
        (401, JevAuthError, None),
        (402, JevCreditError, None),
        (403, JevClientError, None),
        (404, JevClientError, None),
        (429, JevRateLimitError, 45),
        (500, JevResponseError, None),
        (502, JevResponseError, None),
    ],
)
def test_http_error_bodies_never_reach_the_error_and_status_is_kept(status, expected_type, retry_after):
    body = json.dumps({"code": -1, "message": f"{CANARY_BODY} {CANARY_KEY} {CANARY_URL}"})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=body, headers={"Retry-After": "45"})

    with pytest.raises(JevClientError) as excinfo:
        run_call(transport_client(handler), "evaluate")

    error = excinfo.value
    assert type(error) is expected_type
    assert (error.status_code, error.cause_type) == (status, None)
    assert getattr(error, "retry_after", None) == retry_after
    if expected_type in (JevClientError, JevResponseError):  # Generic branches name the status
        assert str(status) in str(error)
    assert_body_absent(error)


def test_non_zero_envelope_code_is_kept_but_its_message_is_not():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": 4321, "message": f"{CANARY_BODY} {CANARY_KEY}"})

    with pytest.raises(JevResponseError) as excinfo:
        run_call(transport_client(handler), "evaluate")

    assert "4321" in str(excinfo.value)
    assert excinfo.value.status_code == 200
    assert_body_absent(excinfo.value)


@pytest.mark.parametrize(
    "body", [f"not json {CANARY_BODY}", '{"code": 0, "data": "' + CANARY_BODY], ids=["text", "truncated"]
)
def test_json_parse_failure_is_sanitized(body):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    with pytest.raises(JevResponseError) as excinfo:
        run_call(transport_client(handler), "evaluate")

    assert excinfo.value.status_code == 200
    assert_sanitized(excinfo.value)


def test_failures_log_nothing_sensitive(caplog):
    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot reach {CANARY_URL} {CANARY_KEY}")

    def bad_request(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text=CANARY_BODY)

    with caplog.at_level(logging.DEBUG):
        for handler in (unreachable, bad_request):
            with pytest.raises(JevClientError):
                run_call(transport_client(handler), "evaluate")

    for canary in CANARIES:
        assert canary not in caplog.text


# --- S2.7 review: what a frame-local capture of a Jev error can reach -----------------------

SKIPPED_REFERENTS = (type, types.ModuleType, types.FunctionType, types.CodeType, types.FrameType, types.MethodType)


def reachable_leaks(roots, depth: int = 10) -> set[str]:
    """Bounded walk of everything reachable from `roots` (gc referents, not just repr): canaries found in
    strings or bytes, and any httpx Request/Response. Types, modules, functions, code and frames are not
    followed (they lead to whole modules), so this covers data structures, not arbitrary globals."""
    found, seen, frontier = set(), set(), list(roots)
    for _ in range(depth):
        next_frontier = []
        for obj in frontier:
            if id(obj) in seen or isinstance(obj, SKIPPED_REFERENTS):
                continue
            seen.add(id(obj))
            if isinstance(obj, (httpx.Request, httpx.Response)):
                found.add(type(obj).__name__)
            if isinstance(obj, (str, bytes, bytearray)):
                text = obj if isinstance(obj, str) else bytes(obj).decode("utf-8", "replace")
                found.update(canary for canary in (CANARY_KEY, CANARY_BODY) if canary in text)
                continue
            next_frontier.extend(gc.get_referents(obj))
        frontier = next_frontier
    return found


def client_frame_locals(error: BaseException) -> list:
    """Locals of the client's own frames on the error's traceback, except `self` (the client must hold
    its key; see test_key_is_not_exposed_through_repr_or_attributes for what `self` exposes)."""
    values = []
    for frame, _ in traceback.walk_tb(error.__traceback__):
        if frame.f_code.co_filename == jev_client.__file__:
            values.extend(value for name, value in frame.f_locals.items() if name != "self")
    return values


def test_reachable_leaks_detects_a_response_and_both_canaries():
    # The scanner itself must find what the tests below assert is absent.
    request = httpx.Request("POST", "https://jev.example.test", headers={"Authorization": f"Bearer {CANARY_KEY}"})
    response = httpx.Response(400, text=CANARY_BODY, request=request)

    assert reachable_leaks([{"kept": response}]) == {"Response", "Request", CANARY_KEY, CANARY_BODY}


RESPONSE_FAILURES = [
    (httpx.Response(400, text=CANARY_BODY), JevClientError, 400),
    (httpx.Response(401, text=CANARY_BODY), JevAuthError, 401),
    (httpx.Response(402, text=CANARY_BODY), JevCreditError, 402),
    (httpx.Response(429, text=CANARY_BODY, headers={"Retry-After": "45"}), JevRateLimitError, 429),
    (httpx.Response(503, text=CANARY_BODY), JevResponseError, 503),
    (httpx.Response(200, json={"code": 7, "message": CANARY_BODY}), JevResponseError, 200),
    (httpx.Response(200, text=f"not json {CANARY_BODY}"), JevResponseError, 200),
    (httpx.Response(200, json=[CANARY_BODY]), JevResponseError, 200),
]


@pytest.mark.parametrize(
    ("template", "expected_type", "status"),
    RESPONSE_FAILURES,
    ids=["400", "401", "402", "429", "503", "envelope-code", "not-json", "not-an-object"],
)
@pytest.mark.parametrize("call", ["evaluate", "check_auth"])
def test_response_errors_leave_no_response_request_key_or_body_in_client_frames(template, expected_type, status, call):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(template.status_code, headers=template.headers, content=template.content)

    with pytest.raises(JevClientError) as excinfo:
        run_call(transport_client(handler), call)

    error = excinfo.value
    assert (type(error), error.status_code, error.cause_type) == (expected_type, status, None)
    assert getattr(error, "retry_after", None) == (45 if status == 429 else None)
    assert error.__cause__ is None and error.__context__ is None
    assert reachable_leaks(client_frame_locals(error)) == set()
    assert reachable_leaks([error, vars(error)]) == set()  # Nor through the exception's own attributes


@pytest.mark.parametrize(
    ("error", "expected_type", "cause_type"), NETWORK_FAILURES, ids=lambda v: getattr(v, "__name__", None)
)
@pytest.mark.parametrize("call", ["evaluate", "check_auth"])
def test_network_errors_leave_no_request_or_key_in_client_frames(error, expected_type, cause_type, call):
    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    with pytest.raises(JevClientError) as excinfo:
        run_call(transport_client(handler), call)

    assert reachable_leaks(client_frame_locals(excinfo.value)) == set()
    assert reachable_leaks([excinfo.value, vars(excinfo.value)]) == set()


def test_after_a_failed_call_the_client_holds_only_its_key():
    # `self` is on every traceback, so this is what a frame capture of `self` can reach: the SecretStr
    # key (by design) and nothing from the failed exchange.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text=CANARY_BODY)

    client = transport_client(handler)
    with pytest.raises(JevClientError):
        run_call(client, "evaluate")

    others = [value for name, value in vars(client).items() if name != "_api_key"]
    assert reachable_leaks(others) == set()
    assert reachable_leaks([client._api_key]) == {CANARY_KEY}  # The one place the key lives
