import asyncio
import json
import os
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.core import jev_client
from app.core.jev_client import (
    JevAuthError,
    JevClient,
    JevClientError,
    JevCreditError,
    JevRateLimitError,
    JevResponseError,
    JevTimeoutError,
    load_api_key_from_env_or_file,
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
        self.assertEqual(client.api_key, "test_key_123")
        self.assertEqual(client.timeout, 8.0)

    async def test_missing_api_key_raises_auth_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("app.core.jev_client.load_api_key_from_env_or_file", return_value=None):
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
            self.assertIn("internal error", str(ctx.exception))

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


def isolate_key_sources(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    """Point both .env search paths at temp dirs so the real repo .env is never read."""
    monkeypatch.delenv("JEVSTATION_API_KEY", raising=False)
    cwd_dir = tmp_path / "cwd"
    root_dir = tmp_path / "root"
    cwd_dir.mkdir()
    (root_dir / "app" / "core").mkdir(parents=True)
    monkeypatch.chdir(cwd_dir)
    monkeypatch.setattr(jev_client, "__file__", str(root_dir / "app" / "core" / "jev_client.py"))
    return cwd_dir / ".env", root_dir / ".env"


def test_load_api_key_prefers_environment_variable(monkeypatch, tmp_path):
    cwd_env, _ = isolate_key_sources(monkeypatch, tmp_path)
    cwd_env.write_text("JEVSTATION_API_KEY=key-from-file\n", encoding="utf-8")
    monkeypatch.setenv("JEVSTATION_API_KEY", "  key-from-env  ")

    assert load_api_key_from_env_or_file() == "key-from-env"


def test_load_api_key_reads_quoted_value_from_cwd_env_file(monkeypatch, tmp_path):
    cwd_env, _ = isolate_key_sources(monkeypatch, tmp_path)
    cwd_env.write_text('OTHER_VAR=x\nJEVSTATION_API_KEY="key-from-cwd"\n', encoding="utf-8")

    assert load_api_key_from_env_or_file() == "key-from-cwd"


def test_load_api_key_skips_undecodable_cwd_file_and_uses_repo_root(monkeypatch, tmp_path):
    cwd_env, root_env = isolate_key_sources(monkeypatch, tmp_path)
    cwd_env.write_bytes(b"\xff\xfe\xfa invalid utf-8")
    root_env.write_text("JEVSTATION_API_KEY=key-from-root\n", encoding="utf-8")

    assert load_api_key_from_env_or_file() == "key-from-root"


def test_load_api_key_returns_none_when_key_missing_or_empty(monkeypatch, tmp_path):
    cwd_env, _ = isolate_key_sources(monkeypatch, tmp_path)
    cwd_env.write_text("OTHER_VAR=x\nJEVSTATION_API_KEY=\n", encoding="utf-8")

    assert load_api_key_from_env_or_file() is None


def test_default_env_search_paths_are_cwd_then_repo_root(monkeypatch, tmp_path):
    # Repo root derived from this test file's location (tests/ -> repo root), independent of jev_client.
    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "pyproject.toml").is_file()
    monkeypatch.delenv("JEVSTATION_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    expected_cwd_env = Path.cwd() / ".env"
    checked: list[Path] = []

    def record_without_opening(self: Path) -> bool:
        checked.append(self)
        return False  # report "no file" so no .env, including the real one, is ever opened

    monkeypatch.setattr(Path, "is_file", record_without_opening)

    assert load_api_key_from_env_or_file() is None
    assert checked == [expected_cwd_env, repo_root / ".env"]
