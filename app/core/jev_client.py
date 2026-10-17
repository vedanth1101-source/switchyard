"""
Asynchronous HTTP client for JevStation's System One API.

Handles authentication, timeouts, error classification, and enforces
client-side limits per technical requirements.

The API key is passed in by the caller (Settings.jevstation_api_key, read from JEVSTATION_API_KEY in
the process environment); the client never reads environment variables or .env files itself. It is held
as a SecretStr and revealed only while building the Authorization header.

Error messages are fixed strings plus status codes: never upstream text (httpx messages can hold URLs,
response bodies and Jev's own `message` can echo input). Every error is raised by evaluate()/check_auth()
outside the `except` block, after the httpx response is dropped: the sanitized error has no __cause__ or
__context__, and no client frame on its traceback holds the upstream exception or response, whose request
carries the Authorization header. `cause_type` keeps the upstream class name.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, Dict, Optional, Union

import certifi
import httpx
from pydantic import SecretStr


class JevClientError(Exception):
    """Base exception for JevStation client errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, cause_type: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        # Class name of the underlying httpx exception for network failures (e.g. "ReadTimeout");
        # safe to log, unlike the message or the chained exception, which holds the request.
        self.cause_type = cause_type


class JevAuthError(JevClientError):
    """Raised when authentication fails (401) or credentials are missing."""

    pass


class JevCreditError(JevClientError):
    """Raised when the account has insufficient credits (402)."""

    pass


class JevRateLimitError(JevClientError):
    """Raised when request is rate limited (429)."""

    def __init__(self, message: str, status_code: int = 429, retry_after: Optional[int] = None):
        super().__init__(message, status_code=status_code)
        self.retry_after = retry_after


class JevTimeoutError(JevClientError):
    """Raised when a request exceeds the time limit."""

    pass


class JevResponseError(JevClientError):
    """Raised when server returns an error response (5xx or non-zero code)."""

    pass


class JevClient:
    """
    Async client for JevStation API (POST /api/v1/systemone).
    """

    DEFAULT_BASE_URL = "https://jevstation.com/api/v1/systemone"
    DEFAULT_TIMEOUT_SECONDS = 8.0  # Settings.jev_timeout_seconds (JEV_TIMEOUT_SECONDS) defaults to this
    MAX_STATE_CHARS = 24000
    DEFAULT_MODEL = "jev"  # evaluate()'s default; part of app.infra.cache.classifier_version()

    def __init__(
        self,
        api_key: Union[SecretStr, str, None] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        client: Optional[httpx.AsyncClient] = None,
    ):
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be a finite number of seconds > 0")
        secret = api_key if isinstance(api_key, SecretStr) else SecretStr(api_key or "")
        if not secret.get_secret_value().strip():
            raise JevAuthError("JEVSTATION_API_KEY is missing. Export it in the process environment.")

        self._api_key = secret
        self.base_url = base_url
        self.timeout = timeout
        import ssl

        ssl_context = ssl.create_default_context(cafile=certifi.where())
        self._client = client or httpx.AsyncClient(
            verify=ssl_context,
            timeout=timeout,
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "JevClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key.get_secret_value()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Switchyard/1.0",
        }

    async def evaluate(
        self,
        state: str,
        questions: Dict[str, Any],
        model: str = DEFAULT_MODEL,
        question_set_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submits an evaluation request to JevStation.
        Consumes credits on the provider side according to tier rules.
        """
        if len(state) > self.MAX_STATE_CHARS:
            raise ValueError(
                f"State length ({len(state)}) exceeds maximum allowed length of {self.MAX_STATE_CHARS} characters."
            )

        payload: Dict[str, Any] = {
            "model": model,
            "state": state,
        }
        if question_set_id:
            payload["question_set_id"] = question_set_id
        else:
            payload["questions"] = questions

        try:
            # httpx applies its timeout per phase (connect, write, read); this bounds the whole call.
            async with asyncio.timeout(self.timeout):
                response = await self._client.post(self.base_url, headers=self._get_headers(), json=payload)
        except (httpx.TimeoutException, TimeoutError) as exc:
            failure = JevTimeoutError(
                f"JevStation evaluation timed out after {self.timeout}s", cause_type=type(exc).__name__
            )
        except httpx.RequestError as exc:
            failure = JevClientError("Network error communicating with JevStation", cause_type=type(exc).__name__)
        else:
            outcome = self._check_response(response)
            del response  # Its request carries the Authorization header: keep it out of any traceback frame
            if not isinstance(outcome, JevClientError):
                return outcome
            failure = outcome
        raise failure from None  # Outside the except block: the upstream exception is not attached

    async def check_auth(self) -> Dict[str, Any]:
        """
        Executes the documented 0-credit GET authentication check.
        Returns account status, remaining credits, and model lists.
        """
        try:
            async with asyncio.timeout(self.timeout):
                response = await self._client.get(self.base_url, headers=self._get_headers())
        except (httpx.TimeoutException, TimeoutError) as exc:
            failure = JevTimeoutError("JevStation auth check timed out", cause_type=type(exc).__name__)
        except httpx.RequestError as exc:
            failure = JevClientError("Network error during auth check", cause_type=type(exc).__name__)
        else:
            outcome = self._check_response(response)
            del response  # Its request carries the Authorization header: keep it out of any traceback frame
            if not isinstance(outcome, JevClientError):
                return outcome
            failure = outcome
        raise failure from None  # Outside the except block: the upstream exception is not attached

    def _check_response(self, response: httpx.Response) -> Union[Dict[str, Any], JevClientError]:
        """The JSON envelope, or the error to raise. Returned, not raised, so the caller can drop `response`
        (its body and its request's Authorization header) before raising: no frame on the traceback holds it."""
        status = response.status_code

        if status == 401:
            return JevAuthError("Unauthorized: invalid or revoked JEVSTATION_API_KEY", status_code=status)
        if status == 402:
            return JevCreditError("Payment required: insufficient JevStation credits", status_code=status)
        if status == 429:
            retry_header = response.headers.get("Retry-After")
            retry_after = int(retry_header) if retry_header and retry_header.isdigit() else None
            return JevRateLimitError(
                "Rate limit exceeded on JevStation API", status_code=status, retry_after=retry_after
            )
        if status >= 500:
            return JevResponseError(f"JevStation server error (HTTP {status})", status_code=status)
        if status >= 400:
            return JevClientError(f"JevStation client error (HTTP {status})", status_code=status)

        try:
            data = response.json()
        except Exception:  # e.g. JSONDecodeError, whose .doc is the whole response body
            return JevResponseError("Failed to parse JSON response", status_code=status)

        if not isinstance(data, dict):
            return JevResponseError("Invalid JSON envelope: expected object", status_code=status)

        code = data.get("code")
        if code is not None and code != 0:
            return JevResponseError(f"JevStation error code {code}", status_code=status)  # Jev's message omitted

        return data
