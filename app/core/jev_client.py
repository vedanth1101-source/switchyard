"""
Asynchronous HTTP client for JevStation's System One API.

Handles authentication, timeouts, error classification, and enforces
client-side limits per technical requirements.
"""

from __future__ import annotations

import asyncio
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

import certifi
import httpx


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


def load_api_key_from_env_or_file() -> Optional[str]:
    """
    Safely retrieves JEVSTATION_API_KEY from environment or .env without leaking/printing.
    """
    # 1. Environment variable
    key = os.environ.get("JEVSTATION_API_KEY")
    if key and key.strip():
        return key.strip()

    # 2. Local .env file in cwd or repo root
    search_paths = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",  # app/core/jev_client.py -> repo root
    ]
    for p in search_paths:
        if p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("JEVSTATION_API_KEY="):
                            val = line.split("=", 1)[1].strip("\"' ")
                            if val:
                                return val
            except (OSError, UnicodeDecodeError):
                continue
    return None


class JevClient:
    """
    Async client for JevStation API (POST /api/v1/systemone).
    """

    DEFAULT_BASE_URL = "https://jevstation.com/api/v1/systemone"
    DEFAULT_TIMEOUT_SECONDS = 8.0  # Settings.jev_timeout_seconds (JEV_TIMEOUT_SECONDS) defaults to this
    MAX_STATE_CHARS = 24000

    def __init__(
        self,
        api_key: Optional[str] = None,
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
        resolved_key = api_key or load_api_key_from_env_or_file()
        if not resolved_key:
            raise JevAuthError("JEVSTATION_API_KEY is missing. Configure it in .env or environment.")

        self.api_key = resolved_key
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
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Switchyard/1.0",
        }

    async def evaluate(
        self,
        state: str,
        questions: Dict[str, Any],
        model: str = "jev",
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

        headers = self._get_headers()

        try:
            # httpx applies its timeout per phase (connect, write, read); this bounds the whole call.
            async with asyncio.timeout(self.timeout):
                response = await self._client.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                )
        except (httpx.TimeoutException, TimeoutError) as exc:
            raise JevTimeoutError(
                f"JevStation evaluation timed out after {self.timeout}s: {exc}", cause_type=type(exc).__name__
            ) from exc
        except httpx.RequestError as exc:
            raise JevClientError(
                f"Network error communicating with JevStation: {exc}", cause_type=type(exc).__name__
            ) from exc

        return self._handle_response(response)

    async def check_auth(self) -> Dict[str, Any]:
        """
        Executes the documented 0-credit GET authentication check.
        Returns account status, remaining credits, and model lists.
        """
        headers = self._get_headers()

        try:
            async with asyncio.timeout(self.timeout):
                response = await self._client.get(
                    self.base_url,
                    headers=headers,
                )
        except (httpx.TimeoutException, TimeoutError) as exc:
            raise JevTimeoutError(f"JevStation auth check timed out: {exc}", cause_type=type(exc).__name__) from exc
        except httpx.RequestError as exc:
            raise JevClientError(f"Network error during auth check: {exc}", cause_type=type(exc).__name__) from exc

        return self._handle_response(response)

    def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        status = response.status_code

        if status == 401:
            raise JevAuthError("Unauthorized: invalid or revoked JEVSTATION_API_KEY", status_code=status)
        if status == 402:
            raise JevCreditError("Payment required: insufficient JevStation credits", status_code=status)
        if status == 429:
            retry_header = response.headers.get("Retry-After")
            retry_after = int(retry_header) if retry_header and retry_header.isdigit() else None
            raise JevRateLimitError(
                "Rate limit exceeded on JevStation API", status_code=status, retry_after=retry_after
            )
        if status >= 500:
            raise JevResponseError(f"JevStation server error (HTTP {status})", status_code=status)
        if status >= 400:
            raise JevClientError(f"JevStation client error (HTTP {status}): {response.text[:200]}", status_code=status)

        try:
            data = response.json()
        except Exception as exc:
            raise JevResponseError(f"Failed to parse JSON response: {exc}", status_code=status) from exc

        if not isinstance(data, dict):
            raise JevResponseError("Invalid JSON envelope: expected object", status_code=status)

        code = data.get("code")
        if code is not None and code != 0:
            msg = data.get("message", "Unknown error")
            raise JevResponseError(f"JevStation error code {code}: {msg}", status_code=status)

        return data
