"""
Async client for the OpenAI-compatible Vercel AI Gateway (Task S1.3).

Sends POST {base_url}/chat/completions with one user message and returns a ModelResult. The
client never retries: the executor (Task S1.4) owns retries, fallback and the request deadline,
and classifies failures with RETRYABLE_ERRORS and FATAL_ERRORS.

Status mapping follows the AI Gateway error documentation: 401/403 authentication or permission
(fatal), 402 credits or budget exhausted (fatal), 429 rate limit, 5xx server error, any other 4xx
a rejected request. Exception messages carry only the model id, HTTP status and exception class
name, never the API key, prompt, response body or URL. Original httpx exceptions are not chained,
because they hold the request (with its Authorization header) and response.
"""

from __future__ import annotations

import asyncio
import math
import re
import ssl
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

import certifi
import httpx
from pydantic import SecretStr

RETRY_AFTER_SECONDS = re.compile(r"\d+(\.\d+)?")
FINISH_REASON = re.compile(r"[a-z_]{1,32}")


@dataclass(frozen=True)
class ModelResult:
    text: str
    model: str  # The requested config model id, not the provider's echoed name
    input_tokens: int
    output_tokens: int
    latency_ms: int
    finish_reason: Optional[str]
    http_status: int  # The gateway's actual 2xx status
    # usage.completion_tokens_details.reasoning_tokens; None when absent or invalid, never 0 then
    reasoning_tokens: Optional[int] = None


class LLMClient(Protocol):
    async def complete(self, model: str, prompt: str, max_tokens: int, timeout: float) -> ModelResult: ...


class LLMError(Exception):
    """Base class for gateway call failures. Messages never contain secrets or payloads."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class LLMRateLimitError(LLMError):
    """HTTP 429. Retryable after a wait."""

    def __init__(self, message: str, status_code: int = 429, retry_after: Optional[float] = None):
        super().__init__(message, status_code=status_code)
        self.retry_after = retry_after


class LLMServerError(LLMError):
    """HTTP 5xx, or a network failure with no status. Retryable."""


class LLMTimeoutError(LLMError):
    """The call did not finish within its timeout. Retryable."""


class LLMClientError(LLMError):
    """Any other HTTP 4xx: this model rejected the request. Not retried; try the next model."""


class LLMAuthError(LLMError):
    """HTTP 401/403, or no API key configured. Fatal: every model shares the key."""


class LLMBudgetExhaustedError(LLMError):
    """HTTP 402: gateway credits or a configured budget are exhausted. Fatal for every model."""


class LLMMalformedResponseError(LLMError):
    """
    A response without usable text or token usage. Not retried; try the next model.

    For a 200 with empty visible text, input_tokens/output_tokens hold the response's usage when
    it is valid (those tokens are billed); otherwise they are None.
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ):
        super().__init__(message, status_code=status_code)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


RETRYABLE_ERRORS = (LLMRateLimitError, LLMServerError, LLMTimeoutError)
FATAL_ERRORS = (LLMAuthError, LLMBudgetExhaustedError)


class GatewayClient:
    """
    LLMClient for the AI Gateway's OpenAI-compatible chat completions endpoint.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[SecretStr],
        *,
        client: Optional[httpx.AsyncClient] = None,
        clock: Callable[[], float] = time.perf_counter,
    ):
        if api_key is None or not api_key.get_secret_value().strip():
            raise LLMAuthError("AI gateway API key is not configured")
        self._api_key = api_key
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._clock = clock
        self._client = client or httpx.AsyncClient(verify=ssl.create_default_context(cafile=certifi.where()))

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "GatewayClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()

    async def complete(self, model: str, prompt: str, max_tokens: int, timeout: float) -> ModelResult:
        """
        Makes exactly one chat completion call.

        The timeout bounds the whole call: it is passed to httpx (per connect/read/write) and also
        enforced as a total limit, so a slowly trickling response cannot outlast it.

        Raises:
            ValueError: If max_tokens or timeout is invalid (no request is sent).
            LLMError: A subclass describing the failure; see the module docstring.
        """
        _validate_arguments(max_tokens, timeout)
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
        headers = {
            "Authorization": f"Bearer {self._api_key.get_secret_value()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        started = self._clock()
        try:
            async with asyncio.timeout(timeout):
                response = await self._client.post(self._url, json=payload, headers=headers, timeout=timeout)
        except (httpx.TimeoutException, TimeoutError):
            raise LLMTimeoutError(f"{model}: request timed out after {timeout}s") from None
        except httpx.RequestError as exc:
            raise LLMServerError(f"{model}: network error ({type(exc).__name__})") from None
        latency_ms = round((self._clock() - started) * 1000)

        _raise_for_status(model, response)
        return _parse_completion(model, response, latency_ms)


def _validate_arguments(max_tokens: Any, timeout: Any) -> None:
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a finite number of seconds > 0")


def _raise_for_status(model: str, response: httpx.Response) -> None:
    status = response.status_code
    if 200 <= status < 300:
        return
    detail = f"{model}: HTTP {status}"
    if status in (401, 403):
        raise LLMAuthError(f"{detail}: authentication or permission failure", status_code=status)
    if status == 402:
        raise LLMBudgetExhaustedError(f"{detail}: gateway credits or budget exhausted", status_code=status)
    if status == 429:
        raise LLMRateLimitError(f"{detail}: rate limited", retry_after=_retry_after(response))
    if status >= 500:
        raise LLMServerError(f"{detail}: server error", status_code=status)
    if status >= 400:
        raise LLMClientError(f"{detail}: request rejected", status_code=status)
    raise LLMMalformedResponseError(f"{detail}: unexpected status", status_code=status)


def _retry_after(response: httpx.Response) -> Optional[float]:
    """Seconds from a numeric Retry-After header; None if absent or in HTTP-date form."""
    value = response.headers.get("Retry-After", "").strip()
    return float(value) if RETRY_AFTER_SECONDS.fullmatch(value) else None


def _parse_completion(model: str, response: httpx.Response, latency_ms: int) -> ModelResult:
    def malformed(
        reason: str, input_tokens: Optional[int] = None, output_tokens: Optional[int] = None
    ) -> LLMMalformedResponseError:
        return LLMMalformedResponseError(
            f"{model}: malformed response ({reason})",
            status_code=response.status_code,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    try:
        data = response.json()
    except ValueError:
        raise malformed("invalid JSON") from None
    if not isinstance(data, dict):
        raise malformed("body is not an object")

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise malformed("missing choices")
    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise malformed("invalid finish_reason")
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise malformed("missing message content")

    usage = data.get("usage")
    usage = usage if isinstance(usage, dict) else None
    input_tokens = _token_count(usage.get("prompt_tokens")) if usage else None
    output_tokens = _token_count(usage.get("completion_tokens")) if usage else None
    usage_valid = input_tokens is not None and output_tokens is not None

    if not content.strip():
        shown = finish_reason if finish_reason and FINISH_REASON.fullmatch(finish_reason) else "unknown"
        # Valid usage is attached: these tokens are billed even though the answer is unusable.
        tokens = (input_tokens, output_tokens) if usage_valid else (None, None)
        raise malformed(f"empty completion, finish_reason={shown}", *tokens)
    if usage is None:
        raise malformed("missing usage")
    if not usage_valid:
        raise malformed("invalid usage token counts")

    details = usage.get("completion_tokens_details")
    reasoning_tokens = _token_count(details.get("reasoning_tokens")) if isinstance(details, dict) else None

    return ModelResult(
        text=content,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        finish_reason=finish_reason,
        http_status=response.status_code,
        reasoning_tokens=reasoning_tokens,
    )


def _token_count(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value
