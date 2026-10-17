"""
Model-chain executor: retries, tier fallback and the request deadline (Task S1.4, TRD §6).

The executor is the only retry layer; GatewayClient makes exactly one call per complete().

- The route's own chain runs first. Only a strong route falls back to the cheap chain for the
  same task once its chain is exhausted; a cheap route never escalates to strong.
- Rate limits, server errors and timeouts are retried with backoff 1 s, then 2 s, but only when
  the deadline still leaves room for the backoff plus a minimum viable call. Otherwise the next
  model is tried immediately.
- Client errors and malformed responses move to the next model without retrying.
- Authentication and gateway-budget errors are fatal: every model shares the key and credits.
- Each call's timeout is min(20 s, remaining - 2 s). When that is below MIN_CALL_SECONDS the
  execution stops: the deadline is shared by every model and only shrinks.

Calls are bounded by (number of chains) x (models per chain) x max_attempts_per_model; with the
validated config that is at most 2 x 2 x 3 = 12.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Literal, Optional, Tuple, Union

from app.core.deadline import Deadline
from app.core.llm_client import (
    FATAL_ERRORS,
    RETRYABLE_ERRORS,
    LLMAuthError,
    LLMBudgetExhaustedError,
    LLMClient,
    LLMClientError,
    LLMError,
    LLMMalformedResponseError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
    ModelResult,
)
from app.core.model_config import ModelConfig
from app.core.policy import RouteDecision

MAX_CALL_SECONDS = 20.0
CALL_MARGIN_SECONDS = 2.0  # Kept free after every call for response handling
MIN_CALL_SECONDS = 2.0  # A call with less time than this is not worth starting
BACKOFF_SECONDS = (1.0, 2.0)  # Before attempt 2, attempt 3 (and the last value after that)

AttemptStatus = Literal[
    "success",
    "rate_limited",
    "server_error",
    "timeout",
    "client_error",
    "malformed",
    "auth_error",
    "budget_exhausted",
    "deadline",
]
ExhaustionReason = Literal["all_failed", "deadline", "fatal"]

_ERROR_STATUS: Tuple[Tuple[type, AttemptStatus], ...] = (
    (LLMRateLimitError, "rate_limited"),
    (LLMServerError, "server_error"),
    (LLMTimeoutError, "timeout"),
    (LLMClientError, "client_error"),
    (LLMMalformedResponseError, "malformed"),
    (LLMAuthError, "auth_error"),
    (LLMBudgetExhaustedError, "budget_exhausted"),
)


@dataclass(frozen=True)
class Attempt:
    tier: str
    model: str
    attempt_no: int  # 1-based, per model
    status: AttemptStatus
    latency_ms: int  # 0 for a "deadline" entry, where no call was made
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None  # Successful calls only; None when the gateway did not report it
    # Gateway HTTP status of the call; None for timeouts, network errors and "deadline"
    http_status: Optional[int] = None


@dataclass(frozen=True)
class ExecutionResult:
    route: RouteDecision  # The policy decision, unchanged
    result: ModelResult
    final_model: str
    final_tier: str
    fell_back_to_cheap: bool
    attempts: Tuple[Attempt, ...]


class ChainExhausted(Exception):
    """No model produced an answer. `attempts` holds the complete, ordered history."""

    def __init__(self, attempts: Tuple[Attempt, ...], reason: ExhaustionReason):
        calls = sum(1 for attempt in attempts if attempt.status != "deadline")
        super().__init__(f"model chain exhausted ({reason}) after {calls} calls")
        self.attempts = attempts
        self.reason = reason


_Stop = Literal["next_model", "deadline", "fatal"]
Sleep = Callable[[float], Awaitable[None]]


def can_start_call(deadline: Deadline) -> bool:
    """Whether the deadline still allows a model call to start under the executor's own rule (at least
    MIN_CALL_SECONDS of call time after keeping CALL_MARGIN_SECONDS free). Says nothing about finishing."""
    return _call_timeout(deadline) >= MIN_CALL_SECONDS


def _call_timeout(deadline: Deadline) -> float:
    return min(MAX_CALL_SECONDS, deadline.remaining() - CALL_MARGIN_SECONDS)


async def execute(
    prompt: str,
    route: RouteDecision,
    config: ModelConfig,
    llm: LLMClient,
    deadline: Deadline,
    *,
    sleep: Sleep = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> ExecutionResult:
    """
    Runs the route's chain (then, for strong routes only, the cheap chain) within the deadline.

    Raises:
        ChainExhausted: No model succeeded; `reason` is "all_failed", "deadline" or "fatal".
    """
    chains: List[Tuple[str, Tuple[str, ...]]] = [(route.tier, tuple(route.model_chain))]
    if route.tier == "strong":
        chains.append(("cheap", tuple(config.routing.get_chain(route.task_type, "cheap"))))

    attempts: List[Attempt] = []
    for tier, chain in chains:
        for model in chain:
            outcome = await _run_model(prompt, model, tier, config, llm, deadline, sleep, clock, attempts)
            if isinstance(outcome, ModelResult):
                return ExecutionResult(
                    route=route,
                    result=outcome,
                    final_model=model,
                    final_tier=tier,
                    fell_back_to_cheap=tier != route.tier,
                    attempts=tuple(attempts),
                )
            if outcome != "next_model":
                raise ChainExhausted(tuple(attempts), reason=outcome)
    raise ChainExhausted(tuple(attempts), reason="all_failed")


async def _run_model(
    prompt: str,
    model: str,
    tier: str,
    config: ModelConfig,
    llm: LLMClient,
    deadline: Deadline,
    sleep: Sleep,
    clock: Callable[[], float],
    attempts: List[Attempt],
) -> Union[ModelResult, _Stop]:
    """Up to max_attempts_per_model calls to one model; appends every attempt to `attempts`."""
    max_attempts = config.limits.max_attempts_per_model
    max_tokens = config.catalog[model].max_output_tokens
    for attempt_no in range(1, max_attempts + 1):
        timeout = _call_timeout(deadline)
        if timeout < MIN_CALL_SECONDS:
            attempts.append(Attempt(tier, model, attempt_no, "deadline", latency_ms=0))
            return "deadline"

        started = clock()
        try:
            result = await llm.complete(model, prompt, max_tokens, timeout)
        except LLMError as exc:
            # Only an empty 200 completion carries usage: those tokens are billed.
            usage = exc if isinstance(exc, LLMMalformedResponseError) else None
            attempts.append(
                Attempt(
                    tier,
                    model,
                    attempt_no,
                    _status(exc),
                    _elapsed_ms(clock, started),
                    input_tokens=usage.input_tokens if usage else None,
                    output_tokens=usage.output_tokens if usage else None,
                    http_status=exc.status_code,
                )
            )
            if isinstance(exc, FATAL_ERRORS):
                return "fatal"
            if not isinstance(exc, RETRYABLE_ERRORS):
                return "next_model"
            if attempt_no == max_attempts:
                break
            backoff = BACKOFF_SECONDS[min(attempt_no, len(BACKOFF_SECONDS)) - 1]
            if deadline.remaining() - backoff - CALL_MARGIN_SECONDS < MIN_CALL_SECONDS:
                return "next_model"
            await sleep(backoff)
            continue

        attempts.append(
            Attempt(
                tier,
                model,
                attempt_no,
                "success",
                _elapsed_ms(clock, started),
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                reasoning_tokens=result.reasoning_tokens,
                http_status=result.http_status,
            )
        )
        return result
    return "next_model"


def _status(error: LLMError) -> AttemptStatus:
    for error_type, status in _ERROR_STATUS:
        if isinstance(error, error_type):
            return status
    raise TypeError(f"Unclassified LLMError subclass: {type(error).__name__}")


def _elapsed_ms(clock: Callable[[], float], started: float) -> int:
    return round((clock() - started) * 1000)
