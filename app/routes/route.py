"""
POST /api/route (Task S1.7; docs/04 Part C).

Runs one prompt through app.core.router.route(). Dependencies (Jev, strong budget, model gateway)
are built per request by get_deps_factory() and closed afterwards; tests override that dependency
with fakes. A dry run skips the model call and never reserves strong budget, but still calls Jev.

Status codes: 200 success or dry run · 422 invalid request · 502 all models failed or the deadline
was hit (body has `attempts`) · 503 the model gateway rejected every model (authentication or
budget), or Jev / the gateway is not configured · 500 invalid model or routing configuration.
Every response from this endpoint carries an X-Request-ID header equal to the body's request_id. The
ID is the one app.main.RequestIdMiddleware assigned to the HTTP request (S2.1); route() and its log
line use it too. Error bodies hold stable error codes only, never exception messages, paths, URLs or
keys. 422 bodies are ValidationErrorResponse (built by app.main), without the submitted input.

`cost` (S1.8) is a list-price estimate for the successful call only (Jev excluded); it is null for
dry runs and when it cannot be calculated, never zero in that case.

Persistence (S2.5): every outcome route() returns (200, 502 and 503 above) is saved before the
response is sent, by the recorder from get_outcome_recorder() (tests override it). Setup errors, 422
and unexpected exceptions have no outcome and are not saved. Persistence fails open: if the database
fails (SQLAlchemyError) or outcome_to_rows refuses the outcome (OutcomeRowsError), the response is
unchanged and one warning with the request ID and exception type is logged. Any other exception is a
bug and reaches the generic 500. Without DATABASE_URL nothing is saved (one INFO line per request).

Rate limit (S2.6): the first statement of the handler, so after FastAPI has validated the body (a 422
is never counted) and before any dependency is built. A refused request gets 429 with a Retry-After
header and a RouteErrorResponse (error=rate_limited); it never builds clients, calls Jev, touches the
strong budget or is persisted. See app.infra.ratelimit for the window, the client identity and why a
missing Redis or salt fails open.

Not implemented yet: cache (`use_cache` is accepted and ignored; `cache_hit` is omitted).
"""

import dataclasses
import functools
import logging
from typing import Any, AsyncIterator, Awaitable, Callable, Literal, Optional, Union

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.exc import SQLAlchemyError
from starlette.concurrency import run_in_threadpool

from app.core.costs import CostEstimate
from app.core.executor import Attempt
from app.core.jev_client import JevAuthError
from app.core.llm_client import LLMAuthError
from app.core.model_config import ModelConfigError
from app.core.policy import RoutingConfigError
from app.core.router import CliConfigurationError, DepsFactory, RouteOutcome, build_cli_deps, route
from app.core.settings import Settings, get_settings
from app.db.repo import OutcomeRowsError, outcome_to_rows, save_outcome
from app.infra.db import get_engine
from app.infra.ratelimit import RateLimiter, create_rate_limiter, resolve_client_ip

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_PROMPT_CHARS = 8000
REQUEST_ID_HEADER = "X-Request-ID"
RETRY_AFTER_HEADER = "Retry-After"

ErrorCode = Literal[
    "all_models_failed",
    "deadline_exceeded",
    "model_gateway_unavailable",
    "model_gateway_not_configured",
    "classifier_not_configured",
    "configuration_error",
    "rate_limited",
]


class RouteRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    dry_run: bool = False

    @field_validator("prompt")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class DecisionOut(BaseModel):
    task_type: str
    tier: str
    reason: str
    model_chain: list[str]
    task_type_probs: Optional[dict[str, float]]  # None when Jev failed
    difficulty: Optional[float]
    needs_reasoning_prob: Optional[float]
    confidence: Optional[float]
    jev_latency_ms: Optional[int]
    jev_error: Optional[str]  # Exception class name only
    jev_http_status: Optional[int]  # HTTP status of a failed Jev call; None if absent
    jev_error_cause: Optional[str]  # httpx exception class behind a Jev timeout/network failure; None if absent


class AttemptOut(BaseModel):
    tier: str
    model: str
    attempt_no: int
    status: str
    latency_ms: int
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    reasoning_tokens: Optional[int]  # Gateway-reported reasoning tokens of a successful call; None when absent
    http_status: Optional[int]  # Gateway HTTP status of the call; None for timeout and network errors


class ModelResponseOut(BaseModel):
    model: str
    tier: str
    text: str
    latency_ms: int
    fell_back_to_cheap: bool
    attempts: list[AttemptOut]


class CostOut(BaseModel):
    """List-price estimate (USD) for the call that produced the answer; Jev credits excluded."""

    est_cost_usd: float
    baseline_cost_usd: float = Field(description="Same tokens priced at baseline_model")
    saved_usd: float = Field(description="baseline_cost_usd - est_cost_usd; negative when pricier than baseline")
    baseline_model: str
    basis: Literal["list_price_estimate"]
    includes_jev: bool


class RouteResponse(BaseModel):
    request_id: str
    dry_run: bool
    decision: DecisionOut
    response: Optional[ModelResponseOut]  # None for dry runs
    strong_reserved: bool
    total_latency_ms: int
    config_version: str
    cost: Optional[CostOut]  # None for dry runs or when the cost cannot be calculated


class ValidationErrorItem(BaseModel):
    loc: list[Union[str, int]]
    msg: str
    type: str


class ValidationErrorResponse(BaseModel):
    """422 body. The submitted values are never echoed (no `input` or `ctx`)."""

    request_id: str
    error: Literal["invalid_request"]
    detail: list[ValidationErrorItem]


class InternalErrorResponse(BaseModel):
    """Generic 500 for an unexpected exception (built by app.main.RequestIdMiddleware); no details."""

    request_id: str
    error: Literal["internal_error"]


class RouteErrorResponse(BaseModel):
    request_id: str
    error: ErrorCode
    decision: Optional[DecisionOut]
    attempts: list[AttemptOut]
    total_latency_ms: Optional[int]
    config_version: Optional[str]


def get_deps_factory() -> DepsFactory:
    """Real per-request dependencies; tests override this with fakes."""
    return build_cli_deps


async def get_rate_limiter(settings: Settings = Depends(get_settings)) -> AsyncIterator[RateLimiter]:
    """Per-request limiter from Settings (built without connecting, closed afterwards); tests override this."""
    limiter = create_rate_limiter(settings)
    try:
        yield limiter
    finally:
        await limiter.aclose()


OutcomeRecorder = Callable[[RouteOutcome, str], Awaitable[None]]


def get_outcome_recorder(settings: Settings = Depends(get_settings)) -> OutcomeRecorder:
    """Real persistence to DATABASE_URL; tests override this."""
    return functools.partial(record_outcome, database_url=settings.database_url)


async def record_outcome(outcome: RouteOutcome, prompt: str, *, database_url: Optional[str]) -> None:
    """Saves one outcome and its attempts; the blocking database work runs in Starlette's thread pool."""
    if database_url is None:
        reason = "database_not_configured"
        logger.info(
            "route persistence skipped request_id=%s reason=%s",
            outcome.request_id,
            reason,
            extra={"request_id": outcome.request_id, "reason": reason},
        )
        return
    rows = outcome_to_rows(outcome, prompt=prompt)
    await run_in_threadpool(save_outcome, get_engine(database_url), rows)


@router.post(
    "/route",
    response_model=RouteResponse,
    responses={
        422: {"model": ValidationErrorResponse, "description": "Invalid request body"},
        429: {"model": RouteErrorResponse, "description": "Rate limited (error=rate_limited); see Retry-After"},
        500: {
            "model": Union[RouteErrorResponse, InternalErrorResponse],
            "description": "Invalid model or routing configuration (RouteErrorResponse, error=configuration_error), "
            "or an unexpected internal error (InternalErrorResponse)",
        },
        502: {"model": RouteErrorResponse, "description": "All models failed or the request deadline was hit"},
        503: {"model": RouteErrorResponse, "description": "Model gateway or Jev unavailable or not configured"},
    },
)
async def route_prompt(
    body: RouteRequest,
    request: Request,
    response: Response,
    settings: Settings = Depends(get_settings),
    deps_factory: DepsFactory = Depends(get_deps_factory),
    recorder: OutcomeRecorder = Depends(get_outcome_recorder),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> Any:
    peer = request.client.host if request.client is not None else None
    limit = await limiter.check(resolve_client_ip(request.headers, peer, settings.client_ip_header))
    request_id: str = request.state.request_id
    if limit.retry_after_seconds is not None:  # Refused; RateLimitDecision guarantees a wait of >= 1 s
        return _rate_limited(request_id, limit.retry_after_seconds)
    try:
        async with deps_factory(settings, dry_run=body.dry_run) as deps:
            deps = dataclasses.replace(deps, new_request_id=lambda: request_id)
            outcome = await route(body.prompt, dry_run=body.dry_run, deps=deps)
    except LLMAuthError as exc:
        return _setup_error(503, "model_gateway_not_configured", exc, request_id)
    except JevAuthError as exc:
        return _setup_error(503, "classifier_not_configured", exc, request_id)
    except (ModelConfigError, CliConfigurationError, RoutingConfigError) as exc:
        return _setup_error(500, "configuration_error", exc, request_id)

    await _persist(recorder, outcome, body.prompt)  # The same prompt string route() received
    if outcome.status == "failed":
        return _failure(outcome)
    response.headers[REQUEST_ID_HEADER] = outcome.request_id
    return _success(outcome)


async def _persist(recorder: OutcomeRecorder, outcome: RouteOutcome, prompt: str) -> None:
    try:
        await recorder(outcome, prompt)
    except (SQLAlchemyError, OutcomeRowsError) as exc:
        # Fail open: routing availability over persistence. Type only: SQLAlchemy messages include SQL
        # (and row values when hide_parameters is off), and mapping messages name refused fields.
        error_type = type(exc).__name__
        logger.warning(
            "route persistence failed request_id=%s error_type=%s",
            outcome.request_id,
            error_type,
            extra={"request_id": outcome.request_id, "error_type": error_type},
        )


def _success(outcome: RouteOutcome) -> RouteResponse:
    execution = outcome.execution
    model_response = None
    if execution is not None:
        model_response = ModelResponseOut(
            model=execution.final_model,
            tier=execution.final_tier,
            text=execution.result.text,
            latency_ms=execution.result.latency_ms,
            fell_back_to_cheap=execution.fell_back_to_cheap,
            attempts=_attempts_out(outcome.attempts),
        )
    return RouteResponse(
        request_id=outcome.request_id,
        dry_run=outcome.status == "dry_run",
        decision=_decision_out(outcome),
        response=model_response,
        strong_reserved=outcome.strong_reserved,
        total_latency_ms=outcome.total_latency_ms,
        config_version=outcome.config_version,
        cost=_cost_out(outcome.cost),
    )


def _cost_out(cost: Optional[CostEstimate]) -> Optional[CostOut]:
    if cost is None:
        return None
    return CostOut(
        est_cost_usd=cost.est_cost_usd,
        baseline_cost_usd=cost.baseline_cost_usd,
        saved_usd=cost.saved_usd,
        baseline_model=cost.baseline_model,
        basis=cost.basis,
        includes_jev=cost.includes_jev,
    )


def _failure(outcome: RouteOutcome) -> JSONResponse:
    # The router has already logged this request.
    if outcome.failure_reason == "fatal":
        status, error = 503, "model_gateway_unavailable"
    elif outcome.failure_reason == "deadline":
        status, error = 502, "deadline_exceeded"
    else:
        status, error = 502, "all_models_failed"
    body = RouteErrorResponse(
        request_id=outcome.request_id,
        error=error,
        decision=_decision_out(outcome),
        attempts=_attempts_out(outcome.attempts),
        total_latency_ms=outcome.total_latency_ms,
        config_version=outcome.config_version,
    )
    return _json(status, body)


def _setup_error(status: int, error: ErrorCode, exc: Exception, request_id: str) -> JSONResponse:
    # Exception type only: messages can contain file paths.
    logger.warning(
        "route request_id=%s status=%s error=%s error_type=%s",
        request_id,
        status,
        error,
        type(exc).__name__,
        extra={"request_id": request_id, "error": error, "error_type": type(exc).__name__},
    )
    body = RouteErrorResponse(
        request_id=request_id, error=error, decision=None, attempts=[], total_latency_ms=None, config_version=None
    )
    return _json(status, body)


def _rate_limited(request_id: str, retry_after_seconds: int) -> JSONResponse:
    error: ErrorCode = "rate_limited"
    # No client IP or identity: the request ID is enough to correlate.
    logger.info(
        "route request_id=%s status=429 error=%s", request_id, error, extra={"request_id": request_id, "error": error}
    )
    body = RouteErrorResponse(
        request_id=request_id, error=error, decision=None, attempts=[], total_latency_ms=None, config_version=None
    )
    response = _json(429, body)
    response.headers[RETRY_AFTER_HEADER] = str(retry_after_seconds)
    return response


def _json(status: int, body: RouteErrorResponse) -> JSONResponse:
    return JSONResponse(status_code=status, content=body.model_dump(), headers={REQUEST_ID_HEADER: body.request_id})


def _decision_out(outcome: RouteOutcome) -> DecisionOut:
    decision, jev = outcome.decision, outcome.jev
    return DecisionOut(
        task_type=decision.task_type,
        tier=decision.tier,
        reason=decision.reason,
        model_chain=list(decision.model_chain),
        task_type_probs=dict(jev.task_type_probs) if jev else None,
        difficulty=decision.difficulty,
        needs_reasoning_prob=decision.needs_reasoning_prob,
        confidence=decision.confidence,
        jev_latency_ms=jev.jev_latency_ms if jev else None,
        jev_error=decision.jev_error,
        jev_http_status=outcome.jev_http_status,
        jev_error_cause=outcome.jev_error_cause,
    )


def _attempts_out(attempts: tuple[Attempt, ...]) -> list[AttemptOut]:
    return [
        AttemptOut(
            tier=attempt.tier,
            model=attempt.model,
            attempt_no=attempt.attempt_no,
            status=attempt.status,
            latency_ms=attempt.latency_ms,
            input_tokens=attempt.input_tokens,
            output_tokens=attempt.output_tokens,
            reasoning_tokens=attempt.reasoning_tokens,
            http_status=attempt.http_status,
        )
        for attempt in attempts
    ]
