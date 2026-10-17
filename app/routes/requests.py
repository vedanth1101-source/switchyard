"""
GET /api/requests/{request_id} (Task S2.13; docs/04 Part C): one saved routing outcome.

Returns what POST /api/route saved for a request: the request and its outcome, the final routing
decision and every model attempt in order. The prompt is never returned in any form: neither its text
(not stored) nor its prompt_hash, an unsalted sha256 that would let anyone holding an ID confirm a
guessed prompt. prompt_chars is kept. USD values are JSON numbers (floats) like /api/route's `cost`;
the database keeps NUMERIC(12,8). NULL stays null: unknown costs, tokens and Jev fields are never 0.

Access: public, keyed only by the request ID (a uuid4), which works as a capability. Anyone who holds
an ID (the client that made the request, or anyone it is shared with or who reads logs carrying it)
can read that request's decision, models, tokens and costs. There is no authentication and no rate
limit, as for /api/health; each read is one short transaction under the 2 s statement timeout.

Status codes: 200 found · 404 no saved request with this ID (error=not_found) · 422 the ID is not a
UUID (ValidationErrorResponse, built by app.main) · 503 the database failed (database_unavailable) or
DATABASE_URL is not set (database_not_configured). Error bodies are {request_id, error}, where
request_id is this HTTP request's own ID (the X-Request-ID header), not the one looked up. A database
failure logs one warning with that ID and the exception type only; any other exception is a bug and
reaches the generic 500.
"""

import functools
import logging
import uuid
from datetime import datetime
from typing import Any, Awaitable, Callable, Literal, Mapping, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError
from starlette.concurrency import run_in_threadpool

from app.core.settings import Settings, get_settings
from app.db.repo import OutcomeRows, load_request
from app.infra.db import get_engine
from app.routes.route import ValidationErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter()

RequestErrorCode = Literal["not_found", "database_unavailable", "database_not_configured"]


class StoredRequestOut(BaseModel):
    id: uuid.UUID
    created_at: datetime
    status: str
    dry_run: bool
    failure_reason: Optional[str]
    final_model: Optional[str]
    final_tier: Optional[str]
    fell_back_to_cheap: Optional[bool]
    strong_reserved: bool
    total_latency_ms: int
    config_version: str
    prompt_chars: int
    est_cost_usd: Optional[float]  # List-price estimate for the successful call; null when unknown or a cache hit
    baseline_cost_usd: Optional[float]
    cache_hit: bool  # Answered from the response cache (S2.14): no model calls and no cost


class StoredDecisionOut(BaseModel):
    task_type: str
    tier: str
    reason: str
    model_chain: list[str]
    confidence: Optional[float]  # Jev fields are null when Jev failed
    difficulty: Optional[float]
    needs_reasoning_prob: Optional[float]
    task_type_probs: Optional[dict[str, float]]
    difficulty_dist: Optional[dict[str, float]]
    jev_model: Optional[str]
    jev_latency_ms: Optional[int]
    jev_input_tokens: Optional[int]
    jev_credits_charged: Optional[int]  # Jev credits, never USD
    jev_error: Optional[str]  # Exception class name only
    jev_http_status: Optional[int]
    jev_error_cause: Optional[str]  # Exception class name only


class StoredAttemptOut(BaseModel):
    seq: int
    tier: str
    model: str
    attempt_no: int
    status: str
    latency_ms: int
    http_status: Optional[int]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    reasoning_tokens: Optional[int]
    est_cost_usd: Optional[float]  # Null when the attempt reported no usage (S2.11)


class StoredRequestResponse(BaseModel):
    request: StoredRequestOut
    decision: StoredDecisionOut
    attempts: list[StoredAttemptOut]  # In order (seq); empty for dry runs


class RequestErrorResponse(BaseModel):
    request_id: str  # This HTTP request's ID (X-Request-ID), not the ID looked up
    error: RequestErrorCode


RequestReader = Callable[[uuid.UUID], Awaitable[Optional[OutcomeRows]]]


def get_request_reader(settings: Settings = Depends(get_settings)) -> Optional[RequestReader]:
    """Reads from DATABASE_URL, or None when it is not set; tests override this."""
    if settings.database_url is None:
        return None
    return functools.partial(read_request, database_url=settings.database_url)


async def read_request(request_id: uuid.UUID, *, database_url: str) -> Optional[OutcomeRows]:
    """The blocking database read runs in Starlette's thread pool."""
    return await run_in_threadpool(load_request, get_engine(database_url), request_id)


@router.get(
    "/requests/{request_id}",
    response_model=StoredRequestResponse,
    responses={
        404: {"model": RequestErrorResponse, "description": "No saved request with this ID (error=not_found)"},
        422: {"model": ValidationErrorResponse, "description": "The ID is not a UUID"},
        503: {
            "model": RequestErrorResponse,
            "description": "The database failed (database_unavailable) or is not configured (database_not_configured)",
        },
    },
)
async def get_stored_request(
    request_id: uuid.UUID, request: Request, reader: Optional[RequestReader] = Depends(get_request_reader)
) -> Any:
    own_id: str = request.state.request_id
    if reader is None:
        return _error(503, "database_not_configured", own_id)
    try:
        rows = await reader(request_id)
    except SQLAlchemyError as exc:
        # Type only: SQLAlchemy messages can include SQL and, without hide_parameters, row values.
        error_type = type(exc).__name__
        logger.warning(
            "request read failed request_id=%s error_type=%s",
            own_id,
            error_type,
            extra={"request_id": own_id, "error_type": error_type},
        )
        return _error(503, "database_unavailable", own_id)
    if rows is None:
        return _error(404, "not_found", own_id)
    return _response(rows)


def _response(rows: OutcomeRows) -> StoredRequestResponse:
    return StoredRequestResponse(
        request=StoredRequestOut(**_pick(rows.request, StoredRequestOut)),
        decision=StoredDecisionOut(**_pick(rows.request, StoredDecisionOut)),
        attempts=[StoredAttemptOut(**_pick(call, StoredAttemptOut)) for call in rows.model_calls],
    )


def _pick(row: Mapping[str, Any], model: type[BaseModel]) -> dict[str, Any]:
    # Only the response model's fields: prompt_hash (and any column added later) is never copied.
    return {name: row[name] for name in model.model_fields}


def _error(status: int, error: RequestErrorCode, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status, content=RequestErrorResponse(request_id=request_id, error=error).model_dump()
    )
