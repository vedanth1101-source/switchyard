"""
Health endpoint: API, PostgreSQL and Redis (Task 0.5).

200 {"status": "ok", "db": "ok", "redis": "ok"} when both services answer; otherwise 503 with
"status": "degraded" and each component as "ok", "error" or "not_configured". Both services are
always checked. Responses and logs never include URLs, credentials or exception messages.

`config_version` from docs/04-data-schema-and-api.md is omitted until Task 1.1 provides it.

The handler and its dependencies are sync, so FastAPI runs them in a worker thread, off the
event loop. Each check has ~2 s connect/operation timeouts, but those are per operation: they
do not add up to a strict deadline for the whole request.
"""

import logging
from typing import Literal

import redis as redis_lib
from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel
from sqlalchemy.exc import ArgumentError, OperationalError

from app.core.settings import Settings, get_settings
from app.infra.db import check_database, get_engine
from app.infra.redis import check_redis, create_redis_client

logger = logging.getLogger(__name__)
router = APIRouter()

ComponentStatus = Literal["ok", "error", "not_configured"]

# Expected failures only: unreachable/timed-out services and unusable configuration.
# Anything else is a programming error and propagates.
DATABASE_ERRORS = (OperationalError, ArgumentError)
REDIS_ERRORS = (redis_lib.exceptions.ConnectionError, redis_lib.exceptions.TimeoutError, ValueError)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    db: ComponentStatus
    redis: ComponentStatus


def _log_failure(component: str, exc: Exception) -> None:
    # Only the component and exception type: driver messages can contain hosts, users or URLs.
    # Both go in the message because default logging does not render `extra` fields.
    error_type = type(exc).__name__
    logger.warning(
        "health check failed component=%s error_type=%s",
        component,
        error_type,
        extra={"component": component, "error_type": error_type},
    )


def database_status(settings: Settings = Depends(get_settings)) -> ComponentStatus:
    if settings.database_url is None:
        return "not_configured"
    try:
        check_database(get_engine(settings.database_url))
    except DATABASE_ERRORS as exc:
        _log_failure("db", exc)
        return "error"
    return "ok"


def redis_status(settings: Settings = Depends(get_settings)) -> ComponentStatus:
    if settings.redis_url is None:
        return "not_configured"
    try:
        client = create_redis_client(settings.redis_url)
    except REDIS_ERRORS as exc:
        _log_failure("redis", exc)
        return "error"
    try:
        check_redis(client)
    except REDIS_ERRORS as exc:
        _log_failure("redis", exc)
        return "error"
    finally:
        client.close()
    return "ok"


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "Degraded: database or Redis is unavailable"}},
)
def health(
    response: Response,
    db: ComponentStatus = Depends(database_status),
    redis: ComponentStatus = Depends(redis_status),
) -> HealthResponse:
    healthy = db == "ok" and redis == "ok"
    if not healthy:
        response.status_code = 503
    return HealthResponse(status="ok" if healthy else "degraded", db=db, redis=redis)
