"""
Health endpoint: API, PostgreSQL, Redis and the model configuration (Task 0.5; S2.8).

200 {"status": "ok", "config_version": "<version>", "db": "ok", "redis": "ok"} when both services
answer and the model configuration loads; otherwise 503 with "status": "degraded". Each service is
"ok", "error" or "not_configured"; `config_version` is load_model_config()'s version for
MODELS_CONFIG_PATH and CONFIDENCE_THRESHOLD (as used by /api/route), or null when that configuration is
invalid (ModelConfigError; decision D7). Everything is always checked. Responses and logs never include
URLs, credentials, file paths or exception messages.

The handler and its dependencies are sync, so FastAPI runs them in a worker thread, off the
event loop. Each check has ~2 s connect/operation timeouts, but those are per operation: they
do not add up to a strict deadline for the whole request.
"""

import logging
from typing import Literal, Optional

import redis as redis_lib
from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel
from sqlalchemy.exc import ArgumentError, OperationalError

from app.core.model_config import ModelConfigError, load_model_config
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
    config_version: Optional[str]  # None when the model configuration is invalid
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


def config_version_status(settings: Settings = Depends(get_settings)) -> Optional[str]:
    try:
        config = load_model_config(settings.models_config_path, confidence_threshold=settings.confidence_threshold)
    except ModelConfigError as exc:  # Its message starts with the file path: log the type only
        _log_failure("config", exc)
        return None
    return config.config_version


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={
        503: {
            "model": HealthResponse,
            "description": "Degraded: database or Redis is unavailable, or the model configuration is invalid",
        }
    },
)
def health(
    response: Response,
    db: ComponentStatus = Depends(database_status),
    redis: ComponentStatus = Depends(redis_status),
    config_version: Optional[str] = Depends(config_version_status),
) -> HealthResponse:
    healthy = db == "ok" and redis == "ok" and config_version is not None
    if not healthy:
        response.status_code = 503
    return HealthResponse(status="ok" if healthy else "degraded", config_version=config_version, db=db, redis=redis)
