"""
Daily cap on strong-tier requests (Task S1.5; docs/04 Part B).

Redis key `budget:strong:{UTC yyyy-mm-dd}`, incremented atomically together with a two-day TTL in
one MULTI/EXEC transaction. A reservation succeeds when the incremented count is at or below
Settings.strong_daily_cap. This counts requests, not dollars: it is not a dollar-budget guarantee.

Usage contract, enforced by the router (Task S1.6): reserve() once per request routed strong,
before execution. There is no refund, because a failed, timed-out or ambiguous call may still be
billed; the cheap fallback inside the executor never reserves. Denied reservations also increment
the counter, so it counts reservation attempts; available() is unaffected because the count is
already at the cap.

Fail closed: when Redis is unconfigured or fails, strong budget is unavailable, so the policy
routes cheap (reason BUDGET). Logs carry only the exception type, never URLs or credentials.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Callable, Protocol

import redis
import redis.asyncio

from app.core.settings import Settings
from app.infra.redis import TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

KEY_PREFIX = "budget:strong"
BUDGET_TTL_SECONDS = 2 * 24 * 60 * 60

# Expected outages: redis-py errors (connection, timeout, server) and raw socket errors.
REDIS_ERRORS = (redis.exceptions.RedisError, OSError)


class StrongBudget(Protocol):
    async def available(self) -> bool: ...

    async def reserve(self) -> bool: ...


def budget_key(day: date) -> str:
    return f"{KEY_PREFIX}:{day.isoformat()}"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class UnavailableStrongBudget:
    """Used when Redis is unconfigured or unusable: strong routing is never available."""

    async def available(self) -> bool:
        return False

    async def reserve(self) -> bool:
        return False


class RedisStrongBudget:
    """
    StrongBudget backed by a Redis counter per UTC day.
    """

    def __init__(
        self,
        client: redis.asyncio.Redis,
        daily_cap: int,
        *,
        now: Callable[[], datetime] = _utc_now,
    ):
        if isinstance(daily_cap, bool) or not isinstance(daily_cap, int) or daily_cap < 0:
            raise ValueError("daily_cap must be an integer >= 0")
        self._client = client
        self.daily_cap = daily_cap
        self._now = now

    async def available(self) -> bool:
        """True while today's count is below the cap. False on any Redis failure."""
        if self.daily_cap == 0:
            return False
        try:
            count = await self._client.get(self._key())
        except REDIS_ERRORS as exc:
            _log_failure("available", exc)
            return False
        return int(count or 0) < self.daily_cap

    async def reserve(self) -> bool:
        """Atomically takes one unit of today's budget. False if over the cap or on Redis failure."""
        if self.daily_cap == 0:
            return False
        key = self._key()
        try:
            async with self._client.pipeline(transaction=True) as pipe:
                pipe.incr(key)
                pipe.expire(key, BUDGET_TTL_SECONDS)
                count, _ = await pipe.execute()
        except REDIS_ERRORS as exc:
            _log_failure("reserve", exc)
            return False
        return count <= self.daily_cap

    async def aclose(self) -> None:
        await self._client.aclose()

    def _key(self) -> str:
        now = self._now()
        if now.tzinfo is None:
            raise ValueError("now() must return a timezone-aware datetime")
        return budget_key(now.astimezone(timezone.utc).date())


def create_strong_budget(settings: Settings) -> RedisStrongBudget | UnavailableStrongBudget:
    """
    Builds the budget from Settings (REDIS_URL, STRONG_DAILY_CAP) without connecting.
    """
    if settings.redis_url is None:
        logger.warning(
            "strong budget unavailable reason=redis_not_configured", extra={"reason": "redis_not_configured"}
        )
        return UnavailableStrongBudget()
    try:
        client = redis.asyncio.Redis.from_url(
            settings.redis_url, socket_connect_timeout=TIMEOUT_SECONDS, socket_timeout=TIMEOUT_SECONDS
        )
    except ValueError as exc:  # Unparseable or unsupported URL
        _log_failure("create", exc)
        return UnavailableStrongBudget()
    return RedisStrongBudget(client, daily_cap=settings.strong_daily_cap)


def _log_failure(operation: str, exc: BaseException) -> None:
    # Only the operation and exception type: redis-py messages can contain hosts, users or URLs.
    error_type = type(exc).__name__
    logger.warning(
        "strong budget unavailable operation=%s error_type=%s",
        operation,
        error_type,
        extra={"operation": operation, "error_type": error_type},
    )
