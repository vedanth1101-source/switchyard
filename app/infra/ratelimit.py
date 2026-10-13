"""
Per-IP rate limit for POST /api/route (Task S2.6; docs/04 Part B).

Fixed one-minute windows aligned to the UTC clock: key `rl:{identity}:{UTC yyyymmddHHMM}`, incremented
together with a 120 s TTL in one MULTI/EXEC transaction (the strong budget's pattern). Redis runs the
queued INCR and EXPIRE back to back with nothing in between and never rolls back, so every counter the
transaction touches gets the TTL, even when INCR fails on a corrupt value. INCR returns each caller a
distinct count, so "count <= limit" admits at most `limit` requests per identity and window however many
arrive at once. Denied checks also increment the counter.

Boundary semantics: a window resets at the start of each UTC minute, so a client can make up to
2 x limit requests in a short burst that straddles a minute boundary. Retry-After is the whole number
of seconds (at least 1) until the next window starts.

Identity: HMAC-SHA256(RATE_LIMIT_IP_SALT, client IP). Raw IPs are never stored or logged. The client IP
comes from the trusted CLIENT_IP_HEADER when it is configured and holds a valid IP, else from the socket
peer; other forwarded headers are never trusted.

Fail open: when REDIS_URL or RATE_LIMIT_IP_SALT is unset, the URL is unusable, or Redis fails, the request
is allowed and exactly one WARNING per check is logged, carrying a reason and at most the exception
type, never URLs, IPs, the salt or exception messages. (The strong budget, by contrast, fails closed.)
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional, Protocol

import redis
import redis.asyncio
from pydantic import SecretStr

from app.core.settings import Settings
from app.infra.redis import TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

KEY_PREFIX = "rl"
WINDOW_SECONDS = 60
KEY_TTL_SECONDS = 2 * WINDOW_SECONDS
UNKNOWN_CLIENT = "unknown"

# Expected outages: redis-py errors (connection, timeout, server) and raw socket errors.
REDIS_ERRORS = (redis.exceptions.RedisError, OSError)


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: Optional[int] = None  # Set only when the request is refused

    def __post_init__(self) -> None:
        refused_without_wait = not self.allowed and (self.retry_after_seconds is None or self.retry_after_seconds < 1)
        if refused_without_wait or (self.allowed and self.retry_after_seconds is not None):
            raise ValueError("retry_after_seconds must be >= 1 when refused and None when allowed")


ALLOWED = RateLimitDecision(allowed=True)


class RateLimiter(Protocol):
    async def check(self, client_ip: str) -> RateLimitDecision: ...

    async def aclose(self) -> None: ...


def resolve_client_ip(headers: Mapping[str, str], peer: Optional[str], trusted_header: Optional[str]) -> str:
    """The trusted header's first address when it is a valid IP, else the socket peer."""
    if trusted_header is not None:
        candidate = (headers.get(trusted_header) or "").split(",")[0].strip()
        try:
            return str(ipaddress.ip_address(candidate))
        except ValueError:
            pass  # Missing or malformed: fall back to the peer
    return peer or UNKNOWN_CLIENT


def client_identity(client_ip: str, salt: SecretStr) -> str:
    return hmac.new(salt.get_secret_value().encode(), client_ip.encode(), hashlib.sha256).hexdigest()


def rate_limit_key(identity: str, now: datetime) -> str:
    return f"{KEY_PREFIX}:{identity}:{_utc(now):%Y%m%d%H%M}"


def retry_after_seconds(now: datetime) -> int:
    utc = _utc(now)
    next_window = utc.replace(second=0, microsecond=0) + timedelta(seconds=WINDOW_SECONDS)
    return max(1, math.ceil((next_window - utc).total_seconds()))


def _utc(now: datetime) -> datetime:
    if now.tzinfo is None:
        raise ValueError("now() must return a timezone-aware datetime")
    return now.astimezone(timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class UnavailableRateLimiter:
    """Used when Redis or the salt is not configured or usable: every request is allowed, with a warning."""

    def __init__(self, reason: str, error_type: Optional[str] = None):
        self.reason = reason
        self.error_type = error_type

    async def check(self, client_ip: str) -> RateLimitDecision:
        _log_unavailable(self.reason, self.error_type)
        return ALLOWED

    async def aclose(self) -> None:
        pass


class RedisRateLimiter:
    """
    RateLimiter backed by one Redis counter per client identity and UTC minute.
    """

    def __init__(
        self,
        client: redis.asyncio.Redis,
        limit: int,
        salt: SecretStr,
        *,
        now: Callable[[], datetime] = _utc_now,
    ):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be an integer >= 1")
        self._client = client
        self.limit = limit
        self._salt = salt
        self._now = now

    async def check(self, client_ip: str) -> RateLimitDecision:
        """Counts this request in the current window; refuses it when the window is over the limit."""
        now = self._now()
        key = rate_limit_key(client_identity(client_ip, self._salt), now)
        try:
            async with self._client.pipeline(transaction=True) as pipe:
                pipe.incr(key)
                pipe.expire(key, KEY_TTL_SECONDS)
                count, _ = await pipe.execute()
        except REDIS_ERRORS as exc:
            _log_unavailable("redis_error", type(exc).__name__)
            return ALLOWED
        if count <= self.limit:
            return ALLOWED
        return RateLimitDecision(allowed=False, retry_after_seconds=retry_after_seconds(now))

    async def aclose(self) -> None:
        await self._client.aclose()


def create_rate_limiter(settings: Settings) -> RedisRateLimiter | UnavailableRateLimiter:
    """
    Builds the limiter from Settings (REDIS_URL, RATE_LIMIT_IP_SALT, RATE_LIMIT_PER_MIN) without connecting.
    """
    if settings.redis_url is None:
        return UnavailableRateLimiter("redis_not_configured")
    if settings.rate_limit_ip_salt is None:
        return UnavailableRateLimiter("salt_not_configured")
    try:
        client = redis.asyncio.Redis.from_url(
            settings.redis_url, socket_connect_timeout=TIMEOUT_SECONDS, socket_timeout=TIMEOUT_SECONDS
        )
    except ValueError as exc:  # Unparseable or unsupported URL
        return UnavailableRateLimiter("invalid_redis_url", type(exc).__name__)
    return RedisRateLimiter(client, limit=settings.rate_limit_per_min, salt=settings.rate_limit_ip_salt)


def _log_unavailable(reason: str, error_type: Optional[str]) -> None:
    # A reason and the exception type only: redis-py messages can contain hosts, users or URLs.
    logger.warning(
        "rate limit unavailable, request allowed reason=%s error_type=%s",
        reason,
        error_type,
        extra={"reason": reason, "error_type": error_type},
    )
