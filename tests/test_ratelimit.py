"""Per-IP rate limit (Task S2.6) against fakeredis; no network access."""

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone

import fakeredis
import pytest
import redis
from pydantic import SecretStr

from app.core.settings import Settings
from app.infra.ratelimit import (
    KEY_TTL_SECONDS,
    RateLimitDecision,
    RedisRateLimiter,
    UnavailableRateLimiter,
    client_identity,
    create_rate_limiter,
    rate_limit_key,
    resolve_client_ip,
    retry_after_seconds,
)

NOW = datetime(2026, 10, 13, 12, 0, 30, tzinfo=timezone.utc)
IP = "203.0.113.7"
OTHER_IP = "198.51.100.23"
SALT = SecretStr("unit-test-salt-not-a-real-secret")
SALT_TEXT = SALT.get_secret_value()
SECRET_URL = "rediss://default:super-secret-password@redis.example.test:6379"
LIMIT = 10


class MutableNow:
    def __init__(self, value: datetime):
        self.value = value

    def __call__(self) -> datetime:
        return self.value


def make_limiter(limit=LIMIT, server=None, now=None, salt=SALT) -> tuple[RedisRateLimiter, fakeredis.FakeAsyncRedis]:
    client = fakeredis.FakeAsyncRedis(server=server or fakeredis.FakeServer())
    return RedisRateLimiter(client, limit=limit, salt=salt, now=now or (lambda: NOW)), client


def key_for(ip: str, now: datetime = NOW, salt: SecretStr = SALT) -> str:
    return rate_limit_key(client_identity(ip, salt), now)


async def check_times(limiter: RedisRateLimiter, ip: str, times: int) -> list[bool]:
    return [(await limiter.check(ip)).allowed for _ in range(times)]


class FailingPipeline:
    def __init__(self, error: Exception):
        self.error = error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    def incr(self, key):
        return self

    def expire(self, key, seconds):
        return self

    async def execute(self):
        raise self.error


class FailingRedis:
    """Async Redis stand-in whose transactions raise `error`."""

    def __init__(self, error: Exception):
        self.error = error

    def pipeline(self, transaction=True):
        return FailingPipeline(self.error)


# --- Limit boundaries -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("earlier_checks", "allowed"),
    [(8, True), (9, True), (10, False)],
    ids=["below-limit", "exact-limit", "over-limit"],
)
def test_the_next_check_is_allowed_up_to_the_limit(earlier_checks, allowed):
    async def scenario():
        limiter, _ = make_limiter()
        await check_times(limiter, IP, earlier_checks)
        return await limiter.check(IP)

    decision = asyncio.run(scenario())

    assert decision.allowed is allowed
    assert decision.retry_after_seconds == (None if allowed else 30)  # NOW is 30 s into its minute


def test_rate_limit_allows_then_blocks():
    async def scenario():
        limiter, client = make_limiter()
        return await check_times(limiter, IP, 12), await client.get(key_for(IP))

    results, count = asyncio.run(scenario())

    assert results == [True] * LIMIT + [False, False]
    assert count == b"12"  # Denied checks count too, like denied strong-budget reservations


def test_rate_limit_resets_next_minute():
    now = MutableNow(NOW)

    async def scenario():
        limiter, _ = make_limiter(now=now)
        first = await check_times(limiter, IP, LIMIT + 1)
        now.value = NOW.replace(second=0) + timedelta(minutes=1)
        return first, await check_times(limiter, IP, LIMIT + 1)

    first, second = asyncio.run(scenario())

    assert first == [True] * LIMIT + [False]
    assert second == [True] * LIMIT + [False]


def test_distinct_ips_are_isolated():
    async def scenario():
        limiter, _ = make_limiter()
        blocked = await check_times(limiter, IP, LIMIT + 1)
        return blocked, await check_times(limiter, OTHER_IP, LIMIT + 1)

    blocked, other = asyncio.run(scenario())

    assert blocked[-1] is False
    assert other == [True] * LIMIT + [False]


def test_concurrent_checks_never_admit_more_than_the_limit():
    async def scenario():
        server = fakeredis.FakeServer()
        limiters = [make_limiter(server=server)[0] for _ in range(4)]  # Independent clients, shared server
        return await asyncio.gather(*(limiters[n % 4].check(IP) for n in range(40)))

    decisions = asyncio.run(scenario())

    assert sum(decision.allowed for decision in decisions) == LIMIT


# --- Keys and TTL ---------------------------------------------------------------------------


def test_key_holds_an_hmac_identity_and_the_utc_minute_never_the_raw_ip():
    async def scenario():
        limiter, client = make_limiter()
        await limiter.check(IP)
        return await client.keys("*")

    (key,) = asyncio.run(scenario())

    assert re.fullmatch(rb"rl:[0-9a-f]{64}:202610131200", key)
    assert IP.encode() not in key
    assert SALT_TEXT.encode() not in key


def test_identity_depends_on_both_ip_and_salt():
    identity = client_identity(IP, SALT)

    assert identity == client_identity(IP, SecretStr(SALT_TEXT))  # Stable across instances
    assert identity != client_identity(OTHER_IP, SALT)
    assert identity != client_identity(IP, SecretStr("another-salt"))
    assert IP not in identity


def test_key_uses_the_utc_minute_of_a_non_utc_clock():
    plus_two = timezone(timedelta(hours=2))

    assert rate_limit_key("abc", datetime(2026, 10, 13, 1, 30, 5, tzinfo=plus_two)) == "rl:abc:202610122330"


def test_naive_clock_is_rejected():
    limiter, _ = make_limiter(now=lambda: datetime(2026, 10, 13, 12, 0))

    with pytest.raises(ValueError, match="timezone-aware"):
        asyncio.run(limiter.check(IP))


def test_check_sets_the_window_ttl():
    async def scenario():
        limiter, client = make_limiter()
        await limiter.check(IP)
        return await client.ttl(key_for(IP))

    assert asyncio.run(scenario()) == KEY_TTL_SECONDS == 120


def test_check_adds_a_ttl_to_an_existing_counter_without_one():
    async def scenario():
        limiter, client = make_limiter()
        await client.set(key_for(IP), 3)
        decision = await limiter.check(IP)
        return decision.allowed, await client.get(key_for(IP)), await client.ttl(key_for(IP))

    assert asyncio.run(scenario()) == (True, b"4", KEY_TTL_SECONDS)


def test_corrupt_counter_fails_open_once_and_still_gets_a_ttl(caplog):
    async def scenario():
        limiter, client = make_limiter()
        await client.set(key_for(IP), b"not-a-number")
        decision = await limiter.check(IP)
        return decision, await client.get(key_for(IP)), await client.ttl(key_for(IP))

    with caplog.at_level(logging.WARNING, logger="app.infra.ratelimit"):
        decision, stored, ttl = asyncio.run(scenario())

    assert decision.allowed is True
    assert stored == b"not-a-number"  # Never repaired or overwritten
    assert ttl == KEY_TTL_SECONDS  # MULTI/EXEC still ran EXPIRE, so the bad key ages out
    assert [(record.levelno, record.reason, record.error_type) for record in caplog.records] == [
        (logging.WARNING, "redis_error", "ResponseError")
    ]


@pytest.mark.parametrize("limit", [0, -1, True, 2.5, "10"])
def test_limit_must_be_a_positive_integer(limit):
    with pytest.raises(ValueError, match="limit"):
        RedisRateLimiter(fakeredis.FakeAsyncRedis(), limit=limit, salt=SALT)


# --- Retry-After ----------------------------------------------------------------------------


@pytest.mark.parametrize(("allowed", "retry_after"), [(False, None), (False, 0), (True, 5)])
def test_a_refusal_needs_a_positive_retry_after_and_an_allowance_has_none(allowed, retry_after):
    with pytest.raises(ValueError, match="retry_after_seconds"):
        RateLimitDecision(allowed=allowed, retry_after_seconds=retry_after)


@pytest.mark.parametrize(
    ("second", "microsecond", "expected"),
    [(0, 0, 60), (0, 500_000, 60), (30, 0, 30), (30, 200_000, 30), (59, 0, 1), (59, 999_999, 1)],
)
def test_retry_after_is_whole_seconds_to_the_next_window_and_at_least_one(second, microsecond, expected):
    now = datetime(2026, 10, 13, 12, 0, second, microsecond, tzinfo=timezone.utc)

    assert retry_after_seconds(now) == expected


# --- Redis failures: fail open, one sanitized WARNING per check -----------------------------


@pytest.mark.parametrize(
    "error",
    [
        redis.exceptions.ConnectionError(f"Error connecting to {SECRET_URL} for {IP}"),
        redis.exceptions.TimeoutError(f"Timeout reading from {SECRET_URL}"),
        redis.exceptions.ResponseError(f"WRONGTYPE {IP} {SALT_TEXT}"),
        ConnectionResetError(f"reset by {SECRET_URL}"),
    ],
    ids=["connection", "timeout", "response", "socket"],
)
def test_redis_failure_fails_open_with_exactly_one_sanitized_warning(caplog, error):
    limiter = RedisRateLimiter(FailingRedis(error), limit=LIMIT, salt=SALT, now=lambda: NOW)

    with caplog.at_level(logging.DEBUG, logger="app.infra.ratelimit"):
        decision = asyncio.run(limiter.check(IP))

    assert (decision.allowed, decision.retry_after_seconds) == (True, None)
    assert [(record.levelno, record.reason, record.error_type) for record in caplog.records] == [
        (logging.WARNING, "redis_error", type(error).__name__)
    ]
    for leaked in ("super-secret-password", "redis.example.test", IP, SALT_TEXT, "WRONGTYPE"):
        assert leaked not in caplog.text


def test_each_failed_check_logs_its_own_warning(caplog):
    limiter = RedisRateLimiter(FailingRedis(redis.exceptions.ConnectionError()), limit=LIMIT, salt=SALT)

    async def scenario():
        return [await limiter.check(IP) for _ in range(3)]

    with caplog.at_level(logging.WARNING, logger="app.infra.ratelimit"):
        decisions = asyncio.run(scenario())

    assert all(decision.allowed for decision in decisions)
    assert len(caplog.records) == 3


def test_disconnected_fake_server_fails_open():
    async def scenario():
        server = fakeredis.FakeServer()
        limiter, _ = make_limiter(server=server)
        server.connected = False
        return await limiter.check(IP)

    assert asyncio.run(scenario()).allowed is True


def test_unexpected_error_is_not_swallowed():
    limiter = RedisRateLimiter(FailingRedis(RuntimeError("bug")), limit=LIMIT, salt=SALT, now=lambda: NOW)

    with pytest.raises(RuntimeError, match="bug"):
        asyncio.run(limiter.check(IP))


# --- Factory: missing configuration fails open ----------------------------------------------


def check_once(limiter, caplog) -> tuple:
    with caplog.at_level(logging.DEBUG, logger="app.infra.ratelimit"):
        decision = asyncio.run(limiter.check(IP))
    records = [(r.levelno, r.reason, r.error_type) for r in caplog.records if r.name == "app.infra.ratelimit"]
    return decision, records


def test_missing_redis_url_fails_open_with_one_warning(caplog):
    limiter = create_rate_limiter(Settings(redis_url=None, rate_limit_ip_salt=SALT_TEXT))

    decision, records = check_once(limiter, caplog)

    assert isinstance(limiter, UnavailableRateLimiter)
    assert (decision.allowed, decision.retry_after_seconds) == (True, None)
    assert records == [(logging.WARNING, "redis_not_configured", None)]
    assert IP not in caplog.text


def test_missing_salt_fails_open_with_one_warning_and_never_builds_a_client(caplog, monkeypatch):
    def no_client(*args, **kwargs):
        raise AssertionError("no Redis client without a salt")

    monkeypatch.setattr(redis.asyncio.Redis, "from_url", no_client)

    limiter = create_rate_limiter(Settings(redis_url=SECRET_URL, rate_limit_ip_salt=None))
    decision, records = check_once(limiter, caplog)

    assert isinstance(limiter, UnavailableRateLimiter)
    assert decision.allowed is True
    assert records == [(logging.WARNING, "salt_not_configured", None)]
    assert "super-secret-password" not in caplog.text


def test_invalid_redis_url_fails_open_with_only_the_error_type(caplog):
    url = "http://user:super-secret-password@not-redis.example.test"

    limiter = create_rate_limiter(Settings(redis_url=url, rate_limit_ip_salt=SALT_TEXT))
    decision, records = check_once(limiter, caplog)

    assert isinstance(limiter, UnavailableRateLimiter)
    assert decision.allowed is True
    assert records == [(logging.WARNING, "invalid_redis_url", "ValueError")]
    assert "super-secret-password" not in caplog.text
    assert "not-redis.example.test" not in caplog.text


def test_unavailable_limiter_logs_once_per_check(caplog):
    limiter = create_rate_limiter(Settings(redis_url=None))

    async def scenario():
        return [await limiter.check(IP) for _ in range(2)]

    with caplog.at_level(logging.WARNING, logger="app.infra.ratelimit"):
        asyncio.run(scenario())
        asyncio.run(limiter.aclose())

    assert len(caplog.records) == 2


def test_configured_limiter_uses_the_settings_limit_without_connecting():
    limiter = create_rate_limiter(
        Settings(redis_url="redis://localhost:6399/0", rate_limit_ip_salt=SALT_TEXT, rate_limit_per_min=7)
    )

    assert isinstance(limiter, RedisRateLimiter)
    assert limiter.limit == 7
    asyncio.run(limiter.aclose())


def test_factory_default_limit_is_ten_per_minute():
    limiter = create_rate_limiter(Settings(redis_url="redis://localhost:6399/0", rate_limit_ip_salt=SALT_TEXT))

    assert limiter.limit == Settings.model_fields["rate_limit_per_min"].default == 10
    asyncio.run(limiter.aclose())


# --- Client IP resolution -------------------------------------------------------------------


def test_socket_peer_is_used_when_no_header_is_configured():
    headers = {"x-forwarded-for": OTHER_IP, "x-real-ip": OTHER_IP}

    assert resolve_client_ip(headers, IP, None) == IP  # Forwarded headers are not trusted by default


def test_configured_header_is_used():
    assert resolve_client_ip({"x-real-ip": OTHER_IP}, IP, "x-real-ip") == OTHER_IP


def test_configured_header_takes_the_first_address_of_a_list():
    assert resolve_client_ip({"x-forwarded-for": f" {OTHER_IP} , 10.0.0.1"}, IP, "x-forwarded-for") == OTHER_IP


def test_configured_header_ipv6_is_normalised():
    assert resolve_client_ip({"x-real-ip": "2001:DB8:0:0::1"}, IP, "x-real-ip") == "2001:db8::1"


@pytest.mark.parametrize(
    "headers",
    [{}, {"x-real-ip": ""}, {"x-real-ip": "  "}, {"x-real-ip": "not-an-ip"}, {"x-real-ip": "300.1.1.1"}],
    ids=["missing", "empty", "blank", "garbage", "out-of-range"],
)
def test_missing_or_malformed_configured_header_falls_back_to_the_socket_peer(headers):
    assert resolve_client_ip(headers, IP, "x-real-ip") == IP


def test_non_ip_or_missing_peer_still_resolves_to_an_identity():
    assert resolve_client_ip({}, "testclient", None) == "testclient"
    assert resolve_client_ip({}, None, None) == "unknown"
