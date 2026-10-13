"""Per-IP rate limit against a real, local Redis (Task S2.6). Skipped by the default `pytest` run.

Runs in CI against the Redis service container. Locally, run it only against a disposable Redis on
this machine, for example:

    REDIS_URL=redis://127.0.0.1:6379/0 python -m pytest -m integration -rA tests/test_ratelimit_integration.py

Isolation: require_local_redis_url (shared with the budget tests) refuses anything but redis:// on a
loopback host. Each test uses a fresh random salt (so identities never match production keys) and a
clock centuries in the future, fails rather than touching a key that already exists, and deletes only
the keys it created. No FLUSHDB, FLUSHALL or key scans.
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest
import redis
import redis.asyncio
from pydantic import SecretStr

from app.core.settings import Settings
from app.infra.ratelimit import (
    KEY_TTL_SECONDS,
    RedisRateLimiter,
    client_identity,
    create_rate_limiter,
    rate_limit_key,
)
from app.infra.redis import TIMEOUT_SECONDS
from app.routes import route as route_api
from tests.test_budget_integration import closed_local_port, require_local_redis_url
from tests.test_route_api import EASY_JEV, PROMPT, ApiHarness

pytestmark = pytest.mark.integration

LIMIT = 10
IP = "203.0.113.7"
OTHER_IP = "198.51.100.23"
FAR_FUTURE = datetime(3000, 1, 1, tzinfo=timezone.utc)
SPREAD_MINUTES = 1_000_000_000  # Random start within ~1,900 years
INDEPENDENT_CLIENTS = 4  # Separate clients, each with its own connection pool (like serverless instances)
TTL_TOLERANCE_SECONDS = 10


@pytest.fixture
def redis_url() -> str:
    return require_local_redis_url(os.environ)


@pytest.fixture
def admin(redis_url):
    """Synchronous client for test setup and inspection only."""
    client = redis.Redis.from_url(redis_url, socket_connect_timeout=TIMEOUT_SECONDS, socket_timeout=TIMEOUT_SECONDS)
    yield client
    client.close()


@pytest.fixture
def window(admin):
    """A fresh salt and a far-future minute; deletes exactly the keys registered through `key()`."""
    salt = SecretStr(f"integration-{uuid.uuid4()}")
    start = (FAR_FUTURE + timedelta(minutes=uuid.uuid4().int % SPREAD_MINUTES)).replace(second=20)
    created: list[str] = []

    class Window:
        def __init__(self):
            self.salt = salt
            self.now = start

        def key(self, ip: str, now: datetime | None = None) -> str:
            name = rate_limit_key(client_identity(ip, salt), now or start)
            if name not in created:
                if admin.exists(name):
                    pytest.fail("A test rate-limit key already exists; refusing to reuse or delete it")
                created.append(name)
            return name

    yield Window()
    if created:
        admin.delete(*created)


def async_client(redis_url: str) -> redis.asyncio.Redis:
    return redis.asyncio.Redis.from_url(
        redis_url, socket_connect_timeout=TIMEOUT_SECONDS, socket_timeout=TIMEOUT_SECONDS
    )


def check_concurrently(redis_url: str, window, ips: list[str]) -> list[bool]:
    """Runs one check per entry of `ips`, all at once, spread over independent clients."""

    async def main():
        clients = [async_client(redis_url) for _ in range(INDEPENDENT_CLIENTS)]
        try:
            limiters = [RedisRateLimiter(c, limit=LIMIT, salt=window.salt, now=lambda: window.now) for c in clients]
            decisions = await asyncio.gather(*(limiters[n % INDEPENDENT_CLIENTS].check(ip) for n, ip in enumerate(ips)))
            return [decision.allowed for decision in decisions]
        finally:
            for client in clients:
                await client.aclose()

    return asyncio.run(main())


def assert_window_ttl(admin, key: str) -> None:
    assert KEY_TTL_SECONDS - TTL_TOLERANCE_SECONDS <= admin.ttl(key) <= KEY_TTL_SECONDS


def test_concurrent_checks_across_independent_clients_admit_exactly_the_limit(redis_url, admin, window):
    key = window.key(IP)

    results = check_concurrently(redis_url, window, [IP] * 40)

    assert results.count(True) == LIMIT
    assert int(admin.get(key)) == 40  # Every check counted, denied ones included
    assert_window_ttl(admin, key)


def test_distinct_identities_stay_isolated_under_concurrency(redis_url, admin, window):
    keys = window.key(IP), window.key(OTHER_IP)

    ips = [IP, OTHER_IP] * 15
    results = check_concurrently(redis_url, window, ips)

    allowed = {ip: sum(ok for ok, each in zip(results, ips) if each == ip) for ip in (IP, OTHER_IP)}
    assert allowed == {IP: LIMIT, OTHER_IP: LIMIT}
    assert [int(admin.get(key)) for key in keys] == [15, 15]


def test_next_window_starts_a_new_counter_with_its_own_ttl(redis_url, admin, window):
    end_of_minute = window.now.replace(second=59, microsecond=999_999)
    next_minute = end_of_minute + timedelta(microseconds=1)
    keys = window.key(IP, end_of_minute), window.key(IP, next_minute)
    now = {"value": end_of_minute}

    async def main():
        client = async_client(redis_url)
        try:
            limiter = RedisRateLimiter(client, limit=LIMIT, salt=window.salt, now=lambda: now["value"])
            first = [await limiter.check(IP) for _ in range(LIMIT + 1)]
            now["value"] = next_minute
            return first, await limiter.check(IP)
        finally:
            await client.aclose()

    first, after = asyncio.run(main())

    assert [d.allowed for d in first] == [True] * LIMIT + [False]
    assert first[-1].retry_after_seconds == 1
    assert after.allowed is True
    assert [int(admin.get(key)) for key in keys] == [LIMIT + 1, 1]
    for key in keys:
        assert_window_ttl(admin, key)


def test_corrupt_counter_fails_open_and_still_gets_a_ttl(redis_url, admin, window, caplog):
    key = window.key(IP)
    admin.set(key, b"not-a-number")  # No TTL yet

    async def main():
        client = async_client(redis_url)
        try:
            return await RedisRateLimiter(client, limit=LIMIT, salt=window.salt, now=lambda: window.now).check(IP)
        finally:
            await client.aclose()

    with caplog.at_level(logging.WARNING, logger="app.infra.ratelimit"):
        decision = asyncio.run(main())

    assert decision.allowed is True
    assert admin.get(key) == b"not-a-number"  # Never repaired or overwritten
    assert_window_ttl(admin, key)  # EXPIRE in the same MULTI/EXEC still ran, so the bad key ages out
    assert [(record.reason, record.error_type) for record in caplog.records] == [("redis_error", "ResponseError")]


def test_closed_local_port_fails_open_with_one_sanitized_warning(caplog):
    url = f"redis://default:super-secret-password@127.0.0.1:{closed_local_port()}/0"
    limiter = create_rate_limiter(Settings(redis_url=url, rate_limit_ip_salt="integration-salt"))
    assert isinstance(limiter, RedisRateLimiter)

    async def main():
        try:
            return await limiter.check(IP)
        finally:
            await limiter.aclose()

    with caplog.at_level(logging.DEBUG, logger="app.infra.ratelimit"):
        decision = asyncio.run(main())

    assert decision.allowed is True
    # Linux refuses a closed loopback port at once (ConnectionError); Windows retries the connect until
    # the 2 s connect timeout fires (TimeoutError). Either way the limiter must fail open, logging once.
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert caplog.records[0].error_type in {"ConnectionError", "TimeoutError"}
    for leaked in ("super-secret-password", "127.0.0.1", IP, "integration-salt"):
        assert leaked not in caplog.text


def test_route_returns_429_after_ten_requests_without_building_dependencies(redis_url, admin, window):
    key = window.key("testclient")  # TestClient's socket peer

    async def real_limiter():
        client = async_client(redis_url)
        try:
            yield RedisRateLimiter(client, limit=LIMIT, salt=window.salt, now=lambda: window.now)
        finally:
            await client.aclose()

    harness = ApiHarness(EASY_JEV)
    harness.client.app.dependency_overrides[route_api.get_rate_limiter] = real_limiter

    statuses = [harness.post({"prompt": PROMPT, "dry_run": True}).status_code for _ in range(LIMIT)]
    blocked = harness.post({"prompt": PROMPT, "dry_run": True})

    assert statuses == [200] * LIMIT
    assert blocked.status_code == 429
    assert blocked.json()["error"] == "rate_limited"
    assert blocked.headers["Retry-After"] == "40"  # window.now is 20 s into its minute
    assert len(harness.factory_calls) == LIMIT  # The 11th request built nothing and never reached Jev
    assert len(harness.jev.calls) == LIMIT
    assert int(admin.get(key)) == LIMIT + 1
    assert_window_ttl(admin, key)
