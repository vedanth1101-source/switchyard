"""Strong daily budget against a real, local Redis (Task S2.2). Skipped by the default `pytest` run.

Runs in CI against the Redis service container. Locally, run it only against a disposable Redis on
this machine, for example:

    REDIS_URL=redis://127.0.0.1:6379/0 python -m pytest -m integration -rA tests/test_budget_integration.py

Isolation: require_local_redis_url refuses anything but redis:// on a loopback host, so never Upstash
(rediss://) or any remote server. Each test uses budget dates centuries in the future (production keys
use today's UTC date), fails rather than touching a key that already exists, and deletes only the keys
it created. No FLUSHDB, FLUSHALL or key scans.
"""

import asyncio
import logging
import os
import socket
import time
import uuid
from collections.abc import Mapping
from datetime import date, datetime, timedelta, timezone
from datetime import time as clock_time
from urllib.parse import urlsplit

import pytest
import redis
import redis.asyncio

from app.core.settings import Settings
from app.infra.budget import BUDGET_TTL_SECONDS, RedisStrongBudget, budget_key, create_strong_budget
from app.infra.redis import TIMEOUT_SECONDS

pytestmark = pytest.mark.integration

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
FAR_FUTURE = date(3000, 1, 1)
SPREAD_DAYS = 1_000_000  # Random start within ~2,700 years; stays well below date.max
INDEPENDENT_CLIENTS = 4  # Separate clients, each with its own connection pool (like serverless instances)
TTL_TOLERANCE_SECONDS = 10


def require_local_redis_url(environ: Mapping[str, str]) -> str:
    """Fail unless REDIS_URL is redis:// on a loopback host. Messages never include the URL."""
    # Hide this frame from tracebacks: pytest would otherwise print the `environ` argument, including
    # REDIS_URL's password and every other variable.
    __tracebackhide__ = True
    redis_url = environ.get("REDIS_URL", "").strip()
    if not redis_url:
        pytest.fail("Budget integration test needs a local Redis service; set REDIS_URL")
    try:
        parts = urlsplit(redis_url)
        scheme, host = parts.scheme, parts.hostname
    except ValueError:
        scheme, host = None, None
    if scheme != "redis":
        pytest.fail("REDIS_URL must use redis:// (rediss:// and other schemes are refused; this test writes keys)")
    if host not in LOCAL_HOSTS:
        pytest.fail("Budget integration test writes keys; it only runs against a local host")
    return redis_url


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
def fresh_days(admin):
    """Allocates consecutive unused far-future dates; deletes exactly their budget keys afterwards."""
    created: list[date] = []

    def allocate(count: int = 1) -> list[date]:
        start = FAR_FUTURE + timedelta(days=uuid.uuid4().int % SPREAD_DAYS)
        days = [start + timedelta(days=offset) for offset in range(count)]
        if admin.exists(*(budget_key(day) for day in days)):
            pytest.fail("A test budget key already exists; refusing to reuse or delete it")
        created.extend(days)
        return days

    yield allocate
    if created:
        admin.delete(*(budget_key(day) for day in created))


class MutableNow:
    def __init__(self, value: datetime):
        self.value = value

    def __call__(self) -> datetime:
        return self.value


def noon(day: date):
    return lambda: datetime.combine(day, clock_time(12), tzinfo=timezone.utc)


def async_client(redis_url: str) -> redis.asyncio.Redis:
    return redis.asyncio.Redis.from_url(
        redis_url, socket_connect_timeout=TIMEOUT_SECONDS, socket_timeout=TIMEOUT_SECONDS
    )


def run_with_budget(redis_url: str, cap: int, now, scenario):
    """Runs `scenario(budget)` on a fresh event loop with its own client, closed afterwards."""

    async def main():
        client = async_client(redis_url)
        try:
            return await scenario(RedisStrongBudget(client, daily_cap=cap, now=now))
        finally:
            await client.aclose()

    return asyncio.run(main())


def reserve_concurrently(redis_url: str, day: date, cap: int, total: int) -> list[bool]:
    async def main():
        clients = [async_client(redis_url) for _ in range(INDEPENDENT_CLIENTS)]
        try:
            budgets = [RedisStrongBudget(client, daily_cap=cap, now=noon(day)) for client in clients]
            return await asyncio.gather(*(budgets[n % INDEPENDENT_CLIENTS].reserve() for n in range(total)))
        finally:
            for client in clients:
                await client.aclose()

    return asyncio.run(main())


def counter(admin, day: date) -> int:
    return int(admin.get(budget_key(day)))


def assert_two_day_ttl(admin, day: date) -> None:
    assert BUDGET_TTL_SECONDS - TTL_TOLERANCE_SECONDS <= admin.ttl(budget_key(day)) <= BUDGET_TTL_SECONDS


# --- Cap ------------------------------------------------------------------------------------


def test_cap_allows_exactly_three_of_five_and_counts_denied_reservations(redis_url, admin, fresh_days):
    (day,) = fresh_days()

    async def scenario(budget):
        before = await budget.available()
        results = [await budget.reserve() for _ in range(5)]
        return before, results, await budget.available()

    before, results, after = run_with_budget(redis_url, 3, noon(day), scenario)

    assert before is True
    assert results == [True, True, True, False, False]
    assert after is False
    assert counter(admin, day) == 5  # Denied reservations still increment the counter


# --- Concurrency ----------------------------------------------------------------------------


def test_concurrent_reservations_across_independent_clients_never_exceed_the_cap(redis_url, admin, fresh_days):
    (day,) = fresh_days()

    results = reserve_concurrently(redis_url, day, cap=5, total=40)

    assert results.count(True) == 5
    assert counter(admin, day) == 40


def test_concurrent_reservations_at_cap_minus_one_allow_exactly_one(redis_url, admin, fresh_days):
    (day,) = fresh_days()
    admin.set(budget_key(day), 4)

    results = reserve_concurrently(redis_url, day, cap=5, total=20)

    assert results.count(True) == 1
    assert counter(admin, day) == 24


# --- TTL and expiry -------------------------------------------------------------------------


def test_reservation_sets_the_two_day_ttl(redis_url, admin, fresh_days):
    (day,) = fresh_days()

    async def scenario(budget):
        return await budget.reserve()

    assert run_with_budget(redis_url, 3, noon(day), scenario) is True
    assert_two_day_ttl(admin, day)


def test_reservation_adds_a_ttl_to_an_existing_key_without_one(redis_url, admin, fresh_days):
    (day,) = fresh_days()
    admin.set(budget_key(day), 1)
    assert admin.ttl(budget_key(day)) == -1  # Exists, no expiry

    async def scenario(budget):
        return await budget.reserve()

    assert run_with_budget(redis_url, 3, noon(day), scenario) is True
    assert counter(admin, day) == 2
    assert_two_day_ttl(admin, day)


def test_expired_counter_resets_the_budget(redis_url, admin, fresh_days):
    (day,) = fresh_days()
    key = budget_key(day)

    async def fill(budget):
        return [await budget.reserve(), await budget.reserve()], await budget.available()

    assert run_with_budget(redis_url, 2, noon(day), fill) == ([True, True], False)

    # Shorten the real expiry instead of waiting two days; poll briefly for Redis to remove the key.
    assert admin.pexpire(key, 100)
    deadline = time.monotonic() + 2.0
    while admin.exists(key) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not admin.exists(key)

    async def after_expiry(budget):
        return await budget.available(), await budget.reserve()

    assert run_with_budget(redis_url, 2, noon(day), after_expiry) == (True, True)
    assert counter(admin, day) == 1
    assert_two_day_ttl(admin, day)


# --- UTC rollover ---------------------------------------------------------------------------


def test_utc_midnight_starts_a_new_key_and_leaves_the_previous_day_unchanged(redis_url, admin, fresh_days):
    previous, today = fresh_days(2)
    clock = MutableNow(datetime.combine(previous, clock_time(23, 59, 59), tzinfo=timezone.utc))

    async def scenario(budget):
        before = [await budget.reserve(), await budget.reserve()]
        clock.value = datetime.combine(today, clock_time(0, 0, 0), tzinfo=timezone.utc)
        return before, await budget.available(), await budget.reserve()

    before, available, reserved = run_with_budget(redis_url, 1, clock, scenario)

    assert before == [True, False]
    assert (available, reserved) == (True, True)
    assert counter(admin, previous) == 2
    assert counter(admin, today) == 1


# --- Failures -------------------------------------------------------------------------------


def closed_local_port() -> int:
    """A loopback port nothing listens on (bound, then released)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_closed_local_port_fails_closed_without_logging_the_url(caplog):
    url = f"redis://default:super-secret-password@127.0.0.1:{closed_local_port()}/0"
    budget = create_strong_budget(Settings(redis_url=url, strong_daily_cap=5))
    assert isinstance(budget, RedisStrongBudget)

    async def scenario():
        try:
            return await budget.available(), await budget.reserve()
        finally:
            await budget.aclose()

    with caplog.at_level(logging.WARNING, logger="app.infra.budget"):
        results = asyncio.run(scenario())

    assert results == (False, False)
    assert [record.operation for record in caplog.records] == ["available", "reserve"]
    # Linux refuses a closed loopback port at once (ConnectionError); Windows retries the connect until
    # the 2 s connect timeout fires (TimeoutError). Either way the budget must fail closed.
    assert {record.error_type for record in caplog.records} <= {"ConnectionError", "TimeoutError"}
    assert "super-secret-password" not in caplog.text
    assert "127.0.0.1" not in caplog.text


def test_corrupt_counter_fails_closed_in_available_and_reserve(redis_url, admin, fresh_days, caplog):
    (day,) = fresh_days()
    admin.set(budget_key(day), b"not-a-number")

    async def scenario(budget):
        return await budget.available(), await budget.reserve()

    with caplog.at_level(logging.WARNING, logger="app.infra.budget"):
        results = run_with_budget(redis_url, 5, noon(day), scenario)

    assert results == (False, False)
    assert admin.get(budget_key(day)) == b"not-a-number"  # Never repaired or overwritten
    assert [(record.operation, record.error_type) for record in caplog.records] == [
        ("available", "ValueError"),
        ("reserve", "ResponseError"),
    ]
    assert "not-a-number" not in caplog.text
