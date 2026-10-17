import asyncio
import logging
from datetime import date, datetime, timedelta, timezone

import fakeredis
import pytest
import redis

from app.core.settings import Settings
from app.infra.budget import (
    BUDGET_TTL_SECONDS,
    RedisStrongBudget,
    UnavailableStrongBudget,
    budget_key,
    create_strong_budget,
)

DAY = datetime(2026, 10, 12, 15, 0, tzinfo=timezone.utc)
SECRET_URL = "rediss://default:super-secret-password@redis.example.test:6379"


class MutableNow:
    def __init__(self, value: datetime):
        self.value = value

    def __call__(self) -> datetime:
        return self.value


def make_budget(cap: int, server=None, now=None) -> tuple[RedisStrongBudget, fakeredis.FakeAsyncRedis]:
    client = fakeredis.FakeAsyncRedis(server=server or fakeredis.FakeServer())
    return RedisStrongBudget(client, daily_cap=cap, now=now or (lambda: DAY)), client


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
    """Client whose every command raises, like redis-py when the server is unreachable."""

    def __init__(self, error: Exception):
        self.error = error

    async def get(self, key):
        raise self.error

    def pipeline(self, transaction=True):
        return FailingPipeline(self.error)


# --- Key and TTL ----------------------------------------------------------------------------


def test_key_is_budget_strong_with_iso_date():
    assert budget_key(date(2026, 10, 12)) == "budget:strong:2026-10-12"


def test_reserve_sets_a_two_day_ttl():
    async def scenario():
        budget, client = make_budget(cap=3)
        assert await budget.reserve() is True
        return await client.ttl(budget_key(DAY.date()))

    assert BUDGET_TTL_SECONDS == 2 * 24 * 60 * 60
    assert asyncio.run(scenario()) == BUDGET_TTL_SECONDS


# --- Cap enforcement ------------------------------------------------------------------------


def test_reserve_allows_up_to_the_cap_then_denies():
    async def scenario():
        budget, _ = make_budget(cap=3)
        availability = [await budget.available()]
        reservations = []
        for _ in range(5):
            reservations.append(await budget.reserve())
            availability.append(await budget.available())
        return reservations, availability

    reservations, availability = asyncio.run(scenario())

    assert reservations == [True, True, True, False, False]
    assert availability == [True, True, True, False, False, False]


def test_concurrent_reservations_never_exceed_the_cap():
    async def scenario():
        server = fakeredis.FakeServer()
        first, _ = make_budget(cap=5, server=server)
        second, _ = make_budget(cap=5, server=server)  # e.g. another serverless instance
        results = await asyncio.gather(*(budget.reserve() for budget in [first, second] * 12 + [first]))
        return results

    results = asyncio.run(scenario())

    assert len(results) == 25
    assert results.count(True) == 5


def test_zero_cap_is_never_available_and_does_not_touch_redis():
    async def scenario():
        budget, client = make_budget(cap=0)
        return await budget.available(), await budget.reserve(), await client.exists(budget_key(DAY.date()))

    assert asyncio.run(scenario()) == (False, False, 0)


def test_negative_or_non_integer_cap_is_rejected():
    client = fakeredis.FakeAsyncRedis()
    for cap in (-1, 1.5, True):
        with pytest.raises(ValueError, match="daily_cap"):
            RedisStrongBudget(client, daily_cap=cap)


# --- UTC date -------------------------------------------------------------------------------


def test_utc_date_rollover_starts_a_fresh_count():
    async def scenario():
        now = MutableNow(DAY)
        budget, client = make_budget(cap=1, now=now)
        first_day = [await budget.reserve(), await budget.reserve(), await budget.available()]
        now.value = DAY + timedelta(days=1)
        next_day = [await budget.available(), await budget.reserve()]
        keys = sorted(key.decode() for key in await client.keys("budget:strong:*"))
        return first_day, next_day, keys

    first_day, next_day, keys = asyncio.run(scenario())

    assert first_day == [True, False, False]
    assert next_day == [True, True]
    assert keys == ["budget:strong:2026-10-12", "budget:strong:2026-10-13"]


def test_non_utc_clock_is_converted_to_the_utc_date():
    # 23:30 at UTC-5 on 12 Oct is 04:30 UTC on 13 Oct.
    local = datetime(2026, 10, 12, 23, 30, tzinfo=timezone(timedelta(hours=-5)))

    async def scenario():
        budget, client = make_budget(cap=3, now=lambda: local)
        await budget.reserve()
        return await client.get("budget:strong:2026-10-13")

    assert asyncio.run(scenario()) == b"1"


def test_naive_clock_is_rejected():
    budget, _ = make_budget(cap=3, now=lambda: datetime(2026, 10, 12, 15, 0))

    with pytest.raises(ValueError, match="timezone-aware"):
        asyncio.run(budget.reserve())


def test_default_clock_uses_current_utc_date():
    async def scenario():
        client = fakeredis.FakeAsyncRedis()
        budget = RedisStrongBudget(client, daily_cap=3)
        await budget.reserve()
        return await client.keys("budget:strong:*")

    keys = asyncio.run(scenario())

    assert keys == [budget_key(datetime.now(timezone.utc).date()).encode()]


# --- Redis failures -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        redis.exceptions.ConnectionError(f"Error connecting to {SECRET_URL}"),
        redis.exceptions.TimeoutError(f"Timeout reading from {SECRET_URL}"),
        ConnectionResetError(f"reset by {SECRET_URL}"),
    ],
    ids=["connection", "timeout", "socket"],
)
def test_redis_failure_fails_closed_and_logs_only_the_error_type(caplog, error):
    budget = RedisStrongBudget(FailingRedis(error), daily_cap=100, now=lambda: DAY)

    with caplog.at_level(logging.WARNING, logger="app.infra.budget"):
        results = asyncio.run(budget.available()), asyncio.run(budget.reserve())

    assert results == (False, False)
    assert len(caplog.records) == 2
    for record in caplog.records:
        assert type(error).__name__ in record.getMessage()
        assert "super-secret-password" not in record.getMessage()
        assert "redis.example.test" not in record.getMessage()


def test_disconnected_fake_server_fails_closed():
    async def scenario():
        server = fakeredis.FakeServer()
        budget, _ = make_budget(cap=100, server=server)
        server.connected = False
        return await budget.available(), await budget.reserve()

    assert asyncio.run(scenario()) == (False, False)


def test_unexpected_error_is_not_swallowed():
    budget = RedisStrongBudget(FailingRedis(RuntimeError("bug")), daily_cap=100, now=lambda: DAY)

    with pytest.raises(RuntimeError, match="bug"):
        asyncio.run(budget.reserve())


# --- Corrupt counter (S2.2): fail closed in available() like reserve() ----------------------


@pytest.mark.parametrize("value", [b"not-a-number", b"3.5", b""], ids=["text", "float", "empty"])
def test_corrupt_counter_fails_closed_and_logs_only_the_error_type(caplog, value):
    async def scenario():
        budget, client = make_budget(cap=5)
        await client.set(budget_key(DAY.date()), value)
        return await budget.available(), await budget.reserve(), await client.get(budget_key(DAY.date()))

    with caplog.at_level(logging.WARNING, logger="app.infra.budget"):
        available, reserved, stored = asyncio.run(scenario())

    assert (available, reserved) == (False, False)
    assert stored == value  # never repaired or overwritten
    assert [(record.operation, record.error_type) for record in caplog.records] == [
        ("available", "ValueError"),
        ("reserve", "ResponseError"),
    ]
    if value:
        assert value.decode() not in caplog.text


def test_naive_clock_is_still_rejected_by_available():
    budget, _ = make_budget(cap=3, now=lambda: datetime(2026, 10, 12, 15, 0))

    with pytest.raises(ValueError, match="timezone-aware"):
        asyncio.run(budget.available())


def test_missing_counter_is_still_available():
    budget, _ = make_budget(cap=1)

    assert asyncio.run(budget.available()) is True


# --- Factory --------------------------------------------------------------------------------


def test_unconfigured_redis_gives_an_unavailable_budget(caplog):
    with caplog.at_level(logging.WARNING, logger="app.infra.budget"):
        budget = create_strong_budget(Settings(redis_url=None, strong_daily_cap=100))

    assert isinstance(budget, UnavailableStrongBudget)
    assert (asyncio.run(budget.available()), asyncio.run(budget.reserve())) == (False, False)
    assert "redis_not_configured" in caplog.text


def test_invalid_redis_url_gives_an_unavailable_budget_without_leaking_it(caplog):
    url = "http://user:super-secret-password@not-redis.example.test"

    with caplog.at_level(logging.WARNING, logger="app.infra.budget"):
        budget = create_strong_budget(Settings(redis_url=url, strong_daily_cap=100))

    assert isinstance(budget, UnavailableStrongBudget)
    assert "ValueError" in caplog.text
    assert "super-secret-password" not in caplog.text
    assert "not-redis.example.test" not in caplog.text


def test_configured_redis_uses_the_settings_cap_without_connecting():
    budget = create_strong_budget(Settings(redis_url="redis://localhost:6399/0", strong_daily_cap=7))

    assert isinstance(budget, RedisStrongBudget)
    assert budget.daily_cap == 7
    asyncio.run(budget.aclose())


def test_factory_default_cap_is_the_settings_default():
    budget = create_strong_budget(Settings(redis_url="redis://localhost:6399/0"))

    assert budget.daily_cap == Settings.model_fields["strong_daily_cap"].default
    asyncio.run(budget.aclose())
