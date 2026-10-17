"""Response cache against a real, local Redis (Task S2.14, step 5). Skipped by the default `pytest` run.

Runs in CI against the Redis service container. Locally, run it only against a disposable Redis on
this machine, for example:

    REDIS_URL=redis://127.0.0.1:6379/0 python -m pytest -m integration -rA tests/test_cache_integration.py

Isolation: require_local_redis_url (shared with the budget and rate-limit tests) refuses anything but
redis:// on a loopback host. Each test uses a random namespace and a random key secret, so its keys can
never match another environment's; it fails rather than touch a key that already exists, and deletes
only the keys it created. No FLUSHDB, FLUSHALL or key scans.
"""

import asyncio
import logging
import os
import time
import uuid

import pytest
import redis
import redis.asyncio
from pydantic import SecretStr

from app.core.settings import Settings
from app.infra.cache import (
    CACHE_TIMEOUT_SECONDS,
    CacheConfig,
    ResponseCache,
    answer_from_outcome,
    cache_key,
    classifier_version,
    create_response_cache,
)
from app.infra.redis import TIMEOUT_SECONDS
from app.routes import route as route_api
from tests.test_budget_integration import closed_local_port, require_local_redis_url
from tests.test_response_cache import CHEAP, CONFIG, EASY_JEV, OUTCOME, PROMPT, ok, run_route
from tests.test_route_api import ApiHarness

pytestmark = pytest.mark.integration

TTL_SECONDS = 600
TTL_TOLERANCE_SECONDS = 10
CLASSIFIER = classifier_version()


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
def space(redis_url, admin):
    """A random namespace and secret; deletes exactly the keys registered through `key()`."""
    config = CacheConfig(
        redis_url=redis_url,
        namespace=f"it-{uuid.uuid4().hex[:24]}",
        key_secret=SecretStr(f"integration-{uuid.uuid4().hex}-{uuid.uuid4().hex}"),
        ttl_seconds=TTL_SECONDS,
    )
    created: list[str] = []

    class Space:
        def __init__(self):
            self.config = config

        def key(self, prompt: str = PROMPT) -> str:
            name = cache_key(config, prompt, config_version=CONFIG.config_version, classifier_version=CLASSIFIER)
            if name not in created:
                if admin.exists(name):
                    pytest.fail("A test cache key already exists; refusing to reuse or delete it")
                created.append(name)
            return name

        def cache(self) -> ResponseCache:
            client = redis.asyncio.Redis.from_url(
                redis_url, socket_connect_timeout=CACHE_TIMEOUT_SECONDS, socket_timeout=CACHE_TIMEOUT_SECONDS
            )
            return ResponseCache(client, config)

    yield Space()
    if created:
        admin.delete(*created)


async def using(cache: ResponseCache, operation):
    try:
        return await operation(cache)
    finally:
        await cache.aclose()


def test_store_then_lookup_round_trips_on_real_redis_with_the_configured_ttl(admin, space):
    key = space.key()

    async def store_and_lookup(cache):
        stored = await cache.store(PROMPT, OUTCOME)
        return stored, await cache.lookup(PROMPT, config_version=CONFIG.config_version)

    stored, found = asyncio.run(using(space.cache(), store_and_lookup))

    assert stored is True
    expected = answer_from_outcome(OUTCOME, classifier_version=CLASSIFIER, now=found.created_at)
    assert found == expected
    assert TTL_SECONDS - TTL_TOLERANCE_SECONDS <= admin.ttl(key) <= TTL_SECONDS


def test_expired_entry_is_a_miss_on_real_redis(admin, space):
    key = space.key()
    asyncio.run(using(space.cache(), lambda cache: cache.store(PROMPT, OUTCOME)))
    assert admin.exists(key)

    admin.pexpire(key, 50)
    time.sleep(0.3)

    assert asyncio.run(using(space.cache(), lambda c: c.lookup(PROMPT, config_version=CONFIG.config_version))) is None
    assert not admin.exists(key)


def test_concurrent_identical_stores_across_independent_clients_leave_one_entry(admin, space):
    key = space.key()
    outcomes = [run_route(EASY_JEV, [ok(CHEAP[0], text=f"ANSWER-{n}")]) for n in range(4)]

    async def store_all():
        caches = [space.cache() for _ in outcomes]  # Separate connections, like serverless instances
        try:
            return await asyncio.gather(*(cache.store(PROMPT, outcome) for cache, outcome in zip(caches, outcomes)))
        finally:
            await asyncio.gather(*(cache.aclose() for cache in caches))

    results = asyncio.run(store_all())
    found = asyncio.run(using(space.cache(), lambda c: c.lookup(PROMPT, config_version=CONFIG.config_version)))

    assert results == [True] * len(outcomes)
    assert admin.exists(key) == 1
    assert found.response.text in {f"ANSWER-{n}" for n in range(len(outcomes))}


def test_closed_local_port_fails_open_quickly_with_sanitized_warnings(caplog):
    url = f"redis://default:super-secret-password@127.0.0.1:{closed_local_port()}/0"
    secret = "integration-secret-0123456789-0123456789"
    cache = create_response_cache(Settings(redis_url=url, cache_namespace="it-closed", cache_key_secret=secret))
    assert isinstance(cache, ResponseCache)

    async def lookup_and_store(cache):
        return await cache.lookup(PROMPT, config_version=CONFIG.config_version), await cache.store(PROMPT, OUTCOME)

    started = time.monotonic()
    with caplog.at_level(logging.DEBUG, logger="app.infra.cache"):
        result = asyncio.run(using(cache, lookup_and_store))
    elapsed = time.monotonic() - started

    assert result == (None, False)
    # Linux refuses a closed loopback port at once (ConnectionError); Windows retries the connect until
    # the cache's 0.3 s connect timeout fires (TimeoutError). A generous bound: two operations, each
    # limited by CACHE_TIMEOUT_SECONDS plus redis-py's own connection retries.
    assert elapsed < 2 * TIMEOUT_SECONDS
    records = [r for r in caplog.records if r.name == "app.infra.cache"]
    assert [(r.operation, r.reason) for r in records] == [("lookup", "redis_error"), ("store", "redis_error")]
    assert {r.error_type for r in records} <= {"ConnectionError", "TimeoutError"}
    for leaked in ("super-secret-password", "127.0.0.1", secret, "PROMPT-MARKER"):
        assert leaked not in caplog.text


def test_route_serves_a_hit_from_real_redis_without_jev_budget_or_model(admin, space):
    key = space.key()

    async def no_persistence(*args):  # Redis only: CI sets DATABASE_URL, and this test must not write rows
        pass

    def harness(script) -> ApiHarness:
        api = ApiHarness(EASY_JEV, script, recorder=no_persistence)
        overrides = api.client.app.dependency_overrides
        overrides[route_api.get_cache_factory] = lambda: lambda settings: space.cache()
        overrides[route_api.get_hit_recorder] = lambda: no_persistence
        return api

    miss = harness([ok(CHEAP[0])]).post({"prompt": PROMPT, "use_cache": True})
    hitting = harness([])
    hit = hitting.post({"prompt": PROMPT, "use_cache": True})

    assert (miss.status_code, miss.json()["cache_hit"]) == (200, False)
    assert admin.exists(key) == 1
    assert (hit.status_code, hit.json()["cache_hit"]) == (200, True)
    assert hit.json()["response"]["text"] == miss.json()["response"]["text"]
    assert (hitting.jev.calls, hitting.llm.calls, hitting.factory_calls) == ([], [], [])
    assert (hitting.budget.available_calls, hitting.budget.reserve_calls) == (0, 0)
