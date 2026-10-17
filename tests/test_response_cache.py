"""Response cache reads and writes (Task S2.14, step 2) against fakeredis; no network access."""

import asyncio
import copy
import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import fakeredis
import pytest
import redis
from pydantic import SecretStr

from app.core.jev_client import JevTimeoutError
from app.core.llm_client import LLMClientError, ModelResult
from app.core.model_config import load_model_config
from app.core.router import RouterDeps, route
from app.core.settings import Settings
from app.infra.cache import (
    CACHE_FORMAT_VERSION,
    CACHE_TIMEOUT_SECONDS,
    CacheConfig,
    CachedAnswer,
    ResponseCache,
    answer_from_outcome,
    cache_key,
    classifier_version,
    create_response_cache,
)
from tests.fakes import FakeBudget, FakeClock, FakeJev, FakeLLM, RecordingSleep

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG = load_model_config()
CHEAP = CONFIG.routing.get_chain("general", "cheap")
STRONG = CONFIG.routing.get_chain("general", "strong")
EASY_JEV = json.loads((FIXTURES / "jev_response_success.json").read_text(encoding="utf-8"))
HARD_JEV = copy.deepcopy(EASY_JEV)
HARD_JEV["data"]["answers"]["difficulty"]["score"] = 3.0
HARD_JEV["data"]["answers"]["difficulty"]["probabilities"] = {"0": 0.0, "1": 0.0, "2": 0.0, "3": 1.0, "4": 0.0}

PROMPT = "Write a function to check primes PROMPT-MARKER-7f3a"
ANSWER = "ANSWER-MARKER-9c1e"
SECRET = "fake-cache-key-secret-for-tests-0123456789"
SECRET_URL = "redis://user:super-secret-password@cache.example.test:6379/0"
NOW = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
CLASSIFIER = classifier_version()


def ok(model: str, text: str = ANSWER) -> ModelResult:
    return ModelResult(
        text=text,
        model=model,
        input_tokens=13,
        output_tokens=115,
        latency_ms=900,
        finish_reason="stop",
        http_status=200,
    )


def client_error() -> LLMClientError:
    return LLMClientError("m: HTTP 400", status_code=400)


def run_route(jev=EASY_JEV, script=(), *, budget=None, dry_run=False):
    clock = FakeClock()
    deps = RouterDeps(
        config=CONFIG,
        jev=FakeJev(jev, clock),
        budget=budget or FakeBudget(),
        llm=FakeLLM(list(script), clock),
        request_budget_seconds=50.0,
        clock=clock,
        sleep=RecordingSleep(clock),
        new_request_id=lambda: str(uuid.uuid4()),
    )
    return asyncio.run(route(PROMPT, dry_run=dry_run, deps=deps))


OUTCOME = run_route(EASY_JEV, [ok(CHEAP[0])])  # A cacheable success; built here, outside any event loop


def make_config(namespace="test", ttl_seconds=86400):
    return CacheConfig(
        redis_url="redis://localhost:6379/0", namespace=namespace, key_secret=SecretStr(SECRET), ttl_seconds=ttl_seconds
    )


def make_cache(server=None, *, client=None, namespace="test", ttl_seconds=86400):
    client = client or fakeredis.FakeAsyncRedis(server=server or fakeredis.FakeServer())
    return ResponseCache(client, make_config(namespace, ttl_seconds), now=lambda: NOW), client


def key_for(prompt=PROMPT, namespace="test", config_version=CONFIG.config_version):
    return cache_key(make_config(namespace), prompt, config_version=config_version, classifier_version=CLASSIFIER)


def cache_records(caplog):
    return [(r.levelno, r.operation, r.reason, r.error_type) for r in caplog.records if r.name == "app.infra.cache"]


class FailingRedis:
    """Async Redis stand-in whose GET and SET raise `error`."""

    def __init__(self, error: Exception):
        self.error = error

    async def get(self, key):
        raise self.error

    async def set(self, key, value, ex=None):
        raise self.error


# --- Which outcomes are cacheable -----------------------------------------------------------


def test_cheap_success_becomes_an_answer_with_its_decision_and_response():
    outcome = run_route(EASY_JEV, [ok(CHEAP[0])])

    answer = answer_from_outcome(outcome, classifier_version=CLASSIFIER, now=NOW)

    assert answer is not None
    assert (answer.v, answer.created_at, answer.source_request_id) == (
        CACHE_FORMAT_VERSION,
        NOW,
        uuid.UUID(outcome.request_id),
    )
    assert (answer.config_version, answer.classifier_version) == (CONFIG.config_version, CLASSIFIER)
    assert (answer.response.model, answer.response.tier, answer.response.text) == (CHEAP[0], "cheap", ANSWER)
    decision = answer.decision
    assert (decision.task_type, decision.tier, decision.reason) == (
        outcome.decision.task_type,
        "cheap",
        outcome.decision.reason,
    )
    assert decision.model_chain == list(outcome.decision.model_chain)
    assert decision.confidence == outcome.decision.confidence
    assert decision.task_type_probs == dict(outcome.jev.task_type_probs)
    assert decision.difficulty_dist == dict(outcome.jev.difficulty_dist)


def test_strong_success_without_fallback_is_cacheable():
    answer = answer_from_outcome(run_route(HARD_JEV, [ok(STRONG[0])]), classifier_version=CLASSIFIER, now=NOW)

    assert (answer.decision.tier, answer.decision.reason, answer.response.tier) == ("strong", "HARD", "strong")


def test_jev_error_success_is_cacheable_with_null_jev_fields():
    error = JevTimeoutError("timed out", cause_type="ReadTimeout")

    answer = answer_from_outcome(run_route(error, [ok(STRONG[0])]), classifier_version=CLASSIFIER, now=NOW)

    assert answer.decision.reason == "JEV_ERROR"
    assert (answer.decision.task_type_probs, answer.decision.difficulty_dist, answer.decision.confidence) == (
        None,
        None,
        None,
    )


@pytest.mark.parametrize(
    "case",
    [
        dict(jev=EASY_JEV, script=[], dry_run=True),
        dict(jev=EASY_JEV, script=[client_error()] * len(CHEAP)),
        dict(jev=HARD_JEV, script=[client_error()] * len(STRONG) + [ok(CHEAP[0])]),
        dict(jev=HARD_JEV, script=[ok(CHEAP[0])], budget=FakeBudget(available=False)),
    ],
    ids=["dry-run", "failed", "fell-back-to-cheap", "budget-downgrade"],
)
def test_dry_runs_failures_fallbacks_and_budget_downgrades_are_never_cached(case):
    outcome = run_route(**case)

    assert answer_from_outcome(outcome, classifier_version=CLASSIFIER, now=NOW) is None
    cache, client = make_cache()
    assert asyncio.run(cache.store(PROMPT, outcome)) is False
    assert asyncio.run(client.keys("*")) == []


# --- Round trip and TTL ---------------------------------------------------------------------


def test_store_then_lookup_round_trips_the_answer():
    outcome = run_route(EASY_JEV, [ok(CHEAP[0])])
    cache, _ = make_cache()

    async def scenario():
        assert await cache.store(PROMPT, outcome) is True
        return await cache.lookup(PROMPT, config_version=CONFIG.config_version)

    found = asyncio.run(scenario())

    assert found == answer_from_outcome(outcome, classifier_version=CLASSIFIER, now=NOW)


def test_entry_is_written_once_under_the_cache_key_with_the_configured_ttl():
    cache, client = make_cache(ttl_seconds=600)

    async def scenario():
        await cache.store(PROMPT, OUTCOME)
        return await client.keys("*"), await client.ttl(key_for())

    keys, ttl = asyncio.run(scenario())

    assert keys == [key_for().encode()]
    assert 0 < ttl <= 600


def test_lookup_matches_a_prompt_differing_only_in_outer_whitespace():
    cache, _ = make_cache()

    async def scenario():
        await cache.store(PROMPT, OUTCOME)
        return await cache.lookup(f"  {PROMPT}\n", config_version=CONFIG.config_version)

    assert asyncio.run(scenario()) is not None


def test_expired_entry_is_a_miss():
    cache, client = make_cache()

    async def scenario():
        await cache.store(PROMPT, OUTCOME)
        await client.pexpire(key_for(), 1)
        await asyncio.sleep(0.05)
        return await cache.lookup(PROMPT, config_version=CONFIG.config_version)

    assert asyncio.run(scenario()) is None


def test_plain_miss_returns_none_without_logging(caplog):
    cache, _ = make_cache()

    with caplog.at_level(logging.DEBUG, logger="app.infra.cache"):
        assert asyncio.run(cache.lookup(PROMPT, config_version=CONFIG.config_version)) is None

    assert cache_records(caplog) == []


def test_other_namespace_config_version_or_prompt_is_a_miss():
    server = fakeredis.FakeServer()
    production, _ = make_cache(server, namespace="production")
    preview, _ = make_cache(server, namespace="preview")

    async def scenario():
        await production.store(PROMPT, OUTCOME)
        return (
            await preview.lookup(PROMPT, config_version=CONFIG.config_version),
            await production.lookup(PROMPT, config_version="f" * 64),
            await production.lookup(PROMPT.upper(), config_version=CONFIG.config_version),
        )

    assert asyncio.run(scenario()) == (None, None, None)


def test_stored_value_holds_neither_the_prompt_nor_its_hash_nor_the_secret():
    cache, client = make_cache()

    async def scenario():
        await cache.store(PROMPT, OUTCOME)
        return (await client.get(key_for())).decode()

    stored = asyncio.run(scenario())

    assert "PROMPT-MARKER" not in stored
    assert hashlib.sha256(PROMPT.encode()).hexdigest() not in stored
    assert SECRET not in stored
    assert ANSWER in stored


def test_concurrent_identical_stores_leave_one_entry_and_the_last_write_wins():
    first = run_route(EASY_JEV, [ok(CHEAP[0], text="FIRST")])
    second = run_route(EASY_JEV, [ok(CHEAP[0], text="SECOND")])
    cache, client = make_cache()

    async def scenario():
        results = await asyncio.gather(cache.store(PROMPT, first), cache.store(PROMPT, second))
        return results, await client.keys("*"), await cache.lookup(PROMPT, config_version=CONFIG.config_version)

    results, keys, found = asyncio.run(scenario())

    assert results == [True, True]
    assert len(keys) == 1
    assert found.response.text == "SECOND"


# --- Malformed or incompatible entries --------------------------------------------------------


def valid_entry() -> dict:
    answer = answer_from_outcome(run_route(EASY_JEV, [ok(CHEAP[0])]), classifier_version=CLASSIFIER, now=NOW)
    return json.loads(answer.model_dump_json())


def with_changes(**changes) -> bytes:
    entry = valid_entry()
    entry.update(changes)
    return json.dumps(entry).encode()


def without(field: str) -> bytes:
    entry = valid_entry()
    del entry[field]
    return json.dumps(entry).encode()


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (b"not json", "malformed_entry"),
        (b"\xff\xfe", "malformed_entry"),
        (b"[]", "malformed_entry"),
        (without("response"), "malformed_entry"),
        (with_changes(response={"model": CHEAP[0], "tier": "cheap", "text": ""}), "malformed_entry"),
        (with_changes(unexpected="field"), "malformed_entry"),
        (with_changes(v=CACHE_FORMAT_VERSION + 1), "malformed_entry"),
        (with_changes(config_version="f" * 64), "version_mismatch"),
        (with_changes(classifier_version="f" * 64), "version_mismatch"),
    ],
    ids=[
        "not-json",
        "not-utf8",
        "not-an-object",
        "missing-field",
        "empty-text",
        "extra-field",
        "other-format-version",
        "other-config-version",
        "other-classifier-version",
    ],
)
def test_malformed_or_incompatible_entry_is_a_miss_with_one_warning(caplog, raw, reason):
    cache, client = make_cache()

    async def scenario():
        await client.set(key_for(), raw)
        return await cache.lookup(PROMPT, config_version=CONFIG.config_version)

    with caplog.at_level(logging.DEBUG, logger="app.infra.cache"):
        assert asyncio.run(scenario()) is None

    records = cache_records(caplog)
    assert len(records) == 1
    assert records[0][:3] == (logging.WARNING, "lookup", reason)
    assert "PROMPT-MARKER" not in caplog.text
    assert ANSWER not in caplog.text


def test_a_new_store_overwrites_a_malformed_entry():
    cache, client = make_cache()

    async def scenario():
        await client.set(key_for(), b"not json")
        await cache.store(PROMPT, OUTCOME)
        return await cache.lookup(PROMPT, config_version=CONFIG.config_version)

    assert asyncio.run(scenario()) is not None


# --- Redis failures fail open -----------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        redis.exceptions.ConnectionError(f"Error connecting to {SECRET_URL}"),
        redis.exceptions.TimeoutError(f"Timeout reading from {SECRET_URL}"),
        OSError(f"socket failure {SECRET_URL}"),
    ],
    ids=["connection", "timeout", "socket"],
)
def test_redis_failure_is_a_miss_and_a_skipped_write_with_one_sanitized_warning_each(caplog, error):
    cache, _ = make_cache(client=FailingRedis(error))
    outcome = run_route(EASY_JEV, [ok(CHEAP[0])])

    async def scenario():
        return await cache.lookup(PROMPT, config_version=CONFIG.config_version), await cache.store(PROMPT, outcome)

    with caplog.at_level(logging.DEBUG, logger="app.infra.cache"):
        assert asyncio.run(scenario()) == (None, False)

    assert cache_records(caplog) == [
        (logging.WARNING, "lookup", "redis_error", type(error).__name__),
        (logging.WARNING, "store", "redis_error", type(error).__name__),
    ]
    assert "super-secret-password" not in caplog.text
    assert "PROMPT-MARKER" not in caplog.text


def test_disconnected_fake_server_fails_open():
    server = fakeredis.FakeServer()
    cache, _ = make_cache(server)
    server.connected = False

    async def scenario():
        return (
            await cache.lookup(PROMPT, config_version=CONFIG.config_version),
            await cache.store(PROMPT, OUTCOME),
        )

    assert asyncio.run(scenario()) == (None, False)


@pytest.mark.parametrize("operation", ["lookup", "store"])
def test_unexpected_errors_are_not_swallowed(operation):
    cache, _ = make_cache(client=FailingRedis(RuntimeError("bug")))

    async def scenario():
        if operation == "lookup":
            await cache.lookup(PROMPT, config_version=CONFIG.config_version)
        else:
            await cache.store(PROMPT, OUTCOME)

    with pytest.raises(RuntimeError, match="bug"):
        asyncio.run(scenario())


def test_naive_clock_is_rejected():
    cache = ResponseCache(fakeredis.FakeAsyncRedis(), make_config(), now=lambda: datetime(2026, 10, 17, 12, 0))

    with pytest.raises(ValueError):
        asyncio.run(cache.store(PROMPT, run_route(EASY_JEV, [ok(CHEAP[0])])))


def test_answer_model_rejects_a_naive_created_at():
    entry = valid_entry()
    entry["created_at"] = "2026-10-17T12:00:00"

    with pytest.raises(ValueError):
        CachedAnswer.model_validate(entry)


# --- Factory ------------------------------------------------------------------------------------


def configured(**overrides):
    values = {"redis_url": "redis://localhost:6399/0", "cache_namespace": "test", "cache_key_secret": SECRET}
    values.update(overrides)
    return Settings(**values)


@pytest.mark.parametrize("missing", ["redis_url", "cache_namespace", "cache_key_secret"])
def test_factory_returns_none_without_building_a_client_when_unconfigured(monkeypatch, caplog, missing):
    def no_client(*args, **kwargs):
        raise AssertionError("no Redis client when the cache is not configured")

    monkeypatch.setattr(redis.asyncio.Redis, "from_url", no_client)

    with caplog.at_level(logging.DEBUG, logger="app.infra.cache"):
        assert create_response_cache(configured(**{missing: None})) is None

    assert cache_records(caplog) == []


def test_factory_builds_a_client_with_the_short_cache_timeouts(monkeypatch):
    calls = []
    real_from_url = redis.asyncio.Redis.from_url

    def spy(url, **kwargs):
        calls.append(kwargs)
        return real_from_url(url, **kwargs)

    monkeypatch.setattr(redis.asyncio.Redis, "from_url", spy)

    cache = create_response_cache(configured(cache_ttl_seconds=60))

    assert isinstance(cache, ResponseCache)
    assert calls == [{"socket_connect_timeout": CACHE_TIMEOUT_SECONDS, "socket_timeout": CACHE_TIMEOUT_SECONDS}]
    assert CACHE_TIMEOUT_SECONDS == 0.3
    asyncio.run(cache.aclose())


def test_factory_with_an_invalid_url_returns_none_and_logs_only_the_error_type(caplog):
    url = "http://user:super-secret-password@not-redis.example.test"

    with caplog.at_level(logging.DEBUG, logger="app.infra.cache"):
        assert create_response_cache(configured(redis_url=url)) is None

    assert cache_records(caplog) == [(logging.WARNING, "create", "invalid_redis_url", "ValueError")]
    assert "super-secret-password" not in caplog.text


def test_default_clock_stamps_entries_in_utc():
    cache = ResponseCache(fakeredis.FakeAsyncRedis(), make_config())
    before = datetime.now(timezone.utc)

    async def scenario():
        await cache.store(PROMPT, OUTCOME)
        return await cache.lookup(PROMPT, config_version=CONFIG.config_version)

    found = asyncio.run(scenario())

    assert found.created_at.utcoffset().total_seconds() == 0
    assert before <= found.created_at <= datetime.now(timezone.utc)
