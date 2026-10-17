"""POST /api/route with the response cache (Task S2.14, step 3). fakeredis and fakes only; no network access."""

import asyncio
import logging
import uuid

import fakeredis
import pytest
import redis
from pydantic import SecretStr

from app.infra.cache import CacheConfig, ResponseCache
from app.routes import route as route_api
from tests.fakes import FakeBudget, FakeRateLimiter
from tests.test_route_api import CHEAP, CONFIG, EASY_JEV, HARD_JEV, PROMPT, STRONG, ApiHarness, client_error, ok

SECRET = "fake-cache-key-secret-for-tests-0123456789"


def make_cache(server, *, client=None, events=None):
    config = CacheConfig(
        redis_url="redis://localhost:6379/0", namespace="test", key_secret=SecretStr(SECRET), ttl_seconds=86400
    )
    return SpyCache(client or fakeredis.FakeAsyncRedis(server=server), config, events=events)


class SpyCache(ResponseCache):
    """A real ResponseCache that records lookup, store and aclose calls."""

    def __init__(self, client, config, *, events=None):
        super().__init__(client, config)
        self.events = [] if events is None else events

    async def lookup(self, prompt, *, config_version):
        self.events.append("lookup")
        return await super().lookup(prompt, config_version=config_version)

    async def store(self, prompt, outcome):
        self.events.append("store")
        return await super().store(prompt, outcome)

    async def aclose(self):
        self.events.append("aclose")
        await super().aclose()


class FailingRedis:
    def __init__(self, error):
        self.error = error

    async def get(self, key):
        raise self.error

    async def set(self, key, value, ex=None):
        raise self.error

    async def aclose(self):
        pass


def with_cache(harness: ApiHarness, cache) -> list:
    """Overrides the cache factory; returns the list of factory calls."""
    calls = []

    def factory(settings):
        calls.append(settings)
        return cache

    harness.client.app.dependency_overrides[route_api.get_cache_factory] = lambda: factory
    return calls


def keys(server) -> list:
    return asyncio.run(fakeredis.FakeAsyncRedis(server=server).keys("*"))


def fill(server, jev=EASY_JEV, script=None) -> dict:
    """One cached answer from a real routed request; returns that response's body."""
    harness = ApiHarness(jev, script or [ok(CHEAP[0])])
    with_cache(harness, make_cache(server))
    response = harness.post({"prompt": PROMPT, "use_cache": True})
    assert response.status_code == 200
    assert response.json()["cache_hit"] is False
    return response.json()


# --- use_cache off (default) and dry runs ---------------------------------------------------------


@pytest.mark.parametrize("body", [{"prompt": PROMPT}, {"prompt": PROMPT, "use_cache": False}], ids=["omitted", "false"])
def test_without_use_cache_the_cache_is_never_built_and_cache_hit_is_false(body):
    server = fakeredis.FakeServer()
    harness = ApiHarness(EASY_JEV, [ok(CHEAP[0])])
    calls = with_cache(harness, make_cache(server))

    response = harness.post(body)

    assert response.status_code == 200
    assert response.json()["cache_hit"] is False
    assert calls == []
    assert keys(server) == []


def test_dry_run_never_reads_or_writes_the_cache():
    server = fakeredis.FakeServer()
    fill(server)
    harness = ApiHarness(EASY_JEV)
    calls = with_cache(harness, make_cache(server))

    response = harness.post({"prompt": PROMPT, "use_cache": True, "dry_run": True})

    assert response.status_code == 200
    assert (response.json()["dry_run"], response.json()["cache_hit"]) == (True, False)
    assert calls == []
    assert len(harness.jev.calls) == 1  # A dry run still classifies


# --- Miss then hit ------------------------------------------------------------------------------


def test_miss_routes_normally_and_writes_one_entry():
    server = fakeredis.FakeServer()

    body = fill(server)

    assert body["response"]["text"] == f"answer from {CHEAP[0]}"
    assert len(keys(server)) == 1


def test_hit_returns_the_cached_answer_without_jev_budget_model_or_dependencies():
    server = fakeredis.FakeServer()
    first = fill(server)
    budget = FakeBudget()
    harness = ApiHarness(EASY_JEV, [], budget=budget)
    with_cache(harness, make_cache(server))

    response = harness.post({"prompt": PROMPT, "use_cache": True})

    assert response.status_code == 200
    body = response.json()
    assert body["cache_hit"] is True
    assert (harness.jev.calls, harness.llm.calls, harness.factory_calls) == ([], [], [])
    assert (budget.available_calls, budget.reserve_calls) == (0, 0)
    assert response.headers["X-Request-ID"] == body["request_id"] != first["request_id"]
    assert uuid.UUID(body["request_id"]).version == 4
    assert (body["dry_run"], body["strong_reserved"], body["cost"]) == (False, False, None)
    assert body["config_version"] == CONFIG.config_version
    assert body["total_latency_ms"] >= 0
    answer = body["response"]
    assert (answer["model"], answer["tier"], answer["text"]) == (CHEAP[0], "cheap", first["response"]["text"])
    assert (answer["fell_back_to_cheap"], answer["attempts"]) == (False, [])
    assert answer["latency_ms"] >= 0


def test_hit_keeps_the_original_decision_and_nulls_this_requests_jev_fields():
    server = fakeredis.FakeServer()
    first = fill(server)
    harness = ApiHarness(EASY_JEV, [])
    with_cache(harness, make_cache(server))

    decision = harness.post({"prompt": PROMPT, "use_cache": True}).json()["decision"]

    original = first["decision"]
    for field in ("task_type", "tier", "reason", "model_chain", "task_type_probs", "difficulty", "confidence"):
        assert decision[field] == original[field], field
    assert decision["needs_reasoning_prob"] == original["needs_reasoning_prob"]
    assert decision["reason"] == "EASY"  # Not "CACHE" (decision U4)
    assert (decision["jev_latency_ms"], decision["jev_error"], decision["jev_http_status"]) == (None, None, None)
    assert decision["jev_error_cause"] is None


def test_strong_answer_hit_never_reserves_strong_budget():
    server = fakeredis.FakeServer()
    fill(server, HARD_JEV, [ok(STRONG[0])])
    budget = FakeBudget()
    harness = ApiHarness(HARD_JEV, [], budget=budget)
    with_cache(harness, make_cache(server))

    body = harness.post({"prompt": PROMPT, "use_cache": True}).json()

    assert (body["cache_hit"], body["decision"]["tier"], body["strong_reserved"]) == (True, "strong", False)
    assert (budget.available_calls, budget.reserve_calls) == (0, 0)


def test_hit_logs_one_line_with_the_request_id_and_no_prompt(caplog):
    server = fakeredis.FakeServer()
    fill(server)
    harness = ApiHarness(EASY_JEV, [])
    with_cache(harness, make_cache(server))

    with caplog.at_level(logging.INFO, logger="app.routes.route"):
        body = harness.post({"prompt": PROMPT, "use_cache": True}).json()

    hits = [r for r in caplog.records if r.name == "app.routes.route" and getattr(r, "status", None) == "cache_hit"]
    assert len(hits) == 1
    assert hits[0].request_id == body["request_id"]
    assert "PROMPT-MARKER" not in caplog.text


# --- What is written ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("jev", "script", "status", "budget"),
    [
        (EASY_JEV, [client_error()] * len(CHEAP), 502, None),
        (HARD_JEV, [client_error()] * len(STRONG) + [ok(CHEAP[0])], 200, None),
        (HARD_JEV, [ok(CHEAP[0])], 200, FakeBudget(available=False)),
    ],
    ids=["failed", "fell-back-to-cheap", "budget-downgrade"],
)
def test_uncacheable_outcomes_are_returned_but_not_written(jev, script, status, budget):
    server = fakeredis.FakeServer()
    harness = ApiHarness(jev, script, budget=budget)
    with_cache(harness, make_cache(server))

    response = harness.post({"prompt": PROMPT, "use_cache": True})

    assert response.status_code == status
    assert keys(server) == []


def test_cache_is_written_after_persistence_and_closed_after_the_request():
    server = fakeredis.FakeServer()
    events = []

    async def recorder(outcome, prompt):
        events.append("persist")

    harness = ApiHarness(EASY_JEV, [ok(CHEAP[0])], recorder=recorder)
    with_cache(harness, make_cache(server, events=events))

    harness.post({"prompt": PROMPT, "use_cache": True})

    assert events == ["lookup", "persist", "store", "aclose"]


def test_cache_is_closed_after_a_hit():
    server = fakeredis.FakeServer()
    fill(server)
    events = []
    harness = ApiHarness(EASY_JEV, [])
    with_cache(harness, make_cache(server, events=events))

    assert harness.post({"prompt": PROMPT, "use_cache": True}).json()["cache_hit"] is True
    assert events == ["lookup", "aclose"]


# --- Rate limit -----------------------------------------------------------------------------------


def test_cache_hits_count_toward_the_rate_limit():
    server = fakeredis.FakeServer()
    fill(server)
    limiter = FakeRateLimiter()
    harness = ApiHarness(EASY_JEV, [], limiter=limiter)
    with_cache(harness, make_cache(server))

    assert harness.post({"prompt": PROMPT, "use_cache": True}).json()["cache_hit"] is True
    assert len(limiter.calls) == 1


def test_rate_limited_request_never_builds_or_reads_the_cache():
    server = fakeredis.FakeServer()
    fill(server)
    harness = ApiHarness(EASY_JEV, [], limiter=FakeRateLimiter(allowed=False, retry_after_seconds=30))
    calls = with_cache(harness, make_cache(server))

    response = harness.post({"prompt": PROMPT, "use_cache": True})

    assert response.status_code == 429
    assert calls == []


# --- Cache unavailable fails open -----------------------------------------------------------------


def test_unconfigured_cache_routes_normally_with_one_info_line(caplog):
    harness = ApiHarness(EASY_JEV, [ok(CHEAP[0])])
    with_cache(harness, None)

    with caplog.at_level(logging.INFO, logger="app.routes.route"):
        response = harness.post({"prompt": PROMPT, "use_cache": True})

    assert response.status_code == 200
    assert response.json()["cache_hit"] is False
    skipped = [r for r in caplog.records if getattr(r, "reason", None) == "cache_not_configured"]
    assert len(skipped) == 1
    assert skipped[0].request_id == response.json()["request_id"]


def test_real_cache_factory_without_settings_routes_normally():
    harness = ApiHarness(EASY_JEV, [ok(CHEAP[0])])  # The real get_cache_factory; CACHE_* are unset in unit tests

    response = harness.post({"prompt": PROMPT, "use_cache": True})

    assert (response.status_code, response.json()["cache_hit"]) == (200, False)


@pytest.mark.parametrize(
    "error",
    [redis.exceptions.ConnectionError("redis://user:super-secret-password@cache.example.test"), OSError("socket")],
    ids=["connection", "socket"],
)
def test_redis_failure_returns_the_normal_response(caplog, error):
    harness = ApiHarness(EASY_JEV, [ok(CHEAP[0])])
    with_cache(harness, make_cache(None, client=FailingRedis(error)))
    reference = ApiHarness(EASY_JEV, [ok(CHEAP[0])]).post({"prompt": PROMPT}).json()

    with caplog.at_level(logging.INFO):
        response = harness.post({"prompt": PROMPT, "use_cache": True})

    assert response.status_code == 200
    body = response.json()
    for field in ("cache_hit", "decision", "response", "cost", "strong_reserved", "config_version"):
        assert body[field] == reference[field], field
    warnings = [r for r in caplog.records if r.name == "app.infra.cache"]
    assert [(r.operation, r.reason) for r in warnings] == [("lookup", "redis_error"), ("store", "redis_error")]
    assert "super-secret-password" not in caplog.text


def test_invalid_model_config_with_use_cache_is_a_500_configuration_error(monkeypatch, tmp_path):
    bad = tmp_path / "models.yaml"
    bad.write_text("not: [valid", encoding="utf-8")
    monkeypatch.setenv("MODELS_CONFIG_PATH", str(bad))
    server = fakeredis.FakeServer()
    harness = ApiHarness(EASY_JEV, [ok(CHEAP[0])])
    with_cache(harness, make_cache(server))

    response = harness.post({"prompt": PROMPT, "use_cache": True})

    assert response.status_code == 500
    assert response.json()["error"] == "configuration_error"
    assert str(bad) not in response.text
    assert harness.factory_calls == []


# --- Contract ---------------------------------------------------------------------------------------


def test_openapi_documents_use_cache_and_cache_hit():
    schemas = ApiHarness().client.get("/api/openapi.json").json()["components"]["schemas"]

    use_cache = schemas["RouteRequest"]["properties"]["use_cache"]
    assert (use_cache["type"], use_cache["default"]) == ("boolean", False)
    assert schemas["RouteResponse"]["properties"]["cache_hit"]["type"] == "boolean"
    assert "cache_hit" in schemas["RouteResponse"]["required"]
