"""
Response cache: configuration, keys and Redis reads/writes (Task S2.14; docs/04 Part B).

/api/route uses it for requests with `use_cache: true` (app.routes.route).

Key: `cache:v{CACHE_FORMAT_VERSION}:{namespace}:{config_version}:{classifier_version}:{prompt_hmac}`.
- namespace: CACHE_NAMESPACE, the deployment environment, so Preview and Production never share entries
  even when they share a Redis database.
- config_version: the model configuration's version (models.yaml and routing thresholds).
- classifier_version: covers what config_version does not, the Jev model identifier and the default
  classifier questions (see classifier_version()).
- prompt_hmac: HMAC-SHA256 under CACHE_KEY_SECRET of the normalized prompt's UTF-8 bytes. A keyed hash,
  not a plain sha256: anyone who can list Redis keys cannot confirm a guessed prompt.

Normalization is Unicode NFC, then trimming leading and trailing whitespace (str.strip). Case and
internal whitespace are kept: they change the meaning of code and of many prompts.

The cache is configured only when REDIS_URL, CACHE_NAMESPACE and CACHE_KEY_SECRET are all set;
cache_config() returns None otherwise, which means the cache is off.

Stored value: a CachedAnswer as JSON, written with SET ... EX CACHE_TTL_SECONDS. It holds the routing
decision and the answer of one fresh successful request, plus the format and versions it was made under.
It never holds the prompt, its hash, the client IP, attempts, tokens or costs. Only successful, non-dry-run
answers are cached, and not when the request fell back to the cheap chain or was downgraded for budget
(reason BUDGET): those answers are worse than the route intended, and a BUDGET answer would outlive the
day's budget.

Failures fail open, with one WARNING carrying the operation, a reason and at most the exception type
(redis-py messages can contain URLs and credentials): a Redis error or timeout makes lookup() a miss and
store() a skipped write, and an entry that does not parse, or was made under other versions, is a miss.
The next successful store() overwrites it. Concurrent identical requests both miss and both write; the
last write wins. Other exceptions are bugs and propagate. The Redis client uses CACHE_TIMEOUT_SECONDS,
shorter than the rate limiter's and budget's, so an unreachable cache adds little latency.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Mapping, Optional

import redis
import redis.asyncio
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, SecretStr

from app.core.classifier import get_default_questions
from app.core.jev_client import JevClient
from app.core.router import RouteOutcome
from app.core.settings import Settings

logger = logging.getLogger(__name__)

KEY_PREFIX = "cache"
CACHE_FORMAT_VERSION = 1  # Bump when the key layout or the stored value changes; old entries are then ignored
CACHE_TIMEOUT_SECONDS = 0.3  # Connect and per-operation Redis timeout for the cache (decision U7)

# Expected outages: redis-py errors (connection, timeout, server) and raw socket errors.
REDIS_ERRORS = (redis.exceptions.RedisError, OSError)

_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class CacheConfig:
    redis_url: str
    namespace: str
    key_secret: SecretStr
    ttl_seconds: int


def cache_config(settings: Settings) -> Optional[CacheConfig]:
    """The cache settings, or None (cache off) when REDIS_URL, CACHE_NAMESPACE or CACHE_KEY_SECRET is unset."""
    if settings.redis_url is None or settings.cache_namespace is None or settings.cache_key_secret is None:
        return None
    return CacheConfig(
        redis_url=settings.redis_url,
        namespace=settings.cache_namespace,
        key_secret=settings.cache_key_secret,
        ttl_seconds=settings.cache_ttl_seconds,
    )


def normalize_prompt(prompt: str) -> str:
    return unicodedata.normalize("NFC", prompt).strip()


def classifier_version(jev_model: str = JevClient.DEFAULT_MODEL, questions: Optional[Mapping[str, Any]] = None) -> str:
    """sha256 hex of the Jev model identifier and the classifier questions the router sends to Jev."""
    canonical = json.dumps(
        {"jev_model": jev_model, "questions": get_default_questions() if questions is None else questions},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def cache_key(config: CacheConfig, prompt: str, *, config_version: str, classifier_version: str) -> str:
    """
    The Redis key for one prompt.

    Raises:
        ValueError: If the prompt is blank after normalization, or a version is not sha256 hex.
    """
    for version in (config_version, classifier_version):
        if not _SHA256_HEX.fullmatch(version):
            raise ValueError("config_version and classifier_version must be sha256 hex digests")
    normalized = normalize_prompt(prompt)
    if not normalized:
        raise ValueError("prompt is blank after normalization")
    digest = hmac.new(
        config.key_secret.get_secret_value().encode("utf-8"), normalized.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return f"{KEY_PREFIX}:v{CACHE_FORMAT_VERSION}:{config.namespace}:{config_version}:{classifier_version}:{digest}"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CachedDecision(_Frozen):
    task_type: str
    tier: str
    reason: str
    model_chain: list[str]
    confidence: Optional[float]  # Jev fields are null when Jev failed (reason JEV_ERROR)
    difficulty: Optional[float]
    needs_reasoning_prob: Optional[float]
    task_type_probs: Optional[dict[str, float]]
    difficulty_dist: Optional[dict[str, float]]


class CachedResponse(_Frozen):
    model: str
    tier: str
    text: str = Field(min_length=1)


class CachedAnswer(_Frozen):
    v: Literal[1]  # CACHE_FORMAT_VERSION
    created_at: AwareDatetime
    config_version: str
    classifier_version: str
    source_request_id: uuid.UUID  # The request whose answer this is
    decision: CachedDecision
    response: CachedResponse


def answer_from_outcome(outcome: RouteOutcome, *, classifier_version: str, now: datetime) -> Optional[CachedAnswer]:
    """The entry to cache for an outcome, or None when the outcome must not be cached."""
    execution = outcome.execution
    if outcome.status != "success" or execution is None:
        return None
    if execution.fell_back_to_cheap or outcome.decision.reason == "BUDGET":
        return None
    decision, jev = outcome.decision, outcome.jev
    return CachedAnswer(
        v=CACHE_FORMAT_VERSION,
        created_at=now,
        config_version=outcome.config_version,
        classifier_version=classifier_version,
        source_request_id=uuid.UUID(outcome.request_id),
        decision=CachedDecision(
            task_type=decision.task_type,
            tier=decision.tier,
            reason=decision.reason,
            model_chain=list(decision.model_chain),
            confidence=decision.confidence,
            difficulty=decision.difficulty,
            needs_reasoning_prob=decision.needs_reasoning_prob,
            task_type_probs=dict(jev.task_type_probs) if jev else None,
            difficulty_dist=dict(jev.difficulty_dist) if jev else None,
        ),
        response=CachedResponse(model=execution.final_model, tier=execution.final_tier, text=execution.result.text),
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ResponseCache:
    """Cached answers in Redis, keyed by cache_key(); every failure is a miss or a skipped write."""

    def __init__(
        self,
        client: redis.asyncio.Redis,
        config: CacheConfig,
        *,
        classifier_version: Optional[str] = None,
        now: Callable[[], datetime] = _utc_now,
    ):
        self._client = client
        self._config = config
        self._classifier_version = classifier_version if classifier_version is not None else _classifier_version()
        self._now = now

    async def lookup(self, prompt: str, *, config_version: str) -> Optional[CachedAnswer]:
        """The cached answer for this prompt under the current versions, or None."""
        key = self._key(prompt, config_version)
        try:
            raw = await self._client.get(key)
        except REDIS_ERRORS as exc:
            _log_unavailable("lookup", "redis_error", type(exc).__name__)
            return None
        if raw is None:
            return None
        try:
            answer = CachedAnswer.model_validate_json(raw)
        except ValueError as exc:  # Not UTF-8 or JSON, or not a valid entry (pydantic's ValidationError)
            _log_unavailable("lookup", "malformed_entry", type(exc).__name__)
            return None
        if answer.config_version != config_version or answer.classifier_version != self._classifier_version:
            _log_unavailable("lookup", "version_mismatch", None)
            return None
        return answer

    async def store(self, prompt: str, outcome: RouteOutcome) -> bool:
        """Caches the outcome's answer when it is cacheable. True only when an entry was written."""
        now = self._now()
        if now.tzinfo is None:
            raise ValueError("now() must return a timezone-aware datetime")
        answer = answer_from_outcome(outcome, classifier_version=self._classifier_version, now=now)
        if answer is None:
            return False
        key = self._key(prompt, outcome.config_version)
        try:
            await self._client.set(key, answer.model_dump_json(), ex=self._config.ttl_seconds)
        except REDIS_ERRORS as exc:
            _log_unavailable("store", "redis_error", type(exc).__name__)
            return False
        return True

    async def aclose(self) -> None:
        await self._client.aclose()

    def _key(self, prompt: str, config_version: str) -> str:
        return cache_key(
            self._config, prompt, config_version=config_version, classifier_version=self._classifier_version
        )


_classifier_version = classifier_version  # ResponseCache's parameter shadows the function's name


def create_response_cache(settings: Settings) -> Optional[ResponseCache]:
    """Builds the cache from Settings without connecting, or None when the cache is off or the URL is unusable."""
    config = cache_config(settings)
    if config is None:
        return None
    try:
        client = redis.asyncio.Redis.from_url(
            config.redis_url, socket_connect_timeout=CACHE_TIMEOUT_SECONDS, socket_timeout=CACHE_TIMEOUT_SECONDS
        )
    except ValueError as exc:  # Unparseable or unsupported URL
        _log_unavailable("create", "invalid_redis_url", type(exc).__name__)
        return None
    return ResponseCache(client, config)


def _log_unavailable(operation: str, reason: str, error_type: Optional[str]) -> None:
    # Operation, reason and exception type only: never the key, the prompt, the URL or exception messages.
    logger.warning(
        "response cache unavailable operation=%s reason=%s error_type=%s",
        operation,
        reason,
        error_type,
        extra={"operation": operation, "reason": reason, "error_type": error_type},
    )
