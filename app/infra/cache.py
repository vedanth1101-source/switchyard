"""
Response cache configuration and keys (Task S2.14, step 1; docs/04 Part B).

Only the settings and the key are implemented here. Nothing reads or writes Redis yet, and /api/route
does not use the cache: `use_cache` is still accepted and ignored.

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
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from pydantic import SecretStr

from app.core.classifier import get_default_questions
from app.core.jev_client import JevClient
from app.core.settings import Settings

KEY_PREFIX = "cache"
CACHE_FORMAT_VERSION = 1  # Bump when the key layout or the stored value changes; old entries are then ignored

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
