import hashlib
import re
import unicodedata

import pytest
from pydantic import SecretStr

from app.core.classifier import get_default_questions
from app.core.jev_client import JevClient
from app.core.settings import Settings
from app.infra.cache import (
    CACHE_FORMAT_VERSION,
    CacheConfig,
    cache_config,
    cache_key,
    classifier_version,
    normalize_prompt,
)

SECRET = "fake-cache-key-secret-for-tests-0123456789"
OTHER_SECRET = "another-fake-cache-key-secret-for-tests-987"
CONFIG_VERSION = "a" * 64
OTHER_CONFIG_VERSION = "b" * 64
CLASSIFIER_VERSION = "c" * 64
OTHER_CLASSIFIER_VERSION = "d" * 64
PROMPT = "Write a function to check if a number is prime"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def make_config(namespace="test", secret=SECRET):
    return CacheConfig(
        redis_url="redis://localhost:6379/0", namespace=namespace, key_secret=SecretStr(secret), ttl_seconds=86400
    )


def key(prompt=PROMPT, *, config=None, config_version=CONFIG_VERSION, classifier=CLASSIFIER_VERSION):
    return cache_key(config or make_config(), prompt, config_version=config_version, classifier_version=classifier)


# --- normalization ------------------------------------------------------------------------


def test_normalize_applies_nfc():
    decomposed = "Café"  # "e" + combining acute accent
    composed = "Café"

    assert decomposed != composed
    assert normalize_prompt(decomposed) == composed
    assert normalize_prompt(composed) == unicodedata.normalize("NFC", composed)


@pytest.mark.parametrize("raw", ["  prompt  ", "\tprompt\n", "\r\n prompt", "prompt 　"])
def test_normalize_trims_leading_and_trailing_whitespace(raw):
    assert normalize_prompt(raw) == "prompt"


def test_normalize_keeps_case_and_internal_whitespace():
    raw = "def f():\n    Return  X"

    assert normalize_prompt(raw) == raw


# --- key determinism and normalization ---------------------------------------------------------


def test_identical_inputs_give_identical_keys():
    assert key() == key()


def test_key_has_the_documented_layout():
    parts = key().split(":")

    assert parts[:5] == ["cache", f"v{CACHE_FORMAT_VERSION}", "test", CONFIG_VERSION, CLASSIFIER_VERSION]
    assert len(parts) == 6
    assert HEX64.match(parts[5])


def test_nfc_equivalent_prompts_share_a_key():
    assert key("Café menu") == key("Café menu")


def test_outer_whitespace_does_not_change_the_key():
    assert key(f"  {PROMPT}\n") == key(PROMPT)


def test_case_changes_the_key():
    assert key(PROMPT.upper()) != key(PROMPT)


def test_internal_whitespace_changes_the_key():
    assert key(PROMPT.replace(" ", "  ", 1)) != key(PROMPT)
    assert key("a\nb") != key("a b")


def test_config_version_changes_the_key():
    assert key(config_version=OTHER_CONFIG_VERSION) != key()


def test_classifier_version_changes_the_key():
    assert key(classifier=OTHER_CLASSIFIER_VERSION) != key()


def test_namespace_changes_the_key():
    assert key(config=make_config(namespace="preview")) != key(config=make_config(namespace="production"))


def test_secret_changes_the_key():
    assert key(config=make_config(secret=OTHER_SECRET)) != key()


def test_key_contains_neither_the_prompt_nor_its_plain_sha256():
    result = key()
    normalized = normalize_prompt(PROMPT)

    for digest in (hashlib.sha256(PROMPT.encode()).hexdigest(), hashlib.sha256(normalized.encode()).hexdigest()):
        assert digest not in result
    assert PROMPT not in result
    assert "prime" not in result


def test_key_never_contains_the_secret():
    assert SECRET not in key()


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"], ids=["empty", "spaces", "whitespace"])
def test_blank_prompt_is_rejected(blank):
    with pytest.raises(ValueError):
        key(blank)


@pytest.mark.parametrize("bad", ["", "abc", "A" * 64, "a" * 63, "a:" + "a" * 62, "a" * 64 + "\n"])
def test_versions_must_be_sha256_hex(bad):
    with pytest.raises(ValueError):
        key(config_version=bad)
    with pytest.raises(ValueError):
        key(classifier=bad)


# --- classifier version ---------------------------------------------------------------------


def test_classifier_version_is_deterministic_sha256_hex():
    assert classifier_version() == classifier_version()
    assert HEX64.match(classifier_version())


def test_classifier_version_defaults_to_the_jev_model_and_default_questions():
    assert classifier_version() == classifier_version(JevClient.DEFAULT_MODEL, get_default_questions())


def test_jev_client_evaluate_uses_the_default_model_the_version_covers():
    import inspect

    assert inspect.signature(JevClient.evaluate).parameters["model"].default == JevClient.DEFAULT_MODEL


def test_classifier_version_changes_with_the_jev_model():
    assert classifier_version("jev-1.13.0") != classifier_version()


def test_classifier_version_changes_with_the_questions():
    questions = get_default_questions()
    questions["task_type"]["instructions"] += " Changed."

    assert classifier_version(questions=questions) != classifier_version()


def test_classifier_version_ignores_question_key_order():
    questions = get_default_questions()
    reordered = dict(reversed(list(questions.items())))

    assert classifier_version(questions=reordered) == classifier_version(questions=questions)


# --- configuration ----------------------------------------------------------------------------

REDIS_URL = "redis://localhost:6379/0"


def configured(**overrides):
    values = {"redis_url": REDIS_URL, "cache_namespace": "test", "cache_key_secret": SECRET}
    values.update(overrides)
    return Settings(**values)


def test_cache_config_from_complete_settings():
    config = cache_config(configured(cache_ttl_seconds=60))

    assert config is not None
    assert (config.redis_url, config.namespace, config.ttl_seconds) == (REDIS_URL, "test", 60)
    assert config.key_secret.get_secret_value() == SECRET


def test_cache_config_ttl_defaults_to_a_day():
    assert cache_config(configured()).ttl_seconds == 86400


@pytest.mark.parametrize("missing", ["redis_url", "cache_namespace", "cache_key_secret"])
def test_cache_is_disabled_when_any_setting_is_missing(missing):
    assert cache_config(configured(**{missing: None})) is None


@pytest.mark.parametrize("missing", ["redis_url", "cache_namespace", "cache_key_secret"])
def test_cache_is_disabled_when_any_setting_is_blank(missing):
    assert cache_config(configured(**{missing: "   "})) is None


def test_cache_is_disabled_by_default():
    assert cache_config(Settings()) is None


def test_cache_config_never_renders_the_secret():
    config = cache_config(configured())

    for rendered in (repr(config), str(config)):
        assert SECRET not in rendered
