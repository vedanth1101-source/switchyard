import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.executor import CALL_MARGIN_SECONDS, MIN_CALL_SECONDS
from app.core.jev_client import JevClient
from app.core.model_config import DEFAULT_CONFIG_PATH, load_model_config
from app.core.policy import RoutingConfig
from app.core.settings import FUNCTION_MAX_DURATION_SECONDS, Settings

ROOT = Path(__file__).resolve().parents[1]
S1_2_ENV_VARS = (
    "STRONG_DAILY_CAP",
    "REQUEST_BUDGET_SECONDS",
    "AI_GATEWAY_API_KEY",
    "MODELS_CONFIG_PATH",
    "CONFIDENCE_THRESHOLD",
    "JEV_TIMEOUT_SECONDS",
)
FAKE_KEY = "fake-gateway-key-for-tests-0123456789"


@pytest.fixture(autouse=True)
def clear_s1_2_env(monkeypatch):
    """Values from the developer's shell must not leak into default checks."""
    for name in S1_2_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_defaults():
    settings = Settings()

    assert settings.strong_daily_cap == 100
    assert settings.request_budget_seconds == 50
    assert settings.ai_gateway_api_key is None
    assert settings.models_config_path == DEFAULT_CONFIG_PATH
    assert settings.confidence_threshold == 0.6
    assert settings.jev_timeout_seconds == 8.0


def test_jev_timeout_default_is_the_jev_client_default():
    assert Settings().jev_timeout_seconds == JevClient.DEFAULT_TIMEOUT_SECONDS == 8.0


def test_confidence_threshold_default_comes_from_routing_config():
    assert Settings().confidence_threshold == RoutingConfig.confidence_threshold


def test_default_settings_reproduce_the_loader_default_config_version():
    settings = Settings()

    from_settings = load_model_config(settings.models_config_path, confidence_threshold=settings.confidence_threshold)

    assert from_settings.config_version == load_model_config().config_version


def test_environment_overrides(monkeypatch, tmp_path):
    config_path = tmp_path / "models.yaml"
    monkeypatch.setenv("STRONG_DAILY_CAP", "5")
    monkeypatch.setenv("REQUEST_BUDGET_SECONDS", "12.5")
    monkeypatch.setenv("AI_GATEWAY_API_KEY", FAKE_KEY)
    monkeypatch.setenv("MODELS_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.75")
    monkeypatch.setenv("JEV_TIMEOUT_SECONDS", "5.5")

    settings = Settings()

    assert settings.strong_daily_cap == 5
    assert settings.request_budget_seconds == 12.5
    assert settings.ai_gateway_api_key is not None
    assert settings.ai_gateway_api_key.get_secret_value() == FAKE_KEY
    assert settings.models_config_path == config_path
    assert settings.confidence_threshold == 0.75
    assert settings.jev_timeout_seconds == 5.5


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"], ids=["empty", "spaces", "whitespace"])
def test_blank_api_key_env_is_unset(monkeypatch, blank):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", blank)

    assert Settings().ai_gateway_api_key is None


def test_blank_api_key_argument_is_unset():
    assert Settings(ai_gateway_api_key="   ").ai_gateway_api_key is None


@pytest.mark.parametrize("name", S1_2_ENV_VARS)
def test_empty_env_var_uses_the_default(monkeypatch, name):
    monkeypatch.setenv(name, "")
    field = name.lower()

    assert getattr(Settings(), field) == Settings.model_fields[field].default


def test_boundary_values_are_accepted(monkeypatch):
    monkeypatch.setenv("STRONG_DAILY_CAP", "0")
    monkeypatch.setenv("REQUEST_BUDGET_SECONDS", "59.9")
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "1")

    settings = Settings()

    assert (settings.strong_daily_cap, settings.request_budget_seconds, settings.confidence_threshold) == (0, 59.9, 1)


def test_api_key_is_never_rendered(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", FAKE_KEY)
    settings = Settings()

    for rendered in (repr(settings), str(settings), settings.model_dump_json(), repr(settings.model_dump())):
        assert FAKE_KEY not in rendered


INVALID_VALUES = [
    ("STRONG_DAILY_CAP", "-1"),
    ("STRONG_DAILY_CAP", "ten"),
    ("STRONG_DAILY_CAP", "1.5"),
    ("REQUEST_BUDGET_SECONDS", "0"),
    ("REQUEST_BUDGET_SECONDS", "-5"),
    ("REQUEST_BUDGET_SECONDS", "60"),
    ("REQUEST_BUDGET_SECONDS", "nan"),
    ("REQUEST_BUDGET_SECONDS", "inf"),
    ("CONFIDENCE_THRESHOLD", "1.5"),
    ("CONFIDENCE_THRESHOLD", "-0.1"),
    ("CONFIDENCE_THRESHOLD", "nan"),
    ("JEV_TIMEOUT_SECONDS", "0"),
    ("JEV_TIMEOUT_SECONDS", "-1"),
    ("JEV_TIMEOUT_SECONDS", "nan"),
    ("JEV_TIMEOUT_SECONDS", "inf"),
    ("JEV_TIMEOUT_SECONDS", "eight"),
]


@pytest.mark.parametrize(("name", "value"), INVALID_VALUES)
def test_invalid_value_fails_naming_the_field_without_the_input(monkeypatch, name, value):
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError) as excinfo:
        Settings()

    message = str(excinfo.value)
    assert name.lower() in message
    assert "input_value" not in message


def test_validation_error_does_not_echo_a_key_pasted_into_the_wrong_variable(monkeypatch):
    monkeypatch.setenv("STRONG_DAILY_CAP", FAKE_KEY)

    with pytest.raises(ValidationError) as excinfo:
        Settings()

    assert FAKE_KEY not in str(excinfo.value)


def test_budget_bound_matches_vercel_max_duration():
    vercel = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))

    assert vercel["functions"]["app/main.py"]["maxDuration"] == FUNCTION_MAX_DURATION_SECONDS
    assert Settings().request_budget_seconds < FUNCTION_MAX_DURATION_SECONDS


# --- S2.6 rate limit ------------------------------------------------------------------------

S2_6_ENV_VARS = ("RATE_LIMIT_PER_MIN", "CLIENT_IP_HEADER", "RATE_LIMIT_IP_SALT")
FAKE_SALT = "fake-rate-limit-salt-for-tests-0123456789"


@pytest.fixture
def clear_s2_6_env(monkeypatch):
    for name in S2_6_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_rate_limit_defaults(clear_s2_6_env):
    settings = Settings()

    assert (settings.rate_limit_per_min, settings.client_ip_header, settings.rate_limit_ip_salt) == (10, None, None)


def test_rate_limit_environment_overrides(clear_s2_6_env, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "25")
    monkeypatch.setenv("CLIENT_IP_HEADER", "x-real-ip")
    monkeypatch.setenv("RATE_LIMIT_IP_SALT", FAKE_SALT)

    settings = Settings()

    assert (settings.rate_limit_per_min, settings.client_ip_header) == (25, "x-real-ip")
    assert settings.rate_limit_ip_salt.get_secret_value() == FAKE_SALT


@pytest.mark.parametrize("name", ["CLIENT_IP_HEADER", "RATE_LIMIT_IP_SALT"])
@pytest.mark.parametrize("blank", ["", "   "], ids=["empty", "spaces"])
def test_blank_rate_limit_strings_are_unset(clear_s2_6_env, monkeypatch, name, blank):
    monkeypatch.setenv(name, blank)

    assert getattr(Settings(), name.lower()) is None


def test_blank_rate_limit_arguments_are_unset():
    settings = Settings(client_ip_header="  ", rate_limit_ip_salt="  ")

    assert (settings.client_ip_header, settings.rate_limit_ip_salt) == (None, None)


def test_rate_limit_salt_is_never_rendered(clear_s2_6_env, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_IP_SALT", FAKE_SALT)
    settings = Settings()

    for rendered in (repr(settings), str(settings), settings.model_dump_json(), repr(settings.model_dump())):
        assert FAKE_SALT not in rendered


@pytest.mark.parametrize("value", ["0", "-1", "ten", "1.5"])
def test_invalid_rate_limit_fails_naming_the_field_without_the_input(clear_s2_6_env, monkeypatch, value):
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", value)

    with pytest.raises(ValidationError) as excinfo:
        Settings()

    assert "rate_limit_per_min" in str(excinfo.value)
    assert "input_value" not in str(excinfo.value)


@pytest.mark.parametrize(
    "raw", ["  x-real-ip  ", "\tx-real-ip\n", "x-real-ip "], ids=["spaces", "tab-newline", "trailing"]
)
def test_client_ip_header_is_stripped_from_the_environment(clear_s2_6_env, monkeypatch, raw):
    monkeypatch.setenv("CLIENT_IP_HEADER", raw)

    assert Settings().client_ip_header == "x-real-ip"


def test_client_ip_header_argument_is_stripped():
    assert Settings(client_ip_header=" x-real-ip ").client_ip_header == "x-real-ip"


# --- S2.7 Jev key ---------------------------------------------------------------------------

JEV_CANARY_KEY = "jev-canary-key-S27-0123456789-not-real"


def test_jev_key_defaults_to_none(monkeypatch):
    monkeypatch.delenv("JEVSTATION_API_KEY", raising=False)

    assert Settings().jevstation_api_key is None


def test_jev_key_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("JEVSTATION_API_KEY", JEV_CANARY_KEY)

    assert Settings().jevstation_api_key.get_secret_value() == JEV_CANARY_KEY


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"], ids=["empty", "spaces", "whitespace"])
def test_blank_jev_key_is_unset(monkeypatch, blank):
    monkeypatch.setenv("JEVSTATION_API_KEY", blank)

    assert Settings().jevstation_api_key is None
    assert Settings(jevstation_api_key=blank).jevstation_api_key is None


def test_jev_key_is_never_rendered(monkeypatch):
    monkeypatch.setenv("JEVSTATION_API_KEY", JEV_CANARY_KEY)
    settings = Settings()

    for rendered in (repr(settings), str(settings), settings.model_dump_json(), repr(settings.model_dump())):
        assert JEV_CANARY_KEY not in rendered


def test_validation_error_does_not_echo_the_jev_key(monkeypatch):
    monkeypatch.setenv("JEVSTATION_API_KEY", JEV_CANARY_KEY)
    monkeypatch.setenv("STRONG_DAILY_CAP", JEV_CANARY_KEY)  # The key pasted into the wrong variable too

    with pytest.raises(ValidationError) as excinfo:
        Settings()

    assert JEV_CANARY_KEY not in str(excinfo.value)


# --- S2.9: the Jev timeout must leave room for a model call ---------------------------------
# Rule (approved plan): jev_timeout_seconds + MIN_CALL_SECONDS + CALL_MARGIN_SECONDS < request_budget_seconds

CALL_START_SECONDS = MIN_CALL_SECONDS + CALL_MARGIN_SECONDS
JEV_TIMEOUT = 46.25  # Distinctive values, so a leaked value would be visible in the message
BOUNDARY_BUDGET = JEV_TIMEOUT + CALL_START_SECONDS  # 50.25: equal, so rejected (strict <)


def test_default_timeouts_leave_room_for_a_model_call():
    settings = Settings()

    assert settings.jev_timeout_seconds + CALL_START_SECONDS < settings.request_budget_seconds


@pytest.mark.parametrize(
    ("jev_timeout", "budget"),
    [(8.0, 50.0), (JEV_TIMEOUT, BOUNDARY_BUDGET + 0.01), (5.5, 12.5), (6.5, 40.0)],
    ids=["defaults", "just-above", "env-override-test", "cli-deps-test"],
)
def test_timeouts_with_room_for_a_call_are_accepted(jev_timeout, budget):
    settings = Settings(jev_timeout_seconds=jev_timeout, request_budget_seconds=budget)

    assert (settings.jev_timeout_seconds, settings.request_budget_seconds) == (jev_timeout, budget)


@pytest.mark.parametrize(
    ("jev_timeout", "budget"),
    [(JEV_TIMEOUT, BOUNDARY_BUDGET), (JEV_TIMEOUT, BOUNDARY_BUDGET - 0.01), (JEV_TIMEOUT, 47.75), (55.5, 50.25)],
    ids=["exact-boundary", "just-below", "far-below", "jev-longer-than-budget"],
)
def test_timeouts_without_room_for_a_call_are_rejected_naming_both_fields_without_values(jev_timeout, budget):
    with pytest.raises(ValidationError) as excinfo:
        Settings(jev_timeout_seconds=jev_timeout, request_budget_seconds=budget)

    message = str(excinfo.value)
    assert "jev_timeout_seconds" in message
    assert "request_budget_seconds" in message
    for value in (jev_timeout, budget):
        assert repr(value) not in message
        assert str(value) not in message
    assert "input_value" not in message


def test_timeouts_from_the_environment_are_cross_checked(monkeypatch):
    monkeypatch.setenv("JEV_TIMEOUT_SECONDS", str(JEV_TIMEOUT))
    monkeypatch.setenv("REQUEST_BUDGET_SECONDS", str(BOUNDARY_BUDGET))

    with pytest.raises(ValidationError) as excinfo:
        Settings()

    assert "jev_timeout_seconds" in str(excinfo.value)
    assert str(JEV_TIMEOUT) not in str(excinfo.value)
