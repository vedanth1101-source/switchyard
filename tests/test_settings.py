import json
from pathlib import Path

import pytest
from pydantic import ValidationError

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
