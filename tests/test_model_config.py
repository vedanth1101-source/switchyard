import copy
import dataclasses
import re
from pathlib import Path

import pytest
import yaml

from app.core.model_config import DEFAULT_CONFIG_PATH, ModelConfigError, load_model_config
from app.core.policy import RoutingConfig, decide_route

REAL_CONFIG = Path(__file__).resolve().parents[1] / "config" / "models.yaml"

DELETE = object()


@pytest.fixture
def raw_config() -> dict:
    """The real config/models.yaml as plain data; tests mutate a fresh copy."""
    return yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))


def write_config(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def write_text(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def placeholders(raw: dict) -> dict:
    return {"cheap": raw["routes"]["general"]["cheap"][0], "strong": raw["routes"]["general"]["strong"][0]}


def resolve(value: object, names: dict) -> object:
    if isinstance(value, str):
        return value.format(**names)
    if isinstance(value, list):
        return [resolve(item, names) for item in value]
    if isinstance(value, dict):
        return {resolve(key, names): resolve(item, names) for key, item in value.items()}
    return value


def apply_change(raw: dict, path: tuple, value: object) -> dict:
    names = placeholders(raw)
    changed = copy.deepcopy(raw)
    keys = [resolve(key, names) for key in path]
    target = changed
    for key in keys[:-1]:
        target = target[key]
    if value is DELETE:
        del target[keys[-1]]
    else:
        target[keys[-1]] = resolve(value, names)
    return changed


# --- Loading the real file ---------------------------------------------------------------


def test_loads_real_models_yaml(raw_config):
    config = load_model_config(REAL_CONFIG)

    assert config.schema_version == raw_config["schema_version"]
    assert config.baseline_model == raw_config["baseline_model"]
    assert set(config.catalog) == set(raw_config["catalog"])
    assert set(config.providers) == set(raw_config["providers"])
    assert config.limits.max_chain_length == raw_config["limits"]["max_chain_length"]
    assert config.limits.max_attempts_per_model == raw_config["limits"]["max_attempts_per_model"]
    for model_id, entry in raw_config["catalog"].items():
        model = config.catalog[model_id]
        assert model.provider == entry["provider"]
        assert model.input_price == entry["input_price"]
        assert model.output_price == entry["output_price"]
        assert model.context_window == entry["context_window"]
        assert model.max_output_tokens == entry["max_output_tokens"]
    for name, entry in raw_config["providers"].items():
        assert config.providers[name].base_url == entry["base_url"]
        assert config.providers[name].api_key_env == entry["api_key_env"]


def test_default_path_is_the_repository_config():
    assert DEFAULT_CONFIG_PATH == REAL_CONFIG
    assert load_model_config() == load_model_config(REAL_CONFIG)


def test_builds_existing_routing_config_from_routes(raw_config):
    routing = load_model_config(REAL_CONFIG).routing

    assert isinstance(routing, RoutingConfig)
    assert routing.models == raw_config["routes"]
    assert routing.general_strong_chain == raw_config["routes"]["general"]["strong"]


def test_routing_thresholds_default_to_routing_config_defaults():
    routing = load_model_config(REAL_CONFIG).routing
    defaults = RoutingConfig()

    assert routing.confidence_threshold == defaults.confidence_threshold
    assert routing.difficulty_threshold == defaults.difficulty_threshold
    assert routing.reasoning_threshold == defaults.reasoning_threshold


def test_routing_thresholds_are_passed_through():
    routing = load_model_config(
        REAL_CONFIG, confidence_threshold=0.7, difficulty_threshold=4.0, reasoning_threshold=0.5
    ).routing

    assert (routing.confidence_threshold, routing.difficulty_threshold, routing.reasoning_threshold) == (0.7, 4.0, 0.5)


def test_task_without_own_route_uses_general_cheap_chain(raw_config):
    routing = load_model_config(REAL_CONFIG).routing

    assert routing.get_chain("code", "cheap") == raw_config["routes"]["general"]["cheap"]


def test_policy_routes_with_loaded_config(raw_config):
    route = decide_route(None, load_model_config(REAL_CONFIG).routing, jev_error="JevTimeoutError")

    assert route.reason == "JEV_ERROR"
    assert route.model_chain == raw_config["routes"]["general"]["strong"]


def test_task_specific_route_is_loaded(tmp_path, raw_config):
    names = placeholders(raw_config)
    raw_config["routes"]["math"] = {"cheap": [names["cheap"]]}

    routing = load_model_config(write_config(tmp_path, raw_config)).routing

    assert routing.get_chain("math", "cheap") == [names["cheap"]]


def test_model_config_is_immutable():
    config = load_model_config(REAL_CONFIG)
    model_id = next(iter(config.catalog))

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.baseline_model = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        config.catalog[model_id] = config.catalog[model_id]  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.catalog[model_id].input_price = 0.0  # type: ignore[misc]


# --- config_version ------------------------------------------------------------------------


def test_config_version_is_sha256_hex_and_deterministic():
    first = load_model_config(REAL_CONFIG).config_version

    assert re.fullmatch(r"[0-9a-f]{64}", first)
    assert load_model_config(REAL_CONFIG).config_version == first


@pytest.mark.parametrize(
    "threshold", [{"confidence_threshold": 0.7}, {"difficulty_threshold": 4.0}, {"reasoning_threshold": 0.5}]
)
def test_config_version_changes_with_threshold(threshold):
    assert load_model_config(REAL_CONFIG, **threshold).config_version != load_model_config(REAL_CONFIG).config_version


def test_config_version_changes_when_a_price_changes(tmp_path, raw_config):
    baseline_version = load_model_config(REAL_CONFIG).config_version
    model_id = placeholders(raw_config)["cheap"]
    raw_config["catalog"][model_id]["input_price"] += 0.01

    assert load_model_config(write_config(tmp_path, raw_config)).config_version != baseline_version


def test_config_version_changes_when_chain_order_changes(tmp_path, raw_config):
    baseline_version = load_model_config(REAL_CONFIG).config_version
    raw_config["routes"]["general"]["cheap"].reverse()

    assert load_model_config(write_config(tmp_path, raw_config)).config_version != baseline_version


def test_config_version_ignores_comments_key_order_and_pricing_metadata(tmp_path, raw_config):
    baseline_version = load_model_config(REAL_CONFIG).config_version
    raw_config["catalog"] = dict(reversed(list(raw_config["catalog"].items())))
    raw_config["pricing"]["as_of"] = "1999-01-01"
    reordered = dict(reversed(list(raw_config.items())))

    # safe_dump drops every comment from the real file.
    assert load_model_config(write_config(tmp_path, reordered)).config_version == baseline_version


# --- Invalid configuration -------------------------------------------------------------------

INVALID_CHANGES = [
    pytest.param(("schema_version",), 2, "schema_version", id="schema-version-unsupported"),
    pytest.param(("schema_version",), "1", "schema_version", id="schema-version-string"),
    pytest.param(("schema_version",), True, "schema_version", id="schema-version-bool"),
    pytest.param(("schema_version",), DELETE, "schema_version", id="schema-version-missing"),
    pytest.param(("providers",), DELETE, "providers", id="providers-missing"),
    pytest.param(("providers",), {}, "providers", id="providers-empty"),
    pytest.param(("providers", "ai_gateway", "base_url"), "http://insecure.example/v1", "base_url", id="base-url-http"),
    pytest.param(("providers", "ai_gateway", "base_url"), DELETE, "base_url", id="base-url-missing"),
    pytest.param(("providers", "ai_gateway", "api_key_env"), DELETE, "api_key_env", id="api-key-env-missing"),
    pytest.param(("providers", "ai_gateway", "api_key_env"), "ai-gateway-key", "api_key_env", id="api-key-env-bad"),
    pytest.param(("catalog",), DELETE, "catalog", id="catalog-missing"),
    pytest.param(("catalog",), {}, "catalog", id="catalog-empty"),
    pytest.param(("catalog", "{cheap}"), "not a mapping", "catalog.{cheap}", id="catalog-entry-not-mapping"),
    pytest.param(("catalog", "{cheap}", "provider"), "unknown_provider", "provider", id="provider-unknown"),
    pytest.param(("catalog", "{cheap}", "vendor"), DELETE, "vendor", id="vendor-missing"),
    pytest.param(("catalog", "{cheap}", "free_tier_eligible"), "yes", "free_tier_eligible", id="free-tier-not-bool"),
    pytest.param(("catalog", "{cheap}", "input_price"), DELETE, "input_price", id="input-price-missing"),
    pytest.param(("catalog", "{cheap}", "input_price"), -0.01, "input_price", id="input-price-negative"),
    pytest.param(("catalog", "{cheap}", "output_price"), True, "output_price", id="output-price-bool"),
    pytest.param(("catalog", "{cheap}", "output_price"), "0.40", "output_price", id="output-price-string"),
    pytest.param(("catalog", "{cheap}", "output_price"), float("nan"), "output_price", id="output-price-nan"),
    pytest.param(("catalog", "{cheap}", "context_window"), 0, "context_window", id="context-window-zero"),
    pytest.param(("catalog", "{cheap}", "max_output_tokens"), 1024.5, "max_output_tokens", id="max-output-float"),
    pytest.param(("catalog", "{cheap}", "max_output_tokens"), 10**9, "max_output_tokens", id="max-output-over-context"),
    pytest.param(("limits",), DELETE, "limits", id="limits-missing"),
    pytest.param(("limits", "max_chain_length"), 0, "max_chain_length", id="max-chain-length-zero"),
    pytest.param(("limits", "max_attempts_per_model"), DELETE, "max_attempts_per_model", id="max-attempts-missing"),
    pytest.param(("limits", "max_attempts_per_model"), True, "max_attempts_per_model", id="max-attempts-bool"),
    pytest.param(("baseline_model",), DELETE, "baseline_model", id="baseline-missing"),
    pytest.param(("baseline_model",), "vendor/not-in-catalog", "baseline_model", id="baseline-not-in-catalog"),
    pytest.param(("routes",), DELETE, "routes", id="routes-missing"),
    pytest.param(("routes",), ["general"], "routes", id="routes-not-mapping"),
    pytest.param(("routes", "cdoe"), {"cheap": ["{cheap}"]}, "routes.cdoe", id="task-type-unknown"),
    pytest.param(("routes", "code"), {}, "routes.code", id="task-without-tiers"),
    pytest.param(("routes", "general", "medium"), ["{cheap}"], "routes.general.medium", id="tier-unknown"),
    pytest.param(("routes", "general", "cheap"), [], "routes.general.cheap", id="chain-empty"),
    pytest.param(("routes", "general", "cheap"), "{cheap}", "routes.general.cheap", id="chain-not-list"),
    pytest.param(("routes", "general", "cheap"), ["{cheap}", "{cheap}"], "duplicate", id="chain-duplicate"),
    pytest.param(
        ("routes", "general", "cheap"), ["vendor/not-in-catalog"], "vendor/not-in-catalog", id="model-unknown"
    ),
    pytest.param(("routes", "general", "cheap"), [42], "routes.general.cheap[0]", id="model-not-string"),
    pytest.param(("limits", "max_chain_length"), 1, "max_chain_length", id="chain-too-long"),
    pytest.param(("routes", "general"), DELETE, "routes.general", id="general-missing"),
    pytest.param(("routes", "general", "cheap"), DELETE, "routes.general.cheap", id="general-cheap-missing"),
    pytest.param(("routes", "general", "strong"), DELETE, "routes.general.strong", id="general-strong-missing"),
    pytest.param(("routes", "code"), {"strong": ["{cheap}"]}, "both cheap and strong", id="cheap-model-in-strong"),
    pytest.param(("routes", "code"), {"cheap": ["{strong}"]}, "both cheap and strong", id="strong-model-in-cheap"),
]


@pytest.mark.parametrize(("path", "value", "fragment"), INVALID_CHANGES)
def test_invalid_config_raises_with_location(tmp_path, raw_config, path, value, fragment):
    config_path = write_config(tmp_path, apply_change(raw_config, path, value))

    with pytest.raises(ModelConfigError) as excinfo:
        load_model_config(config_path)

    message = str(excinfo.value)
    assert resolve(fragment, placeholders(raw_config)) in message
    assert str(config_path) in message


def test_api_key_env_error_does_not_echo_the_value(tmp_path, raw_config):
    secret_like = "sk-live-DO-NOT-ECHO-123"
    raw_config["providers"]["ai_gateway"]["api_key_env"] = secret_like

    with pytest.raises(ModelConfigError) as excinfo:
        load_model_config(write_config(tmp_path, raw_config))

    assert secret_like not in str(excinfo.value)


# --- Invalid YAML / files ----------------------------------------------------------------------


def test_duplicate_top_level_key_is_rejected(tmp_path):
    text = REAL_CONFIG.read_text(encoding="utf-8") + "\nbaseline_model: vendor/duplicate\n"

    with pytest.raises(ModelConfigError, match="duplicate key 'baseline_model'"):
        load_model_config(write_text(tmp_path, text))


def test_duplicate_nested_key_is_rejected(tmp_path):
    text = "catalog:\n  vendor/model:\n    input_price: 1\n  vendor/model:\n    input_price: 2\n"

    with pytest.raises(ModelConfigError, match="duplicate key 'vendor/model'"):
        load_model_config(write_text(tmp_path, text))


def test_merge_key_override_is_not_a_duplicate(tmp_path):
    text = "base: &base {x: 1}\nother:\n  <<: *base\n  x: 2\n"

    # Parsing succeeds; validation then fails on the first missing section.
    with pytest.raises(ModelConfigError, match="schema_version: missing"):
        load_model_config(write_text(tmp_path, text))


def test_unhashable_key_is_rejected(tmp_path):
    with pytest.raises(ModelConfigError, match="invalid YAML"):
        load_model_config(write_text(tmp_path, "? [a, b]\n: 1\n"))


def test_yaml_syntax_error_is_wrapped(tmp_path):
    with pytest.raises(ModelConfigError, match="invalid YAML") as excinfo:
        load_model_config(write_text(tmp_path, "routes: [unclosed\n"))

    assert isinstance(excinfo.value.__cause__, yaml.YAMLError)


def test_unsafe_python_tag_is_rejected(tmp_path):
    text = "schema_version: !!python/object/apply:os.system ['echo unsafe']\n"

    with pytest.raises(ModelConfigError, match="invalid YAML"):
        load_model_config(write_text(tmp_path, text))


@pytest.mark.parametrize("text", ["", "- a\n- b\n", "just a string\n"], ids=["empty", "list", "scalar"])
def test_top_level_must_be_a_mapping(tmp_path, text):
    with pytest.raises(ModelConfigError, match="top level: must be a mapping"):
        load_model_config(write_text(tmp_path, text))


def test_missing_file_is_wrapped(tmp_path):
    with pytest.raises(ModelConfigError, match="cannot read") as excinfo:
        load_model_config(tmp_path / "missing.yaml")

    assert isinstance(excinfo.value.__cause__, FileNotFoundError)


def test_undecodable_file_is_wrapped(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_bytes(b"\xff\xfe\xfa")

    with pytest.raises(ModelConfigError, match="cannot read"):
        load_model_config(path)
