"""
Loader and validator for config/models.yaml (Task S1.1).

Parses the file with a SafeLoader that also rejects duplicate mapping keys, validates every
section, and builds the existing RoutingConfig from `routes`. Routing thresholds are not part of
the YAML: they default to RoutingConfig's values and are included in `config_version`.

Error messages name the offending location but never echo provider settings, which could hold a
mistakenly pasted secret.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping

import yaml

from app.core.classifier import get_default_questions
from app.core.policy import RoutingConfig

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "models.yaml"

SUPPORTED_SCHEMA_VERSION = 1
TIERS = ("cheap", "strong")
GENERAL_TASK = "general"
TASK_TYPES = frozenset(get_default_questions()["task_type"]["criteria"])
ENV_VAR_NAME = re.compile(r"[A-Z][A-Z0-9_]*")

Routes = Dict[str, Dict[str, List[str]]]


class ModelConfigError(ValueError):
    """Raised when config/models.yaml cannot be read, parsed or validated."""


@dataclass(frozen=True)
class Provider:
    base_url: str
    api_key_env: str  # Name of the environment variable holding the key, never the key itself


@dataclass(frozen=True)
class CatalogModel:
    provider: str
    vendor: str
    free_tier_eligible: bool
    input_price: float  # USD per million tokens (list price, for estimates only)
    output_price: float
    context_window: int
    max_output_tokens: int


@dataclass(frozen=True)
class Limits:
    max_chain_length: int
    max_attempts_per_model: int


@dataclass(frozen=True)
class ModelConfig:
    """
    Validated model configuration. Routes live only in `routing` (the existing RoutingConfig).
    """

    schema_version: int
    providers: Mapping[str, Provider]
    catalog: Mapping[str, CatalogModel]
    baseline_model: str
    limits: Limits
    routing: RoutingConfig
    config_version: str


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """SafeLoader that rejects repeated mapping keys instead of silently keeping the last one."""


def _construct_unique_mapping(loader: _UniqueKeySafeLoader, node: yaml.MappingNode, deep: bool = False) -> Any:
    seen = set()
    for key_node, _ in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            continue
        key = loader.construct_object(key_node, deep=deep)
        try:
            is_duplicate = key in seen
        except TypeError:
            continue  # Unhashable key: construct_mapping raises its own error below.
        if is_duplicate:
            raise yaml.constructor.ConstructorError(None, None, f"duplicate key {key!r}", key_node.start_mark)
        seen.add(key)
    return loader.construct_mapping(node, deep=deep)


_UniqueKeySafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def load_model_config(
    path: Path | str = DEFAULT_CONFIG_PATH,
    *,
    confidence_threshold: float = RoutingConfig.confidence_threshold,
    difficulty_threshold: float = RoutingConfig.difficulty_threshold,
    reasoning_threshold: float = RoutingConfig.reasoning_threshold,
) -> ModelConfig:
    """
    Reads, validates and returns the model configuration.

    Raises:
        ModelConfigError: If the file is unreadable, is not valid YAML, or fails validation.
            The message starts with the file path and names the offending key.
    """
    path = Path(path)
    thresholds = {
        "confidence_threshold": confidence_threshold,
        "difficulty_threshold": difficulty_threshold,
        "reasoning_threshold": reasoning_threshold,
    }
    raw = _read_yaml(path)
    try:
        return _build(raw, thresholds)
    except ModelConfigError as exc:
        raise ModelConfigError(f"{path}: {exc}") from exc


def _read_yaml(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ModelConfigError(f"{path}: cannot read model config ({type(exc).__name__})") from exc
    try:
        return yaml.load(text, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise ModelConfigError(f"{path}: invalid YAML: {exc}") from exc


def _build(raw: Any, thresholds: Dict[str, float]) -> ModelConfig:
    document = _mapping(raw, "top level")
    schema_version = _schema_version(_require(document, "schema_version", ""))
    providers = _parse_providers(_require(document, "providers", ""))
    catalog = _parse_catalog(_require(document, "catalog", ""), providers)
    limits = _parse_limits(_require(document, "limits", ""))
    baseline_model = _string(_require(document, "baseline_model", ""), "baseline_model")
    if baseline_model not in catalog:
        raise ModelConfigError(f"baseline_model: unknown model {baseline_model!r}; not in catalog")
    routes = _parse_routes(_require(document, "routes", ""), catalog, limits)

    routing = RoutingConfig(
        models=routes,
        general_strong_chain=list(routes[GENERAL_TASK]["strong"]),
        **thresholds,
    )
    version = _config_version(
        {
            "schema_version": schema_version,
            "providers": {name: asdict(provider) for name, provider in providers.items()},
            "catalog": {model_id: asdict(model) for model_id, model in catalog.items()},
            "routes": routes,
            "baseline_model": baseline_model,
            "limits": asdict(limits),
            "thresholds": thresholds,
        }
    )
    return ModelConfig(
        schema_version=schema_version,
        providers=MappingProxyType(providers),
        catalog=MappingProxyType(catalog),
        baseline_model=baseline_model,
        limits=limits,
        routing=routing,
        config_version=version,
    )


def _schema_version(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != SUPPORTED_SCHEMA_VERSION:
        raise ModelConfigError(f"schema_version: must be the integer {SUPPORTED_SCHEMA_VERSION}")
    return value


def _parse_providers(raw: Any) -> Dict[str, Provider]:
    providers = _mapping(raw, "providers")
    if not providers:
        raise ModelConfigError("providers: must define at least one provider")
    parsed = {}
    for name, entry in providers.items():
        where = f"providers.{name}"
        entry = _mapping(entry, where)
        base_url = _string(_require(entry, "base_url", where), f"{where}.base_url")
        if not base_url.startswith("https://"):
            raise ModelConfigError(f"{where}.base_url: must be an https:// URL")
        api_key_env = _string(_require(entry, "api_key_env", where), f"{where}.api_key_env")
        if not ENV_VAR_NAME.fullmatch(api_key_env):
            raise ModelConfigError(
                f"{where}.api_key_env: must be an environment variable name such as AI_GATEWAY_API_KEY, not a key value"
            )
        parsed[str(name)] = Provider(base_url=base_url, api_key_env=api_key_env)
    return parsed


def _parse_catalog(raw: Any, providers: Mapping[str, Provider]) -> Dict[str, CatalogModel]:
    catalog = _mapping(raw, "catalog")
    if not catalog:
        raise ModelConfigError("catalog: must define at least one model")
    parsed = {}
    for model_id, entry in catalog.items():
        where = f"catalog.{model_id}"
        model_id = _string(model_id, where)
        entry = _mapping(entry, where)
        provider = _string(_require(entry, "provider", where), f"{where}.provider")
        if provider not in providers:
            raise ModelConfigError(f"{where}.provider: unknown provider {provider!r}")
        free_tier_eligible = _require(entry, "free_tier_eligible", where)
        if not isinstance(free_tier_eligible, bool):
            raise ModelConfigError(f"{where}.free_tier_eligible: must be true or false")
        model = CatalogModel(
            provider=provider,
            vendor=_string(_require(entry, "vendor", where), f"{where}.vendor"),
            free_tier_eligible=free_tier_eligible,
            input_price=_price(_require(entry, "input_price", where), f"{where}.input_price"),
            output_price=_price(_require(entry, "output_price", where), f"{where}.output_price"),
            context_window=_positive_int(_require(entry, "context_window", where), f"{where}.context_window"),
            max_output_tokens=_positive_int(_require(entry, "max_output_tokens", where), f"{where}.max_output_tokens"),
        )
        if model.max_output_tokens > model.context_window:
            raise ModelConfigError(f"{where}.max_output_tokens: must not exceed context_window")
        parsed[model_id] = model
    return parsed


def _parse_limits(raw: Any) -> Limits:
    limits = _mapping(raw, "limits")
    return Limits(
        max_chain_length=_positive_int(_require(limits, "max_chain_length", "limits"), "limits.max_chain_length"),
        max_attempts_per_model=_positive_int(
            _require(limits, "max_attempts_per_model", "limits"), "limits.max_attempts_per_model"
        ),
    )


def _parse_routes(raw: Any, catalog: Mapping[str, CatalogModel], limits: Limits) -> Routes:
    routes = _mapping(raw, "routes")
    parsed: Routes = {}
    for task, tiers in routes.items():
        where = f"routes.{task}"
        if task not in TASK_TYPES:
            raise ModelConfigError(f"{where}: unknown task type; expected one of {sorted(TASK_TYPES)}")
        tiers = _mapping(tiers, where)
        if not tiers:
            raise ModelConfigError(f"{where}: must define at least one tier")
        parsed[task] = {}
        for tier, chain in tiers.items():
            if tier not in TIERS:
                raise ModelConfigError(f"{where}.{tier}: unknown tier; expected one of {list(TIERS)}")
            parsed[task][tier] = _parse_chain(chain, f"{where}.{tier}", catalog, limits)

    general = parsed.get(GENERAL_TASK, {})
    for tier in TIERS:
        if tier not in general:
            raise ModelConfigError(f"routes.{GENERAL_TASK}.{tier}: missing; required as the fallback chain")

    cheap = {model for tiers in parsed.values() for model in tiers.get("cheap", [])}
    strong = {model for tiers in parsed.values() for model in tiers.get("strong", [])}
    overlap = sorted(cheap & strong)
    if overlap:
        raise ModelConfigError(
            f"routes: models {overlap} appear in both cheap and strong chains; tiers must not share models"
        )
    return parsed


def _parse_chain(chain: Any, where: str, catalog: Mapping[str, CatalogModel], limits: Limits) -> List[str]:
    if not isinstance(chain, list) or not chain:
        raise ModelConfigError(f"{where}: must be a non-empty list of model ids")
    if len(chain) > limits.max_chain_length:
        raise ModelConfigError(
            f"{where}: {len(chain)} models exceeds limits.max_chain_length={limits.max_chain_length}"
        )
    for index, model in enumerate(chain):
        model = _string(model, f"{where}[{index}]")
        if model not in catalog:
            raise ModelConfigError(f"{where}[{index}]: unknown model {model!r}; not in catalog")
    duplicates = sorted({model for model in chain if chain.count(model) > 1})
    if duplicates:
        raise ModelConfigError(f"{where}: duplicate models {duplicates}")
    return list(chain)


def _config_version(normalized: Dict[str, Any]) -> str:
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _require(mapping: Mapping[Any, Any], key: str, where: str) -> Any:
    if key not in mapping:
        location = f"{where}.{key}" if where else key
        raise ModelConfigError(f"{location}: missing")
    return mapping[key]


def _mapping(value: Any, where: str) -> Dict[Any, Any]:
    if not isinstance(value, dict):
        raise ModelConfigError(f"{where}: must be a mapping")
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelConfigError(f"{where}: must be a non-empty string")
    return value


def _price(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ModelConfigError(f"{where}: must be a finite number >= 0")
    return float(value)


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelConfigError(f"{where}: must be a positive integer")
    return value
