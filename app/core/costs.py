"""
Cost, baseline and savings estimates for one model call (Task S1.8, FR-9).

All values are list-price estimates in USD from the catalog in config/models.yaml (prices per million
tokens). They cover the model call that produced the answer only: Jev is excluded because it is paid
in separate credits, and failed or timed-out attempts report no token usage. The baseline prices the
same token counts at the configured baseline (always-strong) model.

Savings are signed (baseline - actual): a model pricier than the baseline gives negative savings.
A cost that cannot be calculated raises CostError; it is never reported as zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.model_config import CatalogModel, ModelConfig

COST_BASIS = "list_price_estimate"
TOKENS_PER_PRICE_UNIT = 1_000_000


class CostError(ValueError):
    """The cost cannot be calculated (unknown model or invalid token usage)."""


@dataclass(frozen=True)
class CostEstimate:
    model: str
    input_tokens: int
    output_tokens: int
    est_cost_usd: float
    baseline_model: str
    baseline_cost_usd: float
    saved_usd: float  # baseline_cost_usd - est_cost_usd; negative when the model costs more
    basis: str = COST_BASIS
    includes_jev: bool = False


def estimate_cost(config: ModelConfig, model: str, input_tokens: Any, output_tokens: Any) -> CostEstimate:
    """
    Raises:
        CostError: If either model is not in the catalog, or a token count is missing or not an int >= 0.
    """
    input_tokens = _token_count(input_tokens, "input_tokens")
    output_tokens = _token_count(output_tokens, "output_tokens")
    actual = _call_cost(_catalog_entry(config, model), input_tokens, output_tokens)
    baseline = _call_cost(_catalog_entry(config, config.baseline_model), input_tokens, output_tokens)
    return CostEstimate(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        est_cost_usd=actual,
        baseline_model=config.baseline_model,
        baseline_cost_usd=baseline,
        saved_usd=baseline - actual,
    )


def _call_cost(entry: CatalogModel, input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * entry.input_price + output_tokens * entry.output_price) / TOKENS_PER_PRICE_UNIT


def _catalog_entry(config: ModelConfig, model: str) -> CatalogModel:
    entry = config.catalog.get(model)
    if entry is None:
        raise CostError(f"unknown model {model!r}; not in catalog")
    return entry


def _token_count(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CostError(f"{field} must be an integer >= 0")
    return value
