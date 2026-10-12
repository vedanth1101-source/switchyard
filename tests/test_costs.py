import dataclasses
from types import MappingProxyType

import pytest

from app.core.costs import COST_BASIS, CostError, CostEstimate, estimate_cost
from app.core.model_config import CatalogModel, load_model_config

CONFIG = load_model_config()
BASELINE = CONFIG.baseline_model


def priced(input_price: float, output_price: float) -> CatalogModel:
    return CatalogModel(
        provider="ai_gateway",
        vendor="test",
        free_tier_eligible=True,
        input_price=input_price,
        output_price=output_price,
        context_window=100_000,
        max_output_tokens=1_000,
    )


# Round test prices (USD per million tokens) so expected values are easy to read.
TEST_CONFIG = dataclasses.replace(
    CONFIG,
    catalog=MappingProxyType(
        {
            "test/cheap": priced(0.10, 0.40),
            "test/baseline": priced(1.00, 4.00),
            "test/pricier": priced(2.00, 8.00),
        }
    ),
    baseline_model="test/baseline",
)


def test_zero_tokens_cost_nothing():
    estimate = estimate_cost(TEST_CONFIG, "test/cheap", 0, 0)

    assert (estimate.est_cost_usd, estimate.baseline_cost_usd, estimate.saved_usd) == (0.0, 0.0, 0.0)


def test_input_only_cost():
    estimate = estimate_cost(TEST_CONFIG, "test/cheap", 1_000_000, 0)

    assert estimate.est_cost_usd == pytest.approx(0.10)
    assert estimate.baseline_cost_usd == pytest.approx(1.00)


def test_output_only_cost():
    estimate = estimate_cost(TEST_CONFIG, "test/cheap", 0, 500_000)

    assert estimate.est_cost_usd == pytest.approx(0.20)
    assert estimate.baseline_cost_usd == pytest.approx(2.00)


def test_input_and_output_cost_follow_the_per_million_formula():
    estimate = estimate_cost(TEST_CONFIG, "test/cheap", 1_200, 350)

    assert estimate.est_cost_usd == pytest.approx((1_200 * 0.10 + 350 * 0.40) / 1_000_000)
    assert estimate.baseline_cost_usd == pytest.approx((1_200 * 1.00 + 350 * 4.00) / 1_000_000)


def test_baseline_model_has_exactly_zero_savings():
    estimate = estimate_cost(TEST_CONFIG, "test/baseline", 1_234, 567)

    assert estimate.est_cost_usd == estimate.baseline_cost_usd
    assert estimate.saved_usd == 0.0


def test_cheaper_model_has_positive_savings():
    estimate = estimate_cost(TEST_CONFIG, "test/cheap", 1_000, 1_000)

    assert estimate.saved_usd == pytest.approx(estimate.baseline_cost_usd - estimate.est_cost_usd)
    assert estimate.saved_usd > 0


def test_pricier_model_has_negative_savings():
    estimate = estimate_cost(TEST_CONFIG, "test/pricier", 1_000, 1_000)

    assert estimate.saved_usd == pytest.approx(-(2.00 + 8.00 - 1.00 - 4.00) * 1_000 / 1_000_000)
    assert estimate.saved_usd < 0


def test_estimate_records_inputs_and_labels():
    estimate = estimate_cost(TEST_CONFIG, "test/cheap", 10, 20)

    assert estimate == CostEstimate(
        model="test/cheap",
        input_tokens=10,
        output_tokens=20,
        est_cost_usd=estimate.est_cost_usd,
        baseline_model="test/baseline",
        baseline_cost_usd=estimate.baseline_cost_usd,
        saved_usd=estimate.saved_usd,
    )
    assert estimate.basis == COST_BASIS == "list_price_estimate"
    assert estimate.includes_jev is False


@pytest.mark.parametrize("model", list(CONFIG.catalog))
def test_real_catalog_prices_are_used(model):
    entry, baseline = CONFIG.catalog[model], CONFIG.catalog[BASELINE]

    estimate = estimate_cost(CONFIG, model, 2_000, 300)

    assert estimate.est_cost_usd == pytest.approx((2_000 * entry.input_price + 300 * entry.output_price) / 1e6)
    assert estimate.baseline_cost_usd == pytest.approx(
        (2_000 * baseline.input_price + 300 * baseline.output_price) / 1e6
    )
    assert estimate.baseline_model == BASELINE


def test_real_config_has_a_model_pricier_than_the_baseline():
    # Documents why savings must be signed: falling back within the strong chain can cost more.
    baseline = CONFIG.catalog[BASELINE]
    pricier = [
        model
        for model, entry in CONFIG.catalog.items()
        if entry.input_price >= baseline.input_price and entry.output_price > baseline.output_price
    ]

    assert pricier
    assert estimate_cost(CONFIG, pricier[0], 1_000, 1_000).saved_usd < 0


# --- Costs that cannot be calculated ------------------------------------------------------


def test_unknown_model_raises_instead_of_reporting_zero():
    with pytest.raises(CostError, match="unknown model"):
        estimate_cost(TEST_CONFIG, "vendor/not-in-catalog", 10, 10)


def test_unknown_baseline_model_raises():
    config = dataclasses.replace(TEST_CONFIG, baseline_model="vendor/gone")

    with pytest.raises(CostError, match="unknown model"):
        estimate_cost(config, "test/cheap", 10, 10)


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens", "field"),
    [
        (None, 10, "input_tokens"),
        (10, None, "output_tokens"),
        (-1, 10, "input_tokens"),
        (10, -5, "output_tokens"),
        (True, 10, "input_tokens"),
        (1.5, 10, "input_tokens"),
        ("10", 10, "input_tokens"),
    ],
    ids=["input-missing", "output-missing", "input-negative", "output-negative", "bool", "float", "string"],
)
def test_missing_or_invalid_usage_raises_instead_of_reporting_zero(input_tokens, output_tokens, field):
    with pytest.raises(CostError, match=field):
        estimate_cost(TEST_CONFIG, "test/cheap", input_tokens, output_tokens)
