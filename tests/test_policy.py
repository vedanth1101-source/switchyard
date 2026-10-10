import pytest

from app.core.classifier import JevDecision
from app.core.policy import (
    RoutingConfig,
    RoutingConfigError,
    decide_route,
)


@pytest.fixture
def default_config():
    return RoutingConfig(
        confidence_threshold=0.70,
        difficulty_threshold=3.5,
        reasoning_threshold=0.60,
        models={
            "code": {
                "cheap": ["deepseek-coder-6.7b"],
                "strong": ["claude-3-5-sonnet", "gpt-4o"],
            },
            "math": {
                "cheap": ["qwen-math-7b"],
                "strong": ["gpt-4o", "claude-3-5-sonnet"],
            },
            "writing": {
                "cheap": ["llama-3-8b-instruct"],
                "strong": ["claude-3-5-sonnet"],
            },
            "general": {
                "cheap": ["llama-3-8b-instruct"],
                "strong": ["gpt-4o", "claude-3-5-sonnet"],
            },
        },
        general_strong_chain=["gpt-4o", "claude-3-5-sonnet"],
    )


def make_decision(
    task_type: str = "code",
    difficulty: float = 1.01,
    needs_reasoning_prob: float = 0.19,
    confidence: float = 0.99,
) -> JevDecision:
    return JevDecision(
        task_type=task_type,
        task_type_probs={task_type: 1.0},
        difficulty=difficulty,
        raw_difficulty=difficulty - 1.0 if difficulty >= 1.0 else difficulty,
        difficulty_dist={"0": 0.99, "1": 0.01},
        needs_reasoning_prob=needs_reasoning_prob,
        confidence=confidence,
        jev_model="jev-1.13.0",
        jev_latency_ms=1434,
        jev_input_tokens=446,
        credits_charged=1,
        credits_remaining=199,
    )


def test_easy_code_routes_cheap_task_model(default_config):
    decision = make_decision(task_type="code", difficulty=1.01, needs_reasoning_prob=0.19, confidence=0.99)
    route = decide_route(decision, default_config, budget_available=True)

    assert route.tier == "cheap"
    assert route.task_type == "code"
    assert route.reason == "EASY"
    assert route.model_chain == ["deepseek-coder-6.7b"]


def test_high_difficulty_routes_strong_task_model(default_config):
    decision = make_decision(task_type="code", difficulty=4.2, needs_reasoning_prob=0.2, confidence=0.95)
    route = decide_route(decision, default_config, budget_available=True)

    assert route.tier == "strong"
    assert route.task_type == "code"
    assert route.reason == "HARD"
    assert route.model_chain == ["claude-3-5-sonnet", "gpt-4o"]


def test_high_reasoning_prob_routes_strong_task_model(default_config):
    decision = make_decision(task_type="math", difficulty=2.0, needs_reasoning_prob=0.85, confidence=0.90)
    route = decide_route(decision, default_config, budget_available=True)

    assert route.tier == "strong"
    assert route.task_type == "math"
    assert route.reason == "HARD"
    assert route.model_chain == ["gpt-4o", "claude-3-5-sonnet"]


def test_threshold_exact_equality_routes_strong(default_config):
    # difficulty == 3.5 exactly
    decision_diff = make_decision(task_type="writing", difficulty=3.5, needs_reasoning_prob=0.1, confidence=0.8)
    route_diff = decide_route(decision_diff, default_config, budget_available=True)
    assert route_diff.tier == "strong"
    assert route_diff.reason == "HARD"

    # reasoning == 0.60 exactly
    decision_reason = make_decision(task_type="writing", difficulty=2.0, needs_reasoning_prob=0.60, confidence=0.8)
    route_reason = decide_route(decision_reason, default_config, budget_available=True)
    assert route_reason.tier == "strong"
    assert route_reason.reason == "HARD"


def test_low_confidence_routes_strong_general(default_config):
    decision = make_decision(task_type="code", difficulty=1.5, needs_reasoning_prob=0.1, confidence=0.69)
    route = decide_route(decision, default_config, budget_available=True)

    assert route.tier == "strong"
    assert route.task_type == "general"
    assert route.reason == "LOW_CONFIDENCE"
    assert route.model_chain == default_config.general_strong_chain


def test_confidence_threshold_exact_equality(default_config):
    # confidence == 0.70 exactly is considered confident
    decision = make_decision(task_type="code", difficulty=1.5, needs_reasoning_prob=0.1, confidence=0.70)
    route = decide_route(decision, default_config, budget_available=True)

    assert route.tier == "cheap"
    assert route.reason == "EASY"


def test_jev_error_routes_strong_general(default_config):
    # Decision is None (Jev API error)
    route_none = decide_route(None, default_config, budget_available=True, jev_error="Timeout connecting to Jev")
    assert route_none.tier == "strong"
    assert route_none.task_type == "general"
    assert route_none.reason == "JEV_ERROR"
    assert route_none.model_chain == default_config.general_strong_chain

    # Decision present but jev_error explicitly provided
    decision = make_decision()
    route_err = decide_route(decision, default_config, budget_available=True, jev_error="429 Rate Limited")
    assert route_err.tier == "strong"
    assert route_err.reason == "JEV_ERROR"


def test_budget_exhausted_downgrades_strong_to_cheap(default_config):
    decision = make_decision(task_type="code", difficulty=4.5, needs_reasoning_prob=0.9, confidence=0.95)
    route = decide_route(decision, default_config, budget_available=False)

    assert route.tier == "cheap"
    assert route.task_type == "code"
    assert route.reason == "BUDGET"
    assert route.model_chain == ["deepseek-coder-6.7b"]


def test_budget_exhausted_does_not_affect_cheap_routes(default_config):
    decision = make_decision(task_type="code", difficulty=1.2, needs_reasoning_prob=0.1, confidence=0.95)
    route = decide_route(decision, default_config, budget_available=False)

    assert route.tier == "cheap"
    assert route.task_type == "code"
    assert route.reason == "EASY"
    assert route.model_chain == ["deepseek-coder-6.7b"]


def test_all_task_types_route_appropriate_chains(default_config):
    for task in ["code", "math", "writing", "general"]:
        dec_cheap = make_decision(task_type=task, difficulty=1.5, needs_reasoning_prob=0.1, confidence=0.9)
        route_cheap = decide_route(dec_cheap, default_config, budget_available=True)
        assert route_cheap.tier == "cheap"
        assert route_cheap.model_chain == default_config.models[task]["cheap"]

        dec_strong = make_decision(task_type=task, difficulty=4.0, needs_reasoning_prob=0.1, confidence=0.9)
        route_strong = decide_route(dec_strong, default_config, budget_available=True)
        assert route_strong.tier == "strong"
        assert route_strong.model_chain == default_config.models[task]["strong"]


def test_policy_is_pure_and_deterministic(default_config):
    decision = make_decision(task_type="math", difficulty=3.8, needs_reasoning_prob=0.7, confidence=0.88)
    res1 = decide_route(decision, default_config, budget_available=True)
    res2 = decide_route(decision, default_config, budget_available=True)

    assert res1 == res2


def with_default_threshold(config: RoutingConfig) -> RoutingConfig:
    """Same model chains as `config`, but with every threshold left at its RoutingConfig default."""
    return RoutingConfig(models=config.models, general_strong_chain=config.general_strong_chain)


def test_default_confidence_threshold_is_0_6():
    assert RoutingConfig().confidence_threshold == 0.6


def test_confidence_below_default_threshold_routes_low_confidence(default_config):
    config = with_default_threshold(default_config)
    decision = make_decision(task_type="code", difficulty=1.5, needs_reasoning_prob=0.1, confidence=0.59)
    route = decide_route(decision, config, budget_available=True)

    assert route.tier == "strong"
    assert route.task_type == "general"
    assert route.reason == "LOW_CONFIDENCE"


def test_confidence_equal_to_default_threshold_passes_confidence_check(default_config):
    config = with_default_threshold(default_config)
    decision = make_decision(task_type="code", difficulty=1.5, needs_reasoning_prob=0.1, confidence=0.6)
    route = decide_route(decision, config, budget_available=True)

    assert route.tier == "cheap"
    assert route.reason == "EASY"


def test_budget_exhausted_downgrades_low_confidence_to_cheap_general(default_config):
    decision = make_decision(task_type="code", difficulty=1.5, needs_reasoning_prob=0.1, confidence=0.5)
    route = decide_route(decision, default_config, budget_available=False)

    assert route.tier == "cheap"
    assert route.task_type == "general"
    assert route.reason == "BUDGET"
    assert route.model_chain == default_config.models["general"]["cheap"]
    assert route.confidence == 0.5
    assert route.difficulty == 1.5
    assert route.needs_reasoning_prob == 0.1


def test_budget_exhausted_downgrades_jev_error_to_cheap_general(default_config):
    route = decide_route(None, default_config, budget_available=False, jev_error="Timeout connecting to Jev")

    assert route.tier == "cheap"
    assert route.task_type == "general"
    assert route.reason == "BUDGET"
    assert route.model_chain == default_config.models["general"]["cheap"]
    assert route.jev_error == "Timeout connecting to Jev"


def test_budget_exhausted_hard_route_keeps_task_chain_and_diagnostics(default_config):
    decision = make_decision(task_type="math", difficulty=4.5, needs_reasoning_prob=0.9, confidence=0.95)
    route = decide_route(decision, default_config, budget_available=False)

    assert route.tier == "cheap"
    assert route.task_type == "math"
    assert route.reason == "BUDGET"
    assert route.model_chain == default_config.models["math"]["cheap"]
    assert route.confidence == 0.95
    assert route.difficulty == 4.5
    assert route.needs_reasoning_prob == 0.9
    assert route.jev_error is None


STRONG_CHAIN = ["gpt-4o", "claude-3-5-sonnet"]
GENERAL_CHEAP_CHAIN = ["llama-3-8b-instruct"]


def test_missing_task_cheap_chain_falls_back_to_general_cheap():
    config = RoutingConfig(
        models={"code": {"strong": STRONG_CHAIN}, "general": {"cheap": GENERAL_CHEAP_CHAIN}},
        general_strong_chain=STRONG_CHAIN,
    )
    route = decide_route(make_decision(task_type="code", difficulty=1.5), config, budget_available=True)

    assert route.tier == "cheap"
    assert route.reason == "EASY"
    assert route.model_chain == GENERAL_CHEAP_CHAIN


def test_empty_task_cheap_chain_falls_back_to_general_cheap():
    config = RoutingConfig(
        models={"code": {"cheap": [], "strong": STRONG_CHAIN}, "general": {"cheap": GENERAL_CHEAP_CHAIN}},
        general_strong_chain=STRONG_CHAIN,
    )
    route = decide_route(make_decision(task_type="code", difficulty=1.5), config, budget_available=True)

    assert route.tier == "cheap"
    assert route.model_chain == GENERAL_CHEAP_CHAIN


def test_unknown_task_type_cheap_route_uses_general_cheap(default_config):
    route = decide_route(make_decision(task_type="poetry", difficulty=1.5), default_config, budget_available=True)

    assert route.tier == "cheap"
    assert route.task_type == "poetry"
    assert route.reason == "EASY"
    assert route.model_chain == default_config.models["general"]["cheap"]


def test_budget_downgrade_with_missing_task_cheap_chain_uses_general_cheap():
    config = RoutingConfig(
        models={"code": {"strong": STRONG_CHAIN}, "general": {"cheap": GENERAL_CHEAP_CHAIN}},
        general_strong_chain=STRONG_CHAIN,
    )
    decision = make_decision(task_type="code", difficulty=4.5, needs_reasoning_prob=0.9, confidence=0.95)
    route = decide_route(decision, config, budget_available=False)

    assert route.tier == "cheap"
    assert route.reason == "BUDGET"
    assert route.model_chain == GENERAL_CHEAP_CHAIN


def test_no_cheap_chain_configured_raises_routing_config_error():
    config = RoutingConfig(models={"code": {"strong": STRONG_CHAIN}}, general_strong_chain=STRONG_CHAIN)

    with pytest.raises(RoutingConfigError):  # EASY
        decide_route(make_decision(task_type="code", difficulty=1.5), config, budget_available=True)
    with pytest.raises(RoutingConfigError):  # HARD downgraded to BUDGET
        decide_route(make_decision(task_type="code", difficulty=4.5), config, budget_available=False)
    with pytest.raises(RoutingConfigError):  # LOW_CONFIDENCE downgraded to BUDGET
        decide_route(make_decision(confidence=0.1), config, budget_available=False)
    with pytest.raises(RoutingConfigError):  # JEV_ERROR downgraded to BUDGET
        decide_route(None, config, budget_available=False, jev_error="Timeout connecting to Jev")


def test_missing_strong_chain_still_falls_back_to_general_strong():
    config = RoutingConfig(
        models={"code": {"cheap": ["deepseek-coder-6.7b"]}, "general": {"cheap": GENERAL_CHEAP_CHAIN}},
        general_strong_chain=STRONG_CHAIN,
    )
    decision = make_decision(task_type="code", difficulty=4.5, needs_reasoning_prob=0.9, confidence=0.95)
    route = decide_route(decision, config, budget_available=True)

    assert route.tier == "strong"
    assert route.reason == "HARD"
    assert route.model_chain == STRONG_CHAIN
