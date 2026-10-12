import copy
import json
from pathlib import Path

import pytest

from app.core.classifier import (
    JevDecision,
    JevMalformedResponseError,
    get_default_questions,
    parse_jev_response,
)
from app.core.policy import RoutingConfig, decide_route

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    with open(FIXTURES_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


def test_parse_live_fixture_successfully():
    payload = load_fixture("jev_response_success.json")
    decision = parse_jev_response(payload)

    assert isinstance(decision, JevDecision)
    assert decision.task_type == "code"
    assert decision.task_type_probs["code"] == 1.0
    # On 1-5 scale per technical requirements doc: 0.01 -> 1.01
    assert round(decision.difficulty, 2) == 1.01
    assert round(decision.raw_difficulty, 2) == 0.01
    assert decision.needs_reasoning_prob == 0.19
    # min(1.0, 0.99) = 0.99
    assert decision.confidence == 0.99
    assert decision.jev_model == "jev-1.13.0"
    assert decision.jev_latency_ms == 1434
    assert decision.jev_input_tokens == 446
    assert decision.credits_charged == 1
    assert decision.credits_remaining == 199


def test_parse_synthetic_low_confidence_fixture():
    # Synthetic fixture (Task 0.8): same schema as the live one, values hand-written.
    payload = load_fixture("jev_response_low_confidence.synthetic.json")
    decision = parse_jev_response(payload)

    assert decision.task_type == "math"
    assert decision.task_type_probs["math"] == 0.55
    # min(0.55, 0.45) = 0.45
    assert decision.confidence == 0.45
    # 0-4 expected value 3.2 -> 4.2 on the 1-5 scale
    assert round(decision.raw_difficulty, 2) == 3.2
    assert round(decision.difficulty, 2) == 4.2
    assert decision.needs_reasoning_prob == 0.82
    assert decision.jev_model is None
    assert decision.jev_latency_ms is None
    assert decision.jev_input_tokens is None
    assert decision.credits_charged is None
    assert decision.credits_remaining is None


def test_synthetic_fixture_routes_low_confidence():
    payload = load_fixture("jev_response_low_confidence.synthetic.json")
    config = RoutingConfig(
        models={
            "math": {"cheap": ["math-cheap"], "strong": ["math-strong"]},
            "general": {"cheap": ["general-cheap"], "strong": ["general-strong"]},
        },
        general_strong_chain=["general-strong"],
    )

    route = decide_route(parse_jev_response(payload), config)

    # Low confidence wins over HARD (difficulty 4.2, reasoning 0.82): strong general chain.
    assert route.tier == "strong"
    assert route.task_type == "general"
    assert route.reason == "LOW_CONFIDENCE"
    assert route.model_chain == ["general-strong"]
    assert route.confidence == 0.45


def test_raw_0_to_4_scale_option():
    payload = load_fixture("jev_response_success.json")
    decision = parse_jev_response(payload, difficulty_scale="0-4")

    assert round(decision.difficulty, 2) == 0.01
    assert round(decision.raw_difficulty, 2) == 0.01


def test_confidence_is_minimum_of_task_and_difficulty():
    payload = load_fixture("jev_response_success.json")
    # Alter confidences to test min logic
    payload["data"]["answers"]["task_type"]["confidence"] = 0.72
    payload["data"]["answers"]["difficulty"]["confidence"] = 0.88
    decision = parse_jev_response(payload)
    assert decision.confidence == 0.72

    payload["data"]["answers"]["task_type"]["confidence"] = 0.91
    payload["data"]["answers"]["difficulty"]["confidence"] = 0.65
    decision_2 = parse_jev_response(payload)
    assert decision_2.confidence == 0.65


def test_missing_data_envelope_raises_malformed_error():
    with pytest.raises(JevMalformedResponseError, match="Missing 'data' in response"):
        parse_jev_response({"code": 0, "message": "ok"})


def test_missing_task_type_raises_malformed_error():
    payload = load_fixture("jev_response_success.json")
    del payload["data"]["answers"]["task_type"]
    with pytest.raises(JevMalformedResponseError, match="Missing 'task_type' answer"):
        parse_jev_response(payload)


def test_missing_difficulty_raises_malformed_error():
    payload = load_fixture("jev_response_success.json")
    del payload["data"]["answers"]["difficulty"]
    with pytest.raises(JevMalformedResponseError, match="Missing 'difficulty' answer"):
        parse_jev_response(payload)


def test_missing_needs_reasoning_raises_malformed_error():
    payload = load_fixture("jev_response_success.json")
    del payload["data"]["answers"]["needs_reasoning"]
    with pytest.raises(JevMalformedResponseError, match="Missing 'needs_reasoning' answer"):
        parse_jev_response(payload)


def test_malformed_choice_field_raises_error():
    payload = load_fixture("jev_response_success.json")
    payload["data"]["answers"]["task_type"]["choice"] = None
    with pytest.raises(JevMalformedResponseError, match="Invalid or missing 'choice'"):
        parse_jev_response(payload)


def test_malformed_difficulty_score_raises_error():
    payload = load_fixture("jev_response_success.json")
    payload["data"]["answers"]["difficulty"]["score"] = "not-a-number"
    with pytest.raises(JevMalformedResponseError, match="Invalid difficulty score"):
        parse_jev_response(payload)


def test_malformed_noul_raises_error():
    payload = load_fixture("jev_response_success.json")
    payload["data"]["answers"]["needs_reasoning"]["noul"] = 1.5  # probability must be 0-1
    with pytest.raises(JevMalformedResponseError, match="Invalid noul probability"):
        parse_jev_response(payload)


def test_get_default_questions_structure():
    questions = get_default_questions()
    assert "task_type" in questions
    assert questions["task_type"]["type"] == "choice"
    assert set(questions["task_type"]["criteria"].keys()) == {"code", "math", "writing", "general"}

    assert "difficulty" in questions
    assert questions["difficulty"]["type"] == "score"
    assert questions["difficulty"]["criteria"] == ["trivial", "easy", "moderate", "hard", "expert"]

    assert "needs_reasoning" in questions
    assert questions["needs_reasoning"]["type"] == "noul"


def _set(payload: dict, path: tuple, value) -> dict:
    target = payload["data"]
    for key in path[:-1]:
        target = target.setdefault(key, {})
    target[path[-1]] = value
    return payload


MALFORMED_VALUES = [
    pytest.param(
        ("answers", "task_type", "probabilities"), {"code": None}, "task_type.probabilities", id="task-prob-null"
    ),
    pytest.param(
        ("answers", "task_type", "probabilities"), {"code": [1]}, "task_type.probabilities", id="task-prob-list"
    ),
    pytest.param(
        ("answers", "task_type", "probabilities"), {"code": "abc"}, "task_type.probabilities", id="task-prob-text"
    ),
    pytest.param(
        ("answers", "difficulty", "probabilities"), {"0": None}, "difficulty.probabilities", id="diff-prob-null"
    ),
    pytest.param(("latency_ms",), "abc", "latency_ms", id="latency-text"),
    pytest.param(("latency_ms",), [1], "latency_ms", id="latency-list"),
    pytest.param(("usage", "input_tokens"), {}, "usage.input_tokens", id="input-tokens-object"),
    pytest.param(("credits", "charged"), "x", "credits.charged", id="credits-charged-text"),
    pytest.param(("credits", "remaining"), ["x"], "credits.remaining", id="credits-remaining-list"),
]


@pytest.mark.parametrize(("path", "value", "field"), MALFORMED_VALUES)
def test_malformed_numeric_field_raises_malformed_error_naming_only_the_field(path, value, field):
    payload = _set(load_fixture("jev_response_success.json"), path, value)

    with pytest.raises(JevMalformedResponseError) as excinfo:
        parse_jev_response(payload)

    assert field in str(excinfo.value)
    assert "abc" not in str(excinfo.value)
    assert excinfo.value.__cause__ is None


# --- Difficulty: expected value of difficulty.probabilities (docs/05 1.4, TRD 4) ----------------


def with_distribution(dist, score=None) -> dict:
    payload = load_fixture("jev_response_success.json")
    payload["data"]["answers"]["difficulty"]["probabilities"] = dist
    if score is not None:
        payload["data"]["answers"]["difficulty"]["score"] = score
    return payload


def test_expected_difficulty_from_distribution():
    # 0*0.1 + 1*0.2 + 2*0.3 + 3*0.25 + 4*0.15 = 2.15 -> 3.15 on the 1-5 scale
    decision = parse_jev_response(with_distribution({"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.25, "4": 0.15}))

    assert decision.difficulty == pytest.approx(3.15)


@pytest.mark.parametrize(
    ("name", "expected"),
    [("jev_response_success.json", 1.01), ("jev_response_low_confidence.synthetic.json", 4.2)],
    ids=["live-fixture", "synthetic-fixture"],
)
def test_fixture_difficulty_is_one_plus_expected_level(name, expected):
    payload = load_fixture(name)
    dist = payload["data"]["answers"]["difficulty"]["probabilities"]

    decision = parse_jev_response(payload)

    assert decision.difficulty == pytest.approx(expected)
    assert decision.difficulty == pytest.approx(1 + sum(int(k) * v for k, v in dist.items()) / sum(dist.values()))
    assert decision.difficulty_dist == {k: float(v) for k, v in dist.items()}


def test_distribution_wins_when_it_disagrees_with_the_score():
    decision = parse_jev_response(with_distribution({"0": 0, "1": 0, "2": 0, "3": 0, "4": 1.0}, score=0.01))

    assert decision.difficulty == pytest.approx(5.0)
    assert decision.raw_difficulty == 0.01  # the original score, kept for diagnostics only


def test_raw_difficulty_is_the_unmodified_score():
    decision = parse_jev_response(with_distribution({"0": 0, "1": 0, "2": 1.0, "3": 0, "4": 0}, score=2.7))

    assert decision.raw_difficulty == 2.7
    assert decision.difficulty == pytest.approx(3.0)


@pytest.mark.parametrize(
    "dist",
    [
        {"0": 1.0, "1": 0, "2": 0, "3": 0, "4": 1.0},  # sums to 2
        {"0": 0.49, "1": 0, "2": 0, "3": 0, "4": 0.49},  # rounded, sums to 0.98
        {"0": 1, "1": 0, "2": 0, "3": 0, "4": 1},  # integers
    ],
    ids=["sum-2", "sum-0.98", "integers"],
)
def test_probabilities_are_normalized_by_their_total(dist):
    # Equal weight on levels 0 and 4: expected level 2 -> 3.0 whatever the total
    assert parse_jev_response(with_distribution(dist)).difficulty == pytest.approx(3.0)


def test_distribution_is_preserved_with_float_values():
    dist = {"0": 0, "1": 1, "2": 0, "3": 0, "4": 0}

    assert parse_jev_response(with_distribution(dist)).difficulty_dist == {
        "0": 0.0,
        "1": 1.0,
        "2": 0.0,
        "3": 0.0,
        "4": 0.0,
    }


def test_0_to_4_scale_option_uses_the_expected_level_without_the_offset():
    payload = with_distribution({"0": 0, "1": 0, "2": 0.5, "3": 0.5, "4": 0}, score=0.01)

    decision = parse_jev_response(payload, difficulty_scale="0-4")

    assert decision.difficulty == pytest.approx(2.5)
    assert decision.raw_difficulty == 0.01


VALID = {"0": 0.2, "1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2}


def without(key):
    dist = dict(VALID)
    del dist[key]
    return dist


INVALID_DISTRIBUTIONS = [
    pytest.param("absent", id="missing-field"),
    pytest.param(None, id="null"),
    pytest.param([0.2, 0.2, 0.2, 0.2, 0.2], id="list"),
    pytest.param({}, id="empty"),
    pytest.param(without("4"), id="missing-key-4"),
    pytest.param(without("0"), id="missing-key-0"),
    pytest.param({**VALID, "5": 0.0}, id="extra-key"),
    pytest.param({"trivial": 0.2, "easy": 0.2, "moderate": 0.2, "hard": 0.2, "expert": 0.2}, id="legend-keys"),
    pytest.param({**VALID, "2": "0.2"}, id="numeric-string"),
    pytest.param({**VALID, "2": "abc"}, id="text"),
    pytest.param({**VALID, "2": None}, id="none"),
    pytest.param({**VALID, "2": True}, id="bool"),
    pytest.param({**VALID, "2": [0.2]}, id="list-value"),
    pytest.param({**VALID, "2": -0.1}, id="negative"),
    pytest.param({**VALID, "2": float("nan")}, id="nan"),
    pytest.param({**VALID, "2": float("inf")}, id="inf"),
    pytest.param({"0": 0, "1": 0, "2": 0, "3": 0, "4": 0}, id="zero-total"),
    pytest.param({"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0}, id="zero-total-floats"),
]


@pytest.mark.parametrize("dist", INVALID_DISTRIBUTIONS)
def test_malformed_distribution_is_rejected_without_falling_back_to_the_score(dist):
    payload = load_fixture("jev_response_success.json")
    if dist == "absent":
        del payload["data"]["answers"]["difficulty"]["probabilities"]
    else:
        payload["data"]["answers"]["difficulty"]["probabilities"] = copy.deepcopy(dist)

    with pytest.raises(JevMalformedResponseError) as excinfo:
        parse_jev_response(payload)

    message = str(excinfo.value)
    assert "difficulty.probabilities" in message
    assert "abc" not in message and "0.2" not in message  # field named, value never echoed
    assert excinfo.value.__cause__ is None


@pytest.mark.parametrize(
    ("dist", "tier", "reason"),
    [
        ({"0": 0, "1": 0, "2": 0.5, "3": 0.5, "4": 0}, "strong", "HARD"),  # exactly 3.5
        ({"0": 0, "1": 0, "2": 0.51, "3": 0.49, "4": 0}, "cheap", "EASY"),  # 3.49
    ],
    ids=["at-3.5", "below-3.5"],
)
def test_expected_difficulty_drives_the_3_5_threshold(dist, tier, reason):
    # Score deliberately disagrees, so only the distribution can produce this result.
    payload = with_distribution(dist, score=0.01)
    payload["data"]["answers"]["needs_reasoning"]["noul"] = 0.1

    config = RoutingConfig(
        models={"general": {"cheap": ["general-cheap"], "strong": ["general-strong"]}},
        general_strong_chain=["general-strong"],
    )

    decision = parse_jev_response(payload)
    route = decide_route(decision, config, budget_available=True)

    assert (route.tier, route.reason) == (tier, reason)


def test_numeric_strings_that_parsed_before_still_parse():
    payload = load_fixture("jev_response_success.json")
    payload["data"]["answers"]["task_type"]["probabilities"] = {"code": "0.9", "math": 0.1}
    payload["data"]["latency_ms"] = "1434"

    decision = parse_jev_response(payload)

    assert decision.task_type_probs == {"code": 0.9, "math": 0.1}
    assert decision.jev_latency_ms == 1434
