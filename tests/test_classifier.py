import json
from pathlib import Path

import pytest

from app.core.classifier import (
    JevDecision,
    JevMalformedResponseError,
    get_default_questions,
    parse_jev_response,
)

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
