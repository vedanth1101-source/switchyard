"""
Classification and schema normalization for JevStation responses.

Validates the raw JevStation JSON response, computes compound confidence,
maps difficulty levels to the internal representation, and emits a typed JevDecision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.core.jev_client import JevClientError


class JevMalformedResponseError(JevClientError):
    """Raised when JevStation returns an unexpected or incomplete schema."""

    pass


@dataclass(frozen=True)
class JevDecision:
    """
    Normalized, immutable representation of a JevStation routing evaluation.
    """

    task_type: str
    task_type_probs: Dict[str, float]
    difficulty: float  # Normalized scale (default: 1–5 expected value per specification)
    raw_difficulty: float  # Unmodified JevStation scale (0–4 expected value)
    difficulty_dist: Dict[str, float]
    needs_reasoning_prob: float
    confidence: float  # Minimum of task_type and difficulty confidences
    jev_model: Optional[str] = None
    jev_latency_ms: Optional[int] = None
    jev_input_tokens: Optional[int] = None
    credits_charged: Optional[int] = None
    credits_remaining: Optional[int] = None


def get_default_questions() -> Dict[str, Any]:
    """
    Returns the standard 3-question dictionary used for Switchyard prompt routing.
    """
    return {
        "task_type": {
            "type": "choice",
            "instructions": "Classify the user's requested task into exactly one category based on the work required.",
            "criteria": {
                "code": "Programming, software development, code generation, debugging, or script execution",
                "math": "Mathematics, calculation, symbolic logic, proof, or formal problem solving",
                "writing": "Content generation, creative writing, copywriting, editing, or email correspondence",
                "general": "General conversation, facts, broad explanations, or queries that span multiple domains",
            },
        },
        "difficulty": {
            "type": "score",
            "instructions": "How difficult is it to produce a correct and useful answer to the user's prompt?",
            "criteria": [
                "trivial",
                "easy",
                "moderate",
                "hard",
                "expert",
            ],
        },
        "needs_reasoning": {
            "type": "noul",
            "instructions": "Does answering the user's prompt well require multiple dependent reasoning steps?",
        },
    }


def parse_jev_response(data: Dict[str, Any], difficulty_scale: str = "1-5") -> JevDecision:
    """
    Parses and validates a raw JevStation response envelope into a JevDecision.

    Args:
        data: Raw response dict from JevStation POST /api/v1/systemone.
        difficulty_scale: Either "1-5" (converts 0–4 to 1–5 scale per project TRD)
                          or "0-4" (preserves raw score).

    Returns:
        JevDecision: Structured, normalized evaluation metrics.

    Raises:
        JevMalformedResponseError: If required answers or fields are missing/invalid.
    """
    if not isinstance(data, dict) or "data" not in data or not isinstance(data["data"], dict):
        raise JevMalformedResponseError("Missing 'data' in response envelope")

    inner = data["data"]
    answers = inner.get("answers")
    if not isinstance(answers, dict):
        raise JevMalformedResponseError("Missing or invalid 'answers' dictionary in response")

    # 1. Validate task_type (Choice question)
    task_type_ans = answers.get("task_type")
    if not isinstance(task_type_ans, dict):
        raise JevMalformedResponseError("Missing 'task_type' answer in response")

    choice = task_type_ans.get("choice")
    if not isinstance(choice, str) or not choice:
        raise JevMalformedResponseError("Invalid or missing 'choice' string in task_type")

    task_confidence = task_type_ans.get("confidence")
    if not isinstance(task_confidence, (int, float)) or not (0.0 <= task_confidence <= 1.0):
        raise JevMalformedResponseError(f"Invalid confidence value in task_type: {task_confidence}")

    task_probs = task_type_ans.get("probabilities", {})
    if not isinstance(task_probs, dict):
        task_probs = {choice: float(task_confidence)}

    # 2. Validate difficulty (Score question)
    difficulty_ans = answers.get("difficulty")
    if not isinstance(difficulty_ans, dict):
        raise JevMalformedResponseError("Missing 'difficulty' answer in response")

    raw_score = difficulty_ans.get("score")
    if not isinstance(raw_score, (int, float)):
        raise JevMalformedResponseError(f"Invalid difficulty score value: {raw_score}")
    raw_difficulty = float(raw_score)

    diff_confidence = difficulty_ans.get("confidence")
    if not isinstance(diff_confidence, (int, float)) or not (0.0 <= diff_confidence <= 1.0):
        raise JevMalformedResponseError(f"Invalid confidence value in difficulty: {diff_confidence}")

    diff_dist = difficulty_ans.get("probabilities", {})
    if not isinstance(diff_dist, dict):
        diff_dist = {}

    # Map difficulty scale
    if difficulty_scale == "1-5":
        # Per specification (docs/02-technical-requirements.md): Map Score levels to 1–5
        # 0.0 -> 1.0 (trivial), up to 4.0 -> 5.0 (expert)
        difficulty = raw_difficulty + 1.0
    else:
        difficulty = raw_difficulty

    # 3. Validate needs_reasoning (Noul question)
    reasoning_ans = answers.get("needs_reasoning")
    if not isinstance(reasoning_ans, dict):
        raise JevMalformedResponseError("Missing 'needs_reasoning' answer in response")

    noul_val = reasoning_ans.get("noul")
    if not isinstance(noul_val, (int, float)) or not (0.0 <= noul_val <= 1.0):
        raise JevMalformedResponseError(f"Invalid noul probability value in needs_reasoning: {noul_val}")
    needs_reasoning_prob = float(noul_val)

    # Compound confidence: minimum of task and difficulty confidences
    overall_confidence = float(min(task_confidence, diff_confidence))

    # Optional metadata
    model_name = inner.get("model")
    latency_ms = inner.get("latency_ms")
    usage = inner.get("usage", {})
    input_tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
    credits_info = inner.get("credits", {})
    credits_charged = credits_info.get("charged") if isinstance(credits_info, dict) else None
    credits_remaining = credits_info.get("remaining") if isinstance(credits_info, dict) else None

    return JevDecision(
        task_type=choice,
        task_type_probs={str(k): float(v) for k, v in task_probs.items()},
        difficulty=difficulty,
        raw_difficulty=raw_difficulty,
        difficulty_dist={str(k): float(v) for k, v in diff_dist.items()},
        needs_reasoning_prob=needs_reasoning_prob,
        confidence=overall_confidence,
        jev_model=str(model_name) if model_name else None,
        jev_latency_ms=int(latency_ms) if latency_ms is not None else None,
        jev_input_tokens=int(input_tokens) if input_tokens is not None else None,
        credits_charged=int(credits_charged) if credits_charged is not None else None,
        credits_remaining=int(credits_remaining) if credits_remaining is not None else None,
    )
