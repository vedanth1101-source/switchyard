"""
Deterministic routing policy for Switchyard.

Pure functional module mapping Jev classification decisions to LLM model chains
based on documented task difficulty, reasoning needs, confidence thresholds,
and daily strong-model budget constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.core.classifier import JevDecision


class RoutingConfigError(ValueError):
    """Raised when no cheap model chain is configured for a cheap-tier route."""


@dataclass(frozen=True)
class RoutingConfig:
    """
    Configuration parameters for model selection and thresholds.
    """

    confidence_threshold: float = 0.6
    difficulty_threshold: float = 3.5
    reasoning_threshold: float = 0.60
    models: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    general_strong_chain: List[str] = field(default_factory=list)

    def get_chain(self, task_type: str, tier: str) -> List[str]:
        """
        Retrieves ordered candidate model chain for given task type and tier.
        Cheap tier falls back to the general cheap chain and never to a strong chain;
        raises RoutingConfigError if no cheap chain is configured.
        Other tiers fall back to general_strong_chain if task or tier is missing.
        """
        task_models = self.models.get(task_type)
        if isinstance(task_models, dict):
            chain = task_models.get(tier)
            if chain and isinstance(chain, list):
                return list(chain)
        if tier == "cheap":
            general_models = self.models.get("general")
            general_cheap = general_models.get("cheap") if isinstance(general_models, dict) else None
            if general_cheap and isinstance(general_cheap, list):
                return list(general_cheap)
            raise RoutingConfigError(f"No cheap model chain configured for task '{task_type}' or 'general'")
        return list(self.general_strong_chain)


@dataclass(frozen=True)
class RouteDecision:
    """
    Immutable routing decision output for downstream execution and audit logging.
    """

    tier: str  # "cheap" or "strong"
    task_type: str  # "code", "math", "writing", "general"
    reason: str  # "EASY", "HARD", "LOW_CONFIDENCE", "JEV_ERROR", "BUDGET", "CACHE"
    model_chain: List[str]  # Ordered fallback list of model identifiers
    confidence: Optional[float] = None
    difficulty: Optional[float] = None
    needs_reasoning_prob: Optional[float] = None
    jev_error: Optional[str] = None


def decide_route(
    decision: Optional[JevDecision],
    config: RoutingConfig,
    budget_available: bool = True,
    jev_error: Optional[str] = None,
) -> RouteDecision:
    """
    Computes deterministic routing decision.

    Pure function adhering to specification:
    1. If Jev call failed or error supplied -> STRONG general, reason = JEV_ERROR
    2. Elif confidence < threshold -> STRONG general, reason = LOW_CONFIDENCE
    3. Elif difficulty >= 3.5 or p_reasoning >= 0.6 -> STRONG [task], reason = HARD
    4. Else -> CHEAP [task], reason = EASY
    5. If tier == STRONG and daily budget cap reached -> CHEAP [task], reason = BUDGET

    Args:
        decision: Parsed JevDecision, or None if Jev failed.
        config: Routing thresholds and model chain mapping.
        budget_available: False if strong model daily limit has been exhausted.
        jev_error: Optional error message string if upstream Jev call failed.

    Returns:
        RouteDecision: Selected tier, task, reason, and ordered model chain.
    """
    # Rule 1: Jev API failure fallback
    if jev_error is not None or decision is None:
        error_detail = jev_error or "Jev evaluation failed or unavailable"
        # Rule 5: Budget exhaustion override
        if not budget_available:
            return RouteDecision(
                tier="cheap",
                task_type="general",
                reason="BUDGET",
                model_chain=config.get_chain("general", "cheap"),
                jev_error=error_detail,
            )
        return RouteDecision(
            tier="strong",
            task_type="general",
            reason="JEV_ERROR",
            model_chain=list(config.general_strong_chain),
            jev_error=error_detail,
        )

    # Rule 2: Low-confidence fallback
    # Threshold condition: strictly strictly less than threshold triggers fallback;
    # equality counts as confident.
    if decision.confidence < config.confidence_threshold:
        # Rule 5: Budget exhaustion override
        if not budget_available:
            return RouteDecision(
                tier="cheap",
                task_type="general",
                reason="BUDGET",
                model_chain=config.get_chain("general", "cheap"),
                confidence=decision.confidence,
                difficulty=decision.difficulty,
                needs_reasoning_prob=decision.needs_reasoning_prob,
            )
        return RouteDecision(
            tier="strong",
            task_type="general",
            reason="LOW_CONFIDENCE",
            model_chain=list(config.general_strong_chain),
            confidence=decision.confidence,
            difficulty=decision.difficulty,
            needs_reasoning_prob=decision.needs_reasoning_prob,
        )

    # Rule 3: High difficulty or multi-step reasoning
    is_hard = (
        decision.difficulty >= config.difficulty_threshold
        or decision.needs_reasoning_prob >= config.reasoning_threshold
    )

    if is_hard:
        tier = "strong"
        reason = "HARD"
        # Rule 5: Budget exhaustion override
        if not budget_available:
            tier = "cheap"
            reason = "BUDGET"

        chain = config.get_chain(decision.task_type, tier)
        return RouteDecision(
            tier=tier,
            task_type=decision.task_type,
            reason=reason,
            model_chain=chain,
            confidence=decision.confidence,
            difficulty=decision.difficulty,
            needs_reasoning_prob=decision.needs_reasoning_prob,
        )

    # Rule 4: Standard easy route
    tier = "cheap"
    reason = "EASY"
    chain = config.get_chain(decision.task_type, tier)
    return RouteDecision(
        tier=tier,
        task_type=decision.task_type,
        reason=reason,
        model_chain=chain,
        confidence=decision.confidence,
        difficulty=decision.difficulty,
        needs_reasoning_prob=decision.needs_reasoning_prob,
    )
