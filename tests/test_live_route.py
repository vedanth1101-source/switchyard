"""Opt-in live smoke test (Task S1.9). Deselected by the default `pytest` run.

These tests call the real Jev API and the real Vercel AI Gateway and SPEND CREDITS:

- test_live_dry_run_classifies_with_jev: 1 Jev credit, no model call.
- test_live_cheap_prompt_routes_end_to_end: 1 Jev credit and 1 to 6 calls to cheap-chain (free-tier)
  models. Strong routing is disabled for this test, so no strong model is ever called.

Run only deliberately, with both keys exported in the shell (Settings does not read .env):

    JEVSTATION_API_KEY=... AI_GATEWAY_API_KEY=... pytest -m live tests/test_live_route.py -v

They confirm that the real Jev and gateway responses match the parsers built against fixtures and
documentation. Assertion messages contain statuses and class names only, never keys or answers.
"""

import asyncio
import dataclasses
import os
from collections.abc import Mapping

import pytest

from app.core.router import RouteOutcome, build_cli_deps, route
from app.core.settings import get_settings
from app.infra.budget import UnavailableStrongBudget

pytestmark = pytest.mark.live

LIVE_KEYS = ("JEVSTATION_API_KEY", "AI_GATEWAY_API_KEY")
DRY_RUN_PROMPT = "Reverse a string in Python"
CHEAP_PROMPT = "Reply with exactly one word: pong"


def require_live_keys(environ: Mapping[str, str]) -> None:
    """Fail (never skip silently) unless every live key is set; messages name the variable only."""
    # Hide this frame from tracebacks: pytest would otherwise print the `environ` argument,
    # including any key that is set.
    __tracebackhide__ = True
    for name in LIVE_KEYS:
        if not environ.get(name, "").strip():
            pytest.fail(f"Live smoke test needs real credentials; export {name}")


def describe(outcome: RouteOutcome) -> str:
    attempts = ", ".join(f"{a.model}#{a.attempt_no}={a.status}" for a in outcome.attempts) or "none"
    decision = outcome.decision
    return (
        f"status={outcome.status} failure_reason={outcome.failure_reason} tier={decision.tier} "
        f"reason={decision.reason} jev_error={decision.jev_error} attempts=[{attempts}]"
    )


def test_live_dry_run_classifies_with_jev():
    require_live_keys(os.environ)

    async def run() -> RouteOutcome:
        async with build_cli_deps(get_settings(), dry_run=True) as deps:
            return await route(DRY_RUN_PROMPT, dry_run=True, deps=deps)

    outcome = asyncio.run(run())

    assert outcome.status == "dry_run", describe(outcome)
    assert outcome.jev is not None, describe(outcome)  # Jev answered and its response parsed
    assert outcome.decision.jev_error is None, describe(outcome)
    assert outcome.decision.task_type in {"code", "math", "writing", "general"}
    assert 0.0 <= outcome.jev.confidence <= 1.0
    assert outcome.attempts == ()
    assert outcome.strong_reserved is False


def test_live_cheap_prompt_routes_end_to_end():
    require_live_keys(os.environ)

    async def run() -> RouteOutcome:
        async with build_cli_deps(get_settings(), dry_run=False) as deps:
            # No strong budget: the policy downgrades any strong decision to the cheap chain.
            cheap_only = dataclasses.replace(deps, budget=UnavailableStrongBudget())
            return await route(CHEAP_PROMPT, dry_run=False, deps=cheap_only)

    outcome = asyncio.run(run())

    assert outcome.status == "success", describe(outcome)
    assert outcome.decision.tier == "cheap", describe(outcome)
    assert outcome.strong_reserved is False
    assert {attempt.tier for attempt in outcome.attempts} == {"cheap"}, describe(outcome)
    execution = outcome.execution
    assert execution is not None
    assert execution.final_tier == "cheap"
    assert execution.result.text.strip(), describe(outcome)
    assert execution.result.input_tokens > 0, describe(outcome)
    assert execution.result.output_tokens > 0, describe(outcome)
    assert outcome.cost is not None, describe(outcome)
    assert outcome.cost.est_cost_usd >= 0
