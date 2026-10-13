"""
Request router: Jev classification -> strong budget -> routing policy -> executor (Task S1.6).

route() is the single request path. Every external service comes in through RouterDeps, so tests
inject fakes. The request deadline starts before the Jev call and is shared with the executor,
which is the only retry layer.

Strong budget: availability is checked before the policy runs. When the policy picks strong (and
this is not a dry run) one reservation is made before execution; if it is refused (cap reached in a
race, or Redis failed) the policy runs again with budget_available=False. The executor's cheap
fallback never reserves.

Dry runs skip the model call and never reserve strong budget, but they still call Jev. With real
dependencies (the CLI), `--dry-run` therefore spends a Jev credit and contacts Redis when REDIS_URL
is set; it is not an offline mode.

Logging: one line per request with ids, routing fields, counts and latency. Prompts, keys, URLs and
raw exception messages are never logged; Jev failures appear only as their exception class name.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import asdict, dataclass
from typing import Any, AsyncIterator, Callable, Dict, Literal, Optional, Protocol, Tuple

from pydantic import ValidationError

from app.core.classifier import JevDecision, get_default_questions, parse_jev_response
from app.core.costs import CostError, CostEstimate, estimate_cost
from app.core.deadline import Deadline
from app.core.executor import Attempt, ChainExhausted, ExecutionResult, ExhaustionReason, Sleep, execute
from app.core.jev_client import JevAuthError, JevClient, JevClientError
from app.core.llm_client import GatewayClient, LLMAuthError, LLMClient
from app.core.model_config import ModelConfig, ModelConfigError, load_model_config
from app.core.policy import RouteDecision, RoutingConfigError, decide_route
from app.core.settings import Settings, get_settings
from app.infra.budget import RedisStrongBudget, StrongBudget, create_strong_budget

logger = logging.getLogger(__name__)

# Settings.ai_gateway_api_key reads this variable; models.yaml providers must name the same one.
GATEWAY_KEY_ENV = "AI_GATEWAY_API_KEY"

RouteStatus = Literal["success", "dry_run", "failed"]


class JevEvaluator(Protocol):
    async def evaluate(self, state: str, questions: Dict[str, Any]) -> Dict[str, Any]: ...


class CliConfigurationError(Exception):
    """The configuration cannot be turned into real CLI dependencies. Messages hold no secrets."""


def _new_request_id() -> str:
    return str(uuid.uuid4())


@dataclass(frozen=True)
class RouterDeps:
    config: ModelConfig
    jev: JevEvaluator
    budget: StrongBudget
    llm: Optional[LLMClient]  # May be None only for dry runs
    request_budget_seconds: float
    clock: Callable[[], float] = time.monotonic
    sleep: Sleep = asyncio.sleep
    new_request_id: Callable[[], str] = _new_request_id


@dataclass(frozen=True)
class RouteOutcome:
    """
    Result of one routed request; the shape Sprint 2 persists.
    """

    request_id: str
    status: RouteStatus
    decision: RouteDecision  # Final policy decision (after any budget downgrade)
    jev: Optional[JevDecision]  # None when Jev failed (decision.jev_error names the error class)
    strong_reserved: bool  # True when this request consumed one unit of the strong daily budget
    execution: Optional[ExecutionResult]  # Set only when status == "success"
    attempts: Tuple[Attempt, ...]  # Every model attempt, successful or not; empty for dry runs
    failure_reason: Optional[ExhaustionReason]  # Set only when status == "failed"
    total_latency_ms: int
    config_version: str
    # List-price estimate for the successful call; None for dry runs, failures, or when it cannot be
    # calculated (never reported as zero).
    cost: Optional[CostEstimate] = None
    # HTTP status of a failed Jev call; None when Jev succeeded or failed without one (timeout, network)
    jev_http_status: Optional[int] = None
    # httpx exception class behind a Jev timeout or network failure (e.g. "ConnectError"); else None
    jev_error_cause: Optional[str] = None


async def route(prompt: str, *, dry_run: bool, deps: RouterDeps) -> RouteOutcome:
    """
    Classifies, decides, reserves strong budget when needed and executes one prompt.

    Raises:
        ValueError: If deps.llm is missing for a non-dry run (raised before Jev is called).
        RoutingConfigError: If the routing config has no usable chain; logged, never routed around.
    """
    if deps.llm is None and not dry_run:
        raise ValueError("deps.llm is required unless dry_run is set")
    request_id = deps.new_request_id()
    started = deps.clock()
    deadline = Deadline(deps.request_budget_seconds, clock=deps.clock)

    jev, jev_error, jev_http_status, jev_error_cause = await _classify(prompt, deps.jev)
    try:
        decision, reserved = await _decide(jev, jev_error, dry_run=dry_run, deps=deps)
        status, execution, attempts, failure_reason = await _execute(prompt, decision, dry_run, deps, deadline)
    except RoutingConfigError:
        _log_request(
            request_id=request_id,
            status="config_error",
            decision=None,
            jev_error=jev_error,
            jev_error_cause=jev_error_cause,
            final_model=None,
            attempts=0,
            latency_ms=_elapsed_ms(deps.clock, started),
        )
        raise

    outcome = RouteOutcome(
        request_id=request_id,
        status=status,
        decision=decision,
        jev=jev,
        strong_reserved=reserved,
        execution=execution,
        attempts=attempts,
        failure_reason=failure_reason,
        total_latency_ms=_elapsed_ms(deps.clock, started),
        config_version=deps.config.config_version,
        cost=_estimate_cost(execution, deps.config, request_id),
        jev_http_status=jev_http_status,
        jev_error_cause=jev_error_cause,
    )
    _log_request(
        request_id=request_id,
        status=status,
        decision=decision,
        jev_error=jev_error,
        jev_error_cause=jev_error_cause,
        final_model=execution.final_model if execution else None,
        attempts=len(attempts),
        latency_ms=outcome.total_latency_ms,
        failure_reason=failure_reason,
    )
    return outcome


def _estimate_cost(
    execution: Optional[ExecutionResult], config: ModelConfig, request_id: str
) -> Optional[CostEstimate]:
    if execution is None:
        return None
    result = execution.result
    try:
        return estimate_cost(config, execution.final_model, result.input_tokens, result.output_tokens)
    except CostError as exc:
        # The answer is still returned; the cost is unknown, not zero.
        error_type = type(exc).__name__
        logger.warning(
            "route cost unavailable request_id=%s error_type=%s",
            request_id,
            error_type,
            extra={"request_id": request_id, "error_type": error_type},
        )
        return None


async def _classify(
    prompt: str, jev: JevEvaluator
) -> Tuple[Optional[JevDecision], Optional[str], Optional[int], Optional[str]]:
    # Returns (decision, error class, HTTP status, network cause class). Class names and HTTP status
    # only: Jev messages can include response text.
    try:
        raw = await jev.evaluate(prompt, get_default_questions())
    except JevClientError as exc:
        return None, type(exc).__name__, exc.status_code, exc.cause_type
    except ValueError as exc:  # JevClient's prompt-length guard
        return None, type(exc).__name__, None, None
    try:
        return parse_jev_response(raw), None, None, None
    except JevClientError as exc:  # JevMalformedResponseError; any other exception is a bug
        return None, type(exc).__name__, exc.status_code, exc.cause_type


async def _decide(
    jev: Optional[JevDecision], jev_error: Optional[str], *, dry_run: bool, deps: RouterDeps
) -> Tuple[RouteDecision, bool]:
    routing = deps.config.routing
    budget_available = await deps.budget.available()
    decision = decide_route(jev, routing, budget_available=budget_available, jev_error=jev_error)
    if decision.tier != "strong" or dry_run:
        return decision, False
    if await deps.budget.reserve():
        return decision, True
    # Lost the race for the last unit, or Redis failed between the two calls.
    return decide_route(jev, routing, budget_available=False, jev_error=jev_error), False


async def _execute(
    prompt: str, decision: RouteDecision, dry_run: bool, deps: RouterDeps, deadline: Deadline
) -> Tuple[RouteStatus, Optional[ExecutionResult], Tuple[Attempt, ...], Optional[ExhaustionReason]]:
    if dry_run or deps.llm is None:
        return "dry_run", None, (), None
    try:
        execution = await execute(prompt, decision, deps.config, deps.llm, deadline, sleep=deps.sleep, clock=deps.clock)
    except ChainExhausted as exc:
        return "failed", None, exc.attempts, exc.reason
    return "success", execution, execution.attempts, None


def _log_request(
    *,
    request_id: str,
    status: str,
    decision: Optional[RouteDecision],
    jev_error: Optional[str],
    jev_error_cause: Optional[str],
    final_model: Optional[str],
    attempts: int,
    latency_ms: int,
    failure_reason: Optional[str] = None,
) -> None:
    fields = {
        "request_id": request_id,
        "status": status,
        "tier": decision.tier if decision else None,
        "reason": decision.reason if decision else None,
        "task_type": decision.task_type if decision else None,
        "final_model": final_model,
        "attempts": attempts,
        "latency_ms": latency_ms,
        "failure_reason": failure_reason,
        "jev_error": jev_error,
        "jev_error_cause": jev_error_cause,
    }
    level = logging.INFO if status in ("success", "dry_run") else logging.WARNING
    template = "route " + " ".join(f"{key}=%s" for key in fields)
    logger.log(level, template, *("-" if value is None else value for value in fields.values()), extra=fields)


def _elapsed_ms(clock: Callable[[], float], started: float) -> int:
    return round((clock() - started) * 1000)


# --- CLI ----------------------------------------------------------------------------------------


@asynccontextmanager
async def build_cli_deps(settings: Settings, *, dry_run: bool) -> AsyncIterator[RouterDeps]:
    """
    Real dependencies for the CLI; clients are closed on exit. Constructing them makes no calls.
    """
    config = load_model_config(settings.models_config_path, confidence_threshold=settings.confidence_threshold)
    async with AsyncExitStack() as stack:
        jev = await stack.enter_async_context(
            JevClient(api_key=settings.jevstation_api_key, timeout=settings.jev_timeout_seconds)
        )
        llm = None if dry_run else await stack.enter_async_context(_gateway_client(config, settings))
        budget = create_strong_budget(settings)
        if isinstance(budget, RedisStrongBudget):
            stack.push_async_callback(budget.aclose)
        yield RouterDeps(
            config=config,
            jev=jev,
            budget=budget,
            llm=llm,
            request_budget_seconds=settings.request_budget_seconds,
        )


def _gateway_client(config: ModelConfig, settings: Settings) -> GatewayClient:
    provider_names = sorted({model.provider for model in config.catalog.values()})
    if len(provider_names) != 1:
        raise CliConfigurationError(f"exactly one model provider is supported, found {len(provider_names)}")
    provider = config.providers[provider_names[0]]
    if provider.api_key_env != GATEWAY_KEY_ENV:
        raise CliConfigurationError(f"providers.{provider_names[0]}.api_key_env must be {GATEWAY_KEY_ENV}")
    return GatewayClient(provider.base_url, settings.ai_gateway_api_key)


DepsFactory = Callable[..., Any]


def main(argv: Optional[list[str]] = None, *, deps_factory: DepsFactory = build_cli_deps) -> int:
    """
    CLI entry point. Exit codes: 0 success or dry run, 1 failed route, 2 configuration error.
    """
    parser = argparse.ArgumentParser(
        prog="python -m app.core.router",
        description=(
            "Route one prompt through Switchyard. Every run calls Jev (one Jev credit) and contacts Redis "
            "when REDIS_URL is set; the model gateway is called unless --dry-run is given."
        ),
    )
    parser.add_argument("prompt")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="classify and choose a route, but skip the model call (Jev is still called)",
    )
    args = parser.parse_args(argv)

    try:
        outcome = asyncio.run(_run_cli(args.prompt, args.dry_run, deps_factory))
    except ValidationError as exc:  # Settings errors never include the submitted values
        print(f"error: invalid settings\n{exc}", file=sys.stderr)
        return 2
    except JevAuthError:
        print("error: Jev is not configured; set JEVSTATION_API_KEY", file=sys.stderr)
        return 2
    except LLMAuthError:
        print(f"error: the model gateway is not configured; set {GATEWAY_KEY_ENV}", file=sys.stderr)
        return 2
    except (ModelConfigError, CliConfigurationError) as exc:  # Messages name config keys, never values
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RoutingConfigError:
        print("error: the routing configuration has no usable model chain", file=sys.stderr)
        return 2

    print(json.dumps(_summary(outcome), indent=2))
    return 1 if outcome.status == "failed" else 0


async def _run_cli(prompt: str, dry_run: bool, deps_factory: DepsFactory) -> RouteOutcome:
    settings = get_settings()
    async with deps_factory(settings, dry_run=dry_run) as deps:
        return await route(prompt, dry_run=dry_run, deps=deps)


def _summary(outcome: RouteOutcome) -> Dict[str, Any]:
    execution = outcome.execution
    return {
        "request_id": outcome.request_id,
        "status": outcome.status,
        "decision": asdict(outcome.decision),
        "jev_http_status": outcome.jev_http_status,
        "jev_error_cause": outcome.jev_error_cause,
        "strong_reserved": outcome.strong_reserved,
        "final_model": execution.final_model if execution else None,
        "final_tier": execution.final_tier if execution else None,
        "fell_back_to_cheap": execution.fell_back_to_cheap if execution else None,
        "answer": execution.result.text if execution else None,
        "attempts": [asdict(attempt) for attempt in outcome.attempts],
        "failure_reason": outcome.failure_reason,
        "total_latency_ms": outcome.total_latency_ms,
        "config_version": outcome.config_version,
        "cost": asdict(outcome.cost) if outcome.cost else None,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
