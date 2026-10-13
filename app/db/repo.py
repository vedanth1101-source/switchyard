"""
Persisting routed requests (Task S2.4): outcome -> rows, and one transaction to save them.

outcome_to_rows() is pure: no I/O, settings or clock. It maps one RouteOutcome returned by
app.core.router.route() (success, dry run or failed) to one `requests` row and one `model_calls` row
per attempt, in RouteOutcome.attempts order (`seq`). Outcomes route() never returns (setup errors,
RoutingConfigError, unexpected exceptions) have no rows.

Privacy: the prompt is reduced to its sha256 and length; no answer text is mapped; Jev error fields
must be exception class names (anything else is refused, never stored). Values that are unknown stay
None (SQL NULL): costs are never invented, and per-attempt cost is left for S2.11.

save_outcome() writes the rows in one transaction: all of them or none. A repeated save of the same
request_id fails on the primary key (IntegrityError) instead of duplicating rows. Errors propagate to
the caller (S2.5 decides the failure policy); nothing here logs.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Mapping, Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.core.executor import Attempt
from app.core.router import RouteOutcome
from app.db.models import ModelCallRecord, RequestRecord
from app.infra.db import TIMEOUT_SECONDS

_CLASS_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class OutcomeRowsError(ValueError):
    """outcome_to_rows() refused an outcome; a distinct type so callers can tell it from other bugs."""


@dataclass(frozen=True)
class OutcomeRows:
    request: Mapping[str, Any]  # One `requests` row; created_at is set by the database
    model_calls: tuple[Mapping[str, Any], ...]  # One `model_calls` row per attempt, in order


def outcome_to_rows(outcome: RouteOutcome, *, prompt: str) -> OutcomeRows:
    """
    Raises:
        OutcomeRowsError: If request_id is not a UUID, the prompt is empty, or a Jev error field is not
            an exception class name. Messages never include the offending value.
    """
    try:
        request_id = uuid.UUID(outcome.request_id)
    except (ValueError, AttributeError, TypeError):
        raise OutcomeRowsError("request_id must be a UUID") from None
    if not prompt:
        raise OutcomeRowsError("prompt must not be empty")

    decision, jev, execution, cost = outcome.decision, outcome.jev, outcome.execution, outcome.cost
    request = {
        "id": request_id,
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_chars": len(prompt),
        "dry_run": outcome.status == "dry_run",  # route() returns "dry_run" exactly for dry runs
        "status": outcome.status,
        "failure_reason": outcome.failure_reason,
        "final_model": execution.final_model if execution else None,
        "final_tier": execution.final_tier if execution else None,
        "fell_back_to_cheap": execution.fell_back_to_cheap if execution else None,
        "strong_reserved": outcome.strong_reserved,
        "total_latency_ms": outcome.total_latency_ms,
        "config_version": outcome.config_version,
        "est_cost_usd": _usd(cost.est_cost_usd) if cost else None,
        "baseline_cost_usd": _usd(cost.baseline_cost_usd) if cost else None,
        "task_type": decision.task_type,
        "tier": decision.tier,
        "reason": decision.reason,
        "model_chain": list(decision.model_chain),
        "confidence": decision.confidence,
        "difficulty": decision.difficulty,
        "needs_reasoning_prob": decision.needs_reasoning_prob,
        "task_type_probs": dict(jev.task_type_probs) if jev else None,
        "difficulty_dist": dict(jev.difficulty_dist) if jev else None,
        "jev_model": jev.jev_model if jev else None,
        "jev_latency_ms": jev.jev_latency_ms if jev else None,
        "jev_input_tokens": jev.jev_input_tokens if jev else None,
        "jev_credits_charged": jev.credits_charged if jev else None,
        "jev_error": _class_name(decision.jev_error, "jev_error"),
        "jev_http_status": outcome.jev_http_status,
        "jev_error_cause": _class_name(outcome.jev_error_cause, "jev_error_cause"),
    }
    model_calls = tuple(_model_call(request_id, seq, attempt) for seq, attempt in enumerate(outcome.attempts))
    return OutcomeRows(request=MappingProxyType(request), model_calls=model_calls)


def save_outcome(engine: Engine, rows: OutcomeRows) -> None:
    """
    Inserts the request and its attempts in one transaction; on any error nothing is written.

    Raises:
        sqlalchemy.exc.IntegrityError: A constraint rejected a row, including a repeated request_id.
        sqlalchemy.exc.SQLAlchemyError: The database failed or timed out.
    """
    with engine.begin() as connection:
        if connection.dialect.name == "postgresql":
            # Transaction-scoped, so it also works through Neon's PgBouncer pooler (as in check_database).
            connection.execute(text(f"SET LOCAL statement_timeout = {TIMEOUT_SECONDS * 1000}"))
        connection.execute(RequestRecord.__table__.insert(), [dict(rows.request)])
        if rows.model_calls:
            connection.execute(ModelCallRecord.__table__.insert(), [dict(call) for call in rows.model_calls])


def _model_call(request_id: uuid.UUID, seq: int, attempt: Attempt) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "request_id": request_id,
            "seq": seq,
            "tier": attempt.tier,
            "model": attempt.model,
            "attempt_no": attempt.attempt_no,
            "status": attempt.status,
            "latency_ms": attempt.latency_ms,
            "http_status": attempt.http_status,
            "input_tokens": attempt.input_tokens,
            "output_tokens": attempt.output_tokens,
            "reasoning_tokens": attempt.reasoning_tokens,
            "est_cost_usd": None,  # S2.11
        }
    )


def _usd(value: float) -> Decimal:
    # Via str(): the float's shortest repr, not its binary expansion (0.00004665, not 0.0000466499...).
    return Decimal(str(value))


def _class_name(value: Optional[str], field: str) -> Optional[str]:
    if value is not None and not _CLASS_NAME.fullmatch(value):
        raise OutcomeRowsError(f"{field} must be an exception class name")
    return value
