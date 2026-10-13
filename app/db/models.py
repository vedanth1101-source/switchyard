"""
PostgreSQL tables for routed requests (Task S2.3; docs/04 Part A, trimmed by the S2.3 decisions).

`requests` holds one row per outcome returned by app.core.router.route() (success, dry run or failed),
including its routing decision; `model_calls` holds one row per model attempt, in order (`seq`). Rows
are written once, in one transaction, after route() returns (S2.4/S2.5) and never updated. Nothing is
written when route() raises: setup errors, RoutingConfigError and unexpected exceptions leave logs only.

Stored: a sha256 of the prompt and its length, never the prompt or the answer; exception class names
and HTTP statuses, never messages; no client IP. Costs are list-price estimates in USD; NULL means
unknown, never zero. Jev credits are a count, never converted to USD.

CHECK constraints pin the closed value sets the routing code defines (tests/test_db_models.py keeps
them equal to the code's Literal types); a new value needs a migration. task_type is not constrained:
the classifier passes through whatever choice Jev returns.
"""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel

# Deterministic constraint names, so later migrations can refer to them. Set before any table exists.
SQLModel.metadata.naming_convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

ROUTE_STATUSES = ("success", "dry_run", "failed")  # app.core.router.RouteStatus
FAILURE_REASONS = ("all_failed", "deadline", "fatal")  # app.core.executor.ExhaustionReason
TIERS = ("cheap", "strong")  # app.core.policy.RouteDecision.tier
ROUTE_REASONS = ("EASY", "HARD", "LOW_CONFIDENCE", "JEV_ERROR", "BUDGET")  # policy reasons; CACHE not yet used
ATTEMPT_STATUSES = (  # app.core.executor.AttemptStatus
    "success",
    "rate_limited",
    "server_error",
    "timeout",
    "client_error",
    "malformed",
    "auth_error",
    "budget_exhausted",
    "deadline",
)

USD = sa.Numeric(12, 8)


def _one_of(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(value) for value in values)})"


class RequestRecord(SQLModel, table=True):
    __tablename__ = "requests"
    __table_args__ = (
        sa.CheckConstraint("prompt_hash ~ '^[0-9a-f]{64}$'", name="prompt_hash_hex"),
        sa.CheckConstraint("prompt_chars > 0", name="prompt_chars_positive"),
        sa.CheckConstraint(_one_of("status", ROUTE_STATUSES), name="status"),
        sa.CheckConstraint(_one_of("failure_reason", FAILURE_REASONS), name="failure_reason"),
        sa.CheckConstraint("(status = 'failed') = (failure_reason IS NOT NULL)", name="failure_reason_iff_failed"),
        sa.CheckConstraint(_one_of("final_tier", TIERS), name="final_tier"),
        sa.CheckConstraint(
            "(status = 'success') = (final_model IS NOT NULL)"
            " AND (status = 'success') = (final_tier IS NOT NULL)"
            " AND (status = 'success') = (fell_back_to_cheap IS NOT NULL)",
            name="final_fields_iff_success",
        ),
        sa.CheckConstraint("total_latency_ms >= 0", name="total_latency_ms_non_negative"),
        sa.CheckConstraint("(est_cost_usd IS NULL) = (baseline_cost_usd IS NULL)", name="costs_together"),
        sa.CheckConstraint("est_cost_usd IS NULL OR status = 'success'", name="costs_only_on_success"),
        sa.CheckConstraint("est_cost_usd >= 0 AND baseline_cost_usd >= 0", name="costs_non_negative"),
        sa.CheckConstraint(_one_of("tier", TIERS), name="tier"),
        sa.CheckConstraint(_one_of("reason", ROUTE_REASONS), name="reason"),
    )

    # Request and outcome
    id: uuid.UUID = Field(sa_column=sa.Column(sa.Uuid, primary_key=True))  # The API request_id
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    prompt_hash: str = Field(sa_column=sa.Column(sa.CHAR(64), nullable=False))  # sha256 hex of the prompt
    prompt_chars: int = Field(sa_column=sa.Column(sa.Integer, nullable=False))
    dry_run: bool = Field(sa_column=sa.Column(sa.Boolean, nullable=False))
    status: str = Field(sa_column=sa.Column(sa.Text, nullable=False))
    failure_reason: Optional[str] = Field(default=None, sa_column=sa.Column(sa.Text))
    final_model: Optional[str] = Field(default=None, sa_column=sa.Column(sa.Text))
    final_tier: Optional[str] = Field(default=None, sa_column=sa.Column(sa.Text))
    fell_back_to_cheap: Optional[bool] = Field(default=None, sa_column=sa.Column(sa.Boolean))
    strong_reserved: bool = Field(sa_column=sa.Column(sa.Boolean, nullable=False))
    total_latency_ms: int = Field(sa_column=sa.Column(sa.Integer, nullable=False))
    config_version: str = Field(sa_column=sa.Column(sa.Text, nullable=False))
    est_cost_usd: Optional[Decimal] = Field(default=None, sa_column=sa.Column(USD))
    baseline_cost_usd: Optional[Decimal] = Field(default=None, sa_column=sa.Column(USD))

    # Routing decision (final, after any budget downgrade)
    task_type: str = Field(sa_column=sa.Column(sa.Text, nullable=False))
    tier: str = Field(sa_column=sa.Column(sa.Text, nullable=False))
    reason: str = Field(sa_column=sa.Column(sa.Text, nullable=False))
    model_chain: list[str] = Field(sa_column=sa.Column(JSONB, nullable=False))
    confidence: Optional[float] = Field(default=None, sa_column=sa.Column(sa.Double))  # NULL when Jev failed
    difficulty: Optional[float] = Field(default=None, sa_column=sa.Column(sa.Double))
    needs_reasoning_prob: Optional[float] = Field(default=None, sa_column=sa.Column(sa.Double))
    task_type_probs: Optional[dict[str, Any]] = Field(default=None, sa_column=sa.Column(JSONB))
    difficulty_dist: Optional[dict[str, Any]] = Field(default=None, sa_column=sa.Column(JSONB))

    # Jev call
    jev_model: Optional[str] = Field(default=None, sa_column=sa.Column(sa.Text))
    jev_latency_ms: Optional[int] = Field(default=None, sa_column=sa.Column(sa.Integer))
    jev_input_tokens: Optional[int] = Field(default=None, sa_column=sa.Column(sa.Integer))
    jev_credits_charged: Optional[int] = Field(default=None, sa_column=sa.Column(sa.Integer))
    jev_error: Optional[str] = Field(default=None, sa_column=sa.Column(sa.Text))  # Exception class name only
    jev_http_status: Optional[int] = Field(default=None, sa_column=sa.Column(sa.Integer))
    jev_error_cause: Optional[str] = Field(default=None, sa_column=sa.Column(sa.Text))  # Exception class name only


class ModelCallRecord(SQLModel, table=True):
    __tablename__ = "model_calls"
    __table_args__ = (
        sa.CheckConstraint("seq >= 0", name="seq_non_negative"),
        sa.CheckConstraint(_one_of("tier", TIERS), name="tier"),
        sa.CheckConstraint("attempt_no >= 1", name="attempt_no_positive"),
        sa.CheckConstraint(_one_of("status", ATTEMPT_STATUSES), name="status"),
        sa.CheckConstraint("latency_ms >= 0", name="latency_ms_non_negative"),
        sa.CheckConstraint("http_status BETWEEN 100 AND 599", name="http_status_range"),
        sa.CheckConstraint(
            "input_tokens >= 0 AND output_tokens >= 0 AND reasoning_tokens >= 0", name="tokens_non_negative"
        ),
        sa.CheckConstraint("est_cost_usd >= 0", name="est_cost_non_negative"),
    )

    request_id: uuid.UUID = Field(
        sa_column=sa.Column(sa.Uuid, sa.ForeignKey("requests.id", ondelete="CASCADE"), primary_key=True)
    )
    seq: int = Field(sa_column=sa.Column(sa.Integer, primary_key=True))  # Position in RouteOutcome.attempts
    tier: str = Field(sa_column=sa.Column(sa.Text, nullable=False))
    model: str = Field(sa_column=sa.Column(sa.Text, nullable=False))
    attempt_no: int = Field(sa_column=sa.Column(sa.Integer, nullable=False))  # 1-based, per model
    status: str = Field(sa_column=sa.Column(sa.Text, nullable=False))
    latency_ms: int = Field(sa_column=sa.Column(sa.Integer, nullable=False))  # 0 for a "deadline" entry
    http_status: Optional[int] = Field(default=None, sa_column=sa.Column(sa.Integer))
    input_tokens: Optional[int] = Field(default=None, sa_column=sa.Column(sa.Integer))
    output_tokens: Optional[int] = Field(default=None, sa_column=sa.Column(sa.Integer))
    reasoning_tokens: Optional[int] = Field(default=None, sa_column=sa.Column(sa.Integer))
    est_cost_usd: Optional[Decimal] = Field(default=None, sa_column=sa.Column(USD))  # Filled from S2.11
