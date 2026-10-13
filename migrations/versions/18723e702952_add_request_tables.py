"""add request tables

Revision ID: 18723e702952
Revises: f969e942a969
Create Date: 2026-10-13 12:00:00.000000

Task S2.3: `requests` (one row per routed outcome, including its routing decision) and `model_calls`
(one row per model attempt). Additive only. Values and names are frozen here on purpose: migrations must
not import app code that keeps changing; tests/test_migrations.py checks this DDL against app/db/models.py.
Downgrade drops both tables and their data: for local and CI databases only.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "18723e702952"
down_revision: Union[str, Sequence[str], None] = "f969e942a969"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "requests",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("prompt_hash", sa.CHAR(64), nullable=False),
        sa.Column("prompt_chars", sa.Integer(), nullable=False),
        sa.Column("dry_run", sa.Boolean(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("final_model", sa.Text(), nullable=True),
        sa.Column("final_tier", sa.Text(), nullable=True),
        sa.Column("fell_back_to_cheap", sa.Boolean(), nullable=True),
        sa.Column("strong_reserved", sa.Boolean(), nullable=False),
        sa.Column("total_latency_ms", sa.Integer(), nullable=False),
        sa.Column("config_version", sa.Text(), nullable=False),
        sa.Column("est_cost_usd", sa.Numeric(12, 8), nullable=True),
        sa.Column("baseline_cost_usd", sa.Numeric(12, 8), nullable=True),
        sa.Column("task_type", sa.Text(), nullable=False),
        sa.Column("tier", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("model_chain", postgresql.JSONB(), nullable=False),
        sa.Column("confidence", sa.Double(), nullable=True),
        sa.Column("difficulty", sa.Double(), nullable=True),
        sa.Column("needs_reasoning_prob", sa.Double(), nullable=True),
        sa.Column("task_type_probs", postgresql.JSONB(), nullable=True),
        sa.Column("difficulty_dist", postgresql.JSONB(), nullable=True),
        sa.Column("jev_model", sa.Text(), nullable=True),
        sa.Column("jev_latency_ms", sa.Integer(), nullable=True),
        sa.Column("jev_input_tokens", sa.Integer(), nullable=True),
        sa.Column("jev_credits_charged", sa.Integer(), nullable=True),
        sa.Column("jev_error", sa.Text(), nullable=True),
        sa.Column("jev_http_status", sa.Integer(), nullable=True),
        sa.Column("jev_error_cause", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_requests")),
        sa.CheckConstraint("prompt_hash ~ '^[0-9a-f]{64}$'", name=op.f("ck_requests_prompt_hash_hex")),
        sa.CheckConstraint("prompt_chars > 0", name=op.f("ck_requests_prompt_chars_positive")),
        sa.CheckConstraint("status IN ('success', 'dry_run', 'failed')", name=op.f("ck_requests_status")),
        sa.CheckConstraint(
            "failure_reason IN ('all_failed', 'deadline', 'fatal')", name=op.f("ck_requests_failure_reason")
        ),
        sa.CheckConstraint(
            "(status = 'failed') = (failure_reason IS NOT NULL)", name=op.f("ck_requests_failure_reason_iff_failed")
        ),
        sa.CheckConstraint("final_tier IN ('cheap', 'strong')", name=op.f("ck_requests_final_tier")),
        sa.CheckConstraint(
            "(status = 'success') = (final_model IS NOT NULL)"
            " AND (status = 'success') = (final_tier IS NOT NULL)"
            " AND (status = 'success') = (fell_back_to_cheap IS NOT NULL)",
            name=op.f("ck_requests_final_fields_iff_success"),
        ),
        sa.CheckConstraint("total_latency_ms >= 0", name=op.f("ck_requests_total_latency_ms_non_negative")),
        sa.CheckConstraint(
            "(est_cost_usd IS NULL) = (baseline_cost_usd IS NULL)", name=op.f("ck_requests_costs_together")
        ),
        sa.CheckConstraint(
            "est_cost_usd IS NULL OR status = 'success'", name=op.f("ck_requests_costs_only_on_success")
        ),
        sa.CheckConstraint("est_cost_usd >= 0 AND baseline_cost_usd >= 0", name=op.f("ck_requests_costs_non_negative")),
        sa.CheckConstraint("tier IN ('cheap', 'strong')", name=op.f("ck_requests_tier")),
        sa.CheckConstraint(
            "reason IN ('EASY', 'HARD', 'LOW_CONFIDENCE', 'JEV_ERROR', 'BUDGET')", name=op.f("ck_requests_reason")
        ),
    )
    op.create_table(
        "model_calls",
        sa.Column("request_id", sa.Uuid(), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("tier", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("attempt_no", sa.Integer(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("http_status", sa.Integer(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("reasoning_tokens", sa.Integer(), nullable=True),
        sa.Column("est_cost_usd", sa.Numeric(12, 8), nullable=True),
        sa.PrimaryKeyConstraint("request_id", "seq", name=op.f("pk_model_calls")),
        sa.CheckConstraint("seq >= 0", name=op.f("ck_model_calls_seq_non_negative")),
        sa.CheckConstraint("tier IN ('cheap', 'strong')", name=op.f("ck_model_calls_tier")),
        sa.CheckConstraint("attempt_no >= 1", name=op.f("ck_model_calls_attempt_no_positive")),
        sa.CheckConstraint(
            "status IN ('success', 'rate_limited', 'server_error', 'timeout', 'client_error', 'malformed', "
            "'auth_error', 'budget_exhausted', 'deadline')",
            name=op.f("ck_model_calls_status"),
        ),
        sa.CheckConstraint("latency_ms >= 0", name=op.f("ck_model_calls_latency_ms_non_negative")),
        sa.CheckConstraint("http_status BETWEEN 100 AND 599", name=op.f("ck_model_calls_http_status_range")),
        sa.CheckConstraint(
            "input_tokens >= 0 AND output_tokens >= 0 AND reasoning_tokens >= 0",
            name=op.f("ck_model_calls_tokens_non_negative"),
        ),
        sa.CheckConstraint("est_cost_usd >= 0", name=op.f("ck_model_calls_est_cost_non_negative")),
        sa.ForeignKeyConstraint(
            ["request_id"],
            ["requests.id"],
            name=op.f("fk_model_calls_request_id_requests"),
            ondelete="CASCADE",
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("model_calls")
    op.drop_table("requests")
