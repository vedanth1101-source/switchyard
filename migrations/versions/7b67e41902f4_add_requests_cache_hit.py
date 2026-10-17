"""add requests.cache_hit

Revision ID: 7b67e41902f4
Revises: 18723e702952
Create Date: 2026-10-17 12:00:00.000000

Task S2.14 step 4: `requests.cache_hit` marks a request answered from the response cache. Additive only:
existing rows get the default `false`, which the new CHECK accepts. Values and names are frozen here on
purpose, as in 18723e702952; tests/test_migrations.py checks this DDL against app/db/models.py.
Downgrade drops the CHECK and the column, and with it the record of which requests were cache hits.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7b67e41902f4"
down_revision: Union[str, Sequence[str], None] = "18723e702952"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("requests", sa.Column("cache_hit", sa.Boolean(), server_default=sa.false(), nullable=False))
    op.create_check_constraint(
        op.f("ck_requests_cache_hit_shape"),
        "requests",
        "NOT cache_hit OR (status = 'success' AND NOT dry_run AND NOT strong_reserved"
        " AND est_cost_usd IS NULL AND NOT fell_back_to_cheap)",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(op.f("ck_requests_cache_hit_shape"), "requests", type_="check")
    op.drop_column("requests", "cache_hit")
