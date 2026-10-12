"""initial empty revision

Revision ID: f969e942a969
Revises:
Create Date: 2026-10-11 18:05:43.401486

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "f969e942a969"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
