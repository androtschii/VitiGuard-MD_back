"""treatment waiting days

Revision ID: c4d8a1e6f2b7
Revises: b7e1f4c2a9d3
Create Date: 2026-10-30 21:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c4d8a1e6f2b7"
down_revision: str | Sequence[str] | None = "b7e1f4c2a9d3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("treatments", sa.Column("waiting_days", sa.SmallInteger()))


def downgrade() -> None:
    op.drop_column("treatments", "waiting_days")
