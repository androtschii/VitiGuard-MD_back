"""Расширение PostGIS

Revision ID: d032fb2c1b6d
Revises:
Create Date: 2026-10-07 08:16:18.547935

"""

from collections.abc import Sequence

from alembic import op

revision: str = "d032fb2c1b6d"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS postgis")
