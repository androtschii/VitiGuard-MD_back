"""telegram alerts

Revision ID: 80a52ce0cd3e
Revises: 87d8592cea7d
Create Date: 2026-10-21 19:25:48.246947

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "80a52ce0cd3e"
down_revision: str | Sequence[str] | None = "87d8592cea7d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "alerts", sa.Column("notified_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column(
        "vineyards",
        sa.Column(
            "telegram_alerts",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("vineyards", "telegram_alerts")
    op.drop_column("alerts", "notified_at")
