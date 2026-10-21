"""alert level of user

Revision ID: 665f6b2f2b95
Revises: 7e6afdf79deb
Create Date: 2026-10-21 09:24:00.365396

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "665f6b2f2b95"
down_revision: str | Sequence[str] | None = "7e6afdf79deb"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "alert_level",
            sa.Enum(
                "low",
                "medium",
                "high",
                name="alert_level",
                native_enum=False,
                length=20,
            ),
            server_default="high",
            nullable=True,
        ),
    )
    op.create_check_constraint(
        op.f("ck_users_alert_level"),
        "users",
        "alert_level IN ('low', 'medium', 'high')",
    )


def downgrade() -> None:
    op.drop_constraint(op.f("ck_users_alert_level"), "users", type_="check")
    op.drop_column("users", "alert_level")
