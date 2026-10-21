"""risk alerts

Revision ID: 7e6afdf79deb
Revises: c7d2f4a9e613
Create Date: 2026-10-21 08:51:51.161182

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "7e6afdf79deb"
down_revision: str | Sequence[str] | None = "c7d2f4a9e613"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "alerts",
        sa.Column("id", sa.Uuid(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("vineyard_id", sa.Uuid(), nullable=False),
        sa.Column(
            "disease",
            sa.Enum(
                "downy_mildew",
                "powdery_mildew",
                name="disease",
                native_enum=False,
                length=20,
            ),
            nullable=False,
        ),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column(
            "level",
            sa.Enum(
                "low",
                "medium",
                "high",
                name="risk_level",
                native_enum=False,
                length=20,
            ),
            nullable=False,
        ),
        sa.Column("read_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "disease IN ('downy_mildew', 'powdery_mildew')",
            name=op.f("ck_alerts_disease"),
        ),
        sa.CheckConstraint(
            "level IN ('low', 'medium', 'high')", name=op.f("ck_alerts_risk_level")
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_alerts_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["vineyard_id"],
            ["vineyards.id"],
            name=op.f("fk_alerts_vineyard_id_vineyards"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_alerts")),
        sa.UniqueConstraint(
            "vineyard_id",
            "disease",
            "day",
            name=op.f("uq_alerts_vineyard_id_disease_day"),
        ),
    )
    op.create_index(op.f("ix_alerts_user_id"), "alerts", ["user_id"], unique=False)
    op.create_index(
        op.f("ix_alerts_vineyard_id"), "alerts", ["vineyard_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_alerts_vineyard_id"), table_name="alerts")
    op.drop_index(op.f("ix_alerts_user_id"), table_name="alerts")
    op.drop_table("alerts")
