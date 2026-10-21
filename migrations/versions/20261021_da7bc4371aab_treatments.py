"""treatments

Revision ID: da7bc4371aab
Revises: 665f6b2f2b95
Create Date: 2026-10-21 19:07:18.532902

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "da7bc4371aab"
down_revision: str | Sequence[str] | None = "665f6b2f2b95"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "treatments",
        sa.Column("id", sa.Uuid(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("vineyard_id", sa.Uuid(), nullable=False),
        sa.Column("applied_on", sa.Date(), nullable=False),
        sa.Column(
            "fungicide",
            sa.Enum(
                "contact",
                "systemic",
                "sulfur",
                name="fungicide",
                native_enum=False,
                length=20,
            ),
            nullable=False,
        ),
        sa.Column("product", sa.String(length=200), nullable=True),
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
            "fungicide IN ('contact', 'systemic', 'sulfur')",
            name=op.f("ck_treatments_fungicide"),
        ),
        sa.ForeignKeyConstraint(
            ["vineyard_id"],
            ["vineyards.id"],
            name=op.f("fk_treatments_vineyard_id_vineyards"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_treatments")),
    )
    op.create_index(
        op.f("ix_treatments_vineyard_id"), "treatments", ["vineyard_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_treatments_vineyard_id"), table_name="treatments")
    op.drop_table("treatments")
