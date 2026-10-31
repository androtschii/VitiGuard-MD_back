"""harvest records

Revision ID: a3b7d1e5c8f2
Revises: f2a6c9d3e7b1
Create Date: 2026-10-31 03:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a3b7d1e5c8f2"
down_revision: str | Sequence[str] | None = "f2a6c9d3e7b1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "harvest_records",
        sa.Column("id", sa.Uuid(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("vineyard_id", sa.Uuid(), nullable=False),
        sa.Column("season", sa.SmallInteger(), nullable=False),
        sa.Column("estimated_kg", sa.Float(), nullable=True),
        sa.Column("actual_kg", sa.Float(), nullable=True),
        sa.Column("sugar_brix", sa.Float(), nullable=True),
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
        sa.ForeignKeyConstraint(
            ["vineyard_id"],
            ["vineyards.id"],
            name=op.f("fk_harvest_records_vineyard_id_vineyards"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_harvest_records")),
        sa.UniqueConstraint(
            "vineyard_id",
            "season",
            name=op.f("uq_harvest_records_vineyard_id_season"),
        ),
    )
    op.create_index(
        op.f("ix_harvest_records_vineyard_id"), "harvest_records", ["vineyard_id"]
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_harvest_records_vineyard_id"), table_name="harvest_records")
    op.drop_table("harvest_records")
