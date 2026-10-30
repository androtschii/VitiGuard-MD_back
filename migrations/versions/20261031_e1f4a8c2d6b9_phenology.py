"""phenology observations

Revision ID: e1f4a8c2d6b9
Revises: d9e3b5a7c1f4
Create Date: 2026-10-31 00:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e1f4a8c2d6b9"
down_revision: str | Sequence[str] | None = "d9e3b5a7c1f4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "phenology_observations",
        sa.Column("id", sa.Uuid(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("vineyard_id", sa.Uuid(), nullable=False),
        sa.Column("observed_on", sa.Date(), nullable=False),
        sa.Column("stage", sa.SmallInteger(), nullable=False),
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
            name=op.f("fk_phenology_observations_vineyard_id_vineyards"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_phenology_observations")),
        sa.UniqueConstraint(
            "vineyard_id",
            "observed_on",
            name=op.f("uq_phenology_observations_vineyard_id_observed_on"),
        ),
    )
    op.create_index(
        op.f("ix_phenology_observations_vineyard_id"),
        "phenology_observations",
        ["vineyard_id"],
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_phenology_observations_vineyard_id"),
        table_name="phenology_observations",
    )
    op.drop_table("phenology_observations")
