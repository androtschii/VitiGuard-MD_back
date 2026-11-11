"""vine checks

Revision ID: c7e1a9d4b2f6
Revises: b4c8e2f6a1d3
Create Date: 2026-11-11 15:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "c7e1a9d4b2f6"
down_revision: str | Sequence[str] | None = "b4c8e2f6a1d3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "vine_checks",
        sa.Column("id", sa.Uuid(), server_default=sa.text("uuidv7()"), nullable=False),
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
        sa.Column("owner_id", sa.Uuid(), nullable=False),
        sa.Column("vineyard_id", sa.Uuid(), nullable=True),
        sa.Column("storage_key", sa.String(length=512), nullable=False),
        sa.Column("content_type", sa.String(length=50), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("model_name", sa.String(length=100), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("trunks", sa.Integer(), nullable=False),
        sa.Column("bunches", sa.Integer(), nullable=False),
        sa.Column(
            "detections", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["users.id"],
            name=op.f("fk_vine_checks_owner_id_users"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["vineyard_id"],
            ["vineyards.id"],
            name=op.f("fk_vine_checks_vineyard_id_vineyards"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_vine_checks")),
        sa.UniqueConstraint("storage_key", name=op.f("uq_vine_checks_storage_key")),
    )
    op.create_index(
        op.f("ix_vine_checks_owner_id"), "vine_checks", ["owner_id"], unique=False
    )
    op.create_index(
        op.f("ix_vine_checks_vineyard_id"), "vine_checks", ["vineyard_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_vine_checks_vineyard_id"), table_name="vine_checks")
    op.drop_index(op.f("ix_vine_checks_owner_id"), table_name="vine_checks")
    op.drop_table("vine_checks")
