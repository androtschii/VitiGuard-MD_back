"""field notes

Revision ID: d9e3b5a7c1f4
Revises: c4d8a1e6f2b7
Create Date: 2026-10-30 23:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d9e3b5a7c1f4"
down_revision: str | Sequence[str] | None = "c4d8a1e6f2b7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "field_notes",
        sa.Column("vineyard_id", sa.Uuid(), nullable=False),
        sa.Column("noted_on", sa.Date(), nullable=False),
        sa.Column("text", sa.String(length=2000), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=True),
        sa.Column("longitude", sa.Float(), nullable=True),
        sa.Column("photo_key", sa.String(length=512), nullable=True),
        sa.Column("photo_type", sa.String(length=32), nullable=True),
        sa.Column("client_id", sa.Uuid(), nullable=True),
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
        sa.ForeignKeyConstraint(
            ["vineyard_id"],
            ["vineyards.id"],
            name=op.f("fk_field_notes_vineyard_id_vineyards"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_field_notes")),
        sa.UniqueConstraint("photo_key", name=op.f("uq_field_notes_photo_key")),
        sa.UniqueConstraint(
            "vineyard_id",
            "client_id",
            name=op.f("uq_field_notes_vineyard_id_client_id"),
        ),
    )
    op.create_index(op.f("ix_field_notes_vineyard_id"), "field_notes", ["vineyard_id"])


def downgrade() -> None:
    op.drop_index(op.f("ix_field_notes_vineyard_id"), table_name="field_notes")
    op.drop_table("field_notes")
