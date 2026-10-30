"""weather events

Revision ID: f2a6c9d3e7b1
Revises: e1f4a8c2d6b9
Create Date: 2026-10-31 02:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f2a6c9d3e7b1"
down_revision: str | Sequence[str] | None = "e1f4a8c2d6b9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "weather_events",
        sa.Column("id", sa.Uuid(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("vineyard_id", sa.Uuid(), nullable=False),
        sa.Column("occurred_on", sa.Date(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("severity", sa.SmallInteger(), nullable=False),
        sa.Column("note", sa.String(length=500), nullable=True),
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
            name=op.f("fk_weather_events_vineyard_id_vineyards"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_weather_events")),
    )
    op.create_index(
        op.f("ix_weather_events_vineyard_id"), "weather_events", ["vineyard_id"]
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_weather_events_vineyard_id"), table_name="weather_events")
    op.drop_table("weather_events")
