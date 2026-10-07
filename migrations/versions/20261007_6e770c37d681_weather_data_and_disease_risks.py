"""Метеоданные и риск болезней

Revision ID: 6e770c37d681
Revises: dc904f398929
Create Date: 2026-10-07 12:35:52.141883

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "6e770c37d681"
down_revision: str | Sequence[str] | None = "dc904f398929"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "disease_risks",
        sa.Column("id", sa.Uuid(), server_default=sa.text("uuidv7()"), nullable=False),
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
        sa.Column("target_date", sa.Date(), nullable=False),
        sa.Column("method", sa.String(length=100), nullable=False),
        sa.Column("score", sa.Double(), nullable=False),
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
        sa.Column("factors", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
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
            name=op.f("ck_disease_risks_disease"),
        ),
        sa.CheckConstraint(
            "level IN ('low', 'medium', 'high')",
            name=op.f("ck_disease_risks_risk_level"),
        ),
        sa.CheckConstraint(
            "score BETWEEN 0 AND 1", name=op.f("ck_disease_risks_score_range")
        ),
        sa.ForeignKeyConstraint(
            ["vineyard_id"],
            ["vineyards.id"],
            name=op.f("fk_disease_risks_vineyard_id_vineyards"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_disease_risks")),
        sa.UniqueConstraint(
            "vineyard_id",
            "disease",
            "target_date",
            "method",
            name=op.f("uq_disease_risks_vineyard_id_disease_target_date_method"),
        ),
    )
    op.create_index(
        op.f("ix_disease_risks_vineyard_id"),
        "disease_risks",
        ["vineyard_id"],
        unique=False,
    )
    op.create_table(
        "weather_data",
        sa.Column("id", sa.Uuid(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("vineyard_id", sa.Uuid(), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("temperature_c", sa.Double(), nullable=False),
        sa.Column("relative_humidity", sa.Double(), nullable=False),
        sa.Column("precipitation_mm", sa.Double(), nullable=False),
        sa.Column(
            "is_forecast", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
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
            "precipitation_mm >= 0",
            name=op.f("ck_weather_data_precipitation_non_negative"),
        ),
        sa.CheckConstraint(
            "relative_humidity BETWEEN 0 AND 100",
            name=op.f("ck_weather_data_humidity_range"),
        ),
        sa.ForeignKeyConstraint(
            ["vineyard_id"],
            ["vineyards.id"],
            name=op.f("fk_weather_data_vineyard_id_vineyards"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_weather_data")),
        sa.UniqueConstraint(
            "vineyard_id",
            "observed_at",
            name=op.f("uq_weather_data_vineyard_id_observed_at"),
        ),
    )
    op.create_index(
        op.f("ix_weather_data_vineyard_id"),
        "weather_data",
        ["vineyard_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_weather_data_vineyard_id"), table_name="weather_data")
    op.drop_table("weather_data")
    op.drop_index(op.f("ix_disease_risks_vineyard_id"), table_name="disease_risks")
    op.drop_table("disease_risks")
