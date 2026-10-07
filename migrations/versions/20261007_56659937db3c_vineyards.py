"""Участки виноградников

Revision ID: 56659937db3c
Revises: 06213c5b2182
Create Date: 2026-10-07 11:46:53.046938

"""

# Операции *_geospatial_* GeoAlchemy2 регистрирует в alembic.op во время выполнения,
# в заглушках типов Alembic их нет
# mypy: disable-error-code="attr-defined"

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from geoalchemy2 import Geometry

revision: str = "56659937db3c"
down_revision: str | Sequence[str] | None = "06213c5b2182"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_geospatial_table(
        "vineyards",
        sa.Column("id", sa.Uuid(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column(
            "geom",
            Geometry(
                geometry_type="MULTIPOLYGON",
                srid=4326,
                dimension=2,
                spatial_index=False,
                from_text="ST_GeomFromEWKT",
                name="geometry",
                nullable=False,
            ),
            nullable=False,
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
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["users.id"],
            name=op.f("fk_vineyards_owner_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_vineyards")),
    )
    op.create_geospatial_index(
        "idx_vineyards_geom",
        "vineyards",
        ["geom"],
        unique=False,
        postgresql_using="gist",
        postgresql_ops={},
    )
    op.create_index(
        op.f("ix_vineyards_owner_id"), "vineyards", ["owner_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_vineyards_owner_id"), table_name="vineyards")
    op.drop_geospatial_index(
        "idx_vineyards_geom",
        table_name="vineyards",
        postgresql_using="gist",
        column_name="geom",
    )
    op.drop_geospatial_table("vineyards")
