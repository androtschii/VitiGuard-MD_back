"""Виноградники OpenStreetMap

Revision ID: c4ae495676e7
Revises: db56d2b803bd
Create Date: 2026-10-13 12:51:12.682514

"""

# Операции *_geospatial_* GeoAlchemy2 регистрирует в alembic.op во время выполнения,
# в заглушках типов Alembic их нет
# mypy: disable-error-code="attr-defined"

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from geoalchemy2 import Geometry

revision: str = "c4ae495676e7"
down_revision: str | Sequence[str] | None = "db56d2b803bd"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_geospatial_table(
        "osm_vineyards",
        sa.Column("id", sa.Uuid(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("osm_id", sa.String(length=40), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=True),
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
        sa.PrimaryKeyConstraint("id", name=op.f("pk_osm_vineyards")),
        sa.UniqueConstraint("osm_id", name=op.f("uq_osm_vineyards_osm_id")),
    )
    op.create_geospatial_index(
        "idx_osm_vineyards_geom",
        "osm_vineyards",
        ["geom"],
        unique=False,
        postgresql_using="gist",
        postgresql_ops={},
    )


def downgrade() -> None:
    op.drop_geospatial_index(
        "idx_osm_vineyards_geom",
        table_name="osm_vineyards",
        postgresql_using="gist",
        column_name="geom",
    )
    op.drop_geospatial_table("osm_vineyards")
