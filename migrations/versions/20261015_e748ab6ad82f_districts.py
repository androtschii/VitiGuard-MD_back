"""Районы Молдовы

Revision ID: e748ab6ad82f
Revises: c4ae495676e7
Create Date: 2026-10-15 11:24:12.137746

"""

# Операции *_geospatial_* GeoAlchemy2 регистрирует в alembic.op во время выполнения,
# в заглушках типов Alembic их нет
# mypy: disable-error-code="attr-defined"

import json
from collections.abc import Sequence
from pathlib import Path

import sqlalchemy as sa
from alembic import op
from geoalchemy2 import Geometry

revision: str = "e748ab6ad82f"
down_revision: str | Sequence[str] | None = "c4ae495676e7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Границы меняются редко, поэтому загружаются вместе со схемой из файла в репозитории
DISTRICTS = (
    Path(__file__).resolve().parents[2] / "app" / "data" / "moldova-districts.geojson"
)


def upgrade() -> None:
    op.create_geospatial_table(
        "districts",
        sa.Column("iso", sa.String(length=8), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
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
        sa.PrimaryKeyConstraint("iso", name=op.f("pk_districts")),
    )
    op.create_geospatial_index(
        "idx_districts_geom",
        "districts",
        ["geom"],
        unique=False,
        postgresql_using="gist",
        postgresql_ops={},
    )
    features = json.loads(DISTRICTS.read_text(encoding="utf-8"))["features"]
    op.get_bind().execute(
        sa.text(
            "INSERT INTO districts (iso, name, geom) "
            "VALUES (:iso, :name, ST_SetSRID(ST_GeomFromGeoJSON(:geometry), 4326))"
        ),
        [
            {
                "iso": feature["properties"]["iso"],
                "name": feature["properties"]["name"],
                "geometry": json.dumps(feature["geometry"]),
            }
            for feature in features
        ],
    )


def downgrade() -> None:
    op.drop_geospatial_index(
        "idx_districts_geom",
        table_name="districts",
        postgresql_using="gist",
        column_name="geom",
    )
    op.drop_geospatial_table("districts")
