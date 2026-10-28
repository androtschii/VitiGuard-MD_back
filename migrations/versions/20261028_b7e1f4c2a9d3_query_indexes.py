"""query indexes

Revision ID: b7e1f4c2a9d3
Revises: 80a52ce0cd3e
Create Date: 2026-10-28 09:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b7e1f4c2a9d3"
down_revision: str | Sequence[str] | None = "80a52ce0cd3e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_osm_vineyards_geom_geography",
        "osm_vineyards",
        [sa.text("(CAST(geom AS geography(GEOMETRY,-1)))")],
        postgresql_using="gist",
    )
    op.create_index(
        "ix_satellite_scans_vineyard_id_acquired_at",
        "satellite_scans",
        ["vineyard_id", "acquired_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_satellite_scans_vineyard_id_acquired_at", "satellite_scans")
    op.drop_index("ix_osm_vineyards_geom_geography", "osm_vineyards")
