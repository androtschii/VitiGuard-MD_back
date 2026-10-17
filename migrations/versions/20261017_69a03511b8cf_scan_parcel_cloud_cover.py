"""parcel cloud cover and platform

Revision ID: 69a03511b8cf
Revises: e748ab6ad82f
Create Date: 2026-10-17 14:34:40.949950

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "69a03511b8cf"
down_revision: str | Sequence[str] | None = "e748ab6ad82f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "satellite_scans", sa.Column("parcel_cloud_cover", sa.Double(), nullable=True)
    )
    op.add_column(
        "satellite_scans", sa.Column("platform", sa.String(length=20), nullable=True)
    )
    op.create_check_constraint(
        op.f("ck_satellite_scans_parcel_cloud_cover_range"),
        "satellite_scans",
        "parcel_cloud_cover BETWEEN 0 AND 100",
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_satellite_scans_parcel_cloud_cover_range"),
        "satellite_scans",
        type_="check",
    )
    op.drop_column("satellite_scans", "platform")
    op.drop_column("satellite_scans", "parcel_cloud_cover")
