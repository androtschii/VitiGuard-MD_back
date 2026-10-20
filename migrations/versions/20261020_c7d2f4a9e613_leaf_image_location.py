"""leaf image location from exif

Revision ID: c7d2f4a9e613
Revises: a41c9e7d2b58
Create Date: 2026-10-20 21:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c7d2f4a9e613"
down_revision: str | Sequence[str] | None = "a41c9e7d2b58"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("leaf_images", sa.Column("latitude", sa.Double(), nullable=True))
    op.add_column("leaf_images", sa.Column("longitude", sa.Double(), nullable=True))
    op.create_check_constraint(
        op.f("ck_leaf_images_location_valid"),
        "leaf_images",
        "(latitude IS NULL) = (longitude IS NULL)"
        " AND latitude BETWEEN -90 AND 90 AND longitude BETWEEN -180 AND 180",
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_leaf_images_location_valid"), "leaf_images", type_="check"
    )
    op.drop_column("leaf_images", "longitude")
    op.drop_column("leaf_images", "latitude")
