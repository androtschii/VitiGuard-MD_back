"""ndvi map of satellite scan

Revision ID: 3e9a7c5f1b62
Revises: 8b3f6d21c0a4
Create Date: 2026-10-20 18:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "3e9a7c5f1b62"
down_revision: str | Sequence[str] | None = "8b3f6d21c0a4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "satellite_scans", sa.Column("ndvi_image", sa.LargeBinary(), nullable=True)
    )
    op.add_column(
        "satellite_scans",
        sa.Column(
            "ndvi_corners", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
    )


def downgrade() -> None:
    op.drop_column("satellite_scans", "ndvi_corners")
    op.drop_column("satellite_scans", "ndvi_image")
