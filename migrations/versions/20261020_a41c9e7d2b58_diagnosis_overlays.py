"""heatmap and lesion mask keys of diagnosis results

Revision ID: a41c9e7d2b58
Revises: 3e9a7c5f1b62
Create Date: 2026-10-20 20:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a41c9e7d2b58"
down_revision: str | Sequence[str] | None = "3e9a7c5f1b62"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "disease_results",
        sa.Column("heatmap_key", sa.String(length=512), nullable=True),
    )
    op.add_column(
        "disease_results",
        sa.Column("lesion_key", sa.String(length=512), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("disease_results", "lesion_key")
    op.drop_column("disease_results", "heatmap_key")
