"""leaf lesion severity in diagnosis results

Revision ID: 8b3f6d21c0a4
Revises: 5d1c8e2a9f47
Create Date: 2026-10-20 17:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "8b3f6d21c0a4"
down_revision: str | Sequence[str] | None = "5d1c8e2a9f47"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("disease_results", sa.Column("severity", sa.Double(), nullable=True))
    op.create_check_constraint(
        op.f("ck_disease_results_severity_range"),
        "disease_results",
        "severity BETWEEN 0 AND 1",
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_disease_results_severity_range"), "disease_results", type_="check"
    )
    op.drop_column("disease_results", "severity")
