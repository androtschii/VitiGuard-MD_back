"""diagnosis review by agronomist

Revision ID: 5d1c8e2a9f47
Revises: 69a03511b8cf
Create Date: 2026-10-19 15:10:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "5d1c8e2a9f47"
down_revision: str | Sequence[str] | None = "69a03511b8cf"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "disease_results",
        sa.Column("expert_class", sa.String(length=50), nullable=True),
    )
    op.add_column(
        "disease_results",
        sa.Column("expert_comment", sa.String(length=1000), nullable=True),
    )
    op.add_column(
        "disease_results", sa.Column("reviewed_by_id", sa.Uuid(), nullable=True)
    )
    op.add_column(
        "disease_results",
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_foreign_key(
        op.f("fk_disease_results_reviewed_by_id_users"),
        "disease_results",
        "users",
        ["reviewed_by_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_check_constraint(
        op.f("ck_disease_results_review_complete"),
        "disease_results",
        "(expert_class IS NULL) = (reviewed_at IS NULL)",
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_disease_results_review_complete"), "disease_results", type_="check"
    )
    op.drop_constraint(
        op.f("fk_disease_results_reviewed_by_id_users"),
        "disease_results",
        type_="foreignkey",
    )
    op.drop_column("disease_results", "reviewed_at")
    op.drop_column("disease_results", "reviewed_by_id")
    op.drop_column("disease_results", "expert_comment")
    op.drop_column("disease_results", "expert_class")
