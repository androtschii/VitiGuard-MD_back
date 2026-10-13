"""vineyard variety and planting year

Revision ID: db56d2b803bd
Revises: e75bb696d004
Create Date: 2026-10-13 11:00:41.385721

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "db56d2b803bd"
down_revision: str | Sequence[str] | None = "e75bb696d004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("vineyards", sa.Column("variety_id", sa.Uuid(), nullable=True))
    op.add_column(
        "vineyards", sa.Column("planting_year", sa.SmallInteger(), nullable=True)
    )
    op.create_index(
        op.f("ix_vineyards_variety_id"), "vineyards", ["variety_id"], unique=False
    )
    op.create_foreign_key(
        op.f("fk_vineyards_variety_id_varieties"),
        "vineyards",
        "varieties",
        ["variety_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_check_constraint(
        op.f("ck_vineyards_planting_year_range"), "vineyards", "planting_year >= 1900"
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_vineyards_planting_year_range"), "vineyards", type_="check"
    )
    op.drop_constraint(
        op.f("fk_vineyards_variety_id_varieties"), "vineyards", type_="foreignkey"
    )
    op.drop_index(op.f("ix_vineyards_variety_id"), table_name="vineyards")
    op.drop_column("vineyards", "planting_year")
    op.drop_column("vineyards", "variety_id")
