"""telegram chat of user

Revision ID: 87d8592cea7d
Revises: da7bc4371aab
Create Date: 2026-10-21 19:14:51.798037

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "87d8592cea7d"
down_revision: str | Sequence[str] | None = "da7bc4371aab"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "users", sa.Column("telegram_chat_id", sa.BigInteger(), nullable=True)
    )
    op.create_unique_constraint(
        op.f("uq_users_telegram_chat_id"), "users", ["telegram_chat_id"]
    )


def downgrade() -> None:
    op.drop_constraint(op.f("uq_users_telegram_chat_id"), "users", type_="unique")
    op.drop_column("users", "telegram_chat_id")
