"""Email пользователей только в нижнем регистре

Revision ID: f9a98f29e548
Revises: 6e770c37d681
Create Date: 2026-10-08 12:05:18.604465

"""

from collections.abc import Sequence

from alembic import op

revision: str = "f9a98f29e548"
down_revision: str | Sequence[str] | None = "6e770c37d681"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Уникальность email учитывает регистр, поэтому без этого ограничения
    # «Grower@x.md» и «grower@x.md» стали бы двумя разными аккаунтами
    op.execute("UPDATE users SET email = lower(email) WHERE email <> lower(email)")
    op.create_check_constraint(
        op.f("ck_users_email_lowercase"), "users", "email = lower(email)"
    )


def downgrade() -> None:
    op.drop_constraint(op.f("ck_users_email_lowercase"), "users", type_="check")
