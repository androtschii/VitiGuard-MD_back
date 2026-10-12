"""varieties

Revision ID: e75bb696d004
Revises: 059027aaa62b
Create Date: 2026-10-13 01:00:56.764458

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e75bb696d004"
down_revision: str | Sequence[str] | None = "059027aaa62b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "varieties",
        sa.Column("id", sa.Uuid(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column(
            "berry_color",
            sa.Enum(
                "white",
                "rose",
                "red",
                name="berry_color",
                native_enum=False,
                length=20,
            ),
            nullable=False,
        ),
        sa.Column(
            "downy_mildew_susceptibility",
            sa.Enum(
                "low",
                "medium",
                "high",
                name="downy_mildew_susceptibility",
                native_enum=False,
                length=20,
            ),
            nullable=True,
        ),
        sa.Column(
            "powdery_mildew_susceptibility",
            sa.Enum(
                "low",
                "medium",
                "high",
                name="powdery_mildew_susceptibility",
                native_enum=False,
                length=20,
            ),
            nullable=True,
        ),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "berry_color IN ('white', 'rose', 'red')",
            name=op.f("ck_varieties_berry_color"),
        ),
        sa.CheckConstraint(
            "downy_mildew_susceptibility IN ('low', 'medium', 'high')",
            name=op.f("ck_varieties_downy_mildew_susceptibility"),
        ),
        sa.CheckConstraint(
            "powdery_mildew_susceptibility IN ('low', 'medium', 'high')",
            name=op.f("ck_varieties_powdery_mildew_susceptibility"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_varieties")),
    )
    op.create_index(
        "uq_varieties_name_lower",
        "varieties",
        [sa.text("lower(name)")],
        unique=True,
    )

    # Основные сорта Молдовы: название и цвет ягоды. Восприимчивость к милдью
    # и оидиуму не заполняется — её указывает агроном по своим данным
    varieties = sa.table(
        "varieties",
        sa.column("name", sa.String),
        sa.column("berry_color", sa.String),
    )
    op.bulk_insert(
        varieties,
        [
            {"name": name, "berry_color": color}
            for name, color in [
                ("Fetească Neagră", "red"),
                ("Rara Neagră", "red"),
                ("Cabernet Sauvignon", "red"),
                ("Merlot", "red"),
                ("Pinot Noir", "red"),
                ("Saperavi", "red"),
                ("Isabella", "red"),
                ("Moldova", "red"),
                ("Fetească Albă", "white"),
                ("Fetească Regală", "white"),
                ("Chardonnay", "white"),
                ("Sauvignon Blanc", "white"),
                ("Aligoté", "white"),
                ("Riesling", "white"),
                ("Muscat Ottonel", "white"),
                ("Traminer Roz", "rose"),
                ("Pinot Gris", "rose"),
            ]
        ],
    )


def downgrade() -> None:
    op.drop_index("uq_varieties_name_lower", table_name="varieties")
    op.drop_table("varieties")
