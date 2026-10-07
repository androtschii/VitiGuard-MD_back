"""Фото листьев и результаты диагностики

Revision ID: 732d42489445
Revises: 56659937db3c
Create Date: 2026-10-07 12:09:41.803867

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "732d42489445"
down_revision: str | Sequence[str] | None = "56659937db3c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "leaf_images",
        sa.Column("id", sa.Uuid(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=False),
        sa.Column("vineyard_id", sa.Uuid(), nullable=True),
        sa.Column("storage_key", sa.String(length=512), nullable=False),
        sa.Column("content_type", sa.String(length=50), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
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
        sa.CheckConstraint("size_bytes > 0", name=op.f("ck_leaf_images_size_positive")),
        sa.CheckConstraint(
            "width > 0 AND height > 0", name=op.f("ck_leaf_images_dimensions_positive")
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["users.id"],
            name=op.f("fk_leaf_images_owner_id_users"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["vineyard_id"],
            ["vineyards.id"],
            name=op.f("fk_leaf_images_vineyard_id_vineyards"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_leaf_images")),
        sa.UniqueConstraint("storage_key", name=op.f("uq_leaf_images_storage_key")),
    )
    op.create_index(
        op.f("ix_leaf_images_owner_id"), "leaf_images", ["owner_id"], unique=False
    )
    op.create_index(
        op.f("ix_leaf_images_vineyard_id"), "leaf_images", ["vineyard_id"], unique=False
    )
    op.create_table(
        "disease_results",
        sa.Column("id", sa.Uuid(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("image_id", sa.Uuid(), nullable=False),
        sa.Column("model_name", sa.String(length=100), nullable=False),
        sa.Column("model_version", sa.String(length=50), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "pending",
                "completed",
                "failed",
                name="diagnosis_status",
                native_enum=False,
                length=20,
            ),
            server_default="pending",
            nullable=False,
        ),
        sa.Column("predicted_class", sa.String(length=50), nullable=True),
        sa.Column("confidence", sa.Double(), nullable=True),
        sa.Column(
            "probabilities", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("error", sa.String(length=1000), nullable=True),
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
            "status <> 'completed' OR (predicted_class IS NOT NULL AND confidence IS NOT NULL)",
            name=op.f("ck_disease_results_completed_has_prediction"),
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'completed', 'failed')",
            name=op.f("ck_disease_results_diagnosis_status"),
        ),
        sa.CheckConstraint(
            "confidence BETWEEN 0 AND 1",
            name=op.f("ck_disease_results_confidence_range"),
        ),
        sa.ForeignKeyConstraint(
            ["image_id"],
            ["leaf_images.id"],
            name=op.f("fk_disease_results_image_id_leaf_images"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_disease_results")),
    )
    op.create_index(
        op.f("ix_disease_results_image_id"),
        "disease_results",
        ["image_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_disease_results_image_id"), table_name="disease_results")
    op.drop_table("disease_results")
    op.drop_index(op.f("ix_leaf_images_vineyard_id"), table_name="leaf_images")
    op.drop_index(op.f("ix_leaf_images_owner_id"), table_name="leaf_images")
    op.drop_table("leaf_images")
