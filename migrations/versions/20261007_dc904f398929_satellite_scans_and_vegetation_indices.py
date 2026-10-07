"""Спутниковые снимки и вегетационные индексы

Revision ID: dc904f398929
Revises: 732d42489445
Create Date: 2026-10-07 12:28:03.535208

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "dc904f398929"
down_revision: str | Sequence[str] | None = "732d42489445"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "satellite_scans",
        sa.Column("id", sa.Uuid(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("vineyard_id", sa.Uuid(), nullable=False),
        sa.Column("product_id", sa.String(length=255), nullable=False),
        sa.Column("acquired_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("cloud_cover", sa.Double(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "pending",
                "completed",
                "failed",
                name="scan_status",
                native_enum=False,
                length=20,
            ),
            server_default="pending",
            nullable=False,
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
            "status IN ('pending', 'completed', 'failed')",
            name=op.f("ck_satellite_scans_scan_status"),
        ),
        sa.CheckConstraint(
            "cloud_cover BETWEEN 0 AND 100",
            name=op.f("ck_satellite_scans_cloud_cover_range"),
        ),
        sa.ForeignKeyConstraint(
            ["vineyard_id"],
            ["vineyards.id"],
            name=op.f("fk_satellite_scans_vineyard_id_vineyards"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_satellite_scans")),
        sa.UniqueConstraint(
            "vineyard_id",
            "product_id",
            name=op.f("uq_satellite_scans_vineyard_id_product_id"),
        ),
    )
    op.create_index(
        op.f("ix_satellite_scans_vineyard_id"),
        "satellite_scans",
        ["vineyard_id"],
        unique=False,
    )
    op.create_table(
        "vegetation_indices",
        sa.Column("id", sa.Uuid(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("scan_id", sa.Uuid(), nullable=False),
        sa.Column(
            "index_type",
            sa.Enum(
                "ndvi",
                "ndre",
                name="index_type",
                native_enum=False,
                length=20,
            ),
            nullable=False,
        ),
        sa.Column("mean_value", sa.Double(), nullable=False),
        sa.Column("min_value", sa.Double(), nullable=False),
        sa.Column("max_value", sa.Double(), nullable=False),
        sa.Column("std_value", sa.Double(), nullable=False),
        sa.Column("raster_key", sa.String(length=512), nullable=True),
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
            "index_type IN ('ndvi', 'ndre')",
            name=op.f("ck_vegetation_indices_index_type"),
        ),
        sa.CheckConstraint(
            "min_value BETWEEN -1 AND 1 AND max_value BETWEEN -1 AND 1 AND min_value <= mean_value AND mean_value <= max_value",
            name=op.f("ck_vegetation_indices_values_range"),
        ),
        sa.CheckConstraint(
            "std_value >= 0", name=op.f("ck_vegetation_indices_std_non_negative")
        ),
        sa.ForeignKeyConstraint(
            ["scan_id"],
            ["satellite_scans.id"],
            name=op.f("fk_vegetation_indices_scan_id_satellite_scans"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_vegetation_indices")),
        sa.UniqueConstraint(
            "scan_id",
            "index_type",
            name=op.f("uq_vegetation_indices_scan_id_index_type"),
        ),
    )
    op.create_index(
        op.f("ix_vegetation_indices_scan_id"),
        "vegetation_indices",
        ["scan_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_vegetation_indices_scan_id"), table_name="vegetation_indices"
    )
    op.drop_table("vegetation_indices")
    op.drop_index(op.f("ix_satellite_scans_vineyard_id"), table_name="satellite_scans")
    op.drop_table("satellite_scans")
