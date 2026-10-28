import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Index,
    LargeBinary,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import ProcessingStatus, string_enum

if TYPE_CHECKING:
    from app.models.vineyard import Vineyard


class IndexType(enum.StrEnum):
    NDVI = "ndvi"
    NDRE = "ndre"


class SatelliteScan(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Снимок Sentinel-2, обработанный для одного участка."""

    __tablename__ = "satellite_scans"
    __table_args__ = (
        # Один и тот же снимок по участку не обрабатывается дважды
        UniqueConstraint("vineyard_id", "product_id"),
        # Снимки участка по времени: последний чистый снимок для тайлов NDVI,
        # история индексов и аномалии (pr-091)
        Index(
            "ix_satellite_scans_vineyard_id_acquired_at", "vineyard_id", "acquired_at"
        ),
        CheckConstraint("cloud_cover BETWEEN 0 AND 100", name="cloud_cover_range"),
        CheckConstraint(
            "parcel_cloud_cover BETWEEN 0 AND 100", name="parcel_cloud_cover_range"
        ),
    )

    vineyard_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("vineyards.id", ondelete="CASCADE"), index=True
    )
    # Идентификатор продукта в Copernicus Data Space Ecosystem
    product_id: Mapped[str] = mapped_column(String(255))
    acquired_at: Mapped[datetime]
    cloud_cover: Mapped[float]
    # Облачность над самим участком по маске SCL, %: облачность сцены 110 × 110 км
    # о конкретном участке почти ничего не говорит
    parcel_cloud_cover: Mapped[float | None]
    # Спутник: Sentinel-2A и 2B бывают над участком в один день
    platform: Mapped[str | None] = mapped_column(String(20))
    status: Mapped[ProcessingStatus] = mapped_column(
        string_enum(ProcessingStatus, "scan_status"),
        server_default=ProcessingStatus.PENDING.value,
    )
    error: Mapped[str | None] = mapped_column(String(1000))
    # Картограмма NDVI (PNG, единицы килобайт) и её углы [долгота, широта]:
    # читается только по запросу картинки, не со списком снимков
    ndvi_image: Mapped[bytes | None] = mapped_column(LargeBinary, deferred=True)
    ndvi_corners: Mapped[list[list[float]] | None] = mapped_column(JSONB)

    vineyard: Mapped["Vineyard"] = relationship(back_populates="satellite_scans")
    indices: Mapped[list["VegetationIndex"]] = relationship(
        back_populates="scan", cascade="all, delete-orphan", passive_deletes=True
    )


class VegetationIndex(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Статистика вегетационного индекса по пикселям участка на одном снимке."""

    __tablename__ = "vegetation_indices"
    __table_args__ = (
        UniqueConstraint("scan_id", "index_type"),
        CheckConstraint(
            "min_value BETWEEN -1 AND 1 AND max_value BETWEEN -1 AND 1"
            " AND min_value <= mean_value AND mean_value <= max_value",
            name="values_range",
        ),
        CheckConstraint("std_value >= 0", name="std_non_negative"),
    )

    scan_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("satellite_scans.id", ondelete="CASCADE"), index=True
    )
    index_type: Mapped[IndexType] = mapped_column(string_enum(IndexType, "index_type"))
    mean_value: Mapped[float]
    min_value: Mapped[float]
    max_value: Mapped[float]
    std_value: Mapped[float]
    # Обрезанный по участку растр в объектном хранилище (pr-062)
    raster_key: Mapped[str | None] = mapped_column(String(512))

    scan: Mapped["SatelliteScan"] = relationship(back_populates="indices")
