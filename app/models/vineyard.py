import uuid
from typing import TYPE_CHECKING

from geoalchemy2 import Geometry, WKBElement
from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.diagnosis import LeafImage
    from app.models.satellite import SatelliteScan
    from app.models.user import User


class Vineyard(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "vineyards"

    owner_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    # Участок может состоять из нескольких контуров; SRID 4326 — как в GeoJSON и веб-картах
    geom: Mapped[WKBElement] = mapped_column(Geometry("MULTIPOLYGON", srid=4326))

    owner: Mapped["User"] = relationship(back_populates="vineyards")
    # Фото остаются у владельца, если участок удалён (vineyard_id → NULL)
    leaf_images: Mapped[list["LeafImage"]] = relationship(
        back_populates="vineyard", passive_deletes=True
    )
    satellite_scans: Mapped[list["SatelliteScan"]] = relationship(
        back_populates="vineyard", cascade="all, delete-orphan", passive_deletes=True
    )
