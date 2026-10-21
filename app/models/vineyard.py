import uuid
from typing import TYPE_CHECKING

from geoalchemy2 import Geometry, WKBElement
from sqlalchemy import CheckConstraint, ForeignKey, SmallInteger, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.diagnosis import LeafImage
    from app.models.satellite import SatelliteScan
    from app.models.treatment import Treatment
    from app.models.user import User
    from app.models.variety import Variety
    from app.models.weather import DiseaseRisk, WeatherData


class Vineyard(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "vineyards"
    __table_args__ = (
        # Раньше 1900 года посадок, которые ещё плодоносят, не бывает; верхняя
        # граница «не в будущем» проверяется в схеме — CHECK не может зависеть от даты
        CheckConstraint("planting_year >= 1900", name="planting_year_range"),
    )

    owner_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    # Участок может состоять из нескольких контуров; SRID 4326 — как в GeoJSON и веб-картах
    geom: Mapped[WKBElement] = mapped_column(Geometry("MULTIPOLYGON", srid=4326))
    # Сорт необязателен: участок можно завести сразу, а сорт указать потом.
    # Удаление сорта из справочника не удаляет участки — у них сорт становится пустым
    variety_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("varieties.id", ondelete="SET NULL"), index=True
    )
    planting_year: Mapped[int | None] = mapped_column(SmallInteger)

    owner: Mapped["User"] = relationship(back_populates="vineyards")
    variety: Mapped["Variety | None"] = relationship()
    # Фото остаются у владельца, если участок удалён (vineyard_id → NULL)
    leaf_images: Mapped[list["LeafImage"]] = relationship(
        back_populates="vineyard", passive_deletes=True
    )
    satellite_scans: Mapped[list["SatelliteScan"]] = relationship(
        back_populates="vineyard", cascade="all, delete-orphan", passive_deletes=True
    )
    weather: Mapped[list["WeatherData"]] = relationship(
        back_populates="vineyard", cascade="all, delete-orphan", passive_deletes=True
    )
    disease_risks: Mapped[list["DiseaseRisk"]] = relationship(
        back_populates="vineyard", cascade="all, delete-orphan", passive_deletes=True
    )
    treatments: Mapped[list["Treatment"]] = relationship(
        back_populates="vineyard", cascade="all, delete-orphan", passive_deletes=True
    )
