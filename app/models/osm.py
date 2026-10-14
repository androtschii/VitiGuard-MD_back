from geoalchemy2 import Geometry, WKBElement
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class OsmVineyard(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Виноградник из OpenStreetMap — справочный слой, а не участок пользователя:
    по нему показываются виноградники страны на карте и контекст местности."""

    __tablename__ = "osm_vineyards"

    # «way/123» или «relation/456»: по нему повторный импорт обновляет запись
    osm_id: Mapped[str] = mapped_column(String(40), unique=True)
    name: Mapped[str | None] = mapped_column(String(255))
    geom: Mapped[WKBElement] = mapped_column(Geometry("MULTIPOLYGON", srid=4326))
