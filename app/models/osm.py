from geoalchemy2 import Geography, Geometry, WKBElement
from sqlalchemy import Index, String, cast
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


# Соседние виноградники ищутся в метрах (ST_DWithin и ST_Distance по
# geography), а GiST-индекс по geometry такое условие не использует: без этого
# индекса каждый запрос перебирал все контуры страны (pr-091)
Index(
    "ix_osm_vineyards_geom_geography",
    cast(OsmVineyard.geom, Geography),
    postgresql_using="gist",
)
