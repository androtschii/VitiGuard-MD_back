from geoalchemy2 import Geometry, WKBElement
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class District(Base):
    """Район, муниципий или АТО Молдовы (geoBoundaries ADM1) — для сводной
    статистики по региону и определения района участка."""

    __tablename__ = "districts"

    # ISO 3166-2: «MD-CU» (Кишинёв), «MD-GA» (Гагаузия)
    iso: Mapped[str] = mapped_column(String(8), primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    geom: Mapped[WKBElement] = mapped_column(Geometry("MULTIPOLYGON", srid=4326))
