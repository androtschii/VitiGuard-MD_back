import enum

from sqlalchemy import Index, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import string_enum


class BerryColor(enum.StrEnum):
    """Цвет ягод сорта: белый, розовый или красный."""

    WHITE = "white"
    # Розовые и серые ягоды (Traminer Roz, Pinot Gris)
    ROSE = "rose"
    RED = "red"


class Susceptibility(enum.StrEnum):
    """Восприимчивость сорта к болезни: низкая, средняя или высокая."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Variety(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Сорт винограда. Восприимчивость к милдью и оидиуму учитывает модель
    риска: одна и та же погода опаснее для восприимчивого сорта. Пусто —
    восприимчивость не указана, и модель считает риск без поправки на сорт."""

    __tablename__ = "varieties"

    name: Mapped[str] = mapped_column(String(100))
    berry_color: Mapped[BerryColor] = mapped_column(
        string_enum(BerryColor, "berry_color")
    )
    downy_mildew_susceptibility: Mapped[Susceptibility | None] = mapped_column(
        string_enum(Susceptibility, "downy_mildew_susceptibility")
    )
    powdery_mildew_susceptibility: Mapped[Susceptibility | None] = mapped_column(
        string_enum(Susceptibility, "powdery_mildew_susceptibility")
    )
    description: Mapped[str | None] = mapped_column(Text)


# «Merlot» и «merlot» — один сорт: уникальность названия без учёта регистра
Index("uq_varieties_name_lower", func.lower(Variety.name), unique=True)
