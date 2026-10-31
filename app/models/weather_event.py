import uuid
from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, SmallInteger, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.vineyard import Vineyard


class WeatherEvent(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Заморозок, град или другая беда, которую хозяин отметил на участке."""

    __tablename__ = "weather_events"

    vineyard_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("vineyards.id", ondelete="CASCADE"), index=True
    )
    occurred_on: Mapped[date]
    kind: Mapped[str] = mapped_column(String(16))
    # Доля повреждений глазами хозяина: 1 — слабо, 2 — заметно, 3 — сильно
    severity: Mapped[int] = mapped_column(SmallInteger)
    note: Mapped[str | None] = mapped_column(String(500))

    vineyard: Mapped["Vineyard"] = relationship(back_populates="weather_events")
