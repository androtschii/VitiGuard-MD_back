import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import Disease, RiskLevel, string_enum

if TYPE_CHECKING:
    from app.models.user import User
    from app.models.vineyard import Vineyard


class Alert(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Предупреждение о риске болезни на участке на день (pr-080).

    Одно на участок, болезнь и день: пересчёт каждые 6 часов не плодит
    повторов, а при росте уровня предупреждение обновляется и снова
    становится непрочитанным."""

    __tablename__ = "alerts"
    __table_args__ = (UniqueConstraint("vineyard_id", "disease", "day"),)

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    vineyard_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("vineyards.id", ondelete="CASCADE"), index=True
    )
    disease: Mapped[Disease] = mapped_column(string_enum(Disease, "disease"))
    # День, на который ожидается риск: сегодня или по прогнозу на 1-2 дня вперёд
    day: Mapped[date]
    level: Mapped[RiskLevel] = mapped_column(string_enum(RiskLevel, "risk_level"))
    read_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped["User"] = relationship()
    vineyard: Mapped["Vineyard"] = relationship()
