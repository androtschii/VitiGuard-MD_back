import uuid
from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, String, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.vineyard import Vineyard


class FieldNote(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Заметка с обхода участка: текст, фото и точка, где её сделали."""

    __tablename__ = "field_notes"
    __table_args__ = (UniqueConstraint("vineyard_id", "client_id"),)

    vineyard_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("vineyards.id", ondelete="CASCADE"), index=True
    )
    noted_on: Mapped[date]
    text: Mapped[str] = mapped_column(String(2000))
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    photo_key: Mapped[str | None] = mapped_column(String(512), unique=True)
    photo_type: Mapped[str | None] = mapped_column(String(32))
    # Метка устройства: ответ мог потеряться, и очередь повторит ту же заметку
    client_id: Mapped[uuid.UUID | None] = mapped_column(Uuid)

    vineyard: Mapped["Vineyard"] = relationship(back_populates="field_notes")
