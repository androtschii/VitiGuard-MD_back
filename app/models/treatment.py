import uuid
from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, SmallInteger, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import Fungicide, string_enum

if TYPE_CHECKING:
    from app.models.vineyard import Vineyard


class Treatment(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Обработка участка фунгицидом (pr-078): по журналу обработок видно,
    защищён ли участок в день риска и когда обрабатывать снова."""

    __tablename__ = "treatments"

    vineyard_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("vineyards.id", ondelete="CASCADE"), index=True
    )
    applied_on: Mapped[date]
    fungicide: Mapped[Fungicide] = mapped_column(string_enum(Fungicide, "fungicide"))
    # Название препарата — для журнала, на расчёт не влияет
    product: Mapped[str | None] = mapped_column(String(200))
    # Срок ожидания до сбора с этикетки препарата: зависит от препарата и
    # страны, поэтому его вводит пользователь
    waiting_days: Mapped[int | None] = mapped_column(SmallInteger)

    vineyard: Mapped["Vineyard"] = relationship(back_populates="treatments")
