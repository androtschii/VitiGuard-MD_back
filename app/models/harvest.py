import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, SmallInteger, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.vineyard import Vineyard


class HarvestRecord(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Урожай участка за сезон: прогноз весной и летом, факт после сбора."""

    __tablename__ = "harvest_records"
    __table_args__ = (UniqueConstraint("vineyard_id", "season"),)

    vineyard_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("vineyards.id", ondelete="CASCADE"), index=True
    )
    season: Mapped[int] = mapped_column(SmallInteger)
    estimated_kg: Mapped[float | None] = mapped_column(Float)
    actual_kg: Mapped[float | None] = mapped_column(Float)
    sugar_brix: Mapped[float | None] = mapped_column(Float)

    vineyard: Mapped["Vineyard"] = relationship(back_populates="harvests")
