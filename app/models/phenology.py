import uuid
from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, SmallInteger, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.vineyard import Vineyard


class PhenologyObservation(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Фаза развития лозы, отмеченная на участке (шкала Эйхгорна–Ленца)."""

    __tablename__ = "phenology_observations"
    __table_args__ = (UniqueConstraint("vineyard_id", "observed_on"),)

    vineyard_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("vineyards.id", ondelete="CASCADE"), index=True
    )
    observed_on: Mapped[date]
    stage: Mapped[int] = mapped_column(SmallInteger)

    vineyard: Mapped["Vineyard"] = relationship(back_populates="phenology")
