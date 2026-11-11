import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.vineyard import Vineyard


class VineCheck(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Анализ фото ряда: стволы и грозди, найденные моделью лозы.

    Фото лежит в хранилище, как фото листа; рамки — в долях снимка, поэтому
    их можно нарисовать поверх фото любого размера."""

    __tablename__ = "vine_checks"

    owner_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    vineyard_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("vineyards.id", ondelete="SET NULL"), index=True
    )
    storage_key: Mapped[str] = mapped_column(String(512), unique=True)
    content_type: Mapped[str] = mapped_column(String(50))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    model_name: Mapped[str] = mapped_column(String(100))
    model_version: Mapped[str] = mapped_column(String(64))
    trunks: Mapped[int] = mapped_column(Integer)
    bunches: Mapped[int] = mapped_column(Integer)
    # [{"label": "bunch", "confidence": 0.81, "box": [x1, y1, x2, y2]}, ...]
    detections: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)

    vineyard: Mapped["Vineyard | None"] = relationship()
