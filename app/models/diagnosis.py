import enum
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, Enum, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.user import User
    from app.models.vineyard import Vineyard


class DiagnosisStatus(enum.StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class LeafImage(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Фото листа. Сам файл лежит в объектном хранилище, здесь — ключ и метаданные."""

    __tablename__ = "leaf_images"
    __table_args__ = (
        CheckConstraint("size_bytes > 0", name="size_positive"),
        CheckConstraint("width > 0 AND height > 0", name="dimensions_positive"),
    )

    owner_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    # Фото из Telegram-бота может прийти без привязки к участку
    vineyard_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("vineyards.id", ondelete="SET NULL"), index=True
    )
    storage_key: Mapped[str] = mapped_column(String(512), unique=True)
    content_type: Mapped[str] = mapped_column(String(50))
    size_bytes: Mapped[int] = mapped_column(Integer)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)

    owner: Mapped["User"] = relationship(back_populates="leaf_images")
    vineyard: Mapped["Vineyard | None"] = relationship(back_populates="leaf_images")
    results: Mapped[list["DiseaseResult"]] = relationship(
        back_populates="image", cascade="all, delete-orphan", passive_deletes=True
    )


class DiseaseResult(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Результат диагностики фото одной версией модели.

    Фото можно прогнать заново новой моделью, поэтому результатов у него несколько:
    история нужна для сравнения версий и журнала дообучения.
    """

    __tablename__ = "disease_results"
    __table_args__ = (
        CheckConstraint("confidence BETWEEN 0 AND 1", name="confidence_range"),
        CheckConstraint(
            "status <> 'completed'"
            " OR (predicted_class IS NOT NULL AND confidence IS NOT NULL)",
            name="completed_has_prediction",
        ),
    )

    image_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("leaf_images.id", ondelete="CASCADE"), index=True
    )
    model_name: Mapped[str] = mapped_column(String(100))
    model_version: Mapped[str] = mapped_column(String(50))
    status: Mapped[DiagnosisStatus] = mapped_column(
        Enum(
            DiagnosisStatus,
            name="diagnosis_status",
            native_enum=False,
            create_constraint=True,
            length=20,
            values_callable=lambda statuses: [status.value for status in statuses],
        ),
        server_default=DiagnosisStatus.PENDING.value,
    )
    # Строка, а не перечисление: набор классов задаёт обученная модель и он меняется
    predicted_class: Mapped[str | None] = mapped_column(String(50))
    confidence: Mapped[float | None]
    # Top-K вероятностей: {"downy_mildew": 0.91, "healthy": 0.06, ...}
    probabilities: Mapped[dict[str, float] | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(String(1000))

    image: Mapped["LeafImage"] = relationship(back_populates="results")
